#!/usr/bin/env python3
"""
Hybrid Kernelized Expectation Maximization (HKEM) reconstruction for single bed position.

This script implements HKEM reconstruction using the SETR framework with either:
1. KOSMAPOSL (Kernelized Ordered Subsets Maximum A-Posteriori One Step Late)
2. ISTA-based reconstruction with kernel preconditioner

The kernel operator provides anatomical guidance for improved reconstruction quality.
"""

import logging
import os
from types import SimpleNamespace

import numpy as np
from sirf.contrib.partitioner import partitioner

# SIRF imports
from sirf.STIR import AcquisitionData, ImageData, MessageRedirector

AcquisitionData.set_storage_scheme("memory")

# CIL imports
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    SGFunction,
)
from cil.optimisation.operators import (
    CompositionOperator,
)
from cil.optimisation.utilities import Sampler

# SETR imports
from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
)
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import TruncationOperator
from setr.cil_extensions.preconditioners import SubsetKernelisedEMPreconditioner
from setr.kernel import get_kernel_operator
from setr.utils import get_pet_am, get_pet_data, get_spect_am, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters

# Configure CLI and load config
cli = parse_cli()
cfg_dict = load_config(cli.config)
cfg_dict = apply_overrides(cfg_dict, cli.override)
args = SimpleNamespace(**cfg_dict)

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(args.working_path, exist_ok=True)

# Attach the new update method to ISTA
ISTA.update = ista_update_step


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def prepare_data(args):
    """
    Prepare the data for reconstruction.

    Returns:
        data: Dictionary containing all necessary data
        guidance: Guidance image for kernel operator
    """
    if args.modality.upper() == "PET":
        data = get_pet_data(args.data_path)

        # Get guidance image
        if args.guidance == "emission":
            # Load SPECT data for emission guidance
            spect_data = get_spect_data(args.spect_data_path)
            guidance = spect_data["initial_image"]
        else:
            guidance = get_attn_and_normalise(args)
    else:  # SPECT
        data = get_spect_data(args.data_path)
        guidance = get_attn_and_normalise(args)
    # Apply filters to initial images
    cyl, gauss = get_filters()
    gauss.apply(data["initial_image"])
    cyl.apply(data["initial_image"])

    data["initial_image"].write("initial_image.hv")

    # Check for NaNs in all data
    for key, value in data.items():
        if hasattr(value, "as_array") and np.isnan(value.as_array()).any():
            logging.warning(f"Data '{key}' contains NaNs")

    return data, guidance


def get_attn_and_normalise(args):
    # Use attenuation map
    result = ImageData(os.path.join(args.data_path, "umap_zoomed.hv"))
    result += (-result).max()
    result /= result.max()

    return result


def get_kernel_hyperparams(args):
    """Return kernel hyperparameters dictionary."""
    return {
        "num_neighbours": args.num_neighbours,
        "num_non_zero_features": args.num_non_zero_features,
        "sigma_anat": args.sigma_anatomical,
        "sigma_emission": args.sigma_emission,
        "sigma_dist": args.sigma_distance_anatomical,
        "normalize_features": args.normalize_features,
        "normalize_kernel": args.normalize_kernel,
        "use_mask": args.use_mask,
        "mask_k": args.mask_k,
        "recalc_mask": args.recalc_mask,
        "distance_weighting": args.distance_weighting,
        "hybrid": args.hybrid,
    }


def run_ista(args, data, guidance, hyperparams):
    """Run ISTA-based reconstruction with kernel preconditioner."""

    # Get acquisition model function
    if args.modality.upper() == "PET":
        get_am = lambda: get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.gauss_fwhm)
    else:
        get_am = lambda: get_spect_am(data, args.spect_res, True, args.gauss_fwhm)

    # Handle SPECT normalisation
    if args.modality.upper() == "SPECT":
        data["normalisation"] = data["acquisition_data"].get_uniform_copy(1)

    # Partition data
    _, _, objs = partitioner.data_partition(
        data["acquisition_data"],
        data["additive"],
        data["normalisation"],
        args.num_subsets,
        mode=args.sampling,
        create_acq_model=get_am,
    )

    for obj in objs:
        obj.set_up(data["initial_image"])

    # Create kernel operator
    K = get_kernel_operator(
        data["initial_image"], backend=args.kernel_backend, **hyperparams
    )
    K.set_anatomical_image(guidance)

    # Set up objective functions with kernel operator
    truncate = TruncationOperator(data["initial_image"])
    f_list = [
        OperatorCompositionFunction(obj, CompositionOperator(K, truncate))
        for obj in objs
    ]

    sampler = Sampler.sequential(args.num_subsets)
    f = -SGFunction(f_list, sampler)
    g = BlockIndicatorBox(lower=0)

    # Get sensitivities for preconditioner
    sensitivities = []
    for obj in objs:
        sens = obj.get_subset_sensitivity(0)
        sens = sens.maximum(0)
        sensitivities.append(sens)

    # Create preconditioner
    precond = SubsetKernelisedEMPreconditioner(
        args.num_subsets,
        sensitivities,
        K,
        freeze_iter=args.freeze_iter,
        epsilon=data["initial_image"].max() * 1e-12,
    )

    # Initialize alpha
    init_alpha = data["initial_image"].get_uniform_copy(1)
    truncate.direct(init_alpha, out=init_alpha)  # Apply truncation

    num_subiterations = args.num_subsets * args.num_epochs

    # Set up algorithm
    algo = ISTA(
        init_alpha,
        f,
        g,
        step_size=args.step_size,
        preconditioner=precond,
        max_iteration=num_subiterations,
        update_objective_interval=args.num_subsets,
    )

    # Set up callbacks
    class SaveKernelisedImageCallback:
        """Save the kernelised image (x = K(alpha)) to disk."""

        def __init__(self, filename, interval, kernel_op):
            self.filename = filename
            self.interval = interval
            self.kernel_op = kernel_op

        def __call__(self, algo):
            if algo.iteration % self.interval != 0:
                return
            # Save the kernelised image
            image = self.kernel_op.direct(algo.solution)
            image.write(f"{self.filename}_{algo.iteration}.hv")

    callbacks = [
        SaveImageCallback(
            os.path.join(args.output_path, "alpha"),
            interval=args.num_subsets,
        ),
        SaveKernelisedImageCallback(
            os.path.join(args.output_path, "x"),
            interval=args.num_subsets,
            kernel_op=K,
        ),
        PrintObjectiveCallback(interval=args.num_subsets),
        SaveObjectiveCallback(
            os.path.join(args.output_path, "objective"),
            interval=args.num_subsets,
        ),
    ]

    logging.info("Running ISTA reconstruction...")
    algo.run(num_subiterations, verbose=True, callbacks=callbacks)

    # Get final results
    output_alpha = algo.solution
    output_x = K.direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main():
    """Main function to run HKEM reconstruction."""
    configure_logging()

    # Redirect messages
    msg = MessageRedirector()

    # Save arguments
    save_args(args, "hkem_args.csv")

    logging.info(f"Starting HKEM {args.method.upper()} reconstruction")
    logging.info(f"Modality: {args.modality}")
    logging.info(f"Guidance: {args.guidance}")

    # Prepare data
    data, guidance = prepare_data(args)

    # Get kernel hyperparameters
    hyperparams = get_kernel_hyperparams(args)

    # Run reconstruction
    if args.method.lower() == "kosmaposl":
        raise NotImplementedError(
            "KOSMAPOSL reconstruction is not implemented for my KEM prior."
        )
    elif args.method.lower() == "ista":
        output_alpha, output_x = run_ista(args, data, guidance, hyperparams)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logging.info("HKEM reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
