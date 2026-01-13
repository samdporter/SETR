#!/usr/bin/env python3
"""SETR HKEM reconstruction for single bed position - Simplified version using shared modules."""

import logging
import os
from types import SimpleNamespace

from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction, SGFunction
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import TruncationOperator
from setr.cil_extensions.preconditioners import SubsetKernelisedEMPreconditioner
from setr.scripts.common import (
    configure_logging,
    init_run_env,
)
from setr.scripts.hkem_common import (
    get_attn_and_normalise,
    get_kernel_hyperparams,
    get_kernel_operator,
    run_kosmaposl,
)
from setr.utils import get_pet_data, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_pet_am, get_spect_am
from setr.cil_extensions.operators.blurring import create_gaussian_blur_operator

ISTA.update = ista_update_step  # Patch ISTA with our custom update step

DEBUG = False  # Set to True to save more debugging information


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
            guidance = data["spect"]
        else:
            guidance = get_attn_and_normalise(args)
    else:  # SPECT
        data = get_spect_data(args.data_path)
        guidance = get_attn_and_normalise(args)

    return data, guidance


def run_ista(args, data, guidance, hyperparams):
    """Run ISTA-based reconstruction with kernel preconditioner."""

    # Get acquisition model function
    if args.modality.upper() == "PET":

        def get_am():
            return get_pet_am(gpu=not args.no_gpu, gauss_fwhm=None)
    else:

        def get_am():
            return get_spect_am(data, args.spect_res, True, gauss_fwhm=args.gauss_fwhm)

    # Handle SPECT normalisation
    if args.modality.upper() == "SPECT":
        data["normalisation"] = data["acquisition_data"].get_uniform_copy(1)

    # Partition data
    _, _, objs = partitioner.data_partition(
        data["acquisition_data"],
        data["additive"] if args.use_scatter else data["additive"].get_uniform_copy(0),
        data["normalisation"],
        args.num_subsets,
        mode=args.sampling,
        create_acq_model=get_am,
    )

    for obj in objs:
        obj.set_up(data["initial_image"])

    # Create Gaussian blurring operator for PET only
    # SPECT uses image_data_processor which works correctly for SPECT projectors
    blur_op = None
    if args.modality.upper() == "PET":
        blur_op = create_gaussian_blur_operator(args.gauss_fwhm, data["initial_image"])

        # Wrap PET objectives with Gaussian blurring operator (if specified)
        if blur_op is not None:
            objs = [OperatorCompositionFunction(obj, blur_op) for obj in objs]

    K = get_kernel_operator(
        args, guidance, data["initial_image"], data["acquisition_data"], hyperparams
    )

    # Set up objective functions with kernel operator
    f_list = [OperatorCompositionFunction(obj, K) for obj in objs]

    sampler = Sampler.sequential(args.num_subsets)
    f = -SGFunction(f_list, sampler)
    g = BlockIndicatorBox(lower=0)

    # Get sensitivities for preconditioner
    sensitivities = []
    for obj in objs:
        # Extract underlying function if wrapped in OperatorCompositionFunction
        obj_fn = obj.function if isinstance(obj, OperatorCompositionFunction) else obj
        sens = obj_fn.get_subset_sensitivity(0)
        sens = sens.maximum(0)
        if blur_op is not None:
            sens = blur_op.adjoint(sens)
        sensitivities.append(sens * args.num_subsets)  # Scale by number of subsets

    # Create preconditioner
    precond = SubsetKernelisedEMPreconditioner(
        args.num_subsets,
        sensitivities,
        K,
        freeze_iter=args.freeze_iter,
        epsilon=data["initial_image"].max() * 1e-12,
    )

    # Initialize alpha
    truncate = TruncationOperator(data["initial_image"])
    init_alpha = data["initial_image"].get_uniform_copy(1)
    truncate.direct(init_alpha, out=init_alpha)  # Apply truncation

    # Use modality-specific epochs if available, otherwise fall back to num_epochs
    if args.modality.upper() == "PET":
        num_epochs = getattr(args, "num_epochs_pet", args.num_epochs)
        logging.info(f"Using PET epochs: {num_epochs}")
    else:  # SPECT
        num_epochs = getattr(args, "num_epochs_spect", args.num_epochs)
        logging.info(f"Using SPECT epochs: {num_epochs}")

    num_subiterations = args.num_subsets * num_epochs
    logging.info(f"Total subiterations: {num_subiterations} ({num_epochs} epochs × {args.num_subsets} subsets)")

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

    interval = 1 if DEBUG else args.num_subsets

    callbacks = [
        SaveImageCallback(os.path.join(args.output_path, "alpha"), interval=interval),
        SaveKernelisedImageCallback(
            os.path.join(args.output_path, "x"), interval=interval, kernel_op=K
        ),
        PrintObjectiveCallback(interval=args.num_subsets),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), interval=interval),
    ]

    if DEBUG:  # Only save preconditioner and gradient if debugging
        callbacks.extend(
            [
                SavePreconditionerCallback(
                    os.path.join(args.output_path, "preconditioner"), interval=interval
                ),
                SaveGradientUpdateCallback(
                    os.path.join(args.output_path, "gradient"), interval=interval
                ),
            ]
        )

    logging.info("Running ISTA reconstruction...")
    algo.run(num_subiterations, verbose=True, callbacks=callbacks)

    # Get final results
    output_alpha = algo.solution
    output_x = K.direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main(args):
    """Main function to run HKEM reconstruction."""
    configure_logging()

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
        output_alpha, output_x = run_kosmaposl(args, data, guidance, hyperparams)
    elif args.method.lower() == "ista":
        output_alpha, output_x = run_ista(args, data, guidance, hyperparams)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logging.info("HKEM reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    cli = parse_cli()
    cfg_dict = load_config(cli.config)
    cfg_dict = apply_overrides(cfg_dict, cli.override)

    args = SimpleNamespace(**cfg_dict)

    msg = init_run_env(args)

    main(args)
