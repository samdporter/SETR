#!/usr/bin/env python3
"""SETR HKEM reconstruction for single bed position using MyKEM - Simplified version."""

import logging
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction, SGFunction
from cil.optimisation.operators import CompositionOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from sirf.STIR import MessageRedirector

from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import TruncationOperator
from setr.kernel.python.my_kem import KernelOperator
from setr.scripts.common import configure_logging, init_run_env
from setr.scripts.hkem_common import get_attn_and_normalise, get_kernel_hyperparams
from setr.utils import get_pet_data, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters, get_pet_am, get_spect_am


def prepare_data(args):
    """Prepare the data for reconstruction."""
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

    data["initial_image"].write(
        os.path.join(args.output_path, "initial_image.hv")
    )

    # Check for NaNs in all data
    for key, value in data.items():
        if hasattr(value, "as_array") and np.isnan(value.as_array()).any():
            logging.warning(f"Data '{key}' contains NaNs")

    return data, guidance


# get_kernel_hyperparams now imported from hkem_common


def run_ista(args, data, guidance, hyperparams):
    """Run ISTA-based reconstruction with MyKEM kernel preconditioner."""
    logging.info("Running ISTA algorithm with MyKEM")

    # Get acquisition model function
    if args.modality.upper() == "PET":

        def get_am():
            return get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.gauss_fwhm)
    else:

        def get_am():
            return get_spect_am(data, args.spect_res, True, args.gauss_fwhm)

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
    K = KernelOperator(data["initial_image"], **hyperparams)
    K.set_anatomical_image(guidance)

    # Set up objective functions with kernel operator
    truncate = TruncationOperator(data["initial_image"])
    f_list = [OperatorCompositionFunction(obj, CompositionOperator(K, truncate)) for obj in objs]

    sampler = Sampler.sequential(args.num_subsets)
    f = -SGFunction(f_list, sampler)
    g = BlockIndicatorBox(lower=0)

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
        max_iteration=num_subiterations,
        update_objective_interval=args.num_subsets,
    )

    logging.info("Running ISTA reconstruction...")
    algo.run(num_subiterations, verbose=True)

    # Get final results
    output_alpha = algo.solution
    output_x = K.direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main():
    """Main MyKEM reconstruction pipeline."""
    configure_logging()

    # Parse arguments and configuration
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = SimpleNamespace(**config)

    # Initialize run environment
    msg = init_run_env(args)

    # Redirect messages
    MessageRedirector()

    # Save arguments
    save_args(args, "hkem_args.csv")

    logging.info(f"Starting HKEM {args.method.upper()} reconstruction")
    logging.info(f"Modality: {args.modality}")
    logging.info(f"Guidance: {args.guidance}")

    # Prepare data
    data, guidance = prepare_data(args)

    # Get hyperparameters
    hyperparams = get_kernel_hyperparams(args)

    # Run MyKEM ISTA reconstruction
    if args.method.lower() == "ista":
        output_alpha, output_x = run_ista(args, data, guidance, hyperparams)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logging.info("HKEM reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
