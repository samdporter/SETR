#!/usr/bin/env python3
"""SETR RDP reconstruction for single bed position - CIL-based implementation."""

import logging
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    ScaledFunction,
    SumFunction,
    SVRGFunction,
)
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
)
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.priors import RelativeDifferencePrior
from setr.scripts.common import (
    attach_prior_hessian,
    configure_logging,
    init_run_env,
)
from setr.utils import get_pet_data, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_array, get_filters, get_pet_am, get_spect_am


def prepare_data(args):
    """
    Prepare the data for reconstruction.

    Returns:
        data: Dictionary containing all necessary data
    """
    if args.modality.upper() == "PET":
        data = get_pet_data(args.data_path)
    else:  # SPECT
        data = get_spect_data(args.data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters()
    gauss.apply(data["initial_image"])
    cyl.apply(data["initial_image"])

    data["initial_image"].write(os.path.join(args.output_path, "initial_image.hv"))

    # Check for NaNs in all data
    for key, value in data.items():
        if hasattr(value, "as_array") and np.isnan(get_array(value)).any():
            logging.warning(f"Data '{key}' contains NaNs")

    return data


def run_rdp_ista(args, data):
    """Run ISTA-based reconstruction with RDP regularization and preconditioners."""
    logging.info("Running ISTA algorithm with RDP regularization")

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

    # Compute sensitivity inverse for BSREM preconditioner
    from setr.scripts.common import get_sensitivity_from_subset_objs

    sensitivity = get_sensitivity_from_subset_objs(objs)

    # Create sensitivity inverse
    s_inv = sensitivity.clone()
    sens_array = get_array(sensitivity)
    s_inv.fill(np.reciprocal(sens_array, where=sens_array != 0))

    # Apply filters to sensitivity
    cyl, _ = get_filters()
    cyl.apply(s_inv)

    # Save sensitivity inverse
    s_inv.write(os.path.join(args.output_path, "s_inv.hv"))
    logging.info(f"Writing s_inv with max {s_inv.max()}")

    # Set up RDP prior
    rdp_prior = RelativeDifferencePrior(
        domain_geometry=data["initial_image"],
        beta=1.0,  # Will be scaled by penalty factor
        gamma=args.gamma,
        stencil=args.stencil,
        anatomical=None if args.directional else data["attenuation"],
        both_directions=args.both_directions,
    )

    # Scale and attach Hessian to prior (following DTNV pattern)
    scaled_prior = ScaledFunction(rdp_prior, args.beta)
    prior = 1 / len(objs) * scaled_prior  # Normalize for subset count
    attach_prior_hessian(prior)

    # Set up objective functions with prior added to each subset
    all_funs = []
    for obj in objs:
        all_funs.append(SumFunction(obj, prior))

    update_interval = len(objs)

    # Set up data fidelity function
    sampler = Sampler.sequential(args.num_subsets)
    f_obj = -SVRGFunction(all_funs, sampler, snapshot_update_interval=2 * update_interval)

    # BSREM preconditioner using sensitivity
    bsrem_precond = BSREMPreconditioner(s_inv, update_interval=update_interval)

    # Prior preconditioner using RDP inverse Hessian
    prior_precond = ImageFunctionPreconditioner(
        prior.inv_hessian_diag, update_interval=update_interval
    )

    # Combined preconditioner using LehmerMean
    preconditioner = LehmerMeanPreconditioner(
        [bsrem_precond, prior_precond],
        update_interval=update_interval,
        freeze_iter=len(objs) * 10,
    )

    # Set up constraint (positivity)
    g = BlockIndicatorBox(lower=0, upper=np.inf)

    # Initialize reconstruction
    initial_estimate = data["initial_image"]

    # Set up callbacks
    callbacks = []

    # Save images callback
    if args.save_images:
        save_callback = SaveImageCallback(
            interval=update_interval,
            filename=os.path.join(args.output_path, "rdp_image"),
        )
        callbacks.append(save_callback)

    # Save objective callback
    obj_callback = SaveObjectiveCallback(
        interval=update_interval,
        filename=os.path.join(args.output_path, "objective.csv"),
    )
    callbacks.append(obj_callback)

    # Print progress callback
    print_callback = PrintObjectiveCallback(interval=update_interval)
    callbacks.append(print_callback)

    # Set up algorithm with preconditioner
    algo = ISTA(
        initial=initial_estimate,
        f=f_obj,
        g=g,
        preconditioner=preconditioner,  # Use LehmerMean preconditioner
        step_size=LinearDecayStepSizeRule(args.initial_step_size, args.relaxation_eta),
        update_objective_interval=update_interval,
    )

    num_subiterations = args.num_epochs * update_interval
    logging.info(f"Running RDP-ISTA reconstruction for {num_subiterations} iterations...")
    algo.run(num_subiterations, verbose=True, callbacks=callbacks)

    # Get final results
    output_image = algo.solution

    # Save final results
    output_image.write(os.path.join(args.output_path, "reconstruction_rdp.hv"))

    logging.info("RDP reconstruction completed successfully")
    return output_image


def main():
    """Main function to run RDP single bed reconstruction."""
    configure_logging()

    # Parse arguments and configuration
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = SimpleNamespace(**config)

    # Initialize run environment
    msg = init_run_env(args)

    # Save arguments
    save_args(args, "rdp_1bpos_args.csv")

    logging.info("Starting RDP single bed reconstruction")
    logging.info(f"Modality: {args.modality}")
    logging.info(f"Beta (penalty factor): {args.beta}")
    logging.info(f"Gamma (shape parameter): {args.gamma}")

    # Prepare data
    data = prepare_data(args)

    # Run reconstruction
    output_image = run_rdp_ista(args, data)

    logging.info("RDP single bed reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
