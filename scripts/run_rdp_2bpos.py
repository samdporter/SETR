#!/usr/bin/env python3
"""SETR RDP reconstruction for multiple bed positions - CIL-based implementation."""

import logging
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    ScaledFunction,
    SumFunction,
    SVRGFunction,
)
from cil.optimisation.operators import CompositionOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
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
    get_resampling_operators,
    get_shift_operators,
    init_run_env,
)
from setr.utils import get_pet_data_multiple_bed_pos
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters, get_pet_am


def prepare_data(args):
    """Prepare the multi-bed PET data."""
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    pet_data["initial_image"].write(
        os.path.join(args.output_path, "initial_image.hv")
    )

    # Create initial estimates - for RDP we just need PET
    initial_estimates = pet_data["initial_image"]

    if np.isnan(initial_estimates.as_array()).any():
        logging.warning("Initial image contains NaNs")

    return pet_data, initial_estimates


def get_data_fidelity(args, pet_data, uncombine_op, unshift_ops, choose_ops):
    """
    Set up data fidelity functions for multi-bed reconstruction.
    
    Returns:
        all_funs: List of block objective functions (PET all beds)
        s_inv: Sensitivity inverse images (combined across beds)
    """
    # Get acquisition model function
    def get_pet_am_with_res():
        return get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.pet_gauss_fwhm)

    # Partition PET data by bed position
    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=args.num_subsets,
            mode="staggered",
            create_acq_model=get_pet_am_with_res,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # Set up subset objectives on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # Get sensitivities for each bed position
    from setr.scripts.common import get_sensitivity_from_subset_objs
    pet_sens = [
        get_sensitivity_from_subset_objs(df)
        for df in pet_dfs
    ]

    # Unshift and combine PET sensitivities to common PET grid
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )

    # Create sensitivity inverse
    s_inv = pet_sens_combined.clone()
    sens_array = pet_sens_combined.as_array()
    s_inv.fill(np.reciprocal(sens_array, where=sens_array != 0))

    # Apply cylindrical filter
    cyl, _ = get_filters()
    cyl.apply(s_inv)

    # Save sensitivity inverse
    s_inv.write(os.path.join(args.output_path, "s_inv.hv"))
    logging.info(f"Writing s_inv with max {s_inv.max()}")

    # Wrap PET objectives with uncombine/choose/unshift operators
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                ),
            )

    # Flatten the list to get one function per subset per bed
    all_funs = [df for bed in pet_dfs for df in bed]

    return all_funs, s_inv


def run_rdp_ista(args, pet_data, initial_estimates):
    """Run ISTA-based RDP reconstruction with multi-bed handling."""

    # Set up shift operators
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    # Set up data fidelity functions
    all_funs, s_inv = get_data_fidelity(
        args,
        pet_data,
        uncombine_op,
        unshift_ops,
        choose_ops,
    )

    # Set up objective function
    update_interval = len(all_funs)
    probs = [1 / update_interval] * len(all_funs)

    data_fidelity = -SVRGFunction(
        all_funs,
        sampler=Sampler.random_with_replacement(len(all_funs), prob=probs),
        snapshot_update_interval=update_interval * 2,
        store_gradients=True,
    )

    # Set up RDP prior in combined image space
    rdp_prior = RelativeDifferencePrior(
        domain_geometry=initial_estimates,
        beta=1.0,  # Will be scaled by penalty factor
        gamma=args.gamma,
        stencil=args.stencil,
        anatomical=None if args.directional else pet_data["attenuation"],
        both_directions=args.both_directions,
    )

    # Scale and attach Hessian to prior (following DTNV pattern)
    scaled_prior = ScaledFunction(rdp_prior, args.beta)
    prior = 1 / len(all_funs) * scaled_prior  # Normalize for subset count
    attach_prior_hessian(prior)

    # Add prior to each subset objective
    for i, fun in enumerate(all_funs):
        all_funs[i] = SumFunction(fun, prior)

    # Set up data fidelity function
    f_obj = -SVRGFunction(
        all_funs,
        sampler=Sampler.random_with_replacement(len(all_funs), prob=probs),
        snapshot_update_interval=update_interval * 2,
        store_gradients=True,
    )

    # Set up preconditioners following DTNV pattern
    # BSREM preconditioner using sensitivity
    bsrem_precond = BSREMPreconditioner(
        s_inv, 
        update_interval=update_interval
    )
    
    # Prior preconditioner using RDP inverse Hessian
    prior_precond = ImageFunctionPreconditioner(
        prior.inv_hessian_diag,
        update_interval=update_interval
    )
    
    # Combined preconditioner using LehmerMean
    preconditioner = LehmerMeanPreconditioner(
        [bsrem_precond, prior_precond],
        update_interval=update_interval,
        freeze_iter=len(all_funs) * 10,
    )

    # Set up constraint
    g = BlockIndicatorBox(lower=0, upper=np.inf)

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
    
    print_callback = PrintObjectiveCallback(interval=update_interval)
    callbacks.append(print_callback)

    # Set up algorithm with preconditioner
    algo = ISTA(
        initial=initial_estimates,
        f=f_obj,
        g=g,
        preconditioner=preconditioner,  # Use LehmerMean preconditioner
        step_size=LinearDecayStepSizeRule(args.initial_step_size, args.relaxation_eta),
        update_objective_interval=update_interval,
    )

    num_subiterations = args.num_epochs * update_interval
    logging.info("Running RDP-ISTA multi-bed reconstruction...")
    algo.run(num_subiterations, verbose=True, callbacks=callbacks)

    # Get final results
    output_image = algo.solution

    # Save final results
    output_image.write(os.path.join(args.output_path, "reconstruction_rdp.hv"))

    return output_image


def main():
    """Main function to run RDP multi-bed reconstruction."""
    configure_logging()

    # Parse arguments and configuration
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = SimpleNamespace(**config)

    # Initialize run environment
    msg = init_run_env(args)

    # Save arguments
    save_args(args, "rdp_2bpos_args.csv")

    logging.info("Starting RDP multi-bed reconstruction")
    logging.info(f"Beta (penalty factor): {args.beta}")
    logging.info(f"Gamma (shape parameter): {args.gamma}")

    # Prepare data
    pet_data, initial_estimates = prepare_data(args)

    # Run reconstruction
    output_image = run_rdp_ista(args, pet_data, initial_estimates)

    logging.info("RDP multi-bed reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()