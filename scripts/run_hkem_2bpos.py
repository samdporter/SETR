#!/usr/bin/env python3
"""SETR HKEM reconstruction for multiple bed positions - Simplified version using shared modules."""

import logging
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction, SVRGFunction
from cil.optimisation.operators import CompositionOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from sirf.STIR import MessageRedirector

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.preconditioners import (
    DualModalitySubsetKernelisedEMPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.scripts.common import (
    configure_logging,
    get_sensitivity_from_subset_objs,
    get_shift_operators,
    init_run_env,
)
from setr.scripts.hkem_common import get_kernel_hyperparams, get_kernel_operator
from setr.utils import get_pet_data_multiple_bed_pos
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters, get_pet_am


def prepare_data(args):
    """Prepare the multi-bed PET data and guidance image."""
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    # Use attenuation map as guidance
    guidance = pet_data["attenuation"]
    guidance += (-guidance).max()
    guidance /= guidance.max()

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    pet_data["initial_image"].write("initial_image.hv")

    # Create initial estimates - for HKEM we just need PET
    initial_estimates = pet_data["initial_image"]

    if np.isnan(initial_estimates.as_array()).any():
        logging.warning("Initial image contains NaNs")

    return pet_data, guidance, initial_estimates


def get_data_fidelity(
    args, pet_data, get_pet_am, num_subsets, uncombine_op, unshift_ops, choose_ops
):
    """
    Set up data fidelity (objective) functions for multi-bed reconstruction.

    Returns:
        all_funs: List of block objective functions (PET all beds)
        s_inv: Sensitivity inverse images (combined across beds)
        sensitivities: List of sensitivity images for each bed/subset
    """
    # Partition PET data by bed position
    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=num_subsets,
            mode="staggered",
            create_acq_model=get_pet_am,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # Set up subset objectives on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # Get sensitivities for each bed position
    pet_sens = [
        get_sensitivity_from_subset_objs(df, pet_data["bed_positions"][suffix]["template_image"])
        for df, suffix in zip(pet_dfs, pet_data["bed_positions"])
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

    # Create list of sensitivities for preconditioner
    # We need to transform the bed-level sensitivities using the same operators
    sens_bdcs = []
    for bed_idx in range(len(pet_dfs)):  # For each bed
        for _ in range(len(pet_dfs[bed_idx])):
            # Transform the sensitivity from bed coordinates to combined coordinates
            bed_sens = pet_sens[bed_idx]  # Sensitivity for this bed
            combined_sens = uncombine_op.adjoint(
                EnhancedBlockDataContainer(
                    *[
                        unshift_ops[i].adjoint(bed_sens)
                        if i == bed_idx
                        else pet_data["initial_image"].get_uniform_copy(0)
                        for i in range(len(unshift_ops))
                    ]
                )
            )
            sens_bdcs.append(combined_sens)

    return all_funs, s_inv, sens_bdcs


def run_hkem_ista(args, pet_data, guidance, initial_estimates):
    """Run ISTA-based HKEM reconstruction with kernel preconditioner."""

    # Get acquisition model function
    def get_pet_am_with_res():
        return get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.pet_gauss_fwhm)

    # Set up shift operators
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    # Set up data fidelity functions
    all_funs, s_inv, sens_bdcs = get_data_fidelity(
        args,
        pet_data,
        get_pet_am_with_res,
        args.num_subsets,
        uncombine_op,
        unshift_ops,
        choose_ops,
    )

    # Create kernel operators for each bed position
    hyperparams = get_kernel_hyperparams(args)
    kernels = []
    for suffix in pet_data["bed_positions"]:
        guide = (
            pet_data["bed_positions"][suffix]["attenuation"]
            if args.guidance == "attenuation"
            else pet_data["bed_positions"][suffix]["spect"]
        )
        kernel = get_kernel_operator(
            args,
            guide,
            pet_data["bed_positions"][suffix]["template_image"],
            pet_data["bed_positions"][suffix]["acquisition_data"],
            hyperparams,
        )
        kernel.set_anatomical_image(guidance)  # Use same guidance for all beds
        kernels.append(kernel)

    # Set up objective function
    update_interval = len(all_funs)
    probs = [1 / update_interval] * len(all_funs)

    f_obj = -SVRGFunction(
        all_funs,
        sampler=Sampler.random_with_replacement(len(all_funs), prob=probs),
        snapshot_update_interval=update_interval * 2,
        store_gradients=True,
    )

    # Set up preconditioners
    max_val = initial_estimates.max()

    # Create dual modality kernel preconditioner
    dual_precond = DualModalitySubsetKernelisedEMPreconditioner(
        sens_bdcs=sens_bdcs,
        kernel=kernels,
        uncombine_ops=unshift_ops,
        num_subsets=len(all_funs),
        update_interval=update_interval,
        freeze_iter=args.freeze_iter,
        epsilon=max_val * 1e-12,
    )

    # Set up algorithm
    algo = ISTA(
        initial=initial_estimates,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=dual_precond,  # Use kernel preconditioner
        step_size=LinearDecayStepSizeRule(args.initial_step_size, args.relaxation_eta),
        update_objective_interval=update_interval,
    )

    num_subiterations = args.num_epochs * update_interval
    logging.info("Running HKEM-ISTA reconstruction...")
    algo.run(num_subiterations, verbose=True)

    # Get final results
    output_alpha = algo.solution

    # Apply first kernel to get kernelised image
    output_x = kernels[0].direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main():
    """Main function to run HKEM multi-bed reconstruction."""
    configure_logging()

    # Parse arguments and configuration
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = SimpleNamespace(**config)

    # Initialize run environment
    init_run_env(args)

    # Redirect messages
    MessageRedirector()

    # Save arguments
    save_args(args, "hkem_2bpos_args.csv")

    logging.info("Starting HKEM multi-bed reconstruction")

    # Prepare data
    pet_data, guidance, initial_estimates = prepare_data(args)

    # Run reconstruction (only ISTA supported for multi-bed)
    output_alpha, output_x = run_hkem_ista(args, pet_data, guidance, initial_estimates)

    logging.info("HKEM multi-bed reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
