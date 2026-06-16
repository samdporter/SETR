#!/usr/bin/env python3
"""SETR DTNV reconstruction for multiple bed positions - Simplified version using shared modules."""

import argparse
import cProfile
import logging
import os
import pstats

import numpy as np
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction
from cil.optimisation.operators import CompositionOperator
from sirf.contrib.partitioner import partitioner
from sirf.STIR import SeparableGaussianImageFilter

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.cil_extensions.operators import FlipOperator
from recon_core.cil_extensions.utilities import LinearDecayStepSizeRule
from recon_experiments.runners.common import (
    apply_combine_sensitivities,
    build_shared_initial_estimates,
    configure_logging,
    get_pet_to_spect_operator,
    get_resampling_operators,
    get_sensitivity_from_subset_objs,
    get_shift_operators,
    init_run_env,
    save_results,
    save_native_spect_image,
)
from recon_experiments.runners.dtnv_common import (
    apply_dynamic_range_scaling,
    build_support_mask_from_spect_attenuation,
    build_support_mask_from_s_inv,
    build_variance_reduced_function,
    combine_support_masks,
    dynamic_range_scale_sirf,
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_kappa_squareds,
    get_preconditioners,
    get_prior,
    normalise_kappa_squares,
    set_auto_delta_from_scaled_images,
)
from recon_core.utils import (
    get_pet_am,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
)
from recon_core.utils.io import apply_overrides, load_config, parse_cli, save_args
from recon_core.utils.sirf import get_array, get_filters, get_s_inv_from_subset_objs
from recon_core.cil_extensions.operators.blurring import create_gaussian_blur_operator


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def prepare_data(args):
    """
    Prepare the umap image, PET and SPECT data, and initial estimates.

    Returns:
        ct: Normalizedumapimage.
        pet_data: Dictionary containing PET data.
        spect_data: Dictionary containing SPECT data.
        initial_estimates: BlockDataContainer combining PET and SPECT initial images.
        cyl, gauss: Filter objects.
    """

    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    umap = pet_data["attenuation"]
    umap += (-umap).max()
    umap /= umap.max()
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((2, 2, 2))
    ct_smooth.apply(umap)
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))

    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    pet_data["initial_image"].write(os.path.join(args.output_path, "initial_image_0.hv"))
    spect_data["initial_image"].write(os.path.join(args.output_path, "initial_image_1.hv"))

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    # check for nans in all data
    for data in [umap, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("An image contains NaNs")
            break

    return umap, pet_data, spect_data, initial_estimates


def get_data_fidelity(
    args,
    pet_data,
    spect_data,
    get_pet_am,
    get_spect_am,
    num_subsets,
    uncombine_op,
    unshift_ops,
    choose_ops,
    shared_initial_estimates,
    pet_to_spect,
):
    """
    Set up data fidelity (objective) functions.

    Returns:
        all_funs: list of block objective functions (PET all beds, then SPECT).
        s_inv: shared-space inverse sensitivity images.
        kappas: shared-space κ² images.
    """
    # --- partition PET by bed ---
    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=num_subsets[0],
            mode="staggered",
            create_acq_model=get_pet_am,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # set_up subset objs on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # --- partition SPECT ---
    spect_dfs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )[2]
    for obj_fun in spect_dfs:
        obj_fun.set_up(spect_data["initial_image"])

    # Create Gaussian blurring operators for PET only
    # SPECT uses image_data_processor which works correctly for SPECT projectors
    pet_blur_ops = [
        create_gaussian_blur_operator(
            args.pet_gauss_fwhm,
            pet_data["bed_positions"][suffix]["template_image"]
        )
        for suffix in pet_data["bed_positions"]
    ]

    pet_sens = [
        get_sensitivity_from_subset_objs(df, adjoint_operator=op)
        for df, op in zip(pet_dfs, pet_blur_ops)
    ]

    #apply_combine_sensitivities(pet_data, pet_sens)

    spect_s_inv = get_s_inv_from_subset_objs(
        spect_dfs,
        shared_initial_estimates[1],
        clamp_percentile=99.5,
        adjoint_operator=pet_to_spect,
    )

    # unshift+combine PET sensitivities to common PET grid
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )
    # Combined sensitivity can pick up tiny negatives from interpolation/adjoints.
    pet_sens_combined.maximum(0, out=pet_sens_combined)
    pet_s_inv = pet_sens_combined.clone()
    pet_sens_array = get_array(pet_sens_combined)
    pet_s_inv_array = np.zeros_like(
        pet_sens_array, dtype=np.result_type(pet_sens_array, np.float32)
    )
    np.reciprocal(
        pet_sens_array,
        out=pet_s_inv_array,
        where=pet_sens_array != 0,
    )
    pet_s_inv.fill(pet_s_inv_array)
    cyl, _ = get_filters()
    cyl.apply(pet_s_inv)

    s_inv = EnhancedBlockDataContainer(pet_s_inv, spect_s_inv)

    # save s_inv images (unchanged)
    for i, image in enumerate(s_inv.containers):
        image.write(os.path.join(args.output_path, f"s_inv_{i}.hv"))
        logging.info(f"Writing s_inv_{i} with max {image.max()}")

    # --- now wrap PET objectives with blur + uncombine/choose/unshift ---
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            # Build operator chain: blur -> unshift -> choose -> uncombine
            if pet_blur_ops[i] is not None:
                op_chain = CompositionOperator(
                    pet_blur_ops[i],
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op
                )
            else:
                op_chain = CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op
                )
            pet_dfs[i][j] = OperatorCompositionFunction(pet_dfs[i][j], op_chain)

    # flatten beds
    pet_combined_dfs = [df for bed in pet_dfs for df in bed]

    # SPECT objectives are pulled back to the shared PET grid.
    spect_dfs = [
        OperatorCompositionFunction(obj_fun, pet_to_spect)
        for obj_fun in spect_dfs
    ]

    # block objectives
    pet_dfs_block = [
        get_block_objective(
            shared_initial_estimates[0],
            shared_initial_estimates[1],
            df,
            order=0,
        )
        for df in pet_combined_dfs
    ]
    spect_dfs_block = [
        get_block_objective(
            shared_initial_estimates[1],
            shared_initial_estimates[0],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_dfs
    ]

    all_funs = pet_dfs_block + spect_dfs_block

    if args.use_kappa:
        kappas = get_kappa_squareds(
            [pet_dfs_block, spect_dfs_block],
            [shared_initial_estimates[0], shared_initial_estimates[1]],
        )
    else:
        kappas = None

    return all_funs, s_inv, kappas


def main(args) -> None:
    """Main DTNV 2bpos reconstruction pipeline."""
    configure_logging()

    # Initialize run environment (creates dirs, sets storage scheme, redirects messages)
    _ = init_run_env(args)

    # Prepare data
    umap, pet_data, spect_data, _ = prepare_data(args)

    # Set up operators for multiple bed positions
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    # Set up resampling operators
    spect2pet = get_resampling_operators(
        args,
        pet_data, spect_data
    )
    if getattr(args, "flip", False):
        spect2pet = CompositionOperator(
            spect2pet, FlipOperator(axis=(0, 2), image=["initial_image"])
        )
    pet_to_spect = get_pet_to_spect_operator(spect2pet)
    initial_estimates = build_shared_initial_estimates(
        pet_data["initial_image"],
        spect_data["initial_image"],
        spect2pet,
    )

    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=None,
        )

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.keep_all_views_in_cache,
            gauss_fwhm=args.spect_gauss_fwhm,
            attenuation=True,
        )

    # Set up data fidelity
    all_funs, s_inv, kappas = get_data_fidelity(
        args,
        pet_data,
        spect_data,
        get_pet_am_with_res,
        get_spect_am_with_res,
        args.num_subsets,
        uncombine_op,
        unshift_ops,
        choose_ops,
        initial_estimates,
        pet_to_spect,
    )
    kappas = normalise_kappa_squares(kappas) if kappas is not None else None
    combined = initial_estimates

    pet_scale, spect_scale = dynamic_range_scale_sirf(
        combined[0],
        combined[1],
    )

    # Apply consistent scaling to all prior weights
    # Skip this for log TNV since the log transform already normalizes dynamic range
    # Skip this for local weighting since voxel-wise weights are applied directly
    use_log_tnv = getattr(args, "use_log_tnv", False)
    use_local_weighting = getattr(args, "use_local_weighting", False)
    if not use_log_tnv and not use_local_weighting:
        apply_dynamic_range_scaling(args, pet_scale, spect_scale)
        logging.info(
            "Applied dynamic range scaling: pet_scale=%.6g, spect_scale=%.6g",
            pet_scale,
            spect_scale,
        )
    else:
        if use_log_tnv:
            logging.info(
                "Skipping dynamic range scaling for log-TNV "
                "(alpha=%.6g, beta=%.6g remain as configured)",
                args.alpha,
                args.beta,
            )
        if use_local_weighting:
            logging.info(
                "Skipping dynamic range scaling for local weighting "
                "(alpha=%.6g, beta=%.6g will be applied voxel-wise)",
                args.alpha,
                args.beta,
            )

    set_auto_delta_from_scaled_images(args, combined)

    save_args(args, "args.csv")

    # write κ² images
    if kappas is not None:
        for i, image in enumerate(kappas.containers):
            image.write(os.path.join(args.output_path, f"kappa_sq_{i}.hv"))

    if args.no_prior:
        prior = None
        priors_list = []
    else:
        # Set up the prior.
        priors_list = get_prior(
            args, umap, combined, kappas,
            pet_scale=pet_scale, spect_scale=spect_scale
        )
        prior = -SumFunction(*priors_list)

    ui = getattr(args, "update_interval", None)
    update_interval = len(all_funs) if ui is None else ui

    # Set up preconditioners
    precond = get_preconditioners(
        args, s_inv, all_funs, update_interval, priors_list, initial_estimates
    )

    support_mask = None
    if _as_bool(getattr(args, "support_mask_from_sensitivity", False)):
        mask_rel = float(getattr(args, "support_mask_rel_threshold", 1e-3))
        mask_abs = float(getattr(args, "support_mask_abs_threshold", 0.0))
        support_mask_sens = build_support_mask_from_s_inv(
            s_inv,
            rel_threshold=mask_rel,
            abs_threshold=mask_abs,
        )
        support_mask = combine_support_masks(support_mask, support_mask_sens)
        logging.info(
            "Enabled sensitivity support mask (rel_threshold=%.3g, abs_threshold=%.3g).",
            mask_rel,
            mask_abs,
        )
    if _as_bool(getattr(args, "support_mask_from_spect_attenuation", False)):
        attn_rel = float(getattr(args, "support_mask_spect_attn_rel_threshold", 1e-3))
        attn_abs = float(getattr(args, "support_mask_spect_attn_abs_threshold", 1e-2))
        support_mask_attn = build_support_mask_from_spect_attenuation(
            template=s_inv,
            spect_attenuation=spect2pet.direct(spect_data["attenuation"]),
            spect_index=1,
            rel_threshold=attn_rel,
            abs_threshold=attn_abs,
        )
        support_mask = combine_support_masks(support_mask, support_mask_attn)
        logging.info(
            "Enabled SPECT attenuation support mask (rel_threshold=%.3g, abs_threshold=%.3g).",
            attn_rel,
            attn_abs,
        )
    if support_mask is not None and _as_bool(getattr(args, "save_support_mask", False)):
        for i, el in enumerate(support_mask.containers):
            el.write(os.path.join(args.output_path, f"support_mask_{i}.hv"))

    epoch_length = len(all_funs)

    f_obj, probs, prior_prob, prior_in_sampler = build_variance_reduced_function(
        args, all_funs, prior, args.num_subsets, epoch_length, bpos=2
    )

    variance_reduction = getattr(args, "variance_reduction", "saga")
    logging.info(
        "Variance reduction: %s | stochastic functions: %d",
        variance_reduction,
        getattr(f_obj, "num_functions", len(all_funs) + int(prior_in_sampler)),
    )

    if prior_in_sampler and prior_prob is not None:
        target_updates = getattr(args, "prior_updates_per_epoch", None)
        target_str = f"{float(target_updates):.3f}" if target_updates not in (None, False) else "n/a"
        expected_updates = (
            prior_prob * epoch_length / (1.0 - prior_prob) if prior_prob < 1 else float("inf")
        )
        logging.info(
            "Prior sampled with probability %.6f (target=%s updates/epoch, expected≈%.3f).",
            prior_prob,
            target_str,
            expected_updates,
        )
    elif prior is not None:
        logging.info("Prior evaluated deterministically each iteration.")

    objective = -f_obj if prior_in_sampler or prior is None else -SumFunction(f_obj, prior)

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks using shared function
    callbacks = get_callbacks(args, update_interval)

    # Run algorithm using shared function
    subiterations = args.num_epochs * epoch_length
    algo = get_algorithm(
        initial_estimates,
        objective,
        precond,
        step_size,
        update_interval,
        subiterations,
        callbacks,
        support_mask=support_mask,
    )

    # Save results using shared function
    save_results(algo, args)
    save_native_spect_image(
        algo.solution.containers[1],
        pet_to_spect,
        os.path.join(args.output_path, "final_image_1_native.hv"),
    )

    logging.info("Reconstruction complete")


if __name__ == "__main__":
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = argparse.Namespace(**config)

    if getattr(args, "profile", False):
        logging.info("Profiling is enabled. This may slow down the execution.")
        profiler = cProfile.Profile()
        profiler.enable()

        main(args)

        profiler.disable()
        profiler.dump_stats(f"{args.output_path}/profile_data.prof")
        # Output results to a file
        output_file = os.path.join(args.output_path, "profiling_results.txt")
        with open(output_file, "w") as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats("cumulative").print_stats()
        logging.info(f"Profiling results saved to {output_file}")
    else:
        logging.info("Profiling is disabled.")
        main(args)
    logging.info("Execution completed.")
