#!/usr/bin/env python3
"""SETR DTNV reconstruction for multiple bed positions - Simplified version using shared modules."""

import argparse
import cProfile
import logging
import os
import pstats
from typing import Any, List

import numpy as np
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction, SVRGFunction
from cil.optimisation.operators import (
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.priors import WeightedVectorialTotalVariation
from setr.scripts.common import (
    attach_prior_hessian,
    configure_logging,
    get_resampling_operators,
    get_sensitivity_from_subset_objs,
    get_shift_operators,
    init_run_env,
    save_results,
)
from setr.scripts.dtnv_common import (
    compute_kappa_squared_image_from_partitioned_objective,
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_probabilities,
    get_s_inv_from_subset_objs,
    normalise_kappa_squares,
)
from setr.utils import (
    get_pet_am,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
)
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters


def prepare_data(args):
    """
    Prepare theumapimage, PET and SPECT data, and initial estimates.

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
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))

    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    pet_data["initial_image"].write(
        os.path.join(args.output_path, 'initial_image_0.hv')
    )
    spect_data["initial_image"].write(
        os.path.join(args.output_path, 'initial_image_1.hv')
    )

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        args.delta = max(
            pet_data["initial_image"].max() / 1e4,
            spect_data["initial_image"].max() / 1e4,
        ) * min(args.alpha, args.beta)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    # check for nans in all data
    for data in [umap, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(data.as_array()).any():
            logging.warning("An image contains NaNs")
            break

    return umap, pet_data, spect_data, initial_estimates


def get_prior(args, umap, pet_data, spect_data, initial_estimates, bo):
    """Set up vectorial total variation prior for 2bpos."""
    logging.info("Setting up prior")

    # Get kappa weighting - simplified version
    kappa_bdc = bo.direct(initial_estimates)
    kappas = EnhancedBlockDataContainer(*kappa_bdc.containers).get_uniform_copy(1.0)

    # multiply first kappa by alpha/beta for TNV prior
    for i, (ab, el) in enumerate(zip([args.alpha, args.beta], kappas.containers)):
        kappas.containers[i].fill(float(ab) * el)
    logging.info("Kappa images scaled.")

    vtv = WeightedVectorialTotalVariation(
        bo.direct(initial_estimates),
        kappas,
        args.delta,
        anatomical=umap,
        stable=True,
        tail_singular_values=getattr(args, "tail_singular_values", None),
        stencil=getattr(args, "tnv_stencil", args.stencil),
        both_directions=getattr(args, "tnv_both_directions", args.both_directions),
    )
    logging.info("Weighted Vectorial Total Variation prior set up.")
    prior = OperatorCompositionFunction(vtv, bo)
    logging.info("Prior function composed with block operator.")
    return prior


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
):
    """
    Set up data fidelity (objective) functions.

    Returns:
        all_funs: list of block objective functions (PET all beds, then SPECT).
        s_inv:    EnhancedBlockDataContainer of 1/sensitivity images (PET,SPECT).
        kappa_sq_block: EnhancedBlockDataContainer of κ² images (PET,SPECT) in common PET space.
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

    # keep raw copies for κ before operator wrapping
    pet_dfs_raw = [list(df_list) for df_list in pet_dfs]

    # set_up subset objs on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # --- partition SPECT ---
    spect_dfs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"],
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )[2]
    for obj_fun in spect_dfs:
        obj_fun.set_up(spect_data["initial_image"])

    # =========================
    # κ² build *before* op wrapping
    # =========================
    # per-bed PET κ² in bed coords
    pet_kappa_bed_sq = []
    for df_list, suffix in zip(pet_dfs_raw, pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        pet_kappa_bed_sq.append(
            compute_kappa_squared_image_from_partitioned_objective(df_list, tmpl)
        )

    # add across beds (Fisher additivity)
    pet_kappa_sq = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(pet_kappa_bed_sq[i]) for i, unshift_op in enumerate(unshift_ops)]
        )
    )

    logging.info(f"PET κ² images computed and uncombined with shape {pet_kappa_sq.shape}.")

    # SPECT κ²
    spect_kappa_sq = compute_kappa_squared_image_from_partitioned_objective(
        spect_dfs, spect_data["initial_image"]
    )

    logging.info(f"SPECT κ² image computed with shape {spect_kappa_sq.shape}.")

    pet_sens = [get_sensitivity_from_subset_objs(df) for df in pet_dfs]

    spect_s_inv = get_s_inv_from_subset_objs(spect_dfs, spect_data["initial_image"])

    # unshift+combine PET sensitivities to common PET grid
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )
    pet_s_inv = pet_sens_combined.clone()
    pet_sens_array = pet_sens_combined.as_array()
    pet_s_inv.fill(np.reciprocal(pet_sens_array, where=pet_sens_array != 0))
    cyl, _ = get_filters()
    cyl.apply(pet_s_inv)

    s_inv = EnhancedBlockDataContainer(pet_s_inv, spect_s_inv)

    # save s_inv images (unchanged)
    for i, image in enumerate(s_inv.containers):
        image.write(os.path.join(args.output_path, f"s_inv_{i}.hv"))
        logging.info(f"Writing s_inv_{i} with max {image.max()}")

    # --- now wrap PET objectives with uncombine/choose/unshift ---
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(unshift_ops[i], choose_ops[i], uncombine_op),
            )

    # flatten beds
    pet_combined_dfs = [df for bed in pet_dfs for df in bed]

    # block objectives
    pet_dfs_block = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            df,
            order=0,
        )
        for df in pet_combined_dfs
    ]
    spect_dfs_block = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_dfs
    ]

    all_funs = pet_dfs_block + spect_dfs_block

    # bundle κ² (PET in PET space; SPECT still in SPECT space—transform later in get_prior)
    kappa_sq_block = EnhancedBlockDataContainer(pet_kappa_sq, spect_kappa_sq)

    return all_funs, s_inv, kappa_sq_block


def get_preconditioners(
    args: argparse.Namespace,
    s_inv: Any,
    all_funs: List[Any],
    update_interval: int,
    prior: Any,
    initial_estimates: EnhancedBlockDataContainer,
) -> Any:
    """Set up preconditioners for 2bpos."""
    max_vals = [el.max() for el in initial_estimates.containers]
    epsilon = min(el.max() for el in initial_estimates.containers) * 1e-3

    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=epsilon,
        max_vals=max_vals,
        smooth=True,
    )
    if prior is None:
        return bsrem_precond

    prior_precond = ImageFunctionPreconditioner(
        prior.inv_hessian_diag,
        1.0,
        update_interval,
        freeze_iter=np.inf,
        epsilon=epsilon,
    )

    return LehmerMeanPreconditioner(
        [bsrem_precond, prior_precond],
        update_interval=update_interval,
        freeze_iter=len(all_funs) * 10,
    )


def main(args) -> None:
    """Main DTNV 2bpos reconstruction pipeline."""
    configure_logging()

    # Initialize run environment (creates dirs, sets storage scheme, redirects messages)
    msg = init_run_env(args)
    save_args(args, "args.csv")

    # Prepare data
    umap, pet_data, spect_data, initial_estimates = prepare_data(args)

    # find alpha weighting using dynamic range of the initial images (95th percentile)
    pet_max = np.percentile(pet_data["initial_image"].as_array(), 95)
    spect_max = np.percentile(spect_data["initial_image"].as_array(), 95)
    args.alpha = args.alpha * spect_max / pet_max
    logging.info(f"Setting alpha to {args.alpha} based on initial images")

    # Set up operators for multiple bed positions
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    # Set up resampling operators
    spect2pet = get_resampling_operators(pet_data, spect_data)

    # Create combined block operator
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=args.pet_gauss_fwhm,
        )

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.stop_keep_all_views_in_cache,
            gauss_fwhm=args.spect_gauss_fwhm,
            attenuation=True,
        )

    # Set up data fidelity
    all_funs, s_inv, kappa_sq_block = get_data_fidelity(
        args,
        pet_data,
        spect_data,
        get_pet_am_with_res,
        get_spect_am_with_res,
        args.num_subsets,
        uncombine_op,
        unshift_ops,
        choose_ops,
    )

    # cross-modal scaling (XXth pct)
    kappa_sq_block = normalise_kappa_squares(
        kappa_sq_block,
        pct=50,
    )

    # write κ² images
    for i, image in enumerate(kappa_sq_block.containers):
        image.write(os.path.join(args.output_path, f"kappa_sq_{i}.hv"))

    if not args.no_prior:
        # Set up prior
        prior = get_prior(args, umap, pet_data, spect_data, initial_estimates, bo)

        # Scale and attach Hessian to the prior
        prior = -1 / len(all_funs) * prior
        attach_prior_hessian(prior)

        for i, fun in enumerate(all_funs):
            all_funs[i] = SumFunction(fun, prior)
    else:
        prior = None

    update_interval = len(all_funs)

    # Set up preconditioners
    precond = get_preconditioners(args, s_inv, all_funs, update_interval, prior, initial_estimates)

    probs = get_probabilities(args, args.num_subsets, update_interval, bpos=2)

    f_obj = -SVRGFunction(
        all_funs,
        sampler=Sampler.random_with_replacement(
            len(all_funs),
            prob=probs,
        ),
    )

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks using shared function
    callbacks = get_callbacks(args, update_interval)

    # Run algorithm using shared function
    subiterations = args.num_epochs * len(all_funs)
    bsrem = get_algorithm(
        initial_estimates, f_obj, precond, step_size, update_interval, subiterations, callbacks
    )

    # Save results using shared function
    save_results(bsrem, args)

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
