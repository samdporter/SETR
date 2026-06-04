#!/usr/bin/env python3
"""
Subset selection experiment script for DTNV reconstruction.

Tests different combinations of:
- Subset organization (paired vs separate)
- Prior update frequency (always vs as subset)
- Preconditioner types (data-only vs TNV-aware majorisers)

This script reuses code from run_dtnv_1bpos.py and shared modules,
keeping only the experimental logic isolated here.
"""

import logging
import math
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction
from cil.optimisation.operators import (
    CompositionOperator,
)
from sirf.contrib.partitioner import partitioner
from sirf.STIR import ImageData, SeparableGaussianImageFilter

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.cil_extensions.operators import FlipOperator
from recon_core.cil_extensions.utilities import LinearDecayStepSizeRule
from recon_experiments.runners.common import (
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
    build_support_mask_from_s_inv,
    build_variance_reduced_function,
    combine_support_masks,
    dynamic_range_scale_sirf,
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_kappa_squareds,
    get_prior,
    get_preconditioners,
    normalise_kappa_squares,
)
from recon_core.utils import (
    get_pet_am,
    get_pet_data,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
)
from recon_core.utils.io import apply_overrides, load_config, parse_cli, save_args
from recon_core.utils.sirf import get_array, get_filters, get_s_inv_from_objs
from recon_core.cil_extensions.operators.blurring import create_gaussian_blur_operator


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _resolve_bpos(args) -> int:
    bpos = int(getattr(args, "bpos", 1))
    if bpos not in (1, 2):
        raise ValueError(f"bpos must be 1 or 2, got {bpos}")
    return bpos


def _validate_paired_subset_configuration(bpos: int, num_subsets) -> None:
    if bpos == 2 and int(num_subsets[0]) != int(num_subsets[1]):
        raise ValueError(
            "2bpos paired subset mode requires equal PET and SPECT subset counts so that "
            "each combined PET subset group can be paired with one SPECT subset. "
            f"Got num_subsets={list(num_subsets)}."
        )


def _iter_projection_data_for_nan_check(pet_data, spect_data):
    """Yield projection-like data objects for NaN checks across 1- and 2-bed PET data."""
    if "bed_positions" in pet_data:
        for bed in pet_data["bed_positions"].values():
            for key in ("acquisition_data", "normalisation", "additive"):
                if key in bed and bed[key] is not None:
                    yield bed[key]
    else:
        for key in ("acquisition_data", "normalisation", "additive"):
            if key in pet_data and pet_data[key] is not None:
                yield pet_data[key]

    for key in ("acquisition_data", "additive"):
        if key in spect_data and spect_data[key] is not None:
            yield spect_data[key]


def prepare_data(args):
    """
    Prepare the CT image, PET and SPECT data, and initial estimates.

    This is adapted from run_dtnv_1bpos.py with minor adjustments.
    """
    bpos = _resolve_bpos(args)

    if bpos == 2:
        pet_data = get_pet_data_multiple_bed_pos(
            args.pet_data_path,
            tof=getattr(args, "use_tof", False),
            suffixes=["_f1b1", "_f2b1"],
        )

        ct = pet_data["attenuation"]
        ct += (-ct).max()
        ct /= ct.max()
        ct_smooth = SeparableGaussianImageFilter()
        ct_smooth.set_fwhms((2, 2, 2))
        ct_smooth.apply(ct)
        spect_data = get_spect_data(args.spect_data_path)

        cyl, gauss = get_filters(fwhms=(20, 20, 20))
        gauss.apply(spect_data["initial_image"])
        gauss.apply(pet_data["initial_image"])
        cyl.apply(pet_data["initial_image"])

        for data in [ct, pet_data["initial_image"], spect_data["initial_image"]]:
            if np.isnan(get_array(data)).any():
                logging.warning("An image contains NaNs")
                break
        for data in _iter_projection_data_for_nan_check(pet_data, spect_data):
            if np.isnan(get_array(data)).any():
                logging.warning("A ProjData contains NaNs")
                break

        return ct, pet_data, spect_data

    # Get guidance image
    ct = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    ct += (-ct).max()
    ct /= ct.max()
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((0.5, 0.5, 0.5))
    ct_smooth.apply(ct)

    pet_data = get_pet_data(args.pet_data_path)
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    # Check for NaNs
    for data in [ct, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("An image contains NaNs")
            break
    for data in [
        pet_data["acquisition_data"],
        spect_data["acquisition_data"],
        pet_data["normalisation"],
        pet_data["additive"],
        spect_data["additive"],
    ]:
        if np.isnan(get_array(data)).any():
            logging.warning("A ProjData contains NaNs")
            break

    return ct, pet_data, spect_data


def get_data_fidelity_separate(
    args,
    pet_data,
    spect_data,
    get_pet_am,
    get_spect_am,
    num_subsets,
    shared_initial_estimates,
    pet_to_spect,
):
    """
    EXPERIMENTAL: Set up data fidelity with SEPARATE PET and SPECT subsets.

    Returns list: [pet_subset_0, ..., pet_subset_N, spect_subset_0, ..., spect_subset_M]

    This is the experimental subset organization mode.
    """
    logging.info("Partitioning PET data")
    _, _, pet_obj_funs = partitioner.data_partition(
        pet_data["acquisition_data"],
        pet_data["additive"],
        pet_data["normalisation"],
        num_batches=num_subsets[0],
        mode="staggered",
        create_acq_model=get_pet_am,
    )
    logging.info("PET data partitioned; setting up PET objective functions")
    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])
    logging.info("PET objective functions set up; partitioning SPECT data")
    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )
    logging.info("SPECT data partitioned; setting up SPECT objective functions")
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])
    logging.info("SPECT objective functions set up")

    # Create Gaussian blurring operator for PET only
    pet_blur_op = create_gaussian_blur_operator(args.pet_gauss_fwhm, pet_data["initial_image"])

    # Get sensitivity
    s_inv = get_s_inv_from_objs(
        [pet_obj_funs, spect_obj_funs],
        shared_initial_estimates,
        adjoint_ops=[pet_blur_op, pet_to_spect],
    )

    # Wrap PET objectives with Gaussian blurring operator (if specified)
    if pet_blur_op is not None:
        pet_obj_funs = [
            OperatorCompositionFunction(obj_fun, pet_blur_op)
            for obj_fun in pet_obj_funs
        ]
    spect_obj_funs = [
        OperatorCompositionFunction(obj_fun, pet_to_spect)
        for obj_fun in spect_obj_funs
    ]

    # Get kappas if needed (using shared function from dtnv_common)
    if args.use_kappa:
        kappa = get_kappa_squareds(
            [pet_obj_funs, spect_obj_funs],
            [shared_initial_estimates[0], shared_initial_estimates[1]],
        )
        _, gauss = get_filters()
        for kappa_image in kappa.containers:
            gauss.apply(kappa_image)
    else:
        kappa = None

    # Wrap as block objectives (using shared function from dtnv_common)
    pet_obj_funs = [
        get_block_objective(
            shared_initial_estimates[0],
            shared_initial_estimates[1],
            obj_fun,
            order=0,
        )
        for obj_fun in pet_obj_funs
    ]
    spect_obj_funs = [
        get_block_objective(
            shared_initial_estimates[1],
            shared_initial_estimates[0],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_obj_funs
    ]

    all_funs = pet_obj_funs + spect_obj_funs

    return all_funs, s_inv, kappa


def get_data_fidelity_paired(
    args,
    pet_data,
    spect_data,
    get_pet_am,
    get_spect_am,
    num_subsets,
    shared_initial_estimates,
    pet_to_spect,
):
    """
    EXPERIMENTAL: Set up data fidelity with PAIRED PET+SPECT subsets.

    Returns list: [SumFunction(pet_subset_i, spect_subset_i) for i in range(N)]

    This is the experimental subset organization mode.
    """
    _, _, pet_obj_funs = partitioner.data_partition(
        pet_data["acquisition_data"],
        pet_data["additive"],
        pet_data["normalisation"],
        num_batches=num_subsets[0],
        mode="staggered",
        create_acq_model=get_pet_am,
    )
    # STIR's SPECTUBMatrix setup corrupts global projector state that PET set_up reads.
    # Must call PET set_up before SPECT data_partition to avoid SIGSEGV.
    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])

    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    # Create Gaussian blurring operator for PET only
    pet_blur_op = create_gaussian_blur_operator(args.pet_gauss_fwhm, pet_data["initial_image"])

    # Get sensitivity
    s_inv = get_s_inv_from_objs(
        [pet_obj_funs, spect_obj_funs],
        shared_initial_estimates,
        adjoint_ops=[pet_blur_op, pet_to_spect],
    )

    # Wrap PET objectives with Gaussian blurring operator (if specified)
    if pet_blur_op is not None:
        pet_obj_funs = [
            OperatorCompositionFunction(obj_fun, pet_blur_op)
            for obj_fun in pet_obj_funs
        ]
    spect_obj_funs = [
        OperatorCompositionFunction(obj_fun, pet_to_spect)
        for obj_fun in spect_obj_funs
    ]

    # Get kappas if needed (using shared function from dtnv_common)
    if args.use_kappa:
        kappa = get_kappa_squareds(
            [pet_obj_funs, spect_obj_funs],
            [shared_initial_estimates[0], shared_initial_estimates[1]],
        )
        _, gauss = get_filters()
        for kappa_image in kappa.containers:
            gauss.apply(kappa_image)
    else:
        kappa = None

    # Wrap as block objectives (using shared function from dtnv_common)
    pet_obj_funs_block = [
        get_block_objective(
            shared_initial_estimates[0],
            shared_initial_estimates[1],
            obj_fun,
            order=0,
        )
        for obj_fun in pet_obj_funs
    ]
    spect_obj_funs_block = [
        get_block_objective(
            shared_initial_estimates[1],
            shared_initial_estimates[0],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_obj_funs
    ]

    # Create paired SumFunctions
    assert len(pet_obj_funs_block) == len(spect_obj_funs_block), (
        f"PET and SPECT must have same number of subsets for pairing. "
        f"Got {len(pet_obj_funs_block)} PET, {len(spect_obj_funs_block)} SPECT"
    )

    paired_funs = [
        SumFunction(pet_fun, spect_fun)
        for pet_fun, spect_fun in zip(pet_obj_funs_block, spect_obj_funs_block)
    ]

    return paired_funs, s_inv, kappa


def _build_2bpos_block_objectives(
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

    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for obj_fun in pet_dfs[i]:
            obj_fun.set_up(tmpl)

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

    pet_blur_ops = [
        create_gaussian_blur_operator(
            args.pet_gauss_fwhm,
            pet_data["bed_positions"][suffix]["template_image"],
        )
        for suffix in pet_data["bed_positions"]
    ]

    pet_sens = [
        get_sensitivity_from_subset_objs(df, adjoint_operator=op)
        for df, op in zip(pet_dfs, pet_blur_ops)
    ]

    spect_s_inv = get_s_inv_from_subset_objs(
        spect_dfs,
        shared_initial_estimates[1],
        clamp_percentile=99.5,
        adjoint_operator=pet_to_spect,
    )

    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )
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

    for i, _suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            if pet_blur_ops[i] is not None:
                op_chain = CompositionOperator(
                    pet_blur_ops[i],
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                )
            else:
                op_chain = CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                )
            pet_dfs[i][j] = OperatorCompositionFunction(pet_dfs[i][j], op_chain)

    spect_dfs = [
        OperatorCompositionFunction(obj_fun, pet_to_spect)
        for obj_fun in spect_dfs
    ]

    pet_dfs_block_by_bed = [
        [
            get_block_objective(
                shared_initial_estimates[0],
                shared_initial_estimates[1],
                df,
                order=0,
            )
            for df in bed
        ]
        for bed in pet_dfs
    ]
    pet_dfs_block = [df for bed in pet_dfs_block_by_bed for df in bed]
    spect_dfs_block = [
        get_block_objective(
            shared_initial_estimates[1],
            shared_initial_estimates[0],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_dfs
    ]

    if args.use_kappa:
        kappas = get_kappa_squareds(
            [pet_dfs_block, spect_dfs_block],
            [shared_initial_estimates[0], shared_initial_estimates[1]],
        )
    else:
        kappas = None

    return pet_dfs_block_by_bed, pet_dfs_block, spect_dfs_block, s_inv, kappas


def get_data_fidelity_separate_2bpos(
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
    _, pet_dfs_block, spect_dfs_block, s_inv, kappas = _build_2bpos_block_objectives(
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
    )
    return pet_dfs_block + spect_dfs_block, s_inv, kappas


def get_data_fidelity_paired_2bpos(
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
    pet_dfs_block_by_bed, _pet_dfs_block, spect_dfs_block, s_inv, kappas = (
        _build_2bpos_block_objectives(
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
        )
    )

    num_pet_subset_groups = len(pet_dfs_block_by_bed[0]) if pet_dfs_block_by_bed else 0
    if any(len(bed) != num_pet_subset_groups for bed in pet_dfs_block_by_bed):
        raise ValueError("All PET beds must have the same number of subsets for 2bpos pairing.")
    if num_pet_subset_groups != len(spect_dfs_block):
        raise ValueError(
            "2bpos paired subset mode requires equal PET and SPECT subset counts after "
            f"partitioning. Got PET groups={num_pet_subset_groups}, SPECT={len(spect_dfs_block)}."
        )

    paired_funs = [
        SumFunction(*([bed[j] for bed in pet_dfs_block_by_bed] + [spect_dfs_block[j]]))
        for j in range(num_pet_subset_groups)
    ]

    return paired_funs, s_inv, kappas


def get_data_sampling_probabilities(all_funs):
    """Sample uniformly over the actual stochastic data-function list."""
    num_funs = len(all_funs)
    if num_funs <= 0:
        raise ValueError("Subset-selection study requires at least one stochastic data function.")
    prob = 1.0 / num_funs
    return [prob] * num_funs


def get_preconditioner(args, s_inv, all_funs, update_interval, priors_list, initial_estimates):
    """Use the shared DTNV preconditioner builder for current supported methods."""
    return get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=all_funs,
        update_interval=update_interval,
        priors_list=priors_list,
        initial_estimates=initial_estimates,
    )


def calculate_epoch_length_and_prior_updates(subset_mode, prior_mode, num_data_funs, args):
    """
    EXPERIMENTAL: Calculate epoch length and prior updates based on experimental modes.

    This translates the experimental subset_mode and prior_mode into parameters
    that the shared build_variance_reduced_function can understand.

    Returns:
        epoch_length: Total stochastic iterations per epoch
        prior_updates_per_epoch: Number of prior updates per epoch (or None)
    """
    base_epoch_length = num_data_funs

    if prior_mode == "always":
        # Prior evaluated deterministically every iteration (outside sampler)
        # Use the standard epoch length
        epoch_length = base_epoch_length
        prior_updates_per_epoch = None
    elif prior_mode == "subset":
        # Prior sampled stochastically as a subset
        # Check if user specified custom target prior updates
        target_prior_updates = getattr(args, "prior_updates_per_epoch", None)

        if target_prior_updates not in (None, False):
            try:
                prior_updates_per_epoch = float(target_prior_updates)
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid prior_updates_per_epoch=%s. Using default for subset_mode=%s.",
                    target_prior_updates,
                    subset_mode,
                )
                target_prior_updates = None

        if target_prior_updates is None or target_prior_updates is False:
            # Use default ratios based on subset_mode
            if subset_mode == "paired":
                # 18 pairs + 1 prior, ratio 1:2
                # prob(prior) = 1/2, prob(each pair) = 1/2 / 18 = 1/36
                # Expected prior updates per epoch = 18 (when each pair is visited once)
                prior_updates_per_epoch = base_epoch_length
            elif subset_mode == "separate":
                # 18 PET + 18 SPECT + 1 prior, ratio 1:3
                # prob(prior) = 1/3, prob(each data) = 2/3 / 36 = 1/54
                # Expected prior updates per epoch = 18 (when each data subset is visited once)
                prior_updates_per_epoch = base_epoch_length / 2.0
            else:
                raise ValueError(f"Unknown subset_mode: {subset_mode}")

        # Calculate epoch length accounting for prior sampling
        # When prior has probability p, expected iterations for one full epoch is:
        # base_epoch_length / (1 - p) where p = prior_updates / (base_epoch_length + prior_updates)
        prior_prob = prior_updates_per_epoch / (base_epoch_length + prior_updates_per_epoch)
        epoch_length = math.ceil(base_epoch_length / (1.0 - prior_prob))
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    return epoch_length, prior_updates_per_epoch


def main(args) -> None:
    """Main function."""
    configure_logging()
    bpos = _resolve_bpos(args)

    # Validate experimental parameters
    subset_mode = getattr(args, "subset_mode", "separate")
    prior_mode = getattr(args, "prior_mode", "always")
    precond_type = getattr(args, "precond_type", "bsrem")

    if subset_mode not in ["separate", "paired"]:
        raise ValueError(f"subset_mode must be 'separate' or 'paired', got {subset_mode}")
    if prior_mode not in ["always", "subset"]:
        raise ValueError(f"prior_mode must be 'always' or 'subset', got {prior_mode}")
    if subset_mode == "paired":
        _validate_paired_subset_configuration(bpos, args.num_subsets)

    logging.info("=" * 60)
    logging.info("SUBSET SELECTION EXPERIMENT")
    logging.info(f"  Bed positions: {bpos}")
    logging.info(f"  Subset mode: {subset_mode}")
    logging.info(f"  Prior mode: {prior_mode}")
    logging.info(f"  Preconditioner: {precond_type}")
    logging.info(f"  Gamma: {args.gamma_tnv}")
    logging.info("=" * 60)

    # Data preparation
    umap, pet_data, spect_data = prepare_data(args)

    if bpos == 2:
        uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)
    else:
        uncombine_op = None
        unshift_ops = None
        choose_ops = None

    # Set up resampling operators
    spect2pet = get_resampling_operators(args, pet_data, spect_data)
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

    # Set up data fidelity based on EXPERIMENTAL subset mode
    num_subsets = [int(i) for i in args.num_subsets]

    if bpos == 1:
        if subset_mode == "separate":
            all_funs, s_inv, kappas = get_data_fidelity_separate(
                args,
                pet_data,
                spect_data,
                get_pet_am_with_res,
                get_spect_am_with_res,
                num_subsets,
                initial_estimates,
                pet_to_spect,
            )
        else:
            all_funs, s_inv, kappas = get_data_fidelity_paired(
                args,
                pet_data,
                spect_data,
                get_pet_am_with_res,
                get_spect_am_with_res,
                num_subsets,
                initial_estimates,
                pet_to_spect,
            )
    else:
        if subset_mode == "separate":
            all_funs, s_inv, kappas = get_data_fidelity_separate_2bpos(
                args,
                pet_data,
                spect_data,
                get_pet_am_with_res,
                get_spect_am_with_res,
                num_subsets,
                uncombine_op,
                unshift_ops,
                choose_ops,
                initial_estimates,
                pet_to_spect,
            )
        else:
            all_funs, s_inv, kappas = get_data_fidelity_paired_2bpos(
                args,
                pet_data,
                spect_data,
                get_pet_am_with_res,
                get_spect_am_with_res,
                num_subsets,
                uncombine_op,
                unshift_ops,
                choose_ops,
                initial_estimates,
                pet_to_spect,
            )

    # Normalize kappas and scale images (shared functions from dtnv_common)
    kappas = normalise_kappa_squares(kappas) if kappas is not None else None
    combined = initial_estimates
    pet_scale, spect_scale = dynamic_range_scale_sirf(combined[0], combined[1])
    apply_dynamic_range_scaling(args, pet_scale, spect_scale)

    # Set delta (same as run_dtnv_1bpos.py)
    if args.delta is None:
        args.delta = (
            min(
                args.alpha * initial_estimates.containers[0].max(),
                args.beta * initial_estimates.containers[1].max(),
            )
            / 1e3
        )

    save_args(args, "args.csv")

    if kappas is not None:
        for i, kappa in enumerate(kappas.containers):
            logging.info(f"Writing kappa {i} with max {kappa.max()}")
            kappa.write(os.path.join(args.output_path, f"kappa_sq_{i}.hv"))

    for i, el in enumerate(s_inv.containers):
        s_inv.containers[i].write(os.path.join(args.output_path, f"s_inv_{i}.hv"))

    # Set up prior (shared function from dtnv_common)
    if args.no_prior:
        prior = None
        priors_list = []
    else:
        priors_list = get_prior(args, umap, combined, kappas)
        prior = -SumFunction(*priors_list)

    # Calculate epoch length and prior updates based on EXPERIMENTAL modes
    data_epoch_length = len(all_funs)
    epoch_length, prior_updates_per_epoch = calculate_epoch_length_and_prior_updates(
        subset_mode, prior_mode, data_epoch_length, args
    )
    data_probs = get_data_sampling_probabilities(all_funs)

    # Temporarily set prior_updates_per_epoch for build_variance_reduced_function
    original_prior_updates = getattr(args, "prior_updates_per_epoch", None)
    args.prior_updates_per_epoch = prior_updates_per_epoch

    # Build variance-reduced function (shared function from dtnv_common)
    # This handles the prior sampling logic based on prior_updates_per_epoch
    f_obj, probs, prior_prob, prior_in_sampler = build_variance_reduced_function(
        args,
        all_funs,
        prior,
        num_subsets,
        data_epoch_length,
        bpos=bpos,
        data_probs=data_probs,
    )

    # Restore original value
    args.prior_updates_per_epoch = original_prior_updates

    variance_reduction = getattr(args, "variance_reduction", "svrg")
    logging.info(
        "Variance reduction: %s | data functions: %d | stochastic functions: %d",
        variance_reduction,
        len(all_funs),
        getattr(f_obj, "num_functions", len(all_funs) + int(prior_in_sampler)),
    )

    if prior_in_sampler and prior_prob is not None:
        expected_updates = prior_prob * epoch_length
        logging.info(
            "Prior mode: subset (sampled) | prob=%.6f | expected updates/epoch≈%.3f",
            prior_prob,
            expected_updates,
        )
    elif prior is not None:
        logging.info("Prior mode: always (evaluated deterministically each iteration)")

    objective = -f_obj if prior_in_sampler or prior is None else -SumFunction(f_obj, prior)

    # Set up EXPERIMENTAL preconditioner
    ui = getattr(args, "update_interval", None)
    update_interval = epoch_length if ui is None else ui

    precond = get_preconditioner(
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
    if support_mask is not None and _as_bool(getattr(args, "save_support_mask", False)):
        for i, el in enumerate(support_mask.containers):
            el.write(os.path.join(args.output_path, f"support_mask_{i}.hv"))

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks (shared function from dtnv_common)
    callbacks = get_callbacks(args, update_interval)

    # Run algorithm (shared function from dtnv_common)
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

    save_results(algo, args)
    save_native_spect_image(
        algo.solution.containers[1],
        pet_to_spect,
        os.path.join(args.output_path, "final_image_1_native.hv"),
    )
    logging.info("Done")


if __name__ == "__main__":
    cli = parse_cli()
    cfg_dict = load_config(cli.config)
    cfg_dict = apply_overrides(cfg_dict, cli.override)

    args = SimpleNamespace(**cfg_dict)

    msg = init_run_env(args)

    main(args)
