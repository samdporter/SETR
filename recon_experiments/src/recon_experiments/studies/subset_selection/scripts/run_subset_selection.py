#!/usr/bin/env python3
"""
Subset selection experiment script for DTNV reconstruction.

Tests different combinations of:
- Subset organization (paired vs separate)
- Prior update frequency (always vs as subset)
- Preconditioner types (BSREM vs VTV variants)

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
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)
from sirf.contrib.partitioner import partitioner
from sirf.STIR import ImageData, SeparableGaussianImageFilter

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.cil_extensions.operators import FlipOperator
from recon_core.cil_extensions.utilities import LinearDecayStepSizeRule
from recon_core.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from recon_experiments.runners.common import (
    attach_prior_hessian,
    configure_logging,
    get_resampling_operators,
    init_run_env,
    save_results,
)
from recon_experiments.runners.dtnv_common import (
    apply_dynamic_range_scaling,
    build_variance_reduced_function,
    dynamic_range_scale_sirf,
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_kappa_squareds,
    get_prior,
    normalise_kappa_squares,
)
from recon_core.utils import get_pet_am, get_pet_data, get_spect_am, get_spect_data
from recon_core.utils.io import apply_overrides, load_config, parse_cli, save_args
from recon_core.utils.sirf import get_array, get_filters, get_s_inv_from_objs
from recon_core.cil_extensions.operators.blurring import create_gaussian_blur_operator


def prepare_data(args):
    """
    Prepare the CT image, PET and SPECT data, and initial estimates.

    This is adapted from run_dtnv_1bpos.py with minor adjustments.
    """
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


def get_data_fidelity_separate(args, pet_data, spect_data, get_pet_am, get_spect_am, num_subsets):
    """
    EXPERIMENTAL: Set up data fidelity with SEPARATE PET and SPECT subsets.

    Returns list: [pet_subset_0, ..., pet_subset_N, spect_subset_0, ..., spect_subset_M]

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
    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )

    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    # Create Gaussian blurring operator for PET only
    pet_blur_op = create_gaussian_blur_operator(args.pet_gauss_fwhm, pet_data["initial_image"])

    # Get sensitivity
    s_inv = get_s_inv_from_objs(
        [pet_obj_funs, spect_obj_funs],
        EnhancedBlockDataContainer(pet_data["initial_image"], spect_data["initial_image"]),
        adjoint_ops=[pet_blur_op, None],
    )

    # Wrap PET objectives with Gaussian blurring operator (if specified)
    if pet_blur_op is not None:
        pet_obj_funs = [
            OperatorCompositionFunction(obj_fun, pet_blur_op)
            for obj_fun in pet_obj_funs
        ]

    # Get kappas if needed (using shared function from dtnv_common)
    if args.use_kappa:
        kappa = get_kappa_squareds(
            [pet_obj_funs, spect_obj_funs],
            [pet_data["initial_image"], spect_data["initial_image"]],
        )
        _, gauss = get_filters()
        for kappa_image in kappa.containers:
            gauss.apply(kappa_image)
    else:
        kappa = None

    # Wrap as block objectives (using shared function from dtnv_common)
    pet_obj_funs = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            obj_fun,
            order=0,
        )
        for obj_fun in pet_obj_funs
    ]
    spect_obj_funs = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_obj_funs
    ]

    all_funs = pet_obj_funs + spect_obj_funs

    return all_funs, s_inv, kappa


def get_data_fidelity_paired(args, pet_data, spect_data, get_pet_am, get_spect_am, num_subsets):
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
    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )

    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    # Create Gaussian blurring operator for PET only
    pet_blur_op = create_gaussian_blur_operator(args.pet_gauss_fwhm, pet_data["initial_image"])

    # Get sensitivity
    s_inv = get_s_inv_from_objs(
        [pet_obj_funs, spect_obj_funs],
        EnhancedBlockDataContainer(pet_data["initial_image"], spect_data["initial_image"]),
        adjoint_ops=[pet_blur_op, None],
    )

    # Wrap PET objectives with Gaussian blurring operator (if specified)
    if pet_blur_op is not None:
        pet_obj_funs = [
            OperatorCompositionFunction(obj_fun, pet_blur_op)
            for obj_fun in pet_obj_funs
        ]

    # Get kappas if needed (using shared function from dtnv_common)
    if args.use_kappa:
        kappa = get_kappa_squareds(
            [pet_obj_funs, spect_obj_funs],
            [pet_data["initial_image"], spect_data["initial_image"]],
        )
        _, gauss = get_filters()
        for kappa_image in kappa.containers:
            gauss.apply(kappa_image)
    else:
        kappa = None

    # Wrap as block objectives (using shared function from dtnv_common)
    pet_obj_funs_block = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            obj_fun,
            order=0,
        )
        for obj_fun in pet_obj_funs
    ]
    spect_obj_funs_block = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
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


def get_preconditioner(args, s_inv, all_funs, update_interval, priors_list, initial_estimates):
    """
    EXPERIMENTAL: Create preconditioner based on precond_type parameter.

    Supports:
    - "bsrem": BSREM preconditioner only
    - "vtv_svd_principal_alpha": BSREM + VTV Hessian (SVD principal alpha variant)
    - "vtv_frobenius_surrogate_pd": BSREM + VTV Hessian (Frobenius surrogate PD variant)

    This is the experimental preconditioner variant logic.
    """
    precond_type = getattr(args, "precond_type", "bsrem")

    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=0,
        smooth=True,
    )

    if precond_type == "bsrem":
        return bsrem_precond

    # VTV preconditioner variants - need prior preconditioners
    if priors_list is None or len(priors_list) == 0:
        logging.warning(
            f"Requested precond_type={precond_type} but no priors available. Using BSREM."
        )
        return bsrem_precond

    # Determine hessian type based on precond_type
    if precond_type == "vtv_svd_principal_alpha":
        hessian_type = "svd_principal_alpha"
    elif precond_type == "vtv_frobenius_surrogate_pd":
        hessian_type = "frobenius_surrogate_pd"
    else:
        logging.warning(f"Unknown precond_type={precond_type}. Using BSREM.")
        return bsrem_precond

    # Re-create priors with specified hessian type for preconditioner
    logging.info(f"Creating prior preconditioners with hessian_type={hessian_type}")

    # CRITICAL: Use max_value to cap the inverse Hessian preconditioner
    # Without this, 1/(small Hessian at FOV edges) → huge preconditioner → divergence
    max_precond_value = 10.0 * max(
        con.max() * s_inv_con.max()
        for con, s_inv_con in zip(initial_estimates.containers, s_inv.containers)
    )

    prior_precond = [
        ImageFunctionPreconditioner(
            p.inv_hessian_diag,
            1,
            freeze_iter=np.inf,
            epsilon=0,
            max_value=max_precond_value,
        )
        for p in priors_list
    ]

    return LehmerMeanPreconditioner(
        [bsrem_precond, *prior_precond],
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=0,
        p=0.0,
    )


def calculate_epoch_length_and_prior_updates(subset_mode, prior_mode, num_data_funs, args):
    """
    EXPERIMENTAL: Calculate epoch length and prior updates based on experimental modes.

    This translates the experimental subset_mode and prior_mode into parameters
    that the shared build_variance_reduced_function can understand.

    Returns:
        epoch_length: Number of iterations per epoch
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

    # Validate experimental parameters
    subset_mode = getattr(args, "subset_mode", "separate")
    prior_mode = getattr(args, "prior_mode", "always")
    precond_type = getattr(args, "precond_type", "bsrem")

    if subset_mode not in ["separate", "paired"]:
        raise ValueError(f"subset_mode must be 'separate' or 'paired', got {subset_mode}")
    if prior_mode not in ["always", "subset"]:
        raise ValueError(f"prior_mode must be 'always' or 'subset', got {prior_mode}")

    logging.info("=" * 60)
    logging.info("SUBSET SELECTION EXPERIMENT")
    logging.info(f"  Subset mode: {subset_mode}")
    logging.info(f"  Prior mode: {prior_mode}")
    logging.info(f"  Preconditioner: {precond_type}")
    logging.info(f"  Gamma: {args.gamma_tnv}")
    logging.info("=" * 60)

    # Data preparation (reused from run_dtnv_1bpos.py)
    umap, pet_data, spect_data = prepare_data(args)

    # Set up resampling operators (shared function)
    spect2pet = get_resampling_operators(args, pet_data, spect_data)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
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

    if subset_mode == "separate":
        all_funs, s_inv, kappas = get_data_fidelity_separate(
            args, pet_data, spect_data, get_pet_am_with_res, get_spect_am_with_res, num_subsets
        )
    elif subset_mode == "paired":
        all_funs, s_inv, kappas = get_data_fidelity_paired(
            args, pet_data, spect_data, get_pet_am_with_res, get_spect_am_with_res, num_subsets
        )

    if getattr(args, "flip", False):
        spect2pet = CompositionOperator(
            spect2pet, FlipOperator(spect_data["initial_image"], axis=(0, 2))
        )

    # Block operator (same as run_dtnv_1bpos.py)
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    # Normalize kappas and scale images (shared functions from dtnv_common)
    kappas = normalise_kappa_squares(bo.direct(kappas)) if kappas else None
    combined = EnhancedBlockDataContainer(*bo.direct(initial_estimates).containers)
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

    if kappas:
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
        priors_list = get_prior(args, umap, combined, bo, kappas)
        for i, p in enumerate(priors_list):
            attach_prior_hessian(priors_list[i])
        prior = -SumFunction(*priors_list)

    # Calculate epoch length and prior updates based on EXPERIMENTAL modes
    base_epoch_length = len(all_funs)
    epoch_length, prior_updates_per_epoch = calculate_epoch_length_and_prior_updates(
        subset_mode, prior_mode, base_epoch_length, args
    )

    # Temporarily set prior_updates_per_epoch for build_variance_reduced_function
    original_prior_updates = getattr(args, "prior_updates_per_epoch", None)
    args.prior_updates_per_epoch = prior_updates_per_epoch

    # Build variance-reduced function (shared function from dtnv_common)
    # This handles the prior sampling logic based on prior_updates_per_epoch
    f_obj, probs, prior_prob, prior_in_sampler = build_variance_reduced_function(
        args, all_funs, prior, num_subsets, epoch_length, bpos=1
    )

    # Restore original value
    args.prior_updates_per_epoch = original_prior_updates

    variance_reduction = getattr(args, "variance_reduction", "svrg")
    logging.info(
        "Variance reduction: %s | stochastic functions: %d",
        variance_reduction,
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
    )

    save_results(algo, args)
    logging.info("Done")


if __name__ == "__main__":
    cli = parse_cli()
    cfg_dict = load_config(cli.config)
    cfg_dict = apply_overrides(cfg_dict, cli.override)

    args = SimpleNamespace(**cfg_dict)

    msg = init_run_env(args)

    main(args)
