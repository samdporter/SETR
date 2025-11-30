#!/usr/bin/env python3
"""
Subset selection experiment script for DTNV reconstruction.

Tests different combinations of:
- Subset organization (paired vs separate)
- Prior update frequency (always vs as subset)
- Preconditioner types (BSREM vs VTV variants)
"""

import logging
import math
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.functions import OperatorCompositionFunction, SAGAFunction, SumFunction, SVRGFunction
from cil.optimisation.operators import (
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from sirf.STIR import ImageData

from setr.cil_extensions.callbacks import ComputeMetricsCallback
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.cil_extensions.operators import FlipOperator, TruncationOperator
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.scripts.common import (
    attach_prior_hessian,
    configure_logging,
    get_resampling_operators,
    init_run_env,
    save_results,
)
from setr.scripts.dtnv_common import (
    apply_dynamic_range_scaling,
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_kappa_squareds,
    get_prior,
    dynamic_range_scale_sirf,
    normalise_kappa_squares,
)
from setr.utils import get_pet_am, get_pet_data, get_spect_am, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.metrics import create_mask_from_threshold
from setr.utils.sirf import get_array, get_filters, get_s_inv_from_objs
from setr.cil_extensions.operators.blurring import create_gaussian_blur_operator


def prepare_data(args):
    """Prepare CT, PET and SPECT data."""
    # Get guidance image
    ct = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    ct += (-ct).max()
    ct /= ct.max()

    pet_data = get_pet_data(args.pet_data_path)
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters()
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    # Check for NaNs
    for data in [ct, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("An image contains NaNs")
            break

    return ct, pet_data, spect_data


def get_data_fidelity_separate(args, pet_data, spect_data, get_pet_am, get_spect_am, num_subsets):
    """
    Set up data fidelity with SEPARATE PET and SPECT subsets.
    Returns list: [pet_subset_0, ..., pet_subset_N, spect_subset_0, ..., spect_subset_M]
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
    # SPECT uses image_data_processor which works correctly for SPECT projectors
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

    # Get kappas if needed
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

    # Wrap as block objectives
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
    Set up data fidelity with PAIRED PET+SPECT subsets.
    Returns list: [SumFunction(pet_subset_i, spect_subset_i) for i in range(N)]
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
    # SPECT uses image_data_processor which works correctly for SPECT projectors
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

    # Get kappas if needed
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

    # Wrap as block objectives
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
    Create preconditioner based on precond_type parameter.

    Args:
        args: Namespace with precond_type in {"bsrem", "vtv_svd_principal_alpha", "vtv_frobenius_surrogate_pd"}
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
    # (objective always uses fast hessian)
    logging.info(f"Creating prior preconditioners with hessian_type={hessian_type}")

    # CRITICAL: Use max_value to cap the inverse Hessian preconditioner
    # Without this, 1/(small Hessian at FOV edges) → huge preconditioner → divergence
    # Cap at the scale of the BSREM preconditioner to keep both on same scale
    # BSREM scale ≈ x / sensitivity ≈ x * s_inv, so use max(x) * max(s_inv) as upper bound
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
    )


def get_probabilities_for_mode(subset_mode, prior_mode, num_data_funs, target_prior_updates=None):
    """
    Calculate sampling probabilities based on subset and prior modes.

    Args:
        subset_mode: "separate" or "paired"
        prior_mode: "always" or "subset"
        num_data_funs: Number of data fidelity functions
        target_prior_updates: Desired prior updates per epoch (optional)

    Returns:
        probs: List of probabilities (None if prior_mode="always")
        prior_prob: Probability for prior (None if prior_mode="always")
    """
    if prior_mode == "always":
        # Prior in outer SumFunction, not sampled
        probs = [1.0 / num_data_funs] * num_data_funs
        prior_prob = None
    elif prior_mode == "subset":
        prob_each = None
        if target_prior_updates not in (None, False):
            try:
                target_prior_updates = float(target_prior_updates)
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid prior_updates_per_epoch=%s. Falling back to default prior probability.",
                    target_prior_updates,
                )
                target_prior_updates = None

        if target_prior_updates is not None and target_prior_updates > 0:
            prior_prob = target_prior_updates / (num_data_funs + target_prior_updates)
            prob_each = (1.0 - prior_prob) / num_data_funs
        else:
            if subset_mode == "paired":
                # 18 pairs + 1 prior, ratio 1:2
                # prob(prior) = 1/2, prob(each pair) = 1/2 / 18 = 1/36
                prior_prob = 0.5
                prob_each = 0.5 / num_data_funs
            elif subset_mode == "separate":
                # 18 PET + 18 SPECT + 1 prior, ratio 1:3
                # prob(prior) = 1/3, prob(each data) = 2/3 / 36 = 1/54
                prior_prob = 1.0 / 3.0
                prob_each = (1.0 - prior_prob) / num_data_funs
            else:
                raise ValueError(f"Unknown subset_mode: {subset_mode}")

        probs = [prob_each] * num_data_funs
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    # Verify probabilities sum correctly
    total_prob = sum(probs) + (prior_prob if prior_prob is not None else 0)
    assert abs(total_prob - 1.0) < 1e-10, f"Probabilities sum to {total_prob}, not 1.0"

    return probs, prior_prob


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

    # Data preparation
    umap, pet_data, spect_data = prepare_data(args)

    # Set up resampling operators
    spect2pet = get_resampling_operators(pet_data, spect_data)

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

    # Set up data fidelity based on subset mode
    num_subsets = [int(i) for i in args.num_subsets]

    if subset_mode == "separate":
        all_funs, s_inv, kappas = get_data_fidelity_separate(
            args, pet_data, spect_data, get_pet_am_with_res, get_spect_am_with_res, num_subsets
        )
    elif subset_mode == "paired":
        all_funs, s_inv, kappas = get_data_fidelity_paired(
            args, pet_data, spect_data, get_pet_am_with_res, get_spect_am_with_res, num_subsets
        )

    if args.flip:
        spect2pet = CompositionOperator(
            spect2pet, FlipOperator(spect_data["initial_image"], axis=(0, 2))
        )

    bo = BlockOperator(
        CompositionOperator(
            IdentityOperator(pet_data["initial_image"]),
            TruncationOperator(pet_data["initial_image"]),
        ),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    kappas = normalise_kappa_squares(bo.direct(kappas)) if kappas else None
    combined = bo.direct(initial_estimates)
    pet_scale, spect_scale = dynamic_range_scale_sirf(
        combined[0],
        combined[1],
    )
    apply_dynamic_range_scaling(args, pet_scale, spect_scale)

    # Set delta
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

    # Set up prior
    if args.no_prior:
        prior = None
        priors_list = []
    else:
        # Create priors with appropriate hessian type
        # For objective, always use "fast" (default)
        priors_list = get_prior(args, umap, combined, bo, kappas)
        for i, p in enumerate(priors_list):
            attach_prior_hessian(priors_list[i])
        prior = -SumFunction(*priors_list)

    # Set up probabilities and stochastic objective based on prior mode
    raw_target_prior_updates = getattr(args, "prior_updates_per_epoch", None)
    target_prior_updates = None
    if raw_target_prior_updates not in (None, False):
        try:
            target_prior_updates = float(raw_target_prior_updates)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid prior_updates_per_epoch=%s. Falling back to default prior probability.",
                raw_target_prior_updates,
            )
    data_probs, prior_prob = get_probabilities_for_mode(
        subset_mode, prior_mode, len(all_funs), target_prior_updates
    )

    base_epoch_length = len(all_funs)
    epoch_length = base_epoch_length

    if prior_mode == "subset":
        # Expected number of iterations required for each data subset to be
        # visited once when the prior is sampled with probability prior_prob.
        epoch_length = math.ceil(base_epoch_length / (1.0 - prior_prob))

    ui = getattr(args, "update_interval", None)
    update_interval = ui if ui is not None else epoch_length

    variance_reduction = getattr(args, "variance_reduction", "svrg")
    variance_reduction = str(variance_reduction).lower()
    snapshot_factor = getattr(args, "snapshot_interval_factor", None)

    def _create_sampler(num_functions, probs):
        return Sampler.random_with_replacement(num_functions, prob=probs)

    def _create_vr_function(functions, probs):
        sampler = _create_sampler(len(functions), probs)
        if variance_reduction == "svrg":
            if snapshot_factor is None:
                snapshot_interval = epoch_length * 2
            else:
                try:
                    snapshot_interval = max(
                        1, int(round(epoch_length * float(snapshot_factor)))
                    )
                except (TypeError, ValueError):
                    logging.warning(
                        "Invalid snapshot_interval_factor=%s; defaulting to 2 * epoch_length.",
                        snapshot_factor,
                    )
                    snapshot_interval = epoch_length * 2

            return SVRGFunction(
                functions,
                sampler=sampler,
                snapshot_update_interval=snapshot_interval,
                store_gradients=True,
            )
        elif variance_reduction == "saga":
            return SAGAFunction(functions, sampler=sampler)
        else:
            raise ValueError("variance_reduction must be 'svrg' or 'saga'")

    # Set up preconditioner
    precond = get_preconditioner(
        args, s_inv, all_funs, update_interval, priors_list, initial_estimates
    )

    if prior_mode == "always":
        # Prior in outer SumFunction
        logging.info(
            "Prior mode: always (evaluated every iteration) | variance reduction: %s",
            variance_reduction,
        )
        f_obj = _create_vr_function(all_funs, data_probs)
        objective = -SumFunction(f_obj, prior) if prior else -f_obj
    elif prior_mode == "subset":
        # Prior as a separate subset
        logging.info(
            "Prior mode: subset (prob=%.6f) | variance reduction: %s",
            prior_prob,
            variance_reduction,
        )
        if prior is None:
            raise ValueError("Cannot use prior_mode='subset' with no_prior=True")

        # Add prior to list of functions
        all_funs_with_prior = all_funs + [prior]
        probs_with_prior = data_probs + [prior_prob]

        # epoch_length already accounts for the additional iterations needed
        # when the prior is sampled; reuse it for logging and snapshot cadence.

        logging.info(
            f"Total functions: {len(all_funs_with_prior)} ({len(all_funs)} data + 1 prior)"
        )
        logging.info(f"Prior probability: {prior_prob:.4f}")
        logging.info(f"Each data function probability: {data_probs[0]:.6f}")
        if target_prior_updates not in (None, False) and prior_prob is not None:
            logging.info(
                "Target prior updates per epoch: %.3f | Expected ≈ %.3f (based on sampler).",
                float(target_prior_updates),
                prior_prob * epoch_length,
            )
        else:
            logging.info(
                "Expected prior updates per epoch (based on sampler): %.3f.",
                prior_prob * epoch_length,
            )

        f_obj = _create_vr_function(
            all_funs_with_prior,
            probs_with_prior,
        )
        objective = -f_obj

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks
    callbacks = get_callbacks(args, update_interval)

    # Add metrics callback if reference path is provided
    if getattr(args, "compute_metrics", False) and getattr(args, "reference_path", None):
        logging.info("Setting up metrics computation...")

        try:
            # Load reference image (should be BlockDataContainer saved from convergence run)
            reference_pet = ImageData(os.path.join(args.reference_path, "image_0_final.hv"))
            reference_spect = ImageData(os.path.join(args.reference_path, "image_1_final.hv"))
            reference = EnhancedBlockDataContainer(reference_pet, reference_spect)

            # Create mask from reference if threshold specified
            mask = None
            mask_threshold = getattr(args, "mask_threshold", None)
            if mask_threshold is not None and mask_threshold > 0:
                logging.info(f"Creating mask with threshold {mask_threshold}")
                # Create mask for each modality
                mask_pet = create_mask_from_threshold(
                    reference_pet,
                    threshold=mask_threshold * float(reference_pet.max()),
                    mode="greater",
                )
                mask_spect = create_mask_from_threshold(
                    reference_spect,
                    threshold=mask_threshold * float(reference_spect.max()),
                    mode="greater",
                )
                mask = EnhancedBlockDataContainer(mask_pet, mask_spect)

                # Log mask coverage
                mask_pet_arr = get_array(mask_pet)
                mask_spect_arr = get_array(mask_spect)
                logging.info(
                    f"Mask coverage: PET {100 * mask_pet_arr.sum() / mask_pet_arr.size:.1f}%, "
                    f"SPECT {100 * mask_spect_arr.sum() / mask_spect_arr.size:.1f}%"
                )

            metrics_interval = getattr(args, "metrics_interval", None) or update_interval
            metrics_normalization = getattr(args, "metrics_normalization", "range")

            metrics_callback = ComputeMetricsCallback(
                reference=reference,
                filename=os.path.join(args.output_path, "metrics"),
                interval=metrics_interval,
                mask=mask,
                normalization=metrics_normalization,
                verbose=True,
            )
            callbacks.append(metrics_callback)
            logging.info(f"Metrics will be computed every {metrics_interval} iterations")

        except Exception as e:
            logging.warning(f"Failed to set up metrics callback: {e}")
            logging.warning("Continuing without metrics computation")

    # Run algorithm
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
