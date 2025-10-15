"""Shared utilities for DTNV (run_dtnv_*) reconstruction scripts.

Only contains functions that are IDENTICAL between run_dtnv_1bpos.py and run_dtnv_2bpos.py.
"""

import argparse
import logging
import os
from typing import Any, List

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    SAGAFunction,
    SVRGFunction,
    KullbackLeibler,
    OperatorCompositionFunction,
)
from cil.optimisation.operators import (
    BlockOperator,
    IdentityOperator,
    ZeroOperator,
)
from cil.optimisation.utilities import Sampler

from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import ScalingOperator
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.priors import (
    WeightedRDP,
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
)
from setr.utils.sirf import get_array

ISTA.update = ista_update_step


def get_kappa_squareds(obj_funs_list, image_list, normalise=True):
    """
    Compute the kappa squared images for each objective function.

    Returns:
        kappa_squareds: List of kappa squared images.
    """
    kappa_squareds = []
    kappa_squareds.extend(
        compute_kappa_squared_image_from_partitioned_objective(obj_funs, image)
        for obj_funs, image in zip(obj_funs_list, image_list)
    )
    return EnhancedBlockDataContainer(*kappa_squareds)


def get_callbacks(args, update_interval: int) -> List[Any]:
    """Set up callbacks for DTNV algorithm monitoring."""
    callbacks = [
        SaveImageCallback(os.path.join(args.output_path, "image"), update_interval),
        PrintObjectiveCallback(update_interval),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), update_interval),
    ]

    if getattr(args, "save_gradients", False):
        callbacks.append(
            SaveGradientUpdateCallback(os.path.join(args.output_path, "gradient"), update_interval)
        )

    if getattr(args, "save_preconditioners", False):
        callbacks.append(
            SavePreconditionerCallback(
                os.path.join(args.output_path, "preconditioner"), update_interval
            )
        )

    return callbacks


def get_algorithm(
    init_solution: Any,
    f_obj: Any,
    precond: Any,
    step_size: Any,
    update_interval: int,
    subiterations: int,
    callbacks: List[Any],
) -> ISTA:
    """Create and run DTNV ISTA algorithm (identical in both DTNV scripts)."""
    algo = ISTA(
        initial=init_solution,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size,
        update_objective_interval=update_interval,
    )
    algo.run(subiterations, verbose=1, callbacks=callbacks)
    return algo


def get_block_objective(desired_image, other_image, obj_fun, scale=1, order=0):
    """Returns a block CIL objective function for the given SIRF objective function"""

    # Set up zero operators
    o2d_zero = ZeroOperator(other_image, desired_image)
    if scale == 1:
        d2d_id = IdentityOperator(desired_image)
    else:
        d2d_id = ScalingOperator(scale, desired_image)

    if order == 0:
        return OperatorCompositionFunction(obj_fun, BlockOperator(d2d_id, o2d_zero, shape=(1, 2)))
    elif order == 1:
        return OperatorCompositionFunction(obj_fun, BlockOperator(o2d_zero, d2d_id, shape=(1, 2)))
    else:
        raise ValueError("Order must be 0 or 1")


def get_preconditioners(
    args: argparse.Namespace,
    s_inv: Any,
    all_funs: List[Any],
    update_interval: int,
    priors_list: Any,
    initial_estimates: EnhancedBlockDataContainer,
) -> Any:
    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=0,
        smooth=True,
    )
    if priors_list is None:
        return bsrem_precond

    prior_precond = [
        ImageFunctionPreconditioner(
            p.inv_hessian_diag,
            1,
            freeze_iter=np.inf,
            epsilon=0,
        )
        for p in priors_list
    ]

    return LehmerMeanPreconditioner(
        [bsrem_precond, *prior_precond],
        update_interval=1,
        freeze_iter=len(all_funs) * 10,
        epsilon=0,
    )


def get_probabilities(args, num_subsets, update_interval, bpos=1):
    pet_probs = [1 / update_interval] * (num_subsets[0] * bpos)
    spect_probs = [1 / update_interval] * num_subsets[1]

    probs = pet_probs + spect_probs
    assert abs(sum(probs) - 1) < 1e-10, (
        f"Probabilities do not sum to 1: {sum(probs)}. "
        f"Pet: {sum(pet_probs)}, Spect: {sum(spect_probs)}"
    )
    return probs


def build_variance_reduced_function(
    args: argparse.Namespace,
    all_funs: List[Any],
    prior: Any,
    num_subsets: List[int],
    epoch_length: int,
    bpos: int = 1,
):
    """
    Construct a variance-reduced stochastic function (SVRG or SAGA) with optional prior sampling.

    Args:
        args: Configuration namespace.
        all_funs: List of data fidelity functions.
        prior: Prior function (already signed as required by the caller). Use ``None`` to disable.
        num_subsets: [n_pet, n_spect] subset counts.
        epoch_length: Number of data-function evaluations per epoch (typically len(all_funs)).
        bpos: Bed positions multiplier for PET subsets.

    Returns:
        f_obj: Instantiated variance-reduced function.
        probs: Sampling probabilities passed to the sampler.
        prior_prob: Sampling probability allocated to the prior (``None`` if not sampled).
        prior_in_sampler: ``True`` when the prior participates in stochastic updates.
    """

    data_probs = get_probabilities(args, num_subsets, epoch_length, bpos=bpos)

    prior_updates = getattr(args, "prior_updates_per_epoch", None)
    prior_in_sampler = prior is not None and prior_updates not in (None, False)
    prior_prob = None

    if prior_in_sampler:
        try:
            prior_updates = float(prior_updates)
        except (TypeError, ValueError):
            logging.warning(
                "Ignoring prior_updates_per_epoch=%s (non-numeric); falling back to prior outside sampler.",
                prior_updates,
            )
            prior_in_sampler = False
        else:
            if prior_updates <= 0:
                logging.info(
                    "prior_updates_per_epoch <= 0; keeping prior outside stochastic sampler."
                )
                prior_in_sampler = False

    stochastic_functions = list(all_funs)
    if prior_in_sampler:
        prior_prob = prior_updates / (epoch_length + prior_updates)
        data_scale = 1.0 - prior_prob
        data_probs = [p * data_scale for p in data_probs]
        probs = data_probs + [prior_prob]
        stochastic_functions.append(prior)
    else:
        probs = data_probs

    sampler = Sampler.random_with_replacement(len(stochastic_functions), prob=probs)
    variance_reduction = getattr(args, "variance_reduction", "svrg")
    variance_reduction = str(variance_reduction).lower()

    if variance_reduction == "svrg":
        snapshot_factor = getattr(args, "snapshot_interval_factor", None)
        if snapshot_factor is None:
            snapshot_interval = epoch_length * 2
        else:
            try:
                snapshot_interval = max(1, int(round(epoch_length * float(snapshot_factor))))
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid snapshot_interval_factor=%s; defaulting to 2 * epoch_length.",
                    snapshot_factor,
                )
                snapshot_interval = epoch_length * 2

        f_obj = SVRGFunction(
            stochastic_functions,
            sampler=sampler,
            snapshot_update_interval=snapshot_interval,
            store_gradients=True,
        )
    elif variance_reduction == "saga":
        f_obj = SAGAFunction(stochastic_functions, sampler=sampler)
    else:
        raise ValueError("variance_reduction must be 'svrg' or 'saga'")

    return f_obj, probs, prior_prob, prior_in_sampler


def compute_kappa_squared_image_from_partitioned_objective(obj_funs, init_img):
    """Compute kappa-squared weighting image from objective function Hessians.

    Computes κ²(x) = Σ_i H_i(init_img) · 1 where H_i represents the Hessian
    of each objective function component. This provides voxel-wise weighting
    for cross-modal regularization in synergistic reconstruction.

    Args:
        obj_funs: List of objective functions that support multiply_with_Hessian method.
        init_img: Initial image estimate used for Hessian evaluation.

    Returns:
        ImageData: Kappa-squared weighting image with absolute values applied.
    """
    out = init_img.get_uniform_copy(0)  # accumulator zeros
    ones = init_img.get_uniform_copy(1)  # vector of ones

    for obj_fun in obj_funs:
        g = obj_fun
        while hasattr(g, "function"):
            g = g.function

        h1 = g.multiply_with_Hessian(init_img, ones)
        out += h1

    out = out.abs()
    return out


def normalise_kappa_squares(kappa_block, pct=95):
    """
    Scale each κ² image so its `pct` percentile == 1.
    """
    arrays = [get_array(im) for im in kappa_block.containers]
    pvals = [np.percentile(a, pct) for a in arrays]
    for im, p in zip(kappa_block.containers, pvals):
        if p > 1e-12:
            logging.info(
                f"Normalising kappa image with max {im.max()} to percentile {pct} value {p}"
            )
            im *= 1.0 / p
    return kappa_block


def set_up_partitioned_objectives(pet_data, spect_data, pet_obj_funs, spect_obj_funs):
    """Returns a CIL SumFunction for the partitioned objective functions"""

    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])

    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    return pet_obj_funs, spect_obj_funs


def set_up_kl_objectives(
    pet_data, spect_data, pet_datas, pet_norms, spect_datas, pet_ams, spect_ams
):
    """
    Returns a CIL SumFunction using KL objective functions
    for the PET and SPECT data and acq models
    """

    for d, am in zip(pet_datas, pet_ams):
        am.set_up(d, pet_data["initial_image"])

    for d, am in zip(spect_datas, spect_ams):
        am.set_up(d, spect_data["initial_image"])

    pet_ads = [am.get_additive_term() * norm for am, norm in zip(pet_ams, pet_norms)]
    spect_ads = [
        am.get_additive_term() for am in spect_ams
    ]  # Do I somehow need to apply the normalisation here?

    pet_ams = [am.get_linear_acquisition_model() for am in pet_ams]
    spect_ams = [am.get_linear_acquisition_model() for am in spect_ams]

    pet_obj_funs = [
        OperatorCompositionFunction(KullbackLeibler(data, eta=add + add.max() / 1e3), am)
        for data, add, am in zip(pet_datas, pet_ads, pet_ams)
    ]
    spect_obj_funs = [
        OperatorCompositionFunction(KullbackLeibler(data, eta=add + add.max() / 1e3), am)
        for data, add, am in zip(spect_datas, spect_ads, spect_ams)
    ]

    return pet_obj_funs, spect_obj_funs


def get_s_inv_from_objs(obj_funs, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates.get_uniform_copy(0)
    for i, el in enumerate(s_inv.containers):
        for j, obj_fun in enumerate(obj_funs[i]):
            if j == 0:
                sens = obj_fun.get_subset_sensitivity(0)
            else:
                sens += obj_fun.get_subset_sensitivity(0)
        # Compute maximum with zero (returning a new container)
        sens.maximum(0, out=sens)
        sens_arr = get_array(sens).astype(np.float32)
        # We can afford to avoid zeros because
        # a zero sensitivity means we're outside the FOV
        inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
        # there really shouldn't be any NaNs, but just in case
        s_inv.containers[i].fill(np.nan_to_num(inv_sens_arr))
    return s_inv


def get_s_inv_from_am(ams, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates.get_uniform_copy(0)
    for i, el in enumerate(s_inv.containers):
        for am in ams[i]:
            one = am.forward(initial_estimates[i]).get_uniform_copy(1)
            tmp = am.backward(one)
            el += tmp
        el = el.maximum(0)
        el_arr = get_array(el)
        el_arr = np.reciprocal(el_arr, where=el_arr != 0)
        el.fill(np.nan_to_num(el_arr))
    return s_inv


def get_s_inv_from_subset_objs(obj_funs, initial_estimate):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimate.get_uniform_copy(0)
    for j, obj_fun in enumerate(obj_funs):
        if j == 0:
            sens = obj_fun.get_subset_sensitivity(0)
        else:
            sens += obj_fun.get_subset_sensitivity(0)
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    sens_arr = get_array(sens).astype(np.float32)
    # We can afford to avoid zeros because
    # a zero sensitivity means we're outside the FOV
    inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
    # there really shouldn't be any NaNs, but just in case
    s_inv.fill(np.nan_to_num(inv_sens_arr))
    return s_inv


def compute_inv_hessian_diagonals(bdc, obj_funs_list):
    outputs = []

    for image, obj_funs in zip(bdc.containers, obj_funs_list):
        # Initialize uniform copies
        ones_image = image.get_uniform_copy(1)
        hessian_diag = ones_image.get_uniform_copy(0)

        # Accumulate Hessian contributions
        for obj_fun in obj_funs:
            hessian_diag += obj_fun.function.multiply_with_Hessian(image, ones_image)

        # Take absolute values and write the result
        hessian_diag = hessian_diag.abs()

        hessian_diag_arr = get_array(hessian_diag)
        hessian_diag.fill(np.reciprocal(hessian_diag_arr, where=hessian_diag_arr != 0))

        outputs.append(hessian_diag)

    return EnhancedBlockDataContainer(*outputs)


def gradient_energy_scale_sirf(
    x_pet, x_spect, kappa_pet=None, kappa_spect=None, beta=1.0, mask=None, eps=1e-12
):
    """
    Compute alpha* that balances κ-weighted gradient energies for TNV:
        alpha* = < κ_pet ∇x_pet , beta κ_spect ∇x_spect > / || κ_pet ∇x_pet ||^2

    Parameters
    ----------
    x_pet, x_spect : sirf.ImageData
        Images in the SAME geometry (map SPECT→PET first if needed).
    kappa_pet, kappa_spect : sirf.ImageData or None
        Base per-voxel weights used by the TNV (before multiplying by alpha/beta).
        If None, treated as all-ones.
    beta : float
        If you keep beta fixed ≠ 1, include it here.
    mask : sirf.ImageData or None
        Optional boolean/{0,1} mask to limit the fit.
    eps : float
        Numerical stabiliser.

    Returns
    -------
    float
        alpha* scale.
    """
    import numpy as np

    xr = get_array(x_pet)
    xs = get_array(x_spect)

    vz, vy, vx = x_pet.voxel_sizes()  # (z,y,x) spacings in mm

    # Physical gradients
    gr_z, gr_y, gr_x = np.gradient(xr, vz, vy, vx, edge_order=1)
    gs_z, gs_y, gs_x = np.gradient(xs, vz, vy, vx, edge_order=1)

    # Stack as (3, Z, Y, X)
    gr = np.stack([gr_z, gr_y, gr_x], axis=0)
    gs = np.stack([gs_z, gs_y, gs_x], axis=0)

    # Apply base κ weights (broadcast over gradient components)
    if kappa_pet is not None:
        kp = get_array(kappa_pet)
        gr = gr * kp
    if kappa_spect is not None:
        ks = get_array(kappa_spect)
        gs = gs * ks

    # Optional mask
    if mask is not None:
        m = get_array(mask).astype(bool)
        gr = gr[:, m]
        gs = gs[:, m]
    else:
        gr = gr.reshape(3, -1)
        gs = gs.reshape(3, -1)

    # Compute alpha*
    num = float(np.dot(gr.ravel(), (beta * gs).ravel()))
    den = float(np.dot(gr.ravel(), gr.ravel())) + eps
    return num / den


def apply_gradient_energy_scaling(args, scale):
    """Apply gradient energy scaling to all prior weightings consistently."""

    # Scale TNV weightings
    args.alpha *= scale
    logging.info(f"Adjusted alpha to {args.alpha:.6g} using gradient-energy scaling")

    # Scale modality-specific TV weightings if they exist
    if hasattr(args, "gamma_pet"):
        args.gamma_pet *= scale
        logging.info(f"Adjusted gamma_pet to {args.gamma_pet:.6g} using gradient-energy scaling")


def estimate_delta_from_gradients(images, scales=None, percentile=95, divisor=10.0):
    """Estimate delta from scaled image-gradient magnitudes.

    Parameters
    ----------
    images : EnhancedBlockDataContainer or iterable of ImageData
        Images in a common geometry.
    scales : sequence of float or None
        Optional per-image scaling factors (e.g. alpha, beta).
    percentile : float
        Percentile of the gradient magnitude distribution to use (robust to outliers).
    divisor : float
        Factor by which to divide the chosen magnitude to obtain delta.

    Returns
    -------
    float or None
        Suggested delta value, or None if estimation failed.
    """

    if images is None:
        return None

    # Allow direct iterable of images
    containers = getattr(images, "containers", images)

    grad_stats = []

    for idx, image in enumerate(containers):
        arr = get_array(image)
        if np.all(arr == 0):
            continue

        vz, vy, vx = image.voxel_sizes()
        grads = np.gradient(arr, vz, vy, vx, edge_order=1)
        grad_mag = np.sqrt(sum(g * g for g in grads))

        scale = 1.0
        if scales is not None and idx < len(scales):
            scale = float(np.abs(scales[idx]))

        stat = np.percentile(grad_mag, percentile)
        grad_stats.append(scale * stat)

    grad_stats = [val for val in grad_stats if val > 0 and np.isfinite(val)]
    if not grad_stats:
        return None

    ref_stat = max(grad_stats)
    if ref_stat <= 0:
        return None

    return ref_stat / divisor


def get_prior(
    args,
    umap,
    initial_estimates,
    bo,
    kappas=None,
):
    """
    Set up the prior function for image reconstruction.

    Supports three types of priors that can be used independently or combined:
    - PET TV prior (weighted by gamma_pet)
    - SPECT TV prior (weighted by gamma_spect)
    - TNV vectorial prior (weighted by gamma_tnv, uses alpha/beta for kappa weighting)

    Each prior can have independent directional settings.

    Returns:
        prior: The constructed prior function (SumFunction if multiple priors).
        priors: List of individual prior functions for preconditioner setup.
    """

    use_kappa = getattr(args, "use_kappa", getattr(args, "use_kappas", True))

    if not use_kappa or kappas is None:
        kappas = initial_estimates.get_uniform_copy(1)
    else:
        # Ensure κ images live in the same geometry as `initial_estimates`
        shapes_match = all(
            kap.shape == est.shape
            for kap, est in zip(kappas.containers, initial_estimates.containers)
        )
        if not shapes_match:
            kappas = EnhancedBlockDataContainer(*bo.direct(kappas).containers)
        elif hasattr(kappas, "clone"):
            kappas = kappas.clone()
        else:
            kappas = EnhancedBlockDataContainer(
                *[kap.clone() for kap in kappas.containers]
            )

    priors = []

    # TNV (vectorial) prior - uses alpha/beta weighting
    if getattr(args, "use_tnv_prior", True) and getattr(args, "gamma_tnv", 1.0) > 0:
        # Create kappa weights for TNV with alpha/beta scaling
        tnv_kappas = EnhancedBlockDataContainer(
            initial_estimates[0].get_uniform_copy(args.alpha * args.gamma_tnv),
            initial_estimates[1].get_uniform_copy(args.beta * args.gamma_tnv),
        )

        # Apply base kappa weights
        for i, el in enumerate(tnv_kappas.containers):
            el.multiply(kappas.containers[i], out=el)

        vtv = WeightedVectorialTotalVariation(
            initial_estimates,
            tnv_kappas,
            args.delta,
            anatomical=umap if args.directional_tnv else None,
            stable=getattr(args, "stable", True),
            tail_singular_values=getattr(args, "tail_singular_values", None),
            both_directions=getattr(args, "tnv_both_directions", True),
            stencil=getattr(args, "tnv_stencil", "6"),
            hessian=getattr(args, "hessian_type", "slow"),
            bnd_cond=getattr(args, "tnv_bnd_cond", "Periodic"),
        )
        tnv_prior = OperatorCompositionFunction(vtv, bo)

        # Apply TNV weighting

        priors.append(tnv_prior)

    # Modality-specific TV priors
    if getattr(args, "use_modality_specific_priors", False):
        # Get gamma weights (these should already be scaled by gradient energy scaling in main())
        gamma_pet = getattr(args, "gamma_pet", 0.0)
        gamma_spect = getattr(args, "gamma_spect", 0.0)

        if gamma_pet > 0 or gamma_spect > 0:
            # Create separate kappa weights for modality-specific priors using gamma weights
            tv_kappas = EnhancedBlockDataContainer(
                initial_estimates[0].get_uniform_copy(gamma_pet),
                initial_estimates[1].get_uniform_copy(gamma_spect),
            )

            # Apply base kappa weights
            for i, el in enumerate(tv_kappas.containers):
                el.multiply(kappas.containers[i], out=el)

            if getattr(args, "prior", "tv") == "rdp":
                combined_tv = WeightedRDP(
                    initial_estimates,
                    tv_kappas,
                    epsilon=getattr(args, "delta"),
                    anatomical=umap if args.directional_tv else None,
                    stencil=getattr(args, "tv_stencil", "6"),
                    both_directions=getattr(args, "tv_both_directions", False),
                )
            else:
                combined_tv = WeightedTotalVariation(
                    initial_estimates,
                    tv_kappas,
                    delta=getattr(args, "delta"),
                    anatomical=umap if args.directional_tv else None,
                    stencil=getattr(args, "tv_stencil", "6"),
                    both_directions=getattr(args, "tv_both_directions", False),
                    bnd_cond=getattr(args, "tv_bnd_cond", "Periodic"),
                )

            combined_tv_prior = OperatorCompositionFunction(combined_tv, bo)
            priors.append(combined_tv_prior)

    # Combine priors
    if not priors:
        raise ValueError(
            "No priors enabled. Set use_tnv_prior=True or enable TV priors with gamma > 0"
        )
    else:
        return priors
