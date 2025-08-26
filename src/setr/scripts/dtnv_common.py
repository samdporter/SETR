"""Shared utilities for DTNV (run_dtnv_*) reconstruction scripts.

Only contains functions that are IDENTICAL between run_dtnv_1bpos.py and run_dtnv_2bpos.py.
"""

import argparse
import logging
import os
from types import MethodType
from typing import Any, List

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    KullbackLeibler, 
    OperatorCompositionFunction,
    SumFunction,
)
from cil.optimisation.operators import (
    BlockOperator,
    IdentityOperator,
    ZeroOperator,
)

from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.priors import (
    WeightedVectorialTotalVariation,
    WeightedTotalVariation,
    WeightedRDP,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import ScalingOperator

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
    """Set up callbacks for DTNV algorithm monitoring (identical in both DTNV scripts)."""
    return [
        SaveImageCallback(os.path.join(args.output_path, "image"), update_interval),
        SaveGradientUpdateCallback(os.path.join(args.output_path, "gradient"), update_interval),
        SavePreconditionerCallback(
            os.path.join(args.output_path, "preconditioner"), update_interval
        ),
        PrintObjectiveCallback(update_interval),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), update_interval),
    ]


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
    """Set up preconditioners for 2bpos."""
    max_vals = [el.max() for el in initial_estimates.containers]

    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=0,
        max_vals=max_vals,
        smooth=True,
    )
    if priors_list is None:
        return bsrem_precond

    prior_precond = [
        ImageFunctionPreconditioner(
            p.inv_hessian_diag,
            1.0,
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
    """Get sampling probabilities - 1bpos version."""
    pet_probs = [1 / update_interval] * num_subsets[0] * bpos
    spect_probs = [1 / update_interval] * num_subsets[1]
    probs = pet_probs + spect_probs
    assert abs(sum(probs) - 1) < 1e-10, (
        f"Probabilities do not sum to 1: {sum(probs)}. "
        f"Pet: {sum(pet_probs)}, Spect: {sum(spect_probs)}"
    )
    return probs


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
    arrays = [im.as_array() for im in kappa_block.containers]
    pvals = [np.percentile(a, pct) for a in arrays]
    for im, p in zip(kappa_block.containers, pvals):
        if p > 1e-12:
            logging.info(
                f"Normalising kappa image with max {im.max()} to percentile {pct} value {p}"
            )
            im *= 1.0 / p
    return kappa_block


def attach_prior_hessian(prior, epsilon=0) -> None:
    """Attach an inv_hessian_diag method to the prior function."""

    def inv_hessian_diag(self, x, out=None, epsilon=epsilon):
        ret = self.function.operator.adjoint(
            self.function.function.inv_hessian_diag(
                self.function.operator.direct(x),
            )
        )
        ret = ret.abs()
        if out is not None:
            out.fill(ret)
        return ret

    def hessian_diag(self, x, out=None, epsilon=epsilon):
        ret = self.function.operator.adjoint(
            self.function.function.hessian_diag(
                self.function.operator.direct(x),
            )
        )
        ret = ret.abs()
        if out is not None:
            out.fill(ret)
        return ret

    prior.inv_hessian_diag = MethodType(inv_hessian_diag, prior)
    prior.hessian_diag = MethodType(hessian_diag, prior)


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
    """Returns a CIL SumFunction using KL objective functions for the PET and SPECT data and acq models"""

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
        sens_arr = sens.as_array().astype(np.float32)
        # We can afford to avoid zeros because
        # a zero sensitivity means we're outside the FOV
        inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
        # there really shouldn't be any NaNs, but just in case
        s_inv.containers[i].fill(np.nan_to_num(inv_sens_arr))
    return s_inv


def get_s_inv_from_am(ams, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates * 0
    for i, el in enumerate(s_inv.containers):
        for am in ams[i]:
            one = am.forward(initial_estimates[i]).get_uniform_copy(1)
            tmp = am.backward(one)
            el += tmp
        el = el.maximum(0)
        el_arr = el.as_array()
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
    sens_arr = sens.as_array().astype(np.float32)
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

        hessian_diag_arr = hessian_diag.as_array()
        hessian_diag.fill(np.reciprocal(hessian_diag_arr, where=hessian_diag_arr != 0))

        outputs.append(hessian_diag)

    return EnhancedBlockDataContainer(*outputs)


def gradient_energy_scale_sirf(x_pet, x_spect, kappa_pet=None, kappa_spect=None,
                                        beta=1.0, mask=None, eps=1e-12):
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
    xr = x_pet.as_array()
    xs = x_spect.as_array()

    vz, vy, vx = x_pet.voxel_sizes()  # (z,y,x) spacings in mm

    # Physical gradients
    gr_z, gr_y, gr_x = np.gradient(xr, vz, vy, vx, edge_order=1)
    gs_z, gs_y, gs_x = np.gradient(xs, vz, vy, vx, edge_order=1)

    # Stack as (3, Z, Y, X)
    gr = np.stack([gr_z, gr_y, gr_x], axis=0)
    gs = np.stack([gs_z, gs_y, gs_x], axis=0)

    # Apply base κ weights (broadcast over gradient components)
    if kappa_pet is not None:
        kp = kappa_pet.as_array()
        gr = gr * kp
    if kappa_spect is not None:
        ks = kappa_spect.as_array()
        gs = gs * ks

    # Optional mask
    if mask is not None:
        m = mask.as_array().astype(bool)
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
    if hasattr(args, 'gamma_pet'):
        args.gamma_pet *= scale
        logging.info(f"Adjusted gamma_pet to {args.gamma_pet:.6g} using gradient-energy scaling")

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

    if kappas is None:
        kappas = initial_estimates.get_uniform_copy(1)

    priors = []

    # TNV (vectorial) prior - uses alpha/beta weighting
    if getattr(args, "use_tnv_prior", True) and getattr(args, "gamma_tnv", 1.0) > 0:
        # Create kappa weights for TNV with alpha/beta scaling
        tnv_kappas = EnhancedBlockDataContainer(
            initial_estimates[0].get_uniform_copy(args.alpha*args.gamma_tnv),
            initial_estimates[1].get_uniform_copy(args.beta*args.gamma_tnv),
        )

        # Apply base kappa weights
        for i, el in enumerate(tnv_kappas.containers):
            el.fill(kappas.containers[i] * el)

        vtv = WeightedVectorialTotalVariation(
            initial_estimates,
            tnv_kappas,
            args.delta,
            anatomical=umap if args.directional_tnv else None,
            stable=getattr(args, "stable", True),
            tail_singular_values=getattr(args, "tail_singular_values", None),
            both_directions=getattr(args, "tnv_both_directions", False),
            stencil=getattr(args, "tnv_stencil", '6'),
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
                el.fill(kappas.containers[i] * el)

            if getattr(args, "prior", "tv") == "rdp":
                combined_tv = WeightedRDP(
                    initial_estimates,
                    tv_kappas,
                    epsilon=getattr(args, "delta_tv"),
                    anatomical=umap if args.directional_tv else None,
                    stencil=getattr(args, "tv_stencil", '6'),
                    both_directions=getattr(args, "tv_both_directions", False),
                )
            else:
                combined_tv = WeightedTotalVariation(
                    initial_estimates,
                    tv_kappas,
                    delta=getattr(args, "delta_tv"),
                    anatomical=umap if args.directional_tv else None,
                    stencil=getattr(args, "tv_stencil", '6'),
                    both_directions=getattr(args, "tv_both_directions", False),
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