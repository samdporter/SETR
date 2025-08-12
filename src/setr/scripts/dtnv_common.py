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
from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction
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
    priors_list: List[Any],
    initial_estimates: EnhancedBlockDataContainer,
) -> Any:
    """
    Set up the preconditioners.

    Args:
        priors_list: List of individual prior functions (before combining with SumFunction)

    Returns:
        The combined preconditioner.
    """

    from setr.cil_extensions.preconditioners import (
        BSREMPreconditioner,
        ImageFunctionPreconditioner,
        LehmerMeanPreconditioner,
    )

    max_vals = [el.max() for el in initial_estimates.containers]
    minmax_val = min(max_vals)

    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=minmax_val / 1000,
        max_vals=max_vals,
        smooth=True,
    )

    if not priors_list:
        return bsrem_precond

    # Create preconditioners for each individual prior's Hessian
    precond_list = [bsrem_precond]

    for prior in priors_list:
        prior_precond = ImageFunctionPreconditioner(
            prior.inv_hessian_diag,
            1.0,
            update_interval=update_interval,
            epsilon=0,
            freeze_iter=np.inf,
        )
        precond_list.append(prior_precond)

    return LehmerMeanPreconditioner(
        precond_list,
        update_interval=update_interval,
        freeze_iter=len(all_funs) * 10,
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
