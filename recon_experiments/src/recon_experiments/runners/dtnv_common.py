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
from cil.framework import BlockDataContainer
from recon_core.cil_extensions.algorithms import ista_update_step
from recon_core.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.cil_extensions.functions import BlockIndicatorBox
from recon_core.cil_extensions.operators import ScalingOperator
from recon_core.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    BlockDiagonalPriorPreconditioner,
    BlockLehmerMeanPreconditioner,
    HarmonicMeanPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
    MajorisingHessianBlockPreconditioner,
    MajorisingHessianDiagonalPreconditioner,
    MaGeZPreconditioner,
)
from recon_core.priors import (
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
    WeightedLogVectorialTotalVariation,
)
from recon_core.utils.dynamic_range import (
    apply_dynamic_range_scaling,
    apply_gradient_energy_scaling,
    dynamic_range_scale_sirf,
)
from recon_core.utils.sirf import (
    get_array,
    get_s_inv_from_am,
    get_s_inv_from_objs,
    get_s_inv_from_subset_objs,
)

ISTA.update = ista_update_step

_PRECOND_DIAG_METHODS = {"mm_diag_tight", "mm_diag_gershgorin_maj"}
_PRECOND_BLOCK_METHODS = {"mm_diag_block_maj", "mm_diag_block_tight"}
_ALL_PRECOND_METHODS = _PRECOND_DIAG_METHODS | _PRECOND_BLOCK_METHODS
_PRECOND_CONTRACTS = {
    # Proper MM majorisers (for the prior surrogate), when used with combine='majoriser'.
    "mm_diag_gershgorin_maj": "majoriser",
    "mm_diag_block_maj": "majoriser",
    # Curvature estimates / heuristics (no strict majorisation guarantee end-to-end).
    "bsrem": "hessian_estimate",
    "mm_diag_tight": "hessian_estimate",
    "mm_diag_block_tight": "hessian_estimate",
}


def _canonical_preconditioner_type(precond_type: str) -> str:
    return {
        "bsrem": "bsrem",
        "mm_diag_tight": "mm_diag_tight",
        "mm_diag_gershgorin_maj": "mm_diag_gershgorin_maj",
        "mm_diag_block_maj": "mm_diag_block_maj",
        "mm_diag_block_tight": "mm_diag_block_tight",
    }.get(precond_type, precond_type)


def _resolve_precond_method_from_args(args: argparse.Namespace) -> str:
    explicit = getattr(args, "precond_method", None)
    if explicit is not None:
        return _canonical_preconditioner_type(explicit)
    canonical = _canonical_preconditioner_type(getattr(args, "precond_type", "mm_diag_tight"))
    if canonical == "bsrem":
        return "mm_diag_tight"
    return canonical


def _is_block_method(method: str) -> bool:
    return method in _PRECOND_BLOCK_METHODS


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _normalise_fwhm(value, default=(5.0, 5.0, 5.0)):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, v, v)
    try:
        vals = tuple(float(v) for v in value)
    except TypeError:
        return default
    if len(vals) != 3:
        return default
    return vals


def _positive_quantile(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    positive = finite[finite > 0]
    if positive.size == 0:
        return float(np.max(finite))
    q = float(np.clip(q, 0.0, 100.0))
    return float(np.percentile(positive, q))


def _initial_image_cap(initial_estimates, quantile: float) -> float:
    caps = []
    containers = (
        initial_estimates.containers
        if isinstance(initial_estimates, BlockDataContainer)
        else getattr(initial_estimates, "containers", (initial_estimates,))
    )
    for con in containers:
        try:
            arr = get_array(con).astype(np.float64, copy=False)
        except Exception:
            try:
                caps.append(float(con.max()))
            except Exception:
                continue
        else:
            caps.append(_positive_quantile(arr, quantile))
    if not caps:
        return np.inf
    cap = max(caps)
    return cap if np.isfinite(cap) and cap > 0 else np.inf


def _log_preconditioner_contract(precond_method: str, combine: str):
    contract = _PRECOND_CONTRACTS.get(precond_method, "hessian_estimate")
    if combine == "majoriser" and contract == "majoriser":
        logging.info(
            "Preconditioner contract: method=%s, combine=%s -> majorising surrogate.",
            precond_method,
            combine,
        )
        return
    if combine == "majoriser" and contract != "majoriser":
        logging.warning(
            "Preconditioner contract: method=%s is a Hessian estimate/heuristic. "
            "combine='majoriser' still inverts a summed surrogate but does not upgrade this "
            "method to a strict prior majoriser.",
            precond_method,
        )
        return
    logging.info(
        "Preconditioner contract: method=%s, combine=%s -> Hessian estimate/heuristic blend.",
        precond_method,
        combine,
    )


def _clone_container_like(obj):
    if hasattr(obj, "copy"):
        return obj.copy()
    if hasattr(obj, "clone"):
        return obj.clone()
    raise AttributeError(f"Object of type {type(obj)} does not expose copy() or clone().")


def _to_numpy_array(arr_like):
    if hasattr(arr_like, "detach"):
        return arr_like.detach().cpu().numpy()
    return np.asarray(arr_like)


def _is_diag_curvature_capable(prior) -> bool:
    return any(
        hasattr(prior, attr)
        for attr in ("preconditioner_diag", "hessian_diag", "inv_preconditioner_diag", "inv_hessian_diag")
    )


def _is_block_curvature_capable(prior) -> bool:
    return any(
        hasattr(prior, attr)
        for attr in ("preconditioner_block", "hessian_block_diag", "inv_preconditioner_block", "inv_hessian_block")
    )


class _SummedPriorCurvature:
    """Aggregate shared-space prior curvature across multiple active priors."""

    def __init__(self, priors, hessian_floor: float = 1e-8):
        self.priors = list(priors)
        self.hessian_floor = float(hessian_floor)

    def _call_with_optional_epsilon(self, fn, image, epsilon):
        try:
            return fn(image, epsilon=epsilon)
        except TypeError as exc:
            if "epsilon" not in str(exc):
                raise
            return fn(image)

    def _diag_from_inverse(self, inv_diag, epsilon):
        diag = _clone_container_like(inv_diag)
        containers = diag.containers if isinstance(diag, BlockDataContainer) else (diag,)
        for con in containers:
            arr = get_array(con).astype(np.float64, copy=False)
            np.maximum(arr, epsilon, out=arr)
            np.reciprocal(arr, out=arr)
            con.fill(arr)
        return diag

    def _evaluate_prior_diag(self, prior, image, epsilon):
        for attr in ("preconditioner_diag", "hessian_diag"):
            if hasattr(prior, attr):
                diag = self._call_with_optional_epsilon(getattr(prior, attr), image, epsilon)
                return diag.abs() if hasattr(diag, "abs") else diag

        for attr in ("inv_preconditioner_diag", "inv_hessian_diag"):
            if hasattr(prior, attr):
                inv_diag = self._call_with_optional_epsilon(getattr(prior, attr), image, epsilon)
                return self._diag_from_inverse(inv_diag, epsilon)

        return None

    def _sum_diag(self, image, epsilon):
        total = None
        for prior in self.priors:
            diag = self._evaluate_prior_diag(prior, image, epsilon)
            if diag is None:
                continue
            if total is None:
                total = _clone_container_like(diag)
            else:
                total = total + diag
        if total is None:
            raise AttributeError("No active prior exposes diagonal curvature helpers.")
        return total.abs() if hasattr(total, "abs") else total

    def preconditioner_diag(self, image, epsilon=1e-8):
        return self._sum_diag(image, epsilon)

    def hessian_diag(self, image, epsilon=1e-8):
        return self._sum_diag(image, epsilon)

    def inv_preconditioner_diag(self, image, epsilon=1e-8):
        diag = self._sum_diag(image, epsilon)
        containers = diag.containers if isinstance(diag, BlockDataContainer) else (diag,)
        for con in containers:
            arr = get_array(con).astype(np.float64, copy=False)
            np.maximum(arr, epsilon, out=arr)
            np.reciprocal(arr, out=arr)
            con.fill(arr)
        return diag

    def inv_hessian_diag(self, image, epsilon=1e-8):
        return self.inv_preconditioner_diag(image, epsilon=epsilon)

    def _invert_block_hessian(self, block_arr: np.ndarray, epsilon: float) -> np.ndarray:
        block_arr = 0.5 * (block_arr + np.swapaxes(block_arr, -1, -2))
        eigvals, eigvecs = np.linalg.eigh(block_arr.astype(np.float64, copy=False))
        np.maximum(eigvals, epsilon, out=eigvals)
        inv_eigs = 1.0 / eigvals
        inv_block = (eigvecs * inv_eigs[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
        return 0.5 * (inv_block + np.swapaxes(inv_block, -1, -2))

    def _block_from_diag(self, diag) -> np.ndarray:
        if not isinstance(diag, BlockDataContainer):
            raise TypeError(
                f"Expected BlockDataContainer diagonal curvature, got {type(diag)}."
            )
        if len(diag.containers) != 2:
            raise ValueError(
                f"Expected 2 modalities for block curvature assembly, got {len(diag.containers)}."
            )
        arr0 = get_array(diag.containers[0]).astype(np.float64, copy=False)
        arr1 = get_array(diag.containers[1]).astype(np.float64, copy=False)
        if arr0.shape != arr1.shape:
            raise ValueError(
                f"Diagonal prior geometry mismatch: PET {arr0.shape}, SPECT {arr1.shape}."
            )
        block = np.zeros((*arr0.shape, 2, 2), dtype=np.float64)
        block[..., 0, 0] = arr0
        block[..., 1, 1] = arr1
        return block

    def _evaluate_prior_block(self, prior, image, epsilon):
        for attr in ("preconditioner_block", "hessian_block_diag"):
            if hasattr(prior, attr):
                block = self._call_with_optional_epsilon(getattr(prior, attr), image, epsilon)
                block_arr = _to_numpy_array(block).astype(np.float64, copy=False)
                return 0.5 * (block_arr + np.swapaxes(block_arr, -1, -2))

        for attr in ("inv_preconditioner_block", "inv_hessian_block"):
            if hasattr(prior, attr):
                inv_block = self._call_with_optional_epsilon(getattr(prior, attr), image, epsilon)
                inv_block_arr = _to_numpy_array(inv_block).astype(np.float64, copy=False)
                return self._invert_block_hessian(inv_block_arr, epsilon)

        diag = self._evaluate_prior_diag(prior, image, epsilon)
        if diag is not None:
            return self._block_from_diag(diag)

        return None

    def _sum_block(self, image, epsilon):
        total = None
        for prior in self.priors:
            block = self._evaluate_prior_block(prior, image, epsilon)
            if block is None:
                continue
            if total is None:
                total = np.array(block, dtype=np.float64, copy=True)
            else:
                total += block
        if total is None:
            raise AttributeError("No active prior exposes block or diagonal curvature helpers.")
        return 0.5 * (total + np.swapaxes(total, -1, -2))

    def preconditioner_block(self, image, epsilon=1e-8):
        return self._sum_block(image, epsilon)

    def hessian_block_diag(self, image, epsilon=1e-8):
        return self._sum_block(image, epsilon)

    def inv_preconditioner_block(self, image, epsilon=1e-8):
        return self._invert_block_hessian(self._sum_block(image, epsilon), epsilon)


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


def build_support_mask_from_s_inv(
    s_inv: Any,
    rel_threshold: float = 1e-3,
    abs_threshold: float = 0.0,
):
    """Create a binary support mask from inverse sensitivities.

    The support is defined in sensitivity space:
        support = { voxels | sensitivity >= max(abs_threshold, rel_threshold * sensitivity_max) }

    where sensitivity = 1 / s_inv on voxels with positive s_inv.
    """
    rel_threshold = float(rel_threshold)
    abs_threshold = float(abs_threshold)
    if rel_threshold < 0:
        raise ValueError("rel_threshold must be >= 0.")
    if abs_threshold < 0:
        raise ValueError("abs_threshold must be >= 0.")

    mask = s_inv.copy()
    if isinstance(mask, BlockDataContainer):
        mask_containers = mask.containers
        source_containers = s_inv.containers
    else:
        mask_containers = (mask,)
        source_containers = (s_inv,)

    for idx, (dst, src) in enumerate(zip(mask_containers, source_containers)):
        s_inv_arr = get_array(src).astype(np.float64, copy=False)
        sens = np.zeros_like(s_inv_arr, dtype=np.float64)
        np.reciprocal(s_inv_arr, out=sens, where=s_inv_arr > 0)

        finite = sens[np.isfinite(sens)]
        if finite.size == 0:
            mask_arr = np.ones_like(s_inv_arr, dtype=np.float32)
            threshold = 0.0
            sens_max = 0.0
        else:
            sens_max = float(np.max(finite))
            threshold = max(abs_threshold, rel_threshold * sens_max)
            if threshold <= 0:
                mask_arr = (sens > 0).astype(np.float32)
            else:
                mask_arr = (sens >= threshold).astype(np.float32)

        coverage = float(np.mean(mask_arr > 0))
        logging.info(
            "Support mask modality %d: sensitivity_max=%.6g threshold=%.6g coverage=%.3f",
            idx,
            sens_max,
            threshold,
            coverage,
        )
        dst.fill(mask_arr)

    return mask


def build_support_mask_from_spect_attenuation(
    template: Any,
    spect_attenuation: Any,
    spect_index: int = 1,
    rel_threshold: float = 1e-3,
    abs_threshold: float = 1e-2,
):
    """Create a binary support mask from the SPECT attenuation image.

    Voxels outside the attenuation support are projected to zero by BlockIndicatorBox.
    Non-SPECT modalities are set to all-ones in the returned mask.
    """
    rel_threshold = float(rel_threshold)
    abs_threshold = float(abs_threshold)
    if rel_threshold < 0:
        raise ValueError("rel_threshold must be >= 0.")
    if abs_threshold < 0:
        raise ValueError("abs_threshold must be >= 0.")

    mask = template.copy()
    mask_containers = mask.containers if isinstance(mask, BlockDataContainer) else (mask,)
    if not 0 <= spect_index < len(mask_containers):
        raise IndexError(
            f"spect_index={spect_index} out of range for {len(mask_containers)} mask containers."
        )

    spect_mask = mask_containers[spect_index]
    attn_arr = get_array(spect_attenuation).astype(np.float64, copy=False)
    spect_shape = get_array(spect_mask).shape
    if attn_arr.shape != spect_shape:
        raise ValueError(
            f"SPECT attenuation shape {attn_arr.shape} does not match mask shape {spect_shape}."
        )

    finite = attn_arr[np.isfinite(attn_arr)]
    if finite.size == 0:
        attn_max = 0.0
        threshold = 0.0
        spect_mask_arr = np.zeros_like(attn_arr, dtype=np.float32)
    else:
        attn_max = float(np.max(np.maximum(finite, 0.0)))
        threshold = max(abs_threshold, rel_threshold * attn_max)
        if threshold <= 0:
            spect_mask_arr = (attn_arr > 0).astype(np.float32)
        else:
            spect_mask_arr = (attn_arr >= threshold).astype(np.float32)

    for idx, dst in enumerate(mask_containers):
        if idx == spect_index:
            dst.fill(spect_mask_arr)
        else:
            dst.fill(np.ones_like(get_array(dst), dtype=np.float32))

    coverage = float(np.mean(spect_mask_arr > 0))
    logging.info(
        "Support mask from SPECT attenuation: max=%.6g threshold=%.6g coverage=%.3f",
        attn_max,
        threshold,
        coverage,
    )
    return mask


def combine_support_masks(mask_a: Any, mask_b: Any):
    """Combine two support masks via element-wise multiplication."""
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a

    out = mask_a.copy()
    out_containers = out.containers if isinstance(out, BlockDataContainer) else (out,)
    a_containers = mask_a.containers if isinstance(mask_a, BlockDataContainer) else (mask_a,)
    b_containers = mask_b.containers if isinstance(mask_b, BlockDataContainer) else (mask_b,)
    if len(out_containers) != len(b_containers):
        raise ValueError(
            f"Support mask container count mismatch: {len(out_containers)} vs {len(b_containers)}."
        )

    for dst, a_el, b_el in zip(out_containers, a_containers, b_containers):
        dst.fill(a_el)
        dst.multiply(b_el, out=dst)
    return out


def get_callbacks(args, update_interval: int, iteration_offset: int = 0) -> List[Any]:
    """Set up callbacks for DTNV algorithm monitoring."""
    callbacks = [
        SaveImageCallback(
            os.path.join(args.output_path, "image"),
            update_interval,
            iteration_offset=iteration_offset,
        ),
        PrintObjectiveCallback(update_interval),
        SaveObjectiveCallback(
            os.path.join(args.output_path, "objective"),
            update_interval,
            iteration_offset=iteration_offset,
        ),
    ]

    if getattr(args, "save_gradients", False):
        callbacks.append(
            SaveGradientUpdateCallback(
                os.path.join(args.output_path, "gradient"),
                update_interval,
                iteration_offset=iteration_offset,
            )
        )

    if getattr(args, "save_preconditioners", False):
        callbacks.append(
            SavePreconditionerCallback(
                os.path.join(args.output_path, "preconditioner"),
                update_interval,
                iteration_offset=iteration_offset,
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
    support_mask: Any = None,
) -> ISTA:
    """Create and run DTNV ISTA algorithm (identical in both DTNV scripts)."""
    initial = init_solution.copy()
    if support_mask is not None:
        initial_containers = (
            initial.containers if isinstance(initial, BlockDataContainer) else (initial,)
        )
        mask_containers = (
            support_mask.containers if isinstance(support_mask, BlockDataContainer) else (support_mask,)
        )
        if len(initial_containers) != len(mask_containers):
            raise ValueError(
                "Support mask container count does not match initial solution container count: "
                f"{len(mask_containers)} vs {len(initial_containers)}."
            )

        for idx, (initial_el, mask_el) in enumerate(zip(initial_containers, mask_containers)):
            initial_arr = get_array(initial_el)
            mask_arr = get_array(mask_el)
            off_support = mask_arr <= 0
            if np.any(off_support):
                off_support_abs_sum = float(np.sum(np.abs(initial_arr[off_support])))
                if off_support_abs_sum > 0:
                    logging.warning(
                        "Initial solution modality %d has off-support mass %.6g; "
                        "projecting initial iterate onto support mask to avoid objective=inf at iter 0.",
                        idx,
                        off_support_abs_sum,
                    )
            initial_el.multiply(mask_el, out=initial_el)

    algo = ISTA(
        initial=initial,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf, mask=support_mask),
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
    precond_type = getattr(args, "precond_type", "mm_diag_block_maj")
    canonical = _canonical_preconditioner_type(precond_type)
    precond_method = _resolve_precond_method_from_args(args)
    if precond_method not in _ALL_PRECOND_METHODS and canonical != "bsrem":
        raise ValueError(
            f"Unknown preconditioner method '{precond_method}'. "
            f"Options: {sorted(_ALL_PRECOND_METHODS)}"
        )

    combine = str(getattr(args, "precond_combine", "majoriser")).strip().lower()
    valid_combines = {"none", "harmonic", "lehmer", "magez", "majoriser"}
    if combine not in valid_combines:
        logging.warning(
            "Unknown precond_combine=%s; falling back to 'majoriser'. Options: %s",
            combine,
            sorted(valid_combines),
        )
        combine = "majoriser"
    if combine == "none":
        if canonical == "bsrem":
            logging.info("precond_combine='none' with precond_type='bsrem': using BSREM only.")
        else:
            logging.warning(
                "precond_combine='none' is only valid for precond_type='bsrem'; "
                "falling back to combine='majoriser'."
            )
            combine = "majoriser"
    if combine == "harmonic":
        logging.info(
            "precond_combine='harmonic' is equivalent to Hessian-sum inversion here; "
            "using canonical combine='majoriser'."
        )
        combine = "majoriser"
    lehmer_p = float(getattr(args, "lehmer_p", 0.0))
    scalar_reduction = getattr(args, "block_scalar_reduction", "diag")
    if scalar_reduction == "diag" and combine not in {"majoriser", "none"}:
        logging.warning(
            "Diagonal scalar blending requires p=0 inverse-sum; forcing combine='majoriser'."
        )
        combine = "majoriser"
    precond_data_epsilon = float(getattr(args, "precond_data_epsilon", 1e-8))
    precond_safety_scale = float(getattr(args, "precond_safety_scale", 1.0))
    if not np.isfinite(precond_safety_scale) or precond_safety_scale <= 0:
        logging.warning(
            "Invalid precond_safety_scale=%s; using 1.0.",
            precond_safety_scale,
        )
        precond_safety_scale = 1.0
    elif precond_safety_scale > 1.0:
        logging.warning(
            "precond_safety_scale=%.6g > 1 may be aggressive; consider <= 1.",
            precond_safety_scale,
        )
    try:
        epoch_update_interval = int(update_interval)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid preconditioner update_interval=%s; falling back to len(all_funs)=%d.",
            update_interval,
            len(all_funs),
        )
        epoch_update_interval = len(all_funs)
    if epoch_update_interval <= 0:
        logging.warning(
            "Non-positive preconditioner update_interval=%s; falling back to len(all_funs)=%d.",
            update_interval,
            len(all_funs),
        )
        epoch_update_interval = len(all_funs)
    variance_reduction = str(getattr(args, "variance_reduction", "svrg")).lower()
    if canonical != "bsrem" and combine == "majoriser" and variance_reduction in {"svrg", "saga"}:
        logging.warning(
            "Majoriser preconditioner is used with stochastic variance-reduced gradients "
            "(%s). Per-iteration descent is not guaranteed end-to-end.",
            variance_reduction,
        )
    if canonical != "bsrem" and combine == "majoriser" and epoch_update_interval > 1:
        logging.warning(
            "Majoriser preconditioner update_interval=%d (once per epoch). "
            "This is a stale-curvature approximation between updates.",
            epoch_update_interval,
        )
    _log_preconditioner_contract(canonical, combine)

    em_precond_smooth = _as_bool(getattr(args, "em_precond_smooth", True))
    em_precond_smoothing_fwhm = _normalise_fwhm(
        getattr(args, "em_precond_smoothing_fwhm", (5.0, 5.0, 5.0)),
        default=(5.0, 5.0, 5.0),
    )
    em_precond_cap_to_initial = _as_bool(getattr(args, "em_precond_cap_to_initial_max", True))
    em_precond_cap_initial_factor = float(
        getattr(args, "em_precond_cap_to_initial_max_factor", 1.0)
    )
    if em_precond_cap_initial_factor <= 0:
        logging.warning(
            "Invalid em_precond_cap_to_initial_max_factor=%s; using 1.0.",
            em_precond_cap_initial_factor,
        )
        em_precond_cap_initial_factor = 1.0
    if getattr(args, "em_precond_x_cap_quantile", None) is not None:
        logging.warning(
            "em_precond_x_cap_quantile is deprecated and ignored; "
            "use em_precond_cap_to_initial_max_factor instead."
        )
    em_precond_max_val = None
    if em_precond_cap_to_initial:
        initial_cap = _initial_image_cap(initial_estimates, quantile=100.0)
        em_precond_max_val = em_precond_cap_initial_factor * initial_cap
        if np.isfinite(em_precond_max_val):
            logging.info(
                "EM preconditioner x-cap enabled: initial_cap=%.6g factor=%.6g max_val=%.6g",
                initial_cap,
                em_precond_cap_initial_factor,
                em_precond_max_val,
            )
        else:
            em_precond_max_val = None
            logging.warning(
                "EM preconditioner x-cap requested but initial-image cap was non-finite; disabling cap."
            )

    freeze_epochs = getattr(args, "precond_freeze_epochs", None)
    if freeze_epochs is None:
        precond_freeze_iter = np.inf
    else:
        try:
            freeze_epochs = float(freeze_epochs)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid precond_freeze_epochs=%s; using no preconditioner freeze.",
                freeze_epochs,
            )
            precond_freeze_iter = np.inf
        else:
            if freeze_epochs <= 0:
                precond_freeze_iter = np.inf
            else:
                precond_freeze_iter = max(
                    epoch_update_interval,
                    int(round(freeze_epochs * epoch_update_interval)),
                )

    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=0,
        smooth=em_precond_smooth,
        max_val=em_precond_max_val,
        smoothing_fwhm=em_precond_smoothing_fwhm,
    )
    if canonical == "bsrem" or priors_list is None:
        return bsrem_precond

    legacy_max_precond_value = 10.0 * max(
        con.max() * s_inv_con.max()
        for con, s_inv_con in zip(initial_estimates.containers, s_inv.containers)
    )
    precond_cap_to_initial = _as_bool(getattr(args, "precond_cap_to_initial_max", True))
    precond_cap_initial_factor = float(getattr(args, "precond_cap_to_initial_max_factor", 1.0))
    if precond_cap_initial_factor <= 0:
        logging.warning(
            "Invalid precond_cap_to_initial_max_factor=%s; using 1.0.",
            precond_cap_initial_factor,
        )
        precond_cap_initial_factor = 1.0
    max_precond_value = legacy_max_precond_value
    if precond_cap_to_initial:
        initial_cap = _initial_image_cap(initial_estimates, quantile=100.0)
        if np.isfinite(initial_cap):
            initial_bound = precond_cap_initial_factor * initial_cap
            max_precond_value = min(max_precond_value, initial_bound)
            logging.info(
                "Preconditioner max-value cap: legacy=%.6g initial_bound=%.6g -> using %.6g",
                legacy_max_precond_value,
                initial_bound,
                max_precond_value,
            )
        else:
            logging.warning(
                "precond_cap_to_initial_max enabled but initial cap was non-finite; "
                "using legacy max_precond_value=%.6g",
                legacy_max_precond_value,
            )

    supported_priors = [
        p
        for p in (priors_list or [])
        if _is_diag_curvature_capable(p) or _is_block_curvature_capable(p)
    ]

    if _is_block_method(precond_method):
        if not supported_priors:
            logging.warning(
                "No curvature-capable prior found for precond_method=%s. Falling back to diagonal path.",
                precond_method,
            )
        else:
            if len(supported_priors) > 1:
                logging.warning(
                    "Multiple priors found; summing shared-space block/diagonal curvature across %d priors.",
                    len(supported_priors),
                )
            composite_prior = _SummedPriorCurvature(supported_priors, hessian_floor=1e-8)
            block_precond = BlockDiagonalPriorPreconditioner(
                composite_prior,
                update_interval=epoch_update_interval,
                freeze_iter=precond_freeze_iter,
                epsilon=1e-8,
                max_value=max_precond_value,
            )
            if combine == "majoriser":
                return MajorisingHessianBlockPreconditioner(
                    s_inv=s_inv,
                    prior=composite_prior,
                    update_interval=epoch_update_interval,
                    freeze_iter=precond_freeze_iter,
                    x_epsilon=precond_data_epsilon,
                    hessian_floor=1e-8,
                    max_value=max_precond_value,
                    safety_scale=precond_safety_scale,
                )
            if combine == "magez":
                logging.warning(
                    "MaGeZ averaging is diagonal-only; using block majoriser path instead."
                )
                return MajorisingHessianBlockPreconditioner(
                    s_inv=s_inv,
                    prior=composite_prior,
                    update_interval=epoch_update_interval,
                    freeze_iter=precond_freeze_iter,
                    x_epsilon=precond_data_epsilon,
                    hessian_floor=1e-8,
                    max_value=max_precond_value,
                    safety_scale=precond_safety_scale,
                )
            p_val = 0.0 if combine == "harmonic" else lehmer_p
            return BlockLehmerMeanPreconditioner(
                block_preconditioner=block_precond,
                scalar_preconditioner=bsrem_precond,
                p=p_val,
                epsilon=1e-12,
                max_value=max_precond_value,
                update_interval=epoch_update_interval,
                freeze_iter=precond_freeze_iter,
                scalar_reduction=scalar_reduction,
            )

    # Diagonal preconditioner path (single TNV prior)
    prior_candidates = [p for p in supported_priors if _is_diag_curvature_capable(p)]
    prior_for_precond = None
    if prior_candidates:
        if len(prior_candidates) > 1:
            logging.warning(
                "Multiple priors found; summing shared-space diagonal curvature across %d priors.",
                len(prior_candidates),
            )
        prior_for_precond = _SummedPriorCurvature(prior_candidates, hessian_floor=1e-8)
    if prior_for_precond is not None and len(prior_candidates) == 1:
        logging.info(
            "Using diagonal curvature from prior %s.",
            type(prior_candidates[0]).__name__,
        )

    if prior_for_precond is None:
        return bsrem_precond

    prior_precond = ImageFunctionPreconditioner(
        prior_for_precond.inv_preconditioner_diag,
        1,
        freeze_iter=np.inf,
        epsilon=0,
        max_value=max_precond_value,
    )

    if combine == "majoriser":
        return MajorisingHessianDiagonalPreconditioner(
            s_inv=s_inv,
            prior=prior_for_precond,
            update_interval=epoch_update_interval,
            freeze_iter=precond_freeze_iter,
            x_epsilon=precond_data_epsilon,
            hessian_floor=1e-8,
            max_value=max_precond_value,
            safety_scale=precond_safety_scale,
        )

    if combine == "magez":
        return MaGeZPreconditioner(
            s_inv,
            prior_for_precond,
            hessian_scale=getattr(args, "hessian_scale", 1.5),
            delta=getattr(args, "magez_delta", 1e-8),
            update_interval=epoch_update_interval,
            freeze_iter=precond_freeze_iter,
        )

    if combine == "harmonic":
        return HarmonicMeanPreconditioner(
            [bsrem_precond, prior_precond],
            update_interval=epoch_update_interval,
            freeze_iter=precond_freeze_iter,
            epsilon=1e-6,
        )

    # default Lehmer
    return LehmerMeanPreconditioner(
        [bsrem_precond, prior_precond],
        update_interval=epoch_update_interval,
        freeze_iter=precond_freeze_iter,
        epsilon=0,
        p=lehmer_p,
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
    data_probs: List[float] | None = None,
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
        data_probs: Optional explicit sampling probabilities for the data functions in ``all_funs``.
            When omitted, probabilities are inferred from ``num_subsets`` and ``bpos``.

    Returns:
        f_obj: Instantiated variance-reduced function.
        probs: Sampling probabilities passed to the sampler.
        prior_prob: Sampling probability allocated to the prior (``None`` if not sampled).
        prior_in_sampler: ``True`` when the prior participates in stochastic updates.
    """

    if data_probs is None:
        data_probs = get_probabilities(args, num_subsets, epoch_length, bpos=bpos)
    else:
        data_probs = [float(p) for p in data_probs]
        if len(data_probs) != len(all_funs):
            raise ValueError(
                f"Explicit data_probs length {len(data_probs)} does not match "
                f"number of stochastic data functions {len(all_funs)}."
            )
        if any(p < 0 for p in data_probs):
            raise ValueError("Explicit data_probs must be non-negative.")
        prob_sum = float(sum(data_probs))
        if not np.isclose(prob_sum, 1.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"Explicit data_probs must sum to 1.0, got {prob_sum:.12g}."
            )

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

    variance_reduction = getattr(args, "variance_reduction", "saga")
    variance_reduction = str(variance_reduction).lower()

    sampler = Sampler.random_with_replacement(len(stochastic_functions), prob=probs)

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


def _multiply_with_composed_hessian(obj_fun, x, direction):
    """Recursively evaluate Hessian-vector products through operator compositions."""
    if isinstance(obj_fun, OperatorCompositionFunction):
        mapped_x = obj_fun.operator.direct(x)
        mapped_direction = obj_fun.operator.direct(direction)
        hv = _multiply_with_composed_hessian(obj_fun.function, mapped_x, mapped_direction)
        return obj_fun.operator.adjoint(hv)
    return obj_fun.multiply_with_Hessian(x, direction)


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
        h1 = _multiply_with_composed_hessian(obj_fun, init_img, ones)
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
        inv_hessian_arr = np.zeros_like(
            hessian_diag_arr, dtype=np.result_type(hessian_diag_arr, np.float32)
        )
        np.reciprocal(
            hessian_diag_arr,
            out=inv_hessian_arr,
            where=hessian_diag_arr != 0,
        )
        hessian_diag.fill(inv_hessian_arr)

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
    kappas=None,
    pet_scale=None,
    spect_scale=None,
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
        if hasattr(kappas, "clone"):
            kappas = kappas.clone()
        else:
            kappas = EnhancedBlockDataContainer(
                *[kap.clone() for kap in kappas.containers]
            )

    priors = []

    # TNV (vectorial) prior - uses alpha/beta weighting
    if getattr(args, "use_tnv_prior", True) and getattr(args, "gamma_tnv", 1.0) > 0:
        precond_method = _resolve_precond_method_from_args(args)
        # Check for local weighting
        use_log_tnv = getattr(args, "use_log_tnv", False)
        use_local_weighting = getattr(args, "use_local_weighting", False)

        # Prevent using local weighting with log-TNV
        if use_log_tnv and use_local_weighting:
            logging.warning(
                "use_local_weighting is ignored for log-TNV "
                "(log transform already provides local weighting)"
            )
            use_local_weighting = False

        # Create kappa weights based on weighting type
        if use_local_weighting:

            pet_osem = get_array(initial_estimates[0])
            spect_osem = get_array(initial_estimates[1])
            
            # Compute small percentile-based epsilons to avoid division by zero
            pet_eps = np.percentile(pet_osem[pet_osem > 0], 0.5)
            spect_eps = np.percentile(spect_osem[spect_osem > 0], 0.5)

            # Create voxel-wise alpha/beta weight maps
            pet_weight_map = initial_estimates[0].clone()
            spect_weight_map = initial_estimates[1].clone()

            pet_weight_map.fill(args.alpha * args.gamma_tnv / np.maximum(pet_osem, pet_eps))
            spect_weight_map.fill(args.beta * args.gamma_tnv / np.maximum(spect_osem, spect_eps))

            # Apply local weights to existing kappas
            tnv_kappas = kappas.clone()
            tnv_kappas.containers[0].multiply(pet_weight_map, out=tnv_kappas.containers[0])
            tnv_kappas.containers[1].multiply(spect_weight_map, out=tnv_kappas.containers[1])

            logging.info(
                "Using local weighting for TNV: kappa * (alpha/osem) for PET, kappa * (beta/osem) for SPECT"
            )
        else:
            # Global weighting: uniform alpha/beta scaling
            tnv_kappas = EnhancedBlockDataContainer(
                initial_estimates[0].get_uniform_copy(args.alpha * args.gamma_tnv),
                initial_estimates[1].get_uniform_copy(args.beta * args.gamma_tnv),
            )

            # Apply base kappa weights
            for i, el in enumerate(tnv_kappas.containers):
                el.multiply(kappas.containers[i], out=el)

        # Select TNV variant: standard or log-domain
        use_log_tnv = getattr(args, "use_log_tnv", False)

        if use_log_tnv:
            # Compute per-modality epsilon from dynamic ranges
            epsilon_divisor = float(getattr(args, "epsilon_divisor", 100.0))

            # Extract arrays from initial_estimates (already in common geometry)
            pet_arr = get_array(initial_estimates[0])
            spect_arr = get_array(initial_estimates[1])

            # Filter for valid (positive, finite) values before percentile computation
            pet_valid = pet_arr[np.isfinite(pet_arr) & (pet_arr > 0)]
            spect_valid = spect_arr[np.isfinite(spect_arr) & (spect_arr > 0)]

            # Use pet_scale and spect_scale if provided, otherwise compute from percentiles
            # NOTE: pet_scale and spect_scale are INVERSE scaling factors (1/dynamic_range)
            # so we need to invert them to get actual dynamic ranges
            if pet_scale is not None and spect_scale is not None:
                pet_range = 1.0 / pet_scale
                spect_range = 1.0 / spect_scale
            else:
                # Fallback: compute from images directly
                dynamic_percentile = getattr(args, "dynamic_percentile", 95.0)
                pet_range = (
                    np.percentile(pet_valid, dynamic_percentile)
                    if len(pet_valid) > 0 else 1.0
                )
                spect_range = (
                    np.percentile(spect_valid, dynamic_percentile)
                    if len(spect_valid) > 0 else 1.0
                )

            # Compute epsilon values with fallback for zero/near-zero ranges
            log_eps_pet = max(pet_range / epsilon_divisor, 1e-10)
            log_eps_spect = max(spect_range / epsilon_divisor, 1e-10)
            log_eps_values = [log_eps_pet, log_eps_spect]

            logging.info(
                "Auto-computed log_eps: PET=%.6g, SPECT=%.6g (from dynamic ranges / %.3g)",
                log_eps_pet, log_eps_spect, epsilon_divisor
            )

            log_hessian = {
                "mm_diag_tight": "mm_jensen",
                "mm_diag_gershgorin_maj": "mm_jensen",
            }.get(precond_method, "mm_jensen")
            if precond_method not in {"mm_diag_tight", "mm_diag_gershgorin_maj"}:
                logging.warning(
                    "Log-TNV only supports diagonal MM-Jensen preconditioners; "
                    "falling back to %s.",
                    log_hessian,
                )
            vtv = WeightedLogVectorialTotalVariation(
                initial_estimates,
                tnv_kappas,
                args.delta,
                log_eps_values=log_eps_values,
                smoothing=getattr(args, "smoothing", "charbonnier"),
                anatomical=umap if args.directional_tnv else None,
                stable=getattr(args, "stable", True),
                tail_singular_values=getattr(args, "tail_singular_values", None),
                both_directions=getattr(args, "tnv_both_directions", True),
                stencil=getattr(args, "tnv_stencil", "6"),
                max_step=getattr(args, "tnv_max_step", 1),
                hessian=log_hessian,
                bnd_cond=getattr(args, "tnv_bnd_cond", "Periodic"),
            )
        else:
            vtv = WeightedVectorialTotalVariation(
                initial_estimates,
                tnv_kappas,
                args.delta,
                smoothing=getattr(args, "smoothing", "charbonnier"),
                anatomical=umap if args.directional_tnv else None,
                stable=getattr(args, "stable", True),
                tail_singular_values=getattr(args, "tail_singular_values", None),
                both_directions=getattr(args, "tnv_both_directions", True),
                stencil=getattr(args, "tnv_stencil", "6"),
                max_step=getattr(args, "tnv_max_step", 1),
                precond_method=precond_method,
                bnd_cond=getattr(args, "tnv_bnd_cond", "Periodic"),
            )

        priors.append(vtv)

    # Modality-specific TV priors
    if getattr(args, "use_modality_specific_priors", False):
        # Get gamma weights (these should already be scaled by gradient energy scaling in main())
        gamma_pet = getattr(args, "alpha", 0.0)
        gamma_spect = getattr(args, "beta", 0.0)

        if gamma_pet > 0 or gamma_spect > 0:
            # Create separate kappa weights for modality-specific priors using gamma weights
            tv_kappas = EnhancedBlockDataContainer(
                initial_estimates[0].get_uniform_copy(gamma_pet),
                initial_estimates[1].get_uniform_copy(gamma_spect),
            )

            # Apply base kappa weights
            for i, el in enumerate(tv_kappas.containers):
                el.multiply(kappas.containers[i], out=el)

            prior_kind = getattr(args, "prior", "tv")
            if prior_kind == "rdp":
                logging.warning(
                    "prior='rdp' requested but WeightedRDP is no longer available; "
                    "using weighted TV instead."
                )

            combined_tv = WeightedTotalVariation(
                initial_estimates,
                tv_kappas,
                delta=getattr(args, "delta"),
                anatomical=umap if args.directional_tv else None,
                stencil=getattr(args, "tv_stencil", "6"),
                both_directions=getattr(args, "tv_both_directions", False),
                max_step=getattr(args, "tv_max_step", getattr(args, "tnv_max_step", 1)),
                bnd_cond=getattr(args, "tv_bnd_cond", "Periodic"),
            )

            priors.append(combined_tv)

    # Combine priors
    if not priors:
        raise ValueError(
            "No priors enabled. Set use_tnv_prior=True or enable TV priors with gamma > 0"
        )
    else:
        return priors
