import logging
from typing import Tuple

import numpy as np


def dynamic_range_scale_sirf(
    x_pet,
    x_spect,
    percentile_high: float = 99.0,
    percentile_low: float = 1.0,
    mask=None,
    use_absolute: bool = True,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Compute per-modality inverse dynamic-range scales for PET and SPECT.

    Returns ``(scale_pet, scale_spect)`` so consumers can normalise each
    modality independently.
    """

    def _get_array(obj):
        if hasattr(obj, "asarray"):
            return obj.asarray()
        if hasattr(obj, "as_array"):
            return obj.as_array()
        raise AttributeError(
            f"Object of type {type(obj)} has neither asarray() nor as_array() method"
        )

    def _prepare(arr):
        if mask is not None:
            m = _get_array(mask).astype(bool)
            arr = arr[m]
        else:
            arr = arr.ravel()
        if use_absolute:
            arr = np.abs(arr)
        arr = arr[np.isfinite(arr)]
        return arr

    xp = _prepare(_get_array(x_pet))
    xs = _prepare(_get_array(x_spect))

    if xp.size == 0 or xs.size == 0:
        logging.warning(
            "Dynamic range scaling fallback to 1: empty or invalid data encountered."
        )
        return 1.0, 1.0

    def _range(stat_arr):
        hi = (
            np.percentile(stat_arr, percentile_high)
            if percentile_high is not None
            else np.max(stat_arr)
        )
        lo = (
            np.percentile(stat_arr, percentile_low)
            if percentile_low is not None
            else np.min(stat_arr)
        )
        return max(float(hi - lo), eps)

    pet_range = _range(xp)
    spect_range = _range(xs)
    pet_scale = 1.0 / max(pet_range, eps)
    spect_scale = 1.0 / max(spect_range, eps)
    return pet_scale, spect_scale


def apply_dynamic_range_scaling(args, pet_scale: float, spect_scale: float) -> None:
    """Apply per-modality dynamic range scaling to all prior weights."""

    args.alpha *= pet_scale
    logging.info(
        "Adjusted alpha to %s using PET dynamic-range scaling", f"{args.alpha:.6g}"
    )

    args.beta *= spect_scale
    logging.info(
        "Adjusted beta to %s using SPECT dynamic-range scaling", f"{args.beta:.6g}"
    )

    if hasattr(args, "gamma_pet"):
        args.gamma_pet *= pet_scale
        logging.info(
            "Adjusted gamma_pet to %s using PET dynamic-range scaling",
            f"{args.gamma_pet:.6g}",
        )

    if hasattr(args, "gamma_spect"):
        args.gamma_spect *= spect_scale
        logging.info(
            "Adjusted gamma_spect to %s using SPECT dynamic-range scaling",
            f"{args.gamma_spect:.6g}",
        )


def apply_gradient_energy_scaling(args, pet_scale: float, spect_scale: float) -> None:
    """Backward-compatible alias for older gradient scaling usage."""

    apply_dynamic_range_scaling(args, pet_scale, spect_scale)
