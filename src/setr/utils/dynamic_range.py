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

    def _scale_attr(attr: str, scale: float, source: str) -> None:
        if not hasattr(args, attr):
            return

        value = getattr(args, attr)
        if value is None:
            return

        initial_attr = f"{attr}_initial"
        scaled_attr = f"{attr}_scaled"

        # Preserve the very first value we saw so the CLI input is retained.
        if not hasattr(args, initial_attr):
            setattr(args, initial_attr, value)

        initial_value = getattr(args, initial_attr)
        scaled_value = initial_value * scale
        setattr(args, attr, scaled_value)
        setattr(args, scaled_attr, scaled_value)

        logging.info(
            "Adjusted %s from %s to %s using %s dynamic-range scaling",
            attr,
            f"{initial_value:.6g}",
            f"{scaled_value:.6g}",
            source,
        )

    _scale_attr("alpha", pet_scale, "PET")
    _scale_attr("beta", spect_scale, "SPECT")
    _scale_attr("gamma_pet", pet_scale, "PET")
    _scale_attr("gamma_spect", spect_scale, "SPECT")


def apply_gradient_energy_scaling(args, pet_scale: float, spect_scale: float) -> None:
    """Backward-compatible alias for older gradient scaling usage."""

    apply_dynamic_range_scaling(args, pet_scale, spect_scale)
