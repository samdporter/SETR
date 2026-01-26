"""
Mask generation utilities operating on flat voxel buffers.

The helpers below cover the most common ROI definitions we need for quick
reconstruction sweeps:

* ``nonzero`` – selects every voxel with a strictly positive value.
* ``fraction`` – keeps voxels above a fraction of the global maximum.
* ``absolute`` – keeps voxels above an absolute intensity threshold.
* ``percentile`` – retains voxels above a percentile computed over all voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math


@dataclass
class MaskConfig:
    """Configuration for mask creation."""

    method: str = "fraction"
    fraction: float = 0.5
    absolute_threshold: float | None = None
    percentile: float = 90.0

    def clamp(self) -> "MaskConfig":
        fraction = min(max(self.fraction, 0.0), 1.0)
        percentile = min(max(self.percentile, 0.0), 100.0)
        return MaskConfig(
            method=self.method,
            fraction=fraction,
            absolute_threshold=self.absolute_threshold,
            percentile=percentile,
        )


def create_mask(values: Sequence[float], config: MaskConfig) -> List[bool]:
    """
    Generate a boolean mask according to *config*.

    The returned list mirrors the length of *values*.  All mask operations work
    on the flattened voxel buffer – this keeps the implementation simple and is
    sufficient for global region-of-interest measurements.
    """

    n = len(values)
    if n == 0:
        return []

    config = config.clamp()
    method = config.method.lower()
    mask = [False] * n

    if method == "nonzero":
        for idx, value in enumerate(values):
            mask[idx] = float(value) > 0.0
        return mask

    if method == "absolute":
        if config.absolute_threshold is None:
            raise ValueError("Absolute threshold requested but 'absolute_threshold' is not set.")
        threshold = float(config.absolute_threshold)
    elif method == "fraction":
        maximum = max(float(v) for v in values)
        threshold = maximum * config.fraction
    elif method == "percentile":
        ordered = sorted(float(v) for v in values)
        if not ordered:
            return mask
        rank = config.percentile / 100.0
        rank = min(max(rank, 0.0), 1.0)
        index = int(math.floor((len(ordered) - 1) * rank))
        threshold = ordered[index]
    else:
        raise ValueError(f"Unknown mask method: {config.method!r}")

    for idx, value in enumerate(values):
        mask[idx] = float(value) >= threshold
    return mask


__all__ = ["MaskConfig", "create_mask"]
