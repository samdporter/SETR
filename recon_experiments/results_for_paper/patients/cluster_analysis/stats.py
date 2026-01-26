"""
Numerically stable statistics for flattened voxel buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence
import math


def compute_basic_stats(values: Sequence[float], mask: Sequence[bool] | None = None) -> Dict[str, float]:
    """
    Compute summary statistics for *values* restricted to ``mask`` (if
    provided).

    The implementation uses Welford's online algorithm for numerical
    stability.  ``mask`` must either be ``None`` or a sequence with the same
    length as *values*.
    """

    if mask is not None and len(mask) != len(values):
        raise ValueError("Mask length does not match the number of voxels")

    count = 0
    mean = 0.0
    m2 = 0.0
    total = 0.0
    sum_squares = 0.0
    min_value: float | None = None
    max_value: float | None = None

    iterator: Iterable[tuple[int, float]]
    iterator = enumerate(values)

    for idx, raw_value in iterator:
        if mask is not None and not mask[idx]:
            continue
        value = float(raw_value)
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
        total += value
        sum_squares += value * value
        if min_value is None or value < min_value:
            min_value = value
        if max_value is None or value > max_value:
            max_value = value

    if count == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "m2": 0.0,
            "coefficient_of_variation": math.inf,
            "l2_norm": 0.0,
        }

    variance = m2 / (count - 1) if count > 1 else 0.0
    std = math.sqrt(variance)
    cov = math.inf if mean == 0 else std / abs(mean)

    return {
        "count": float(count),
        "mean": mean,
        "variance": variance,
        "std": std,
        "min": min_value if min_value is not None else 0.0,
        "max": max_value if max_value is not None else 0.0,
        "sum": total,
        "sum_squares": sum_squares,
        "m2": m2,
        "coefficient_of_variation": cov,
        "l2_norm": math.sqrt(sum_squares),
    }


def combine_partial_stats(lhs: Dict[str, float], rhs: Dict[str, float]) -> Dict[str, float]:
    """
    Combine two statistics dictionaries as returned by :func:`compute_basic_stats`.

    The function returns a *new* dictionary; the inputs remain untouched.
    """

    count_a = int(lhs["count"])
    count_b = int(rhs["count"])
    if count_a == 0:
        return dict(rhs)
    if count_b == 0:
        return dict(lhs)

    total_count = count_a + count_b
    mean_a = lhs["mean"]
    mean_b = rhs["mean"]
    delta = mean_b - mean_a
    mean = mean_a + delta * (count_b / total_count)
    m2 = lhs["m2"] + rhs["m2"] + delta * delta * (count_a * count_b / total_count)
    total_sum = lhs["sum"] + rhs["sum"]
    sum_squares = lhs["sum_squares"] + rhs["sum_squares"]
    min_value = min(lhs["min"], rhs["min"])
    max_value = max(lhs["max"], rhs["max"])

    variance = m2 / (total_count - 1) if total_count > 1 else 0.0
    std = math.sqrt(variance)
    cov = math.inf if mean == 0 else std / abs(mean)

    return {
        "count": float(total_count),
        "mean": mean,
        "variance": variance,
        "std": std,
        "min": min_value,
        "max": max_value,
        "sum": total_sum,
        "sum_squares": sum_squares,
        "m2": m2,
        "coefficient_of_variation": cov,
        "l2_norm": math.sqrt(sum_squares),
    }


__all__ = ["compute_basic_stats", "combine_partial_stats"]
