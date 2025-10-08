"""VTV (Vectorial Total Variation) priors and utilities."""

from .vtv import (
    TotalVariation,
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
)

__all__ = [
    "WeightedVectorialTotalVariation",
    "WeightedTotalVariation",
    "TotalVariation",
]
