"""
setr.priors
Priors for synergistic reconstruction.
"""

from .mutual_information import (
    MutualInformationGradientPrior,
    MutualInformationImagePrior,
)
from .vtv.vtv import (
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
)

__all__ = [
    "WeightedTotalVariation",
    "WeightedVectorialTotalVariation",
    "MutualInformationGradientPrior",
    "MutualInformationImagePrior",
]
