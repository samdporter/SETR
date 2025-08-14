"""
setr.priors
Priors for synergistic reconstruction.
"""

from .mutual_information import (
    MutualInformationGradientPrior,
    MutualInformationImagePrior,
)
from .rdp import RelativeDifferencePrior, WeightedRDP
from .vtv.vtv import (
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
)

__all__ = [
    "WeightedTotalVariation",
    "WeightedVectorialTotalVariation",
    "MutualInformationGradientPrior",
    "MutualInformationImagePrior",
    "RelativeDifferencePrior",
    "WeightedRDP",
]
