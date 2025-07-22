"""
setr.priors
Priors for synergistic reconstruction.
"""

from .vtv.vtv import (
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
)

from .mutual_information import (
    MutualInformationGradientPrior,
    MutualInformationObjective,
)

__all__ = [
    "WeightedTotalVariation",
    "WeightedVectorialTotalVariation",
    "MutualInformationGradientPrior",
    "MutualInformationObjective",
]