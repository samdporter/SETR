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
from .vtv import (
    schatten_norm_gpu_slow,
    schatten_norm_gpu_stable,
    schatten_norm_gpu,
)

__all__ = [
    "WeightedTotalVariation",
    "WeightedVectorialTotalVariation",
    "MutualInformationGradientPrior",
    "MutualInformationImagePrior",
    "RelativeDifferencePrior",
    "WeightedRDP",
    "schatten_norm_gpu_slow",
    "schatten_norm_gpu_stable",
    "schatten_norm_gpu",
]
