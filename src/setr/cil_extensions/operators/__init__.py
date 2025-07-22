"""
setr.cil_extensions.operators
"""

from .operators import (
    AdjointOperator,
    ScalingOperator,
    ZeroEndSlicesOperator,
    NaNToZeroOperator,
    TruncationOperator,
    DirectionalOperator,
    NiftyResampleOperator,
    CouchShiftOperator,
    ImageCombineOperator,
    ImageResampleOperator,
    ImageSummationOperator,
)

__all__ = [
    "AdjointOperator",
    "ScalingOperator",
    "ZeroEndSlicesOperator",
    "NaNToZeroOperator",
    "TruncationOperator",
    "DirectionalOperator",
    "NiftyResampleOperator",
    "CouchShiftOperator",
    "ImageCombineOperator",
    "ImageResampleOperator",
    "ImageSummationOperator",
]
