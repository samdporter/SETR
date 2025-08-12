"""
setr.cil_extensions.operators
"""

from .operators import (
    AdjointOperator,
    CouchShiftOperator,
    DirectionalOperator,
    ImageCombineOperator,
    ImageResampleOperator,
    ImageSummationOperator,
    NaNToZeroOperator,
    NiftyResampleOperator,
    ScalingOperator,
    TruncationOperator,
    ZeroEndSlicesOperator,
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
