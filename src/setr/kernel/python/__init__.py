"""
setr.kernel.python

Pure-Python and Numba-based kernel operators.
"""

from .my_kem import (
    NUMBA_AVAIL,
    SLIDING_WINDOW_AVAIL,
    BaseKernelOperator,
    KernelOperator,
    NumbaKernelOperator,
    get_kernel_operator,
)

__all__ = [
    "get_kernel_operator",
    "BaseKernelOperator",
    "KernelOperator",
    "NumbaKernelOperator",
    "NUMBA_AVAIL",
    "SLIDING_WINDOW_AVAIL",
]
