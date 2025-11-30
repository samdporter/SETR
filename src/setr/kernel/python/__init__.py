"""
setr.kernel.python

Numba-based kernel operators (KRL implementation).
"""

from .my_kem import (
    NUMBA_AVAIL,
    BaseKernelOperator,
    KernelOperator,
    get_kernel_operator,
)

# Backward compatibility alias
NumbaKernelOperator = KernelOperator
SLIDING_WINDOW_AVAIL = False  # No longer used

__all__ = [
    "get_kernel_operator",
    "BaseKernelOperator",
    "KernelOperator",
    "NumbaKernelOperator",
    "NUMBA_AVAIL",
    "SLIDING_WINDOW_AVAIL",
]
