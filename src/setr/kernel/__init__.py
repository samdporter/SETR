"""
synergistic_recon.kernel

Top-level kernel package. Exposes factory and subpackages.
"""

from .python import (
    BaseKernelOperator,
    KernelOperator as PythonKernelOperator,
    NumbaKernelOperator,
    get_kernel_operator,
)
from .stir import STIRKernelOperator

__all__ = [
    "get_kernel_operator",
    "BaseKernelOperator",
    "PythonKernelOperator",
    "NumbaKernelOperator",
    "STIRKernelOperator",
]
