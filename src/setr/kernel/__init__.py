"""
synergistic_recon.kernel

Top-level kernel package. Exposes factory and subpackages.
"""

from .python import (
    get_kernel_operator,
    BaseKernelOperator,
    KernelOperator as PythonKernelOperator,
    NumbaKernelOperator
)
from .stir import STIRKernelOperator

__all__ = [
    "get_kernel_operator",
    "BaseKernelOperator",
    "PythonKernelOperator",
    "NumbaKernelOperator",
    "STIRKernelOperator",
]
