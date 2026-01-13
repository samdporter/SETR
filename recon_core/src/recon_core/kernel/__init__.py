"""
synergistic_recon.kernel

Top-level kernel package. Exposes factory and subpackages.
"""

from .python import (  # noqa: F401
    BaseKernelOperator,
    KernelOperator as PythonKernelOperator,
    NumbaKernelOperator,
    get_kernel_operator,
)

try:  # pragma: no cover - optional dependency
    from .stir import STIRKernelOperator
except Exception:  # pragma: no cover - optional dependency
    STIRKernelOperator = None  # type: ignore[misc]

__all__ = [
    "get_kernel_operator",
    "BaseKernelOperator",
    "PythonKernelOperator",
    "NumbaKernelOperator",
    "STIRKernelOperator",
]
