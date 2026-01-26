"""
synergistic_recon.kernel

Top-level kernel package. Exposes factory and subpackages.
"""

try:  # pragma: no cover - optional dependency
    from .stir import STIRKernelOperator
except Exception:  # pragma: no cover - optional dependency
    STIRKernelOperator = None  # type: ignore[misc]

__all__ = [
    "STIRKernelOperator",
]
