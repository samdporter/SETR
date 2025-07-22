"""
synergistic_recon.kernel.stir

STIR-based kernel operator.
"""

from .stir_kem import KernelOperator as STIRKernelOperator

__all__ = [
    "STIRKernelOperator",
]
