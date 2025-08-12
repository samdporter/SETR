"""
synergistic_recon.core.gradients

GPU-accelerated and reference CPU gradient-based operators.
"""

from .gradients import (
    AdjointOperator,
    CompositionOperator,
    CPUFiniteDifferenceOperator,
    DirectionalGradient,
    Gradient,
    Jacobian,
    Operator,
    ScaledOperator,
    directional_op,
    fast_norm_parallel_3d,
    fast_norm_parallel_4d,
    gpu_directional_op,
    power_iteration,
)

__all__ = [
    "Operator",
    "AdjointOperator",
    "ScaledOperator",
    "CompositionOperator",
    "Jacobian",
    "Gradient",
    "DirectionalGradient",
    "CPUFiniteDifferenceOperator",
    "directional_op",
    "gpu_directional_op",
    "power_iteration",
    "fast_norm_parallel_3d",
    "fast_norm_parallel_4d",
]
