"""
Numerical constants for VTV prior implementation.

This module provides dtype-aware epsilon values for numerical stability in
the Vectorial Total Variation (VTV) prior implementation. These constants
are critical for avoiding division-by-zero and numerical instability issues
in low-gradient regions (e.g., background in PET/SPECT images).

Design considerations:
- image scales approximately 1
- Float32 machine epsilon: ~1.2e-7
- Epsilon values must be: << image scale, >> machine epsilon
"""

import torch


def get_sqrt_epsilon(dtype=torch.float32):
    """
    Epsilon for sqrt stabilization: sqrt(x^2 + eps)

    Used in operations like:
        r = sqrt(||gradient||^2 + eps)

    Must satisfy:
        - << typical image values (1)
        - >> machine epsilon (1.2e-7 for float32)

    Args:
        dtype: torch dtype (float32 or float64)

    Returns:
        float: Appropriate epsilon value for the given dtype
    """
    if dtype == torch.float32:
        # Keep comfortably above float32 machine epsilon to avoid
        # amplification from near-zero gradients in background regions.
        return 1e-6
    else:  # float64
        return 1e-12


def get_division_epsilon(dtype=torch.float32):
    """
    Epsilon for division stabilization: x / (y + eps)

    Used as threshold in torch.where() operations:
        result = torch.where(S > eps_div, f(S) / S, 0)

    Values below this threshold are considered effectively zero.

    Args:
        dtype: torch dtype (float32 or float64)

    Returns:
        float: Appropriate epsilon value for the given dtype
    """
    if dtype == torch.float32:
        # Division guard should exceed float32 machine epsilon.
        return 1e-6
    else:  # float64
        return 1e-14


def get_gradient_floor(dtype=torch.float32):
    """
    Minimum meaningful gradient magnitude.

    Gradients smaller than this are treated as effectively zero.
    Used in operations like:
        grad_safe = torch.maximum(||grad||, eps_floor)
        result = f(||grad||) / grad_safe

    Args:
        dtype: torch dtype (float32 or float64)

    Returns:
        float: Appropriate floor value for the given dtype
    """
    if dtype == torch.float32:
        # Minimum meaningful gradient magnitude for stable normalization.
        return 1e-6
    else:  # float64
        return 1e-14
