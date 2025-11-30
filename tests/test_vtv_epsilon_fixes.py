"""
Test numerical stability fixes for VTV prior.

This tests that the epsilon value changes prevent division-by-near-zero
from amplifying background noise.
"""

import numpy as np
import torch

from setr.priors.vtv.numerical_constants import (
    get_division_epsilon,
    get_gradient_floor,
    get_sqrt_epsilon,
)


def test_epsilon_values_reasonable():
    """Verify epsilon values are reasonable for float32 and float64."""
    # Epsilon values should be above machine epsilon
    machine_eps_32 = np.finfo(np.float32).eps
    machine_eps_64 = np.finfo(np.float64).eps

    # Float32
    eps_sqrt_32 = get_sqrt_epsilon(torch.float32)
    eps_div_32 = get_division_epsilon(torch.float32)
    eps_floor_32 = get_gradient_floor(torch.float32)

    assert eps_sqrt_32 > machine_eps_32, "sqrt epsilon should be above machine epsilon"
    assert eps_div_32 > machine_eps_32, "division epsilon should be above machine epsilon"
    assert eps_floor_32 > machine_eps_32, "gradient floor should be above machine epsilon"

    # Float64 should use smaller values than float32
    eps_sqrt_64 = get_sqrt_epsilon(torch.float64)
    eps_div_64 = get_division_epsilon(torch.float64)
    eps_floor_64 = get_gradient_floor(torch.float64)

    assert eps_sqrt_64 < eps_sqrt_32, "float64 sqrt epsilon should be smaller than float32"
    assert eps_div_64 < eps_div_32, "float64 division epsilon should be smaller than float32"
    assert eps_floor_64 < eps_floor_32, "float64 gradient floor should be smaller than float32"


def test_division_by_zero_protection():
    """Verify that near-zero singular values don't cause infinities."""
    from setr.priors.vtv.schatten_norm_gpu_stable import GPUVectorialTotalVariation

    # Create a synthetic field with near-zero gradients
    # Shape: (batch, M, d) where M=2 modalities, d=3 spatial directions
    batch_size = 10
    M, d = 2, 3

    # Create matrices with very small singular values (background-like)
    # These should be order 1e-15 (numerical noise level)
    x = torch.randn(batch_size, M, d, dtype=torch.float32) * 1e-15

    # Initialize VTV with nuclear norm
    vtv = GPUVectorialTotalVariation(
        eps=1e-6,  # delta parameter
        norm="nuclear",
        smoothing_function="charbonnier",
        numpy_out=False,
    )

    # Compute gradient - should NOT produce infinities or NaN
    grad = vtv.gradient(x)

    assert torch.all(torch.isfinite(grad)), "Gradient contains NaN or Inf values"
    assert torch.all(torch.abs(grad) < 1e10), f"Gradient too large: max={torch.max(torch.abs(grad))}"


def test_frobenius_gradient_floor():
    """Verify Frobenius norm gradient uses appropriate floor."""
    from setr.priors.vtv.schatten_norm_gpu_stable import GPUVectorialTotalVariation

    # Create field with truly zero gradients
    batch_size = 5
    M, d = 2, 3
    x = torch.zeros(batch_size, M, d, dtype=torch.float32)

    # Initialize VTV with Frobenius norm
    vtv = GPUVectorialTotalVariation(
        eps=1e-6,
        norm="frobenius",
        smoothing_function="charbonnier",
        numpy_out=False,
    )

    # Compute gradient - should be bounded and finite
    grad = vtv.gradient(x)

    assert torch.all(torch.isfinite(grad)), "Frobenius gradient contains NaN or Inf"
    assert torch.all(grad == 0.0), f"Zero input should give zero gradient, got max={torch.max(torch.abs(grad))}"


def test_background_region_gradient_bounded():
    """
    Verify gradients in uniform background regions are near-zero.

    This is the key test: background regions should not have amplified noise
    from division by near-zero values.
    """
    from setr.priors.vtv.schatten_norm_gpu_stable import GPUVectorialTotalVariation

    # Create a synthetic phantom with uniform background
    # Spatial: 16x16x8, 2 modalities, 3 directions
    nx, ny, nz = 16, 16, 8
    M, d = 2, 3

    # Background value: 1e-8 (100x smaller than typical PET activity)
    background_value = 1e-8

    # Create uniform field (constant in space, so gradients should be zero)
    x = torch.full((nx, ny, nz, M, d), background_value, dtype=torch.float32)

    # Add tiny numerical noise (order machine epsilon)
    noise = torch.randn(nx, ny, nz, M, d, dtype=torch.float32) * 1e-10
    x = x + noise

    vtv = GPUVectorialTotalVariation(
        eps=1e-6,
        norm="nuclear",
        smoothing_function="charbonnier",
        numpy_out=False,
    )

    # Compute gradient
    grad = vtv.gradient(x)

    # Gradient magnitude should be near zero (not amplified to 1e3 or higher)
    grad_mag = torch.linalg.norm(grad, dim=(-2, -1))  # Frobenius norm per voxel
    max_grad = torch.max(grad_mag)
    mean_grad = torch.mean(grad_mag)

    # With proper epsilon values, gradient should be bounded by O(1)
    # Old code with 1e-12 epsilon would amplify to ~1e3 or higher
    # With 1e-8 epsilon, gradient should stay O(0.01-0.1) for uniform regions
    assert max_grad < 1.0, f"Background gradient too large: max={max_grad} (severe noise amplification detected)"
    assert mean_grad < 0.1, f"Mean background gradient too large: mean={mean_grad}"

    # Verify no extreme outliers (would indicate division by near-zero)
    assert torch.all(torch.isfinite(grad)), "Gradient contains NaN or Inf"

    print(f"✓ Background gradient bounded: max={max_grad:.2e}, mean={mean_grad:.2e}")


if __name__ == "__main__":
    # Run tests
    print("Testing epsilon values...")
    test_epsilon_values_reasonable()
    print("✓ Epsilon values reasonable\n")

    print("Testing division-by-zero protection...")
    test_division_by_zero_protection()
    print("✓ Division-by-zero protection working\n")

    print("Testing Frobenius gradient floor...")
    test_frobenius_gradient_floor()
    print("✓ Frobenius gradient floor working\n")

    print("Testing background region gradients...")
    test_background_region_gradient_bounded()
    print("✓ Background gradients bounded\n")

    print("=" * 60)
    print("All numerical stability tests PASSED!")
    print("=" * 60)
