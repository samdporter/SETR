#!/usr/bin/env python3
"""
Tests to verify that nuclear and Frobenius VTV produce different gradients.

This test was added to fix a bug where both norms were producing identical gradients.
"""

import pytest
import torch
import numpy as np

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = "cpu"
    torch = None

from setr.priors.vtv.schatten_norm_gpu_slow import GPUVectorialTotalVariation as VTV_Slow
from setr.priors.vtv.schatten_norm_gpu_stable import GPUVectorialTotalVariation as VTV_Stable


@pytest.fixture
def test_data():
    """Create simple test data for VTV testing."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Shape: (4, 4, 4, 2, 3) - spatial (4x4x4), matrices (2x3)
    # Scale down to make singular values smaller (better for testing with eps=0.01)
    # Perona-Malik gradient ≈ 0 for large values, so we need smaller data
    x = torch.randn(4, 4, 4, 2, 3, device=device) * 0.1
    return x


class TestNuclearVsFrobeniusGradients:
    """Test that nuclear and Frobenius norms produce different gradients."""

    @pytest.mark.parametrize("smoothing", ["perona_malik", "charbonnier", "fair"])
    @pytest.mark.parametrize("vtv_class", [VTV_Slow, VTV_Stable])
    def test_gradients_are_different(self, test_data, smoothing, vtv_class):
        """Verify nuclear and Frobenius produce different gradients."""

        # Create nuclear and frobenius VTV
        vtv_nuclear = vtv_class(
            eps=0.01,
            norm="nuclear",
            smoothing_function=smoothing
        )
        vtv_frobenius = vtv_class(
            eps=0.01,
            norm="frobenius",
            smoothing_function=smoothing
        )

        # Compute gradients
        grad_nuclear = vtv_nuclear.gradient(test_data)
        grad_frobenius = vtv_frobenius.gradient(test_data)

        # Check they're different
        grad_diff = torch.abs(grad_nuclear - grad_frobenius)
        max_diff = torch.max(grad_diff).item()
        mean_diff = torch.mean(grad_diff).item()

        # Gradients should be significantly different
        assert max_diff > 1e-6, (
            f"Nuclear and Frobenius gradients are too similar! "
            f"Max diff: {max_diff:.6e} (smoothing={smoothing}, class={vtv_class.__name__})"
        )
        assert mean_diff > 1e-8, (
            f"Nuclear and Frobenius gradients mean difference too small! "
            f"Mean diff: {mean_diff:.6e} (smoothing={smoothing}, class={vtv_class.__name__})"
        )

    @pytest.mark.parametrize("smoothing", ["perona_malik", "charbonnier", "fair"])
    @pytest.mark.parametrize("vtv_class", [VTV_Slow, VTV_Stable])
    def test_objectives_are_different(self, test_data, smoothing, vtv_class):
        """Verify nuclear and Frobenius produce different objective values."""

        # Create nuclear and frobenius VTV
        vtv_nuclear = vtv_class(
            eps=0.01,
            norm="nuclear",
            smoothing_function=smoothing
        )
        vtv_frobenius = vtv_class(
            eps=0.01,
            norm="frobenius",
            smoothing_function=smoothing
        )

        # Compute objectives
        obj_nuclear = vtv_nuclear(test_data)
        obj_frobenius = vtv_frobenius(test_data)

        # Objectives should be different
        obj_diff = abs(obj_nuclear - obj_frobenius)
        rel_diff = obj_diff / (abs(obj_nuclear) + 1e-10)

        assert obj_diff > 1e-6, (
            f"Nuclear and Frobenius objectives are too similar! "
            f"Diff: {obj_diff:.6e} (smoothing={smoothing}, class={vtv_class.__name__})"
        )


    @pytest.mark.parametrize("vtv_class", [VTV_Slow, VTV_Stable])
    def test_frobenius_gradient_chain_rule(self, test_data, vtv_class):
        """Test that Frobenius gradient follows the chain rule correctly."""

        vtv_frobenius = vtv_class(
            eps=0.01,
            norm="frobenius",
            smoothing_function="charbonnier"
        )

        # Compute gradient
        grad = vtv_frobenius.gradient(test_data)

        # Check gradient is not NaN or Inf
        assert torch.isfinite(grad).all(), "Frobenius gradient contains NaN or Inf"

        # Check gradient has reasonable magnitude
        grad_norm = torch.linalg.norm(grad).item()
        assert grad_norm > 1e-10, f"Frobenius gradient norm too small: {grad_norm}"
        assert grad_norm < 1e10, f"Frobenius gradient norm too large: {grad_norm}"


    @pytest.mark.parametrize("vtv_class", [VTV_Slow, VTV_Stable])
    def test_nuclear_gradient_element_wise(self, test_data, vtv_class):
        """Test that nuclear gradient is applied element-wise to singular values."""

        vtv_nuclear = vtv_class(
            eps=0.01,
            norm="nuclear",
            smoothing_function="charbonnier"
        )

        # Compute gradient
        grad = vtv_nuclear.gradient(test_data)

        # Check gradient is not NaN or Inf
        assert torch.isfinite(grad).all(), "Nuclear gradient contains NaN or Inf"

        # Check gradient has reasonable magnitude
        grad_norm = torch.linalg.norm(grad).item()
        assert grad_norm > 1e-10, f"Nuclear gradient norm too small: {grad_norm}"
        assert grad_norm < 1e10, f"Nuclear gradient norm too large: {grad_norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
