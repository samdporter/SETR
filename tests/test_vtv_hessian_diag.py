import math
from types import SimpleNamespace

import pytest

try:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    pytest.skip(f"Skipping VTV Hessian tests; PyTorch not available: {e}", allow_module_level=True)

try:
    from setr.priors.vtv.vtv import WeightedVectorialTotalVariation
except Exception as e:
    pytest.skip(f"Skipping VTV Hessian tests; dependency not available: {e}", allow_module_level=True)
from setr.priors.vtv.schatten_norm_gpu_slow import (
    GPUVectorialTotalVariation as VTVSlowBackend,
)
from setr.priors.vtv.common import fair_hessian_diag


class MockJacobian:
    """Minimal Jacobian stub for testing diagonal preconditioners.

    Provides fixed-gradient forward, simple zero adjoint (as configured),
    and constant sensitivities to make expectations easy to compute.
    """

    def __init__(self, J_field: torch.Tensor, S_field: torch.Tensor):
        # Shapes: J_field (..., M, d), S_field (..., M, d)
        self._J = J_field
        self._S = S_field

    def direct(self, x_arr: torch.Tensor) -> torch.Tensor:
        # Return fixed gradient field, independent of x
        return self._J

    def adjoint(self, grad_field: torch.Tensor) -> torch.Tensor:
        # For tests, use a zero adjoint when J is zero; else return zeros by default
        # This is sufficient for the slow-method test where J is zero
        return torch.zeros_like(self._J[..., 0])  # (..., M)

    def sensitivity(self, x_arr: torch.Tensor) -> torch.Tensor:
        return self._S


def _make_wvtv_harness(J_field: torch.Tensor, S_field: torch.Tensor, weights: torch.Tensor, eps=1.0):
    """Create a WeightedVectorialTotalVariation-like object without running __init__.

    Only the core internals needed by the tested private methods are set.
    """
    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.jacobian = MockJacobian(J_field=J_field, S_field=S_field)
    vtv.weights = weights
    vtv.smoothing = "fair"
    vtv.vtv = VTVSlowBackend(eps=eps, norm="nuclear", smoothing_function="fair", numpy_out=False)
    return vtv


def test_m1_fastest_exact_matches_vector_norm():
    # Single-modality equivalence: fastest_exact should reduce to vector-norm α,β structure
    nx = ny = nz = 1
    M, d = 1, 3

    # Create a simple nontrivial gradient across directions
    J_field = torch.tensor([[[[[1.0, 2.0, 0.5]]]]], device=device)  # shape (1,1,1,1,3)
    S_field = torch.tensor([[[[[1.0, 2.0, 3.0]]]]], device=device)  # varying sensitivities
    weights = torch.tensor([[[[2.0]]]], device=device)  # (1,1,1,1)

    vtv = _make_wvtv_harness(J_field, S_field, weights, eps=1.0)

    x_arr = torch.zeros(nx, ny, nz, M, device=device)
    H = vtv._preconditioner_weights_core_fastest_exact(x_arr, epsilon=1e-12)

    # Expected vector-norm mapping per direction: h_dir = α + β * (U_d^2 / r^2)
    A = weights.unsqueeze(-1) * J_field  # (..., 1, d)
    r2 = torch.sum(A * A, dim=-1)
    r = torch.sqrt(r2 + 1e-12)

    # Fair smoother
    from setr.priors.vtv.common import fair_grad

    alpha = fair_grad(r, torch.tensor(1.0)) / r
    beta = fair_hessian_diag(r, torch.tensor(1.0)) - alpha
    frac = (A * A) / (r2.unsqueeze(-1) + 1e-12)
    h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac

    expected = torch.sum((weights * weights).unsqueeze(-1) * (S_field * S_field) * h_dir, dim=-1)

    assert torch.allclose(H, expected, rtol=1e-6, atol=1e-6)


def test_slow_includes_isotropic_alpha_term_when_J_zero():
    # When J == 0, principal backprojection terms vanish; only the isotropic α term remains.
    nx = ny = nz = 1
    M, d = 2, 2
    eps = 1.0

    J_field = torch.zeros(nx, ny, nz, M, d, device=device)  # zero gradients
    S_field = torch.ones(nx, ny, nz, M, d, device=device)   # unit sensitivities
    weights = torch.tensor([[[[2.0, 3.0]]]], device=device)  # shape (1,1,1,M)

    vtv = _make_wvtv_harness(J_field, S_field, weights, eps=eps)

    x_arr = torch.zeros(nx, ny, nz, M, device=device)
    H = vtv._preconditioner_weights_core_slow(x_arr)

    # α_total = Σ_k φ'(σ_k)/σ_k ; for fair and σ_k=0, φ'(0)/0 -> 1/eps; r = min(M,d) = 2
    alpha_total = 2.0 / eps
    S_jm = torch.sum(S_field * S_field, dim=-1)  # equals d for each modality
    expected = alpha_total * (weights * weights) * S_jm

    assert torch.allclose(H, expected, rtol=1e-6, atol=1e-6)
