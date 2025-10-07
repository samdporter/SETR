import pytest

from tests._vtv_test_utils import DEVICE, make_vtv_harness, require_vtv, torch

require_vtv()

from setr.priors.vtv.common import fair_grad, fair_hessian_diag


def _compute_diag(vtv, method, x_arr):
    if method == "svd_principal_alpha":
        return vtv._preconditioner_weights_core_slow(x_arr)
    if method == "mm_jensen":
        return vtv._preconditioner_weights_core_fast(x_arr, eta=1.0, epsilon=1e-12)
    if method == "frobenius_surrogate_pd":
        return vtv._preconditioner_weights_core_fastest_positive(x_arr, eta=1.0, epsilon=1e-12)
    if method == "vector_tv_per_modality":
        return vtv._preconditioner_weights_core_fastest_exact(x_arr, epsilon=1e-12)
    raise ValueError(f"Unknown method: {method}")


@pytest.mark.parametrize(
    "method",
    ["svd_principal_alpha", "mm_jensen", "frobenius_surrogate_pd", "vector_tv_per_modality"],
)
def test_preconditioner_outputs_are_finite_and_nonnegative(method):
    """All diagonals should be finite and non-negative for Fair smoothing."""

    torch.manual_seed(1)

    J_field = torch.randn(1, 1, 1, 2, 3, device=DEVICE)
    S_field = torch.abs(torch.randn(1, 1, 1, 2, 3, device=DEVICE)) + 0.1
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.5
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = make_vtv_harness(J_field, S_field, weights, eps=1.0)
    diag = _compute_diag(vtv, method, x_arr)

    assert torch.all(torch.isfinite(diag))
    assert torch.all(diag >= 0)


def _expected_mm_jensen(vtv, J_field, S_field, weights):
    A = weights.unsqueeze(-1) * J_field
    r = min(A.shape[-2], A.shape[-1])
    sigma_avg_sq = torch.sum(A * A, dim=(-2, -1)) / r
    sigma_avg = torch.sqrt(sigma_avg_sq + 1e-12)
    omega = fair_grad(sigma_avg, torch.tensor(1.0, device=DEVICE)) / (2.0 * sigma_avg + 1e-12)

    S = torch.as_tensor(S_field, device=DEVICE, dtype=A.dtype)
    participation = vtv._compute_directional_participation_counts(
        A.shape[:-2] + (A.shape[-1],), DEVICE, A.dtype
    )
    participation = participation.unsqueeze(-2).expand_as(S)
    S_jm = torch.sum(S * S * participation, dim=-1)
    return omega.unsqueeze(-1) * S_jm * (weights * weights)


def _expected_frobenius_surrogate_pd(vtv, J_field, S_field, weights):
    A = weights.unsqueeze(-1) * J_field
    A_frob_sq = torch.sum(A * A, dim=(-2, -1))
    A_frob = torch.sqrt(A_frob_sq + 1e-12)
    phi_prime = fair_grad(A_frob, torch.tensor(1.0, device=DEVICE))
    M = A.shape[-2]
    omega = M * phi_prime / (A_frob + 1e-12)

    S = torch.as_tensor(S_field, device=DEVICE, dtype=A.dtype)
    participation = vtv._compute_directional_participation_counts(
        A.shape[:-2] + (A.shape[-1],), DEVICE, A.dtype
    )
    participation = participation.unsqueeze(-2).expand_as(S)
    S_jm = torch.sum(S * S * participation, dim=-1)
    return omega.unsqueeze(-1) * S_jm * (weights * weights)


def _expected_vector_tv_per_modality(J_field, S_field, weights):
    A = weights.unsqueeze(-1) * J_field
    r2 = torch.sum(A * A, dim=-1)
    r = torch.sqrt(r2 + 1e-12)
    alpha = fair_grad(r, torch.tensor(1.0, device=DEVICE)) / (r + 1e-12)
    beta = fair_hessian_diag(r, torch.tensor(1.0, device=DEVICE)) - alpha
    frac = (A * A) / (r2.unsqueeze(-1) + 1e-12)
    h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac
    S = torch.as_tensor(S_field, device=DEVICE, dtype=A.dtype)
    return torch.sum((weights.unsqueeze(-1) ** 2) * (S * S) * h_dir, dim=-1)


def test_mm_jensen_matches_formula():
    torch.manual_seed(2)
    J_field = torch.randn(1, 1, 1, 2, 3, device=DEVICE)
    S_field = torch.abs(torch.randn(1, 1, 1, 2, 3, device=DEVICE)) + 0.5
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.2
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = make_vtv_harness(J_field, S_field, weights, eps=1.0)
    diag = vtv._preconditioner_weights_core_fast(x_arr, eta=1.0, epsilon=1e-12)
    expected = _expected_mm_jensen(vtv, J_field, S_field, weights)

    assert torch.allclose(diag, expected, rtol=1e-5, atol=1e-6)


def test_frobenius_surrogate_pd_matches_formula():
    torch.manual_seed(3)
    J_field = torch.randn(1, 1, 1, 2, 3, device=DEVICE)
    S_field = torch.abs(torch.randn(1, 1, 1, 2, 3, device=DEVICE)) + 0.5
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.2
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = make_vtv_harness(J_field, S_field, weights, eps=1.0)
    diag = vtv._preconditioner_weights_core_fastest_positive(x_arr, eta=1.0, epsilon=1e-12)
    expected = _expected_frobenius_surrogate_pd(vtv, J_field, S_field, weights)

    assert torch.allclose(diag, expected, rtol=1e-5, atol=1e-6)


def test_vector_tv_per_modality_matches_formula_multi_modality():
    torch.manual_seed(4)
    J_field = torch.randn(1, 1, 1, 2, 3, device=DEVICE)
    S_field = torch.abs(torch.randn(1, 1, 1, 2, 3, device=DEVICE)) + 0.1
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.3
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = make_vtv_harness(J_field, S_field, weights, eps=1.0)
    diag = vtv._preconditioner_weights_core_fastest_exact(x_arr, epsilon=1e-12)
    expected = _expected_vector_tv_per_modality(J_field, S_field, weights)

    assert torch.allclose(diag, expected, rtol=1e-5, atol=1e-6)
