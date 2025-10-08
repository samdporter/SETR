from tests._vtv_test_utils import (
    DEVICE,
    make_vtv_harness,
    require_vtv,
    torch,
)

require_vtv()

from setr.priors.vtv.common import fair_grad, fair_hessian_diag


def test_vector_tv_per_modality_matches_vector_norm_single_modality():
    """Per-modality preconditioner must collapse to vector-norm Hessian when M=1."""

    nx = ny = nz = 1
    M, d = 1, 3

    J_field = torch.tensor([[[[[1.0, 2.0, 0.5]]]]], device=DEVICE)
    S_field = torch.tensor([[[[[1.0, 2.0, 3.0]]]]], device=DEVICE)
    weights = torch.tensor([[[[2.0]]]], device=DEVICE)

    vtv = make_vtv_harness(J_field, S_field, weights, eps=1.0)

    x_arr = torch.zeros(nx, ny, nz, M, device=DEVICE)
    diag = vtv._preconditioner_weights_core_fastest_exact(x_arr, epsilon=1e-12)

    A = weights.unsqueeze(-1) * J_field
    r2 = torch.sum(A * A, dim=-1)
    r = torch.sqrt(r2 + 1e-12)

    alpha = fair_grad(r, torch.tensor(1.0, device=DEVICE)) / r
    beta = fair_hessian_diag(r, torch.tensor(1.0, device=DEVICE)) - alpha
    frac = (A * A) / (r2.unsqueeze(-1) + 1e-12)
    h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac

    expected = torch.sum((weights * weights).unsqueeze(-1) * (S_field * S_field) * h_dir, dim=-1)

    assert torch.allclose(diag, expected, rtol=1e-6, atol=1e-6)


def test_svd_principal_alpha_reduces_to_isotropic_when_grad_zero():
    """When gradients vanish, only the isotropic α term should remain."""

    nx = ny = nz = 1
    M, d = 2, 2
    eps = 1.0

    J_field = torch.zeros(nx, ny, nz, M, d, device=DEVICE)
    S_field = torch.ones(nx, ny, nz, M, d, device=DEVICE)
    weights = torch.tensor([[[[2.0, 3.0]]]], device=DEVICE)

    vtv = make_vtv_harness(J_field, S_field, weights, eps=eps)

    x_arr = torch.zeros(nx, ny, nz, M, device=DEVICE)
    diag = vtv._preconditioner_weights_core_slow(x_arr)

    alpha_total = 2.0 / eps  # two singular values each contributing 1/eps
    expected = alpha_total * (weights * weights) * torch.sum(S_field * S_field, dim=-1)

    assert torch.allclose(diag, expected, rtol=1e-6, atol=1e-6)
