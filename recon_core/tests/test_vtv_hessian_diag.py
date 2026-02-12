import pytest

from tests._vtv_test_utils import require_vtv, torch

require_vtv()

DEVICE = torch.device("cpu")

from recon_core.priors.vtv.preconditioners import (
    _ls_blocks_from_hessian,
    _ls_blocks_lowmem,
    _ls_hessian_matrix,
)


def test_mm_diag_gershgorin_row_sum_majorises():
    torch.manual_seed(21)
    A = torch.randn(10, 2, 2, device=DEVICE, dtype=torch.float64)
    W = A @ A.transpose(-1, -2) + 0.1 * torch.eye(2, device=DEVICE, dtype=torch.float64)

    w12 = torch.abs(W[..., 0, 1])
    D = torch.zeros_like(W)
    D[..., 0, 0] = W[..., 0, 0] + w12
    D[..., 1, 1] = W[..., 1, 1] + w12

    eigvals = torch.linalg.eigvalsh(D - W)
    assert torch.all(eigvals >= -1e-10)


@pytest.mark.parametrize("d", [2, 3])
def test_ls_lowmem_matches_full(d):
    torch.manual_seed(40 + d)
    J = torch.randn(2, d, 2, device=DEVICE)
    H = _ls_hessian_matrix(J, eps=1.0)
    blocks_full = _ls_blocks_from_hessian(H, d)
    blocks_lowmem = _ls_blocks_lowmem(J, eps=1.0)

    assert torch.allclose(blocks_lowmem, blocks_full, rtol=1e-5, atol=1e-6)
