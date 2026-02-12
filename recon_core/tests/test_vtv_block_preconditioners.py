from types import SimpleNamespace

import pytest

from tests._vtv_test_utils import require_vtv, torch

require_vtv()

DEVICE = torch.device("cpu")

from recon_core.priors.vtv.vtv import WeightedVectorialTotalVariation
from recon_core.priors.vtv.preconditioners import compute_precond_block


class _GradInfo:
    def __init__(self, directions, bnd_cond):
        self.directions = directions
        self.bnd_cond = bnd_cond


class _JacobianStub:
    def __init__(self, j_field, s_field, directions=None, bnd_cond="Periodic"):
        self._j = j_field
        self._s = s_field
        if directions is None:
            d = j_field.shape[-1]
            if d == 2:
                directions = [(1, 0, 0), (0, 1, 0)]
            else:
                directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self.grad = _GradInfo(
            directions=directions,
            bnd_cond=bnd_cond,
        )

    def direct(self, _x):
        return self._j

    def sensitivity(self, _x):
        return self._s


class _IdentityBDC2A:
    @staticmethod
    def direct(x):
        return x


def _make_block_vtv(j_field, s_field, weights, eps=1.0):
    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.jacobian = _JacobianStub(j_field, s_field)
    vtv.weights = weights
    vtv.vtv = SimpleNamespace(eps=eps)
    vtv.smoothing = "charbonnier"
    vtv._dV = 1.0
    vtv.bdc2a = _IdentityBDC2A()
    return vtv


def test_mm_block_returns_symmetric_positive_blocks():
    torch.manual_seed(11)
    j_field = torch.randn(3, 2, 2, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(3, 2, 2, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(3, 2, 2, 2, device=DEVICE)) + 0.5

    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    blocks = compute_precond_block(vtv, weights, "mm_block_diag", epsilon=1e-8)

    assert blocks.shape == (*weights.shape[:-1], 2, 2)
    assert torch.max(torch.abs(blocks[..., 0, 1] - blocks[..., 1, 0])) < 1e-5
    eigvals = torch.linalg.eigvalsh(blocks)
    assert torch.all(eigvals > 0)


def test_ls_block_diag_returns_symmetric_positive_blocks():
    torch.manual_seed(12)
    j_field = torch.randn(1, 1, 1, 2, 2, device=DEVICE)
    s_field = torch.abs(torch.randn(1, 1, 1, 2, 2, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.3

    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    blocks = compute_precond_block(vtv, weights, "ls_block_diag", epsilon=1e-8)

    assert blocks.shape == (*weights.shape[:-1], 2, 2)
    assert torch.max(torch.abs(blocks[..., 0, 1] - blocks[..., 1, 0])) < 1e-5
    eigvals = torch.linalg.eigvalsh(blocks)
    assert torch.all(eigvals > 0)


@pytest.mark.parametrize("method", ["mm_block_diag", "ls_block_diag"])
def test_block_inverse_matches_identity(method):
    torch.manual_seed(13)
    j_field = torch.randn(1, 1, 1, 2, 2, device=DEVICE)
    s_field = torch.abs(torch.randn(1, 1, 1, 2, 2, device=DEVICE)) + 0.3
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.3

    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    vtv.precond_method = method
    h = vtv.preconditioner_block(weights, epsilon=1e-8)
    h_inv = vtv.inv_preconditioner_block(weights, epsilon=1e-8)

    ident = torch.einsum("...ij,...jk->...ik", h, h_inv)
    target = torch.eye(2, device=DEVICE, dtype=ident.dtype)
    target = target.view(*((1,) * (ident.ndim - 2)), 2, 2).expand_as(ident)
    assert torch.allclose(ident, target, atol=1e-3, rtol=1e-3)
