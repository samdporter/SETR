from types import SimpleNamespace

import pytest

from tests._vtv_test_utils import require_vtv, torch

require_vtv()

DEVICE = torch.device("cpu")

from recon_core.priors.vtv.vtv import WeightedVectorialTotalVariation
from recon_core.priors.vtv.preconditioners import (
    _ls_blocks_lowmem,
    _ls_hessian_matrix,
    compute_precond_block,
)
from recon_core.core.gradients import Jacobian


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
    blocks = compute_precond_block(vtv, weights, "mm_diag_block_maj", epsilon=1e-8)

    assert blocks.shape == (*weights.shape[:-1], 2, 2)
    assert torch.max(torch.abs(blocks[..., 0, 1] - blocks[..., 1, 0])) < 1e-5
    eigvals = torch.linalg.eigvalsh(blocks)
    assert torch.all(eigvals > 0)


def test_mm_block_projector_returns_symmetric_positive_blocks():
    torch.manual_seed(111)
    j_field = torch.randn(3, 2, 2, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(3, 2, 2, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(3, 2, 2, 2, device=DEVICE)) + 0.5

    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    blocks = compute_precond_block(vtv, weights, "mm_diag_block_tight", epsilon=1e-8)

    assert blocks.shape == (*weights.shape[:-1], 2, 2)
    assert torch.max(torch.abs(blocks[..., 0, 1] - blocks[..., 1, 0])) < 1e-5
    eigvals = torch.linalg.eigvalsh(blocks)
    assert torch.all(eigvals > 0)


def test_directional_participation_counts_follow_actual_stencil_order():
    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.jacobian = Jacobian(
        voxel_sizes=(1.0, 1.0, 1.0),
        stencil="6",
        both_directions=False,
        bnd_cond="Neumann",
    )

    counts = vtv._compute_directional_participation_counts(
        (3, 4, 5, 3),
        device=DEVICE,
        dtype=torch.float32,
    )

    directions = vtv.jacobian.grad.directions
    assert directions == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    # Channel 0 is z, not x. The old implementation used nx for this channel.
    assert torch.all(counts[:, :, 0, 0] == 1)
    assert torch.all(counts[:, :, -1, 0] == 1)
    assert torch.all(counts[:, :, 1:-1, 0] == 2)

    # Channel 2 is x.
    assert torch.all(counts[0, :, :, 2] == 1)
    assert torch.all(counts[-1, :, :, 2] == 1)
    assert torch.all(counts[1:-1, :, :, 2] == 2)


@pytest.mark.parametrize("method", ["mm_diag_block_maj", "mm_diag_block_tight"])
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


def test_ls_block_methods_registered_on_weighted_vectorial_total_variation():
    for method in ("ls_block_diag", "ls_block_gershgorin"):
        assert method in WeightedVectorialTotalVariation._PRECOND_BLOCK_METHODS
        assert method in WeightedVectorialTotalVariation._PRECOND_METHODS


@pytest.mark.parametrize("method", ["ls_block_diag", "ls_block_gershgorin"])
def test_ls_block_dispatch_returns_finite_blocks(method):
    torch.manual_seed(29)
    j_field = torch.randn(2, 2, 2, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(2, 2, 2, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(2, 2, 2, 2, device=DEVICE)) + 0.5

    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    blocks = compute_precond_block(vtv, weights, method, epsilon=1e-8)
    assert blocks.shape == (*weights.shape[:-1], 2, 2)
    assert torch.all(torch.isfinite(blocks))

    vtv.precond_method = method
    via_method = vtv.preconditioner_block(weights, epsilon=1e-8)
    assert torch.all(torch.isfinite(via_method))


@pytest.mark.parametrize("method", ["ls_block_diag", "ls_block_gershgorin"])
def test_ls_block_returns_symmetric_spd_blocks_with_epsilon_floor(method):
    torch.manual_seed(31)
    j_field = torch.randn(3, 2, 2, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(3, 2, 2, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(3, 2, 2, 2, device=DEVICE)) + 0.5

    epsilon = 1e-6
    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    blocks = compute_precond_block(vtv, weights, method, epsilon=epsilon)

    assert torch.max(torch.abs(blocks[..., 0, 1] - blocks[..., 1, 0])) < 1e-5
    eigvals = torch.linalg.eigvalsh(blocks)
    assert torch.all(eigvals >= epsilon - 1e-6)


@pytest.mark.parametrize("method", ["ls_block_diag", "ls_block_gershgorin"])
def test_ls_block_matches_legacy_preconditioners_old(method):
    from recon_core.priors.vtv import preconditioners_old

    torch.manual_seed(23)
    j_field = torch.randn(3, 2, 2, 2, 3, dtype=torch.float64, device=DEVICE)
    s_field = torch.abs(torch.randn(3, 2, 2, 2, 3, dtype=torch.float64, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(3, 2, 2, 2, dtype=torch.float64, device=DEVICE)) + 0.5

    vtv = _make_block_vtv(j_field, s_field, weights, eps=1.0)
    active = compute_precond_block(vtv, weights, method, epsilon=1e-8)
    legacy = preconditioners_old.compute_precond_block(vtv, weights, method, epsilon=1e-8)

    assert torch.allclose(active, legacy, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ls_hessian_matrix_matches_autograd_hessian(seed):
    torch.manual_seed(seed)
    d = 3
    eps = 1e-2
    J = torch.randn(d, 2, dtype=torch.float64, device=DEVICE)

    def rho(vec):
        Jm = vec.reshape(d, 2)
        s = torch.linalg.svdvals(Jm)
        return torch.sum(torch.sqrt(s * s + eps * eps))

    H_autograd = torch.autograd.functional.hessian(rho, J.reshape(-1))
    H_ls = _ls_hessian_matrix(J, eps)

    assert torch.allclose(H_ls, H_autograd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["ls_block_diag", "ls_block_gershgorin"])
def test_ls_block_nan_robustness(method):
    nx, ny, nz = 4, 4, 4
    M = 2
    d = 3

    x_arr = torch.rand((nx, ny, nz, M), dtype=torch.float32, device=DEVICE)
    x_arr[1, 1, 1, :] = float("nan")
    x_arr[2, 2, 2, :] = float("inf")

    class DummyJacobian:
        def __init__(self):
            self.grad = type(
                "Grad", (), {"directions": [(1, 0, 0), (0, 1, 0), (0, 0, 1)], "bnd_cond": "Neumann"}
            )()

        def direct(self, x):
            out = torch.rand((nx, ny, nz, M, d), dtype=torch.float32, device=DEVICE)
            out[1, 1, 1, :, :] = float("nan")
            out[2, 2, 2, :, :] = float("inf")
            return out

        def sensitivity(self, x):
            return torch.ones((nx, ny, nz, M, d), dtype=torch.float32, device=DEVICE)

    class DummyWVTV:
        def __init__(self):
            self.jacobian = DummyJacobian()
            self.smoothing = "charbonnier"
            self.vtv = type("VTV", (), {"eps": 1e-4})()
            self.weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEVICE)

    wvtv = DummyWVTV()
    res = compute_precond_block(wvtv, x_arr, method=method, epsilon=1e-8)

    assert res is not None
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()


def test_ls_block_diag_is_not_global_loewner_majoriser_of_full_ls_hessian():
    """
    LS block-diagonal extraction keeps only per-direction 2x2 blocks of the
    full LS Hessian, so it is generally *not* a global Loewner majoriser.
    """
    j = torch.tensor(
        [
            [0.12573022, -0.13210486],
            [0.64042265, 0.10490012],
            [-0.53566937, 0.36159505],
        ],
        dtype=torch.float64,
        device=DEVICE,
    )  # shape (d=3, modalities=2)
    eps = 1e-3

    h_full = _ls_hessian_matrix(j, eps)
    b_blocks = _ls_blocks_lowmem(j, eps)

    d = j.shape[0]
    h_block_diag = torch.zeros((2 * d, 2 * d), dtype=torch.float64, device=DEVICE)
    for q in range(d):
        h_block_diag[2 * q : 2 * q + 2, 2 * q : 2 * q + 2] = b_blocks[q]

    delta = 0.5 * ((h_block_diag - h_full) + (h_block_diag - h_full).T)
    eigvals = torch.linalg.eigvalsh(delta)

    # Negative eigenvalue => block-diagonal approximation does not majorise full Hessian.
    assert float(eigvals.min()) < -1e-2
