import pytest

from tests._vtv_test_utils import require_vtv, torch

require_vtv()

from types import SimpleNamespace

from recon_core.priors.vtv import preconditioners as vtv_precond
from recon_core.priors.vtv.preconditioners import (
    _ls_blocks_from_hessian,
    _ls_blocks_lowmem,
    _ls_hessian_matrix,
    _mm_block_weight,
    compute_precond_block,
    compute_precond_diag,
)
from recon_core.priors.vtv.vtv import WeightedVectorialTotalVariation, _accumulate_over_neighborhood_block


DEVICE = torch.device("cpu")


class _GradInfo:
    def __init__(self, directions, bnd_cond):
        self.directions = directions
        self.bnd_cond = bnd_cond


class _JacobianStub:
    def __init__(self, j_field, s_field, directions=None, bnd_cond="Neumann"):
        self._j = j_field
        self._s = s_field
        if directions is None:
            d = j_field.shape[-1]
            if d == 2:
                directions = [(1, 0, 0), (0, 1, 0)]
            elif d == 3:
                directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            else:
                directions = [(1, 0, 0)] * d
        self.grad = _GradInfo(directions=directions, bnd_cond=bnd_cond)

    def direct(self, _x):
        return self._j

    def sensitivity(self, _x):
        return self._s


class _IdentityBDC2A:
    @staticmethod
    def direct(x):
        return x

    @staticmethod
    def adjoint(x, out=None):
        if out is None:
            return x
        out.copy_(x)
        return out


def _make_vtv_stub(j_field, s_field, weights, eps=1.0, smoothing="charbonnier", bnd_cond="Neumann"):
    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.jacobian = _JacobianStub(j_field, s_field, bnd_cond=bnd_cond)
    vtv.weights = weights
    vtv.vtv = SimpleNamespace(eps=eps)
    vtv.smoothing = smoothing
    vtv._dV = 1.0
    vtv.bdc2a = _IdentityBDC2A()
    return vtv


def phi_nuclear(Y, eps):
    gram = Y.transpose(-1, -2) @ Y
    eye = torch.eye(2, device=Y.device, dtype=Y.dtype)
    eigvals = torch.linalg.eigvalsh(gram + (eps * eps) * eye)
    return torch.sum(torch.sqrt(eigvals))


def W_mm(Y0, eps):
    gram = Y0.transpose(-1, -2) @ Y0
    eye = torch.eye(2, device=Y0.device, dtype=Y0.dtype)
    eigvals, eigvecs = torch.linalg.eigh(gram + (eps * eps) * eye)
    inv_sqrt = torch.diag(torch.reciprocal(torch.sqrt(eigvals)))
    return eigvecs @ inv_sqrt @ eigvecs.transpose(-1, -2)


def mm_surrogate(Y, Y0, eps):
    W0 = W_mm(Y0, eps)
    gram0 = Y0.transpose(-1, -2) @ Y0
    gram = Y.transpose(-1, -2) @ Y
    const = phi_nuclear(Y0, eps) - 0.5 * torch.sum(W0 * gram0)
    return 0.5 * torch.sum(W0 * gram) + const


def is_psd_2x2(B, tol=1e-8):
    eigvals = torch.linalg.eigvalsh(B)
    return torch.min(eigvals).item() >= -tol


def quad_form_samples(A, B, n=32, tol=1e-8):
    for _ in range(n):
        x = torch.randn(2, device=A.device, dtype=A.dtype)
        val = x @ (A - B) @ x
        if val.item() < -tol:
            return False
    return True


def _shift_with_zeros(x, sh):
    dx, dy, dz = (int(v) for v in sh)
    nx, ny, nz = x.shape[:3]
    out = torch.zeros_like(x)

    def _slices(n, delta):
        if delta > 0:
            return slice(0, n - delta), slice(delta, n)
        if delta < 0:
            return slice(-delta, n), slice(0, n + delta)
        return slice(0, n), slice(0, n)

    xs, xd = _slices(nx, dx)
    ys, yd = _slices(ny, dy)
    zs, zd = _slices(nz, dz)
    out[xd, yd, zd, ...] = x[xs, ys, zs, ...]
    return out


def _shift_with_clamp(x, sh):
    dx, dy, dz = (int(v) for v in sh)
    nx, ny, nz = x.shape[:3]
    xs = torch.arange(nx, device=x.device) - dx
    ys = torch.arange(ny, device=x.device) - dy
    zs = torch.arange(nz, device=x.device) - dz
    xs = torch.clamp(xs, 0, nx - 1)
    ys = torch.clamp(ys, 0, ny - 1)
    zs = torch.clamp(zs, 0, nz - 1)
    return x[xs[:, None, None], ys[None, :, None], zs[None, None, :], ...]


def _ref_accumulate(term_block, directions, periodic, neumann=False):
    p = torch.zeros_like(term_block[..., 0, :, :])
    for ch, sh in enumerate(directions):
        t = term_block[..., ch, :, :]
        p = p + t
        if periodic:
            p = p + torch.roll(t, shifts=sh, dims=(0, 1, 2))
        elif neumann:
            p = p + _shift_with_clamp(t, sh)
        else:
            p = p + _shift_with_zeros(t, sh)
    return p


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 2, 2, 6),
        (2, 2, 2, 2, 3),
    ],
)
@pytest.mark.parametrize("method", ["mm_diag", "mm_diag_gershgorin", "frob_diag"])
def test_diag_preconditioners_finite_and_positive(shape, method):
    torch.manual_seed(1)
    j_field = torch.randn(*shape, device=DEVICE)
    s_field = torch.abs(torch.randn(*shape, device=DEVICE)) + 0.1
    weights = torch.abs(torch.randn(*shape[:-1], device=DEVICE)) + 0.5
    x_arr = torch.zeros(*shape[:-1], device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights)
    diag = compute_precond_diag(vtv, x_arr, method, epsilon=1e-8)

    assert torch.all(torch.isfinite(diag))
    assert torch.all(diag >= 0)


@pytest.mark.parametrize("method", ["mm_block_diag", "ls_block_diag"])
def test_block_preconditioners_symmetric_psd(method):
    torch.manual_seed(2)
    j_field = torch.randn(2, 2, 1, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(2, 2, 1, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(2, 2, 1, 2, device=DEVICE)) + 0.5
    x_arr = torch.zeros(2, 2, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights)
    blocks = compute_precond_block(vtv, x_arr, method, epsilon=1e-8)

    assert blocks.shape[-2:] == (2, 2)
    assert torch.max(torch.abs(blocks[..., 0, 1] - blocks[..., 1, 0])) < 1e-5
    eigvals = torch.linalg.eigvalsh(blocks)
    assert torch.all(eigvals >= -1e-6)


def test_mm_surrogate_majorizes_local_nuclear():
    torch.manual_seed(3)
    eps = 1.0
    Y0 = torch.randn(5, 2, device=DEVICE)

    for _ in range(8):
        H = torch.randn_like(Y0)
        for t in (0.0, 0.1, 0.2, 0.4):
            Y = Y0 + t * H
            phi_val = phi_nuclear(Y, eps)
            surrogate = mm_surrogate(Y, Y0, eps)
            assert phi_val <= surrogate + 1e-4

    W0 = W_mm(Y0, eps)
    assert torch.all(torch.isfinite(W0))
    assert is_psd_2x2(W0, tol=1e-8)


def test_mm_diag_gershgorin_loewner_majorizer():
    torch.manual_seed(4)
    eps = 1.0
    for _ in range(20):
        Y0 = torch.randn(4, 2, device=DEVICE, dtype=torch.float64)
        W0 = W_mm(Y0, eps)
        w12 = torch.abs(W0[0, 1])
        D = torch.diag(torch.stack([W0[0, 0] + w12, W0[1, 1] + w12]))
        assert is_psd_2x2(D - W0, tol=1e-10)
        assert quad_form_samples(D, W0, n=32, tol=1e-10)


def test_mm_diag_is_not_loewner_majorizer():
    torch.manual_seed(5)
    eps = 1.0
    found = False
    for _ in range(50):
        Y0 = torch.randn(4, 2, device=DEVICE)
        W0 = W_mm(Y0, eps)
        D = torch.diag(torch.stack([W0[0, 0], W0[1, 1]]))
        if not is_psd_2x2(D - W0, tol=1e-8):
            found = True
            break
    assert found, "Expected a counterexample where diag(W) - W is indefinite."


@pytest.mark.parametrize("d", [2, 3])
def test_ls_block_diag_not_majorizer(d):
    torch.manual_seed(6 + d)
    found = False
    for _ in range(40):
        J = torch.randn(2, d, 2, device=DEVICE)
        H = _ls_hessian_matrix(J, eps=1.0)
        blocks = _ls_blocks_from_hessian(H, d)
        batch = H.shape[0] if H.ndim == 3 else 1
        for b in range(batch):
            Hb = H[b] if H.ndim == 3 else H
            Bb = blocks[b] if blocks.ndim == 4 else blocks
            B = torch.zeros((2 * d, 2 * d), device=DEVICE)
            for p in range(d):
                B[2 * p : 2 * p + 2, 2 * p : 2 * p + 2] = Bb[p]
            eigvals = torch.linalg.eigvalsh(B - Hb)
            if torch.min(eigvals).item() < -1e-6:
                found = True
                break
        if found:
            break

    assert found, "Expected a counterexample where LS block diagonal does not dominate full LS Hessian."


def test_frob_diag_monotonic_scaling():
    torch.manual_seed(7)
    base = torch.randn(2, 2, 1, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(2, 2, 1, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(2, 2, 1, 2, device=DEVICE)) + 0.5
    x_arr = torch.zeros(2, 2, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(base, s_field, weights)

    diag_small = compute_precond_diag(vtv, x_arr, "frob_diag", epsilon=1e-8)
    vtv.jacobian._j = 2.0 * base
    diag_large = compute_precond_diag(vtv, x_arr, "frob_diag", epsilon=1e-8)

    assert torch.all(diag_large <= diag_small + 1e-6)


def test_edge_accumulation_matches_reference():
    torch.manual_seed(8)
    nx, ny, nz = 3, 2, 1
    directions = [(1, 0, 0), (0, 1, 0)]
    term_block = torch.randn(nx, ny, nz, len(directions), 2, 2, device=DEVICE)

    ref = _ref_accumulate(term_block, directions, periodic=False, neumann=True)
    out = _accumulate_over_neighborhood_block(term_block, directions, "Neumann")
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)

    ref_p = _ref_accumulate(term_block, directions, periodic=True)
    out_p = _accumulate_over_neighborhood_block(term_block, directions, "Periodic")
    assert torch.allclose(out_p, ref_p, atol=1e-6, rtol=1e-6)


def test_neumann_accumulation_clamps_boundaries():
    nx, ny, nz = 3, 1, 1
    directions = [(-1, 0, 0)]
    term_block = torch.zeros(nx, ny, nz, len(directions), 2, 2, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 5.0], device=DEVICE)
    for x in range(nx):
        term_block[x, 0, 0, 0, :, :] = values[x]

    out_neu = _accumulate_over_neighborhood_block(term_block, directions, "Neumann")
    ref_clamp = _ref_accumulate(term_block, directions, periodic=False, neumann=True)
    ref_zero = _ref_accumulate(term_block, directions, periodic=False, neumann=False)

    assert torch.allclose(out_neu, ref_clamp, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out_neu, ref_zero)


def test_periodic_vs_neumann_accumulation_differs():
    nx, ny, nz = 3, 1, 1
    directions = [(-1, 0, 0)]
    term_block = torch.zeros(nx, ny, nz, len(directions), 2, 2, device=DEVICE)
    values = torch.tensor([1.0, 2.0, 5.0], device=DEVICE)
    for x in range(nx):
        term_block[x, 0, 0, 0, :, :] = values[x]

    out_neu = _accumulate_over_neighborhood_block(term_block, directions, "Neumann")
    out_per = _accumulate_over_neighborhood_block(term_block, directions, "Periodic")
    assert not torch.allclose(out_neu, out_per)

def test_periodic_participation_is_uniform():
    torch.manual_seed(9)
    j_field = torch.ones(2, 2, 1, 2, 3, device=DEVICE)
    s_field = torch.ones(2, 2, 1, 2, 3, device=DEVICE)
    weights = torch.ones(2, 2, 1, 2, device=DEVICE)
    x_arr = torch.zeros(2, 2, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights, bnd_cond="Periodic")
    diag = compute_precond_diag(vtv, x_arr, "frob_diag", epsilon=1e-8)

    ref = diag[0, 0, 0]
    assert torch.allclose(diag, ref, rtol=1e-6, atol=1e-6)


def test_scaling_weights_monotone():
    torch.manual_seed(10)
    j_field = torch.randn(2, 1, 1, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(2, 1, 1, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(2, 1, 1, 2, device=DEVICE)) + 0.3
    x_arr = torch.zeros(2, 1, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights)
    diag = compute_precond_diag(vtv, x_arr, "mm_diag", epsilon=1e-8)

    scale = 3.0
    vtv.weights = weights * scale
    diag_scaled = compute_precond_diag(vtv, x_arr, "mm_diag", epsilon=1e-8)

    # Scaling weights by c should not decrease the MM diagonal (monotone in c).
    assert torch.all(diag_scaled >= diag - 1e-6)


def test_directional_projector_identity_case():
    torch.manual_seed(11)
    j_field = torch.randn(2, 2, 1, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(2, 2, 1, 2, 3, device=DEVICE)) + 0.1
    weights = torch.ones(2, 2, 1, 2, device=DEVICE)
    x_arr = torch.zeros(2, 2, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights, bnd_cond="Neumann")
    like = vtv.jacobian.direct(x_arr)
    S_jm = vtv_precond._aggregate_directional_sensitivity(vtv, x_arr, like)

    nx, ny, nz, M, d = s_field.shape
    participation = vtv._compute_directional_participation_counts((nx, ny, nz, d), DEVICE, s_field.dtype)
    participation = participation.unsqueeze(-2).expand(nx, ny, nz, M, d)
    expected = (s_field * s_field * participation).sum(dim=-1)

    assert torch.allclose(S_jm, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["mm_diag", "mm_diag_gershgorin", "frob_diag"])
def test_diag_quadratic_minimiser(method):
    torch.manual_seed(12)
    j_field = torch.randn(1, 1, 1, 2, 3, device=DEVICE)
    s_field = torch.abs(torch.randn(1, 1, 1, 2, 3, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.3
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights)
    diag = compute_precond_diag(vtv, x_arr, method, epsilon=1e-8)

    g = torch.randn_like(diag)
    x_star = g / diag
    f_star = 0.5 * torch.sum(diag * x_star * x_star) - torch.sum(g * x_star)

    for _ in range(10):
        delta = 0.05 * torch.randn_like(x_star)
        x_test = x_star + delta
        f_test = 0.5 * torch.sum(diag * x_test * x_test) - torch.sum(g * x_test)
        assert f_star <= f_test + 1e-6


@pytest.mark.parametrize("method", ["mm_block_diag", "ls_block_diag"])
def test_block_inverse_and_minimiser(method):
    torch.manual_seed(13)
    j_field = torch.randn(1, 1, 1, 2, 2, device=DEVICE)
    s_field = torch.abs(torch.randn(1, 1, 1, 2, 2, device=DEVICE)) + 0.2
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 0.3
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights)
    blocks = compute_precond_block(vtv, x_arr, method, epsilon=1e-8)

    p12 = 0.5 * (blocks[..., 0, 1] + blocks[..., 1, 0])
    p11 = blocks[..., 0, 0]
    p22 = blocks[..., 1, 1]
    det = p11 * p22 - p12 * p12
    det = torch.clamp(det, min=1e-12)
    inv = torch.stack(
        [torch.stack([p22, -p12], dim=-1), torch.stack([-p12, p11], dim=-1)], dim=-2
    ) / det.unsqueeze(-1).unsqueeze(-1)

    assert torch.max(torch.abs(inv[..., 0, 1] - inv[..., 1, 0])) < 1e-6

    ident = torch.einsum("...ij,...jk->...ik", blocks, inv)
    eye = torch.eye(2, device=DEVICE, dtype=ident.dtype)
    eye = eye.view(*((1,) * (ident.ndim - 2)), 2, 2).expand_as(ident)
    assert torch.allclose(ident, eye, rtol=1e-3, atol=1e-3)

    g = torch.randn_like(blocks[..., 0])
    x_star = torch.einsum("...ij,...j->...i", inv, g)
    f_star = 0.5 * torch.einsum("...i,...ij,...j->...", x_star, blocks, x_star) - torch.einsum(
        "...i,...i->...", g, x_star
    )

    for _ in range(10):
        delta = 0.05 * torch.randn_like(x_star)
        x_test = x_star + delta
        f_test = 0.5 * torch.einsum("...i,...ij,...j->...", x_test, blocks, x_test) - torch.einsum(
            "...i,...i->...", g, x_test
        )
        assert torch.all(f_star <= f_test + 1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mm_weight_dtype_spd(dtype):
    torch.manual_seed(14)
    Y0 = torch.randn(4, 2, device=DEVICE, dtype=dtype)
    W0 = W_mm(Y0, eps=1.0)
    assert torch.all(torch.isfinite(W0))
    assert is_psd_2x2(W0, tol=1e-10)


def test_extreme_values_are_finite():
    torch.manual_seed(15)
    j_field = torch.randn(1, 1, 1, 2, 3, device=DEVICE) * 1.0e6
    s_field = torch.abs(torch.randn(1, 1, 1, 2, 3, device=DEVICE)) + 1e-3
    weights = torch.abs(torch.randn(1, 1, 1, 2, device=DEVICE)) + 1e-3
    x_arr = torch.zeros(1, 1, 1, 2, device=DEVICE)

    vtv = _make_vtv_stub(j_field, s_field, weights)
    diag = compute_precond_diag(vtv, x_arr, "frob_diag", epsilon=1e-8)
    blocks = compute_precond_block(vtv, x_arr, "mm_block_diag", epsilon=1e-8)

    assert torch.all(torch.isfinite(diag))
    assert torch.all(torch.isfinite(blocks))


@pytest.mark.parametrize("d", [2, 3])
def test_ls_lowmem_matches_full(d):
    torch.manual_seed(40 + d)
    J = torch.randn(2, d, 2, device=DEVICE)
    H = _ls_hessian_matrix(J, eps=1.0)
    blocks_full = _ls_blocks_from_hessian(H, d)
    blocks_lowmem = _ls_blocks_lowmem(J, eps=1.0)

    assert torch.allclose(blocks_lowmem, blocks_full, rtol=1e-5, atol=1e-6)


def test_mm_block_weight_matches_helper():
    torch.manual_seed(16)
    A = torch.randn(2, 3, device=DEVICE)
    W0 = _mm_block_weight(A, "charbonnier", 1.0)
    W1 = W_mm(A.transpose(-1, -2), eps=1.0)
    assert torch.allclose(W0, W1, rtol=1e-5, atol=1e-6)
