import numpy as np
import pytest

from tests._vtv_test_utils import require_vtv, torch

require_vtv()

from recon_core.core.gradients import Jacobian
from recon_core.priors.vtv.preconditioners import compute_precond_block, compute_precond_diag
from recon_core.priors.vtv.schatten_norm_gpu_slow import (
    GPUVectorialTotalVariation as VTVSlowBackend,
)
from recon_core.priors.vtv.schatten_norm_gpu_stable import (
    GPUVectorialTotalVariation as VTVStableBackend,
)
from recon_core.priors.vtv.vtv import WeightedVectorialTotalVariation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _IdentityBDC2A:
    @staticmethod
    def direct(x):
        return x

    @staticmethod
    def adjoint(arr, out=None):
        if out is None:
            return arr
        out.copy_(arr)
        return out


def _normalised_direction(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = torch.as_tensor(rng.standard_normal(shape), device=DEVICE, dtype=torch.float64)
    nrm = torch.sqrt(torch.sum(arr * arr))
    return arr / torch.clamp(nrm, min=1e-20)


def _directional_fd(fun, x: torch.Tensor, h: torch.Tensor, eps: float) -> float:
    return float((fun(x + eps * h) - fun(x - eps * h)) / (2.0 * eps))


def _rel_err(a: float, b: float, eps: float = 1e-12) -> float:
    return abs(a - b) / max(abs(a), abs(b), eps)


def _make_wvtv_harness(
    shape: tuple[int, int, int, int],
    voxel_sizes: tuple[float, float, float],
    backend: str,
    directional: bool,
    bnd_cond: str = "Neumann",
    stencil: str = "6",
) -> WeightedVectorialTotalVariation:
    if shape[-1] != 2:
        raise ValueError("Edge-case harness is defined for two modalities only.")

    anatomical = None
    if directional:
        rng = np.random.default_rng(909)
        anatomical = np.abs(rng.standard_normal(shape[:-1])).astype(np.float32)

    wvtv = object.__new__(WeightedVectorialTotalVariation)
    wvtv._dV = float(np.prod(voxel_sizes))
    wvtv.jacobian = Jacobian(
        voxel_sizes=voxel_sizes,
        anatomical=anatomical,
        stencil=stencil,
        both_directions=False,
        max_step=1,
        bnd_cond=bnd_cond,
    )
    wvtv.bdc2a = _IdentityBDC2A()
    wvtv.smoothing = "charbonnier"
    wvtv.precond_method = "mm_diag_gershgorin_maj"

    weights = torch.ones(shape, device=DEVICE, dtype=torch.float64)
    weights[..., 0] *= 1.3
    weights[..., 1] *= 0.9
    wvtv.weights = weights

    if backend == "stable":
        backend_cls = VTVStableBackend
    elif backend == "slow":
        backend_cls = VTVSlowBackend
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    wvtv.vtv = backend_cls(
        eps=1e-3,
        norm="nuclear",
        smoothing_function="charbonnier",
        numpy_out=False,
    )
    return wvtv


def _salt_pepper_field(
    shape: tuple[int, int, int],
    salt_prob: float,
    pepper_prob: float,
    salt_val: float,
    pepper_val: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float64)
    salt = rng.random(shape) < salt_prob
    pepper = rng.random(shape) < pepper_prob
    field[salt] = salt_val
    field[pepper] = pepper_val
    return field


def _make_edge_case_input(
    case: str,
    shape: tuple[int, int, int, int] = (8, 7, 6, 2),
    seed: int = 11,
) -> torch.Tensor:
    if shape[-1] != 2:
        raise ValueError("Edge-case inputs require shape (..., 2).")

    x = np.zeros(shape, dtype=np.float64)
    spatial = shape[:-1]
    rng = np.random.default_rng(seed)

    if case == "zeros_vs_zeros":
        pass
    elif case == "salt_pepper_vs_zeros":
        x[..., 0] = _salt_pepper_field(
            spatial,
            salt_prob=0.04,
            pepper_prob=0.04,
            salt_val=8.0,
            pepper_val=-8.0,
            seed=seed,
        )
    elif case == "impulse_train_vs_zeros":
        impulses = np.zeros(spatial, dtype=np.float64)
        n_impulses = max(4, int(0.02 * np.prod(spatial)))
        flat_idx = rng.choice(np.prod(spatial), size=n_impulses, replace=False)
        impulses.reshape(-1)[flat_idx] = rng.choice([-20.0, 20.0], size=n_impulses)
        x[..., 0] = impulses
    elif case == "checkerboard_vs_tiny_constant":
        grid = np.indices(spatial).sum(axis=0)
        x[..., 0] = np.where((grid % 2) == 0, 1.0, -1.0)
        x[..., 1] = 1.0e-6
    elif case == "high_dynamic_range_mismatch":
        x[..., 0] = rng.uniform(0.0, 1.0e3, size=spatial)
        spikes = rng.random(spatial) < 0.01
        x[..., 0][spikes] = 1.0e4
        x[..., 1] = rng.normal(loc=0.0, scale=1.0e-4, size=spatial)
    else:
        raise ValueError(f"Unknown edge case: {case}")

    return torch.as_tensor(x, device=DEVICE, dtype=torch.float64)


EDGE_CASES = [
    "zeros_vs_zeros",
    "salt_pepper_vs_zeros",
    "impulse_train_vs_zeros",
    "checkerboard_vs_tiny_constant",
    "high_dynamic_range_mismatch",
]


@pytest.mark.parametrize("backend", ["stable", "slow"])
@pytest.mark.parametrize("directional", [False, True])
@pytest.mark.parametrize(
    "case",
    ["zeros_vs_zeros", "salt_pepper_vs_zeros", "high_dynamic_range_mismatch"],
)
def test_wvtv_value_gradient_are_finite_for_pathological_inputs(
    backend: str,
    directional: bool,
    case: str,
):
    shape = (8, 7, 6, 2)
    x = _make_edge_case_input(case, shape=shape, seed=21)
    wvtv = _make_wvtv_harness(
        shape=shape,
        voxel_sizes=(2.4, 2.4, 2.8),
        backend=backend,
        directional=directional,
        bnd_cond="Neumann",
        stencil="6",
    )

    val = float(wvtv(x).item())
    grad = wvtv.gradient(x)
    grad_norm = float(torch.sqrt(torch.sum(grad * grad)).item())

    assert np.isfinite(val), f"{backend}/{case}/directional={directional}: non-finite value"
    assert torch.all(torch.isfinite(grad)), (
        f"{backend}/{case}/directional={directional}: non-finite gradient entries"
    )
    assert grad_norm < 1e12, (
        f"{backend}/{case}/directional={directional}: suspiciously large gradient norm {grad_norm:.3e}"
    )

    if case == "zeros_vs_zeros":
        assert abs(val) < 1e-10, f"Zero input should have near-zero value, got {val:.3e}"
        assert grad_norm < 1e-10, f"Zero input should have near-zero gradient, got {grad_norm:.3e}"


@pytest.mark.parametrize("case", EDGE_CASES)
def test_wvtv_stable_slow_parity_on_pathological_inputs(case: str):
    shape = (8, 7, 6, 2)
    x = _make_edge_case_input(case, shape=shape, seed=22)
    w_stable = _make_wvtv_harness(
        shape=shape,
        voxel_sizes=(2.4, 2.4, 2.8),
        backend="stable",
        directional=True,
        bnd_cond="Neumann",
        stencil="6",
    )
    w_slow = _make_wvtv_harness(
        shape=shape,
        voxel_sizes=(2.4, 2.4, 2.8),
        backend="slow",
        directional=True,
        bnd_cond="Neumann",
        stencil="6",
    )

    f_stable = float(w_stable(x).item())
    f_slow = float(w_slow(x).item())
    g_stable = w_stable.gradient(x)
    g_slow = w_slow.gradient(x)

    value_rel = _rel_err(f_stable, f_slow)
    grad_rel = float(torch.norm(g_stable - g_slow) / torch.clamp(torch.norm(g_slow), min=1e-12))

    assert value_rel < 1e-3, (
        f"{case}: stable/slow value mismatch too large (rel={value_rel:.3e}, "
        f"stable={f_stable:.6e}, slow={f_slow:.6e})"
    )
    assert grad_rel < 8e-2, f"{case}: stable/slow gradient mismatch too large (rel={grad_rel:.3e})"


@pytest.mark.parametrize("backend, tol", [("stable", 8e-2), ("slow", 1e-1)])
def test_wvtv_fd_agreement_salt_pepper_vs_zero_modality(backend: str, tol: float):
    shape = (8, 7, 6, 2)
    x = _make_edge_case_input("salt_pepper_vs_zeros", shape=shape, seed=23)
    h = _normalised_direction(shape, seed=502)
    wvtv = _make_wvtv_harness(
        shape=shape,
        voxel_sizes=(2.4, 2.4, 2.8),
        backend=backend,
        directional=True,
        bnd_cond="Neumann",
        stencil="6",
    )

    grad = wvtv.gradient(x)
    inner = float(torch.sum(grad * h).item())

    best_err = float("inf")
    best_step = None
    best_fd = None
    for step in [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
        fd = _directional_fd(lambda z: float(wvtv(z).item()), x, h, step)
        err = _rel_err(inner, fd)
        if err < best_err:
            best_err = err
            best_step = step
            best_fd = fd

    assert best_err < tol, (
        f"{backend}: salt-pepper/zero edge case gradient-FD mismatch too large "
        f"(best rel err={best_err:.3e}, step={best_step:.1e}, inner={inner:.6e}, fd={best_fd:.6e})"
    )


@pytest.mark.parametrize(
    "method",
    [
        "mm_diag_tight",
        "mm_diag_gershgorin_maj",
        "mm_diag_block_maj",
        "mm_diag_block_tight",
    ],
)
@pytest.mark.parametrize(
    "case",
    ["zeros_vs_zeros", "salt_pepper_vs_zeros", "high_dynamic_range_mismatch"],
)
def test_preconditioners_are_finite_on_pathological_inputs(method: str, case: str):
    shape = (6, 5, 4, 2)
    x = _make_edge_case_input(case, shape=shape, seed=24)
    wvtv = _make_wvtv_harness(
        shape=shape,
        voxel_sizes=(2.2, 2.1, 2.6),
        backend="stable",
        directional=True,
        bnd_cond="Neumann",
        stencil="6",
    )
    x_arr = wvtv.bdc2a.direct(x)

    if method in {"mm_diag_tight", "mm_diag_gershgorin_maj"}:
        diag = compute_precond_diag(wvtv, x_arr, method, epsilon=1e-8)
        assert torch.all(torch.isfinite(diag)), f"{method}/{case}: non-finite diagonal preconditioner"
        assert torch.all(diag >= 0), f"{method}/{case}: diagonal preconditioner has negative entries"
    else:
        blocks = compute_precond_block(wvtv, x_arr, method, epsilon=1e-8)
        assert torch.all(torch.isfinite(blocks)), f"{method}/{case}: non-finite block preconditioner"
        assert blocks.shape[-2:] == (2, 2), f"{method}/{case}: unexpected block shape {blocks.shape}"
        eigvals = torch.linalg.eigvalsh(blocks)
        assert torch.all(eigvals >= -1e-6), (
            f"{method}/{case}: block preconditioner not PSD (min eig={torch.min(eigvals).item():.3e})"
        )
