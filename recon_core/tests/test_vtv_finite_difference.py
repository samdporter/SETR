import os
from typing import Callable

import numpy as np
import pytest

from tests._vtv_test_utils import require_vtv, torch

require_vtv()

from recon_core.cil_extensions.operators import NiftyResampleOperator
from recon_core.core.gradients import Jacobian
from recon_core.priors.vtv.schatten_norm_gpu_slow import (
    GPUVectorialTotalVariation as VTVSlowBackend,
)
from recon_core.priors.vtv.schatten_norm_gpu_stable import (
    GPUVectorialTotalVariation as VTVStableBackend,
)
from recon_core.priors.vtv.small_eig import eigenvalsh_2x2, eigenvecsh_2x2
from recon_core.priors.vtv.vtv import WeightedVectorialTotalVariation
from recon_core.utils.sirf import get_array, get_pet_data, get_spect_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pathological near-repeated-singular-value 2x3 case that previously broke
# the analytic/stable backend gradient in float32.
PATHOLOGICAL_2X3_X = np.array(
    [
        [-0.134422411501, 0.262926201435, -0.227354924088],
        [-0.102809514637, -0.262932396097, -0.243286564869],
    ],
    dtype=np.float64,
)
PATHOLOGICAL_2X3_H = np.array(
    [
        [1.115928762475, 1.387218666188, 0.542519363936],
        [-1.289143255613, 0.365171742364, 0.273641821727],
    ],
    dtype=np.float64,
)


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


def _directional_fd(fun: Callable[[torch.Tensor], float], x: torch.Tensor, h: torch.Tensor, eps: float) -> float:
    return float((fun(x + eps * h) - fun(x - eps * h)) / (2.0 * eps))


def _rel_err(a: float, b: float, eps: float = 1e-12) -> float:
    return abs(a - b) / max(abs(a), abs(b), eps)


def _normalised_direction(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = torch.as_tensor(rng.standard_normal(shape), device=DEVICE, dtype=torch.float64)
    nrm = torch.sqrt(torch.sum(arr * arr))
    return arr / torch.clamp(nrm, min=1e-20)


def _make_wvtv_harness(
    shape: tuple[int, int, int, int],
    voxel_sizes: tuple[float, float, float],
    backend: str,
    smoothing: str = "charbonnier",
    delta: float = 1e-3,
    stencil: str = "6",
    both_directions: bool = False,
    bnd_cond: str = "Neumann",
    anatomical: np.ndarray | None = None,
) -> WeightedVectorialTotalVariation:
    if shape[-1] != 2:
        raise ValueError("These checks are currently set up for two-modality TNV.")

    wvtv = object.__new__(WeightedVectorialTotalVariation)
    wvtv._dV = float(np.prod(voxel_sizes))
    wvtv.jacobian = Jacobian(
        voxel_sizes=voxel_sizes,
        anatomical=anatomical,
        stencil=stencil,
        both_directions=both_directions,
        max_step=1,
        bnd_cond=bnd_cond,
    )
    wvtv.bdc2a = _IdentityBDC2A()
    wvtv.smoothing = smoothing
    wvtv.precond_method = "mm_diag"

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
        eps=delta,
        norm="nuclear",
        smoothing_function=smoothing,
        numpy_out=False,
    )
    return wvtv


def _best_fd_match(
    fun: Callable[[torch.Tensor], float],
    x: torch.Tensor,
    h: torch.Tensor,
    inner: float,
    fd_steps: list[float],
) -> tuple[float, float, float]:
    best_step = None
    best_fd = None
    best_err = float("inf")
    for step in fd_steps:
        fd = _directional_fd(fun, x, h, eps=step)
        err = _rel_err(inner, fd)
        if err < best_err:
            best_err = err
            best_step = step
            best_fd = fd
    return float(best_step), float(best_fd), float(best_err)


def _simulated_inputs(
    shape: tuple[int, int, int, int],
    seed: int,
    near_rank_deficient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = 0.2 * rng.standard_normal(shape) + 0.5
    if near_rank_deficient:
        x_np[..., 1] = x_np[..., 0] + 1e-4 * rng.standard_normal(shape[:-1])
    x = torch.as_tensor(x_np, device=DEVICE, dtype=torch.float64)
    h = _normalised_direction(shape, seed=seed + 1000)
    return x, h


SIMULATED_JACOBIAN_CASES = [
    pytest.param(
        {
            "shape": (8, 7, 6, 2),
            "voxel_sizes": (2.4, 2.4, 2.8),
            "stencil": "6",
            "both_directions": False,
            "bnd_cond": "Neumann",
            "directional": False,
            "near_rank_deficient": False,
        },
        id="stencil6-neumann",
    ),
    pytest.param(
        {
            "shape": (8, 7, 6, 2),
            "voxel_sizes": (2.4, 2.4, 2.8),
            "stencil": "18",
            "both_directions": False,
            "bnd_cond": "Neumann",
            "directional": False,
            "near_rank_deficient": False,
        },
        id="stencil18-neumann",
    ),
    pytest.param(
        {
            "shape": (7, 6, 5, 2),
            "voxel_sizes": (2.0, 2.2, 2.6),
            "stencil": "6",
            "both_directions": False,
            "bnd_cond": "Periodic",
            "directional": False,
            "near_rank_deficient": False,
        },
        id="stencil6-periodic",
    ),
    pytest.param(
        {
            "shape": (6, 5, 4, 2),
            "voxel_sizes": (2.1, 2.0, 2.3),
            "stencil": "26",
            "both_directions": True,
            "bnd_cond": "Neumann",
            "directional": True,
            "near_rank_deficient": True,
        },
        id="stencil26-bidirectional-directional-rankdef",
    ),
]


def test_eigenvecsh_2x2_pathological_near_degenerate_is_orthonormal():
    x = torch.as_tensor(PATHOLOGICAL_2X3_X, device=DEVICE, dtype=torch.float32)
    h = x @ x.transpose(-1, -2)

    eigvals = eigenvalsh_2x2(h)
    eigvecs = eigenvecsh_2x2(h, eigvals)

    eye = torch.eye(2, device=DEVICE, dtype=torch.float32)
    ortho_err = float(torch.linalg.norm(eigvecs.transpose(-1, -2) @ eigvecs - eye).item())
    recon = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    denom = torch.clamp(torch.linalg.norm(h), min=1e-12)
    recon_err = float((torch.linalg.norm(h - recon) / denom).item())

    assert ortho_err < 1e-3, f"2x2 eigenvectors lost orthogonality: {ortho_err:.3e}"
    assert recon_err < 1e-3, f"2x2 eigendecomposition reconstruction error too large: {recon_err:.3e}"


@pytest.mark.parametrize("backend", ["stable", "slow"])
def test_backend_gradient_matches_fd_pathological_near_degenerate_2x3(backend: str):
    if backend == "stable":
        backend_cls = VTVStableBackend
    elif backend == "slow":
        backend_cls = VTVSlowBackend
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    vtv = backend_cls(
        eps=1e-3,
        norm="nuclear",
        smoothing_function="charbonnier",
        numpy_out=False,
    )

    x = torch.as_tensor(PATHOLOGICAL_2X3_X, device=DEVICE, dtype=torch.float32).view(1, 1, 1, 2, 3)
    h = torch.as_tensor(PATHOLOGICAL_2X3_H, device=DEVICE, dtype=torch.float32).view(1, 1, 1, 2, 3)
    h = h / torch.clamp(torch.sqrt(torch.sum(h * h)), min=1e-20)

    grad = vtv.gradient(x)
    inner = float(torch.sum(grad * h).item())
    step, fd, err = _best_fd_match(
        lambda z: float(vtv(z).item()),
        x,
        h,
        inner,
        fd_steps=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    )

    assert err < 2e-2, (
        f"{backend}: pathological near-degenerate 2x3 gradient/FD mismatch too large "
        f"(best rel err={err:.3e}, step={step:.1e}, inner={inner:.6e}, fd={fd:.6e})"
    )


@pytest.mark.parametrize("backend", ["stable", "slow"])
def test_wvtv_gradient_matches_finite_difference_simulated(backend: str):
    shape = (8, 7, 6, 2)
    voxel_sizes = (2.4, 2.4, 2.8)
    wvtv = _make_wvtv_harness(shape, voxel_sizes, backend=backend, smoothing="charbonnier", delta=1e-3)

    rng = np.random.default_rng(5)
    x = torch.as_tensor(rng.standard_normal(shape), device=DEVICE, dtype=torch.float64)
    x = 0.2 * x + 0.5
    h = _normalised_direction(shape, seed=77)

    grad = wvtv.gradient(x)
    inner = float(torch.sum(grad * h).item())
    step, fd, err = _best_fd_match(
        lambda z: float(wvtv(z).item()),
        x,
        h,
        inner,
        fd_steps=[2e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    )

    assert err < 1e-2, (
        f"{backend}: gradient/FD mismatch too large "
        f"(best rel err={err:.3e}, step={step:.1e}, inner={inner:.6e}, fd={fd:.6e})"
    )


@pytest.mark.parametrize("case", SIMULATED_JACOBIAN_CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wvtv_stable_slow_parity_simulated_jacobian_sweep(case: dict, seed: int):
    shape = case["shape"]
    anatomical = None
    if case["directional"]:
        rng = np.random.default_rng(seed + 500)
        anatomical = np.abs(rng.standard_normal(shape[:-1])).astype(np.float32)

    x, _ = _simulated_inputs(shape, seed=seed, near_rank_deficient=case["near_rank_deficient"])

    common_kwargs = dict(
        shape=shape,
        voxel_sizes=case["voxel_sizes"],
        smoothing="charbonnier",
        delta=1e-3,
        stencil=case["stencil"],
        both_directions=case["both_directions"],
        bnd_cond=case["bnd_cond"],
        anatomical=anatomical,
    )
    wvtv_slow = _make_wvtv_harness(backend="slow", **common_kwargs)
    wvtv_stable = _make_wvtv_harness(backend="stable", **common_kwargs)

    f_slow = float(wvtv_slow(x).item())
    f_stable = float(wvtv_stable(x).item())
    g_slow = wvtv_slow.gradient(x)
    g_stable = wvtv_stable.gradient(x)

    value_rel = _rel_err(f_slow, f_stable)
    grad_rel = float(torch.norm(g_slow - g_stable) / torch.clamp(torch.norm(g_slow), min=1e-12))

    assert value_rel < 1e-5, (
        f"value parity failure for case={case}, seed={seed}: "
        f"rel_err={value_rel:.3e}, slow={f_slow:.6e}, stable={f_stable:.6e}"
    )
    assert grad_rel < 5e-2, (
        f"gradient parity failure for case={case}, seed={seed}: "
        f"rel_err={grad_rel:.3e}"
    )


@pytest.mark.parametrize("case", SIMULATED_JACOBIAN_CASES)
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("backend, tol", [("stable", 8e-2), ("slow", 1e-1)])
def test_wvtv_gradient_matches_fd_simulated_jacobian_sweep(
    case: dict,
    seed: int,
    backend: str,
    tol: float,
):
    shape = case["shape"]
    anatomical = None
    if case["directional"]:
        rng = np.random.default_rng(seed + 500)
        anatomical = np.abs(rng.standard_normal(shape[:-1])).astype(np.float32)

    x, h = _simulated_inputs(shape, seed=seed, near_rank_deficient=case["near_rank_deficient"])
    wvtv = _make_wvtv_harness(
        shape=shape,
        voxel_sizes=case["voxel_sizes"],
        backend=backend,
        smoothing="charbonnier",
        delta=1e-3,
        stencil=case["stencil"],
        both_directions=case["both_directions"],
        bnd_cond=case["bnd_cond"],
        anatomical=anatomical,
    )

    grad = wvtv.gradient(x)
    inner = float(torch.sum(grad * h).item())
    step, fd, err = _best_fd_match(
        lambda z: float(wvtv(z).item()),
        x,
        h,
        inner,
        fd_steps=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    )

    assert err < tol, (
        f"{backend}: sweep gradient/FD mismatch too large for seed={seed}, case={case}. "
        f"(best rel err={err:.3e}, step={step:.1e}, inner={inner:.6e}, fd={fd:.6e})"
    )


@pytest.mark.parametrize(
    "backend, tol",
    [
        ("stable", 5e-2),
        ("slow", 2e-2),
    ],
)
def test_wvtv_gradient_matches_finite_difference_real_data(backend: str, tol: float):
    pet_path = os.getenv(
        "SETR_TEST_PET_PATH",
        "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/PET/phantom",
    )
    spect_path = os.getenv(
        "SETR_TEST_SPECT_PATH",
        "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140",
    )

    if not (os.path.isdir(pet_path) and os.path.isdir(spect_path)):
        pytest.skip(
            f"Prepared data unavailable for real-data finite-difference test. "
            f"PET={pet_path!r}, SPECT={spect_path!r}"
        )

    pet_data = get_pet_data(pet_path, load_sinos=False)
    spect_data = get_spect_data(spect_path, load_sinos=False)
    transform = spect_data.get("no_zoom_displacement")
    if transform is None:
        pytest.skip("No spect2pet no-zoom displacement field available in real-data test set.")

    spect2pet = NiftyResampleOperator(
        reference=pet_data["initial_image"],
        floating=spect_data["initial_image"],
        transform=transform,
    )
    spect_in_pet = spect2pet.direct(spect_data["initial_image"])

    pet_arr = np.array(get_array(pet_data["initial_image"]), dtype=np.float64)
    spect_arr = np.array(get_array(spect_in_pet), dtype=np.float64)

    # Use a compact central patch to keep test runtime bounded.
    cz, cy, cx = [s // 2 for s in pet_arr.shape]
    dz, dy, dx = 6, 6, 6
    sl = (
        slice(cz - dz, cz + dz),
        slice(cy - dy, cy + dy),
        slice(cx - dx, cx + dx),
    )
    pet_patch = pet_arr[sl]
    spect_patch = spect_arr[sl]

    # Normalise each modality to O(1) scale for numerically stable FD.
    pet_scale = max(np.percentile(np.abs(pet_patch), 99.0), 1e-12)
    spect_scale = max(np.percentile(np.abs(spect_patch), 99.0), 1e-12)
    pet_patch = pet_patch / pet_scale
    spect_patch = spect_patch / spect_scale

    x_np = np.stack([pet_patch, spect_patch], axis=-1)
    x = torch.as_tensor(x_np, device=DEVICE, dtype=torch.float64)
    h = _normalised_direction(x.shape, seed=101)

    voxel_sizes = tuple(float(v) for v in pet_data["initial_image"].voxel_sizes())
    wvtv = _make_wvtv_harness(
        x.shape,
        voxel_sizes,
        backend=backend,
        smoothing="charbonnier",
        delta=1e-3,
    )

    grad = wvtv.gradient(x)
    inner = float(torch.sum(grad * h).item())
    step, fd, err = _best_fd_match(
        lambda z: float(wvtv(z).item()),
        x,
        h,
        inner,
        fd_steps=[1e-3, 2e-3, 5e-3, 1e-2, 2e-2],
    )

    assert err < tol, (
        f"{backend}: real-data gradient/FD mismatch too large "
        f"(best rel err={err:.3e}, step={step:.1e}, inner={inner:.6e}, fd={fd:.6e})"
    )
