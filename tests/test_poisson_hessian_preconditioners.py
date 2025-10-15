import numpy as np
import pytest
from types import SimpleNamespace

try:
    from sirf.STIR import ImageData
except (OSError, ImportError) as exc:  # pragma: no cover - SIRF/STIR not available
    pytest.skip(f"SIRF/STIR ImageData unavailable: {exc}", allow_module_level=True)

from setr.cil_extensions.preconditioners import (
    PoissonHessianPreconditioner,
    SubsetPoissonHessianPreconditioner,
)
from setr.utils.sirf import get_array


def _to_imagedata(array: np.ndarray) -> ImageData:
    """Allocate a STIR ImageData populated with the provided 3D array."""
    if array.ndim != 3:
        raise ValueError("STIR ImageData initialisation expects a (z, y, x) array.")
    container = ImageData()
    container.initialise(dim=tuple(int(d) for d in array.shape))
    container.fill(array.astype(np.float32, copy=False))
    return container


class DummyPoissonObjective:
    """Minimal objective exposing multiply_with_Hessian."""

    def __init__(self, diag: np.ndarray):
        self.diag = np.array(diag, dtype=np.float32, copy=True)

    def multiply_with_Hessian(self, image, vector):
        out = image.get_uniform_copy(0)
        out.fill(self.diag)
        return out


def test_poisson_hessian_preconditioner_basic():
    shape = (1, 2, 3)
    base = np.ones(shape, dtype=np.float32)
    image = _to_imagedata(base)

    objectives = [
        DummyPoissonObjective(np.full(shape, 2.0, dtype=np.float32)),
        DummyPoissonObjective(np.full(shape, 3.0, dtype=np.float32)),
    ]

    precond = PoissonHessianPreconditioner(objectives, hessian_floor=1e-9)
    algo = SimpleNamespace(solution=image, iteration=0)

    result = precond.compute_preconditioner(algo)
    arr = get_array(result).copy()
    assert np.allclose(arr, 1.0 / 5.0)


def test_poisson_hessian_preconditioner_floor_and_clamp():
    shape = (1, 2, 2)
    image = _to_imagedata(np.ones(shape, dtype=np.float32))
    objectives = [DummyPoissonObjective(np.full(shape, 0.0, dtype=np.float32))]

    precond = PoissonHessianPreconditioner(
        objectives, hessian_floor=0.5, epsilon=0.1, max_value=1.5
    )
    algo = SimpleNamespace(solution=image, iteration=0)

    arr = get_array(precond.compute_preconditioner(algo)).copy()
    assert np.allclose(arr, 1.5)


def test_subset_poisson_hessian_preconditioner_mean_scaling():
    shape = (1, 2, 2)
    image = _to_imagedata(np.ones(shape, dtype=np.float32))
    objectives = [
        DummyPoissonObjective(np.full(shape, 2.0, dtype=np.float32)),
        DummyPoissonObjective(np.full(shape, 4.0, dtype=np.float32)),
    ]

    subset_precond = SubsetPoissonHessianPreconditioner(
        objectives, mode="mean", scale_to_full=True, hessian_floor=1e-9
    )

    algo = SimpleNamespace(
        solution=image,
        iteration=0,
        f=SimpleNamespace(data_passes_indices=[(0,)]),
    )

    # First subset: estimate should scale single subset contribution to full data.
    arr0 = get_array(subset_precond.compute_preconditioner(algo)).copy()
    assert np.allclose(arr0, 1.0 / (2.0 * 2.0))  # approximated with doubling

    # Second subset: full aggregate should reproduce inverse of total Hessian diagonal.
    algo.f.data_passes_indices = [(1,)]
    arr1 = get_array(subset_precond.compute_preconditioner(algo)).copy()
    assert np.allclose(arr1, 1.0 / (2.0 + 4.0))
