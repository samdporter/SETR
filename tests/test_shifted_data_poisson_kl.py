import numpy as np
import pytest

try:
    from sirf.STIR import ImageData
except (OSError, ImportError) as exc:  # pragma: no cover - SIRF/STIR not available
    pytest.skip(f"SIRF/STIR ImageData unavailable: {exc}", allow_module_level=True)

from setr.cil_extensions.functions.shifted_data_poisson_KL import (
    ShiftedDataShiftedPoissonKullbackLeibler,
)

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except Exception:  # pragma: no cover - defensive
    HAS_NUMBA = False

BACKENDS = ["numpy"] + (["numba"] if HAS_NUMBA else [])


def _to_imagedata(array: np.ndarray) -> ImageData:
    if array.ndim != 3:
        raise ValueError("STIR ImageData initialisation expects a (z, y, x) array.")
    container = ImageData()
    container.initialise(dim=tuple(int(d) for d in array.shape))
    container.fill(array.astype(np.float32, copy=False))
    return container


def _directional_derivative(
    func,
    x_dc: ImageData,
    direction: np.ndarray,
    eps_sequence=(1e-3, 5e-4, 1e-4),
):
    if direction.ndim != 3:
        raise ValueError("direction must have shape (z, y, x).")

    base = x_dc.as_array().astype(np.float32, copy=True)
    direction = direction.astype(np.float32, copy=False)

    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("direction must be non-zero.")
    direction /= norm

    grad = func.gradient(x_dc).as_array().astype(np.float64, copy=False)
    inner = float(np.sum(grad * direction.astype(np.float64, copy=False)))

    best_fd = None
    best_gap = None
    for eps in eps_sequence:
        x_plus = x_dc.copy()
        x_minus = x_dc.copy()
        x_plus.fill((base + eps * direction).astype(np.float32, copy=False))
        x_minus.fill((base - eps * direction).astype(np.float32, copy=False))

        val_plus = func(x_plus)
        val_minus = func(x_minus)
        fd = (val_plus - val_minus) / (2.0 * eps)

        gap = abs(fd - inner)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_fd = fd

    return best_fd, inner


@pytest.mark.parametrize("backend", BACKENDS)
def test_shifted_poisson_gradient_matches_fd(backend):
    rng = np.random.default_rng(42)
    shape = (1, 4, 5)

    f_arr = rng.uniform(-3.0, 6.0, size=shape)
    additive_arr = rng.uniform(0.5, 1.5, size=shape)
    shift_arr = np.maximum(1.0, -f_arr + 0.5)  # ensure f + shift > 0
    x_arr = rng.uniform(0.8, 1.2, size=shape)

    f_dc = _to_imagedata(f_arr)
    additive_dc = _to_imagedata(additive_arr)
    shift_dc = _to_imagedata(shift_arr)
    x_dc = _to_imagedata(x_arr)

    sdsp = ShiftedDataShiftedPoissonKullbackLeibler(
        f=f_dc,
        additive=additive_dc,
        shift=shift_dc,
        backend=backend,
    )

    direction = rng.normal(size=shape)
    fd, inner = _directional_derivative(sdsp, x_dc, direction)
    assert fd == pytest.approx(inner, rel=5e-4, abs=5e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_shifted_poisson_value_minimiser_zero(backend):
    rng = np.random.default_rng(1337)
    shape = (1, 3, 4)

    f_arr = rng.uniform(-5.0, 8.0, size=shape)
    additive_arr = rng.uniform(0.2, 1.0, size=shape)
    shift_arr = np.maximum(2.0, -f_arr + 0.25)
    mu_arr = f_arr
    x_arr = mu_arr - additive_arr

    f_dc = _to_imagedata(f_arr)
    additive_dc = _to_imagedata(additive_arr)
    shift_dc = _to_imagedata(shift_arr)
    x_dc = _to_imagedata(x_arr)

    sdsp = ShiftedDataShiftedPoissonKullbackLeibler(
        f=f_dc,
        additive=additive_dc,
        shift=shift_dc,
        backend=backend,
    )

    value = sdsp(x_dc)
    assert value == pytest.approx(0.0, abs=5e-6)
    grad = sdsp.gradient(x_dc).as_array()
    assert np.allclose(grad, 0.0, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_shifted_poisson_domain_violation(backend):
    shape = (1, 2, 3)
    f_arr = np.zeros(shape, dtype=np.float64)
    additive_arr = np.zeros(shape, dtype=np.float64)
    shift_arr = np.ones(shape, dtype=np.float64) * 0.5

    f_dc = _to_imagedata(f_arr)
    additive_dc = _to_imagedata(additive_arr)
    shift_dc = _to_imagedata(shift_arr)
    x_dc = _to_imagedata(-np.ones(shape, dtype=np.float64))

    sdsp = ShiftedDataShiftedPoissonKullbackLeibler(
        f=f_dc,
        additive=additive_dc,
        shift=shift_dc,
        backend=backend,
    )

    assert np.isinf(sdsp(x_dc))
    with pytest.raises(ValueError):
        sdsp.gradient(x_dc)

