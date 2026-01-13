import numpy as np
import pytest

try:
    from sirf.STIR import ImageData
except (OSError, ImportError) as exc:  # pragma: no cover - SIRF/STIR not available
    pytest.skip(f"SIRF/STIR ImageData unavailable: {exc}", allow_module_level=True)

from recon_core.cil_extensions.functions.residual_aware_KL import ResidualAwareKullbackLeibler

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except Exception:  # pragma: no cover - defensive fallback
    HAS_NUMBA = False

BACKENDS = ["numpy"] + (["numba"] if HAS_NUMBA else [])


def _to_imagedata(array: np.ndarray) -> ImageData:
    """Allocate a STIR ImageData populated with the provided 3D array."""
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
    """Finite-difference directional derivative using central differences."""
    if direction.ndim != 3:
        raise ValueError("direction must have shape (z, y, x).")

    # Ensure float32 consistency with STIR storage.
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
def test_residual_corrected_kl_gradient_zero_residual(backend):
    rng = np.random.default_rng(123)
    shape = (1, 4, 5)  # (z, y, x)

    f_arr = rng.uniform(5.0, 10.0, size=shape)
    additive_arr = rng.uniform(0.5, 1.5, size=shape)
    x_arr = rng.uniform(1.0, 2.0, size=shape)
    residual_arr = np.zeros(shape, dtype=np.float64)

    f_dc = _to_imagedata(f_arr)
    additive_dc = _to_imagedata(additive_arr)
    residual_dc = _to_imagedata(residual_arr)
    x_dc = _to_imagedata(x_arr)

    rckl = ResidualAwareKullbackLeibler(
        f=f_dc,
        additive=additive_dc,
        residual=residual_dc,
        backend=backend,
    )

    direction = rng.normal(size=shape)

    fd, inner = _directional_derivative(rckl, x_dc, direction)
    # Float32 storage introduces ~1e-4 relative noise in the finite-difference check.
    assert fd == pytest.approx(inner, rel=5e-4, abs=5e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_corrected_kl_gradient_signed_residual(backend):
    rng = np.random.default_rng(321)
    shape = (1, 3, 4)

    f_arr = rng.uniform(6.0, 12.0, size=shape)
    additive_arr = rng.uniform(1.0, 2.0, size=shape)
    x_arr = rng.uniform(2.0, 3.0, size=shape)
    residual_arr = rng.uniform(-0.3, 0.3, size=shape)

    f_dc = _to_imagedata(f_arr)
    additive_dc = _to_imagedata(additive_arr)
    residual_dc = _to_imagedata(residual_arr)
    x_dc = _to_imagedata(x_arr)

    rckl = ResidualAwareKullbackLeibler(
        f=f_dc,
        additive=additive_dc,
        residual=residual_dc,
        backend=backend,
    )

    direction = rng.normal(size=shape)

    fd, inner = _directional_derivative(rckl, x_dc, direction)
    assert fd == pytest.approx(inner, rel=5e-4, abs=5e-6)
