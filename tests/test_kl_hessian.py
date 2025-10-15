import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from setr.cil_extensions.functions import ensure_kl_hessian_support
    from setr.cil_extensions.functions.residual_aware_KL import ResidualAwareKullbackLeibler
    from cil.optimisation.functions.KullbackLeibler import KullbackLeibler
    from cil.optimisation.functions import OperatorCompositionFunction
except (ImportError, OSError) as exc:  # pragma: no cover - external dependency
    pytest.skip(f"CIL dependencies unavailable: {exc}", allow_module_level=True)


class ArrayContainer:
    """Light-weight array-backed container mimicking essential DataContainer API."""

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float64, copy=True)

    def as_array(self):
        return self._data

    def fill(self, data):
        self._data[...] = np.array(data, dtype=self._data.dtype, copy=False)

    def copy(self):
        return ArrayContainer(self._data.copy())

    def get_uniform_copy(self, value):
        return ArrayContainer(np.full_like(self._data, value, dtype=self._data.dtype))

    def __mul__(self, other):
        return ArrayContainer(self._data * other)

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, ArrayContainer):
            return ArrayContainer(self._data + other._data)
        return ArrayContainer(self._data + other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, ArrayContainer):
            return ArrayContainer(self._data - other._data)
        return ArrayContainer(self._data - other)

    def maximum(self, other):
        arr = other._data if isinstance(other, ArrayContainer) else other
        return ArrayContainer(np.maximum(self._data, arr))

    def minimum(self, other):
        arr = other._data if isinstance(other, ArrayContainer) else other
        return ArrayContainer(np.minimum(self._data, arr))


class ArrayGeometry:
    def __init__(self, length):
        self.length = length

    def allocate(self, value=0.0):
        return ArrayContainer(np.full(self.length, value, dtype=np.float64))


class DenseArrayOperator:
    """Simple dense operator acting on ArrayContainer instances."""

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)
        m, n = self.matrix.shape
        self._domain = ArrayGeometry(n)
        self._range = ArrayGeometry(m)

    def range_geometry(self):
        return self._range

    def domain_geometry(self):
        return self._domain

    def direct(self, x, out=None):
        vec = self.matrix @ x.as_array().reshape(-1)
        if out is None:
            out = self._range.allocate()
        out.fill(vec.reshape(out.as_array().shape))
        return out

    def adjoint(self, y, out=None):
        vec = self.matrix.T @ y.as_array().reshape(-1)
        if out is None:
            out = self._domain.allocate()
        out.fill(vec.reshape(out.as_array().shape))
        return out


def test_kl_multiply_with_hessian_matches_diagonal():
    ensure_kl_hessian_support()

    b = ArrayContainer([5.0, 8.0])
    eta = ArrayContainer([0.5, 0.25])
    kl = KullbackLeibler(b=b, eta=eta, backend="numpy")

    x = ArrayContainer([3.0, 4.0])
    v = ArrayContainer([0.2, 0.5])

    result = kl.multiply_with_Hessian(x, v)
    result_arr = result.as_array()

    diag = b.as_array() / (x.as_array() + eta.as_array()) ** 2
    expected = v.as_array() * diag

    assert np.allclose(result_arr, expected)


def test_operator_composition_hessian():
    ensure_kl_hessian_support()

    A = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
    operator = DenseArrayOperator(A)

    b = ArrayContainer([10.0, 6.0])
    eta = ArrayContainer([0.0, 0.0])
    kl = KullbackLeibler(b=b, eta=eta, backend="numpy")

    composed = OperatorCompositionFunction(kl, operator)

    x = ArrayContainer([1.0, 1.5])
    direction = ArrayContainer([0.3, -0.2])

    result = composed.multiply_with_Hessian(x, direction)

    Ax = A @ x.as_array()
    weights = b.as_array() / (Ax + eta.as_array()) ** 2
    Av = A @ direction.as_array()
    expected = A.T @ (weights * Av)

    assert np.allclose(result.as_array(), expected)


def test_residual_aware_hessian_matches_formula():
    f = ArrayContainer([12.0, 4.0])
    additive = ArrayContainer([1.0, 0.5])
    residual = ArrayContainer([0.2, -0.1])
    x = ArrayContainer([2.5, 3.5])
    direction = ArrayContainer([1.0, -0.3])

    rckl = ResidualAwareKullbackLeibler(
        f=f, additive=additive, residual=residual, backend="numpy"
    )

    result = rckl.multiply_with_Hessian(x, direction)

    _, mu_eff = rckl._mu_raw_and_mu_eff(x)
    mask, ok = rckl._mask_ok(mu_eff)
    assert ok
    weights = np.zeros_like(mu_eff)
    weights[mask] = rckl._f_eff_np[mask] / (mu_eff[mask] ** 2)

    expected = direction.as_array() * weights
    assert np.allclose(result.as_array(), expected)
