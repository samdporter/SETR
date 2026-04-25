import math

import numpy as np

from recon_core.cil_extensions.functions import BlockIndicatorBox


class _FakeContainer:
    def __init__(self, array):
        self._arr = np.asarray(array, dtype=np.float32).copy()

    def asarray(self):
        return self._arr

    def as_array(self):
        return self._arr

    def copy(self):
        return _FakeContainer(self._arr.copy())

    def fill(self, array):
        np.copyto(self._arr, np.asarray(array, dtype=self._arr.dtype))

    def maximum(self, other, out=None):
        if out is None:
            out = self.copy()
        other_arr = other.asarray() if hasattr(other, "asarray") else other
        out.fill(np.maximum(self._arr, other_arr))
        return out

    def minimum(self, other, out=None):
        if out is None:
            out = self.copy()
        other_arr = other.asarray() if hasattr(other, "asarray") else other
        out.fill(np.minimum(self._arr, other_arr))
        return out

    def multiply(self, other, out=None):
        if out is None:
            out = self.copy()
        other_arr = other.asarray() if hasattr(other, "asarray") else other
        out.fill(self._arr * other_arr)
        return out


def test_proximal_applies_bounds_and_support_mask():
    x = _FakeContainer([-1.0, 1.5, 3.0, 0.4])
    mask = _FakeContainer([1.0, 0.0, 1.0, 0.0])
    g = BlockIndicatorBox(lower=0.0, upper=2.0, mask=mask)

    out = g.proximal(x, tau=1.0)
    np.testing.assert_allclose(out.asarray(), np.array([0.0, 0.0, 2.0, 0.0], dtype=np.float32))


def test_indicator_returns_inf_for_off_support_mass():
    x = _FakeContainer([0.0, 1.0, 0.5])
    mask = _FakeContainer([1.0, 0.0, 1.0])
    g = BlockIndicatorBox(lower=0.0, upper=np.inf, mask=mask)

    val = g(x)
    assert math.isinf(val)


def test_indicator_respects_mask_tolerance():
    x = _FakeContainer([0.0, 1e-9, 0.5])
    mask = _FakeContainer([1.0, 0.0, 1.0])
    g = BlockIndicatorBox(lower=0.0, upper=np.inf, mask=mask, mask_tolerance=1e-8)

    assert g(x) == 0.0
