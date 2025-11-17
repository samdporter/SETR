import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Ensure src/ is on sys.path when tests run from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setr.utils.dynamic_range import (
    apply_dynamic_range_scaling,
    dynamic_range_scale_sirf,
)


class DummyImage:
    """Minimal SIRF-like object exposing asarray."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def asarray(self):
        return self._array


def test_dynamic_range_scale_returns_inverse_ranges():
    pet = DummyImage(np.linspace(0, 9, 10).reshape(2, 5))
    spect = DummyImage(np.linspace(0, 19, 20).reshape(4, 5))

    pet_scale, spect_scale = dynamic_range_scale_sirf(
        pet,
        spect,
        percentile_high=None,
        percentile_low=None,
        use_absolute=False,
    )

    assert pet_scale == pytest.approx(1.0 / 9.0)
    assert spect_scale == pytest.approx(1.0 / 19.0)


def test_dynamic_range_scale_respects_mask_and_absolute():
    pet = DummyImage([[-10.0, np.nan], [5.0, -1.0]])
    spect = DummyImage([[2.0, -8.0], [np.nan, -4.0]])
    mask = DummyImage([[True, False], [False, True]])

    pet_scale, spect_scale = dynamic_range_scale_sirf(
        pet,
        spect,
        mask=mask,
        percentile_high=100.0,
        percentile_low=0.0,
        use_absolute=True,
    )

    assert pet_scale == pytest.approx(1.0 / 9.0)
    assert spect_scale == pytest.approx(1.0 / 2.0)


def test_dynamic_range_scale_empty_mask_returns_unity():
    mask = DummyImage(np.zeros((2, 2), dtype=bool))
    pet_scale, spect_scale = dynamic_range_scale_sirf(
        DummyImage(np.ones((2, 2))),
        DummyImage(np.ones((2, 2))),
        mask=mask,
    )

    assert pet_scale == pytest.approx(1.0)
    assert spect_scale == pytest.approx(1.0)


def test_apply_dynamic_range_scaling_updates_weights():
    args = SimpleNamespace(alpha=2.0, beta=4.0, gamma_pet=10.0, gamma_spect=5.0)

    apply_dynamic_range_scaling(args, pet_scale=0.5, spect_scale=0.25)

    assert args.alpha == pytest.approx(1.0)
    assert args.beta == pytest.approx(1.0)
    assert args.gamma_pet == pytest.approx(5.0)
    assert args.gamma_spect == pytest.approx(1.25)


def test_apply_dynamic_range_scaling_handles_missing_gamma_weights():
    args = SimpleNamespace(alpha=3.0, beta=6.0)

    apply_dynamic_range_scaling(args, pet_scale=0.1, spect_scale=0.2)

    assert args.alpha == pytest.approx(0.3)
    assert args.beta == pytest.approx(1.2)
    assert not hasattr(args, "gamma_pet")
    assert not hasattr(args, "gamma_spect")
