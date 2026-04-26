import numpy as np
import pytest

pytest.importorskip("torch")

from recon_core.priors.vtv.vtv import (
    WeightedLogVectorialTotalVariation,
    WeightedVectorialTotalVariation,
)


class _FakeImage:
    def __init__(self, array, voxel_sizes=(1.0, 1.0, 1.0)):
        self._array = np.asarray(array, dtype=np.float32)
        self._voxel_sizes = tuple(voxel_sizes)

    def asarray(self):
        return self._array

    def as_array(self):
        return self._array

    def fill(self, array):
        np.copyto(self._array, np.asarray(array, dtype=self._array.dtype))

    def voxel_sizes(self):
        return self._voxel_sizes


class _FakeBlock:
    def __init__(self, containers):
        self.containers = containers


def _make_fake_block(shape=(4, 4, 4), num_modalities=2, value=1.0):
    return _FakeBlock(
        [_FakeImage(np.full(shape, value, dtype=np.float32)) for _ in range(num_modalities)]
    )


@pytest.mark.parametrize(
    "stable, expected_module_suffix",
    [
        (True, "schatten_norm_gpu_slow"),
        (False, "schatten_norm_gpu_stable"),
    ],
)
def test_wvtv_backend_selection_matches_stable_flag(stable, expected_module_suffix):
    geometry = _make_fake_block()
    weights = _make_fake_block(value=1.0)
    prior = WeightedVectorialTotalVariation(
        geometry=geometry,
        weights=weights,
        delta=1e-3,
        smoothing="charbonnier",
        anatomical=None,
        stable=stable,
        stencil="6",
        both_directions=False,
        max_step=1,
        precond_method="mm_diag_tight",
        bnd_cond="Neumann",
    )

    assert expected_module_suffix in prior.vtv.__class__.__module__


@pytest.mark.parametrize(
    "stable, expected_module_suffix",
    [
        (True, "schatten_norm_gpu_slow"),
        (False, "schatten_norm_gpu_stable"),
    ],
)
def test_log_wvtv_backend_selection_matches_stable_flag(stable, expected_module_suffix):
    geometry = _make_fake_block()
    weights = _make_fake_block(value=1.0)
    prior = WeightedLogVectorialTotalVariation(
        geometry=geometry,
        weights=weights,
        delta=1e-3,
        log_eps_values=[1e-3, 1e-3],
        smoothing="charbonnier",
        anatomical=None,
        stable=stable,
        stencil="6",
        both_directions=False,
        max_step=1,
        hessian="mm_jensen",
        bnd_cond="Neumann",
    )

    assert expected_module_suffix in prior.vtv.__class__.__module__
