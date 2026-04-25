import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP_SRC = ROOT.parent / "recon_experiments" / "src"
if str(EXP_SRC) not in sys.path:
    sys.path.insert(0, str(EXP_SRC))

try:
    from recon_experiments.runners.scripts.run_dtnv_1bpos import (
        _coerce_restart_estimates_to_shared_grid,
    )
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"DTNV runner utilities unavailable: {exc}", allow_module_level=True)


class _MockImage:
    def __init__(self, shape, voxel_sizes=(1.0, 1.0, 1.0), name=""):
        self.shape = tuple(shape)
        self._voxel_sizes = tuple(voxel_sizes)
        self.name = name

    def voxel_sizes(self):
        return self._voxel_sizes


class _MockSpect2Pet:
    def __init__(self, shared_template):
        self.shared_template = shared_template

    def direct(self, image):
        return _MockImage(
            self.shared_template.shape,
            voxel_sizes=self.shared_template.voxel_sizes(),
            name=f"mapped:{image.name}",
        )


def test_restart_estimates_map_native_spect_to_shared_grid():
    pet_template = _MockImage((8, 7, 6), name="pet_shared")
    spect_template = _MockImage((5, 4, 3), name="spect_native")
    spect2pet = _MockSpect2Pet(pet_template)

    pet_restart = _MockImage((8, 7, 6), name="pet_restart")
    spect_restart_native = _MockImage((5, 4, 3), name="spect_restart_native")

    restart = _coerce_restart_estimates_to_shared_grid(
        pet_restart,
        spect_restart_native,
        pet_template,
        spect_template,
        spect2pet,
    )

    assert restart.containers[0] is pet_restart
    assert restart.containers[1].shape == pet_template.shape
    assert restart.containers[1].name == "mapped:spect_restart_native"


def test_restart_estimates_reject_pet_restart_off_shared_grid():
    pet_template = _MockImage((8, 7, 6), name="pet_shared")
    spect_template = _MockImage((5, 4, 3), name="spect_native")
    spect2pet = _MockSpect2Pet(pet_template)

    pet_restart_bad = _MockImage((5, 4, 3), name="pet_restart_bad")
    spect_restart_native = _MockImage((5, 4, 3), name="spect_restart_native")

    with pytest.raises(ValueError, match="PET restart image must already be on the shared PET grid"):
        _coerce_restart_estimates_to_shared_grid(
            pet_restart_bad,
            spect_restart_native,
            pet_template,
            spect_template,
            spect2pet,
        )
