import os
import sys
from pathlib import Path

import pytest

if os.getenv("SETR_RUN_REAL_GRID_INTEGRATION", "0") != "1":
    pytest.skip(
        "Set SETR_RUN_REAL_GRID_INTEGRATION=1 to run real PET/SPECT grid integration checks.",
        allow_module_level=True,
    )


def _maybe_add_recon_experiments_src() -> None:
    candidates = [
        Path(__file__).resolve().parents[2] / "recon_experiments" / "src",
        Path("/home/sam/working/synergistic_recon/recon_experiments/src"),
    ]
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            return


_maybe_add_recon_experiments_src()

sirf = pytest.importorskip("sirf.STIR", reason="SIRF/STIR is required")
from cil.optimisation.functions import OperatorCompositionFunction
from sirf.contrib.partitioner import partitioner
from sirf.STIR import AcquisitionData, MessageRedirector

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.utils.sirf import get_pet_data, get_spect_am, get_spect_data
from recon_experiments.runners.common import (
    build_shared_initial_estimates,
    get_pet_to_spect_operator,
    get_resampling_operators,
)
from recon_experiments.runners.dtnv_common import get_block_objective


MessageRedirector()


def test_real_non_aligned_spect_subset_gradient_through_pet_grid_adjoint():
    pet_path = os.getenv(
        "SETR_TEST_PET_PATH",
        "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/PET/phantom_short",
    )
    spect_path = os.getenv(
        "SETR_TEST_SPECT_PATH",
        "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140",
    )

    if not (Path(pet_path).is_dir() and Path(spect_path).is_dir()):
        pytest.skip(
            "Prepared PET/SPECT data unavailable for real grid integration test: "
            f"PET={pet_path!r}, SPECT={spect_path!r}"
        )

    AcquisitionData.set_storage_scheme("memory")
    pet_data = get_pet_data(pet_path, load_sinos=False)
    spect_data = get_spect_data(spect_path, load_sinos=True)
    spect2pet = get_resampling_operators(pet_data, spect_data)
    pet_to_spect = get_pet_to_spect_operator(spect2pet)
    x = build_shared_initial_estimates(
        pet_data["initial_image"],
        spect_data["initial_image"],
        spect2pet,
    )

    def create_spect_am():
        return get_spect_am(
            spect_data,
            res=[1.31, 0.027, False],
            keep_all_views_in_cache=True,
            gauss_fwhm=[6.8, 6.8, 6.8],
            attenuation=True,
        )

    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"],
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=18,
        mode="staggered",
        create_acq_model=create_spect_am,
    )
    spect_obj_funs[0].set_up(spect_data["initial_image"])

    spect_fun_on_pet_grid = OperatorCompositionFunction(
        spect_obj_funs[0],
        pet_to_spect,
    )
    block_fun = get_block_objective(
        x[1],
        x[0],
        spect_fun_on_pet_grid,
        order=1,
    )

    grad = block_fun.gradient(EnhancedBlockDataContainer(x[0].copy(), x[1].copy()))

    assert grad.containers[0].shape == x[0].shape
    assert grad.containers[1].shape == x[1].shape
    assert grad.norm() > 0
