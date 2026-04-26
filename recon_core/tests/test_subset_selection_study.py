import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP_SRC = ROOT.parent / "recon_experiments" / "src"
if str(EXP_SRC) not in sys.path:
    sys.path.insert(0, str(EXP_SRC))

try:
    from recon_experiments.runners import dtnv_common
    from recon_experiments.studies.subset_selection.scripts.analyze_subset_sweep import (
        parse_config_from_dirname,
    )
    from recon_experiments.studies.subset_selection.scripts.run_subset_selection import (
        _iter_projection_data_for_nan_check,
        _resolve_bpos,
        _validate_paired_subset_configuration,
        calculate_epoch_length_and_prior_updates,
        get_data_sampling_probabilities,
    )
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Subset-selection study utilities unavailable: {exc}", allow_module_level=True)


class _DummySVRG:
    def __init__(self, stochastic_functions, sampler, snapshot_update_interval, store_gradients):
        self.num_functions = len(stochastic_functions)
        self.sampler = sampler
        self.snapshot_update_interval = snapshot_update_interval
        self.store_gradients = store_gradients


def test_get_data_sampling_probabilities_match_actual_function_count():
    probs = get_data_sampling_probabilities([object()] * 18)

    assert len(probs) == 18
    assert sum(probs) == pytest.approx(1.0)
    assert all(p == pytest.approx(1.0 / 18.0) for p in probs)


def test_resolve_bpos_accepts_one_and_two():
    assert _resolve_bpos(SimpleNamespace(bpos=1)) == 1
    assert _resolve_bpos(SimpleNamespace(bpos=2)) == 2


def test_validate_paired_subset_configuration_requires_equal_counts_for_2bpos():
    _validate_paired_subset_configuration(1, [18, 12])
    _validate_paired_subset_configuration(2, [9, 9])

    with pytest.raises(ValueError, match="requires equal PET and SPECT subset counts"):
        _validate_paired_subset_configuration(2, [9, 12])


def test_projection_nan_check_iterates_per_bed_pet_data_for_2bpos():
    pet_data = {
        "bed_positions": {
            "_f1b1": {
                "acquisition_data": "pet-prompts-1",
                "normalisation": "pet-norm-1",
                "additive": "pet-additive-1",
            },
            "_f2b1": {
                "acquisition_data": "pet-prompts-2",
                "normalisation": "pet-norm-2",
                "additive": "pet-additive-2",
            },
        }
    }
    spect_data = {
        "acquisition_data": "spect-prompts",
        "additive": "spect-additive",
    }

    assert list(_iter_projection_data_for_nan_check(pet_data, spect_data)) == [
        "pet-prompts-1",
        "pet-norm-1",
        "pet-additive-1",
        "pet-prompts-2",
        "pet-norm-2",
        "pet-additive-2",
        "spect-prompts",
        "spect-additive",
    ]


def test_build_variance_reduced_function_supports_paired_subset_sampling(monkeypatch):
    captured = {}

    def fake_sampler(num_indices, prob=None, seed=None):
        captured["num_indices"] = num_indices
        captured["prob"] = list(prob)
        captured["seed"] = seed
        return "sampler"

    monkeypatch.setattr(dtnv_common.Sampler, "random_with_replacement", fake_sampler)
    monkeypatch.setattr(dtnv_common, "SVRGFunction", _DummySVRG)

    args = SimpleNamespace(
        variance_reduction="svrg",
        prior_updates_per_epoch=18,
        snapshot_interval_factor=None,
    )
    total_epoch_length, prior_updates = calculate_epoch_length_and_prior_updates(
        "paired",
        "subset",
        18,
        args,
    )

    f_obj, probs, prior_prob, prior_in_sampler = dtnv_common.build_variance_reduced_function(
        args=args,
        all_funs=[object()] * 18,
        prior=object(),
        num_subsets=[18, 18],
        epoch_length=18,
        bpos=1,
        data_probs=get_data_sampling_probabilities([object()] * 18),
    )

    assert total_epoch_length == 36
    assert prior_updates == 18
    assert prior_in_sampler is True
    assert prior_prob == pytest.approx(0.5)
    assert len(probs) == 19
    assert captured["num_indices"] == 19
    assert len(captured["prob"]) == 19
    assert captured["prob"][-1] == pytest.approx(0.5)
    assert all(p == pytest.approx(1.0 / 36.0) for p in captured["prob"][:-1])
    assert f_obj.num_functions == 19


def test_parse_config_from_dirname_accepts_current_preconditioner_names():
    parsed = parse_config_from_dirname(
        "subset_paired_prior_subset_precond_mm_diag_block_maj_gamma_1000"
    )

    assert parsed["subset_mode"] == "paired"
    assert parsed["prior_mode"] == "subset"
    assert parsed["precond_type"] == "mm_diag_block_maj"
    assert parsed["gamma_tnv"] == pytest.approx(1000.0)
