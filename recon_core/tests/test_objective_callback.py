import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXP_SRC = ROOT.parent / "recon_experiments" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXP_SRC) not in sys.path:
    sys.path.insert(0, str(EXP_SRC))

try:
    from recon_core.cil_extensions.callbacks.callbacks import SaveObjectiveCallback
except (ImportError, OSError) as exc:  # pragma: no cover - external dependency
    pytest.skip(f"CIL/SIRF dependencies unavailable: {exc}", allow_module_level=True)


def test_save_objective_callback_rewrites_with_full_history(tmp_path):
    out_prefix = tmp_path / "objective"
    cb = SaveObjectiveCallback(str(out_prefix), interval=2)

    algo = SimpleNamespace(
        iteration=0,
        max_iteration=4,
        update_objective_interval=2,
        objective=[],
    )

    values = [10.0, 9.5, 9.0, 8.8, 8.7]
    for i, v in enumerate(values):
        algo.iteration = i
        algo.objective.append(v)
        cb(algo)

    out_file = out_prefix.with_suffix(".csv")
    assert out_file.exists()

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    # header + all objective entries
    assert len(lines) == len(values) + 1
    assert lines[-1].endswith(str(values[-1]))


def test_save_objective_callback_appends_existing_history_on_restart(tmp_path):
    out_prefix = tmp_path / "objective"
    out_prefix.with_suffix(".csv").write_text("0\n3.0\n2.5\n", encoding="utf-8")

    cb = SaveObjectiveCallback(str(out_prefix), interval=1, iteration_offset=2)
    algo = SimpleNamespace(
        iteration=0,
        max_iteration=1,
        update_objective_interval=1,
        objective=[2.0],
    )

    cb(algo)

    lines = out_prefix.with_suffix(".csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["0", "3.0", "2.5", "2.0"]


def test_get_callbacks_threads_restart_offset_to_objective_callback(tmp_path):
    try:
        from recon_experiments.runners.dtnv_common import get_callbacks
    except (ImportError, OSError) as exc:  # pragma: no cover - external dependency
        pytest.skip(f"recon_experiments dependencies unavailable: {exc}")

    args = SimpleNamespace(
        output_path=str(tmp_path),
        save_gradients=False,
        save_preconditioners=False,
    )

    callbacks = get_callbacks(args, update_interval=2, iteration_offset=12)
    objective_callbacks = [cb for cb in callbacks if isinstance(cb, SaveObjectiveCallback)]

    assert len(objective_callbacks) == 1
    assert objective_callbacks[0].iteration_offset == 12
