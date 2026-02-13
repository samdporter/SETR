import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if os.getenv("SETR_RUN_FULL_PIPELINE_FD", "0") != "1":
    pytest.skip(
        "Set SETR_RUN_FULL_PIPELINE_FD=1 to run full DTNV pipeline finite-difference checks.",
        allow_module_level=True,
    )


def _maybe_add_recon_experiments_src() -> None:
    candidates = []
    env_path = os.getenv("SETR_RECON_EXPERIMENTS_SRC")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parents[2] / "recon_experiments" / "src")
    candidates.append(Path("/home/sam/working/synergistic_recon/recon_experiments/src"))

    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            return


_maybe_add_recon_experiments_src()
pytest.importorskip("recon_experiments", reason="recon_experiments package is required for pipeline assembly")

try:
    from cil.optimisation.functions import SumFunction
    from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator
except Exception as exc:  # pragma: no cover - environment-dependent imports
    pytest.skip(f"CIL unavailable for pipeline FD checks: {exc}", allow_module_level=True)

try:
    from sirf.STIR import AcquisitionData
except Exception as exc:  # pragma: no cover - environment-dependent imports
    pytest.skip(f"SIRF/STIR unavailable for pipeline FD checks: {exc}", allow_module_level=True)

try:
    from recon_experiments.runners.dtnv_common import get_prior
    from recon_experiments.runners.scripts.run_dtnv_1bpos import get_data_fidelity, prepare_data
    from recon_experiments.runners.common import get_resampling_operators
except Exception as exc:  # pragma: no cover - environment-dependent imports
    pytest.skip(f"DTNV runner utilities unavailable: {exc}", allow_module_level=True)

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.utils.dynamic_range import apply_dynamic_range_scaling, dynamic_range_scale_sirf
from recon_core.utils.io import load_config
from recon_core.utils.sirf import get_array, get_pet_am, get_spect_am


def _objective_value(fun, x) -> float:
    return float(fun(x))


def _block_dot(x, direction_arrays) -> float:
    val = 0.0
    for container, direction in zip(x.containers, direction_arrays):
        arr = np.asarray(get_array(container), dtype=np.float64)
        val += float(np.vdot(arr.ravel(), direction.ravel()).real)
    return val


def _direction_like(x, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    dirs = []
    sq_norm = 0.0
    for container in x.containers:
        arr = rng.standard_normal(container.shape).astype(np.float32)
        dirs.append(arr)
        sq_norm += float(np.vdot(arr.ravel(), arr.ravel()).real)
    sq_norm = max(sq_norm, 1e-20)
    scale = 1.0 / np.sqrt(sq_norm)
    return [d * scale for d in dirs]


def _safe_eps_cap(base_arrays, direction_arrays) -> float:
    cap = np.inf
    for base, direction in zip(base_arrays, direction_arrays):
        mask = direction > 0
        if np.any(mask):
            cap = min(cap, float(np.min(base[mask] / direction[mask])))
    if not np.isfinite(cap):
        return 1.0
    return max(1e-5, 0.25 * cap)


def _best_fd_match(fun, x, base_arrays, direction_arrays, inner, steps):
    best = None
    for eps in steps:
        x_plus = x.clone()
        x_minus = x.clone()
        for c_plus, c_minus, base, direction in zip(
            x_plus.containers, x_minus.containers, base_arrays, direction_arrays
        ):
            c_plus.fill((base + eps * direction).astype(np.float32, copy=False))
            c_minus.fill((base - eps * direction).astype(np.float32, copy=False))

        fd = (_objective_value(fun, x_plus) - _objective_value(fun, x_minus)) / (2.0 * eps)
        rel_err = abs(fd - inner) / max(abs(fd), abs(inner), 1e-12)

        if best is None or rel_err < best["rel_err"]:
            best = {"step": eps, "fd": fd, "rel_err": rel_err}
    return best


def _auto_set_delta_like_runner(args, combined):
    if args.delta is not None:
        return
    weighted_pet = args.alpha * get_array(combined[0])
    weighted_spect = args.beta * get_array(combined[1])
    percentile = float(getattr(args, "delta_percentile", 99.0))
    divisor = float(getattr(args, "delta_divisor", 100.0))

    pet_vals = weighted_pet[weighted_pet > 0]
    spect_vals = weighted_spect[weighted_spect > 0]
    if pet_vals.size == 0 or spect_vals.size == 0:
        args.delta = 1e-3
        return
    pet_val = float(np.percentile(pet_vals, percentile))
    spect_val = float(np.percentile(spect_vals, percentile))
    args.delta = min(pet_val, spect_val) / max(divisor, 1e-12)


def test_full_pipeline_objective_gradient_matches_finite_difference_real_data(tmp_path, monkeypatch):
    config_path = Path(
        os.getenv(
            "SETR_PIPELINE_CONFIG_PATH",
            "/home/sam/working/synergistic_recon/recon_experiments/configs/config_1bpos_anthro_long.yaml",
        )
    )
    if not config_path.is_file():
        pytest.skip(f"Pipeline config not found: {config_path}")

    cfg = load_config(str(config_path))
    cfg["pet_data_path"] = os.getenv(
        "SETR_TEST_PET_PATH",
        cfg.get(
            "pet_data_path",
            "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/PET/phantom",
        ),
    )
    cfg["spect_data_path"] = os.getenv(
        "SETR_TEST_SPECT_PATH",
        cfg.get(
            "spect_data_path",
            "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140",
        ),
    )
    if not (Path(cfg["pet_data_path"]).is_dir() and Path(cfg["spect_data_path"]).is_dir()):
        pytest.skip(
            "Prepared PET/SPECT data paths not available for full pipeline FD test: "
            f"PET={cfg['pet_data_path']!r}, SPECT={cfg['spect_data_path']!r}"
        )

    cfg["output_path"] = str(tmp_path / "output")
    cfg["working_path"] = str(tmp_path / "work")
    cfg["save_images"] = False
    cfg["save_preconditioners"] = False
    cfg["save_gradients"] = False
    os.makedirs(cfg["output_path"], exist_ok=True)
    os.makedirs(cfg["working_path"], exist_ok=True)
    monkeypatch.chdir(cfg["working_path"])

    args = SimpleNamespace(**cfg)
    AcquisitionData.set_storage_scheme("memory")

    subset_override = os.getenv("SETR_PIPELINE_NUM_SUBSETS")
    if subset_override:
        vals = [int(v.strip()) for v in subset_override.split(",") if v.strip()]
        if len(vals) != 2:
            raise ValueError(
                "SETR_PIPELINE_NUM_SUBSETS must contain two comma-separated integers, e.g. '4,4'."
            )
        args.num_subsets = vals

    # Assemble the exact 1bpos DTNV objective stack as in the runner.
    umap, pet_data, spect_data = prepare_data(args)
    spect2pet = get_resampling_operators(args, pet_data, spect_data)
    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    def get_pet_am_with_res():
        return get_pet_am(not args.no_gpu, gauss_fwhm=None)

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.keep_all_views_in_cache,
            gauss_fwhm=args.spect_gauss_fwhm,
            attenuation=True,
        )

    num_subsets = [int(i) for i in args.num_subsets]
    all_funs, _, kappas = get_data_fidelity(
        args,
        pet_data,
        spect_data,
        get_pet_am_with_res,
        get_spect_am_with_res,
        num_subsets,
    )

    # Mirror composition used in run_dtnv_1bpos.py.
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )
    combined = EnhancedBlockDataContainer(*bo.direct(initial_estimates).containers)

    pet_scale, spect_scale = dynamic_range_scale_sirf(combined[0], combined[1])
    use_log_tnv = bool(getattr(args, "use_log_tnv", False))
    use_local_weighting = bool(getattr(args, "use_local_weighting", False))
    if not use_log_tnv and not use_local_weighting:
        apply_dynamic_range_scaling(args, pet_scale, spect_scale)
    _auto_set_delta_like_runner(args, combined)

    priors_list = get_prior(
        args,
        umap,
        combined,
        bo,
        kappas,
        pet_scale=pet_scale,
        spect_scale=spect_scale,
    )
    prior_signed = -SumFunction(*priors_list)
    data_sum = SumFunction(*all_funs)
    objective = -SumFunction(data_sum, prior_signed)

    # Use a strictly positive interior point to avoid boundary/nonnegativity effects in FD checks.
    x = initial_estimates.clone()
    base_arrays = []
    for container in x.containers:
        arr = np.array(get_array(container), dtype=np.float32, copy=True)
        arr = np.maximum(arr, 1e-3)
        container.fill(arr)
        base_arrays.append(arr)

    direction_arrays = _direction_like(x, seed=87)
    grad = objective.gradient(x)
    inner = _block_dot(grad, direction_arrays)

    eps_cap = _safe_eps_cap(base_arrays, direction_arrays)
    trial_steps = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
    fd_steps_override = os.getenv("SETR_PIPELINE_FD_STEPS")
    if fd_steps_override:
        trial_steps = [float(v.strip()) for v in fd_steps_override.split(",") if v.strip()]
    trial_steps = [s for s in trial_steps if s <= eps_cap]
    if not trial_steps:
        trial_steps = [max(1e-4, 0.5 * eps_cap)]

    best = _best_fd_match(objective, x, base_arrays, direction_arrays, inner, trial_steps)
    print(
        "Full pipeline FD check:",
        f"best_rel_err={best['rel_err']:.6e}",
        f"step={best['step']:.6e}",
        f"inner={inner:.6e}",
        f"fd={best['fd']:.6e}",
    )
    tol = float(os.getenv("SETR_PIPELINE_FD_TOL", "1e-1"))

    assert best["rel_err"] < tol, (
        "Full pipeline objective gradient/FD mismatch too large: "
        f"best rel err={best['rel_err']:.3e}, step={best['step']:.1e}, "
        f"inner={inner:.6e}, fd={best['fd']:.6e}."
    )
