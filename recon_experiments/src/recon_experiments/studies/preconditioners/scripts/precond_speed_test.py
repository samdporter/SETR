#!/usr/bin/env python3
"""Measure preconditioner evaluation speed on 1bpos/2bpos datasets."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import numpy as np
import yaml
from sirf.STIR import ImageData, SeparableGaussianImageFilter

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.utils import get_pet_data, get_pet_data_multiple_bed_pos, get_spect_data
from recon_core.utils.dynamic_range import apply_dynamic_range_scaling, dynamic_range_scale_sirf
from recon_core.utils.sirf import get_array, get_filters
from recon_experiments.runners.common import (
    build_shared_initial_estimates,
    get_resampling_operators,
)
from recon_experiments.runners.dtnv_common import get_preconditioners, get_prior

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _resolve_base_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src" / "recon_experiments").is_dir():
            return parent
    raise RuntimeError("Could not locate recon_experiments base directory.")


def _load_sweep_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_base_config(base_dir: Path, config_name: str, fixed_params: dict | None) -> SimpleNamespace:
    config_path = base_dir / "configs" / config_name
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in (fixed_params or {}).items():
        cfg[key] = value
    if "alpha" in cfg and "beta" not in cfg:
        cfg["beta"] = cfg["alpha"]
    return SimpleNamespace(**cfg)


def _prepare_data_1bpos(args: SimpleNamespace):
    ct = ImageData(str(Path(args.pet_data_path) / "umap_zoomed.hv"))
    ct += (-ct).max()
    ct /= ct.max()
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((0.5, 0.5, 0.5))
    ct_smooth.apply(ct)

    pet_data = get_pet_data(args.pet_data_path)
    spect_data = get_spect_data(args.spect_data_path)

    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )
    return ct, pet_data, spect_data, initial_estimates


def _prepare_data_2bpos(args: SimpleNamespace):
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=getattr(args, "use_tof", False), suffixes=["_f1b1", "_f2b1"]
    )
    ct = pet_data["attenuation"]
    ct += (-ct).max()
    ct /= ct.max()
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((2, 2, 2))
    ct_smooth.apply(ct)

    spect_data = get_spect_data(args.spect_data_path)

    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )
    return ct, pet_data, spect_data, initial_estimates


def _build_geometry(args: SimpleNamespace, bpos: int):
    if bpos == 1:
        ct, pet_data, spect_data, initial_estimates = _prepare_data_1bpos(args)
    else:
        ct, pet_data, spect_data, initial_estimates = _prepare_data_2bpos(args)

    spect2pet = get_resampling_operators(args, pet_data, spect_data)
    shared_initial_estimates = build_shared_initial_estimates(
        pet_data["initial_image"],
        spect_data["initial_image"],
        spect2pet,
    )
    return ct, shared_initial_estimates, shared_initial_estimates


def _apply_scaling_and_delta(args: SimpleNamespace, combined: EnhancedBlockDataContainer) -> Tuple[float, float]:
    use_log_tnv = getattr(args, "use_log_tnv", False)
    use_local_weighting = getattr(args, "use_local_weighting", False)

    pet_scale, spect_scale = dynamic_range_scale_sirf(combined[0], combined[1])
    if not use_log_tnv and not use_local_weighting:
        apply_dynamic_range_scaling(args, pet_scale, spect_scale)

    if getattr(args, "delta", None) is not None:
        return pet_scale, spect_scale

    if use_log_tnv:
        divisor = float(getattr(args, "delta_divisor", 10.0))
        args.delta = 1.0 / divisor
        return pet_scale, spect_scale

    percentile = float(getattr(args, "delta_percentile", getattr(args, "dynamic_percentile", 99.0)))
    divisor = float(getattr(args, "delta_divisor", 100.0))

    weighted_pet = float(args.alpha) * get_array(combined[0])
    weighted_spect = float(args.beta) * get_array(combined[1])

    def _safe_percentile(arr: np.ndarray, pct: float):
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            return None
        return float(np.percentile(arr, pct))

    pet_val = _safe_percentile(weighted_pet, percentile)
    spect_val = _safe_percentile(weighted_spect, percentile)
    if pet_val is None and spect_val is None:
        args.delta = 1.0 / divisor
        return pet_scale, spect_scale
    if pet_val is None:
        scale_val = spect_val
    elif spect_val is None:
        scale_val = pet_val
    else:
        scale_val = min(pet_val, spect_val)
    args.delta = scale_val / divisor
    return pet_scale, spect_scale


def _build_priors(
    args: SimpleNamespace,
    ct,
    combined: EnhancedBlockDataContainer,
    pet_scale: float,
    spect_scale: float,
):
    priors = get_prior(
        args,
        ct,
        combined,
        kappas=None,
        pet_scale=pet_scale,
        spect_scale=spect_scale,
    )
    return priors


def _make_uniform_sinv(initial_estimates: EnhancedBlockDataContainer) -> EnhancedBlockDataContainer:
    if hasattr(initial_estimates, "get_uniform_copy"):
        return initial_estimates.get_uniform_copy(1)
    return EnhancedBlockDataContainer(
        *[el.get_uniform_copy(1) for el in initial_estimates.containers]
    )


def _make_random_solution(
    rng: np.random.Generator, initial_estimates: EnhancedBlockDataContainer
) -> EnhancedBlockDataContainer:
    containers = []
    for el in initial_estimates.containers:
        arr = get_array(el)
        rand = rng.random(arr.shape).astype(arr.dtype, copy=False)
        img = el.clone()
        img.fill(rand)
        containers.append(img)
    return EnhancedBlockDataContainer(*containers)


def _synchronize_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_preconditioner(precond, algorithm, repeats: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        precond.compute_preconditioner(algorithm)
        _synchronize_cuda()

    timings = []
    for _ in range(repeats):
        _synchronize_cuda()
        start = time.perf_counter()
        precond.compute_preconditioner(algorithm)
        _synchronize_cuda()
        timings.append(time.perf_counter() - start)
    return timings


def _summarize(times: Iterable[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(times), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std, float(np.min(arr)), float(np.max(arr))


def _load_precond_types(param_path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    with param_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        precond = parts[0]
        combine = parts[1] if len(parts) > 1 else ""
        entries.append((precond, combine))
    return entries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DTNV preconditioner evaluation speed.")
    parser.add_argument(
        "--sweep-configs",
        nargs="*",
        default=["precond_sweep_1bpos.yaml", "precond_sweep_2bpos.yaml"],
        help="Sweep config(s) under studies/preconditioners/configs",
    )
    parser.add_argument(
        "--precond-types-file",
        default="parameters/precond_types.csv",
        help="CSV with precond_type,combine (relative to studies/preconditioners)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats per method")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per method")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    base_dir = _resolve_base_dir()
    study_dir = base_dir / "src" / "recon_experiments" / "studies" / "preconditioners"
    output_dir = study_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    precond_file = Path(args.precond_types_file)
    if not precond_file.is_absolute():
        precond_file = study_dir / precond_file
    precond_entries = _load_precond_types(precond_file)
    rng = np.random.default_rng(args.seed)

    for sweep_name in args.sweep_configs:
        sweep_path = Path(sweep_name)
        if not sweep_path.is_absolute():
            sweep_path = study_dir / "configs" / sweep_name
        sweep_cfg = _load_sweep_config(sweep_path)
        base_config = sweep_cfg["base_config"]
        fixed_params = sweep_cfg.get("fixed_params", {}) or {}
        sweep_label = sweep_cfg.get("sweep_name", sweep_path.stem)

        bpos = 2 if "2bpos" in base_config.lower() else 1

        cfg_args = _load_base_config(base_dir, base_config, fixed_params)

        ct, initial_estimates, combined = _build_geometry(cfg_args, bpos)
        pet_scale, spect_scale = _apply_scaling_and_delta(cfg_args, combined)
        s_inv = _make_uniform_sinv(initial_estimates)
        all_funs = [None]

        solution = _make_random_solution(rng, initial_estimates)

        class _DummyAlg:
            def __init__(self, sol):
                self.solution = sol
                self.iteration = 0

        algo = _DummyAlg(solution)

        results = []
        for precond_type, combine in precond_entries:
            cfg_args.precond_type = precond_type
            cfg_args.precond_combine = combine or "majoriser"
            status = "success"
            error = ""
            mean = std = min_t = max_t = float("nan")

            try:
                # Rebuild priors per method so the prior's internal preconditioner
                # mode matches cfg_args.precond_type for this timing entry.
                priors = _build_priors(cfg_args, ct, combined, pet_scale, spect_scale)
                precond = get_preconditioners(
                    cfg_args,
                    s_inv,
                    all_funs,
                    update_interval=1,
                    priors_list=priors,
                    initial_estimates=initial_estimates,
                )
                timings = _time_preconditioner(precond, algo, args.repeats, args.warmup)
                mean, std, min_t, max_t = _summarize(timings)
            except Exception as exc:  # pragma: no cover - defensive benchmark robustness
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                logging.exception(
                    "Benchmark failed for precond_type=%s combine=%s",
                    precond_type,
                    combine,
                )

            results.append(
                {
                    "sweep": sweep_label,
                    "base_config": base_config,
                    "bpos": bpos,
                    "precond_type": precond_type,
                    "combine": combine,
                    "mean_s": mean,
                    "std_s": std,
                    "min_s": min_t,
                    "max_s": max_t,
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "pet_shape": str(get_array(initial_estimates[0]).shape),
                    "spect_shape": str(get_array(initial_estimates[1]).shape),
                    "device": "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
                    "status": status,
                    "error": error,
                }
            )

        out_path = output_dir / f"precond_speed_{sweep_label}.csv"
        if results:
            headers = list(results[0].keys())
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results)
        logging.info("Wrote %d rows to %s", len(results), out_path)


if __name__ == "__main__":
    main()
