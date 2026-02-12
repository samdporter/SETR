#!/usr/bin/env python3
"""Benchmark VTV preconditioner timings on phantom and patient data."""

from __future__ import annotations

import argparse
import logging
import os
import time
from types import SimpleNamespace
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator
from sirf.STIR import ImageData, SeparableGaussianImageFilter

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.utils import get_pet_data, get_pet_data_multiple_bed_pos, get_spect_data
from recon_core.utils.dynamic_range import apply_dynamic_range_scaling, dynamic_range_scale_sirf
from recon_core.utils.io import load_config
from recon_core.utils.sirf import get_array, get_filters
from recon_experiments.runners.common import attach_prior_hessian, get_resampling_operators
from recon_experiments.runners.dtnv_common import get_prior

try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent on some systems
    torch = None


HESSIAN_TYPES = (
    "frob_diag",
    "mm_diag",
    "mm_diag_gershgorin",
    "mm_block_diag",
    "ls_block_diag",
)

_BLOCK_METHODS = {"mm_block_diag", "ls_block_diag"}


def _parse_hessian_types(raw: str) -> List[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or list(HESSIAN_TYPES)


def _prepare_data(args: SimpleNamespace, use_multi_bed: bool):
    if use_multi_bed:
        pet_data = get_pet_data_multiple_bed_pos(
            args.pet_data_path,
            suffixes=["_f1b1", "_f2b1"],
            tof=False,
            load_sinos=False,
        )
    else:
        pet_data = get_pet_data(args.pet_data_path, load_sinos=False)
    spect_data = get_spect_data(args.spect_data_path, load_sinos=False)

    ct = pet_data['attenuation']
    ct += (-ct).max()
    ct /= ct.max()
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((0.5, 0.5, 0.5))
    ct_smooth.apply(ct)

    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    return ct, pet_data, spect_data


def _build_geometry(args: SimpleNamespace, use_multi_bed: bool):
    umap, pet_data, spect_data = _prepare_data(args, use_multi_bed)
    spect2pet = get_resampling_operators(args, pet_data, spect_data)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    combined = EnhancedBlockDataContainer(*bo.direct(initial_estimates).containers)
    return umap, initial_estimates, combined, bo


def _apply_scaling_and_delta(
    args: SimpleNamespace, combined: EnhancedBlockDataContainer
) -> Tuple[float, float]:
    use_log_tnv = getattr(args, "use_log_tnv", False)
    use_local_weighting = getattr(args, "use_local_weighting", False)

    pet_scale, spect_scale = dynamic_range_scale_sirf(combined[0], combined[1])
    if not use_log_tnv and not use_local_weighting:
        apply_dynamic_range_scaling(args, pet_scale, spect_scale)

    if args.delta is not None:
        return pet_scale, spect_scale

    if use_log_tnv:
        divisor = getattr(args, "delta_divisor", 10.0)
        args.delta = 1.0 / divisor
        return pet_scale, spect_scale

    percentile = getattr(args, "delta_percentile", 99.0)
    divisor = getattr(args, "delta_divisor", 100.0)

    weighted_pet = args.alpha * get_array(combined[0])
    weighted_spect = args.beta * get_array(combined[1])

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


def _build_tnv_prior(
    args: SimpleNamespace,
    umap,
    combined: EnhancedBlockDataContainer,
    bo: BlockOperator,
    pet_scale: float,
    spect_scale: float,
):
    args.directional_tnv = True
    if not getattr(args, "use_tnv_prior", True):
        raise RuntimeError("use_tnv_prior is disabled; VTV prior cannot be constructed.")

    priors = get_prior(
        args,
        umap,
        combined,
        bo,
        kappas=None,
        pet_scale=pet_scale,
        spect_scale=spect_scale,
    )
    if not priors:
        raise RuntimeError("No priors returned from get_prior; check config settings.")

    for prior in priors:
        attach_prior_hessian(prior)

    return priors[0]


def _synchronize_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_preconditioner(
    prior,
    initial_estimates,
    method: str,
    repeats: int,
    warmup: int,
) -> List[float]:
    for _ in range(warmup):
        if method in _BLOCK_METHODS:
            _ = prior.inv_hessian_block_diag(initial_estimates)
        else:
            _ = prior.inv_hessian_diag(initial_estimates)
        _synchronize_cuda()

    timings = []
    for _ in range(repeats):
        _synchronize_cuda()
        start = time.perf_counter()
        if method in _BLOCK_METHODS:
            _ = prior.inv_hessian_block_diag(initial_estimates)
        else:
            _ = prior.inv_hessian_diag(initial_estimates)
        _synchronize_cuda()
        timings.append(time.perf_counter() - start)

    return timings


def _summarize(times: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.array(times, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std, float(np.min(arr)), float(np.max(arr))


def _run_config(
    label: str, config_path: str, hessian_types: Iterable[str], repeats: int, warmup: int
):
    logging.info("Loading %s config: %s", label, config_path)
    cfg = load_config(config_path)
    args = SimpleNamespace(**cfg)

    if not os.path.isdir(args.pet_data_path):
        raise FileNotFoundError(f"PET data path not found: {args.pet_data_path}")
    if not os.path.isdir(args.spect_data_path):
        raise FileNotFoundError(f"SPECT data path not found: {args.spect_data_path}")

    args.use_log_tnv = False

    use_multi_bed = label == "patient"
    umap, initial_estimates, combined, bo = _build_geometry(args, use_multi_bed)
    pet_scale, spect_scale = _apply_scaling_and_delta(args, combined)

    logging.info(
        "%s geometry: PET %s | SPECT %s",
        label,
        initial_estimates[0].shape,
        initial_estimates[1].shape,
    )

    results = []
    for hessian_type in hessian_types:
        args.precond_method = hessian_type
        prior = _build_tnv_prior(args, umap, combined, bo, pet_scale, spect_scale)
        timings = _time_preconditioner(prior, initial_estimates, hessian_type, repeats, warmup)
        mean, std, min_t, max_t = _summarize(timings)
        results.append((hessian_type, mean, std, min_t, max_t))

    print(f"\n{label} ({config_path})")
    for hessian_type, mean, std, min_t, max_t in results:
        print(
            f"- precond_method={hessian_type} mean={mean:.4f}s std={std:.4f}s "
            f"min={min_t:.4f}s max={max_t:.4f}s"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time VTV preconditioners on patient/phantom data.")
    parser.add_argument(
        "--phantom-config",
        default="recon_experiments/configs/config_1bpos_anthro_long.yaml",
        help="Phantom config path",
    )
    parser.add_argument(
        "--patient-config",
        default="recon_experiments/configs/config_2bpos.yaml",
        help="Patient config path",
    )
    parser.add_argument(
        "--hessian-types",
        default=",".join(HESSIAN_TYPES),
        help="Comma-separated preconditioner methods to test (diag or block)",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Timing repeats per type")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per type")
    parser.add_argument(
        "--only",
        choices=("phantom", "patient", "both"),
        default="both",
        help="Which dataset(s) to run",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    hessian_types = _parse_hessian_types(args.hessian_types)
    if args.only in ("phantom", "both"):
        _run_config("phantom", args.phantom_config, hessian_types, args.repeats, args.warmup)
    if args.only in ("patient", "both"):
        _run_config("patient", args.patient_config, hessian_types, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
