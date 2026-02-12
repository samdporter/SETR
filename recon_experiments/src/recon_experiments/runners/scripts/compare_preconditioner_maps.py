#!/usr/bin/env python3
"""Visual comparison of TNV preconditioners on phantom and patient data."""

from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator
from sirf.STIR import ImageData, SeparableGaussianImageFilter

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.utils import get_pet_data, get_spect_data
from recon_core.utils.dynamic_range import apply_dynamic_range_scaling, dynamic_range_scale_sirf
from recon_core.utils.io import load_config
from recon_core.utils.sirf import get_array, get_filters
from recon_experiments.runners.common import attach_prior_hessian, get_resampling_operators
from recon_experiments.runners.dtnv_common import get_prior

try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent on some systems
    torch = None


_PRECOND_DIAG_METHODS = {"mm_diag", "mm_diag_gershgorin", "frob_diag"}
_PRECOND_BLOCK_METHODS = {"mm_block_diag", "ls_block_diag"}
_HESSIAN_TYPES = (
    "svd_principal_alpha",
    "mm_jensen",
    "frobenius_surrogate_pd",
    "vector_tv_per_modality",
)


def _parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _prepare_data(args: SimpleNamespace):
    ct = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    ct += (-ct).max()
    ct /= ct.max()
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((0.5, 0.5, 0.5))
    ct_smooth.apply(ct)

    pet_data = get_pet_data(args.pet_data_path, load_sinos=False)
    spect_data = get_spect_data(args.spect_data_path, load_sinos=False)

    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    return ct, pet_data, spect_data


def _build_geometry(args: SimpleNamespace):
    umap, pet_data, spect_data = _prepare_data(args)
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


def _resolve_methods(args: SimpleNamespace, precond_methods: str | None, hessian_types: str | None):
    if precond_methods and hessian_types:
        raise ValueError("Use either --precond-methods or --hessian-types, not both.")
    if hessian_types:
        if not getattr(args, "use_log_tnv", False):
            raise ValueError(
                "--hessian-types requires use_log_tnv=true, but this script forces use_log_tnv=False. "
                "Use --precond-methods instead."
            )
        methods = _parse_list(hessian_types)
        return "hessian", methods
    if precond_methods:
        return "precond", _parse_list(precond_methods)
    if getattr(args, "use_log_tnv", False):
        return "precond", ["mm_diag", "frob_diag"]
    return "precond", ["mm_diag", "frob_diag", "mm_block_diag"]


def _canonical_hessian(prior, hessian_type: str) -> str:
    if hasattr(prior.function, "_HESSIAN_CANONICAL"):
        canonical = hessian_type
        if hessian_type not in prior.function._HESSIAN_CANONICAL:
            canonical = prior.function._HESSIAN_ALIASES.get(hessian_type, hessian_type)
        return canonical
    return hessian_type


def _as_numpy(arr) -> np.ndarray:
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _extract_diag_arrays(diag) -> List[np.ndarray]:
    if hasattr(diag, "containers"):
        return [get_array(container) for container in diag.containers]
    return [np.asarray(diag)]


def _extract_block_arrays(blocks) -> List[np.ndarray]:
    arr = _as_numpy(blocks)
    p11 = arr[..., 0, 0]
    p22 = arr[..., 1, 1]
    p12 = 0.5 * (arr[..., 0, 1] + arr[..., 1, 0])
    return [p11, p12, p22]


def _slice_from_volume(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def _percentile_limits(images: Sequence[np.ndarray], percentile: float) -> Tuple[float, float]:
    flattened = []
    for img in images:
        if img.size == 0:
            continue
        flattened.append(img[np.isfinite(img)].ravel())
    if not flattened:
        return 0.0, 1.0
    stacked = np.concatenate(flattened)
    if stacked.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(stacked, 100.0 - percentile))
    vmax = float(np.nanpercentile(stacked, percentile))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_grid(
    output_path: Path,
    title: str,
    method_names: Sequence[str],
    channel_names: Sequence[str],
    slices: Sequence[Sequence[np.ndarray]],
    percentile: float,
    cmap: str,
) -> None:
    rows = len(channel_names)
    cols = len(method_names)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row, channel in enumerate(channel_names):
        row_images = [slices[row][col] for col in range(cols)]
        vmin, vmax = _percentile_limits(row_images, percentile)
        im = None
        for col, method in enumerate(method_names):
            ax = axes[row, col]
            im = ax.imshow(row_images[col], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method)
            if col == 0:
                ax.set_ylabel(channel)
        if im is not None:
            fig.colorbar(im, ax=axes[row, :], shrink=0.7, pad=0.01)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diff_grid(
    output_path: Path,
    title: str,
    method_names: Sequence[str],
    channel_names: Sequence[str],
    slices: Sequence[Sequence[np.ndarray]],
    percentile: float,
    cmap: str,
) -> None:
    rows = len(channel_names)
    cols = len(method_names)
    baseline = [row[0] for row in slices]
    diffs: List[List[np.ndarray]] = []
    for row_idx in range(rows):
        row_diffs = []
        for col_idx in range(cols):
            row_diffs.append(slices[row_idx][col_idx] - baseline[row_idx])
        diffs.append(row_diffs)

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row, channel in enumerate(channel_names):
        row_images = [diffs[row][col] for col in range(cols)]
        flat = np.concatenate([img.ravel() for img in row_images]) if row_images else np.array([0.0])
        max_abs = float(np.nanpercentile(np.abs(flat), percentile))
        if max_abs == 0:
            max_abs = 1.0
        im = None
        for col, method in enumerate(method_names):
            ax = axes[row, col]
            im = ax.imshow(row_images[col], cmap=cmap, vmin=-max_abs, vmax=max_abs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method)
            if col == 0:
                ax.set_ylabel(channel)
        if im is not None:
            fig.colorbar(im, ax=axes[row, :], shrink=0.7, pad=0.01)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _collect_slices(
    arrays_by_method: Sequence[Sequence[np.ndarray]],
    axis: int,
    index: int,
) -> List[List[np.ndarray]]:
    slices: List[List[np.ndarray]] = []
    for channel_idx in range(len(arrays_by_method[0])):
        row = []
        for method_idx in range(len(arrays_by_method)):
            volume = arrays_by_method[method_idx][channel_idx]
            row.append(_slice_from_volume(volume, axis, index))
        slices.append(row)
    return slices


def _run_config(
    label: str,
    config_path: str,
    precond_methods: str | None,
    hessian_types: str | None,
    axis: int,
    slice_index: int | None,
    percentile: float,
    output_dir: Path | None,
) -> None:
    logging.info("Loading %s config: %s", label, config_path)
    cfg = load_config(config_path)
    args = SimpleNamespace(**cfg)

    if not os.path.isdir(args.pet_data_path):
        raise FileNotFoundError(f"PET data path not found: {args.pet_data_path}")
    if not os.path.isdir(args.spect_data_path):
        raise FileNotFoundError(f"SPECT data path not found: {args.spect_data_path}")

    args.use_log_tnv = False

    umap, initial_estimates, combined, bo = _build_geometry(args)
    pet_scale, spect_scale = _apply_scaling_and_delta(args, combined)

    mode, methods = _resolve_methods(args, precond_methods, hessian_types)
    logging.info("Comparison mode=%s methods=%s", mode, methods)

    diag_methods: List[str] = []
    diag_arrays: List[List[np.ndarray]] = []
    block_methods: List[str] = []
    block_arrays: List[List[np.ndarray]] = []

    for method in methods:
        method_args = deepcopy(args)
        if mode == "precond":
            method_args.precond_method = method
        elif mode == "hessian":
            method_args.hessian_type = method

        if mode == "precond" and getattr(method_args, "use_log_tnv", False):
            if method in _PRECOND_BLOCK_METHODS:
                logging.warning("Skipping block method %s (log-TNV does not support block preconditioners).", method)
                continue

        prior = _build_tnv_prior(method_args, umap, combined, bo, pet_scale, spect_scale)

        if mode == "hessian":
            if not hasattr(prior.function, "hessian"):
                logging.warning("Prior does not expose hessian attribute; skipping %s.", method)
                continue
            prior.function.hessian = _canonical_hessian(prior, method)

        if mode == "precond" and method in _PRECOND_BLOCK_METHODS:
            blocks = prior.inv_hessian_block_diag(initial_estimates)
            block_arrays.append(_extract_block_arrays(blocks))
            block_methods.append(method)
        else:
            diag = prior.inv_hessian_diag(initial_estimates)
            diag_arrays.append(_extract_diag_arrays(diag))
            diag_methods.append(method)

    if output_dir is None:
        base_out = getattr(args, "output_path", None)
        if base_out:
            output_dir = Path(base_out) / "preconditioner_compare"
        else:
            output_dir = Path("output") / "preconditioner_compare"

    if diag_arrays:
        channel_names = ["PET", "SPECT"]
        diag_slice = slice_index
        if diag_slice is None:
            diag_slice = diag_arrays[0][0].shape[axis] // 2
        slices = _collect_slices(diag_arrays, axis, diag_slice)
        title = f"{label} diag preconditioners (axis={axis}, slice={diag_slice})"
        _plot_grid(
            output_dir / f"precond_diag_{label}_abs.png",
            title,
            diag_methods,
            channel_names,
            slices,
            percentile,
            cmap="viridis",
        )
        if len(diag_methods) > 1:
            _plot_diff_grid(
                output_dir / f"precond_diag_{label}_diff.png",
                f"{label} diag preconditioners (difference vs {diag_methods[0]})",
                diag_methods,
                channel_names,
                slices,
                percentile,
                cmap="coolwarm",
            )

    if block_arrays:
        channel_names = ["PET-PET", "PET-SPECT", "SPECT-SPECT"]
        block_slice = slice_index
        if block_slice is None:
            block_slice = block_arrays[0][0].shape[axis] // 2
        slices = _collect_slices(block_arrays, axis, block_slice)
        title = f"{label} block preconditioners (axis={axis}, slice={block_slice})"
        _plot_grid(
            output_dir / f"precond_block_{label}_abs.png",
            title,
            block_methods,
            channel_names,
            slices,
            percentile,
            cmap="viridis",
        )
        if len(block_methods) > 1:
            _plot_diff_grid(
                output_dir / f"precond_block_{label}_diff.png",
                f"{label} block preconditioners (difference vs {block_methods[0]})",
                block_methods,
                channel_names,
                slices,
                percentile,
                cmap="coolwarm",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual comparison of TNV preconditioner maps on phantom/patient data."
    )
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
        "--precond-methods",
        default=None,
        help="Comma-separated preconditioner methods to compare (diag or block).",
    )
    parser.add_argument(
        "--hessian-types",
        default=None,
        help=f"Comma-separated hessian types to compare (log-TNV only). Options: {', '.join(_HESSIAN_TYPES)}",
    )
    parser.add_argument(
        "--only",
        choices=("phantom", "patient", "both"),
        default="both",
        help="Which dataset(s) to run",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Slice axis (0=z, 1=y, 2=x).",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index along the chosen axis (defaults to center).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile for color scaling (symmetric for diff plots).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures (defaults to config output_path/preconditioner_compare).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.only in ("phantom", "both"):
        _run_config(
            "phantom",
            args.phantom_config,
            args.precond_methods,
            args.hessian_types,
            args.axis,
            args.slice_index,
            args.percentile,
            output_dir,
        )
    if args.only in ("patient", "both"):
        _run_config(
            "patient",
            args.patient_config,
            args.precond_methods,
            args.hessian_types,
            args.axis,
            args.slice_index,
            args.percentile,
            output_dir,
        )


if __name__ == "__main__":
    main()
