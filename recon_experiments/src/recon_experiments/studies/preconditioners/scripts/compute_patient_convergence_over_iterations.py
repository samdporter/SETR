#!/usr/bin/env python3
"""
Compute convergence-over-iterations for patient sweeps using lesion/background VOIs.

This mirrors the phantom convergence script but sources VOIs from:
  /home/storage/cluster/patient_sweeps/lesion_masks/<sirtX>/<method>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sirf.STIR import ImageData

import compute_convergence_over_iterations as base


DEFAULT_LESION_MASKS_ROOT = Path("/home/storage/cluster/patient_sweeps/lesion_masks")
PREFERRED_LESION_METHODS: Tuple[str, ...] = (
    "intensity_combined_recon",
    "intensity",
    "radial_gradient_combined_recon",
    "radial_gradient",
)


def _infer_patient_id_from_text(value: object) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    match = re.search(r"(sirt\d+)", text)
    if match:
        return match.group(1)
    return None


def _infer_patient_id_for_alpha(
    baseline_dir: Path,
    alpha: float,
    explicit_patient_id: Optional[str],
) -> Optional[str]:
    if explicit_patient_id:
        return explicit_patient_id.strip().lower()

    baseline_path = baseline_dir / f"baseline_alpha_{alpha}"
    args_row = base._read_first_csv_row(baseline_path / "args.csv")
    if args_row is None:
        return None

    for key in ("pet_data_path", "spect_data_path", "output_path", "working_path"):
        patient_id = _infer_patient_id_from_text(args_row.get(key, ""))
        if patient_id:
            return patient_id
    return None


def _resolve_lesion_method_dir(
    lesion_masks_root: Path,
    patient_id: str,
    lesion_method: Optional[str],
) -> Tuple[Optional[Path], Optional[str]]:
    patient_dir = lesion_masks_root / patient_id
    if not patient_dir.exists():
        return None, None

    if lesion_method:
        method_dir = patient_dir / lesion_method
        if method_dir.exists():
            return method_dir, lesion_method
        return None, None

    for method in PREFERRED_LESION_METHODS:
        method_dir = patient_dir / method
        if method_dir.exists():
            return method_dir, method

    fallback_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    if fallback_dirs:
        return fallback_dirs[0], fallback_dirs[0].name
    return None, None


def _lesion_sort_key(lesion_name: str) -> Tuple[int, str]:
    match = re.match(r"^lesion_(\d+)$", lesion_name.lower())
    if not match:
        return 10**9, lesion_name
    return int(match.group(1)), lesion_name


def _discover_lesion_paths(lesion_method_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for lesion_path in lesion_method_dir.glob("lesion_*.hv"):
        out[lesion_path.stem] = lesion_path
    return dict(sorted(out.items(), key=lambda item: _lesion_sort_key(item[0])))


def _load_mask_hv(mask_path: Path, pet_shape: Tuple[int, ...], mask_name: str) -> np.ndarray:
    arr = ImageData(str(mask_path)).as_array() > 0.5
    if tuple(arr.shape) != tuple(pet_shape):
        raise ValueError(
            f"VOI mask shape mismatch for '{mask_name}': mask={arr.shape}, pet={pet_shape}, path={mask_path}"
        )
    return arr


def _build_background_mask(background_roi_path: Path, pet_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    if not background_roi_path.exists():
        return None

    with background_roi_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    if str(spec.get("type", "")).strip().lower() != "ellipsoid":
        raise ValueError(f"Unsupported background ROI type in {background_roi_path}: {spec.get('type')}")

    cx = float(spec["center_x"])
    cy = float(spec["center_y"])
    cz = float(spec["center_z"])
    rx = max(float(spec["radius_x"]), 1e-6)
    ry = max(float(spec["radius_y"]), 1e-6)
    rz = max(float(spec["radius_z"]), 1e-6)

    # ROI spec is in voxel coordinates (x, y, z) while arrays are indexed as (z, y, x).
    zz, yy, xx = np.indices(pet_shape, dtype=np.float64)
    background_mask = (
        ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2 <= 1.0
    )
    return background_mask.astype(bool)


def _default_patient_vois(lesion_method_dir: Path) -> List[str]:
    lesion_paths = _discover_lesion_paths(lesion_method_dir)
    voi_names = list(lesion_paths.keys())
    if lesion_paths:
        voi_names.append("lesion_union")
    voi_names.append("background")
    return voi_names


def load_patient_voi_masks(
    lesion_method_dir: Path,
    voi_names: Sequence[str],
    pet_shape: Tuple[int, ...],
    support_mask: Optional[np.ndarray] = None,
    clip_to_support: bool = True,
) -> Dict[str, np.ndarray]:
    lesion_paths = _discover_lesion_paths(lesion_method_dir)
    lesion_masks: Dict[str, np.ndarray] = {}
    for lesion_name, lesion_path in lesion_paths.items():
        lesion_masks[lesion_name] = _load_mask_hv(lesion_path, pet_shape, lesion_name)

    if lesion_masks:
        lesion_union = np.zeros(pet_shape, dtype=bool)
        for lesion_mask in lesion_masks.values():
            lesion_union |= lesion_mask
    else:
        lesion_union = None

    background_mask = _build_background_mask(lesion_method_dir.parent / "background_roi.json", pet_shape)

    if support_mask is not None and tuple(support_mask.shape) != tuple(pet_shape):
        raise ValueError(
            f"Support mask shape mismatch: support={support_mask.shape}, pet={pet_shape}"
        )

    out: Dict[str, np.ndarray] = {}
    for voi in voi_names:
        if voi in {"lesion", "lesion_union"}:
            if lesion_union is None:
                print(f"Warning: Could not build '{voi}' (no lesion_*.hv files found in {lesion_method_dir}).")
                continue
            mask = lesion_union
            source = "derived-lesion-union"
        elif voi == "background":
            if background_mask is None:
                print(f"Warning: Missing background ROI json in {lesion_method_dir.parent}")
                continue
            mask = background_mask
            source = "background-ellipsoid"
        elif voi in lesion_masks:
            mask = lesion_masks[voi]
            source = "lesion-file"
        else:
            print(f"Warning: Unsupported/unknown patient VOI '{voi}'")
            continue

        adjusted = mask.copy()
        if clip_to_support and support_mask is not None:
            adjusted = adjusted & support_mask
        out[voi] = adjusted
        print(
            f"Loaded patient VOI '{voi}' ({source}) with {int(mask.sum())} voxels "
            f"-> adjusted to {int(adjusted.sum())} voxels"
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute patient convergence-over-iterations with lesion/background VOIs."
    )
    parser.add_argument("--sweep", type=str, required=True, help="Sweep directory name, e.g. precond_2bpos")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline directory name, e.g. baselines_2bpos")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <sweep>_analysis_patient)",
    )
    parser.add_argument(
        "--lesion-masks-root",
        type=str,
        default=str(DEFAULT_LESION_MASKS_ROOT),
        help="Root containing per-patient lesion masks (sirt*/<method>/lesion_*.hv).",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional explicit patient id (e.g. sirt3). If omitted, inferred from baseline args.csv.",
    )
    parser.add_argument(
        "--lesion-method",
        type=str,
        default=None,
        help=(
            "Optional lesion mask method subfolder under patient dir "
            "(e.g. intensity, intensity_combined_recon)."
        ),
    )
    parser.add_argument(
        "--vois",
        nargs="+",
        default=None,
        help="Patient VOIs to evaluate. Defaults to lesion_*, lesion_union, background.",
    )
    parser.add_argument(
        "--spect2pet-transform",
        type=str,
        default=None,
        help="Optional explicit path to SPECT->PET no-zoom transform (nii/tfm).",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default=None,
        help="Comma-separated alpha filter. If omitted, uses parameters/alphas.csv when available.",
    )
    parser.add_argument(
        "--step-sizes",
        type=str,
        default=None,
        help="Comma-separated step-size filter (e.g. 1 or 1,0.5). If omitted, all step sizes are used.",
    )
    parser.add_argument(
        "--summary-percentiles",
        type=str,
        default="10,50,90",
        help="Percentiles for aggregated CSV summaries (default: 10,50,90).",
    )
    parser.add_argument(
        "--curve-inner-band",
        type=str,
        default="10,90",
        help="Inner percentile band for convergence plots (default: 10,90).",
    )
    parser.add_argument(
        "--no-whole-image",
        action="store_true",
        help="Disable whole-image metrics (default includes VOIs + whole image).",
    )
    parser.add_argument(
        "--include-bsrem",
        action="store_true",
        help="Include BSREM runs in analysis (default excludes BSREM).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of sweep runs processed (debugging).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional cap on per-run iteration index processed for convergence metrics.",
    )
    parser.add_argument(
        "--nrmse-threshold",
        type=float,
        default=None,
        help="Plot truncation threshold on NRMSE; CSV metrics still use all iterations (default: 1e-2).",
    )
    parser.add_argument(
        "--relative-error-threshold",
        type=float,
        default=None,
        help="Deprecated alias for --nrmse-threshold.",
    )
    parser.add_argument(
        "--initial-cap-factor",
        type=float,
        default=1.1,
        help="Upper y-limit cap as multiplier of initial mean value (default: 1.1, set <=0 to disable).",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate VOI + final image figures only (skip convergence CSV computation).",
    )
    parser.add_argument(
        "--no-final-images",
        action="store_true",
        help="Disable per-preconditioner final PET/SPECT pair figures.",
    )
    parser.add_argument(
        "--final-vmax-pet",
        type=float,
        default=0.002,
        help="vmax for PET panel in final pair figures (default: 0.002).",
    )
    parser.add_argument(
        "--final-vmax-spect",
        type=float,
        default=0.55,
        help="vmax for resampled SPECT panel in final pair figures (default: 0.55).",
    )
    parser.add_argument(
        "--final-fig-dpi",
        type=int,
        default=300,
        help="DPI for final pair figures (default: 300).",
    )
    parser.add_argument(
        "--mean-baseline-from",
        type=str,
        default=None,
        help="Use mean final image from the given precond_type instead of baseline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even when cached convergence/objective CSVs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.nrmse_threshold is not None:
        nrmse_threshold = float(args.nrmse_threshold)
        if args.relative_error_threshold is not None:
            print("Note: --relative-error-threshold ignored because --nrmse-threshold was provided.")
    elif args.relative_error_threshold is not None:
        nrmse_threshold = float(args.relative_error_threshold)
        print("Note: --relative-error-threshold is deprecated; use --nrmse-threshold instead.")
    else:
        nrmse_threshold = 1e-2

    summary_percentiles = base._parse_percentiles(args.summary_percentiles, base.DEFAULT_SUMMARY_PERCENTILES)
    curve_band = base._parse_percentiles(args.curve_inner_band, base.DEFAULT_CURVE_BAND)
    if len(curve_band) != 2:
        raise ValueError("--curve-inner-band must contain exactly two values, e.g. 10,90")
    curve_band_tuple = (curve_band[0], curve_band[1])
    include_whole_image = not args.no_whole_image

    study_dir = Path(__file__).resolve().parent.parent
    allowed_alphas = base._parse_alpha_values(args.alpha_values)
    if allowed_alphas is None:
        allowed_alphas = base._load_allowed_alphas_from_csv(study_dir / "parameters" / "alphas.csv")
    allowed_step_sizes = base._parse_step_sizes(args.step_sizes)

    base_output_dir = study_dir / "output"
    sweep_dir = base_output_dir / args.sweep
    baseline_dir = base_output_dir / args.baseline
    output_dir = Path(args.output) if args.output else base_output_dir / f"{args.sweep}_analysis_patient"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / "convergence_over_iterations_voi.csv"
    agg_csv = output_dir / "convergence_over_iterations_voi_aggregated.csv"
    objective_raw_csv = output_dir / "convergence_over_iterations_objective.csv"
    objective_agg_csv = output_dir / "convergence_over_iterations_objective_aggregated.csv"

    cached_metrics_available = all(
        path.exists() for path in (raw_csv, agg_csv, objective_raw_csv, objective_agg_csv)
    )
    allowed_vois_for_cache = list(args.vois) if args.vois else []
    if not args.figures_only and not args.force and cached_metrics_available:
        print("Using cached convergence/objective CSVs (pass --force to recompute).")
        all_rows = base._read_rows_csv(raw_csv)
        aggregated_rows = base._read_rows_csv(agg_csv)
        objective_rows = base._read_rows_csv(objective_raw_csv)
        aggregated_objective_rows = base._read_rows_csv(objective_agg_csv)

        all_rows = base._filter_cached_convergence_rows(
            all_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
            include_whole_image=include_whole_image,
            allowed_vois=allowed_vois_for_cache,
        )
        aggregated_rows = base._filter_cached_convergence_rows(
            aggregated_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
            include_whole_image=include_whole_image,
            allowed_vois=allowed_vois_for_cache,
        )
        objective_rows = base._filter_cached_objective_rows(
            objective_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
        )
        aggregated_objective_rows = base._filter_cached_objective_rows(
            aggregated_objective_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
        )

        if not aggregated_rows:
            print("No cached aggregated rows matched current filters.")
            print("Re-run with --force to regenerate filtered metrics.")
            return

        base.plot_convergence_bands(
            aggregated_rows=aggregated_rows,
            output_dir=output_dir,
            curve_band=curve_band_tuple,
            nrmse_threshold=nrmse_threshold,
            initial_cap_factor=float(args.initial_cap_factor),
        )
        if aggregated_objective_rows:
            base.plot_objective_bands(
                aggregated_objective_rows=aggregated_objective_rows,
                output_dir=output_dir,
                curve_band=curve_band_tuple,
                initial_cap_factor=float(args.initial_cap_factor),
            )
        else:
            print(f"Warning: cached objective aggregated CSV is empty: {objective_agg_csv}")

        print("=" * 70)
        print("Convergence plots refreshed from cached CSVs (after applying filters).")
        print(f"Raw rows: {raw_csv} ({len(all_rows)} rows)")
        print(f"Aggregated rows: {agg_csv} ({len(aggregated_rows)} rows)")
        print(f"Objective rows: {objective_raw_csv} ({len(objective_rows)} rows)")
        print(f"Objective aggregated rows: {objective_agg_csv} ({len(aggregated_objective_rows)} rows)")
        print(f"Plots + VOI overlays: {output_dir}")
        print("=" * 70)
        return

    lesion_masks_root = Path(args.lesion_masks_root)
    if not lesion_masks_root.exists():
        raise FileNotFoundError(f"Lesion mask root not found: {lesion_masks_root}")
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    print(f"Sweep directory: {sweep_dir}")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Lesion masks root: {lesion_masks_root}")
    print(f"Patient id override: {args.patient_id or '<auto>'}")
    print(f"Lesion method override: {args.lesion_method or '<auto>'}")
    if args.vois:
        print(f"VOIs override: {args.vois}")
    else:
        print("VOIs override: <auto>")
    print(f"Include BSREM: {bool(args.include_bsrem)}")

    run_dirs = base._gather_runs(sweep_dir)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    print(f"Found {len(run_dirs)} sweep run directories")

    baseline_cache: Dict[float, Optional[base.BaselineContext]] = {}
    masks_cache: Dict[float, Dict[str, np.ndarray]] = {}
    overlay_done_for_alpha: Dict[float, bool] = {}
    representative_runs: Dict[Tuple[float, str], Tuple[int, Path, Dict[str, str]]] = {}
    grouped_runs: Dict[Tuple[float, str], List[Tuple[Path, Dict[str, str]]]] = {}
    all_rows: List[Dict[str, object]] = []
    objective_rows: List[Dict[str, object]] = []
    vois_by_alpha: Dict[float, List[str]] = {}

    for idx, run_dir in enumerate(run_dirs, start=1):
        run_info = base._load_run_info(run_dir)
        if run_info is None:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: no result.csv")
            continue

        precond_type = str(run_info.get("precond_type", "")).strip().lower()
        if precond_type == "bsrem" and not args.include_bsrem:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: BSREM excluded by default")
            continue

        alpha = base._coerce_float(run_info.get("alpha"))
        if alpha is None:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: missing alpha")
            continue
        if allowed_alphas is not None and alpha not in set(allowed_alphas):
            continue

        step_size = base._coerce_float(run_info.get("step_size"))
        if allowed_step_sizes is not None and not base._float_matches(step_size, allowed_step_sizes):
            continue

        if alpha not in baseline_cache:
            baseline_cache[alpha] = base._build_baseline_context(
                baseline_dir=baseline_dir,
                alpha=alpha,
                explicit_transform=args.spect2pet_transform,
                sweep_dir=sweep_dir,
                mean_baseline_from=args.mean_baseline_from,
            )
            ctx = baseline_cache[alpha]
            if ctx is not None:
                print(f"alpha={alpha}: using transform {ctx.transform_path}")
                print(f"alpha={alpha}: baseline PET image {ctx.pet_final_path}")
                print(f"alpha={alpha}: baseline SPECT image {ctx.spect_final_path}")
                if ctx.pet_support_mask is not None:
                    print(f"alpha={alpha}: PET support voxels {int(ctx.pet_support_mask.sum())}")

        baseline_ctx = baseline_cache[alpha]
        if baseline_ctx is None:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: baseline context unavailable")
            continue

        if alpha not in masks_cache:
            patient_id = _infer_patient_id_for_alpha(baseline_dir, alpha, args.patient_id)
            if patient_id is None:
                print(f"Warning: could not infer patient id for alpha={alpha}; no VOI masks loaded.")
                masks_cache[alpha] = {}
                vois_by_alpha[alpha] = []
            else:
                lesion_method_dir, resolved_method = _resolve_lesion_method_dir(
                    lesion_masks_root=lesion_masks_root,
                    patient_id=patient_id,
                    lesion_method=args.lesion_method,
                )
                if lesion_method_dir is None or resolved_method is None:
                    print(
                        f"Warning: no lesion method directory found for patient={patient_id} "
                        f"(override={args.lesion_method})."
                    )
                    masks_cache[alpha] = {}
                    vois_by_alpha[alpha] = []
                else:
                    voi_names = list(args.vois) if args.vois else _default_patient_vois(lesion_method_dir)
                    print(
                        f"alpha={alpha}: patient={patient_id}, lesion_method={resolved_method}, "
                        f"VOIs={voi_names}"
                    )
                    masks_cache[alpha] = load_patient_voi_masks(
                        lesion_method_dir=lesion_method_dir,
                        voi_names=voi_names,
                        pet_shape=baseline_ctx.pet_final_array.shape,
                        support_mask=baseline_ctx.pet_support_mask,
                        clip_to_support=True,
                    )
                    vois_by_alpha[alpha] = voi_names
                    if not masks_cache[alpha]:
                        print(f"Warning: no patient VOI masks loaded for alpha={alpha}")

        voi_masks = masks_cache[alpha]
        if not voi_masks and not include_whole_image:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: no VOI masks loaded")
            continue

        if not overlay_done_for_alpha.get(alpha, False):
            pet_overlay_bg = (
                baseline_ctx.pet_umap_array
                if baseline_ctx.pet_umap_array is not None
                else baseline_ctx.pet_final_array
            )
            base.save_voi_overlay_image(
                output_dir=output_dir,
                alpha=alpha,
                pet_background=pet_overlay_bg,
                voi_masks=voi_masks,
                pet_reference_path=baseline_ctx.pet_final_path,
                umap_vmax=0.11,
                dpi=int(args.final_fig_dpi),
            )
            overlay_done_for_alpha[alpha] = True

        setting_id = run_info.get("setting_id", run_dir.name)
        repeat_id = base._coerce_float(run_info.get("repeat_id"))
        repeat_key = int(repeat_id) if repeat_id is not None else 10**9
        rep_key = (alpha, str(setting_id))
        existing = representative_runs.get(rep_key)
        if existing is None or repeat_key < existing[0]:
            representative_runs[rep_key] = (repeat_key, run_dir, run_info)
        grouped_runs.setdefault(rep_key, []).append((run_dir, run_info))

        if args.figures_only:
            print(f"[{idx}/{len(run_dirs)}] Registered {run_dir.name} for figure generation")
            continue

        run_rows = base.compute_convergence_rows_for_run(
            run_dir=run_dir,
            run_info=run_info,
            baseline_ctx=baseline_ctx,
            voi_masks=voi_masks,
            include_whole_image=include_whole_image,
            max_iterations=args.max_iterations,
        )
        run_objective_rows = base.compute_objective_rows_for_run(
            run_dir=run_dir,
            run_info=run_info,
            max_iterations=args.max_iterations,
        )
        all_rows.extend(run_rows)
        objective_rows.extend(run_objective_rows)
        print(
            f"[{idx}/{len(run_dirs)}] Processed {run_dir.name}: "
            f"{len(run_rows)} convergence rows, {len(run_objective_rows)} objective rows"
        )

    if not args.no_final_images:
        base.generate_final_images_per_preconditioner(
            output_dir=output_dir,
            baseline_cache=baseline_cache,
            representative_runs=representative_runs,
            pet_vmax=float(args.final_vmax_pet),
            spect_vmax=float(args.final_vmax_spect),
            dpi=int(args.final_fig_dpi),
        )
        base.generate_mean_final_and_difference_images_per_preconditioner(
            output_dir=output_dir,
            baseline_cache=baseline_cache,
            grouped_runs=grouped_runs,
            pet_vmax=float(args.final_vmax_pet),
            spect_vmax=float(args.final_vmax_spect),
            dpi=int(args.final_fig_dpi),
        )

    if args.figures_only:
        print("=" * 70)
        print("Figure generation complete (figures-only mode).")
        if vois_by_alpha:
            for alpha in sorted(vois_by_alpha):
                print(f"alpha={alpha}: VOIs={vois_by_alpha[alpha]}")
        print(f"Plots + VOI overlays + final pair figures: {output_dir}")
        print("=" * 70)
        return

    if not all_rows:
        print("No convergence rows were generated. Exiting.")
        return

    all_rows.sort(
        key=lambda r: (
            float(r["alpha"]) if r["alpha"] is not None else np.inf,
            str(r["precond_label"]),
            float(r["step_size"]) if r["step_size"] is not None else np.inf,
            str(r["modality"]),
            str(r["voi"]),
            int(r["iteration"]),
            str(r["run_dir"]),
        )
    )

    aggregated_rows = base.aggregate_rows(all_rows, summary_percentiles)
    aggregated_objective_rows = base.aggregate_objective_rows(objective_rows, summary_percentiles)

    base._write_rows_csv(raw_csv, all_rows)
    base._write_rows_csv(agg_csv, aggregated_rows)
    base._write_rows_csv(objective_raw_csv, objective_rows)
    base._write_rows_csv(objective_agg_csv, aggregated_objective_rows)

    base.plot_convergence_bands(
        aggregated_rows=aggregated_rows,
        output_dir=output_dir,
        curve_band=curve_band_tuple,
        nrmse_threshold=nrmse_threshold,
        initial_cap_factor=float(args.initial_cap_factor),
    )
    if aggregated_objective_rows:
        base.plot_objective_bands(
            aggregated_objective_rows=aggregated_objective_rows,
            output_dir=output_dir,
            curve_band=curve_band_tuple,
            initial_cap_factor=float(args.initial_cap_factor),
        )
    else:
        print("Warning: no objective rows were generated.")

    print("=" * 70)
    print("Patient convergence-over-iterations with VOIs complete.")
    print(f"Raw rows: {raw_csv}")
    print(f"Aggregated rows: {agg_csv}")
    print(f"Objective rows: {objective_raw_csv}")
    print(f"Objective aggregated rows: {objective_agg_csv}")
    print(f"Plots + VOI overlays: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
