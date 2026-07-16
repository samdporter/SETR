#!/usr/bin/env python3
"""
Compute convergence-over-iterations metrics with PET-space VOIs.

The reconstructions (both PET modality 0 and SPECT modality 1) are performed
directly on the PET grid, so the saved image_1_*.hv snapshots are already in
PET space. Metrics and figures therefore use these arrays directly -- we must
NOT re-apply the SPECT->PET transform, which would double-warp them.

For each sweep run:
1. Load PET and SPECT intermediate images per saved iteration (both already PET-space).
2. Compute RMSE and normalized RMSE (NRMSE) versus baseline final images inside PET-space VOIs.
3. Aggregate repeated runs with mean/min/max and percentile summaries.
4. Save convergence CSVs, convergence plots (with run envelopes), and a VOI overlay image.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
from recon_core.cil_extensions.operators import NiftyResampleOperator
from sirf.Reg import AffineTransformation, NiftiImageData3DDisplacement
from sirf.STIR import ImageData


DEFAULT_VOIS = (
    "hot_sphere",
    "hot_sphere_edge",
    "cold_sphere",
    "cold_sphere_edge",
    "cold",
    "liver",
)
DEFAULT_SUMMARY_PERCENTILES = (10, 50, 90)
DEFAULT_CURVE_BAND = (10, 90)
MODALITY_PET = "pet"
# SPECT is reconstructed on the PET grid; the label reflects PET-space SPECT
# (no resampling is applied). The constant name is retained for compatibility.
MODALITY_SPECT_RESAMPLED = "spect_pet"

# Preconditioners excluded from the analysis by default. mm_diag_block_maj has a
# zero SPECT preconditioner block, so its SPECT iterate never updates and its
# convergence curve is a meaningless flat line. Override via --exclude-precond.
EXCLUDED_PRECOND_TYPES = {"ls_block_gershgorin", "mm_diag_block_maj"}


def _is_excluded_precond(precond_type: object) -> bool:
    return str(precond_type or "").strip().lower() in EXCLUDED_PRECOND_TYPES
PRECOND_LABEL_MAP = {
    "mm_diag_block_maj": r"$\left(P_{\mathcal{D}}^{-1}+P_{\mathrm{blk,maj}}^{-1}\right)^{-1}$",
    "mm_diag_block_tight": r"$\left(P_{\mathcal{D}}^{-1}+P_{\mathrm{blk}}^{-1}\right)^{-1}$",
    "mm_diag_tight": r"$\left(P_{\mathcal{D}}^{-1}+P_{\mathrm{diag}}^{-1}\right)^{-1}$",
    "mm_diag_gershgorin_maj": r"$\left(P_{\mathcal{D}}^{-1}+P_{\mathrm{diag,maj}}^{-1}\right)^{-1}$",
    "ls_block_diag": r"$\left(\mathcal{P}_{D}^{-1}+\mathcal{P}_{\mathcal{LS}}^{-1}\right)^{-1}$",
    "bsrem": r"$P_{\mathcal{D}}$",
}


@dataclass
class BaselineContext:
    alpha: float
    pet_final_path: Path
    spect_final_path: Path
    pet_final_array: np.ndarray
    # PET-space SPECT baseline (image_1 is already on the PET grid; no resampling).
    spect_resampled_final_array: np.ndarray
    # Resampler/transform are only retained for the optional native-SPECT-umap
    # overlay; they are NOT used for metrics or the main figures.
    resampler: Optional[NiftyResampleOperator]
    transform_path: Optional[Path]
    baseline_dir: Path
    pet_support_mask: Optional[np.ndarray]
    pet_umap_array: Optional[np.ndarray]
    spect_registered_umap_array: Optional[np.ndarray]


def _coerce_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _float_matches(value: Optional[float], allowed_values: Tuple[float, ...], tol: float = 1e-9) -> bool:
    if value is None:
        return False
    return any(abs(value - allowed) <= tol for allowed in allowed_values)


def _normalise_combine(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text.lower() == "nan":
        return ""
    # Canonicalise accepted aliases/spellings.
    if text in {"harmonic", "majorise", "majorize"}:
        return "majoriser"
    return text


def _parse_repeat_metadata(result_dir_name: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.*)_rep_(\d+)$", result_dir_name)
    if match:
        return match.group(1), int(match.group(2))
    return result_dir_name, None


def _parse_subset_metadata(result_dir_name: str) -> Dict[str, str]:
    match = re.match(
        r"^subset_(?P<subset_mode>.+?)_prior_(?P<prior_mode>.+?)_precond_(?P<precond_type>.+?)_gamma_(?P<gamma_tnv>.+)$",
        result_dir_name,
    )
    if not match:
        return {}
    return match.groupdict()


RUN_METADATA_KEYS = (
    "subset_mode",
    "prior_mode",
    "gamma_tnv",
    "alpha_scaled",
    "beta_scaled",
)


def _run_metadata_fields(run_info: Dict[str, str]) -> Dict[str, object]:
    return {key: run_info.get(key, "") for key in RUN_METADATA_KEYS}


def _parse_percentiles(spec: Optional[str], default: Tuple[int, ...]) -> Tuple[int, ...]:
    if spec is None:
        return default
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        val = int(token)
        if val < 0 or val > 100:
            raise ValueError(f"Percentile out of range [0, 100]: {val}")
        out.append(val)
    if not out:
        return default
    return tuple(sorted(set(out)))


def _parse_alpha_values(spec: Optional[str]) -> Optional[Tuple[float, ...]]:
    if spec is None:
        return None
    values: List[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        return None
    return tuple(sorted(set(values)))


def _parse_step_sizes(spec: Optional[str]) -> Optional[Tuple[float, ...]]:
    if spec is None:
        return None
    values: List[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        return None
    return tuple(sorted(set(values)))


def _load_allowed_alphas_from_csv(csv_path: Path) -> Optional[Tuple[float, ...]]:
    if not csv_path.exists():
        return None
    rows = _read_csv_rows(csv_path)
    if not rows:
        return None
    header = list(rows[0].keys())
    alpha_key = "alpha" if "alpha" in header else header[0]
    out: List[float] = []
    for row in rows:
        val = _coerce_float(row.get(alpha_key))
        if val is not None:
            out.append(val)
    if not out:
        return None
    return tuple(sorted(set(out)))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_first_csv_row(path: Path) -> Optional[Dict[str, str]]:
    rows = _read_csv_rows(path)
    if not rows:
        return None
    return rows[0]


def _safe_percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def _apply_binary_morphology(mask: np.ndarray, op: str, iterations: int, connectivity: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scipy.ndimage is required for VOI morphology controls "
            "(--cold-erode-iters)."
        ) from exc

    conn = int(np.clip(connectivity, 1, 3))
    structure = generate_binary_structure(rank=3, connectivity=conn)
    if op == "erode":
        return binary_erosion(mask, structure=structure, iterations=iterations).astype(bool)
    if op == "dilate":
        return binary_dilation(mask, structure=structure, iterations=iterations).astype(bool)
    raise ValueError(f"Unsupported morphology op: {op}")


def _keep_central_connected_component(mask: np.ndarray, connectivity: int) -> np.ndarray:
    if not np.any(mask):
        return mask
    try:
        from scipy.ndimage import generate_binary_structure, label
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy.ndimage is required for cold VOI connected-component filtering.") from exc

    conn = int(np.clip(connectivity, 1, 3))
    structure = generate_binary_structure(rank=3, connectivity=conn)
    labels, num_labels = label(mask, structure=structure)
    if num_labels <= 1:
        return mask

    counts = np.bincount(labels.ravel())
    if counts.size <= 1:
        return mask
    counts[0] = 0
    target = int(np.argmax(counts))
    if target <= 0:
        return mask

    return labels == target


def _resolve_mask_dir(study_dir: Path, explicit_mask_dir: Optional[str]) -> Optional[Path]:
    if explicit_mask_dir:
        p = Path(explicit_mask_dir)
        return p if p.exists() else None

    candidates = [
        Path("/home/sam/working/synergistic_recon/recon_experiments/src/recon_experiments/studies/preconditioners"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_mask(
    mask_dir: Path,
    voi_name: str,
    mask_overrides: Optional[Dict[str, Path]] = None,
) -> Optional[np.ndarray]:
    if mask_overrides and voi_name in mask_overrides:
        override = mask_overrides[voi_name]
        if override.exists():
            arr = ImageData(str(override)).as_array()
            return arr > 0.5
        raise FileNotFoundError(f"Mask override for {voi_name} does not exist: {override}")

    aliases: Dict[str, Tuple[str, ...]] = {
        "hot_sphere": ("hot_sphere",),
        "cold": ("cold",),
        "cold_sphere": ("cold_sphere",),
        "liver": ("liver_zoomed", "Liver_zoomed"),
    }
    keys = aliases.get(voi_name, (voi_name,))
    candidates: List[Path] = []
    for key in keys:
        candidates.append(mask_dir / f"{key}_pet.hv")
        candidates.append(mask_dir / f"{key}.hv")
    for c in candidates:
        if c.exists():
            arr = ImageData(str(c)).as_array()
            return arr > 0.5
    return None


def _load_pet_support_mask_from_umap(
    baseline_args_row: Optional[Dict[str, str]],
    pet_shape: Tuple[int, ...],
) -> Optional[np.ndarray]:
    if baseline_args_row is None:
        return None

    pet_data_path = str(baseline_args_row.get("pet_data_path", "")).strip()
    if not pet_data_path:
        return None

    pet_data_dir = Path(pet_data_path)
    candidates = (
        pet_data_dir / "umap_zoomed.hv",
        pet_data_dir / "umap.hv",
    )
    for c in candidates:
        if not c.exists():
            continue
        arr = ImageData(str(c)).as_array()
        if tuple(arr.shape) != tuple(pet_shape):
            print(
                f"Warning: umap support shape mismatch for {c}: "
                f"{arr.shape} vs PET {pet_shape}"
            )
            continue
        support = arr > 0.01
        support = _keep_central_connected_component(support, connectivity=1)
        print(f"Loaded PET support from {c} (umap > 0.01) with {int(support.sum())} voxels")
        return support

    print(f"Warning: could not find umap(.hv) or umap_zoomed(.hv) in {pet_data_dir}")
    return None


def _load_pet_umap_for_overlay(
    baseline_args_row: Optional[Dict[str, str]],
    pet_shape: Tuple[int, ...],
    fallback_dirs: Sequence[Path] = (),
) -> Optional[np.ndarray]:
    # Search the recon's own pet_data_path first, then any fallback dirs
    # (e.g. the mask/study dir) so overlays still render when the recorded
    # cluster path is not reachable locally.
    search_dirs: List[Path] = []
    if baseline_args_row is not None:
        pet_data_path = str(baseline_args_row.get("pet_data_path", "")).strip()
        if pet_data_path:
            search_dirs.append(Path(pet_data_path))
    search_dirs.extend(Path(d) for d in fallback_dirs)

    for base in search_dirs:
        for c in (base / "umap_zoomed.hv", base / "umap.hv"):
            if not c.exists():
                continue
            arr = ImageData(str(c)).as_array()
            if tuple(arr.shape) != tuple(pet_shape):
                continue
            print(f"Loaded PET umap overlay from {c}")
            return arr.astype(np.float64, copy=False)
    return None


def _load_spect_registered_umap_for_overlay(
    baseline_args_row: Optional[Dict[str, str]],
    pet_shape: Tuple[int, ...],
    resampler: NiftyResampleOperator,
) -> Optional[np.ndarray]:
    if baseline_args_row is None:
        return None

    spect_data_path = str(baseline_args_row.get("spect_data_path", "")).strip()
    if not spect_data_path:
        return None

    spect_data_dir = Path(spect_data_path)
    registered_candidates = (
        spect_data_dir / "umap_zoomed_registered.hv",
        spect_data_dir / "umap_registered.hv",
    )
    for c in registered_candidates:
        if not c.exists():
            continue
        arr = ImageData(str(c)).as_array()
        if tuple(arr.shape) != tuple(pet_shape):
            continue
        print(f"Loaded registered SPECT umap overlay from {c}")
        return arr.astype(np.float64, copy=False)

    raw_candidates = (
        spect_data_dir / "umap_zoomed.hv",
        spect_data_dir / "umap.hv",
    )
    for c in raw_candidates:
        if not c.exists():
            continue
        try:
            arr = resampler.direct(ImageData(str(c))).as_array()
        except Exception:
            continue
        if tuple(arr.shape) != tuple(pet_shape):
            continue
        print(f"Loaded SPECT umap overlay by resampling {c} to PET space")
        return arr.astype(np.float64, copy=False)

    return None


def load_pet_voi_masks(
    mask_dir: Path,
    voi_names: Sequence[str],
    pet_shape: Tuple[int, ...],
    cold_erode_iters: int = 3,
    cold_liver_dilate_iters: int = 5,
    cold_bottom_clearance_voxels: int = 10,
    liver_sphere_gap_iters: int = 3,
    morph_connectivity: int = 1,
    support_mask: Optional[np.ndarray] = None,
    clip_to_support: bool = True,
    mask_overrides: Optional[Dict[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    voi_set = set(voi_names)

    hot_sphere_mask: Optional[np.ndarray] = None
    cold_sphere_mask: Optional[np.ndarray] = None
    liver_mask: Optional[np.ndarray] = None

    if {"hot_sphere", "hot_sphere_edge", "liver"} & voi_set:
        hot_sphere_mask = _read_mask(mask_dir, "hot_sphere", mask_overrides=mask_overrides)
        if hot_sphere_mask is not None and tuple(hot_sphere_mask.shape) != tuple(pet_shape):
            raise ValueError(
                f"VOI mask shape mismatch for 'hot_sphere': mask={hot_sphere_mask.shape}, pet={pet_shape}"
            )

    if {"cold_sphere", "cold_sphere_edge", "liver"} & voi_set:
        cold_sphere_mask = _read_mask(mask_dir, "cold_sphere", mask_overrides=mask_overrides)
        if cold_sphere_mask is not None and tuple(cold_sphere_mask.shape) != tuple(pet_shape):
            raise ValueError(
                f"VOI mask shape mismatch for 'cold_sphere': mask={cold_sphere_mask.shape}, pet={pet_shape}"
            )

    if {"cold", "liver", "hot_sphere_edge", "cold_sphere_edge"} & voi_set:
        liver_mask = _read_mask(mask_dir, "liver", mask_overrides=mask_overrides)
        if liver_mask is not None and tuple(liver_mask.shape) != tuple(pet_shape):
            raise ValueError(
                f"VOI mask shape mismatch for 'liver': mask={liver_mask.shape}, pet={pet_shape}"
            )

    liver_for_cold: Optional[np.ndarray] = None
    if "cold" in voi_set:
        liver_for_cold = liver_mask

    spheres_for_liver: List[np.ndarray] = []
    if "liver" in voi_set:
        for sphere in (hot_sphere_mask, cold_sphere_mask):
            if sphere is not None:
                spheres_for_liver.append(sphere)

    hot_exclusion: Optional[np.ndarray] = None
    cold_exclusion: Optional[np.ndarray] = None
    if hot_sphere_mask is not None:
        hot_exclusion = hot_sphere_mask
        if liver_sphere_gap_iters > 0:
            hot_exclusion = _apply_binary_morphology(
                hot_exclusion,
                op="dilate",
                iterations=liver_sphere_gap_iters,
                connectivity=morph_connectivity,
            )
    if cold_sphere_mask is not None:
        cold_exclusion = cold_sphere_mask
        if liver_sphere_gap_iters > 0:
            cold_exclusion = _apply_binary_morphology(
                cold_exclusion,
                op="dilate",
                iterations=liver_sphere_gap_iters,
                connectivity=morph_connectivity,
            )

    for voi in voi_names:
        built_from_umap = voi == "cold" and support_mask is not None
        if voi == "hot_sphere":
            mask = hot_sphere_mask
            if mask is None:
                print(f"Warning: Missing PET VOI mask for '{voi}' in {mask_dir}")
                continue
        elif voi == "cold_sphere":
            mask = cold_sphere_mask
            if mask is None:
                print(f"Warning: Missing PET VOI mask for '{voi}' in {mask_dir}")
                continue
        elif voi == "hot_sphere_edge":
            if liver_mask is None or hot_sphere_mask is None or hot_exclusion is None:
                print(
                    "Warning: Could not build 'hot_sphere_edge' (requires liver + hot_sphere masks)."
                )
                continue
            mask = liver_mask & hot_exclusion & ~hot_sphere_mask
            if cold_exclusion is not None:
                mask = mask & ~cold_exclusion
        elif voi == "cold_sphere_edge":
            if liver_mask is None or cold_sphere_mask is None or cold_exclusion is None:
                print(
                    "Warning: Could not build 'cold_sphere_edge' (requires liver + cold_sphere masks)."
                )
                continue
            mask = liver_mask & cold_exclusion & ~cold_sphere_mask
            if hot_exclusion is not None:
                mask = mask & ~hot_exclusion
        elif built_from_umap:
            mask = support_mask.copy()
        else:
            mask = _read_mask(mask_dir, voi, mask_overrides=mask_overrides)
            if mask is None:
                print(f"Warning: Missing PET VOI mask for '{voi}' in {mask_dir}")
                continue
            if tuple(mask.shape) != tuple(pet_shape):
                raise ValueError(
                    f"VOI mask shape mismatch for '{voi}': mask={mask.shape}, pet={pet_shape}"
                )

        adjusted = mask.copy()
        if voi == "cold":
            if cold_erode_iters > 0:
                adjusted = _apply_binary_morphology(
                    adjusted,
                    op="erode",
                    iterations=cold_erode_iters,
                    connectivity=morph_connectivity,
                )
            if liver_for_cold is not None:
                liver_exclusion = liver_for_cold
                if cold_liver_dilate_iters > 0:
                    liver_exclusion = _apply_binary_morphology(
                        liver_exclusion,
                        op="dilate",
                        iterations=cold_liver_dilate_iters,
                        connectivity=morph_connectivity,
                    )
                adjusted = adjusted & ~liver_exclusion
            if cold_bottom_clearance_voxels > 0:
                z_keep = max(0, int(adjusted.shape[0]) - int(cold_bottom_clearance_voxels))
                adjusted[z_keep:, :, :] = False
            adjusted = _keep_central_connected_component(adjusted, connectivity=morph_connectivity)
        elif voi == "liver" and spheres_for_liver:
            for sphere in spheres_for_liver:
                exclusion = sphere
                if liver_sphere_gap_iters > 0:
                    exclusion = _apply_binary_morphology(
                        exclusion,
                        op="dilate",
                        iterations=liver_sphere_gap_iters,
                        connectivity=morph_connectivity,
                    )
                adjusted = adjusted & ~exclusion

        if clip_to_support and support_mask is not None:
            adjusted = adjusted & support_mask

        masks[voi] = adjusted
        if voi in {"hot_sphere_edge", "cold_sphere_edge"}:
            source = "derived-gap-mask"
        else:
            source = "umap-support" if built_from_umap else "file-mask"
        print(
            f"Loaded VOI '{voi}' ({source}) with {int(mask.sum())} voxels "
            f"-> adjusted to {int(adjusted.sum())} voxels"
        )
    return masks


def _parse_iteration_from_name(filename: str, modality: int) -> Optional[int]:
    match = re.match(rf"image_{modality}_(\d+)\.hv$", filename)
    if not match:
        return None
    return int(match.group(1))


def get_iteration_images(result_dir: Path, modality: int) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for img_path in result_dir.glob(f"image_{modality}_*.hv"):
        it = _parse_iteration_from_name(img_path.name, modality)
        if it is not None:
            out[it] = img_path
    return out


def _find_final_image_path(run_dir: Path, modality: int) -> Optional[Path]:
    direct_final = run_dir / f"final_image_{modality}.hv"
    if direct_final.exists():
        return direct_final
    iter_images = get_iteration_images(run_dir, modality)
    if not iter_images:
        return None
    return iter_images[max(iter_images.keys())]


def _find_highest_iteration_image_path(run_dir: Path, modality: int) -> Optional[Path]:
    """Return highest-iteration image_<modality>_<iter>.hv if available."""
    iter_images = get_iteration_images(run_dir, modality)
    if not iter_images:
        return None
    return iter_images[max(iter_images.keys())]


def _resolve_nozoom_transform(
    explicit_transform: Optional[str],
    baseline_args_row: Optional[Dict[str, str]],
) -> Optional[Path]:
    if explicit_transform:
        p = Path(explicit_transform)
        if p.exists():
            return p
        raise FileNotFoundError(f"--spect2pet-transform does not exist: {p}")

    spect_data_path: Optional[str] = None
    if baseline_args_row is not None:
        spect_data_path = baseline_args_row.get("spect_data_path")
    if not spect_data_path:
        return None

    base = Path(spect_data_path)
    candidates = [
        "spect2pet_nozoom_nonrigid.nii",
        "spect2pet_nozoom_rigid.nii",
        "spect2pet_nozoom.nii",
        "spect2pet_nozoom_nonrigid.tfm",
        "spect2pet_nozoom_rigid.tfm",
        "spect2pet_nozoom.tfm",
    ]
    for name in candidates:
        fp = base / name
        if fp.exists():
            return fp
    return None


def _load_transform(path: Path):
    suffix = path.suffix.lower()
    if suffix in (".nii", ".gz"):
        return NiftiImageData3DDisplacement(str(path))
    return AffineTransformation(str(path))


def _build_resampler(reference_pet_img: ImageData, floating_spect_img: ImageData, transform_path: Path) -> NiftyResampleOperator:
    transform = _load_transform(transform_path)
    return NiftyResampleOperator(
        reference=reference_pet_img,
        floating=floating_spect_img,
        transform=transform,
    )


def _collect_mean_images_from_sweep(
    sweep_dir: Path,
    alpha: float,
    precond_key: str,
    pet_shape: Tuple[int, ...],
    resampler: NiftyResampleOperator,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Path], Optional[Path]]:
    pet_arrays: List[np.ndarray] = []
    pet_ref: Optional[Path] = None
    spect_arrays: List[np.ndarray] = []
    spect_ref: Optional[Path] = None

    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_info = _load_run_info(run_dir)
        if run_info is None:
            continue
        if run_info.get("precond_type") != precond_key:
            continue
        run_alpha = _coerce_float(run_info.get("alpha"))
        if run_alpha is None or abs(run_alpha - alpha) > 1e-9:
            continue

        pet_path = _find_highest_iteration_image_path(run_dir, modality=0) or _find_final_image_path(run_dir, modality=0)
        spect_path = _find_highest_iteration_image_path(run_dir, modality=1) or _find_final_image_path(run_dir, modality=1)
        if not pet_path or not spect_path:
            continue

        pet_arr = ImageData(str(pet_path)).as_array()
        if pet_arr.shape != pet_shape:
            continue
        pet_arrays.append(pet_arr.astype(np.float64))
        pet_ref = pet_ref or pet_path

        # image_1 is already reconstructed on the PET grid; use it directly.
        spect_arr = ImageData(str(spect_path)).as_array()
        if spect_arr.shape != pet_shape:
            continue
        spect_arrays.append(spect_arr.astype(np.float64))
        spect_ref = spect_ref or spect_path

    if not pet_arrays:
        return None, None, None, None

    pet_mean = np.mean(np.stack(pet_arrays, axis=0), axis=0)
    spect_mean = np.mean(np.stack(spect_arrays, axis=0), axis=0) if spect_arrays else None
    return pet_mean, spect_mean, pet_ref, spect_ref


def _compute_metrics(test_arr: np.ndarray, baseline_arr: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[float, float, int]:
    if mask is None:
        test = test_arr.ravel()
        base = baseline_arr.ravel()
    else:
        test = test_arr[mask]
        base = baseline_arr[mask]
    if test.size == 0:
        return np.nan, np.nan, 0
    diff = test - base
    rmse = float(np.sqrt(np.mean(diff**2)))
    base_rms = float(np.sqrt(np.mean(base**2)))
    nrmse = float(rmse / base_rms) if base_rms > 0 else np.inf
    return rmse, nrmse, int(test.size)


def _build_baseline_context(
    baseline_dir: Path,
    alpha: float,
    explicit_transform: Optional[str],
    sweep_dir: Optional[Path] = None,
    mean_baseline_from: Optional[str] = None,
    umap_fallback_dirs: Sequence[Path] = (),
) -> Optional[BaselineContext]:
    baseline_path = baseline_dir / f"baseline_alpha_{alpha}"
    if not baseline_path.exists():
        print(f"Warning: baseline directory not found for alpha={alpha}: {baseline_path}")
        return None

    # Baseline reference should use the highest available iteration snapshot.
    # Fall back to final_image_* only if iteration snapshots are unavailable.
    pet_mean_arr = None
    spect_mean_arr = None
    pet_mean_ref = None
    spect_mean_ref = None

    pet_final = pet_mean_ref or _find_highest_iteration_image_path(baseline_path, modality=0)
    if pet_final is None:
        pet_final = _find_final_image_path(baseline_path, modality=0)
    spect_final = spect_mean_ref or _find_highest_iteration_image_path(baseline_path, modality=1)
    if spect_final is None:
        spect_final = _find_final_image_path(baseline_path, modality=1)
    if pet_final is None or spect_final is None:
        print(f"Warning: missing baseline final images for alpha={alpha}")
        return None

    args_row = _read_first_csv_row(baseline_path / "args.csv")

    pet_final_img = ImageData(str(pet_final))
    spect_final_img = ImageData(str(spect_final))
    pet_shape = pet_final_img.as_array().shape
    pet_support_mask = _load_pet_support_mask_from_umap(args_row, pet_shape)

    # The SPECT->PET transform is NOT needed for metrics: image_1 is already on
    # the PET grid. We only build a resampler (if a transform is available) for
    # the optional native-SPECT-umap display overlay; its absence is non-fatal.
    transform_path = _resolve_nozoom_transform(explicit_transform, args_row)
    resampler = (
        _build_resampler(pet_final_img, spect_final_img, transform_path)
        if transform_path is not None
        else None
    )
    pet_umap_array = _load_pet_umap_for_overlay(args_row, pet_shape, umap_fallback_dirs)
    spect_registered_umap_array = (
        _load_spect_registered_umap_for_overlay(args_row, pet_shape, resampler)
        if resampler is not None
        else None
    )
    pet_ref_path = pet_final
    spect_ref_path = spect_final

    if sweep_dir and mean_baseline_from:
        pet_mean_arr, spect_mean_arr, pet_mean_ref, spect_mean_ref = _collect_mean_images_from_sweep(
            sweep_dir,
            alpha,
            mean_baseline_from,
            pet_shape,
            resampler,
        )
        if pet_mean_arr is not None:
            pet_arr = pet_mean_arr
            if pet_mean_ref is not None:
                pet_ref_path = pet_mean_ref
        else:
            pet_arr = pet_final_img.as_array()
        if spect_mean_arr is not None and spect_mean_arr.shape == pet_shape:
            spect_res_arr = spect_mean_arr
            if spect_mean_ref is not None:
                spect_ref_path = spect_mean_ref
        else:
            # image_1 already PET-space; use directly (no resampling).
            spect_res_arr = spect_final_img.as_array()
    else:
        pet_arr = pet_final_img.as_array()
        # image_1 already PET-space; use directly (no resampling).
        spect_res_arr = spect_final_img.as_array()

    return BaselineContext(
        alpha=alpha,
        pet_final_path=pet_ref_path,
        spect_final_path=spect_ref_path,
        pet_final_array=pet_arr,
        spect_resampled_final_array=spect_res_arr,
        resampler=resampler,
        transform_path=transform_path,
        baseline_dir=baseline_path,
        pet_support_mask=pet_support_mask,
        pet_umap_array=pet_umap_array,
        spect_registered_umap_array=spect_registered_umap_array,
    )


def _gather_runs(sweep_dir: Path) -> List[Path]:
    return sorted(
        [
            d
            for d in sweep_dir.iterdir()
            if d.is_dir() and (d.name.startswith("precond_") or d.name.startswith("subset_"))
        ]
    )


def _load_run_info(run_dir: Path) -> Optional[Dict[str, str]]:
    row = _read_first_csv_row(run_dir / "result.csv")
    if row is None:
        row = _read_first_csv_row(run_dir / "args.csv")
        if row is None:
            return None
        row.update({k: v for k, v in _parse_subset_metadata(run_dir.name).items() if v})

        # Subset-selection args.csv stores scaled objective weights in alpha/beta.
        # Baseline directories are keyed by the sweep/unscaled alpha value.
        row["alpha_scaled"] = row.get("alpha", "")
        row["beta_scaled"] = row.get("beta", "")
        row["alpha"] = row.get("alpha_initial") or row.get("alpha", "")
        row["beta"] = row.get("beta_initial") or row.get("beta", "")
        row.setdefault("step_size", row.get("initial_step_size", ""))
        row.setdefault("precond_combine", row.get("precond_combine", ""))

    precond_type = row.get("precond_type", "unknown")
    precond_combine = _normalise_combine(row.get("precond_combine", "")) or _normalise_combine(row.get("combine", ""))
    precond_label = f"{precond_type}:{precond_combine}" if precond_combine else precond_type
    subset_mode = str(row.get("subset_mode", "")).strip()
    prior_mode = str(row.get("prior_mode", "")).strip()
    if subset_mode or prior_mode:
        subset_label = "+".join(part for part in (subset_mode, prior_mode) if part)
        precond_label = f"{subset_label}:{precond_type}"
    setting_id, repeat_id = _parse_repeat_metadata(run_dir.name)
    row["precond_combine"] = precond_combine
    row["precond_label"] = precond_label
    row["setting_id"] = setting_id
    row["repeat_id"] = "" if repeat_id is None else str(repeat_id)
    return row


def compute_convergence_rows_for_run(
    run_dir: Path,
    run_info: Dict[str, str],
    baseline_ctx: BaselineContext,
    voi_masks: Dict[str, np.ndarray],
    include_whole_image: bool,
    max_iterations: Optional[int] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    pet_images = get_iteration_images(run_dir, modality=0)
    spect_images = get_iteration_images(run_dir, modality=1)
    common_iterations = sorted(set(pet_images.keys()) & set(spect_images.keys()))
    if not common_iterations:
        print(f"Warning: no common PET/SPECT iterations in {run_dir.name}")
        return rows

    for it in common_iterations:
        if max_iterations is not None and it > max_iterations:
            break

        # Defensively skip snapshots whose Interfile binary is unreadable
        # (e.g. a truncated *.v from an interrupted sync) rather than aborting
        # the whole analysis on one bad file.
        try:
            pet_arr = ImageData(str(pet_images[it])).as_array()
            # image_1 is already reconstructed on the PET grid: use directly, do
            # NOT re-apply the SPECT->PET transform (that would double-warp it).
            spect_res_arr = ImageData(str(spect_images[it])).as_array()
        except Exception as exc:  # pragma: no cover - depends on filesystem state
            print(
                f"Warning: skipping unreadable snapshot iter {it} in "
                f"{run_dir.name}: {exc}"
            )
            continue

        modality_arrays = {
            MODALITY_PET: (pet_arr, baseline_ctx.pet_final_array),
            MODALITY_SPECT_RESAMPLED: (spect_res_arr, baseline_ctx.spect_resampled_final_array),
        }

        for modality_name, (test_arr, baseline_arr) in modality_arrays.items():
            for voi_name, voi_mask in voi_masks.items():
                rmse, nrmse, nvox = _compute_metrics(test_arr, baseline_arr, voi_mask)
                rows.append(
                    {
                        "run_dir": run_dir.name,
                        "setting_id": run_info["setting_id"],
                        "repeat_id": _coerce_float(run_info.get("repeat_id")),
                        "precond_type": run_info.get("precond_type", "unknown"),
                        "precond_combine": run_info.get("precond_combine", ""),
                        "precond_label": run_info.get("precond_label", "unknown"),
                        **_run_metadata_fields(run_info),
                        "alpha": _coerce_float(run_info.get("alpha")),
                        "step_size": _coerce_float(run_info.get("step_size")),
                        "iteration": int(it),
                        "modality": modality_name,
                        "voi": voi_name,
                        "rmse": rmse,
                        "nrmse": nrmse,
                        # Backward compatibility for legacy consumers.
                        "relative_error": nrmse,
                        "num_voxels": nvox,
                    }
                )

            if include_whole_image:
                rmse, nrmse, nvox = _compute_metrics(test_arr, baseline_arr, None)
                rows.append(
                    {
                        "run_dir": run_dir.name,
                        "setting_id": run_info["setting_id"],
                        "repeat_id": _coerce_float(run_info.get("repeat_id")),
                        "precond_type": run_info.get("precond_type", "unknown"),
                        "precond_combine": run_info.get("precond_combine", ""),
                        "precond_label": run_info.get("precond_label", "unknown"),
                        **_run_metadata_fields(run_info),
                        "alpha": _coerce_float(run_info.get("alpha")),
                        "step_size": _coerce_float(run_info.get("step_size")),
                        "iteration": int(it),
                        "modality": modality_name,
                        "voi": "whole_image",
                        "rmse": rmse,
                        "nrmse": nrmse,
                        # Backward compatibility for legacy consumers.
                        "relative_error": nrmse,
                        "num_voxels": nvox,
                    }
                )

    return rows


def _read_objective_history_with_iterations(obj_path: Path) -> List[Tuple[int, float]]:
    if not obj_path.exists():
        return []

    out: List[Tuple[int, float]] = []
    with obj_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    if not rows:
        return out

    # SaveObjectiveCallback writes a single unnamed column via pandas with
    # header "0". In that common case, treat rows as objective values and
    # assign sequential iteration indices.
    if all(len(row) == 1 for row in rows):
        start_idx = 0
        first_token = rows[0][0].strip().lower() if rows and rows[0] else ""
        if first_token in ("0", "0.0", "objective"):
            start_idx = 1

        for row in rows[start_idx:]:
            token = row[0].strip() if row and row[0] is not None else ""
            val = _coerce_float(token)
            if val is None:
                continue
            out.append((len(out), float(val)))
        return out

    fallback_it = 0
    for row in rows:
        # Pandas default CSV header for index+single unnamed column: ",0"
        if len(row) >= 2 and row[0].strip() == "" and row[1].strip() in ("0", "0.0"):
            continue

        val = _coerce_float(row[-1].strip() if row[-1] is not None else None)
        if val is None:
            continue

        it = _coerce_float(row[0].strip()) if row and row[0] is not None else None
        if it is None:
            it_idx = fallback_it
        else:
            it_idx = int(it)
        fallback_it = max(fallback_it + 1, it_idx + 1)
        out.append((it_idx, float(val)))
    return out


def compute_objective_rows_for_run(
    run_dir: Path,
    run_info: Dict[str, str],
    max_iterations: Optional[int] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    objective_hist = _read_objective_history_with_iterations(run_dir / "objective.csv")
    for it, obj in objective_hist:
        if max_iterations is not None and it > max_iterations:
            break
        rows.append(
            {
                "run_dir": run_dir.name,
                "setting_id": run_info["setting_id"],
                "repeat_id": _coerce_float(run_info.get("repeat_id")),
                "precond_type": run_info.get("precond_type", "unknown"),
                "precond_combine": run_info.get("precond_combine", ""),
                "precond_label": run_info.get("precond_label", "unknown"),
                **_run_metadata_fields(run_info),
                "alpha": _coerce_float(run_info.get("alpha")),
                "step_size": _coerce_float(run_info.get("step_size")),
                "iteration": int(it),
                "objective": float(obj),
            }
        )
    return rows


def aggregate_rows(
    rows: List[Dict[str, object]],
    percentiles: Tuple[int, ...],
) -> List[Dict[str, object]]:
    if not rows:
        return []

    group_keys = [
        "setting_id",
        "precond_type",
        "precond_combine",
        "precond_label",
        "subset_mode",
        "prior_mode",
        "gamma_tnv",
        "alpha",
        "step_size",
        "iteration",
        "modality",
        "voi",
    ]

    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        buckets.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, group in buckets.items():
        row_out: Dict[str, object] = dict(zip(group_keys, key))
        row_out["num_runs"] = len(group)

        for metric in ("rmse", "nrmse", "relative_error"):
            vals = [float(g[metric]) for g in group if g.get(metric) is not None and np.isfinite(g[metric])]
            if not vals:
                row_out[f"{metric}_mean"] = np.nan
                row_out[f"{metric}_min"] = np.nan
                row_out[f"{metric}_max"] = np.nan
                for p in percentiles:
                    row_out[f"{metric}_p{p}"] = np.nan
                continue
            row_out[f"{metric}_mean"] = float(np.mean(vals))
            row_out[f"{metric}_min"] = float(np.min(vals))
            row_out[f"{metric}_max"] = float(np.max(vals))
            for p in percentiles:
                row_out[f"{metric}_p{p}"] = _safe_percentile(vals, p)

        out.append(row_out)

    out.sort(
        key=lambda r: (
            float(r["alpha"]) if r["alpha"] is not None else np.inf,
            str(r["precond_label"]),
            float(r["step_size"]) if r["step_size"] is not None else np.inf,
            str(r["modality"]),
            str(r["voi"]),
            int(r["iteration"]),
        )
    )
    return out


def aggregate_objective_rows(
    rows: List[Dict[str, object]],
    percentiles: Tuple[int, ...],
) -> List[Dict[str, object]]:
    if not rows:
        return []

    group_keys = [
        "setting_id",
        "precond_type",
        "precond_combine",
        "precond_label",
        "subset_mode",
        "prior_mode",
        "gamma_tnv",
        "alpha",
        "step_size",
        "iteration",
    ]

    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        buckets.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, group in buckets.items():
        row_out: Dict[str, object] = dict(zip(group_keys, key))
        row_out["num_runs"] = len(group)
        vals = [float(g["objective"]) for g in group if g.get("objective") is not None and np.isfinite(g["objective"])]
        if not vals:
            row_out["objective_mean"] = np.nan
            row_out["objective_min"] = np.nan
            row_out["objective_max"] = np.nan
            for p in percentiles:
                row_out[f"objective_p{p}"] = np.nan
        else:
            row_out["objective_mean"] = float(np.mean(vals))
            row_out["objective_min"] = float(np.min(vals))
            row_out["objective_max"] = float(np.max(vals))
            for p in percentiles:
                row_out[f"objective_p{p}"] = _safe_percentile(vals, p)
        out.append(row_out)

    out.sort(
        key=lambda r: (
            float(r["alpha"]) if r["alpha"] is not None else np.inf,
            str(r["precond_label"]),
            float(r["step_size"]) if r["step_size"] is not None else np.inf,
            int(r["iteration"]),
        )
    )
    return out


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        print(f"Warning: no rows to write for {path}")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def _read_rows_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _row_matches_cached_filters(
    row: Dict[str, object],
    *,
    allowed_alphas: Optional[Tuple[float, ...]],
    allowed_step_sizes: Optional[Tuple[float, ...]],
    max_iterations: Optional[int],
    include_bsrem: bool,
) -> bool:
    precond_type = str(row.get("precond_type", "")).strip().lower()
    if _is_excluded_precond(precond_type):
        return False
    if precond_type == "bsrem" and not include_bsrem:
        return False

    alpha = _coerce_float(row.get("alpha"))
    if allowed_alphas is not None and not _float_matches(alpha, allowed_alphas):
        return False

    step_size = _coerce_float(row.get("step_size"))
    if allowed_step_sizes is not None and not _float_matches(step_size, allowed_step_sizes):
        return False

    if max_iterations is not None:
        iteration = _coerce_float(row.get("iteration"))
        if iteration is not None and int(iteration) > int(max_iterations):
            return False

    return True


def _filter_cached_convergence_rows(
    rows: List[Dict[str, object]],
    *,
    allowed_alphas: Optional[Tuple[float, ...]],
    allowed_step_sizes: Optional[Tuple[float, ...]],
    max_iterations: Optional[int],
    include_bsrem: bool,
    include_whole_image: bool,
    allowed_vois: Sequence[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    allowed_voi_set = set(str(v) for v in allowed_vois)
    for row in rows:
        if not _row_matches_cached_filters(
            row,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=max_iterations,
            include_bsrem=include_bsrem,
        ):
            continue

        voi = str(row.get("voi", ""))
        if voi == "whole_image" and not include_whole_image:
            continue
        if voi and voi != "whole_image" and allowed_voi_set and voi not in allowed_voi_set:
            continue

        out.append(row)
    return out


def _filter_cached_objective_rows(
    rows: List[Dict[str, object]],
    *,
    allowed_alphas: Optional[Tuple[float, ...]],
    allowed_step_sizes: Optional[Tuple[float, ...]],
    max_iterations: Optional[int],
    include_bsrem: bool,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        if _row_matches_cached_filters(
            row,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=max_iterations,
            include_bsrem=include_bsrem,
        ):
            out.append(row)
    return out


def _sanitize_token(text: object) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    clean = clean.strip("_")
    return clean or "value"


def _step_sort_key(step_value: object) -> Tuple[int, float, str]:
    step_num = _coerce_float(step_value)
    if step_num is not None:
        return (0, step_num, "")
    return (1, np.inf, str(step_value))


def _normalise_step_style_key(step_value: object) -> object:
    step_num = _coerce_float(step_value)
    if step_num is not None:
        return float(step_num)
    return str(step_value)


def _blend_with_white(color: object, fraction: float) -> Tuple[float, float, float]:
    frac = float(np.clip(fraction, 0.0, 1.0))
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - frac) + frac
    return (float(mixed[0]), float(mixed[1]), float(mixed[2]))


def _compute_plot_ylim(
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    initial_values: Sequence[float],
    initial_cap_factor: float,
    *,
    log_scale: bool,
    pad_fraction: float = 0.05,
) -> Optional[Tuple[float, float]]:
    finite_uppers = [float(v) for v in upper_values if np.isfinite(v)]
    if log_scale:
        finite_uppers = [v for v in finite_uppers if v > 0]
    if not finite_uppers:
        return None

    upper = float(max(finite_uppers))

    if initial_cap_factor > 0:
        finite_initials = [float(v) for v in initial_values if np.isfinite(v)]
        if finite_initials:
            upper_cap = float(max(finite_initials) * initial_cap_factor)
            if np.isfinite(upper_cap) and upper_cap > 0:
                upper = min(upper, upper_cap)

    if log_scale:
        finite_lowers = [float(v) for v in lower_values if np.isfinite(v) and v > 0]
        if finite_lowers:
            lower = float(min(finite_lowers))
        else:
            lower = upper / 1e3 if upper > 0 else 1e-6
        lower = max(lower, float(np.finfo(float).tiny))
        if lower >= upper:
            lower = max(float(np.finfo(float).tiny), upper / 10.0)
        log_lower = float(np.log10(lower))
        log_upper = float(np.log10(upper))
        log_span = max(log_upper - log_lower, 1e-6)
        lower = 10 ** (log_lower - pad_fraction * log_span)
        upper = 10 ** (log_upper + pad_fraction * log_span)
        return max(lower, float(np.finfo(float).tiny)), upper

    finite_lowers = [float(v) for v in lower_values if np.isfinite(v)]
    lower = float(min(finite_lowers)) if finite_lowers else 0.0
    if lower >= upper:
        span = max(abs(upper), 1.0)
        upper = lower + 0.01 * span
    span = upper - lower
    pad = pad_fraction * span if span > 0 else pad_fraction * max(abs(upper), 1.0)
    lower -= pad
    upper += pad
    if finite_lowers and min(finite_lowers) >= 0:
        lower = max(0.0, lower)
    return lower, upper


def _format_precond_plot_label(precond_label: object) -> str:
    raw = str(precond_label)
    precond_type = raw.split(":", 1)[0].strip().lower()
    if precond_type in PRECOND_LABEL_MAP:
        return PRECOND_LABEL_MAP[precond_type]
    if ":" in raw:
        strategy, precond = raw.split(":", 1)
        strategy_label = strategy.replace("+", " + ")
        precond_label = PRECOND_LABEL_MAP.get(precond.strip().lower(), precond.strip())
        return f"{strategy_label} | {precond_label}"
    return raw


def _precond_family_key_from_row(row: Dict[str, object]) -> str:
    subset_mode = str(row.get("subset_mode", "")).strip()
    prior_mode = str(row.get("prior_mode", "")).strip()
    if subset_mode or prior_mode:
        return str(row.get("precond_label", "unknown"))

    precond_type = str(row.get("precond_type", "")).strip().lower()
    if precond_type:
        return precond_type
    precond_label = str(row.get("precond_label", "unknown"))
    inferred = precond_label.split(":", 1)[0].strip().lower()
    return inferred or "unknown"


def _format_step_legend_label(step_key: object) -> str:
    step_num = _coerce_float(step_key)
    if step_num is not None:
        return f"step={step_num:g}"
    return f"step={step_key}"


def _x_values_from_iterations(rows: Sequence[Dict[str, object]], subsets: Optional[float]) -> np.ndarray:
    x = np.array([int(r["iteration"]) for r in rows], dtype=float)
    if subsets is not None:
        x = x / subsets
    return x


def _x_axis_label(subsets: Optional[float]) -> str:
    return "Epoch" if subsets is not None else "Iteration"


def _build_precond_legend_handles(
    color_by_precond: Dict[str, Tuple[float, float, float, float]],
) -> List[Line2D]:
    precond_handles: List[Line2D] = []
    for precond_label in sorted(color_by_precond):
        base_color = color_by_precond[precond_label]
        precond_handles.append(
            Line2D(
                [0],
                [0],
                color=_blend_with_white(base_color, 0.0),
                linestyle="-",
                linewidth=2,
                label=_format_precond_plot_label(precond_label),
            )
        )
    return precond_handles


def _build_step_legend_handles(
    style_by_step: Dict[object, Tuple[str, float]],
) -> List[Line2D]:
    step_handles: List[Line2D] = []
    for step_key, (linestyle, _) in sorted(style_by_step.items(), key=lambda item: _step_sort_key(item[0])):
        step_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=linestyle,
                linewidth=2,
                label=_format_step_legend_label(step_key),
            )
        )
    return step_handles


def _save_legend_only_figure(
    handles: Sequence[Line2D],
    title: str,
    out_path: Path,
) -> None:
    if not handles:
        return

    ncols = min(5, max(1, len(handles)))
    width = max(2.5, 1.4 * ncols)
    fig, ax = plt.subplots(figsize=(width, 1.6))
    ax.axis("off")
    fig.legend(
        handles=handles,
        title=title,
        loc="center",
        ncol=ncols,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.7",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved legend key: {out_path}")


def _save_precond_step_legend_files(
    output_dir: Path,
    stem: str,
    color_by_precond: Dict[str, Tuple[float, float, float, float]],
    style_by_step: Dict[object, Tuple[str, float]],
) -> None:
    legend_dir = output_dir / "legend_keys"
    legend_dir.mkdir(parents=True, exist_ok=True)

    precond_path = legend_dir / f"{stem}_preconditioners.png"
    step_path = legend_dir / f"{stem}_step_sizes.png"

    precond_handles = _build_precond_legend_handles(color_by_precond)
    step_handles = _build_step_legend_handles(style_by_step)

    _save_legend_only_figure(precond_handles, "Preconditioner", precond_path)
    _save_legend_only_figure(step_handles, "Step size", step_path)


def _build_series_style_maps(
    by_setting: Dict[str, List[Dict[str, object]]],
) -> Tuple[
    Dict[str, Tuple[str, object]],
    Dict[str, Tuple[float, float, float, float]],
    Dict[object, Tuple[str, float]],
]:
    setting_meta: Dict[str, Tuple[str, object]] = {}
    precond_labels: List[str] = []
    step_keys: List[object] = []

    for setting_key, setting_rows in by_setting.items():
        if not setting_rows:
            continue
        row0 = setting_rows[0]
        precond_label = _precond_family_key_from_row(row0)
        step_key = _normalise_step_style_key(row0.get("step_size"))
        setting_meta[setting_key] = (precond_label, step_key)
        precond_labels.append(precond_label)
        step_keys.append(step_key)

    unique_preconds = sorted(set(precond_labels))
    precond_palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_preconds))))
    color_by_precond = {label: precond_palette[i] for i, label in enumerate(unique_preconds)}

    unique_steps = sorted(set(step_keys), key=_step_sort_key)
    step_style_cycle: Tuple[Tuple[str, float], ...] = (
        ("-", 0.00),
        ("--", 0.15),
        (":", 0.30),
        ("-.", 0.45),
    )
    style_by_step = {step: step_style_cycle[i % len(step_style_cycle)] for i, step in enumerate(unique_steps)}

    return setting_meta, color_by_precond, style_by_step


def plot_convergence_bands(
    aggregated_rows: List[Dict[str, object]],
    output_dir: Path,
    curve_band: Tuple[int, int],
    nrmse_threshold: float,
    initial_cap_factor: float,
    subsets: Optional[float] = None,
) -> None:
    if not aggregated_rows:
        return

    p_low, p_high = curve_band
    metrics = ("nrmse", "rmse")

    by_alpha_mod_voi: Dict[Tuple[float, str, str], List[Dict[str, object]]] = {}
    for row in aggregated_rows:
        alpha = _coerce_float(row.get("alpha"))
        if alpha is None:
            continue
        key = (alpha, str(row["modality"]), str(row["voi"]))
        by_alpha_mod_voi.setdefault(key, []).append(row)

    for (alpha, modality, voi), group_rows in sorted(by_alpha_mod_voi.items()):
        by_setting: Dict[str, List[Dict[str, object]]] = {}
        for row in group_rows:
            setting_key = f"{row['precond_label']}|{row['step_size']}|{row['setting_id']}"
            by_setting.setdefault(setting_key, []).append(row)
        setting_meta, color_by_precond, style_by_step = _build_series_style_maps(by_setting)
        legend_stem = (
            f"convergence_alpha_{_sanitize_token(alpha)}_"
            f"{_sanitize_token(modality)}_{_sanitize_token(voi)}"
        )
        _save_precond_step_legend_files(
            output_dir=output_dir,
            stem=legend_stem,
            color_by_precond=color_by_precond,
            style_by_step=style_by_step,
        )

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(5, 3))
            plot_lower_values: List[float] = []
            plot_upper_values: List[float] = []
            plot_initial_values: List[float] = []

            for setting_key, setting_rows in sorted(by_setting.items()):
                setting_rows = sorted(setting_rows, key=lambda r: int(r["iteration"]))
                x = _x_values_from_iterations(setting_rows, subsets)
                y_mean = np.array([float(r[f"{metric}_mean"]) for r in setting_rows], dtype=float)
                y_min = np.array([float(r[f"{metric}_min"]) for r in setting_rows], dtype=float)
                y_max = np.array([float(r[f"{metric}_max"]) for r in setting_rows], dtype=float)
                y_low = np.array([float(r[f"{metric}_p{p_low}"]) for r in setting_rows], dtype=float)
                y_high = np.array([float(r[f"{metric}_p{p_high}"]) for r in setting_rows], dtype=float)
                if "nrmse_mean" in setting_rows[0]:
                    y_nrmse_mean = np.array([float(r["nrmse_mean"]) for r in setting_rows], dtype=float)
                else:
                    # Legacy aggregated CSVs may only expose relative_error_mean.
                    y_nrmse_mean = np.array([float(r["relative_error_mean"]) for r in setting_rows], dtype=float)
                precond_label, step_key = setting_meta.get(setting_key, ("unknown", "unknown"))
                base_color = color_by_precond.get(precond_label, (0.1, 0.1, 0.1, 1.0))
                linestyle, shade_shift = style_by_step.get(step_key, ("-", 0.0))
                line_color = _blend_with_white(base_color, shade_shift)
                outer_color = _blend_with_white(base_color, min(0.75, shade_shift + 0.35))
                inner_color = _blend_with_white(base_color, min(0.65, shade_shift + 0.20))

                if np.isfinite(nrmse_threshold):
                    crossing = np.where(np.isfinite(y_nrmse_mean) & (y_nrmse_mean <= nrmse_threshold))[0]
                    if crossing.size > 0:
                        stop_idx = int(crossing[0]) + 1
                        x = x[:stop_idx]
                        y_mean = y_mean[:stop_idx]
                        y_min = y_min[:stop_idx]
                        y_max = y_max[:stop_idx]
                        y_low = y_low[:stop_idx]
                        y_high = y_high[:stop_idx]

                finite_mean = y_mean[np.isfinite(y_mean)]
                if finite_mean.size > 0:
                    plot_initial_values.append(float(finite_mean[0]))

                finite_y_min = y_min[np.isfinite(y_min)]
                if finite_y_min.size > 0:
                    plot_lower_values.append(float(np.min(finite_y_min)))

                finite_y_max = y_max[np.isfinite(y_max)]
                if finite_y_max.size > 0:
                    plot_upper_values.append(float(np.max(finite_y_max)))

                ax.fill_between(x, y_min, y_max, color=outer_color, alpha=0.14, linewidth=0)
                ax.fill_between(x, y_low, y_high, color=inner_color, alpha=0.24, linewidth=0)
                ax.plot(x, y_mean, color=line_color, linestyle=linestyle, linewidth=2)

            metric_label = "NRMSE" if metric == "nrmse" else "RMSE"
            ax.set_xlabel(_x_axis_label(subsets))
            ax.set_ylabel(metric_label)
            #ax.set_title(
            #    f"Convergence ({metric_label}) | alpha={alpha}, {modality}, VOI={voi}\n"
            #    f"Envelope=min-max, inner=P{p_low}-P{p_high}"
            #)
            ax.grid(True, alpha=0.3)
            if metric == "nrmse":
                ax.set_yscale("log")

            plot_ylim = _compute_plot_ylim(
                lower_values=plot_lower_values,
                upper_values=plot_upper_values,
                initial_values=plot_initial_values,
                initial_cap_factor=initial_cap_factor,
                log_scale=(metric == "nrmse"),
            )
            if plot_ylim is not None:
                ax.set_ylim(plot_ylim[0], plot_ylim[1])

            plt.tight_layout()

            out_stem = "nrmse" if metric == "nrmse" else metric
            out_path = output_dir / f"convergence_{out_stem}_alpha_{alpha}_{modality}_{voi}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if metric == "nrmse":
                # Backward-compatible filename.
                legacy_out_path = output_dir / f"convergence_relative_error_alpha_{alpha}_{modality}_{voi}.png"
                fig.savefig(legacy_out_path, dpi=150, bbox_inches="tight")
                print(f"Saved plot: {legacy_out_path}")
            plt.close(fig)
            print(f"Saved plot: {out_path}")


def plot_objective_bands(
    aggregated_objective_rows: List[Dict[str, object]],
    output_dir: Path,
    curve_band: Tuple[int, int],
    initial_cap_factor: float,
    subsets: Optional[float] = None,
) -> None:
    if not aggregated_objective_rows:
        return

    p_low, p_high = curve_band
    by_alpha: Dict[float, List[Dict[str, object]]] = {}
    for row in aggregated_objective_rows:
        alpha = _coerce_float(row.get("alpha"))
        if alpha is None:
            continue
        by_alpha.setdefault(alpha, []).append(row)

    for alpha, rows_alpha in sorted(by_alpha.items()):
        by_setting: Dict[str, List[Dict[str, object]]] = {}
        for row in rows_alpha:
            setting_key = f"{row['precond_label']}|{row['step_size']}|{row['setting_id']}"
            by_setting.setdefault(setting_key, []).append(row)
        setting_meta, color_by_precond, style_by_step = _build_series_style_maps(by_setting)
        legend_stem = f"convergence_objective_alpha_{_sanitize_token(alpha)}"
        _save_precond_step_legend_files(
            output_dir=output_dir,
            stem=legend_stem,
            color_by_precond=color_by_precond,
            style_by_step=style_by_step,
        )

        fig, ax = plt.subplots(figsize=(6, 3.5))
        plot_lower_values: List[float] = []
        plot_upper_values: List[float] = []
        plot_initial_values: List[float] = []

        for setting_key, setting_rows in sorted(by_setting.items()):
            setting_rows = sorted(setting_rows, key=lambda r: int(r["iteration"]))
            # Objective callbacks are written once per epoch, whereas image
            # snapshots are indexed by subset updates.  Do not divide the
            # objective callback index by ``subsets`` a second time.
            x = _x_values_from_iterations(setting_rows, subsets=None)
            y_mean = np.array([float(r["objective_mean"]) for r in setting_rows], dtype=float)
            y_min = np.array([float(r["objective_min"]) for r in setting_rows], dtype=float)
            y_max = np.array([float(r["objective_max"]) for r in setting_rows], dtype=float)
            y_low = np.array([float(r[f"objective_p{p_low}"]) for r in setting_rows], dtype=float)
            y_high = np.array([float(r[f"objective_p{p_high}"]) for r in setting_rows], dtype=float)
            precond_label, step_key = setting_meta.get(setting_key, ("unknown", "unknown"))
            base_color = color_by_precond.get(precond_label, (0.1, 0.1, 0.1, 1.0))
            linestyle, shade_shift = style_by_step.get(step_key, ("-", 0.0))
            line_color = _blend_with_white(base_color, shade_shift)
            outer_color = _blend_with_white(base_color, min(0.75, shade_shift + 0.35))
            inner_color = _blend_with_white(base_color, min(0.65, shade_shift + 0.20))

            finite_mean = y_mean[np.isfinite(y_mean)]
            if finite_mean.size > 0:
                plot_initial_values.append(float(finite_mean[0]))

            finite_y_min = y_min[np.isfinite(y_min)]
            if finite_y_min.size > 0:
                plot_lower_values.append(float(np.min(finite_y_min)))

            finite_y_max = y_max[np.isfinite(y_max)]
            if finite_y_max.size > 0:
                plot_upper_values.append(float(np.max(finite_y_max)))

            ax.fill_between(x, y_min, y_max, color=outer_color, alpha=0.14, linewidth=0)
            ax.fill_between(x, y_low, y_high, color=inner_color, alpha=0.24, linewidth=0)
            ax.plot(x, y_mean, color=line_color, linestyle=linestyle, linewidth=2)

        ax.set_xlabel("Epoch" if subsets is not None else "Iteration")
        ax.set_ylabel("Objective")
        #ax.set_title(
        #    f"Convergence (objective) | alpha={alpha}\n"
        #    f"Envelope=min-max, inner=P{p_low}-P{p_high}"
        #)
        ax.grid(True, alpha=0.3)

        plot_ylim = _compute_plot_ylim(
            lower_values=plot_lower_values,
            upper_values=plot_upper_values,
            initial_values=plot_initial_values,
            initial_cap_factor=initial_cap_factor,
            log_scale=False,
        )
        if plot_ylim is not None:
            ax.set_ylim(plot_ylim[0], plot_ylim[1])

        plt.tight_layout()
        out_path = output_dir / f"convergence_objective_alpha_{alpha}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")


SUBSET_SELECTION_STYLE = {
    ("paired", "always"): ("Paired subsets; prior always", "#1976d2"),
    ("paired", "subset"): ("Paired subsets; prior once/epoch", "#12b886"),
    ("separate", "always"): ("Separate subsets; prior always", "#f0642f"),
    ("separate", "subset"): ("Separate subsets; prior once/epoch", "#4c3fa3"),
}


def _subset_selection_data_passes(
    rows: Sequence[Dict[str, object]],
    subset_mode: str,
    prior_mode: str,
    total_separate_subsets: float,
) -> np.ndarray:
    """Convert optimiser updates to full PET+SPECT data passes.

    A separate data update has 18 PET + 18 SPECT subsets (36 updates/pass),
    while a paired data update has 18 paired subsets (18 updates/pass).  When
    the prior is sampled as its own subset, it adds 18 updates/pass.  Hence the
    denominators are 18 (paired/always), 36 (paired/subset or
    separate/always), and 54 (separate/subset).
    """
    paired_subsets = total_separate_subsets / 2.0
    updates_per_data_pass = (
        paired_subsets if subset_mode == "paired" else total_separate_subsets
    )
    if prior_mode == "subset":
        updates_per_data_pass += paired_subsets
    iterations = np.asarray([int(row["iteration"]) for row in rows], dtype=float)
    return iterations / updates_per_data_pass


def plot_subset_selection_summary(
    aggregated_rows: List[Dict[str, object]],
    figures_dir: Path,
    curve_band: Tuple[int, int],
    subsets: Optional[float],
    *,
    bpos_label: str = "1bpos",
    modality: str = MODALITY_PET,
    filename_suffix: str = "",
    panel_specs: Sequence[Tuple[str, str]] = (
        ("whole_image", "Whole image"),
        ("hot_sphere", "Hot sphere"),
        ("cold_sphere", "Cold sphere"),
    ),
) -> None:
    """Save the paper-style three-panel subset-selection convergence figure.

    ``modality`` selects PET (default, matching exp3's panels) or the
    PET-space SPECT metric; ``filename_suffix`` is appended before the
    extension so PET and SPECT variants can coexist.
    """
    if not aggregated_rows:
        return

    p_low, p_high = curve_band
    by_alpha: Dict[float, List[Dict[str, object]]] = {}
    for row in aggregated_rows:
        alpha = _coerce_float(row.get("alpha"))
        if alpha is not None and str(row.get("modality", "")) == modality:
            by_alpha.setdefault(alpha, []).append(row)

    for alpha, rows_alpha in sorted(by_alpha.items()):
        if not any(str(row.get("subset_mode", "")).strip() for row in rows_alpha):
            continue

        precond_types = sorted({str(row.get("precond_type", "")).strip() for row in rows_alpha})
        for precond_type in precond_types:
            rows_precond = [row for row in rows_alpha if str(row.get("precond_type", "")).strip() == precond_type]
            fig, axes = plt.subplots(1, len(panel_specs), figsize=(14, 4.6), sharey=True)
            all_lower: List[float] = []
            all_upper: List[float] = []
            legend_handles: List[Line2D] = []

            for ax, (voi, title) in zip(axes, panel_specs):
                ax.set_title(title)
                for (subset_mode, prior_mode), (label, color) in SUBSET_SELECTION_STYLE.items():
                    series = [
                        row for row in rows_precond
                        if str(row.get("voi", "")) == voi
                        and str(row.get("subset_mode", "")).strip() == subset_mode
                        and str(row.get("prior_mode", "")).strip() == prior_mode
                    ]
                    if not series:
                        continue
                    series.sort(key=lambda row: int(row["iteration"]))
                    if subsets is None:
                        raise ValueError("Subset-selection summary requires --subsets.")
                    x = _subset_selection_data_passes(series, subset_mode, prior_mode, subsets)
                    mean = np.asarray([float(row["nrmse_mean"]) for row in series], dtype=float)
                    low = np.asarray([float(row[f"nrmse_p{p_low}"]) for row in series], dtype=float)
                    high = np.asarray([float(row[f"nrmse_p{p_high}"]) for row in series], dtype=float)
                    finite_low = low[np.isfinite(low) & (low > 0)]
                    finite_high = high[np.isfinite(high) & (high > 0)]
                    all_lower.extend(finite_low.tolist())
                    all_upper.extend(finite_high.tolist())
                    ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
                    ax.plot(x, mean, color=color, linewidth=2.5, marker="o", markersize=4)

                ax.set_xlabel("Data passes")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

            axes[0].set_ylabel("NRMSE to converged baseline")
            plot_ylim = _compute_plot_ylim(all_lower, all_upper, [], 0.0, log_scale=True)
            if plot_ylim is not None:
                for ax in axes:
                    ax.set_ylim(plot_ylim)

            for _key, (label, color) in SUBSET_SELECTION_STYLE.items():
                legend_handles.append(Line2D([0], [0], color=color, linewidth=2.5, marker="o", markersize=5, label=label))
            fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="lower center", ncol=2, frameon=False)
            fig.tight_layout(rect=(0, 0.12, 1, 1))
            figures_dir.mkdir(parents=True, exist_ok=True)
            out_path = figures_dir / f"exp_subset_selection_{bpos_label}_{_sanitize_token(precond_type)}_alpha_{alpha:g}{filename_suffix}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved subset-selection summary figure: {out_path}")


def _choose_slice_for_overlay(pet_image: np.ndarray, voi_masks: Dict[str, np.ndarray]) -> int:
    if voi_masks:
        union = np.zeros_like(next(iter(voi_masks.values())), dtype=bool)
        for m in voi_masks.values():
            union |= m
        z_idx = np.where(union.any(axis=(1, 2)))[0]
        if z_idx.size > 0:
            return int(z_idx[len(z_idx) // 2])
    max_loc = np.unravel_index(np.argmax(pet_image), pet_image.shape)
    return int(max_loc[0])


def _crop_for_imshow(image_2d: np.ndarray, border: int = 10) -> np.ndarray:
    if image_2d.ndim != 2:
        raise ValueError(f"Expected 2D image for imshow cropping, got shape={image_2d.shape}")
    if border <= 0:
        return image_2d
    if image_2d.shape[1] <= 2 * border:
        return image_2d
    return image_2d[:, border:-border]


def save_voi_overlay_image(
    output_dir: Path,
    alpha: float,
    pet_background: np.ndarray,
    voi_masks: Dict[str, np.ndarray],
    pet_reference_path: Optional[Path] = None,
    umap_vmax: float = 0.11,
    dpi: int = 300,
) -> None:
    if not voi_masks:
        return

    if pet_reference_path is not None:
        coronal_y, aspect = _get_coronal_index_and_aspect(pet_reference_path)
    else:
        # Fallback when image metadata path is not available.
        max_loc = np.unravel_index(np.argmax(pet_background), pet_background.shape)
        coronal_y = int(max_loc[1])
        aspect = 1.0

    coronal_y = int(np.clip(coronal_y, 0, pet_background.shape[1] - 1))
    pet_slice = _crop_for_imshow(pet_background[:, coronal_y], border=10)

    sorted_voi_items = sorted(voi_masks.items())
    default_colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_voi_items)))
    voi_color_overrides = {
        "cold_sphere": "#00BFFF",  # bright blue
        "hot_sphere": "#00FF00",  # bright green
        "cold_sphere_edge": "#FF8C00",  # orange for high contrast
        "hot_sphere_edge": "#FFD700",  # gold for high contrast
    }
    voi_color_map = {
        voi_name: voi_color_overrides.get(voi_name, default_colors[idx])
        for idx, (voi_name, _mask) in enumerate(sorted_voi_items)
    }

    def _overlay_priority(voi_name: str) -> int:
        if voi_name.endswith("_edge"):
            return 2
        if voi_name in {"hot_sphere", "cold_sphere"}:
            return 1
        return 0

    # Draw edges last so they are not obscured by liver/sphere contours.
    draw_voi_items = sorted(sorted_voi_items, key=lambda item: (_overlay_priority(item[0]), item[0]))
    cmap = "gray"
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    axim = ax.imshow(
        pet_slice,
        vmin=0.0,
        vmax=umap_vmax,
        aspect=aspect,
        cmap=cmap,
    )
    ax.axis("off")
    plt.colorbar(axim, ax=ax, shrink=0.9)
    ax.text(
        0.05,
        0.95,
        "uMap",
        color="white",
        fontsize=14,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    legend_handles: List[Line2D] = []
    for voi_name, _mask in sorted_voi_items:
        color = voi_color_map[voi_name]
        is_edge = voi_name.endswith("_edge")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.6 if is_edge else 2.0,
                linestyle="--" if is_edge else "-",
                label=voi_name,
            )
        )

    for voi_name, mask in draw_voi_items:
        color = voi_color_map[voi_name]
        is_edge = voi_name.endswith("_edge")
        mask_slice = _crop_for_imshow(mask[:, coronal_y, :].astype(float), border=10)
        if is_edge:
            fill_rgba = np.array(mcolors.to_rgba(color, alpha=0.22), dtype=float)
            fill_overlay = np.zeros(mask_slice.shape + (4,), dtype=float)
            fill_overlay[..., :3] = fill_rgba[:3]
            fill_overlay[..., 3] = np.where(mask_slice > 0.5, fill_rgba[3], 0.0)
            ax.imshow(
                fill_overlay,
                aspect=aspect,
                interpolation="nearest",
                zorder=4,
            )
        ax.contour(
            mask_slice,
            levels=[0.5],
            colors=[color],
            linewidths=2.0 if is_edge else 1.0,
            linestyles="--" if is_edge else "-",
            zorder=5 if is_edge else 3,
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=max(1, len(legend_handles)))
    #fig.suptitle(f"VOIs on uMap in PET Space (alpha={alpha}, coronal y={coronal_y})", y=1.03, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])

    out_path = output_dir / f"voi_overlay_alpha_{alpha}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VOI overlay image: {out_path}")


def _get_coronal_index_and_aspect(pet_reference_path: Path) -> Tuple[int, float]:
    pet_img = ImageData(str(pet_reference_path))
    arr = pet_img.as_array()

    try:
        # Match notebook style: locate hotspot from a smoothed PET image.
        from sirf.STIR import SeparableGaussianImageFilter

        gauss = SeparableGaussianImageFilter()
        gauss.set_fwhms((10, 10, 10))
        smoothed = pet_img.clone()
        gauss.apply(smoothed)
        arr = smoothed.as_array()
    except Exception:
        pass

    max_loc = np.unravel_index(np.argmax(arr), arr.shape)
    coronal_y = int(max_loc[1])

    vz, vy, _vx = pet_img.voxel_sizes()
    aspect = float(vz) / float(vy) if float(vy) != 0 else 1.0
    return coronal_y, aspect


def _get_fall_cmap():
    try:
        import cmasher

        return cmasher.fall
    except Exception:
        # Fallback when cmasher is unavailable in a given env.
        return "inferno"


def save_final_pet_spect_pair_figure(
    output_dir: Path,
    run_dir: Path,
    run_info: Dict[str, str],
    baseline_ctx: BaselineContext,
    coronal_y: int,
    aspect: float,
    pet_vmax: float,
    spect_vmax: float,
    dpi: int,
) -> Optional[Path]:
    pet_final = _find_final_image_path(run_dir, modality=0)
    spect_final = _find_final_image_path(run_dir, modality=1)
    if pet_final is None or spect_final is None:
        print(f"Warning: missing final images for {run_dir.name}")
        return None

    pet_img = ImageData(str(pet_final))
    spect_img = ImageData(str(spect_final))

    pet_arr = pet_img.as_array()
    # image_1 already PET-space; use directly (no resampling / double-warp).
    spect_res_arr = spect_img.as_array()

    if coronal_y < 0 or coronal_y >= pet_arr.shape[1]:
        coronal_y = int(np.clip(coronal_y, 0, pet_arr.shape[1] - 1))

    cmap = _get_fall_cmap()
    fig, ax = plt.subplots(2, 1, figsize=(6, 3))
    axim0 = ax[0].imshow(
        _crop_for_imshow(pet_arr[:, coronal_y], border=10),
        vmax=pet_vmax,
        aspect=aspect,
        cmap=cmap,
    )
    axim1 = ax[1].imshow(
        _crop_for_imshow(spect_res_arr[:, coronal_y], border=10),
        vmax=spect_vmax,
        aspect=aspect,
        cmap=cmap,
    )

    for a in ax:
        a.axis("off")

    plt.colorbar(axim0, ax=ax[0], shrink=0.9)
    plt.colorbar(axim1, ax=ax[1], shrink=0.9)

    ax[1].text(
        0.05,
        0.95,
        "SPECT",
        color="white",
        fontsize=14,
        transform=ax[1].transAxes,
        ha="left",
        va="top",
    )
    ax[0].text(
        0.05,
        0.95,
        "PET",
        color="white",
        fontsize=14,
        transform=ax[0].transAxes,
        ha="left",
        va="top",
    )

    #fig.suptitle(
    #    f"{_format_precond_plot_label(run_info.get('precond_label', 'unknown'))}, "
    #    f"step={run_info.get('step_size')}, alpha={run_info.get('alpha')}",
    #    fontsize=9,
    #    y=1.02,
    #)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_token = _sanitize_token(run_info.get("alpha"))
    setting_token = _sanitize_token(run_info.get("setting_id", run_dir.name))
    out_path = output_dir / f"final_pair_alpha_{alpha_token}_{setting_token}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _compute_robust_symmetric_vmax(arr: np.ndarray, percentile: float = 99.5) -> float:
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(finite)))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def _compute_abs_symmetric_vmax(arr: np.ndarray) -> float:
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.max(np.abs(finite)))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def _collect_mean_final_images_for_setting(
    run_entries: Sequence[Tuple[Path, Dict[str, str]]],
    baseline_ctx: BaselineContext,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    pet_arrays: List[np.ndarray] = []
    spect_arrays: List[np.ndarray] = []
    pet_shape = baseline_ctx.pet_final_array.shape

    for run_dir, _run_info in run_entries:
        pet_path = _find_highest_iteration_image_path(run_dir, modality=0) or _find_final_image_path(run_dir, modality=0)
        spect_path = _find_highest_iteration_image_path(run_dir, modality=1) or _find_final_image_path(run_dir, modality=1)
        if pet_path is None or spect_path is None:
            continue

        pet_arr = ImageData(str(pet_path)).as_array()
        if tuple(pet_arr.shape) != tuple(pet_shape):
            continue
        pet_arrays.append(pet_arr.astype(np.float64))

        # image_1 already PET-space; use directly (no resampling / double-warp).
        spect_res_arr = ImageData(str(spect_path)).as_array()
        if tuple(spect_res_arr.shape) != tuple(pet_shape):
            continue
        spect_arrays.append(spect_res_arr.astype(np.float64))

    n_used = min(len(pet_arrays), len(spect_arrays))
    if n_used == 0:
        return None, None, 0

    pet_mean = np.mean(np.stack(pet_arrays[:n_used], axis=0), axis=0)
    spect_mean = np.mean(np.stack(spect_arrays[:n_used], axis=0), axis=0)
    return pet_mean, spect_mean, n_used


def save_mean_final_pet_spect_pair_figure(
    output_dir: Path,
    run_info: Dict[str, str],
    alpha: float,
    setting_id: str,
    pet_mean: np.ndarray,
    spect_mean: np.ndarray,
    coronal_y: int,
    aspect: float,
    pet_vmax: float,
    spect_vmax: float,
    n_used: int,
    dpi: int,
) -> Path:
    cmap = _get_fall_cmap()
    fig, ax = plt.subplots(2, 1, figsize=(6, 3))
    axim0 = ax[0].imshow(
        _crop_for_imshow(pet_mean[:, coronal_y], border=10),
        vmax=pet_vmax,
        aspect=aspect,
        cmap=cmap,
    )
    axim1 = ax[1].imshow(
        _crop_for_imshow(spect_mean[:, coronal_y], border=10),
        vmax=spect_vmax,
        aspect=aspect,
        cmap=cmap,
    )

    for a in ax:
        a.axis("off")

    plt.colorbar(axim0, ax=ax[0], shrink=0.9)
    plt.colorbar(axim1, ax=ax[1], shrink=0.9)

    ax[1].text(0.05, 0.95, "SPECT mean", color="white", fontsize=12, transform=ax[1].transAxes, ha="left", va="top")
    ax[0].text(0.05, 0.95, "PET mean", color="white", fontsize=12, transform=ax[0].transAxes, ha="left", va="top")

    #fig.suptitle(
    #    f"{_format_precond_plot_label(run_info.get('precond_label', 'unknown'))}, "
    #    f"step={run_info.get('step_size')}, alpha={alpha}, n={n_used}",
    #    fontsize=9,
    #    y=1.02,
    #)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_token = _sanitize_token(alpha)
    setting_token = _sanitize_token(setting_id)
    out_path = output_dir / f"final_pair_mean_alpha_{alpha_token}_{setting_token}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_mean_diff_to_baseline_figure(
    output_dir: Path,
    run_info: Dict[str, str],
    alpha: float,
    setting_id: str,
    pet_diff: np.ndarray,
    spect_diff: np.ndarray,
    coronal_y: int,
    aspect: float,
    n_used: int,
    dpi: int,
    pet_vlim: Optional[float] = None,
    spect_vlim: Optional[float] = None,
) -> Path:
    cmap = "coolwarm"
    if pet_vlim is None:
        pet_vlim = _compute_robust_symmetric_vmax(pet_diff)
    if spect_vlim is None:
        spect_vlim = _compute_robust_symmetric_vmax(spect_diff)

    fig, ax = plt.subplots(2, 1, figsize=(6, 3))
    axim0 = ax[0].imshow(
        _crop_for_imshow(pet_diff[:, coronal_y], border=10),
        vmin=-pet_vlim,
        vmax=pet_vlim,
        aspect=aspect,
        cmap=cmap,
    )
    axim1 = ax[1].imshow(
        _crop_for_imshow(spect_diff[:, coronal_y], border=10),
        vmin=-spect_vlim,
        vmax=spect_vlim,
        aspect=aspect,
        cmap=cmap,
    )

    for a in ax:
        a.axis("off")

    plt.colorbar(axim0, ax=ax[0], shrink=0.9)
    plt.colorbar(axim1, ax=ax[1], shrink=0.9)

    ax[1].text(0.05, 0.95, "SPECT mean - baseline", color="white", fontsize=10, transform=ax[1].transAxes, ha="left", va="top")
    ax[0].text(0.05, 0.95, "PET mean - baseline", color="white", fontsize=10, transform=ax[0].transAxes, ha="left", va="top")

    #fig.suptitle(
    #    f"Mean difference to baseline | {_format_precond_plot_label(run_info.get('precond_label', 'unknown'))}, "
    #    f"step={run_info.get('step_size')}, alpha={alpha}, n={n_used}",
    #    fontsize=8,
    #    y=1.02,
    #)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_token = _sanitize_token(alpha)
    setting_token = _sanitize_token(setting_id)
    out_path = output_dir / f"final_pair_mean_diff_alpha_{alpha_token}_{setting_token}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_final_images_per_preconditioner(
    output_dir: Path,
    baseline_cache: Dict[float, Optional[BaselineContext]],
    representative_runs: Dict[Tuple[float, str], Tuple[int, Path, Dict[str, str]]],
    pet_vmax: float,
    spect_vmax: float,
    dpi: int,
) -> None:
    if not representative_runs:
        print("No representative runs found for final-image figures.")
        return

    final_images_dir = output_dir / "final_images_per_preconditioner"
    coronal_cache: Dict[float, Tuple[int, float]] = {}

    saved = 0
    for (alpha, setting_id), (_repeat_key, run_dir, run_info) in sorted(
        representative_runs.items(),
        key=lambda item: (item[0][0], item[1][2].get("precond_label", ""), item[1][2].get("step_size", ""), item[0][1]),
    ):
        baseline_ctx = baseline_cache.get(alpha)
        if baseline_ctx is None:
            print(f"Warning: no baseline context for alpha={alpha}, skipping final image for {setting_id}")
            continue
        if alpha not in coronal_cache:
            coronal_cache[alpha] = _get_coronal_index_and_aspect(baseline_ctx.pet_final_path)
        coronal_y, aspect = coronal_cache[alpha]

        out_path = save_final_pet_spect_pair_figure(
            output_dir=final_images_dir,
            run_dir=run_dir,
            run_info=run_info,
            baseline_ctx=baseline_ctx,
            coronal_y=coronal_y,
            aspect=aspect,
            pet_vmax=pet_vmax,
            spect_vmax=spect_vmax,
            dpi=dpi,
        )
        if out_path is not None:
            saved += 1
            print(f"Saved final PET/SPECT pair: {out_path}")

    print(f"Saved {saved} final-image figures to {final_images_dir}")


def generate_mean_final_and_difference_images_per_preconditioner(
    output_dir: Path,
    baseline_cache: Dict[float, Optional[BaselineContext]],
    grouped_runs: Dict[Tuple[float, str], List[Tuple[Path, Dict[str, str]]]],
    pet_vmax: float,
    spect_vmax: float,
    dpi: int,
) -> None:
    if not grouped_runs:
        print("No grouped runs found for mean final-image figures.")
        return

    mean_dir = output_dir / "final_images_mean_per_preconditioner"
    diff_dir = output_dir / "final_images_mean_diff_to_baseline_per_preconditioner"
    coronal_cache: Dict[float, Tuple[int, float]] = {}
    shared_diff_vlims_by_alpha: Dict[float, Tuple[float, float]] = {}

    sorted_grouped_items = sorted(
        grouped_runs.items(),
        key=lambda item: (
            item[0][0],
            item[1][0][1].get("precond_label", "") if item[1] else "",
            item[1][0][1].get("step_size", "") if item[1] else "",
            item[0][1],
        ),
    )

    # Pass 1: compute shared diff color limits per alpha so every setting is
    # rendered with the same scale.
    for (alpha, setting_id), run_entries in sorted_grouped_items:
        baseline_ctx = baseline_cache.get(alpha)
        if baseline_ctx is None:
            continue
        pet_mean, spect_mean, n_used = _collect_mean_final_images_for_setting(run_entries, baseline_ctx)
        if pet_mean is None or spect_mean is None or n_used <= 0:
            continue

        pet_diff = pet_mean - baseline_ctx.pet_final_array
        spect_diff = spect_mean - baseline_ctx.spect_resampled_final_array
        pet_vlim = _compute_abs_symmetric_vmax(pet_diff)
        spect_vlim = _compute_abs_symmetric_vmax(spect_diff)

        prev = shared_diff_vlims_by_alpha.get(alpha)
        if prev is None:
            shared_diff_vlims_by_alpha[alpha] = (pet_vlim, spect_vlim)
        else:
            shared_diff_vlims_by_alpha[alpha] = (max(prev[0], pet_vlim), max(prev[1], spect_vlim))

    for alpha, (pet_vlim, spect_vlim) in sorted(shared_diff_vlims_by_alpha.items()):
        print(
            f"alpha={alpha}: shared mean-diff color limits "
            f"PET=+/-{pet_vlim:.6g}, SPECT=+/-{spect_vlim:.6g}"
        )

    saved_mean = 0
    saved_diff = 0
    for (alpha, setting_id), run_entries in sorted_grouped_items:
        baseline_ctx = baseline_cache.get(alpha)
        if baseline_ctx is None:
            print(f"Warning: no baseline context for alpha={alpha}, skipping mean/diff images for {setting_id}")
            continue

        if alpha not in coronal_cache:
            coronal_cache[alpha] = _get_coronal_index_and_aspect(baseline_ctx.pet_final_path)
        coronal_y, aspect = coronal_cache[alpha]

        pet_mean, spect_mean, n_used = _collect_mean_final_images_for_setting(run_entries, baseline_ctx)
        if pet_mean is None or spect_mean is None or n_used <= 0:
            print(f"Warning: no valid final images for averaging in setting {setting_id} (alpha={alpha})")
            continue

        run_info_ref = run_entries[0][1]
        mean_path = save_mean_final_pet_spect_pair_figure(
            output_dir=mean_dir,
            run_info=run_info_ref,
            alpha=alpha,
            setting_id=setting_id,
            pet_mean=pet_mean,
            spect_mean=spect_mean,
            coronal_y=coronal_y,
            aspect=aspect,
            pet_vmax=pet_vmax,
            spect_vmax=spect_vmax,
            n_used=n_used,
            dpi=dpi,
        )
        saved_mean += 1
        print(f"Saved mean final PET/SPECT pair: {mean_path}")

        pet_diff = pet_mean - baseline_ctx.pet_final_array
        spect_diff = spect_mean - baseline_ctx.spect_resampled_final_array
        diff_path = save_mean_diff_to_baseline_figure(
            output_dir=diff_dir,
            run_info=run_info_ref,
            alpha=alpha,
            setting_id=setting_id,
            pet_diff=pet_diff,
            spect_diff=spect_diff,
            coronal_y=coronal_y,
            aspect=aspect,
            n_used=n_used,
            dpi=dpi,
            pet_vlim=shared_diff_vlims_by_alpha.get(alpha, (None, None))[0],
            spect_vlim=shared_diff_vlims_by_alpha.get(alpha, (None, None))[1],
        )
        saved_diff += 1
        print(f"Saved mean difference-to-baseline PET/SPECT pair: {diff_path}")

    print(f"Saved {saved_mean} mean final-image figures to {mean_dir}")
    print(f"Saved {saved_diff} mean difference-image figures to {diff_dir}")


def main() -> None:
    global EXCLUDED_PRECOND_TYPES
    parser = argparse.ArgumentParser(
        description="Compute convergence-over-iterations with PET-space VOIs and SPECT resampling."
    )
    parser.add_argument("--sweep", type=str, default=None, help="Sweep directory name under the base output directory, e.g. precond_1bpos")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline directory name under the base output directory, e.g. baselines_1bpos")
    parser.add_argument(
        "--sweep-dir",
        action="append",
        default=None,
        help=(
            "Explicit sweep directory to analyse. May be supplied multiple times; "
            "run directories from all supplied sweep dirs are pooled."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Explicit baseline directory containing baseline_alpha_<alpha> subdirectories.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: <sweep>_analysis)")
    parser.add_argument("--masks", type=str, default=None, help="Mask directory containing <voi>_pet.hv")
    parser.add_argument(
        "--umap-path",
        type=str,
        default=None,
        help=(
            "Directory to search for the overlay umap (umap_zoomed.hv/umap.hv) "
            "when the recon's recorded pet_data_path is not reachable locally. "
            "Falls back to the mask directory if unset."
        ),
    )
    parser.add_argument("--vois", nargs="+", default=list(DEFAULT_VOIS), help="VOI names to evaluate")
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
        "--subsets",
        type=float,
        default=None,
        help="If provided, plot x-axis as epochs using iteration/subsets.",
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
        "--exclude-precond",
        type=str,
        default=None,
        help=(
            "Comma-separated precond_type values to exclude "
            f"(default: {','.join(sorted(EXCLUDED_PRECOND_TYPES))}). "
            "Pass 'none' or '' to exclude nothing."
        ),
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
        help="Per-plot upper y-limit cap as multiplier of initial mean value (default: 1.1, set <=0 to disable).",
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
        help="Use mean final image from the given precond_type (e.g. ls_block_diag) instead of baseline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even when cached convergence/objective CSVs already exist.",
    )
    args = parser.parse_args()
    subsets = float(args.subsets) if args.subsets is not None else None
    if subsets is not None and subsets <= 0:
        raise ValueError("--subsets must be positive")
    if args.exclude_precond is not None:
        tokens = [t.strip().lower() for t in args.exclude_precond.split(",") if t.strip()]
        EXCLUDED_PRECOND_TYPES = set() if tokens == ["none"] else set(tokens)
        print(f"Excluded preconditioners: {sorted(EXCLUDED_PRECOND_TYPES) or '<none>'}")
    if args.nrmse_threshold is not None:
        nrmse_threshold = float(args.nrmse_threshold)
        if args.relative_error_threshold is not None:
            print("Note: --relative-error-threshold ignored because --nrmse-threshold was provided.")
    elif args.relative_error_threshold is not None:
        nrmse_threshold = float(args.relative_error_threshold)
        print("Note: --relative-error-threshold is deprecated; use --nrmse-threshold instead.")
    else:
        # Default: no truncation -- plot NRMSE for all iterations. Pass a finite
        # --nrmse-threshold to truncate curves once they drop below it.
        nrmse_threshold = float("inf")

    summary_percentiles = _parse_percentiles(args.summary_percentiles, DEFAULT_SUMMARY_PERCENTILES)
    curve_band = _parse_percentiles(args.curve_inner_band, DEFAULT_CURVE_BAND)
    if len(curve_band) != 2:
        raise ValueError("--curve-inner-band must contain exactly two values, e.g. 10,90")
    curve_band_tuple = (curve_band[0], curve_band[1])
    include_whole_image = not args.no_whole_image
    if len(args.vois) == 1 and str(args.vois[0]).strip().lower() in {"none", "whole_image", "whole-image"}:
        args.vois = []
    if not args.vois and not include_whole_image:
        raise ValueError("No metrics requested: --vois none cannot be combined with --no-whole-image.")

    study_dir = Path(__file__).resolve().parent.parent
    allowed_alphas = _parse_alpha_values(args.alpha_values)
    if allowed_alphas is None:
        allowed_alphas = _load_allowed_alphas_from_csv(study_dir / "parameters" / "alphas.csv")
    allowed_step_sizes = _parse_step_sizes(args.step_sizes)

    base_output_dir = study_dir / "output"
    if args.sweep_dir:
        sweep_dirs = [Path(p).expanduser().resolve() for p in args.sweep_dir]
    elif args.sweep:
        sweep_dirs = [base_output_dir / args.sweep]
    else:
        raise ValueError("Provide either --sweep or at least one --sweep-dir.")

    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    elif args.baseline:
        baseline_dir = base_output_dir / args.baseline
    else:
        raise ValueError("Provide either --baseline or --baseline-dir.")

    output_name = args.sweep if args.sweep else "convergence_over_iterations_analysis"
    output_dir = Path(args.output) if args.output else base_output_dir / f"{output_name}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "convergence_over_iterations_voi.csv"
    agg_csv = output_dir / "convergence_over_iterations_voi_aggregated.csv"
    objective_raw_csv = output_dir / "convergence_over_iterations_objective.csv"
    objective_agg_csv = output_dir / "convergence_over_iterations_objective_aggregated.csv"

    cached_metrics_available = all(
        path.exists() for path in (raw_csv, agg_csv, objective_raw_csv, objective_agg_csv)
    )
    if not args.figures_only and not args.force and cached_metrics_available:
        print("Using cached convergence/objective CSVs (pass --force to recompute).")
        all_rows = _read_rows_csv(raw_csv)
        aggregated_rows = _read_rows_csv(agg_csv)
        objective_rows = _read_rows_csv(objective_raw_csv)
        aggregated_objective_rows = _read_rows_csv(objective_agg_csv)

        all_rows = _filter_cached_convergence_rows(
            all_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
            include_whole_image=include_whole_image,
            allowed_vois=args.vois,
        )
        aggregated_rows = _filter_cached_convergence_rows(
            aggregated_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
            include_whole_image=include_whole_image,
            allowed_vois=args.vois,
        )
        objective_rows = _filter_cached_objective_rows(
            objective_rows,
            allowed_alphas=allowed_alphas,
            allowed_step_sizes=allowed_step_sizes,
            max_iterations=args.max_iterations,
            include_bsrem=bool(args.include_bsrem),
        )
        aggregated_objective_rows = _filter_cached_objective_rows(
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

        plot_convergence_bands(
            aggregated_rows=aggregated_rows,
            output_dir=output_dir,
            curve_band=curve_band_tuple,
            nrmse_threshold=nrmse_threshold,
            initial_cap_factor=float(args.initial_cap_factor),
            subsets=subsets,
        )
        if aggregated_objective_rows:
            plot_objective_bands(
                aggregated_objective_rows=aggregated_objective_rows,
                output_dir=output_dir,
                curve_band=curve_band_tuple,
                initial_cap_factor=float(args.initial_cap_factor),
                subsets=subsets,
            )
        else:
            print(f"Warning: cached objective aggregated CSV is empty: {objective_agg_csv}")
        plot_subset_selection_summary(
            aggregated_rows=aggregated_rows,
            figures_dir=output_dir.parent / "figures",
            curve_band=curve_band_tuple,
            subsets=subsets,
        )

        print("=" * 70)
        print("Convergence plots refreshed from cached CSVs (after applying filters).")
        print(f"Raw rows: {raw_csv} ({len(all_rows)} rows)")
        print(f"Aggregated rows: {agg_csv} ({len(aggregated_rows)} rows)")
        print(f"Objective rows: {objective_raw_csv} ({len(objective_rows)} rows)")
        print(f"Objective aggregated rows: {objective_agg_csv} ({len(aggregated_objective_rows)} rows)")
        print(f"Plots + VOI overlays: {output_dir}")
        print("=" * 70)
        return

    mask_dir = _resolve_mask_dir(study_dir, args.masks) if args.vois else None
    if args.vois and mask_dir is None:
        raise FileNotFoundError(
            "Could not locate mask directory. Provide --masks with a directory containing <voi>_pet.hv."
        )

    # Extra directories to search for the overlay umap when the recon's own
    # (possibly cluster-only) pet_data_path is not reachable locally. An
    # explicit --umap-path dir wins; the mask/study dir is the default fallback.
    umap_fallback_dirs: List[Path] = []
    if getattr(args, "umap_path", None):
        umap_fallback_dirs.append(Path(args.umap_path))
    if mask_dir is not None:
        umap_fallback_dirs.append(mask_dir)

    missing_sweep_dirs = [sweep_dir for sweep_dir in sweep_dirs if not sweep_dir.exists()]
    if missing_sweep_dirs:
        raise FileNotFoundError(
            "Sweep directory not found: " + ", ".join(str(p) for p in missing_sweep_dirs)
        )
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    print("Sweep directories:")
    for sweep_dir in sweep_dirs:
        print(f"  {sweep_dir}")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mask directory: {mask_dir if mask_dir is not None else '<none>'}")
    print(f"VOIs: {args.vois if args.vois else '<none; whole-image only>'}")
    print(f"Include BSREM: {bool(args.include_bsrem)}")
    if allowed_alphas is not None:
        print(f"Alpha filter: {allowed_alphas}")
    else:
        print("Alpha filter: <none>")
    if allowed_step_sizes is not None:
        print(f"Step-size filter: {allowed_step_sizes}")
    else:
        print("Step-size filter: <none>")
    if subsets is not None:
        print(f"Epoch x-axis: iteration/{subsets:g}")

    run_dirs: List[Path] = []
    for sweep_dir in sweep_dirs:
        run_dirs.extend(_gather_runs(sweep_dir))
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    print(f"Found {len(run_dirs)} sweep run directories")

    baseline_cache: Dict[float, Optional[BaselineContext]] = {}
    masks_cache: Dict[float, Dict[str, np.ndarray]] = {}
    overlay_done_for_alpha: Dict[float, bool] = {}
    representative_runs: Dict[Tuple[float, str], Tuple[int, Path, Dict[str, str]]] = {}
    grouped_runs: Dict[Tuple[float, str], List[Tuple[Path, Dict[str, str]]]] = {}
    all_rows: List[Dict[str, object]] = []
    objective_rows: List[Dict[str, object]] = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        run_info = _load_run_info(run_dir)
        if run_info is None:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: no result.csv")
            continue
        precond_type = str(run_info.get("precond_type", "")).strip().lower()
        if _is_excluded_precond(precond_type):
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: precond '{precond_type}' excluded")
            continue
        if precond_type == "bsrem" and not args.include_bsrem:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: BSREM excluded by default")
            continue

        alpha = _coerce_float(run_info.get("alpha"))
        if alpha is None:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: missing alpha")
            continue
        if allowed_alphas is not None and alpha not in set(allowed_alphas):
            continue
        step_size = _coerce_float(run_info.get("step_size"))
        if allowed_step_sizes is not None and not _float_matches(step_size, allowed_step_sizes):
            continue

        if alpha not in baseline_cache:
            baseline_cache[alpha] = _build_baseline_context(
                baseline_dir=baseline_dir,
                alpha=alpha,
                explicit_transform=args.spect2pet_transform,
                sweep_dir=sweep_dirs[0] if len(sweep_dirs) == 1 else None,
                mean_baseline_from=args.mean_baseline_from,
                umap_fallback_dirs=umap_fallback_dirs,
            )
            ctx = baseline_cache[alpha]
            if ctx is not None:
                print(
                    f"alpha={alpha}: SPECT metrics computed in PET space directly "
                    f"(no resampling); overlay transform: {ctx.transform_path}"
                )
                print(f"alpha={alpha}: baseline PET image {ctx.pet_final_path}")
                print(f"alpha={alpha}: baseline SPECT image {ctx.spect_final_path}")
                if ctx.pet_support_mask is not None:
                    print(f"alpha={alpha}: PET support voxels {int(ctx.pet_support_mask.sum())}")
        baseline_ctx = baseline_cache[alpha]
        if baseline_ctx is None:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: baseline context unavailable")
            continue

        if alpha not in masks_cache:
            if args.vois:
                masks_cache[alpha] = load_pet_voi_masks(
                    mask_dir=mask_dir,
                    voi_names=args.vois,
                    pet_shape=baseline_ctx.pet_final_array.shape,
                    support_mask=baseline_ctx.pet_support_mask,
                )
            else:
                masks_cache[alpha] = {}
            if not masks_cache[alpha]:
                print(f"Warning: no VOI masks loaded for alpha={alpha}")
        voi_masks = masks_cache[alpha]
        if not voi_masks and not include_whole_image:
            print(f"[{idx}/{len(run_dirs)}] Skipping {run_dir.name}: no VOI masks loaded")
            continue

        if voi_masks and not overlay_done_for_alpha.get(alpha, False):
            pet_overlay_bg = (
                baseline_ctx.pet_umap_array
                if baseline_ctx.pet_umap_array is not None
                else baseline_ctx.pet_final_array
            )
            save_voi_overlay_image(
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
        repeat_id = _coerce_float(run_info.get("repeat_id"))
        repeat_key = int(repeat_id) if repeat_id is not None else 10**9
        rep_key = (alpha, str(setting_id))
        existing = representative_runs.get(rep_key)
        if existing is None or repeat_key < existing[0]:
            representative_runs[rep_key] = (repeat_key, run_dir, run_info)
        grouped_runs.setdefault(rep_key, []).append((run_dir, run_info))

        if args.figures_only:
            print(f"[{idx}/{len(run_dirs)}] Registered {run_dir.name} for figure generation")
            continue

        run_rows = compute_convergence_rows_for_run(
            run_dir=run_dir,
            run_info=run_info,
            baseline_ctx=baseline_ctx,
            voi_masks=voi_masks,
            include_whole_image=include_whole_image,
            max_iterations=args.max_iterations,
        )
        run_objective_rows = compute_objective_rows_for_run(
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
        generate_final_images_per_preconditioner(
            output_dir=output_dir,
            baseline_cache=baseline_cache,
            representative_runs=representative_runs,
            pet_vmax=float(args.final_vmax_pet),
            spect_vmax=float(args.final_vmax_spect),
            dpi=int(args.final_fig_dpi),
        )
        generate_mean_final_and_difference_images_per_preconditioner(
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

    aggregated_rows = aggregate_rows(all_rows, summary_percentiles)
    aggregated_objective_rows = aggregate_objective_rows(objective_rows, summary_percentiles)

    _write_rows_csv(raw_csv, all_rows)
    _write_rows_csv(agg_csv, aggregated_rows)
    _write_rows_csv(objective_raw_csv, objective_rows)
    _write_rows_csv(objective_agg_csv, aggregated_objective_rows)

    plot_convergence_bands(
        aggregated_rows=aggregated_rows,
        output_dir=output_dir,
        curve_band=curve_band_tuple,
        nrmse_threshold=nrmse_threshold,
        initial_cap_factor=float(args.initial_cap_factor),
        subsets=subsets,
    )
    plot_objective_bands(
        aggregated_objective_rows=aggregated_objective_rows,
        output_dir=output_dir,
        curve_band=curve_band_tuple,
        initial_cap_factor=float(args.initial_cap_factor),
        subsets=subsets,
    )
    plot_subset_selection_summary(
        aggregated_rows=aggregated_rows,
        figures_dir=output_dir.parent / "figures",
        curve_band=curve_band_tuple,
        subsets=subsets,
    )

    print("=" * 70)
    print("Convergence-over-iterations with VOIs complete.")
    print(f"Raw rows: {raw_csv}")
    print(f"Aggregated rows: {agg_csv}")
    print(f"Objective rows: {objective_raw_csv}")
    print(f"Objective aggregated rows: {objective_agg_csv}")
    print(f"Plots + VOI overlays: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
