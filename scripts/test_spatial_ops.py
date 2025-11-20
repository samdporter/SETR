#!/usr/bin/env python3
"""Diagnostic checks for spatial operators used in DTNV reconstructions.

This script runs outside pytest and loads the real patient data specified in a
YAML config (default: configs/config_2bpos.yaml). It verifies three critical
operator groups:
    1. SPECT → PET resampling (enlargement, zoom, deformation resample).
    2. PET bed shifting / combining operators for multi-bed acquisitions.
    3. Block wiring that mixes PET/SPECT components before applying priors.

Each section logs norms, relative errors, and NaN checks so numerical issues can
be spotted quickly when debugging non-convergent behaviour.
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace
from pathlib import Path
import sys
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator

from sirf.STIR import AcquisitionSensitivityModel
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.operators import (
    CouchShiftOperator,
    NiftyResampleOperator,
)
from setr.scripts.common import (
    configure_logging,
    get_resampling_operators,
    get_shift_operators,
    init_run_env,
    apply_combine_sensitivities,
    get_sensitivity_from_subset_objs,
)
from setr.utils.io import apply_overrides, load_config
from setr.utils.sirf import get_array, get_pet_am

# Allow running as a stand-alone script without installing the repository as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))


@dataclass
class AdjointRecord:
    label: str
    forward_inner: float
    adjoint_inner: float
    relative_diff: float


class AdjointnessTracker:
    """Collects adjoint consistency checks for later summarisation."""

    def __init__(self, heading: str):
        self.heading = heading
        self.records: list[AdjointRecord] = []

    def add(self, label: str, forward_inner: float, adjoint_inner: float, rel_diff: float) -> None:
        self.records.append(
            AdjointRecord(label=label, forward_inner=forward_inner, adjoint_inner=adjoint_inner, relative_diff=rel_diff)
        )

    def add_from_inners(self, label: str, forward_inner: float, adjoint_inner: float) -> None:
        denom = max(abs(forward_inner), abs(adjoint_inner), 1e-9)
        rel_diff = abs(forward_inner - adjoint_inner) / denom
        self.add(label, forward_inner, adjoint_inner, rel_diff)

    def log_summary(self) -> None:
        if not self.records:
            logging.info("%s adjoint summary | no checks recorded", self.heading)
            return
        worst = max(self.records, key=lambda rec: rec.relative_diff)
        logging.info(
            "%s adjoint summary | checks=%d worst diff=%.3e (%s)",
            self.heading,
            len(self.records),
            worst.relative_diff,
            worst.label,
        )


class SirfViewerLauncher:
    """Lazy wrapper around sirf_viewer so it stays optional."""

    def __init__(self, enabled: bool, alpha: float = 0.6):
        self.enabled = enabled
        self._alpha = alpha
        self._viewer_cls: Any | None = None
        if not enabled:
            return
        try:
            from sirf_viewer import SIRFViewer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.error("sirf_viewer requested but unavailable: %s", exc)
            self.enabled = False
        else:
            self._viewer_cls = SIRFViewer

    def show(self, image, title: str, background=None) -> None:
        if not self.enabled or self._viewer_cls is None:
            return
        try:
            viewer = self._viewer_cls(image, title=title, background_image=background, alpha=self._alpha)
            viewer.show()
        except Exception as exc:  # pragma: no cover - GUI path
            logging.error("sirf_viewer failed for %s: %s", title, exc)
            self.enabled = False


def _load_prepare_data(mode: str):
    module_name = f"run_dtnv_{mode}"
    spec = importlib.util.spec_from_file_location(
        module_name, _THIS_DIR / f"{module_name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name}.py for mode {mode}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "prepare_data"):
        raise RuntimeError(f"{module_name}.py does not expose prepare_data.")
    return module.prepare_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spatial operator diagnostics using DTNV configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/config_2bpos.yaml",
        help="Path to YAML config used to locate real PET/SPECT data.",
    )
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="*",
        help="Override YAML keys, e.g. output_path=tmp alpha=0.5 beta=0.25",
    )
    parser.add_argument(
        "--sections",
        "-s",
        nargs="+",
        choices=("resampling", "shifts", "block"),
        default=("resampling", "shifts", "block"),
        help="Select which diagnostic sections to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed for RNG that populates synthetic probe images.",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="If set, write intermediate images (hv) for deeper inspection.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open sirf_viewer windows to inspect intermediate ImageData objects.",
    )
    parser.add_argument(
        "--delta-shift-check",
        action="store_true",
        help="Also probe shift/combine operators with delta images to localise leakage.",
    )
    parser.add_argument(
        "--swap-bed-order",
        action="store_true",
        help="Evaluate combine/uncombine by reversing the internal bed order.",
    )
    parser.add_argument(
        "--mode",
        choices=("1bpos", "2bpos"),
        default="2bpos",
        help="Choose which DTNV pipeline to mirror when preparing data.",
    )
    parser.add_argument(
        "--sensitivity-cpu",
        action="store_true",
        help="Use CPU ray-tracing for sensitivity models instead of GPU parallelproj.",
    )
    parser.add_argument(
        "--enlarged-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="Override enlarged shape used before zooming (defaults to 128×256×256).",
    )
    return parser.parse_args()


def _image_norm(image) -> float:
    arr = get_array(image)
    return float(np.linalg.norm(arr.ravel()))


def _image_inner(lhs, rhs) -> float:
    if hasattr(lhs, "containers") and hasattr(rhs, "containers"):
        total = 0.0
        for l, r in zip(lhs.containers, rhs.containers):
            total += _image_inner(l, r)
        return total
    return float(np.vdot(get_array(lhs).ravel(), get_array(rhs).ravel()))


def _random_image_like(image, rng: np.random.Generator):
    if hasattr(image, "containers"):
        samples = [
            _random_image_like(container, rng) for container in image.containers
        ]
        return type(image)(*samples)
    sample = image.copy()
    sample.fill(rng.standard_normal(sample.as_array().shape))
    return sample


def _relative_error(ref, err) -> float:
    denom = max(ref, 1e-9)
    return float(abs(err) / denom)


def _log_image_stats(label: str, image) -> None:
    arr = get_array(image)
    logging.info(
        "%s stats | min=%.6e max=%.6e mean=%.6e std=%.6e norm=%.6e",
        label,
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(arr.std()),
        float(np.linalg.norm(arr.ravel())),
    )


def _log_scaled_image_stats(label: str, image, scale: float) -> None:
    """Log stats for an image after applying a uniform scale (used for SPECT-equivalent views)."""
    arr = get_array(image)
    min_val = float(arr.min())
    max_val = float(arr.max())
    mean_val = float(arr.mean())
    std_val = float(arr.std())
    norm_val = float(np.linalg.norm(arr.ravel()))
    abs_scale = abs(scale)
    if scale >= 0:
        scaled_min = min_val * scale
        scaled_max = max_val * scale
    else:
        scaled_min = max_val * scale
        scaled_max = min_val * scale
    logging.info(
        "%s stats | min=%.6e max=%.6e mean=%.6e std=%.6e norm=%.6e (scale=%.4f)",
        label,
        scaled_min,
        scaled_max,
        mean_val * scale,
        std_val * abs_scale,
        norm_val * abs_scale,
        scale,
    )


def _log_spect_stats(label: str, image, scale: float) -> None:
    """Helper that logs SPECT-domain stats, falling back to raw stats when no scaling is needed."""
    if np.isclose(scale, 1.0):
        _log_image_stats(label, image)
    else:
        _log_scaled_image_stats(label, image, scale)


def _maybe_view(viewer: SirfViewerLauncher | None, image, label: str, background=None) -> None:
    if viewer is None:
        return
    viewer.show(image, label, background=background)


def _save_image(image, dump_dir: Path | None, filename: str):
    if dump_dir is None:
        return
    target = dump_dir / filename
    image.write(str(target))
    logging.info("Wrote %s", target)


def _delta_image(image):
    delta = image.get_uniform_copy(0)
    arr = delta.as_array()
    centre = tuple(int(s // 2) for s in arr.shape)
    arr[centre] = 1.0
    delta.fill(arr)
    return delta


def _build_direct_resampler(pet_data, spect_data):
    displacement = spect_data.get("no_zoom_displacement")
    if displacement is None:
        raise RuntimeError(
            "No SPECT→PET no-zoom displacement available; expected spect2pet_nozoom*.nii in SPECT folder"
        )
    return NiftyResampleOperator(
        reference=pet_data["initial_image"],
        floating=spect_data["initial_image"],
        transform=displacement,
    )


def _compute_bed_sensitivities(pet_data, suffixes, use_gpu=True):
    """Compute per-bed PET sensitivities by running Aᵗ·1 with the PET acquisition model."""
    per_bed = []
    for suffix in suffixes:
        bed = pet_data["bed_positions"][suffix]
        am = get_pet_am(gpu=use_gpu)
        if bed.get("normalisation") is not None:
            asm = AcquisitionSensitivityModel(bed["normalisation"])
            am.set_acquisition_sensitivity(asm)
        if bed.get("additive") is not None:
            am.set_additive_term(bed["additive"])
        am.set_up(bed["acquisition_data"], bed["template_image"])
        ones = bed["acquisition_data"].get_uniform_copy(1)
        sens = am.backward(ones)
        sens.maximum(0, out=sens)
        per_bed.append(sens)
    return per_bed


def _adjoint_rel_diff(
    label,
    op,
    domain_image,
    range_image,
    rng: np.random.Generator,
    tracker: AdjointnessTracker | None = None,
):
    x = _random_image_like(domain_image, rng)
    y = _random_image_like(range_image, rng)
    Ax = op.direct(x)
    Aty = op.adjoint(y)
    lhs = _image_inner(Ax, y)
    rhs = _image_inner(x, Aty)
    denom = max(abs(lhs), abs(rhs), 1e-9)
    diff = abs(lhs - rhs) / denom
    logging.info("%s adjoint rel diff = %.3e", label, diff)
    if tracker is not None:
        tracker.add(label, lhs, rhs, diff)
    return diff


def run_resampling_checks(
    pet_data,
    spect_data,
    rng: np.random.Generator,
    dump_dir: Path | None = None,
    viewer: SirfViewerLauncher | None = None,
):
    logging.info("=" * 80)
    logging.info("Resampling diagnostics (SPECT ➜ PET)")
    spect_img = spect_data["initial_image"]
    pet_img = pet_data["initial_image"]
    _log_image_stats("Initial SPECT", spect_img)
    _log_image_stats("Initial PET", pet_img)
    _maybe_view(viewer, spect_img, "Initial SPECT image")
    _maybe_view(viewer, pet_img, "Initial PET image")
    resampler = _build_direct_resampler(pet_data, spect_data)
    spect_scale = float(getattr(resampler, "scale", 1.0))
    tracker = AdjointnessTracker("Resampling operators [direct-warp]")

    resampled = resampler.direct(spect_img)
    logging.info(
        "Direct-warp | Resampled dims %s vs PET dims %s",
        resampled.dimensions(),
        pet_img.dimensions(),
    )
    logging.info(
        "Direct-warp | PET voxel sizes %s | resampled voxel sizes %s",
        pet_img.voxel_sizes(),
        resampled.voxel_sizes(),
    )
    _log_image_stats("Direct-warp resampled (PET stats)", resampled)
    _log_spect_stats("Direct-warp resampled (SPECT stats)", resampled, spect_scale)
    _maybe_view(viewer, resampled, "Resampled SPECT→PET (direct)", background=pet_img)
    _save_image(resampled, dump_dir, "spect2pet_initial_direct.hv")

    if np.isnan(get_array(resampled)).any():
        logging.warning("Direct-warp | NaNs detected after resampling initial SPECT image.")
    else:
        logging.info("Direct-warp | No NaNs detected in resampled output.")

    probe_spect = _random_image_like(spect_img, rng)
    probe_pet = _random_image_like(pet_img, rng)

    Ax = resampler.direct(probe_spect)
    Aty = resampler.adjoint(probe_pet)
    _log_image_stats("Direct-warp Random Ax", Ax)
    _log_image_stats("Direct-warp Random A^Ty", Aty)
    _save_image(Ax, dump_dir, "resample_Ax_random_direct.hv")
    _save_image(Aty, dump_dir, "resample_At_random_direct.hv")
    inner_forward = _image_inner(Ax, probe_pet)
    inner_adjoint = _image_inner(probe_spect, Aty)

    rel_err = _relative_error(max(abs(inner_forward), abs(inner_adjoint)), inner_forward - inner_adjoint)
    logging.info(
        "Direct-warp | adjoint consistency | <Ax,y>=%.6e, <x,A*y>=%.6e, rel diff=%.3e",
        inner_forward,
        inner_adjoint,
        rel_err,
    )
    tracker.add_from_inners("Full resampler (probe pair)", inner_forward, inner_adjoint)

    backproject = resampler.adjoint(resampled)
    _log_spect_stats("Direct-warp A^T(A initial) (SPECT stats)", backproject, 1.0)
    _maybe_view(viewer, backproject, "Backprojected resampled (direct)", background=spect_img)
    _save_image(backproject, dump_dir, "resample_AtAx_initial_direct.hv")
    logging.info(
        "Direct-warp | Direct/backproject norms | ||Ax||=%.6e, ||A^T(Ax)||=%.6e",
        _image_norm(resampled),
        _image_norm(backproject),
    )

    delta = _delta_image(spect_img)
    delta_resampled = resampler.direct(delta)
    delta_back = resampler.adjoint(delta_resampled)
    _log_image_stats("Direct-warp Delta Ax", delta_resampled)
    _log_image_stats("Direct-warp Delta A^TAx", delta_back)
    _save_image(delta_resampled, dump_dir, "resample_delta_Ax_direct.hv")
    _save_image(delta_back, dump_dir, "resample_delta_AtAx_direct.hv")

    _adjoint_rel_diff("NiftyResample (direct)", resampler, spect_img, pet_img, rng, tracker=tracker)
    tracker.log_summary()


def run_shift_checks(
    pet_data,
    rng: np.random.Generator,
    dump_dir: Path | None = None,
    use_delta: bool = False,
    test_reversed_order: bool = False,
    sensitivity_use_gpu: bool = True,
    viewer: SirfViewerLauncher | None = None,
):
    logging.info("=" * 80)
    logging.info("Shift & combination diagnostics (multi-bed PET)")
    shift_tracker = AdjointnessTracker("Shift operators")

    if "bed_positions" not in pet_data or len(pet_data["bed_positions"]) < 2:
        logging.info("Single-bed dataset detected; skipping multi-bed shift diagnostics.")
        shift_tracker.log_summary()
        return

    suffixes = ["_f1b1", "_f2b1"]
    suffixes = [suffix for suffix in suffixes if suffix in pet_data["bed_positions"]]

    bed_sens = None
    try:
        bed_sens = _compute_bed_sensitivities(
            pet_data,
            suffixes,
            use_gpu=sensitivity_use_gpu,
        )
        logging.info("Computed per-bed sensitivities for combine diagnostics.")
    except Exception as exc:
        logging.warning("Failed to compute combine sensitivities (%s); proceeding unweighted.", exc)

    shifted_templates = []
    template_norms = []

    for suffix in suffixes:
        bed = pet_data["bed_positions"][suffix]
        template = bed["template_image"]
        shift_mm = CouchShiftOperator.get_couch_shift_from_sinogram(bed["acquisition_data"])
        shift_op = CouchShiftOperator(template, shift_mm)
        shifted = shift_op.direct(template)
        _maybe_view(viewer, template, f"{suffix} template image")
        _maybe_view(viewer, shifted, f"{suffix} shifted template")
        _adjoint_rel_diff(
            f"Couch shift [{suffix}]",
            shift_op,
            template,
            shifted,
            rng,
            tracker=shift_tracker,
        )

        orig_off = template.get_geometrical_info().get_offset()[2]
        shifted_off = shifted.get_geometrical_info().get_offset()[2]
        logging.info(
            "[%s] couch shift %.3f mm | z-offset %.3f ➜ %.3f",
            suffix,
            shift_mm,
            orig_off,
            shifted_off,
        )

        back = shift_op.adjoint(shifted)
        ref_norm = _image_norm(template)
        err_norm = _image_norm(back - template)
        rel = _relative_error(ref_norm, err_norm)
        logging.info("[%s] direct∘adjoint drift ||err||/||ref|| = %.3e", suffix, rel)

        shifted_templates.append(shifted)
        template_norms.append(ref_norm)

    shifted_block = EnhancedBlockDataContainer(*shifted_templates)
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)
    combine_op = pet_data.get("combine_operator")
    if combine_op is not None:
        combined_template = combine_op.direct(shifted_block)
        _maybe_view(viewer, combined_template, "Combined PET template")
        _adjoint_rel_diff(
            "Combine operator (unweighted)",
            combine_op,
            shifted_block,
            combined_template,
            rng,
            tracker=shift_tracker,
        )
        if bed_sens is not None:
            apply_combine_sensitivities(pet_data, bed_sens)
            logging.info("Applied sensitivity-weighted combine operator for shift diagnostics.")
            combined_weighted = combine_op.direct(shifted_block)
            _maybe_view(viewer, combined_weighted, "Combined PET template (weighted)")
            _adjoint_rel_diff(
                "Combine operator (weighted)",
                combine_op,
                shifted_block,
                combined_weighted,
                rng,
                tracker=shift_tracker,
            )

    from setr.cil_extensions.operators import ImageCombineOperator, AdjointOperator as _AdjointOperator

    def evaluate_block(
        label: str,
        block: EnhancedBlockDataContainer,
        ref_norms: list[float],
        suffix_labels: list[str],
        uncombine,
        choose_ops_list=None,
        unshift_ops_list=None,
    ):
        logging.info("--- %s ---", label)
        combined = uncombine.adjoint(block)
        _save_image(combined, dump_dir, f"combined_{label}.hv")
        _maybe_view(viewer, combined, f"Combined block {label}")
        recovered = uncombine.direct(combined)
        for idx, suffix in enumerate(suffix_labels):
            err = _image_norm(recovered[idx] - block[idx])
            rel = _relative_error(ref_norms[idx], err)
            logging.info("[%s] combine+uncombine rel err = %.3e", suffix, rel)
            if choose_ops_list is not None:
                selection = choose_ops_list[idx].direct(block)
                selection_img = selection.containers[0] if hasattr(selection, "containers") else selection
                sel_err = _image_norm(selection_img - block[idx])
                logging.info("[%s] selector rel err = %.3e", suffix, _relative_error(ref_norms[idx], sel_err))

        if unshift_ops_list is not None:
            random_beds = EnhancedBlockDataContainer(
                *[_random_image_like(pet_data["bed_positions"][suffix]["template_image"], rng) for suffix in suffix_labels]
            )
            shifted_random = EnhancedBlockDataContainer(
                *[unshift_ops_list[idx].adjoint(random_beds[idx]) for idx in range(len(suffix_labels))]
            )
            combined_random = uncombine.adjoint(shifted_random)
            recovered_random = uncombine.direct(combined_random)
            for idx, suffix in enumerate(suffix_labels):
                err = _image_norm(recovered_random[idx] - shifted_random[idx])
                rel = _relative_error(_image_norm(shifted_random[idx]), err)
                logging.info("[%s] κ/sensitivity reprojection rel err = %.3e", suffix, rel)

    evaluate_block(
        "templates",
        shifted_block,
        template_norms,
        suffixes,
        uncombine_op,
        choose_ops_list=choose_ops,
        unshift_ops_list=unshift_ops,
    )

    if use_delta:
        delta_images = []
        delta_norms = []
        for suffix in suffixes:
            delta_img = _delta_image(pet_data["bed_positions"][suffix]["template_image"])
            delta_images.append(delta_img)
            delta_norms.append(_image_norm(delta_img))
        evaluate_block(
            "delta",
            EnhancedBlockDataContainer(*delta_images),
            delta_norms,
            suffixes,
            uncombine_op,
            choose_ops_list=choose_ops,
            unshift_ops_list=unshift_ops,
        )

    if test_reversed_order:
        reversed_block = EnhancedBlockDataContainer(*shifted_templates[::-1])
        reversed_norms = list(reversed(template_norms))
        reversed_suffixes = list(reversed(suffixes))
        reversed_combine = ImageCombineOperator(EnhancedBlockDataContainer(*shifted_templates[::-1]))
        reversed_uncombine = _AdjointOperator(reversed_combine)
        reversed_combined = reversed_combine.direct(reversed_block)
        _adjoint_rel_diff(
            "Combine operator (reversed)",
            reversed_combine,
            reversed_block,
            reversed_combined,
            rng,
            tracker=shift_tracker,
        )
        evaluate_block(
            "templates_reversed",
            reversed_block,
            reversed_norms,
            reversed_suffixes,
            reversed_uncombine,
        )
    shift_tracker.log_summary()


def run_block_checks(
    pet_data,
    spect_data,
    initial_estimates,
    rng: np.random.Generator,
    viewer: SirfViewerLauncher | None = None,
    args: Any | None = None,
):
    logging.info("=" * 80)
    logging.info("Block operator diagnostics (PET/SPECT coupling)")
    block_tracker = AdjointnessTracker("Block operators")
    if args is None:
        spect2pet = get_resampling_operators(pet_data, spect_data)
    else:
        spect2pet = get_resampling_operators(args, pet_data, spect_data)
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    combined = EnhancedBlockDataContainer(*bo.direct(initial_estimates).containers)
    _maybe_view(viewer, combined[0], "Block PET passthrough")

    pet_err = _image_norm(combined[0] - initial_estimates[0])
    pet_rel = _relative_error(_image_norm(initial_estimates[0]), pet_err)
    logging.info("PET channel passthrough rel err = %.3e", pet_rel)

    spect_resampled = spect2pet.direct(initial_estimates[1])
    spect_err = _image_norm(combined[1] - spect_resampled)
    logging.info("SPECT block resampling rel err = %.3e", _relative_error(_image_norm(spect_resampled), spect_err))
    _maybe_view(viewer, spect_resampled, "Block SPECT→PET resampled", background=initial_estimates[0])

    probe = EnhancedBlockDataContainer(
        _random_image_like(initial_estimates[0], rng),
        _random_image_like(initial_estimates[1], rng),
    )
    probe_dual = EnhancedBlockDataContainer(
        _random_image_like(initial_estimates[0], rng),
        _random_image_like(spect_resampled, rng),
    )

    Ax = EnhancedBlockDataContainer(*bo.direct(probe).containers)
    Aty = EnhancedBlockDataContainer(*bo.adjoint(probe_dual).containers)

    lhs = sum(_image_inner(Ax[i], probe_dual[i]) for i in range(2))
    rhs = sum(_image_inner(probe[i], Aty[i]) for i in range(2))
    rel = _relative_error(max(abs(lhs), abs(rhs)), lhs - rhs)
    logging.info("Block operator adjoint consistency rel diff = %.3e", rel)
    block_tracker.add("Block operator", lhs, rhs, rel)
    block_tracker.log_summary()


def main():
    cli = _parse_args()
    configure_logging()

    config = apply_overrides(load_config(cli.config), cli.override)
    args = SimpleNamespace(**config)

    _ = init_run_env(args)
    prepare_data_fn = _load_prepare_data(cli.mode)
    if cli.mode == "2bpos":
        _, pet_data, spect_data, initial_estimates = prepare_data_fn(args)
    else:
        _, pet_data, spect_data = prepare_data_fn(args)
        initial_estimates = EnhancedBlockDataContainer(
            pet_data["initial_image"], spect_data["initial_image"]
        )
    rng = np.random.default_rng(cli.seed)
    dump_dir = Path(cli.dump_dir).resolve() if cli.dump_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    viewer = SirfViewerLauncher(cli.viewer) if getattr(cli, "viewer", False) else None

    if "resampling" in cli.sections:
        run_resampling_checks(pet_data, spect_data, rng, dump_dir, viewer=viewer)
    if "shifts" in cli.sections:
        run_shift_checks(
            pet_data,
            rng,
            dump_dir=dump_dir,
            use_delta=cli.delta_shift_check,
            test_reversed_order=cli.swap_bed_order,
            sensitivity_use_gpu=not cli.sensitivity_cpu,
            viewer=viewer,
        )
    if "block" in cli.sections:
        run_block_checks(
            pet_data,
            spect_data,
            initial_estimates,
            rng,
            viewer=viewer,
            args=args,
        )

    logging.info("=" * 80)
    logging.info("Spatial diagnostics complete.")


if __name__ == "__main__":
    main()
