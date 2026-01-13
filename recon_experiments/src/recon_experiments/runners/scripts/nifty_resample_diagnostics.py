#!/usr/bin/env python3
"""Deep-dive diagnostics for SIRF NiftyResample with SETR displacement fields."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import importlib.util

import numpy as np
from sirf.Reg import NiftyResample

from recon_experiments.runners.common import configure_logging, init_run_env
from recon_core.utils.io import apply_overrides, load_config
from recon_core.utils.sirf import get_array

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))


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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse NiftyResample forward/backward consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/config_2bpos.yaml",
        help="Config pointing to PET/SPECT data.",
    )
    parser.add_argument(
        "--override",
        "-o",
        nargs="*",
        help="Override YAML keys (key=value).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19,
        help="Seed for RNG probes.",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Optional directory to save intermediate Images.",
    )
    parser.add_argument(
        "--identity-check",
        action="store_true",
        help="Also evaluate NiftyResample with zero displacement (identity).",
    )
    parser.add_argument(
        "--transform-path",
        type=str,
        default=None,
        help="Override SPECT→PET displacement file.",
    )
    parser.add_argument(
        "--mode",
        choices=("1bpos", "2bpos"),
        default="2bpos",
        help="Choose which DTNV pipeline to mirror when loading data.",
    )
    return parser.parse_args()


def _log_image_stats(label: str, image):
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


def _save(image, dump_dir: Path | None, name: str):
    if dump_dir is None:
        return
    dest = dump_dir / name
    image.write(str(dest))
    logging.info("Wrote %s", dest)


def _random_image_like(image, rng: np.random.Generator):
    sample = image.get_uniform_copy(0)
    arr = rng.standard_normal(sample.as_array().shape)
    sample.fill(arr)
    return sample


def _summarise_displacement(transform) -> None:
    if transform is None:
        logging.warning("No displacement field available.")
        return
    arr = transform.as_array()
    # arr shape (3, z, y, x)
    mags = np.linalg.norm(arr, axis=0)
    logging.info(
        "Displacement magnitude | min=%.3f mm max=%.3f mm mean=%.3f mm std=%.3f mm",
        float(mags.min()),
        float(mags.max()),
        float(mags.mean()),
        float(mags.std()),
    )


def _build_resampler(pet_data, spect_data, displacement):
    floating = spect_data["initial_image"]
    resampler = NiftyResample()
    resampler.set_reference_image(pet_data["initial_image"])
    resampler.set_floating_image(floating)
    resampler.set_interpolation_type_to_linear()
    resampler.set_padding_value(0)
    if displacement is not None:
        resampler.add_transformation(displacement)
    return resampler, floating


def _adjoint_probe(label, resampler, floating_img, reference_img, rng, dump_dir: Path | None):
    logging.info("--- %s ---", label)
    x = _random_image_like(floating_img, rng)
    y = _random_image_like(reference_img, rng)
    Ax = resampler.forward(x)
    Aty = resampler.backward(y)

    _log_image_stats("Ax", Ax)
    _log_image_stats("A^Ty", Aty)
    _save(Ax, dump_dir, f"{label}_Ax.hv")
    _save(Aty, dump_dir, f"{label}_ATy.hv")

    lhs = float(np.vdot(get_array(Ax).ravel(), get_array(y).ravel()))
    rhs = float(np.vdot(get_array(x).ravel(), get_array(Aty).ravel()))
    denom = max(abs(lhs), abs(rhs), 1e-9)
    logging.info(
        "%s adjoint diff | <Ax,y>=%.6e <x,A^Ty>=%.6e rel diff=%.3e",
        label,
        lhs,
        rhs,
        abs(lhs - rhs) / denom,
    )

    delta = floating_img.get_uniform_copy(0)
    arr = delta.as_array()
    centre = tuple(int(s // 2) for s in arr.shape)
    arr[centre] = 1.0
    delta.fill(arr)
    Ax_delta = resampler.forward(delta)
    back_delta = resampler.backward(Ax_delta)
    _log_image_stats("Delta Ax", Ax_delta)
    _log_image_stats("Delta A^TAx", back_delta)
    _save(Ax_delta, dump_dir, f"{label}_delta_Ax.hv")
    _save(back_delta, dump_dir, f"{label}_delta_AtAx.hv")


def main():
    cli = _parse_args()
    configure_logging()
    dump_dir = Path(cli.dump_dir).resolve() if cli.dump_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_overrides(load_config(cli.config), cli.override)
    args = argparse.Namespace(**cfg)
    _ = init_run_env(args)

    prepare_fn = _load_prepare_data(cli.mode)
    if cli.mode == "2bpos":
        _, pet_data, spect_data, _ = prepare_fn(args)
    else:
        _, pet_data, spect_data = prepare_fn(args)
    if cli.transform_path:
        from sirf.Reg import NiftiImageData3DDisplacement

        spect_data["no_zoom_displacement"] = NiftiImageData3DDisplacement(cli.transform_path)

    displacement = spect_data.get("no_zoom_displacement")
    _summarise_displacement(displacement)

    rng = np.random.default_rng(cli.seed)
    resampler, floating = _build_resampler(pet_data, spect_data, displacement)
    _adjoint_probe("actual", resampler, floating, pet_data["initial_image"], rng, dump_dir)

    if cli.identity_check:
        logging.info("Running identity displacement check...")
        identity_resampler, identity_floating = _build_resampler(pet_data, spect_data, None)
        _adjoint_probe("identity", identity_resampler, identity_floating, pet_data["initial_image"], rng, dump_dir)


if __name__ == "__main__":
    main()
