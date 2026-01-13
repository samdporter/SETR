#!/usr/bin/env python3
"""Compare composed blur+SIRF acquisition model against native image-based PSF.

This script builds two SPECT forward models using the NEMA phantom data:
1) CompositionOperator(GaussianBlurringOperator, AcquisitionModel(no PSF))
2) AcquisitionModel(with image-based Gaussian PSF)

It compares both the forward projection and the back-projected image
(A* A x) using the same FWHM and resolution modelling parameters and
emits slice plots with x/y profiles through the local maxima.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cil.optimisation.operators import CompositionOperator, LinearOperator

# Ensure local imports work when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_core.cil_extensions.operators.blurring import GaussianBlurringOperator
from recon_core.utils import get_spect_am, get_spect_data
from recon_core.utils.sirf import get_array

try:
    from sirf.STIR import AcquisitionData, ImageData, MessageRedirector
except Exception as exc:  # pragma: no cover - runtime import
    raise SystemExit(f"SIRF/STIR is required for this script: {exc}")


class SIRFAcquisitionModelOperator(LinearOperator):
    """CIL LinearOperator wrapper for SIRF AcquisitionModel forward/backward."""

    def __init__(self, am, acq_data, image):
        super().__init__(domain_geometry=image, range_geometry=acq_data)
        self.am = am

    def direct(self, x, out=None):
        result = self.am.forward(x)
        if out is None:
            return result
        out.fill(get_array(result))
        return out

    def adjoint(self, y, out=None):
        result = self.am.backward(y)
        if out is None:
            return result
        out.fill(get_array(result))
        return out


def _parse_triplet(values: str, name: str) -> List[float]:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must have 3 comma-separated values, got {values!r}")
    return [float(p) for p in parts]


def _parse_spect_res(values: str) -> List[object]:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"spect-res must have 3 comma-separated values, got {values!r}")
    slope = float(parts[0])
    intercept = float(parts[1])
    use_psf = parts[2].lower() in {"1", "true", "t", "yes", "y"}
    return [slope, intercept, use_psf]


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.linalg.norm(a - b)
    denom = max(np.linalg.norm(b), 1e-12)
    return float(diff / denom)


def _random_like(image: ImageData, seed: int) -> ImageData:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(get_array(image).shape).astype(np.float32)
    out = image.clone()
    out.fill(arr)
    return out


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare composed blur+AM against native AM with PSF (SPECT NEMA)."
    )
    parser.add_argument(
        "--spect-data-path",
        default="/home/storage/prepared_data/phantom_data/nema_phantom_data/SPECT",
        help="Path to NEMA SPECT data (default: %(default)s).",
    )
    parser.add_argument(
        "--gauss-fwhm",
        default="6.7,6.7,6.7",
        help="Gaussian FWHM in mm, as 'z,y,x' (default: %(default)s).",
    )
    parser.add_argument(
        "--spect-res",
        default="1.31,0.027,false",
        help="Collimator resolution params 'slope,intercept,use_psf' (default: %(default)s).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "numba", "scipy"],
        default="auto",
        help="Backend for Gaussian blurring operator (default: %(default)s).",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional image to project instead of initial_image.hv.",
    )
    parser.add_argument(
        "--use-random-image",
        action="store_true",
        help="Use a random image instead of initial_image.hv.",
    )
    parser.add_argument(
        "--point-source",
        action="store_true",
        help="Use a point source image instead of initial_image.hv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random image (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/psf_compare",
        help="Directory to save plots (default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of just saving them.",
    )
    return parser.parse_args(list(argv))


def _find_max_in_slice(slice_2d: np.ndarray) -> Tuple[int, int]:
    flat_idx = int(np.argmax(slice_2d))
    return np.unravel_index(flat_idx, slice_2d.shape)


def _plot_compare_slices_with_profiles(
    slice_a: np.ndarray,
    slice_b: np.ndarray,
    slice_c: np.ndarray,
    label_a: str,
    label_b: str,
    label_c: str,
    title: str,
    out_path: str,
    show: bool,
) -> None:
    y_idx, x_idx = _find_max_in_slice(slice_a)
    x_profile_a = slice_a[y_idx, :]
    y_profile_a = slice_a[:, x_idx]
    x_profile_b = slice_b[y_idx, :]
    y_profile_b = slice_b[:, x_idx]
    x_profile_c = slice_c[y_idx, :]
    y_profile_c = slice_c[:, x_idx]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax_img_a, ax_img_b, ax_img_c = axes[0]
    ax_prof_x, ax_prof_y, ax_dummy = axes[1]
    ax_dummy.axis("off")

    im_a = ax_img_a.imshow(slice_a, cmap="inferno", origin="lower")
    ax_img_a.plot(x_idx, y_idx, "c+", markersize=8, markeredgewidth=1.5)
    ax_img_a.set_title(label_a)
    fig.colorbar(im_a, ax=ax_img_a, fraction=0.046, pad=0.04)

    im_b = ax_img_b.imshow(slice_b, cmap="inferno", origin="lower")
    ax_img_b.plot(x_idx, y_idx, "c+", markersize=8, markeredgewidth=1.5)
    ax_img_b.set_title(label_b)
    fig.colorbar(im_b, ax=ax_img_b, fraction=0.046, pad=0.04)

    im_c = ax_img_c.imshow(slice_c, cmap="inferno", origin="lower")
    ax_img_c.plot(x_idx, y_idx, "c+", markersize=8, markeredgewidth=1.5)
    ax_img_c.set_title(label_c)
    fig.colorbar(im_c, ax=ax_img_c, fraction=0.046, pad=0.04)

    ax_prof_x.plot(x_profile_a, color="tab:blue", label=label_a)
    ax_prof_x.plot(x_profile_b, color="tab:orange", label=label_b)
    ax_prof_x.plot(x_profile_c, color="tab:purple", label=label_c)
    ax_prof_x.set_title(f"{title} x profile")
    ax_prof_x.set_xlabel("x")
    ax_prof_x.set_ylabel("intensity")
    ax_prof_x.legend()

    ax_prof_y.plot(y_profile_a, color="tab:green", label=label_a)
    ax_prof_y.plot(y_profile_b, color="tab:red", label=label_b)
    ax_prof_y.plot(y_profile_c, color="tab:brown", label=label_c)
    ax_prof_y.set_title(f"{title} y profile")
    ax_prof_y.set_xlabel("y")
    ax_prof_y.set_ylabel("intensity")
    ax_prof_y.legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    logging.info("Wrote %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_image_domain_compare(
    image_a: np.ndarray,
    image_b: np.ndarray,
    image_c: np.ndarray,
    label_a: str,
    label_b: str,
    label_c: str,
    out_dir: str,
    show: bool,
) -> None:
    if image_a.ndim == 4:
        image_a = image_a[0]
    if image_b.ndim == 4:
        image_b = image_b[0]
    if image_a.ndim != 3 or image_b.ndim != 3 or image_c.ndim != 3:
        raise ValueError(
            "Expected 3D image arrays, got shapes "
            f"{image_a.shape}, {image_b.shape}, {image_c.shape}"
        )
    max_idx = np.unravel_index(int(np.argmax(image_a)), image_a.shape)
    z_idx = max_idx[0]
    slice_a = image_a[z_idx, :, :]
    slice_b = image_b[z_idx, :, :]
    slice_c = image_c[z_idx, :, :]
    out_path = os.path.join(out_dir, f"image_compare_z{z_idx}.png")
    _plot_compare_slices_with_profiles(
        slice_a,
        slice_b,
        slice_c,
        label_a,
        label_b,
        label_c,
        f"Image z={z_idx}",
        out_path,
        show,
    )


def _plot_projection_slices_compare(
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    proj_c: np.ndarray,
    label_a: str,
    label_b: str,
    label_c: str,
    out_dir: str,
    show: bool,
) -> None:
    if proj_a.ndim == 4:
        proj_a = proj_a[0]
    if proj_b.ndim == 4:
        proj_b = proj_b[0]
    if proj_c.ndim == 4:
        proj_c = proj_c[0]
    if proj_a.ndim != 3 or proj_b.ndim != 3 or proj_c.ndim != 3:
        raise ValueError(
            "Expected 4D or 3D projection arrays, got shapes "
            f"{proj_a.shape}, {proj_b.shape}, {proj_c.shape}"
        )

    # Assume (axial, projection, radial) ordering once the leading dim is fixed at 0.
    axial_idx = proj_a.shape[0] // 2
    proj_idx = proj_a.shape[1] // 2
    radial_idx = proj_a.shape[2] // 2

    axial_a = proj_a[axial_idx, :, :]
    axial_b = proj_b[axial_idx, :, :]
    axial_c = proj_c[axial_idx, :, :]
    proj_a_slice = proj_a[:, proj_idx, :]
    proj_b_slice = proj_b[:, proj_idx, :]
    proj_c_slice = proj_c[:, proj_idx, :]
    radial_a = proj_a[:, :, radial_idx]
    radial_b = proj_b[:, :, radial_idx]
    radial_c = proj_c[:, :, radial_idx]

    _plot_compare_slices_with_profiles(
        axial_a,
        axial_b,
        axial_c,
        label_a,
        label_b,
        label_c,
        f"Projection axial={axial_idx}",
        os.path.join(out_dir, f"projection_compare_axial{axial_idx}.png"),
        show,
    )
    _plot_compare_slices_with_profiles(
        proj_a_slice,
        proj_b_slice,
        proj_c_slice,
        label_a,
        label_b,
        label_c,
        f"Projection view={proj_idx}",
        os.path.join(out_dir, f"projection_compare_view{proj_idx}.png"),
        show,
    )
    _plot_compare_slices_with_profiles(
        radial_a,
        radial_b,
        radial_c,
        label_a,
        label_b,
        label_c,
        f"Projection radial={radial_idx}",
        os.path.join(out_dir, f"projection_compare_radial{radial_idx}.png"),
        show,
    )


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Reduce STIR verbosity and use memory storage.
    MessageRedirector()
    AcquisitionData.set_storage_scheme("memory")
    os.makedirs(args.output_dir, exist_ok=True)

    spect_data = get_spect_data(args.spect_data_path)
    acq = spect_data["acquisition_data"]

    if args.image_path:
        image = ImageData(args.image_path)
    else:
        image = spect_data["initial_image"]

    if args.point_source:
        arr = np.zeros(get_array(image).shape, dtype=np.float32)
        center = tuple(s // 2 for s in arr.shape)
        arr[center] = 1.0
        image = image.clone()
        image.fill(arr)
    elif args.use_random_image:
        image = _random_like(image, args.seed)

    gauss_fwhm = _parse_triplet(args.gauss_fwhm, "gauss-fwhm")
    spect_res = _parse_spect_res(args.spect_res)

    am_native = get_spect_am(
        spect_data,
        res=spect_res,
        keep_all_views_in_cache=True,
        gauss_fwhm=gauss_fwhm,
        attenuation=True,
    )
    am_plain = get_spect_am(
        spect_data,
        res=spect_res,
        keep_all_views_in_cache=True,
        gauss_fwhm=None,
        attenuation=True,
    )

    am_native.set_up(acq, image)
    am_plain.set_up(acq, image)

    blur = GaussianBlurringOperator(gauss_fwhm, image, backend=args.backend)
    am_plain_op = SIRFAcquisitionModelOperator(am_plain, acq, image)
    composed = CompositionOperator(am_plain_op, blur)

    proj_comp = composed.direct(image)
    proj_native = am_native.forward(image)
    proj_plain = am_plain.forward(image)

    proj_comp_arr = get_array(proj_comp)
    proj_native_arr = get_array(proj_native)
    proj_plain_arr = get_array(proj_plain)

    back_comp = composed.adjoint(proj_comp)
    back_native = am_native.backward(proj_native)
    back_plain = am_plain.backward(proj_plain)

    back_comp_arr = get_array(back_comp)
    back_native_arr = get_array(back_native)
    back_plain_arr = get_array(back_plain)

    _plot_image_domain_compare(
        back_comp_arr,
        back_native_arr,
        back_plain_arr,
        "composed",
        "native",
        "no_blur",
        args.output_dir,
        args.show,
    )
    _plot_projection_slices_compare(
        proj_comp_arr,
        proj_native_arr,
        proj_plain_arr,
        "composed",
        "native",
        "no_blur",
        args.output_dir,
        args.show,
    )

    logging.info("Relative L2 (forward vs native): %.4e", _relative_l2(proj_comp_arr, proj_native_arr))
    logging.info("Relative L2 (forward vs no_blur): %.4e", _relative_l2(proj_comp_arr, proj_plain_arr))
    logging.info("Relative L2 (adjoint vs native): %.4e", _relative_l2(back_comp_arr, back_native_arr))
    logging.info("Relative L2 (adjoint vs no_blur): %.4e", _relative_l2(back_comp_arr, back_plain_arr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
