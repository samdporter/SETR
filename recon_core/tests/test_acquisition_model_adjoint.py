#!/usr/bin/env python3
"""
Tests mirroring the standalone SIRF PSF behaviour checks and exercising
the custom Gaussian blurring operator used in the CIL forward model.
These tests are skipped automatically when SIRF (and its example data)
is not available in the environment.
"""

import os
from typing import Dict

import numpy as np
import pytest
import tempfile

sirf = pytest.importorskip(
    "sirf.STIR", reason="SIRF is required for PSF behaviour tests"
)
from sirf.STIR import (  # type: ignore # pragma: no cover - runtime import
    AcquisitionData,
    AcquisitionModelUsingRayTracingMatrix,
    AcquisitionModelUsingParallelproj,
    AcquisitionModelUsingMatrix,
    SPECTUBMatrix,
    ImageData,
    SeparableGaussianImageFilter,
    examples_data_path,
    MessageRedirector
)

from recon_core.cil_extensions.operators.blurring import GaussianBlurringOperator
from recon_core.utils.sirf import create_spect_uniform_image

msg = MessageRedirector()

def create_gaussian_blur(fwhm, template_image):
    """Helper to create GaussianBlurringOperator."""
    return GaussianBlurringOperator(fwhm, template_image, backend='auto')

def create_point_source_image(template_image):
    """Create a point source image based on the template."""
    point_image = template_image.clone()
    arr = np.zeros(point_image.as_array().shape, dtype=np.float32)
    center = tuple(s // 2 for s in arr.shape)
    arr[center] = 1_000.0
    point_image.fill(arr)
    return point_image

def _load_pet_data() -> Dict[str, ImageData]:
    """Load acquisition/image templates from SIRF examples or skip."""
    try:
        data_root = examples_data_path("PET")
    except Exception as exc:  # pragma: no cover - guarded by skip below
        pytest.skip(f"SIRF PET example data unavailable: {exc}")

    acq_path = os.path.join(data_root, "simulated_data.hs")
    img_path = os.path.join(data_root, "test_image_PM_QP_6.hv")

    if not (os.path.exists(acq_path) and os.path.exists(img_path)):
        pytest.skip(
            f"SIRF PET example files not found at {data_root}: "
            f"{os.path.basename(acq_path)}, {os.path.basename(img_path)}"
        )

    acquisition_data = AcquisitionData(acq_path)
    initial_image = ImageData(img_path)

    point_image = create_point_source_image(initial_image)

    return {
        "acquisition_data": acquisition_data,
        "initial_image": initial_image,
        "point_source": point_image,
    }

def _load_spect_data() -> Dict[str, ImageData]:
    """Load acquisition/image templates from local SPECT data or skip."""
    # Try to find SPECT data in the project data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_root = os.path.join(project_root, "data")

    acq_path = os.path.join(data_root, "template_sinogram.hs")
    img_path = os.path.join(data_root, "template_image.hv")

    # Check if the data files exist
    if not os.path.exists(data_root):
        pytest.skip(f"SPECT data directory not found at {data_root}")

    if not os.path.exists(acq_path):
        pytest.skip(f"SPECT acquisition template not found: {acq_path}")

    if not os.path.exists(img_path):
        pytest.skip(f"SPECT image template not found: {img_path}")

    try:
        acquisition_data = AcquisitionData(acq_path)
        initial_image = ImageData(img_path)
    except Exception as exc:
        pytest.skip(f"Failed to load SPECT data: {exc}")

    point_image = create_point_source_image(initial_image)

    return {
        "acquisition_data": acquisition_data,
        "initial_image": initial_image,
        "point_source": point_image,
    }


@pytest.fixture(scope="module")
def sirf_pet_data():
    """Provide cached PET acquisition/image data for the PSF tests."""
    return _load_pet_data()


@pytest.fixture(scope="module")
def sirf_spect_data():
    """Provide cached SPECT acquisition/image data for the PSF tests."""
    return _load_spect_data()


@pytest.fixture(
    params=[
        ("pet", AcquisitionModelUsingParallelproj),
        ("pet", AcquisitionModelUsingRayTracingMatrix),
        ("spect", AcquisitionModelUsingMatrix),
    ],
    ids=["PET-parallelproj", "PET-raytracing", "SPECT-matrix"],
)
def modality_and_model(request, sirf_pet_data, sirf_spect_data):
    """Provide modality data and acquisition model class together."""
    modality, model_cls = request.param
    data = sirf_pet_data if modality == "pet" else sirf_spect_data
    return {
        "data": data,
        "modality": modality,
        "model_cls": model_cls,
    }


def setup_acquisition_model(model_cls, modality, acq, initial, psf_fwhm=None):
    """Set up acquisition model with optional PSF."""
    am = model_cls()

    # SPECT needs matrix setup
    if modality == "spect" and model_cls == AcquisitionModelUsingMatrix:
        am.set_matrix(SPECTUBMatrix())

    # Add PSF if specified
    if psf_fwhm is not None:
        psf = SeparableGaussianImageFilter()
        psf.set_fwhms(psf_fwhm)
        am.set_image_data_processor(psf)

    am.set_up(acq, initial)
    return am


@pytest.fixture(
    params=[
        AcquisitionModelUsingParallelproj,
        AcquisitionModelUsingRayTracingMatrix,
        AcquisitionModelUsingMatrix,
    ],
    ids=["parallelproj", "ray_tracing", "matrix"],
)
def acquisition_model_cls(request):
    """Yield acquisition model classes to exercise both STIR projectors."""
    return request.param


@pytest.mark.parametrize("fwhm", ([9, 9, 9], [21, 21, 21]))
def test_acquisition_model_psf_forward_projection(modality_and_model, fwhm):
    """Ensure PSF modifies forward projections."""
    data = modality_and_model["data"]
    modality = modality_and_model["modality"]
    model_cls = modality_and_model["model_cls"]

    acq = data["acquisition_data"]
    initial = data["initial_image"]
    point = data["point_source"]

    am_with_psf = setup_acquisition_model(model_cls, modality, acq, initial, psf_fwhm=fwhm)
    am_without_psf = setup_acquisition_model(model_cls, modality, acq, initial)

    proj_with = am_with_psf.forward(point)
    proj_without = am_without_psf.forward(point)

    diff_norm = (proj_with - proj_without).norm()
    rel = diff_norm / max(proj_without.norm(), 1e-6)

    assert diff_norm > 1e-6
    assert rel > 1e-3


def test_acquisition_model_psf_backward_projection(modality_and_model):
    """PSF should propagate through backward projection."""
    data = modality_and_model["data"]
    modality = modality_and_model["modality"]
    model_cls = modality_and_model["model_cls"]

    acq = data["acquisition_data"]
    initial = data["initial_image"]

    am_with_psf = setup_acquisition_model(model_cls, modality, acq, initial, psf_fwhm=[21, 21, 21])
    am_without_psf = setup_acquisition_model(model_cls, modality, acq, initial)

    rng = np.random.default_rng(0)
    random_proj = acq.clone()
    random_proj.fill(rng.standard_normal(random_proj.shape).astype(np.float32))

    back_with_psf = am_with_psf.backward(random_proj)
    back_without_psf = am_without_psf.backward(random_proj)

    diff = (back_with_psf - back_without_psf).norm()
    rel = diff / max(back_without_psf.norm(), 1e-6)

    assert diff > 1e-6, f"{modality.upper()}/{model_cls.__name__} PSF backward diff too small"
    assert rel > 1e-3, f"{modality.upper()}/{model_cls.__name__} PSF backward rel diff too small"


def test_acquisition_model_psf_fwhm_scaling(modality_and_model):
    """Larger FWHM should produce stronger blurring effects."""
    data = modality_and_model["data"]
    modality = modality_and_model["modality"]
    model_cls = modality_and_model["model_cls"]

    acq = data["acquisition_data"]
    initial = data["initial_image"]
    point = data["point_source"]

    am_small = setup_acquisition_model(model_cls, modality, acq, initial, psf_fwhm=[4, 4, 4])
    am_large = setup_acquisition_model(model_cls, modality, acq, initial, psf_fwhm=[21, 21, 21])
    am_none = setup_acquisition_model(model_cls, modality, acq, initial)

    diff_small = (am_small.forward(point) - am_none.forward(point)).norm()
    diff_large = (am_large.forward(point) - am_none.forward(point)).norm()

    assert diff_large > diff_small
    assert (diff_large / max(diff_small, 1e-6)) > 1


def test_gaussian_blur_operator_changes_forward_projection_with_model(modality_and_model):
    """Our custom Gaussian operator should alter the forward model."""
    data = modality_and_model["data"]
    modality = modality_and_model["modality"]
    model_cls = modality_and_model["model_cls"]

    acq = data["acquisition_data"]
    initial = data["initial_image"]
    point = data["point_source"]

    blur = create_gaussian_blur((6.0, 6.0, 6.0), initial)
    am = setup_acquisition_model(model_cls, modality, acq, initial)

    proj_blurred = am.forward(blur.direct(point))
    proj_plain = am.forward(point)

    diff = (proj_blurred - proj_plain).norm()
    rel = diff / max(proj_plain.norm(), 1e-6)

    assert diff > 1e-6 and rel > 1e-3


@pytest.mark.parametrize("modality", ["pet", "spect"], ids=["PET", "SPECT"])
def test_gaussian_blur_operator_adjoint(modality, sirf_pet_data, sirf_spect_data):
    """Validate <Ax, y> == <x, A*y> for the Gaussian blurring operator."""
    data = sirf_pet_data if modality == "pet" else sirf_spect_data
    initial = data["initial_image"]
    blur = create_gaussian_blur((6.0, 6.0, 6.0), initial)

    rng = np.random.default_rng(1)
    x = initial.clone()
    x.fill(rng.standard_normal(x.shape).astype(np.float32))
    y = initial.clone()
    y.fill(rng.standard_normal(y.shape).astype(np.float32))

    Ax = blur.direct(x)
    Aty = blur.adjoint(y)

    lhs = float(np.vdot(Ax.as_array().ravel(), y.as_array().ravel()))
    rhs = float(np.vdot(x.as_array().ravel(), Aty.as_array().ravel()))

    assert np.isclose(lhs, rhs, rtol=5e-3, atol=1e-3)


def test_acquisition_model_with_psf_adjoint_check(modality_and_model):
    """The native PSF pipeline should remain adjoint-consistent."""
    data = modality_and_model["data"]
    modality = modality_and_model["modality"]
    model_cls = modality_and_model["model_cls"]

    acq = data["acquisition_data"]
    initial = data["initial_image"]

    am = setup_acquisition_model(model_cls, modality, acq, initial, psf_fwhm=[9, 9, 9])

    rng = np.random.default_rng(2)
    x = initial.clone()
    x.fill(rng.standard_normal(x.shape).astype(np.float32))
    y = acq.clone()
    y.fill(rng.standard_normal(y.shape).astype(np.float32))

    Ax = am.forward(x)
    Aty = am.backward(y)

    lhs = float(np.vdot(Ax.as_array().ravel(), y.as_array().ravel()))
    rhs = float(np.vdot(x.as_array().ravel(), Aty.as_array().ravel()))

    rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-6)
    assert rel_diff < 1e-1, f"{modality.upper()}/{model_cls.__name__} violates adjoint with PSF (rel_diff={rel_diff})"


def test_gaussian_blurred_acquisition_model_adjoint_with_model(modality_and_model):
    """Our composed (blur + projection) forward model should remain adjoint-consistent."""
    data = modality_and_model["data"]
    modality = modality_and_model["modality"]
    model_cls = modality_and_model["model_cls"]

    acq = data["acquisition_data"]
    initial = data["initial_image"]

    blur = create_gaussian_blur((6.0, 6.0, 6.0), initial)
    am = setup_acquisition_model(model_cls, modality, acq, initial)

    rng = np.random.default_rng(3)
    x = initial.clone()
    x.fill(rng.standard_normal(x.shape).astype(np.float32))
    y = acq.clone()
    y.fill(rng.standard_normal(y.shape).astype(np.float32))

    Ax = am.forward(blur.direct(x))
    Aty = blur.adjoint(am.backward(y))

    lhs = float(np.vdot(Ax.as_array().ravel(), y.as_array().ravel()))
    rhs = float(np.vdot(x.as_array().ravel(), Aty.as_array().ravel()))

    assert np.isclose(lhs, rhs, rtol=5e-3, atol=1e-3)


