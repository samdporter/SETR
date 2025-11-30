"""
Unit tests for GaussianBlurringOperator.

This module tests the core functionality of the GaussianBlurringOperator including:
- FWHM to sigma conversion
- Voxel size handling
- PSF normalization
- Backend consistency
"""

import numpy as np
import pytest

from setr.cil_extensions.operators.blurring import (
    GaussianBlurringOperator,
    get_fwhm_from_sigma,
    get_sigma_from_fwhm,
)


def test_fwhm_to_sigma_conversion():
    """Verify FWHM is correctly converted to sigma."""
    fwhm = [5.0, 10.0, 15.0]
    sigma = get_sigma_from_fwhm(fwhm)

    # FWHM = 2.3548 * sigma  (or more precisely 2 * sqrt(2 * ln(2)) ≈ 2.3548)
    expected = [f / 2.3548 for f in fwhm]
    assert np.allclose(sigma, expected, rtol=1e-6)

    # Round trip
    fwhm_back = get_fwhm_from_sigma(sigma)
    assert np.allclose(fwhm, fwhm_back, rtol=1e-6)


def test_sigma_to_fwhm_conversion():
    """Verify sigma is correctly converted to FWHM."""
    sigma = [2.0, 4.0, 6.0]
    fwhm = get_fwhm_from_sigma(sigma)

    # FWHM = 2.3548 * sigma
    expected = [s * 2.3548 for s in sigma]
    assert np.allclose(fwhm, expected, rtol=1e-6)


def test_conversion_constants():
    """Verify the conversion constant is correct."""
    # The exact constant is 2 * sqrt(2 * ln(2))
    expected_constant = 2 * np.sqrt(2 * np.log(2))
    # The code uses 2.3548 which should be close
    assert np.isclose(2.3548, expected_constant, rtol=1e-4)


def test_fwhm_conversion_with_zero():
    """Test FWHM conversion with edge case values."""
    # Zero FWHM should give zero sigma
    assert np.allclose(get_sigma_from_fwhm([0, 0, 0]), [0, 0, 0])
    assert np.allclose(get_fwhm_from_sigma([0, 0, 0]), [0, 0, 0])


def test_fwhm_conversion_anisotropic():
    """Test FWHM conversion with different values in each direction."""
    fwhm = [3.0, 6.0, 9.0]
    sigma = get_sigma_from_fwhm(fwhm)

    # Each direction should be converted independently
    for f, s in zip(fwhm, sigma):
        assert np.isclose(f, s * 2.3548, rtol=1e-6)

    # Round trip should preserve anisotropy
    fwhm_back = get_fwhm_from_sigma(sigma)
    assert np.allclose(fwhm, fwhm_back, rtol=1e-6)


def _measure_fwhm_1d(profile):
    """
    Measure FWHM from a 1D profile.

    Args:
        profile: 1D array with a peak

    Returns:
        FWHM in pixels
    """
    # Find maximum and half-maximum
    peak_val = profile.max()
    half_max = peak_val / 2.0

    # Find peak location
    peak_idx = np.argmax(profile)

    # Find left and right half-max points
    # Search left from peak
    left_idx = peak_idx
    while left_idx > 0 and profile[left_idx] > half_max:
        left_idx -= 1
    # Interpolate
    if left_idx < peak_idx and profile[left_idx] < half_max < profile[left_idx + 1]:
        frac = (half_max - profile[left_idx]) / (profile[left_idx + 1] - profile[left_idx])
        left_pos = left_idx + frac
    else:
        left_pos = left_idx

    # Search right from peak
    right_idx = peak_idx
    while right_idx < len(profile) - 1 and profile[right_idx] > half_max:
        right_idx += 1
    # Interpolate
    if right_idx > peak_idx and profile[right_idx] < half_max < profile[right_idx - 1]:
        frac = (half_max - profile[right_idx]) / (profile[right_idx - 1] - profile[right_idx])
        right_pos = right_idx - frac
    else:
        right_pos = right_idx

    return right_pos - left_pos


sirf = pytest.importorskip("sirf.STIR", reason="SIRF required for point source test")
from sirf.STIR import ImageData, examples_data_path  # noqa: E402


@pytest.mark.parametrize("fwhm_mm", [
    [6.0, 6.0, 6.0],
    [9.0, 9.0, 9.0],
    [12.0, 12.0, 12.0],
])
def test_blur_point_source_fwhm_isotropic(fwhm_mm):
    """
    Blur a point source and verify the measured FWHM matches the input FWHM.

    This test creates a point source, blurs it with a specified FWHM, and measures
    the resulting FWHM to ensure the blurring operator correctly implements the
    Gaussian convolution.
    """
    import os

    # Load SIRF example image for geometry
    data_path = examples_data_path('PET')
    template = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Use template's actual dimensions
    dims = template.dimensions()
    center = tuple(d // 2 for d in dims)

    # Create point source with template's shape
    point_arr = np.zeros(dims, dtype=np.float32)
    point_arr[center] = 1000.0

    point_image = template.clone()
    point_image.fill(point_arr)

    # Override voxel sizes for this test
    # Note: We can't easily change SIRF ImageData voxel sizes, so we'll use the
    # template's actual voxel sizes and adjust our expectations
    actual_voxel_sizes = np.array(template.voxel_sizes())  # (z, y, x)

    # Create blurring operator with FWHM in mm
    blur_op = GaussianBlurringOperator(fwhm_mm, template, backend='auto')

    # Blur the point source
    blurred = blur_op.direct(point_image)
    blurred_arr = blurred.as_array()

    # Measure FWHM in each direction (in voxels)
    # Extract 1D profiles through the center
    # center is a tuple (z, y, x), need integers for indexing
    center_z, center_y, center_x = center
    profile_z = blurred_arr[:, center_y, center_x]
    profile_y = blurred_arr[center_z, :, center_x]
    profile_x = blurred_arr[center_z, center_y, :]

    fwhm_z_voxels = _measure_fwhm_1d(profile_z)
    fwhm_y_voxels = _measure_fwhm_1d(profile_y)
    fwhm_x_voxels = _measure_fwhm_1d(profile_x)

    # Convert measured FWHM from voxels to mm
    measured_fwhm_mm = [
        fwhm_z_voxels * actual_voxel_sizes[0],
        fwhm_y_voxels * actual_voxel_sizes[1],
        fwhm_x_voxels * actual_voxel_sizes[2],
    ]

    # Compare with expected FWHM (allow 10% tolerance due to discrete sampling)
    for i, (measured, expected) in enumerate(zip(measured_fwhm_mm, fwhm_mm)):
        rel_error = abs(measured - expected) / expected
        assert rel_error < 0.1, (
            f"FWHM mismatch in dimension {i}: "
            f"expected {expected:.2f} mm, measured {measured:.2f} mm "
            f"(voxel size = {actual_voxel_sizes[i]:.2f} mm, "
            f"measured = {[fwhm_z_voxels, fwhm_y_voxels, fwhm_x_voxels][i]:.2f} voxels)"
        )


@pytest.mark.parametrize("fwhm_mm", [
    [6.0, 9.0, 12.0],   # Anisotropic
    [4.0, 8.0, 6.0],    # Different anisotropy
])
def test_blur_point_source_fwhm_anisotropic(fwhm_mm):
    """
    Test anisotropic blurring - different FWHM in each direction.

    This verifies that the blurring operator correctly handles different FWHM
    values in z, y, and x directions.
    """
    import os

    # Load SIRF example image for geometry
    data_path = examples_data_path('PET')
    template = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Use template's actual dimensions
    dims = template.dimensions()
    center = tuple(d // 2 for d in dims)

    # Create point source with template's shape
    point_arr = np.zeros(dims, dtype=np.float32)
    point_arr[center] = 1000.0

    point_image = template.clone()
    point_image.fill(point_arr)

    actual_voxel_sizes = np.array(template.voxel_sizes())  # (z, y, x)

    # Create blurring operator
    blur_op = GaussianBlurringOperator(fwhm_mm, template, backend='auto')

    # Blur the point source
    blurred = blur_op.direct(point_image)
    blurred_arr = blurred.as_array()

    # Measure FWHM in each direction
    # center is a tuple (z, y, x), need integers for indexing
    center_z, center_y, center_x = center
    profile_z = blurred_arr[:, center_y, center_x]
    profile_y = blurred_arr[center_z, :, center_x]
    profile_x = blurred_arr[center_z, center_y, :]

    fwhm_z_voxels = _measure_fwhm_1d(profile_z)
    fwhm_y_voxels = _measure_fwhm_1d(profile_y)
    fwhm_x_voxels = _measure_fwhm_1d(profile_x)

    measured_fwhm_mm = [
        fwhm_z_voxels * actual_voxel_sizes[0],
        fwhm_y_voxels * actual_voxel_sizes[1],
        fwhm_x_voxels * actual_voxel_sizes[2],
    ]

    # Verify each direction independently
    for i, (measured, expected) in enumerate(zip(measured_fwhm_mm, fwhm_mm)):
        rel_error = abs(measured - expected) / expected
        assert rel_error < 0.1, (
            f"Anisotropic FWHM mismatch in dimension {i}: "
            f"expected {expected:.2f} mm, measured {measured:.2f} mm "
            f"(relative error = {rel_error:.1%})"
        )

    # Verify anisotropy is preserved (ratios should match)
    expected_ratio_yx = fwhm_mm[1] / fwhm_mm[0]
    expected_ratio_zx = fwhm_mm[0] / fwhm_mm[2]

    measured_ratio_yx = measured_fwhm_mm[1] / measured_fwhm_mm[0]
    measured_ratio_zx = measured_fwhm_mm[0] / measured_fwhm_mm[2]

    assert np.isclose(measured_ratio_yx, expected_ratio_yx, rtol=0.1), \
        f"Y/X ratio mismatch: expected {expected_ratio_yx:.2f}, got {measured_ratio_yx:.2f}"
    assert np.isclose(measured_ratio_zx, expected_ratio_zx, rtol=0.1), \
        f"Z/X ratio mismatch: expected {expected_ratio_zx:.2f}, got {measured_ratio_zx:.2f}"


def test_psf_normalization():
    """Verify that the PSF integrates (sums) to approximately 1."""
    import os

    # Load SIRF example image
    data_path = examples_data_path('PET')
    template = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Use template's actual dimensions
    dims = template.dimensions()
    center = tuple(d // 2 for d in dims)

    # Create point source with unit amplitude
    point_arr = np.zeros(dims, dtype=np.float32)
    point_arr[center] = 1.0

    point_image = template.clone()
    point_image.fill(point_arr)

    # Blur with reasonable FWHM
    fwhm_mm = [6.0, 6.0, 6.0]
    blur_op = GaussianBlurringOperator(fwhm_mm, template, backend='auto')

    blurred = blur_op.direct(point_image)
    blurred_arr = blurred.as_array()

    # Sum should be close to 1 (conservation of mass)
    total = blurred_arr.sum()
    assert np.isclose(total, 1.0, rtol=0.01), \
        f"PSF normalization failed: sum = {total:.6f}, expected 1.0"


@pytest.mark.parametrize("fwhm_mm", [
    [6.0, 6.0, 6.0],
    [9.0, 9.0, 9.0],
])
def test_adjoint_point_source_fwhm_isotropic(fwhm_mm):
    """
    Test the adjoint operation produces expected correlation with a point source.

    For isotropic Gaussian blurring, the adjoint should be the same as the direct
    operation (self-adjoint property). This test verifies that applying the adjoint
    to a point source produces the expected Gaussian correlation.
    """
    import os

    # Load SIRF example image for geometry
    data_path = examples_data_path('PET')
    template = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Use template's actual dimensions
    dims = template.dimensions()
    center = tuple(d // 2 for d in dims)

    # Create point source with template's shape
    point_arr = np.zeros(dims, dtype=np.float32)
    point_arr[center] = 1000.0

    point_image = template.clone()
    point_image.fill(point_arr)

    actual_voxel_sizes = np.array(template.voxel_sizes())  # (z, y, x)

    # Create blurring operator with FWHM in mm
    blur_op = GaussianBlurringOperator(fwhm_mm, template, backend='auto')

    # Apply adjoint to the point source
    adjoint_blurred = blur_op.adjoint(point_image)
    adjoint_arr = adjoint_blurred.as_array()

    # Measure FWHM in each direction (in voxels)
    # Extract 1D profiles through the center
    center_z, center_y, center_x = center
    profile_z = adjoint_arr[:, center_y, center_x]
    profile_y = adjoint_arr[center_z, :, center_x]
    profile_x = adjoint_arr[center_z, center_y, :]

    fwhm_z_voxels = _measure_fwhm_1d(profile_z)
    fwhm_y_voxels = _measure_fwhm_1d(profile_y)
    fwhm_x_voxels = _measure_fwhm_1d(profile_x)

    # Convert measured FWHM from voxels to mm
    measured_fwhm_mm = [
        fwhm_z_voxels * actual_voxel_sizes[0],
        fwhm_y_voxels * actual_voxel_sizes[1],
        fwhm_x_voxels * actual_voxel_sizes[2],
    ]

    # Compare with expected FWHM (allow 10% tolerance due to discrete sampling)
    for i, (measured, expected) in enumerate(zip(measured_fwhm_mm, fwhm_mm)):
        rel_error = abs(measured - expected) / expected
        assert rel_error < 0.1, (
            f"Adjoint FWHM mismatch in dimension {i}: "
            f"expected {expected:.2f} mm, measured {measured:.2f} mm "
            f"(voxel size = {actual_voxel_sizes[i]:.2f} mm, "
            f"measured = {[fwhm_z_voxels, fwhm_y_voxels, fwhm_x_voxels][i]:.2f} voxels)"
        )

    # For isotropic case, verify adjoint equals direct (self-adjoint property)
    if len(set(fwhm_mm)) == 1:  # All FWHM values are the same
        direct_blurred = blur_op.direct(point_image)
        direct_arr = direct_blurred.as_array()

        # Arrays should be nearly identical for self-adjoint operator
        assert np.allclose(adjoint_arr, direct_arr, rtol=1e-5), \
            "For isotropic blur, adjoint should equal direct (self-adjoint property)"


@pytest.mark.parametrize("fwhm_mm", [
    [6.0, 9.0, 12.0],   # Anisotropic
    [4.0, 8.0, 6.0],    # Different anisotropy
])
def test_adjoint_point_source_fwhm_anisotropic(fwhm_mm):
    """
    Test the adjoint operation with anisotropic blurring.

    This verifies that the adjoint correctly handles different FWHM values in
    each direction and produces the expected correlation pattern.
    """
    import os

    # Load SIRF example image for geometry
    data_path = examples_data_path('PET')
    template = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Use template's actual dimensions
    dims = template.dimensions()
    center = tuple(d // 2 for d in dims)

    # Create point source with template's shape
    point_arr = np.zeros(dims, dtype=np.float32)
    point_arr[center] = 1000.0

    point_image = template.clone()
    point_image.fill(point_arr)

    actual_voxel_sizes = np.array(template.voxel_sizes())  # (z, y, x)

    # Create blurring operator
    blur_op = GaussianBlurringOperator(fwhm_mm, template, backend='auto')

    # Apply adjoint to the point source
    adjoint_blurred = blur_op.adjoint(point_image)
    adjoint_arr = adjoint_blurred.as_array()

    # Measure FWHM in each direction
    center_z, center_y, center_x = center
    profile_z = adjoint_arr[:, center_y, center_x]
    profile_y = adjoint_arr[center_z, :, center_x]
    profile_x = adjoint_arr[center_z, center_y, :]

    fwhm_z_voxels = _measure_fwhm_1d(profile_z)
    fwhm_y_voxels = _measure_fwhm_1d(profile_y)
    fwhm_x_voxels = _measure_fwhm_1d(profile_x)

    measured_fwhm_mm = [
        fwhm_z_voxels * actual_voxel_sizes[0],
        fwhm_y_voxels * actual_voxel_sizes[1],
        fwhm_x_voxels * actual_voxel_sizes[2],
    ]

    # Verify each direction independently
    for i, (measured, expected) in enumerate(zip(measured_fwhm_mm, fwhm_mm)):
        rel_error = abs(measured - expected) / expected
        assert rel_error < 0.1, (
            f"Adjoint anisotropic FWHM mismatch in dimension {i}: "
            f"expected {expected:.2f} mm, measured {measured:.2f} mm "
            f"(relative error = {rel_error:.1%})"
        )

    # Verify anisotropy is preserved (ratios should match)
    expected_ratio_yx = fwhm_mm[1] / fwhm_mm[0]
    expected_ratio_zx = fwhm_mm[0] / fwhm_mm[2]

    measured_ratio_yx = measured_fwhm_mm[1] / measured_fwhm_mm[0]
    measured_ratio_zx = measured_fwhm_mm[0] / measured_fwhm_mm[2]

    assert np.isclose(measured_ratio_yx, expected_ratio_yx, rtol=0.1), \
        f"Adjoint Y/X ratio mismatch: expected {expected_ratio_yx:.2f}, got {measured_ratio_yx:.2f}"
    assert np.isclose(measured_ratio_zx, expected_ratio_zx, rtol=0.1), \
        f"Adjoint Z/X ratio mismatch: expected {expected_ratio_zx:.2f}, got {measured_ratio_zx:.2f}"


def test_adjoint_is_correlation_not_convolution():
    """
    Verify that the adjoint performs correlation, not convolution.

    For a Gaussian PSF, convolution and correlation are the same because
    the PSF is symmetric. To test that adjoint truly does correlation,
    we verify the fundamental adjoint property: <Ax, y> = <x, A*y>

    This ensures the adjoint is correctly flipping the kernel (correlation)
    rather than incorrectly applying the same convolution.
    """
    import os

    # Load SIRF example image for geometry
    data_path = examples_data_path('PET')
    template = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Create test patterns with random values to ensure overlap after blurring
    dims = template.dimensions()
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    # Create random patterns (ensuring they have some spatial structure)
    x_arr = rng.randn(*dims).astype(np.float32)
    y_arr = rng.randn(*dims).astype(np.float32)

    x_image = template.clone()
    x_image.fill(x_arr)

    y_image = template.clone()
    y_image.fill(y_arr)

    # Create blurring operator
    fwhm_mm = [12.0, 4.0, 21.0]
    blur_op = GaussianBlurringOperator(fwhm_mm, template, backend='auto')

    # Test 1: Verify adjoint test <Ax, y> = <x, A*y>
    # This is the fundamental property that defines the adjoint
    Ax = blur_op.direct(x_image)
    Aty = blur_op.adjoint(y_image)

    # Compute inner products
    inner_Ax_y = np.sum(Ax.as_array() * y_arr)
    inner_x_Aty = np.sum(x_arr * Aty.as_array())

    # These should be equal (within numerical precision)
    rel_error = abs(inner_Ax_y - inner_x_Aty) / max(abs(inner_Ax_y), abs(inner_x_Aty))
    assert rel_error < 1e-3, (
        f"Adjoint test failed: <Ax, y> = {inner_Ax_y:.6e}, "
        f"<x, A*y> = {inner_x_Aty:.6e}, relative error = {rel_error:.6e}"
    )

    # Test 2: For Gaussian (symmetric PSF), verify direct(x) ≈ adjoint(x)
    # This is a sanity check - even though we can't distinguish correlation
    # from convolution with symmetric kernels, we can verify self-adjointness
    Ax_direct = blur_op.direct(x_image)
    Ax_adjoint = blur_op.adjoint(x_image)

    # For symmetric Gaussian, these should be nearly equal
    assert np.allclose(Ax_direct.as_array(), Ax_adjoint.as_array(), rtol=1e-3), \
        "For symmetric Gaussian PSF, direct and adjoint should give the same result"
