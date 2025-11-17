#!/usr/bin/env python3
"""
Standalone script to test SIRF PSF behavior in acquisition models.

NOTE: This is NOT part of the test suite - it tests SIRF/Parallelproj behavior,
not this library's code. It's kept for reference and documentation purposes.

Usage:
    python scripts/test_sirf_psf_behavior.py
"""

import os
import sys
import numpy as np

try:
    from sirf.STIR import (
        AcquisitionModelUsingParallelproj,
        AcquisitionModelUsingRayTracingMatrix,
        SeparableGaussianImageFilter,
        AcquisitionData,
        ImageData,
        examples_data_path,
    )
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False
    print("ERROR: SIRF not available. Please install SIRF to run this script.")
    sys.exit(1)


def get_test_data():
    """Load PET acquisition data and images from SIRF examples."""
    data_path = examples_data_path('PET')
    acquisition_data = AcquisitionData(os.path.join(data_path, "template_sinogram.hs"))
    initial_image = ImageData(os.path.join(data_path, "test_image_PM_QP_6.hv"))

    # Create point source
    point_image = initial_image.clone()
    point_arr = np.zeros(point_image.shape)
    center = tuple(s // 2 for s in point_arr.shape)
    point_arr[center] = 1000.0
    point_image.fill(point_arr)

    return acquisition_data, initial_image, point_image


def test_psf_forward_projection(acquisition_data, initial_image, point_source, fwhm):
    """Test that PSF affects forward projection for ParallelProj."""
    print(f"\n{'='*60}")
    print(f"Test: PSF forward projection with FWHM={fwhm}")
    print(f"{'='*60}")

    # Create acquisition model WITH PSF
    am_with_psf = AcquisitionModelUsingParallelproj()
    psf = SeparableGaussianImageFilter()
    psf.set_fwhms(fwhm)
    am_with_psf.set_image_data_processor(psf)
    am_with_psf.set_up(acquisition_data, initial_image)

    # Create acquisition model WITHOUT PSF
    am_without_psf = AcquisitionModelUsingParallelproj()
    am_without_psf.set_up(acquisition_data, initial_image)

    # Forward project a point source
    proj_with_psf = am_with_psf.forward(point_source)
    proj_without_psf = am_without_psf.forward(point_source)

    # Calculate difference
    diff = proj_with_psf - proj_without_psf
    diff_norm = diff.norm()
    relative_diff = diff_norm / proj_without_psf.norm()

    print(f"  Absolute difference: {diff_norm:.6e}")
    print(f"  Relative difference: {relative_diff*100:.4f}%")

    passed = diff_norm > 1e-6 and relative_diff > 0.001
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def test_psf_backward_projection(acquisition_data, initial_image):
    """Test that PSF affects backward projection."""
    print(f"\n{'='*60}")
    print("Test: PSF backward projection")
    print(f"{'='*60}")

    # Create acquisition models
    am_with_psf = AcquisitionModelUsingParallelproj()
    psf = SeparableGaussianImageFilter()
    psf.set_fwhms([21, 21, 21])
    am_with_psf.set_image_data_processor(psf)
    am_with_psf.set_up(acquisition_data, initial_image)

    am_without_psf = AcquisitionModelUsingParallelproj()
    am_without_psf.set_up(acquisition_data, initial_image)

    # Backward project uniform data
    ones_proj = acquisition_data.get_uniform_copy(1.0)
    back_with_psf = am_with_psf.backward(ones_proj)
    back_without_psf = am_without_psf.backward(ones_proj)

    # Calculate difference
    diff = (back_with_psf - back_without_psf).norm()
    relative_diff = diff / back_without_psf.norm()

    print(f"  Absolute difference: {diff:.6e}")
    print(f"  Relative difference: {relative_diff*100:.6f}%")

    passed = diff > 1e-6 and relative_diff > 0.001
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL (PSF not applied in backward)'}")

    if not passed:
        print("\n  NOTE: PSF does not appear to be applied in backward projection")
        print("  This may be expected behavior in SIRF/Parallelproj")

    return passed


def test_fwhm_scaling(acquisition_data, initial_image, point_source):
    """Test that larger FWHM causes more blurring."""
    print(f"\n{'='*60}")
    print("Test: Larger FWHM has stronger effect")
    print(f"{'='*60}")

    # Small FWHM
    am_small = AcquisitionModelUsingParallelproj()
    psf_small = SeparableGaussianImageFilter()
    psf_small.set_fwhms([4, 4, 4])
    am_small.set_image_data_processor(psf_small)
    am_small.set_up(acquisition_data, initial_image)

    # Large FWHM
    am_large = AcquisitionModelUsingParallelproj()
    psf_large = SeparableGaussianImageFilter()
    psf_large.set_fwhms([21, 21, 21])
    am_large.set_image_data_processor(psf_large)
    am_large.set_up(acquisition_data, initial_image)

    # No PSF
    am_none = AcquisitionModelUsingParallelproj()
    am_none.set_up(acquisition_data, initial_image)

    # Forward project
    proj_small = am_small.forward(point_source)
    proj_large = am_large.forward(point_source)
    proj_none = am_none.forward(point_source)

    # Calculate differences from unblurred
    diff_small = (proj_small - proj_none).norm()
    diff_large = (proj_large - proj_none).norm()
    ratio = diff_large / diff_small if diff_small > 0 else 0

    print(f"  FWHM=4 difference:  {diff_small:.6f}")
    print(f"  FWHM=21 difference: {diff_large:.6f}")
    print(f"  Ratio (large/small): {ratio:.3f}")

    passed = diff_large > diff_small and ratio > 1.3
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def test_raytracing_psf(acquisition_data, initial_image, point_source):
    """Test that PSF works with RayTracing acquisition model."""
    print(f"\n{'='*60}")
    print("Test: PSF with RayTracing projector")
    print(f"{'='*60}")

    try:
        # Create acquisition model WITH PSF
        am_with_psf = AcquisitionModelUsingRayTracingMatrix()
        am_with_psf.set_num_tangential_LORs(10)
        psf = SeparableGaussianImageFilter()
        psf.set_fwhms([21, 21, 21])
        am_with_psf.set_image_data_processor(psf)
        am_with_psf.set_up(acquisition_data, initial_image)

        # Create acquisition model WITHOUT PSF
        am_without_psf = AcquisitionModelUsingRayTracingMatrix()
        am_without_psf.set_num_tangential_LORs(10)
        am_without_psf.set_up(acquisition_data, initial_image)

        # Forward project
        proj_with_psf = am_with_psf.forward(point_source)
        proj_without_psf = am_without_psf.forward(point_source)

        # Calculate difference
        diff_norm = (proj_with_psf - proj_without_psf).norm()

        print(f"  Absolute difference: {diff_norm:.6e}")

        passed = diff_norm > 1e-6
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

        return passed
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def main():
    """Run all PSF tests."""
    if not SIRF_AVAILABLE:
        return 1

    print("="*60)
    print("SIRF PSF Behavior Tests")
    print("="*60)
    print("\nLoading test data from SIRF examples...")

    try:
        acquisition_data, initial_image, point_source = get_test_data()
    except Exception as e:
        print(f"ERROR: Failed to load test data: {e}")
        return 1

    print("✓ Test data loaded successfully")

    # Run tests
    results = {}
    results['Forward FWHM=9'] = test_psf_forward_projection(
        acquisition_data, initial_image, point_source, [9, 9, 9]
    )
    results['Forward FWHM=21'] = test_psf_forward_projection(
        acquisition_data, initial_image, point_source, [21, 21, 21]
    )
    results['Backward'] = test_psf_backward_projection(
        acquisition_data, initial_image
    )
    results['FWHM Scaling'] = test_fwhm_scaling(
        acquisition_data, initial_image, point_source
    )
    results['RayTracing'] = test_raytracing_psf(
        acquisition_data, initial_image, point_source
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  {test_name:20s}: {status}")

    passed_count = sum(results.values())
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
