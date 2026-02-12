#!/usr/bin/env python3
"""Test to verify the weighted averaging in the ImageSummationOperator."""

import numpy as np
from sirf.STIR import ImageData
from cil.framework import BlockDataContainer
from recon_core.cil_extensions.operators import ImageCombineOperator
from recon_core.cil_extensions.framework import EnhancedBlockDataContainer
from recon_core.utils.sirf import get_array


def create_test_image(dims=(10, 10, 10), voxel_sizes=(2.0, 2.0, 2.0), offset=(0.0, 0.0, 0.0), value=0.0):
    """Create a test image with specified geometry and uniform value."""
    img = ImageData()
    img.initialise(dims, voxel_sizes, offset)
    img.fill(value)
    return img


def test_weighted_overlap():
    """Test weighted averaging in overlap region."""
    print("=" * 80)
    print("Testing weighted averaging in overlap region")
    print("=" * 80)

    # Create two images with different z-offsets to create an overlap
    # Image 1: z-offset = -50mm, length = 60mm (30 slices * 2mm/slice)
    # Image 2: z-offset = -10mm, length = 60mm
    # Overlap region: z = -10mm to 10mm (10 slices)

    dims = (30, 10, 10)  # z, y, x
    voxel_sizes = (2.0, 2.0, 2.0)

    img1 = create_test_image(dims, voxel_sizes, offset=(-50.0, 0.0, 0.0), value=10.0)
    img2 = create_test_image(dims, voxel_sizes, offset=(-10.0, 0.0, 0.0), value=20.0)

    # Create sensitivity images (uniform sensitivities for simplicity)
    sens1 = create_test_image(dims, voxel_sizes, offset=(-50.0, 0.0, 0.0), value=1.0)
    sens2 = create_test_image(dims, voxel_sizes, offset=(-10.0, 0.0, 0.0), value=1.0)

    # Test 1: Unweighted combination (simple sum)
    print("\n--- Test 1: Unweighted combination ---")
    images = EnhancedBlockDataContainer(img1, img2)
    combine_op_unweighted = ImageCombineOperator(images, weight_overlap=False)
    combined_unweighted = combine_op_unweighted.direct(images)

    arr_unweighted = get_array(combined_unweighted)
    print(f"Combined image shape: {arr_unweighted.shape}")
    print(f"Combined image min: {arr_unweighted.min():.2f}")
    print(f"Combined image max: {arr_unweighted.max():.2f}")
    print(f"Combined image mean: {arr_unweighted.mean():.2f}")

    # In overlap region, unweighted should give 30.0 (10 + 20)
    # The overlap is at z=0 based on the geometry
    mid_idx = arr_unweighted.shape[0] // 2
    overlap_val_unweighted = arr_unweighted[0, 5, 5]  # Changed from mid_idx to 0
    print(f"Value at z=0 (overlap, unweighted): {overlap_val_unweighted:.2f} (expected: 30.0)")

    # Debug: print values along z-axis to see the overlap pattern
    print(f"Values along z-axis (center pixel):")
    for i in [0, 10, 20, mid_idx, 30, 40, 49]:
        print(f"  z={i}: {arr_unweighted[i, 5, 5]:.2f}")

    # Test 1b: Adjoint consistency for unweighted
    print("\n--- Test 1b: Adjoint consistency check (unweighted) ---")
    rng_unweighted = np.random.default_rng(123)
    x1_unweighted = img1.copy()
    x1_unweighted.fill(rng_unweighted.standard_normal(dims))
    x2_unweighted = img2.copy()
    x2_unweighted.fill(rng_unweighted.standard_normal(dims))
    x_unweighted = EnhancedBlockDataContainer(x1_unweighted, x2_unweighted)

    y_unweighted = combined_unweighted.copy()
    y_unweighted.fill(rng_unweighted.standard_normal(get_array(combined_unweighted).shape))

    Ax_unweighted = combine_op_unweighted.direct(x_unweighted)
    Aty_unweighted = combine_op_unweighted.adjoint(y_unweighted)

    lhs_unweighted = np.vdot(get_array(Ax_unweighted).ravel(), get_array(y_unweighted).ravel())
    rhs_unweighted = sum(np.vdot(get_array(x_unweighted[i]).ravel(), get_array(Aty_unweighted[i]).ravel()) for i in range(2))

    rel_diff_unweighted = abs(lhs_unweighted - rhs_unweighted) / max(abs(lhs_unweighted), abs(rhs_unweighted), 1e-9)
    print(f"<Ax, y> = {lhs_unweighted:.6e}")
    print(f"<x, A^T y> = {rhs_unweighted:.6e}")
    print(f"Relative difference: {rel_diff_unweighted:.3e}")
    if rel_diff_unweighted < 1e-6:
        print("✓ Adjoint test PASSED")
    else:
        print("✗ Adjoint test FAILED")

    # Test 2: Weighted combination
    print("\n--- Test 2: Weighted combination with uniform sensitivities ---")
    # Need to resample sensitivities to the combined geometry first
    combine_op_weighted = ImageCombineOperator(images, weight_overlap=False)
    # Resample sensitivities using the resample operator
    sens_images_orig = EnhancedBlockDataContainer(sens1, sens2)
    sens_images_resampled = combine_op_weighted.resample_op.direct(sens_images_orig)

    # Debug: Check resampled sensitivities
    print(f"Resampled sensitivity shapes: {[get_array(s).shape for s in sens_images_resampled.containers]}")
    sens1_arr = get_array(sens_images_resampled.containers[0])
    sens2_arr = get_array(sens_images_resampled.containers[1])
    print(f"Sens1 values along z: z=0:{sens1_arr[0,5,5]:.2f}, z=10:{sens1_arr[10,5,5]:.2f}, z=20:{sens1_arr[20,5,5]:.2f}")
    print(f"Sens2 values along z: z=0:{sens2_arr[0,5,5]:.2f}, z=10:{sens2_arr[10,5,5]:.2f}, z=20:{sens2_arr[20,5,5]:.2f}")
    print(f"Sum of sens at z=0: {sens1_arr[0,5,5] + sens2_arr[0,5,5]:.2f}")

    # Now set sensitivities and enable weighting
    combine_op_weighted.set_sensitivities(sens_images_resampled)

    # Debug: Manually check what the resampled images look like
    resampled_images = combine_op_weighted.resample_op.direct(images)
    img1_resampled_arr = get_array(resampled_images.containers[0])
    img2_resampled_arr = get_array(resampled_images.containers[1])
    print(f"Resampled img1 at z=0: {img1_resampled_arr[0,5,5]:.2f}")
    print(f"Resampled img2 at z=0: {img2_resampled_arr[0,5,5]:.2f}")
    print(f"Manual weighted calc: (10*1 + 20*1)/(1+1) = {(10*1 + 20*1)/(1+1):.2f}")

    combined_weighted = combine_op_weighted.direct(images)

    arr_weighted = get_array(combined_weighted)
    print(f"Combined image shape: {arr_weighted.shape}")
    print(f"Combined image min: {arr_weighted.min():.2f}")
    print(f"Combined image max: {arr_weighted.max():.2f}")
    print(f"Combined image mean: {arr_weighted.mean():.2f}")

    # In overlap region with uniform sensitivities, weighted should give 15.0
    # Formula: (10*1 + 20*1) / (1+1) = 30/2 = 15.0
    overlap_val_weighted = arr_weighted[0, 5, 5]  # Changed from mid_idx to 0 (actual overlap location)
    print(f"Value at z=0 (overlap, weighted): {overlap_val_weighted:.2f} (expected: 15.0)")
    print(f"Value at z={mid_idx} (weighted): {arr_weighted[mid_idx, 5, 5]:.2f}")

    # Test 3: Weighted combination with non-uniform sensitivities
    print("\n--- Test 3: Weighted combination with non-uniform sensitivities ---")
    # Create sensitivities: bed 1 has 2x sensitivity, bed 2 has 1x
    sens1_nonuniform = create_test_image(dims, voxel_sizes, offset=(-50.0, 0.0, 0.0), value=2.0)
    sens2_nonuniform = create_test_image(dims, voxel_sizes, offset=(-10.0, 0.0, 0.0), value=1.0)

    sens_images_nonuniform_orig = EnhancedBlockDataContainer(sens1_nonuniform, sens2_nonuniform)
    combine_op_nonuniform = ImageCombineOperator(images, weight_overlap=False)
    sens_images_nonuniform_resampled = combine_op_nonuniform.resample_op.direct(sens_images_nonuniform_orig)
    combine_op_nonuniform.set_sensitivities(sens_images_nonuniform_resampled)
    combined_nonuniform = combine_op_nonuniform.direct(images)

    arr_nonuniform = get_array(combined_nonuniform)
    print(f"Combined image shape: {arr_nonuniform.shape}")
    print(f"Combined image min: {arr_nonuniform.min():.2f}")
    print(f"Combined image max: {arr_nonuniform.max():.2f}")
    print(f"Combined image mean: {arr_nonuniform.mean():.2f}")

    # In overlap region with non-uniform sensitivities:
    # Formula: (10*2 + 20*1) / (2+1) = 40/3 = 13.33
    overlap_val_nonuniform = arr_nonuniform[0, 5, 5]  # Changed from mid_idx to 0
    print(f"Value at z=0 (overlap, non-uniform): {overlap_val_nonuniform:.2f} (expected: 13.33)")

    # Test 4: Check adjoint consistency
    print("\n--- Test 4: Adjoint consistency check ---")

    # Create random test images
    rng = np.random.default_rng(42)
    x1 = img1.copy()
    x1.fill(rng.standard_normal(dims))
    x2 = img2.copy()
    x2.fill(rng.standard_normal(dims))
    x = EnhancedBlockDataContainer(x1, x2)

    y = combined_weighted.copy()
    y.fill(rng.standard_normal(get_array(combined_weighted).shape))

    # Compute <Ax, y> and <x, A^T y>
    Ax = combine_op_weighted.direct(x)
    Aty = combine_op_weighted.adjoint(y)

    lhs = np.vdot(get_array(Ax).ravel(), get_array(y).ravel())
    rhs = sum(np.vdot(get_array(x[i]).ravel(), get_array(Aty[i]).ravel()) for i in range(2))

    rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-9)
    print(f"<Ax, y> = {lhs:.6e}")
    print(f"<x, A^T y> = {rhs:.6e}")
    print(f"Relative difference: {rel_diff:.3e}")

    if rel_diff < 1e-6:
        print("✓ Adjoint test PASSED")
    else:
        print("✗ Adjoint test FAILED")

    # Assertions for automated regression testing
    assert abs(overlap_val_unweighted - 30.0) < 1e-3
    assert abs(overlap_val_weighted - 15.0) < 1e-3
    assert abs(overlap_val_nonuniform - (40.0 / 3.0)) < 1e-3
    assert rel_diff_unweighted < 1e-6
    assert rel_diff < 1e-6

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Unweighted overlap value: {overlap_val_unweighted:.2f} (expected: 30.0) - {'✓' if abs(overlap_val_unweighted - 30.0) < 0.1 else '✗'}")
    print(f"Weighted overlap value (uniform): {overlap_val_weighted:.2f} (expected: 15.0) - {'✓' if abs(overlap_val_weighted - 15.0) < 0.1 else '✗'}")
    print(f"Weighted overlap value (non-uniform): {overlap_val_nonuniform:.2f} (expected: 13.33) - {'✓' if abs(overlap_val_nonuniform - 13.33) < 0.1 else '✗'}")
    print(f"Adjoint consistency (unweighted): {rel_diff_unweighted:.3e} - {'✓ PASSED' if rel_diff_unweighted < 1e-6 else '✗ FAILED'}")
    print(f"Adjoint consistency (weighted): {rel_diff:.3e} - {'✓ PASSED' if rel_diff < 1e-6 else '✗ FAILED'}")


if __name__ == "__main__":
    test_weighted_overlap()
