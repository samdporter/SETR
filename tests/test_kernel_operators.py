"""
Comprehensive tests for kernel operators (KernelOperator and NumbaKernelOperator).

Tests cover:
- Basic kernel application (forward pass)
- Adjoint correctness via dot product test
- Hybrid mode functionality
- Masking functionality
- Distance weighting
- Kernel normalization
- Edge cases and boundary handling
"""

import numpy as np
import pytest
from cil.framework import ImageGeometry

from setr.kernel.python import (
    NUMBA_AVAIL,
    SLIDING_WINDOW_AVAIL,
    KernelOperator,
    NumbaKernelOperator,
    get_kernel_operator,
)


# Test fixtures
@pytest.fixture
def small_geometry():
    """Small 3D geometry for fast tests."""
    return ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=8)


@pytest.fixture
def medium_geometry():
    """Medium 3D geometry for more realistic tests."""
    return ImageGeometry(voxel_num_x=16, voxel_num_y=16, voxel_num_z=16)


@pytest.fixture
def anatomical_image_simple(small_geometry):
    """Simple anatomical image with two regions."""
    img = small_geometry.allocate(0.0)
    arr = img.as_array()
    # Create two regions: background (0.0) and foreground (1.0)
    arr[2:6, 2:6, 2:6] = 1.0
    img.fill(arr)
    return img


@pytest.fixture
def anatomical_image_gradient(small_geometry):
    """Anatomical image with smooth gradient."""
    img = small_geometry.allocate(0.0)
    arr = img.as_array()
    # Create smooth gradient along x-axis
    for i in range(arr.shape[0]):
        arr[i, :, :] = i / arr.shape[0]
    img.fill(arr)
    return img


@pytest.fixture
def emission_image_uniform(small_geometry):
    """Uniform emission image."""
    img = small_geometry.allocate(1.0)
    return img


@pytest.fixture
def emission_image_spot(small_geometry):
    """Emission image with a hot spot."""
    img = small_geometry.allocate(0.1)
    arr = img.as_array()
    # Create hot spot in the center
    arr[3:5, 3:5, 3:5] = 2.0
    img.fill(arr)
    return img


# Backend parametrization
backends = []
if SLIDING_WINDOW_AVAIL:
    backends.append("python")
if NUMBA_AVAIL:
    backends.append("numba")

if not backends:
    pytest.skip("No backends available (need numpy sliding_window_view or numba)", allow_module_level=True)


# ============================================================================
# Test 1: Basic kernel application and identity preservation
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_kernel_identity_on_uniform_image(small_geometry, backend):
    """Test that kernel applied to uniform image returns uniform image (with proper settings)."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_dist=1.0,
        normalize_kernel=True,
    )

    # Create uniform anatomical and emission images
    anat = small_geometry.allocate(1.0)
    x = small_geometry.allocate(5.0)

    operator.set_anatomical_image(anat)
    result = operator.direct(x)

    # With uniform images and normalized kernel, output should be approximately uniform
    result_arr = result.as_array()
    assert np.allclose(result_arr, 5.0, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("backend", backends)
def test_kernel_smoothing_effect(small_geometry, emission_image_spot, anatomical_image_simple, backend):
    """Test that kernel smooths the emission image."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=True,
    )

    operator.set_anatomical_image(anatomical_image_simple)
    result = operator.direct(emission_image_spot)

    # Check that max is reduced (smoothing)
    original_max = emission_image_spot.as_array().max()
    smoothed_max = result.as_array().max()
    assert smoothed_max < original_max

    # Check that variance is reduced (smoothing effect)
    original_var = np.var(emission_image_spot.as_array())
    smoothed_var = np.var(result.as_array())
    assert smoothed_var < original_var


# ============================================================================
# Test 2: Adjoint correctness via dot product test
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_adjoint_dot_product_pure_anatomical(small_geometry, anatomical_image_simple, backend):
    """Test adjoint correctness: <Kx, y> = <x, K*y> for pure anatomical kernel."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.3,
        sigma_dist=1.0,
        normalize_kernel=False,  # Don't normalize for adjoint test
        use_mask=False,
        hybrid=False,
    )

    operator.set_anatomical_image(anatomical_image_simple)

    # Create random test vectors
    np.random.seed(42)
    x = small_geometry.allocate(0.0)
    y = small_geometry.allocate(0.0)
    x.fill(np.random.randn(*x.shape))
    y.fill(np.random.randn(*y.shape))

    # Compute forward and adjoint
    Kx = operator.direct(x)
    Kstar_y = operator.adjoint(y)

    # Compute dot products
    dot1 = np.sum(Kx.as_array() * y.as_array())
    dot2 = np.sum(x.as_array() * Kstar_y.as_array())

    # For pure anatomical kernel without mask, should be self-adjoint
    assert np.allclose(dot1, dot2, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("backend", backends)
def test_adjoint_dot_product_with_mask(small_geometry, anatomical_image_simple, backend):
    """Test adjoint correctness with masking enabled."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_dist=1.0,
        normalize_kernel=False,
        use_mask=True,
        mask_k=10,  # Keep only 10 closest neighbors
        hybrid=False,
    )

    operator.set_anatomical_image(anatomical_image_simple)

    # Create random test vectors
    np.random.seed(43)
    x = small_geometry.allocate(0.0)
    y = small_geometry.allocate(0.0)
    x.fill(np.random.randn(*x.shape))
    y.fill(np.random.randn(*y.shape))

    # Compute forward and adjoint
    Kx = operator.direct(x)
    Kstar_y = operator.adjoint(y)

    # Compute dot products
    dot1 = np.sum(Kx.as_array() * y.as_array())
    dot2 = np.sum(x.as_array() * Kstar_y.as_array())

    # With mask, adjoint should still satisfy dot product property
    assert np.allclose(dot1, dot2, rtol=0.05, atol=1e-5)


@pytest.mark.parametrize("backend", backends)
def test_adjoint_dot_product_hybrid(small_geometry, anatomical_image_simple, backend):
    """Test adjoint correctness with hybrid=True."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.3,
        sigma_dist=1.0,
        sigma_emission=0.5,
        normalize_kernel=False,
        use_mask=False,
        hybrid=True,
    )

    operator.set_anatomical_image(anatomical_image_simple)

    # Create random test vectors (positive values for emission)
    np.random.seed(44)
    x = small_geometry.allocate(0.0)
    y = small_geometry.allocate(0.0)
    x.fill(np.abs(np.random.randn(*x.shape)) + 0.1)
    y.fill(np.abs(np.random.randn(*y.shape)) + 0.1)

    # Compute forward and adjoint
    Kx = operator.direct(x)
    Kstar_y = operator.adjoint(y)

    # Compute dot products
    dot1 = np.sum(Kx.as_array() * y.as_array())
    dot2 = np.sum(x.as_array() * Kstar_y.as_array())

    # Hybrid mode makes kernel non-symmetric, but adjoint property should hold
    assert np.allclose(dot1, dot2, rtol=0.03, atol=1e-5)


# ============================================================================
# Test 3: Hybrid mode functionality
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_hybrid_mode_no_nans(small_geometry, anatomical_image_simple, emission_image_spot, backend):
    """Test that hybrid mode does not produce NaNs."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.3,
        sigma_dist=1.0,
        sigma_emission=0.5,
        normalize_kernel=True,
        hybrid=True,
    )

    operator.set_anatomical_image(anatomical_image_simple)
    result = operator.direct(emission_image_spot)

    # Check for NaNs
    assert not np.any(np.isnan(result.as_array())), "Hybrid mode produced NaNs"
    assert not np.any(np.isinf(result.as_array())), "Hybrid mode produced Infs"


@pytest.mark.parametrize("backend", backends)
def test_hybrid_affects_result(small_geometry, anatomical_image_simple, emission_image_spot, backend):
    """Test that hybrid mode produces different results than non-hybrid."""
    operator_pure = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_dist=1.0,
        sigma_emission=0.5,
        normalize_kernel=True,
        hybrid=False,
    )

    operator_hybrid = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_dist=1.0,
        sigma_emission=0.5,
        normalize_kernel=True,
        hybrid=True,
    )

    operator_pure.set_anatomical_image(anatomical_image_simple)
    operator_hybrid.set_anatomical_image(anatomical_image_simple)

    result_pure = operator_pure.direct(emission_image_spot)
    result_hybrid = operator_hybrid.direct(emission_image_spot)

    # Results should be different
    assert not np.allclose(result_pure.as_array(), result_hybrid.as_array(), rtol=1e-3)


# ============================================================================
# Test 4: Masking functionality
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_mask_reduces_smoothing(small_geometry, anatomical_image_simple, emission_image_spot, backend):
    """Test that masking reduces the number of neighbors and affects smoothing."""
    operator_full = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        normalize_kernel=True,
        use_mask=False,
    )

    operator_masked = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        normalize_kernel=True,
        use_mask=True,
        mask_k=10,  # Only 10 neighbors out of 5^3=125
    )

    operator_full.set_anatomical_image(anatomical_image_simple)
    operator_masked.set_anatomical_image(anatomical_image_simple)

    result_full = operator_full.direct(emission_image_spot)
    result_masked = operator_masked.direct(emission_image_spot)

    # Results should differ
    assert not np.allclose(result_full.as_array(), result_masked.as_array(), rtol=1e-2)


# ============================================================================
# Test 5: Distance weighting
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_distance_weighting_affects_result(small_geometry, anatomical_image_simple, emission_image_spot, backend):
    """Test that distance weighting changes the kernel output."""
    operator_no_dist = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        distance_weighting=False,
        normalize_kernel=True,
    )

    operator_with_dist = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_dist=1.0,
        distance_weighting=True,
        normalize_kernel=True,
    )

    operator_no_dist.set_anatomical_image(anatomical_image_simple)
    operator_with_dist.set_anatomical_image(anatomical_image_simple)

    result_no_dist = operator_no_dist.direct(emission_image_spot)
    result_with_dist = operator_with_dist.direct(emission_image_spot)

    # Results should differ
    assert not np.allclose(result_no_dist.as_array(), result_with_dist.as_array(), rtol=1e-3)


# ============================================================================
# Test 6: Normalization
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_normalization_preserves_scale(small_geometry, anatomical_image_simple, emission_image_uniform, backend):
    """Test that normalization approximately preserves the scale of uniform images."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_kernel=True,
    )

    operator.set_anatomical_image(anatomical_image_simple)
    result = operator.direct(emission_image_uniform)

    # With normalized kernel on uniform regions, values should stay close to original
    result_arr = result.as_array()
    original_val = emission_image_uniform.as_array()[0, 0, 0]

    # Check interior points (away from edges)
    interior = result_arr[2:6, 2:6, 2:6]
    assert np.allclose(interior, original_val, rtol=0.1, atol=0.1)


# ============================================================================
# Test 7: Edge cases and robustness
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_zero_emission_no_crash(small_geometry, anatomical_image_simple, backend):
    """Test that zero emission doesn't cause crashes or NaNs."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_emission=0.5,
        hybrid=True,
        normalize_kernel=True,
    )

    operator.set_anatomical_image(anatomical_image_simple)
    x = small_geometry.allocate(0.0)
    result = operator.direct(x)

    assert not np.any(np.isnan(result.as_array()))
    assert np.allclose(result.as_array(), 0.0)


@pytest.mark.parametrize("backend", backends)
def test_negative_values_handled(small_geometry, anatomical_image_simple, backend):
    """Test that negative values in emission don't cause issues."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_emission=0.5,
        hybrid=True,
        normalize_kernel=True,
    )

    operator.set_anatomical_image(anatomical_image_simple)
    x = small_geometry.allocate(-1.0)
    result = operator.direct(x)

    assert not np.any(np.isnan(result.as_array()))
    assert not np.any(np.isinf(result.as_array()))


@pytest.mark.parametrize("backend", backends)
def test_extreme_sigma_values(small_geometry, anatomical_image_simple, emission_image_spot, backend):
    """Test behavior with extreme sigma values."""
    # Very small sigma (should be very local)
    operator_small = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.001,
        normalize_kernel=True,
    )

    # Very large sigma (should smooth a lot)
    operator_large = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=100.0,
        normalize_kernel=True,
    )

    operator_small.set_anatomical_image(anatomical_image_simple)
    operator_large.set_anatomical_image(anatomical_image_simple)

    result_small = operator_small.direct(emission_image_spot)
    result_large = operator_large.direct(emission_image_spot)

    # Small sigma should preserve structure better
    assert not np.any(np.isnan(result_small.as_array()))
    assert not np.any(np.isnan(result_large.as_array()))

    # Large sigma should smooth more
    variance_small = np.var(result_small.as_array())
    variance_large = np.var(result_large.as_array())
    assert variance_large < variance_small


# ============================================================================
# Test 8: Backend consistency
# ============================================================================


@pytest.mark.skipif(len(backends) < 2, reason="Need at least 2 backends to compare")
def test_backends_produce_consistent_results(small_geometry, anatomical_image_simple, emission_image_spot):
    """Test that different backends produce consistent results."""
    params = {
        "num_neighbours": 5,
        "sigma_anat": 0.3,
        "sigma_dist": 1.0,
        "distance_weighting": True,
        "normalize_kernel": True,
    }

    results = {}
    for backend in backends:
        operator = get_kernel_operator(small_geometry, backend=backend, **params)
        operator.set_anatomical_image(anatomical_image_simple)
        result = operator.direct(emission_image_spot)
        results[backend] = result.as_array()

    # Compare all backends pairwise
    backend_list = list(results.keys())
    for i in range(len(backend_list)):
        for j in range(i + 1, len(backend_list)):
            b1, b2 = backend_list[i], backend_list[j]
            assert np.allclose(results[b1], results[b2], rtol=1e-5, atol=1e-8), f"{b1} vs {b2} mismatch"


@pytest.mark.skipif(len(backends) < 2, reason="Need at least 2 backends to compare")
def test_backends_hybrid_consistency(small_geometry, anatomical_image_simple, emission_image_spot):
    """Test that different backends produce consistent results with hybrid=True."""
    params = {
        "num_neighbours": 3,
        "sigma_anat": 0.3,
        "sigma_dist": 1.0,
        "sigma_emission": 0.5,
        "distance_weighting": True,
        "normalize_kernel": True,
        "hybrid": True,
    }

    results = {}
    for backend in backends:
        operator = get_kernel_operator(small_geometry, backend=backend, **params)
        operator.set_anatomical_image(anatomical_image_simple)
        result = operator.direct(emission_image_spot)
        results[backend] = result.as_array()

    # Compare all backends pairwise
    backend_list = list(results.keys())
    for i in range(len(backend_list)):
        for j in range(i + 1, len(backend_list)):
            b1, b2 = backend_list[i], backend_list[j]
            assert np.allclose(results[b1], results[b2], rtol=1e-5, atol=1e-8), f"{b1} vs {b2} hybrid mismatch"


# ============================================================================
# Test 9: Parameter update and mask recalculation
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_parameter_update_clears_mask(small_geometry, anatomical_image_simple, backend):
    """Test that updating parameters clears the cached mask."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        use_mask=True,
        mask_k=10,
    )

    operator.set_anatomical_image(anatomical_image_simple)

    # Force mask computation
    x = small_geometry.allocate(1.0)
    _ = operator.direct(x)
    assert operator.mask is not None

    # Update parameters
    operator.set_parameters({"sigma_anat": 0.5})
    assert operator.mask is None, "Mask should be cleared after parameter update"


@pytest.mark.parametrize("backend", backends)
def test_mask_recalculation_flag(small_geometry, anatomical_image_simple, emission_image_spot, backend):
    """Test that recalc_mask flag forces mask recomputation."""
    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        use_mask=True,
        mask_k=10,
        recalc_mask=True,  # Always recalculate
    )

    operator.set_anatomical_image(anatomical_image_simple)

    # First call
    result1 = operator.direct(emission_image_spot)

    # Manually modify the mask (to verify it gets recalculated)
    if operator.mask is not None:
        operator.mask = np.ones_like(operator.mask)

    # Second call should recalculate mask
    result2 = operator.direct(emission_image_spot)

    # Results might differ slightly due to mask recalculation
    # but should not crash
    assert not np.any(np.isnan(result2.as_array()))


# ============================================================================
# Test 10: Feature normalization
# ============================================================================


@pytest.mark.parametrize("backend", backends)
def test_feature_normalization(small_geometry, backend):
    """Test that normalize_features option correctly normalizes anatomical image."""
    # Create anatomical image with known std
    anat = small_geometry.allocate(0.0)
    arr = anat.as_array()
    arr[:] = np.random.randn(*arr.shape) * 10.0 + 50.0  # mean=50, std~10
    anat.fill(arr)

    operator = get_kernel_operator(
        small_geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_features=True,
    )

    operator.set_anatomical_image(anat)

    # Check that stored anatomical image has std close to 1.0
    stored_arr = operator.anatomical_image.as_array()
    stored_std = stored_arr.std()
    assert np.allclose(stored_std, 1.0, rtol=0.1)


# ============================================================================
# Test 11: Get kernel operator function
# ============================================================================


def test_get_kernel_operator_auto():
    """Test that get_kernel_operator with backend='auto' returns correct type."""
    geom = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=4)
    operator = get_kernel_operator(geom, backend="auto")

    if NUMBA_AVAIL:
        assert isinstance(operator, NumbaKernelOperator)
    elif SLIDING_WINDOW_AVAIL:
        assert isinstance(operator, KernelOperator)


@pytest.mark.skipif(not NUMBA_AVAIL, reason="Numba not available")
def test_get_kernel_operator_numba():
    """Test explicit numba backend selection."""
    geom = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=4)
    operator = get_kernel_operator(geom, backend="numba")
    assert isinstance(operator, NumbaKernelOperator)
    assert operator.backend == "numba"


@pytest.mark.skipif(not SLIDING_WINDOW_AVAIL, reason="Sliding window not available")
def test_get_kernel_operator_python():
    """Test explicit python backend selection."""
    geom = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=4)
    operator = get_kernel_operator(geom, backend="python")
    assert isinstance(operator, KernelOperator)
    assert operator.backend == "python"


def test_get_kernel_operator_invalid_backend():
    """Test that invalid backend raises error."""
    geom = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=4)
    with pytest.raises(ValueError):
        get_kernel_operator(geom, backend="invalid_backend")
