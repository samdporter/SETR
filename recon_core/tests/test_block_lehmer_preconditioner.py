import numpy as np
import pytest

try:
    from cil.framework import VectorGeometry

    from recon_core.cil_extensions.preconditioners.preconditioners import (
        BlockLehmerMeanPreconditioner,
        LehmerMeanPreconditioner,
    )
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"Skipping BlockLehmer preconditioner tests; dependencies unavailable: {exc}",
        allow_module_level=True,
    )


def _make_blender(p=0.1, epsilon=1e-12, max_value=np.inf, output_scale=1.0):
    obj = object.__new__(BlockLehmerMeanPreconditioner)
    obj.p = float(p)
    obj.epsilon = float(epsilon)
    obj.max_value = float(max_value)
    obj.output_scale = float(output_scale)
    return obj


def _parallel_sum(left, right):
    return np.linalg.inv(np.linalg.inv(left) + np.linalg.inv(right))


class _ConstantPreconditioner:
    def __init__(self, value):
        self.value = value

    def compute_preconditioner(self, _algorithm):
        return self.value.copy()


def test_scalar_lehmer_p0_half_scale_matches_parallel_sum():
    geometry = VectorGeometry(4)
    left = geometry.allocate()
    right = geometry.allocate()
    left.fill(np.array([1.0, 2.0, 7.0, 50.0]))
    right.fill(np.array([3.0, 11.0, 0.5, 0.02]))
    lehmer = LehmerMeanPreconditioner(
        [_ConstantPreconditioner(left), _ConstantPreconditioner(right)],
        p=0.0,
        epsilon=1e-12,
        output_scale=0.5,
    )

    out = lehmer.compute_preconditioner(object()).as_array()
    expected = left.as_array() * right.as_array() / (left.as_array() + right.as_array())
    assert np.allclose(out, expected, rtol=5e-7, atol=1e-8)


def test_block_lehmer_keeps_scalar_identity_fixed_point():
    lam = 2.5
    block = np.zeros((3, 2, 2, 2), dtype=np.float64)
    block[..., 0, 0] = lam
    block[..., 1, 1] = lam
    scalar = np.full((3, 2), lam, dtype=np.float64)

    blender = _make_blender(p=0.1)
    out = blender._blend_block_and_scalar(block, scalar)

    assert np.allclose(out[..., 0, 0], lam)
    assert np.allclose(out[..., 1, 1], lam)
    assert np.allclose(out[..., 0, 1], 0.0)
    assert np.allclose(out[..., 1, 0], 0.0)


def test_block_lehmer_p1_matches_arithmetic_mean_in_eigenspace():
    block = np.zeros((2, 1, 2, 2), dtype=np.float64)
    block[..., 0, 0] = np.array([[1.0], [4.0]])
    block[..., 1, 1] = np.array([[3.0], [8.0]])
    scalar = np.array([[5.0], [2.0]], dtype=np.float64)

    blender = _make_blender(p=1.0)
    out = blender._blend_block_and_scalar(block, scalar)

    expected_eig1 = 0.5 * (block[..., 0, 0] + scalar)
    expected_eig2 = 0.5 * (block[..., 1, 1] + scalar)
    assert np.allclose(out[..., 0, 0], expected_eig1)
    assert np.allclose(out[..., 1, 1], expected_eig2)
    assert np.allclose(out[..., 0, 1], 0.0)
    assert np.allclose(out[..., 1, 0], 0.0)


def test_block_lehmer_output_is_symmetric_and_positive():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 3, 2, 2))
    block = np.matmul(a, np.swapaxes(a, -1, -2)) + 1e-3 * np.eye(2)
    scalar = np.abs(rng.normal(size=(4, 3))) + 1e-3

    blender = _make_blender(p=0.1, epsilon=1e-12)
    out = blender._blend_block_and_scalar(block, scalar)

    assert np.allclose(out, np.swapaxes(out, -1, -2))
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_block_lehmer_harmonic_with_diag_scalar():
    block = np.zeros((2, 1, 2, 2), dtype=np.float64)
    block[..., 0, 0] = np.array([[2.0], [4.0]])
    block[..., 1, 1] = np.array([[6.0], [8.0]])
    scalar = np.zeros_like(block)
    scalar[..., 0, 0] = np.array([[10.0], [5.0]])
    scalar[..., 1, 1] = np.array([[12.0], [7.0]])

    blender = _make_blender(p=0.0)
    out = blender._blend_block_and_scalar(block, scalar)

    expected_00 = 2.0 / (1.0 / block[..., 0, 0] + 1.0 / scalar[..., 0, 0])
    expected_11 = 2.0 / (1.0 / block[..., 1, 1] + 1.0 / scalar[..., 1, 1])
    assert np.allclose(out[..., 0, 0], expected_00)
    assert np.allclose(out[..., 1, 1], expected_11)
    assert np.allclose(out[..., 0, 1], 0.0)
    assert np.allclose(out[..., 1, 0], 0.0)


def test_block_lehmer_output_scale_halves_blend():
    block = np.zeros((1, 1, 2, 2), dtype=np.float64)
    block[..., 0, 0] = 2.0
    block[..., 1, 1] = 4.0
    scalar = np.full((1, 1), 3.0, dtype=np.float64)

    unscaled = _make_blender(p=0.1, output_scale=1.0)._blend_block_and_scalar(block, scalar)
    scaled = _make_blender(p=0.1, output_scale=0.5)._blend_block_and_scalar(block, scalar)

    assert np.allclose(scaled, 0.5 * unscaled)


def test_block_lehmer_p0_half_scale_matches_parallel_sum_for_noncommuting_blocks():
    left = np.array(
        [
            [[[4.0, 1.25], [1.25, 2.5]], [[7.0, -1.6], [-1.6, 3.0]]],
            [[[1.8, 0.55], [0.55, 5.0]], [[9.0, 2.0], [2.0, 6.0]]],
        ],
        dtype=np.float64,
    )
    right = np.array(
        [
            [[[0.5, 0.0], [0.0, 9.0]], [[12.0, 0.0], [0.0, 0.75]]],
            [[[15.0, 0.0], [0.0, 0.4]], [[1.25, 0.0], [0.0, 20.0]]],
        ],
        dtype=np.float64,
    )

    blender = _make_blender(p=0.0, output_scale=0.5)
    out = blender._blend_blocks(left, right)

    expected = _parallel_sum(left, right)
    assert np.allclose(out, expected, rtol=1e-11, atol=1e-12)


def test_block_lehmer_p1_matches_arithmetic_mean_for_noncommuting_blocks():
    left = np.array(
        [
            [[4.0, 1.5], [1.5, 3.0]],
            [[2.0, -0.4], [-0.4, 1.0]],
        ],
        dtype=np.float64,
    )
    right = np.array(
        [
            [[1.5, -0.25], [-0.25, 5.0]],
            [[6.0, 1.2], [1.2, 4.0]],
        ],
        dtype=np.float64,
    )

    blender = _make_blender(p=1.0)
    out = blender._blend_blocks(left, right)

    assert np.allclose(out, 0.5 * (left + right), rtol=1e-12, atol=1e-12)


def test_block_lehmer_full_block_output_is_symmetric_and_positive_definite():
    rng = np.random.default_rng(1234)
    left_factors = rng.normal(size=(5, 4, 2, 2))
    right_factors = rng.normal(size=(5, 4, 2, 2))
    identity = np.eye(2, dtype=np.float64)
    left = left_factors @ np.swapaxes(left_factors, -1, -2) + 1e-2 * identity
    right = right_factors @ np.swapaxes(right_factors, -1, -2) + 1e-2 * identity

    blender = _make_blender(p=0.1, output_scale=0.5)
    out = blender._blend_blocks(left, right)

    assert np.allclose(out, np.swapaxes(out, -1, -2), rtol=1e-12, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(out) > 0.0)


def test_block_lehmer_p0_half_scale_handles_high_pet_spect_contrast():
    prior = np.array(
        [
            [[3.0, 0.8], [0.8, 2.0]],
            [[5.0, -1.1], [-1.1, 1.5]],
            [[2.5, 0.6], [0.6, 4.0]],
        ],
        dtype=np.float64,
    )
    data = np.zeros_like(prior)
    data[..., 0, 0] = np.array([1e-6, 1e-5, 1e-4])
    data[..., 1, 1] = np.array([1e4, 1e5, 1e6])

    blender = _make_blender(p=0.0, epsilon=1e-14, output_scale=0.5)
    out = blender._blend_blocks(prior, data)

    expected = _parallel_sum(prior, data)
    assert np.allclose(out, expected, rtol=1e-9, atol=1e-13)
    assert np.all(np.linalg.eigvalsh(out) > 0.0)
