import numpy as np
import pytest

try:
    from recon_core.cil_extensions.preconditioners.preconditioners import (
        BlockLehmerMeanPreconditioner,
    )
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"Skipping BlockLehmer preconditioner tests; dependencies unavailable: {exc}",
        allow_module_level=True,
    )


def _make_blender(p=0.1, epsilon=1e-12, max_value=np.inf):
    obj = object.__new__(BlockLehmerMeanPreconditioner)
    obj.p = float(p)
    obj.epsilon = float(epsilon)
    obj.max_value = float(max_value)
    return obj


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

    expected_00 = 1.0 / (1.0 / block[..., 0, 0] + 1.0 / scalar[..., 0, 0])
    expected_11 = 1.0 / (1.0 / block[..., 1, 1] + 1.0 / scalar[..., 1, 1])
    assert np.allclose(out[..., 0, 0], expected_00)
    assert np.allclose(out[..., 1, 1], expected_11)
    assert np.allclose(out[..., 0, 1], 0.0)
    assert np.allclose(out[..., 1, 0], 0.0)
