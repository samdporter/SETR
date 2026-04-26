import numpy as np
import pytest
import sys
from pathlib import Path


def test_project_then_divide_underestimates_with_anticorrelated_inputs():
    # Two-voxel averaging example: x and s_inv are anti-correlated.
    r = np.array([[0.5, 0.5]], dtype=np.float64)
    x = np.array([100.0, 1.0], dtype=np.float64)
    s_inv = np.array([0.01, 1.0], dtype=np.float64)
    eps = 1e-8

    # Correct shared-space ordering: first pull sensitivity to image space, then apply
    # the per-voxel nonlinearity 1 / ((x + eps) * s_inv).
    h_then_project = (r @ (1.0 / ((x + eps) * s_inv))).item()

    # Incorrect ordering: project x and s_inv first, then apply the nonlinearity.
    project_then_h = (1.0 / (((r @ x) + eps) * (r @ s_inv))).item()

    assert h_then_project > 20.0 * project_then_h


def test_shared_space_data_hessian_matches_voxelwise_formula():
    x_pet = np.array([100.0, 1.0], dtype=np.float64)
    x_spect = np.array([20.0, 2.0], dtype=np.float64)
    s_inv_pet = np.array([0.01, 1.0], dtype=np.float64)
    s_inv_spect = np.array([0.02, 0.5], dtype=np.float64)
    eps = 1e-8

    expected_pet = 1.0 / ((x_pet + eps) * s_inv_pet)
    expected_spect = 1.0 / ((x_spect + eps) * s_inv_spect)

    assert np.allclose(
        expected_pet,
        np.array([1.0 / ((100.0 + eps) * 0.01), 1.0 / ((1.0 + eps) * 1.0)]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        expected_spect,
        np.array([1.0 / ((20.0 + eps) * 0.02), 1.0 / ((2.0 + eps) * 0.5)]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_shared_space_sensitivity_pullback_precedes_em_denominator():
    # Model a native-space sensitivity A^T 1 and a PET-grid pullback W^T.
    w_t = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    native_sens = np.array([4.0, 1.0], dtype=np.float64)
    shared_x = np.array([2.0, 3.0], dtype=np.float64)
    eps = 1e-8

    pulled_sens = w_t @ native_sens
    shared_denominator = (shared_x + eps) * (1.0 / np.maximum(pulled_sens, eps))
    native_denominator = (shared_x + eps) * (1.0 / np.maximum(native_sens, eps))

    assert np.allclose(pulled_sens, np.array([3.4, 1.3]), atol=1e-12, rtol=1e-12)
    assert not np.allclose(shared_denominator, native_denominator, atol=1e-12, rtol=1e-12)


def test_composed_hessian_vector_product_recurses_through_operator_chain():
    exp_src = Path(__file__).resolve().parents[2] / "recon_experiments" / "src"
    if exp_src.is_dir() and str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from cil.optimisation.functions import OperatorCompositionFunction
        from recon_experiments.runners.dtnv_common import _multiply_with_composed_hessian
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CIL/recon_experiments unavailable: {exc}")

    class _Quadratic:
        def multiply_with_Hessian(self, _x, direction):
            h = np.diag([4.0, 9.0])
            return h @ np.asarray(direction, dtype=np.float64)

    class _MatrixOperator:
        def __init__(self, matrix):
            self.matrix = np.array(matrix, dtype=np.float64, copy=True)

        def direct(self, x):
            return self.matrix @ np.asarray(x, dtype=np.float64)

        def adjoint(self, y):
            return self.matrix.T @ np.asarray(y, dtype=np.float64)

    k1 = _MatrixOperator([[1.0, 2.0], [0.0, 1.0]])
    k2 = _MatrixOperator([[2.0, 0.0], [1.0, 1.0]])
    composed = OperatorCompositionFunction(
        OperatorCompositionFunction(_Quadratic(), k1),
        k2,
    )

    x = np.array([0.5, -1.0], dtype=np.float64)
    direction = np.array([1.0, 3.0], dtype=np.float64)
    hv = _multiply_with_composed_hessian(composed, x, direction)

    h = np.diag([4.0, 9.0])
    expected = k2.matrix.T @ k1.matrix.T @ h @ k1.matrix @ k2.matrix @ direction
    assert np.allclose(hv, expected, atol=1e-12, rtol=1e-12)
