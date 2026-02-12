import numpy as np


def _pullback_preconditioner(B: np.ndarray, H_data: np.ndarray, H_prior_common: np.ndarray) -> np.ndarray:
    """Current common-space pullback form: P = B^T (Hc_data + Hc_prior)^(-1) B."""
    H_common = H_prior_common + B @ H_data @ B.T
    return B.T @ np.linalg.inv(H_common) @ B


def _strict_original_preconditioner(
    B: np.ndarray, H_data: np.ndarray, H_prior_common: np.ndarray
) -> np.ndarray:
    """Strict original-space inverse of summed Hessians."""
    H_orig = H_data + B.T @ H_prior_common @ B
    return np.linalg.inv(H_orig)


def test_pullback_matches_strict_for_orthonormal_operator():
    theta = 0.31
    B = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    H_data = np.diag([1.5, 0.8]).astype(np.float64)
    H_prior_common = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=np.float64)

    P_pullback = _pullback_preconditioner(B, H_data, H_prior_common)
    P_strict = _strict_original_preconditioner(B, H_data, H_prior_common)

    assert np.allclose(P_pullback, P_strict, rtol=1e-12, atol=1e-12)


def test_adjointness_does_not_imply_strict_majoriser_for_nonorthonormal_operator():
    # Non-orthonormal map (e.g. scaling-like resampling effect).
    B = np.diag([2.0, 0.5]).astype(np.float64)

    # Adjointness still holds with B^T.
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2)
    y = rng.standard_normal(2)
    lhs = float(np.vdot(B @ x, y))
    rhs = float(np.vdot(x, B.T @ y))
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)

    H_data = np.diag([1.5, 0.8]).astype(np.float64)
    H_prior_common = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=np.float64)

    P_pullback = _pullback_preconditioner(B, H_data, H_prior_common)
    P_strict = _strict_original_preconditioner(B, H_data, H_prior_common)

    # The two preconditioners differ for non-orthonormal B.
    assert not np.allclose(P_pullback, P_strict, rtol=1e-10, atol=1e-10)

    # Equivalently, the implied Hessian from pullback can fail to dominate strict Hessian.
    H_pullback = np.linalg.inv(P_pullback)
    H_strict = np.linalg.inv(P_strict)
    eigvals = np.linalg.eigvalsh(H_pullback - H_strict)
    assert eigvals.min() < -1e-8
