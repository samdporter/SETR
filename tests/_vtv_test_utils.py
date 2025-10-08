"""Shared utilities for VTV preconditioner tests."""

import pytest

try:  # Optional dependency: torch
    import torch  # noqa: F401

    TORCH = torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    TORCH = None
    DEVICE = None
    TORCH_IMPORT_ERROR = exc

torch = TORCH  # expose for convenient import by tests

try:  # Optional dependency: VTV implementation
    from setr.priors.vtv.schatten_norm_gpu_slow import (
        GPUVectorialTotalVariation as VTVSlowBackend,
    )
    from setr.priors.vtv.vtv import WeightedVectorialTotalVariation

    VTV_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on missing deps
    WeightedVectorialTotalVariation = None  # type: ignore
    VTVSlowBackend = None  # type: ignore
    VTV_IMPORT_ERROR = exc


def require_vtv():
    """Skip the current test if torch or VTV backends are unavailable."""

    if TORCH is None:
        pytest.skip(
            f"Skipping VTV tests; PyTorch not available: {TORCH_IMPORT_ERROR}",
            allow_module_level=True,
        )
    if WeightedVectorialTotalVariation is None or VTVSlowBackend is None:
        pytest.skip(
            f"Skipping VTV tests; VTV backend not available: {VTV_IMPORT_ERROR}",
            allow_module_level=True,
        )


class MockJacobian:
    """Minimal Jacobian stub exposing fixed forward/adjoint/sensitivity data."""

    def __init__(self, J_field, S_field, adjoint_field=None):
        self._J = J_field
        self._S = S_field
        self._adjoint = adjoint_field

    def direct(self, _x):
        return self._J

    def adjoint(self, grad_field):
        if self._adjoint is not None:
            return self._adjoint
        # Fallback: return zeros of appropriate shape (used in isotropic-only tests)
        return torch.zeros_like(self._J[..., 0])

    def sensitivity(self, _x):
        return self._S


def make_vtv_harness(J_field, S_field, weights, smoothing="fair", eps=1.0):
    """Instantiate a lightweight WeightedVectorialTotalVariation surrogate."""

    require_vtv()

    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.jacobian = MockJacobian(J_field=J_field, S_field=S_field)
    vtv.weights = weights
    vtv.smoothing = smoothing
    vtv.vtv = VTVSlowBackend(
        eps=eps,
        norm="nuclear",
        smoothing_function=smoothing,
        numpy_out=False,
    )
    return vtv


__all__ = [
    "require_vtv",
    "make_vtv_harness",
    "MockJacobian",
    "TORCH",
    "DEVICE",
    "WeightedVectorialTotalVariation",
]
