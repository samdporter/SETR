import math

import pytest

from tests._vtv_test_utils import (
    DEVICE,
    MockJacobian,
    WeightedVectorialTotalVariation,
    require_vtv,
    torch,
)

require_vtv()


class IdentityBDCA:
    def direct(self, x):
        return x

    def adjoint(self, arr, out=None):
        return arr if out is None else out.copy_(arr)


class ScalarSmoothNorm:
    """Simple smooth norm φ(U) = sqrt(||U||² + δ²)."""

    def __init__(self, delta):
        self.delta = torch.tensor(delta, device=DEVICE)

    def __call__(self, U):
        return torch.sqrt(torch.sum(U * U) + self.delta**2)

    def gradient(self, U):
        denom = torch.sqrt(torch.sum(U * U) + self.delta**2)
        return U / denom


def _make_scalar_wvtv(weight):
    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.bdc2a = IdentityBDCA()

    def direct(x_arr):
        return x_arr.unsqueeze(-1)

    def adjoint(inner):
        return inner.squeeze(-1)

    S_field = torch.ones(1, 1, 1, 1, 1, device=DEVICE)
    vtv.jacobian = MockJacobian(J_field=None, S_field=S_field)
    vtv.jacobian.direct = direct  # type: ignore
    vtv.jacobian.adjoint = adjoint  # type: ignore

    vtv.weights = torch.tensor([[[[weight]]]], device=DEVICE)
    vtv.vtv = ScalarSmoothNorm(delta=0.25)
    vtv.smoothing = "fair"
    return vtv


class LinearJacobian:
    def __init__(self, scale):
        self.scale = scale

    def direct(self, x):
        return x.unsqueeze(-1) * self.scale

    def adjoint(self, grad_field):
        return torch.sum(self.scale * grad_field, dim=-1)

    def sensitivity(self, _):  # compatibility
        return self.scale


class QuadraticVTV:
    def __call__(self, U):
        return 0.5 * torch.sum(U * U)

    def gradient(self, U):
        return U


def test_scalar_objective_matches_expected():
    x = torch.tensor([[[[0.3]]]], device=DEVICE)
    weight = 2.0
    vtv = _make_scalar_wvtv(weight)

    obj = vtv(x)
    expected = math.sqrt((weight * 0.3) ** 2 + 0.25**2)
    assert pytest.approx(expected, rel=1e-6) == obj.item()


def test_scalar_gradient_matches_analytic():
    x = torch.tensor([[[[0.4]]]], device=DEVICE)
    weight = 3.0
    vtv = _make_scalar_wvtv(weight)

    grad = vtv.gradient(x)

    numerator = weight**2 * 0.4
    denom = math.sqrt((weight * 0.4) ** 2 + 0.25**2)
    expected = numerator / denom
    assert grad.shape == x.shape
    assert pytest.approx(expected, rel=1e-6) == grad.item()


def test_scalar_gradient_matches_finite_difference():
    dtype = torch.float64
    x = torch.tensor([[[[0.2]]]], device=DEVICE, dtype=dtype)
    weight = 1.5
    vtv = _make_scalar_wvtv(weight)

    grad = vtv.gradient(x)

    eps = 1e-6
    x_plus = torch.tensor([[[[0.2 + eps]]]], device=DEVICE, dtype=dtype)
    x_minus = torch.tensor([[[[0.2 - eps]]]], device=DEVICE, dtype=dtype)

    f_plus = vtv(x_plus)
    f_minus = vtv(x_minus)
    fd = (f_plus - f_minus) / (2 * eps)

    assert pytest.approx(fd.item(), rel=1e-5, abs=1e-6) == grad.item()


def _make_linear_wvtv(weights, scale):
    vtv = object.__new__(WeightedVectorialTotalVariation)
    vtv.bdc2a = IdentityBDCA()
    vtv.jacobian = LinearJacobian(scale)
    vtv.weights = weights
    vtv.vtv = QuadraticVTV()
    vtv.smoothing = "fair"
    return vtv


def test_linear_jacobian_adjoint_property():
    weights = torch.tensor([[[[1.2, 0.8]]]], device=DEVICE)
    scale = torch.tensor([[[[[1.0, -0.5, 0.25], [0.3, 0.6, -0.2]]]]], device=DEVICE)
    jac = LinearJacobian(scale)

    x = torch.tensor([[[[0.7, -0.4]]]], device=DEVICE)
    y = torch.tensor([[[[[0.5, -0.2, 0.3], [0.1, 0.4, -0.6]]]]], device=DEVICE)

    inner1 = torch.sum(jac.direct(x) * y)
    inner2 = torch.sum(x * jac.adjoint(y))
    assert pytest.approx(inner1.item(), rel=1e-6) == inner2.item()


def test_linear_objective_and_gradient_match_formula():
    weights = torch.tensor([[[[1.5, 0.6]]]], device=DEVICE)
    scale = torch.tensor([[[[[1.0, 2.0], [-1.0, 0.5]]]]], device=DEVICE)
    vtv = _make_linear_wvtv(weights, scale)

    x = torch.tensor([[[[0.2, -0.3]]]], device=DEVICE)

    obj = vtv(x)
    U = weights.unsqueeze(-1) * vtv.jacobian.direct(x)
    expected_obj = 0.5 * torch.sum(U * U)
    assert pytest.approx(expected_obj.item(), rel=1e-6) == obj.item()

    grad = vtv.gradient(x)
    summed = torch.sum(scale * (weights.unsqueeze(-1) ** 2) * vtv.jacobian.direct(x), dim=-1)
    expected_grad = summed
    assert torch.allclose(grad, expected_grad, rtol=1e-6, atol=1e-6)
