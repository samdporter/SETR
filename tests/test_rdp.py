import sys
from itertools import product
from pathlib import Path

import pytest

# Ensure src/ is on the path when running tests from the repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch", reason="RelativeDifferencePrior requires PyTorch")

from setr.priors.rdp import RelativeDifferencePrior


def _make_prior(**kwargs):
    class _DummyGeometry:
        def voxel_sizes(self):
            return (1.0, 1.0, 1.0)

    return RelativeDifferencePrior(_DummyGeometry(), **kwargs)


def _nd_indices(shape):
    return product(*(range(s) for s in shape))


def _finite_difference_grad(prior, x, step=5e-4):
    grad = torch.zeros_like(x)
    for idx in _nd_indices(x.shape):
        direction = torch.zeros_like(x)
        direction[idx] = 1.0
        f_plus = prior._value_tensor(x + step * direction)
        f_minus = prior._value_tensor(x - step * direction)
        grad[idx] = (f_plus - f_minus) / (2.0 * step)
    return grad


@pytest.mark.parametrize("gamma, epsilon", [(0.15, 1e-2), (0.4, 5e-3)])
def test_rdp_gradient_matches_central_difference(gamma, epsilon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = _make_prior(gamma=gamma, epsilon=epsilon)
    x = torch.tensor(
        [[[0.2], [0.8], [0.5]], [[0.4], [0.1], [0.7]]],
        dtype=torch.float32,
        device=device,
    )

    analytic = prior._grad_tensor(x)
    numeric = _finite_difference_grad(prior, x, step=1e-4)

    assert torch.allclose(analytic, numeric, rtol=3e-3, atol=3e-3)


def test_rdp_hessian_vector_matches_finite_difference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = _make_prior(gamma=0.25, epsilon=8e-3)
    x = torch.tensor(
        [[[0.3], [0.6], [0.1]], [[0.5], [0.2], [0.9]]],
        dtype=torch.float32,
        device=device,
    )
    direction = torch.tensor(
        [[[0.5], [-0.2], [0.1]], [[-0.3], [0.4], [0.7]]],
        dtype=torch.float32,
        device=device,
    )

    hv = prior._hess_vec_tensor(x, direction)

    step = 2e-3
    grad_plus = prior._grad_tensor(x + step * direction)
    grad_minus = prior._grad_tensor(x - step * direction)
    numeric = (grad_plus - grad_minus) / (2.0 * step)

    assert torch.allclose(hv, numeric, rtol=8e-3, atol=8e-3)


def test_rdp_hessian_diagonal_matches_basis_columns():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = _make_prior(gamma=0.35, epsilon=1e-2)
    x = torch.tensor(
        [[[0.1], [0.4], [0.7]], [[0.3], [0.2], [0.5]]],
        dtype=torch.float32,
        device=device,
    )

    diag = prior._hess_diag_tensor(x)

    step = 5e-4
    for idx in _nd_indices(x.shape):
        basis = torch.zeros_like(x)
        basis[idx] = 1.0
        grad_plus = prior._grad_tensor(x + step * basis)
        grad_minus = prior._grad_tensor(x - step * basis)
        numeric = (grad_plus[idx] - grad_minus[idx]) / (2.0 * step)
        assert pytest.approx(diag[idx].item(), rel=2e-2, abs=2e-2) == numeric.item()
