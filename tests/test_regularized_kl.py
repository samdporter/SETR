import numpy as np
import pytest
from cil.framework import ImageData
from cil.optimisation.functions import KullbackLeibler

from setr.cil_extensions.functions import RegularizedKL, ShiftedKullbackLeibler


def setup_data(shape=(10, 20)):
    """Creates basic test data."""
    ig = ImageData(np.zeros(shape))
    b = ig.copy()
    b.fill(np.random.uniform(0.1, 10, size=shape))
    x = ig.copy()
    x.fill(np.random.uniform(0.1, 10, size=shape))
    return ig, b, x


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("softplus_beta", [None, 0.1])
def test_rkl_minimum_at_zero(backend, softplus_beta):
    """Test that RegularizedKL has a value of ~0 at its minimum."""
    ig, b, _ = setup_data()
    shift = 0.0
    eta = 0.0

    # The minimum is at x + eta + shift = b + shift => x = b
    x_min = b.copy()

    rkl = RegularizedKL(
        b=b,
        eta=eta,
        shift=shift,
        softplus_beta=softplus_beta,
        backend=backend,
    )

    # Value at the minimum should be close to zero
    val_at_min = rkl(x_min)
    assert val_at_min == pytest.approx(0.0, abs=1e-6)

    # Gradient at the minimum should be zero
    grad_at_min = rkl.gradient(x_min)
    np.testing.assert_allclose(grad_at_min.as_array(), 0.0, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_rkl_hard_floor_behavior(backend):
    """Test the hard floor behavior of RegularizedKL."""
    ig, b, _ = setup_data()
    floor_eps = 1e-4

    rkl = RegularizedKL(
        b=b,
        floor_eps=floor_eps,
        softplus_beta=None,  # Hard floor
        backend=backend,
    )

    # Case 1: x is well above the floor
    x_above = b.copy() * 2
    s_raw_above = x_above.as_array()
    s_floored_above = np.maximum(s_raw_above, floor_eps)
    grad_above = rkl.gradient(x_above).as_array()
    expected_grad_above = 1.0 - b.as_array() / s_floored_above
    np.testing.assert_allclose(grad_above, expected_grad_above)

    # Case 2: x is such that s_raw is below the floor
    x_below = ig.copy()
    x_below.fill(floor_eps / 2)
    grad_below = rkl.gradient(x_below).as_array()
    # Gradient should be zero because of the chain rule factor
    np.testing.assert_allclose(grad_below, 0.0, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_rkl_soft_floor_behavior(backend):
    """Test the soft floor behavior of RegularizedKL."""
    ig, b, _ = setup_data()
    floor_eps = 1e-4
    beta = 0.01

    rkl = RegularizedKL(
        b=b,
        floor_eps=floor_eps,
        softplus_beta=beta,  # Soft floor
        backend=backend,
    )

    # Case 1: x is well above the floor
    x_above = b.copy() * 2
    s_raw_above = x_above.as_array()
    t_above = (s_raw_above - floor_eps) / beta
    chain_above = 1.0 / (1.0 + np.exp(-t_above))
    # Chain factor should be close to 1
    assert np.all(chain_above > 0.999)

    # Case 2: x is well below the floor
    x_below = ig.copy()
    x_below.fill(-1.0)
    s_raw_below = x_below.as_array()
    t_below = (s_raw_below - floor_eps) / beta
    chain_below = 1.0 / (1.0 + np.exp(-t_below))
    # Chain factor should be close to 0
    assert np.all(chain_below < 1e-6)

    grad_below = rkl.gradient(x_below).as_array()
    # Gradient should be close to zero due to the small chain factor
    np.testing.assert_allclose(grad_below, 0.0, atol=1e-5)


def test_rkl_vs_skl_positive_domain():
    """Compare RegularizedKL (hard floor) with ShiftedKullbackLeibler
    in the positive domain where they should match."""
    ig, b, x = setup_data()

    # Ensure x is positive so no flooring occurs
    x.maximum(0.1, out=x)

    # ShiftedKL (the reference)
    skl = ShiftedKullbackLeibler(b=b, shift=0.0, eta=0.0)
    skl_val = skl(x)
    skl_grad = skl.gradient(x)

    # RegularizedKL with a very small floor_eps
    rkl = RegularizedKL(
        b=b,
        shift=0.0,
        eta=0.0,
        floor_eps=1e-12,
        softplus_beta=None,  # Hard floor
    )
    rkl_val = rkl(x)
    rkl_grad = rkl.gradient(x)

    # The values should be almost identical
    assert rkl_val == pytest.approx(skl_val, rel=1e-6)

    # The gradients should be almost identical
    np.testing.assert_allclose(rkl_grad.as_array(), skl_grad.as_array(), rtol=1e-6)