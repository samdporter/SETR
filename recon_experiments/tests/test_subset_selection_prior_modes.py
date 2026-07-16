"""Tests for the uniform-sampling prior modes of the subset-selection study.

CIL's SVRG/SAGA scale sampled gradient differences by ``num_functions``,
which is only unbiased under uniform sampling. These tests verify that each
prior mode builds a UNIFORMLY-sampled function list that sums exactly to
(data + prior), which together make the stochastic gradient estimator
unbiased for the intended objective.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from cil.framework import VectorData
from cil.optimisation.functions import LeastSquares
from cil.optimisation.operators import MatrixOperator

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "recon_experiments"
    / "studies"
    / "subset_selection"
    / "scripts"
    / "run_subset_selection.py"
)


@pytest.fixture(scope="module")
def runner_module():
    spec = importlib.util.spec_from_file_location("run_subset_selection", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(1234)
    dim = 8
    num_data = 6
    data_funs = [
        LeastSquares(
            MatrixOperator(rng.normal(size=(dim, dim)).astype(np.float64)),
            VectorData(rng.normal(size=dim).astype(np.float64)),
        )
        for _ in range(num_data)
    ]
    prior = LeastSquares(
        MatrixOperator(rng.normal(size=(dim, dim)).astype(np.float64)),
        VectorData(rng.normal(size=dim).astype(np.float64)),
    )
    x = VectorData(rng.normal(size=dim).astype(np.float64))
    snapshot = VectorData(rng.normal(size=dim).astype(np.float64))
    return data_funs, prior, x, snapshot


def _full_gradient(funs, x):
    return sum(f.gradient(x).as_array() for f in funs)


def _target_gradient(data_funs, prior, x):
    return _full_gradient(data_funs, x) + prior.gradient(x).as_array()


@pytest.mark.parametrize(
    "mode,expected_n,expected_epoch",
    [("folded", 6, 6), ("half", 12, 12), ("epoch", 7, 7)],
)
def test_list_structure_and_epoch_length(
    runner_module, problem, mode, expected_n, expected_epoch
):
    data_funs, prior, _, _ = problem
    funs, epoch_length = runner_module.build_prior_mode_functions(
        mode, data_funs, prior
    )
    assert len(funs) == expected_n
    assert epoch_length == expected_epoch


@pytest.mark.parametrize("mode", ["folded", "half", "epoch"])
def test_function_list_sums_to_data_plus_prior(runner_module, problem, mode):
    data_funs, prior, x, _ = problem
    funs, _ = runner_module.build_prior_mode_functions(mode, data_funs, prior)

    target_value = sum(f(x) for f in data_funs) + prior(x)
    assert np.isclose(sum(f(x) for f in funs), target_value, rtol=1e-12)

    target_grad = _target_gradient(data_funs, prior, x)
    np.testing.assert_allclose(_full_gradient(funs, x), target_grad, rtol=1e-10)


@pytest.mark.parametrize("mode", ["folded", "half", "epoch"])
def test_svrg_estimator_is_unbiased_under_cil_scaling(runner_module, problem, mode):
    """Expected value of CIL's SVRG estimator equals the true full gradient.

    CIL's estimator for sampled index i is
        n * (grad_i(x) - grad_i(snapshot)) + full_gradient(snapshot).
    With the uniform probabilities the study uses, its expectation over i must
    equal the gradient of (data + prior) at x.
    """
    data_funs, prior, x, snapshot = problem
    funs, _ = runner_module.build_prior_mode_functions(mode, data_funs, prior)
    probs = runner_module.get_data_sampling_probabilities(funs)
    assert np.allclose(probs, 1.0 / len(funs))

    n = len(funs)
    anchor = _full_gradient(funs, snapshot)
    expectation = anchor.copy()
    for p, f in zip(probs, funs):
        expectation = expectation + p * n * (
            f.gradient(x).as_array() - f.gradient(snapshot).as_array()
        )

    np.testing.assert_allclose(
        expectation, _target_gradient(data_funs, prior, x), rtol=1e-10
    )


def test_half_mode_prior_probability_is_half(runner_module, problem):
    data_funs, prior, _, _ = problem
    funs, _ = runner_module.build_prior_mode_functions("half", data_funs, prior)
    num_prior_slots = len(funs) - len(data_funs)
    assert num_prior_slots / len(funs) == pytest.approx(0.5)


def test_unknown_mode_raises(runner_module, problem):
    data_funs, prior, _, _ = problem
    with pytest.raises(ValueError):
        runner_module.build_prior_mode_functions("always", data_funs, prior)
