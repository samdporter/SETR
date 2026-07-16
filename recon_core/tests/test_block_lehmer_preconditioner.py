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


def _make_blender(
    p=0.1, epsilon=1e-12, max_value=np.inf, output_scale=1.0, max_relative_contrast=np.inf
):
    obj = object.__new__(BlockLehmerMeanPreconditioner)
    obj.p = float(p)
    obj.epsilon = float(epsilon)
    obj.max_value = float(max_value)
    obj.output_scale = float(output_scale)
    obj.max_relative_contrast = float(max_relative_contrast)
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


def test_scalar_lehmer_max_relative_contrast_floors_result_and_is_order_independent():
    """max_relative_contrast gives a genuine floor, order-independently.

    Unlike the block-case cap (which is deliberately one-sided in (prior, data)
    order because production always calls it that way), the scalar cap is
    value-based (max/min), so it must give the same result regardless of which
    operand comes first -- both orderings appear in this codebase
    (BlockLehmerMeanPreconditioner's internal hessian-provider path is always called
    as (prior, data), but the diagonal LehmerMeanPreconditioner is called as
    (bsrem/data, prior) in dtnv_common.get_preconditioners).
    """
    geometry = VectorGeometry(1)
    p = 0.3
    max_contrast = 100.0
    big_val = 1.0
    expected_floor = 0.5 * (big_val / max_contrast) * (
        (max_contrast**p + 1) / (max_contrast ** (p - 1) + 1)
    )

    def make(values, order):
        a = geometry.allocate()
        b = geometry.allocate()
        a.fill(np.array([values[0]]))
        b.fill(np.array([values[1]]))
        preconds = [a, b] if order == "ab" else [b, a]
        return LehmerMeanPreconditioner(
            [_ConstantPreconditioner(v) for v in preconds],
            p=p,
            epsilon=1e-14,
            output_scale=0.5,
            max_relative_contrast=max_contrast,
        )

    results = []
    for small_val in [1e-2, 1e-4, 1e-8, 1e-12]:
        for order in ("ab", "ba"):
            lehmer = make([big_val, small_val], order)
            out = lehmer.compute_preconditioner(object()).as_array()[0]
            results.append(float(out))

    # All results (across contrasts >= max_contrast and both orderings) must sit at
    # the same analytic floor -- the output stops tracking the degenerate operand.
    assert np.allclose(results, expected_floor, rtol=1e-4)


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


def _generalized_eigvals(numerator, denominator):
    """Generalized eigenvalues of (numerator, denominator) for SPD 2x2 blocks."""
    chol = np.linalg.cholesky(denominator)
    inv_chol = np.linalg.inv(chol)
    sandwich = inv_chol @ numerator @ np.swapaxes(inv_chol, -1, -2)
    return np.linalg.eigvalsh(sandwich)


def test_block_lehmer_p0_is_generalized_unit_relative_to_parallel_sum():
    """L_0 at output_scale=0.5 must equal the parallel sum exactly (safety baseline)."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(6, 2, 2))
    b = rng.normal(size=(6, 2, 2))
    left = a @ np.swapaxes(a, -1, -2) + 1e-2 * np.eye(2)
    right = b @ np.swapaxes(b, -1, -2) + 1e-2 * np.eye(2)

    blender = _make_blender(p=0.0, epsilon=1e-14, output_scale=0.5)
    out = blender._blend_blocks(left, right)
    parallel = _parallel_sum(left, right)

    gen_eigs = _generalized_eigvals(out, parallel)
    assert np.allclose(gen_eigs, 1.0, rtol=1e-8)


def test_block_lehmer_p_gt0_overshoot_grows_with_relative_contrast():
    """Positive p overshoots the parallel sum by ~0.5*contrast**p on the small operand.

    This is the mechanism documented in EXPERIMENT_FINDINGS.md Exp 1: any p>0 that
    escapes stagnation on a degenerate operand overshoots the majoriser bound by the
    same factor, whichever operand happens to be small.
    """
    left = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    contrasts = [1e2, 1e4, 1e6]
    p = 0.1
    for r in contrasts:
        right = np.array([[[1.0 / r, 0.0], [0.0, 1.0 / r]]])
        blender = _make_blender(p=p, epsilon=1e-14, output_scale=0.5)
        out = blender._blend_blocks(left, right)
        parallel = _parallel_sum(left, right)
        gen_eigs = _generalized_eigvals(out, parallel)
        expected_overshoot = 0.5 * r**p * 2  # factor of 2 from L_0 vs 0.5*L_0 baseline
        # Overshoot should scale like r**p and be substantially > 1 for p>0.
        assert gen_eigs.max() > 1.0 + 1e-6
        assert gen_eigs.max() < expected_overshoot + 1.0


def test_block_lehmer_max_relative_contrast_floors_result_as_data_degenerates():
    """max_relative_contrast gives a genuine floor, not just a bounded overshoot ratio.

    ``_blend_blocks`` is always called as ``_blend_blocks(prior, data)`` in production
    (see ``_compute_from_hessian_provider``), so the failure mode this guards against is
    one-sided: the data (right) operand collapsing relative to the prior (left) operand.
    Without the cap, the blended eigenvalue keeps shrinking to zero along with the data
    operand (just at a softened rate, ~data**(1-p)). With the cap, once the prior/data
    contrast exceeds max_relative_contrast the result plateaus at a fixed value
    proportional to the prior operand -- it stops tracking the data operand's magnitude
    at all, which is the actual point of a stagnation-escape floor.
    """
    p = 0.3
    max_contrast = 100.0
    prior = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    expected_floor = (1.0 / max_contrast) * (max_contrast**p + 1) / (max_contrast ** (p - 1) + 1)
    expected_floor *= 0.5  # output_scale

    unclipped_eigs = []
    clipped_eigs = []
    for data_scale in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-14]:
        data = prior * data_scale
        unclipped = _make_blender(p=p, epsilon=1e-14, output_scale=0.5)
        clipped = _make_blender(
            p=p, epsilon=1e-14, output_scale=0.5, max_relative_contrast=max_contrast
        )
        unclipped_eigs.append(unclipped._blend_blocks(prior, data)[0, 0, 0])
        clipped_eigs.append(clipped._blend_blocks(prior, data)[0, 0, 0])

    # Unclipped result keeps shrinking as the data operand shrinks (no floor).
    assert unclipped_eigs[-1] < unclipped_eigs[0] / 1e6

    # Clipped result plateaus at the analytic floor value once contrast exceeds the
    # cap (data_scale <= 1/max_contrast == 1e-2), independent of further shrinkage.
    assert np.allclose(clipped_eigs, expected_floor, rtol=1e-8)


def test_block_lehmer_max_relative_contrast_applies_at_p0():
    """A finite cap must not be silently ignored on the p=0 fast path.

    p=0 with the default infinite cap keeps the exact parallel-sum fast path
    (covered by the p0 equality tests above); p=0 with a finite cap must floor
    like any other p -- consistent with the scalar LehmerMeanPreconditioner,
    which applies its cap for every p. At the cap the blend saturates at
    0.5 * L_0(prior, prior/M) = prior / (1 + M).
    """
    max_contrast = 100.0
    prior = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    expected_floor = 2.0 / (1.0 + max_contrast)

    capped_eigs = []
    for data_scale in [1e-3, 1e-6, 1e-10]:
        blender = _make_blender(
            p=0.0, epsilon=1e-14, output_scale=0.5, max_relative_contrast=max_contrast
        )
        out = blender._blend_blocks(prior, prior * data_scale)
        capped_eigs.append(out[0, 0, 0])
    assert np.allclose(capped_eigs, expected_floor, rtol=1e-8)

    # While the contrast stays below the cap, the capped p=0 blend must still be
    # the exact parallel sum.
    data = prior * 0.5
    blender = _make_blender(
        p=0.0, epsilon=1e-14, output_scale=0.5, max_relative_contrast=max_contrast
    )
    out = blender._blend_blocks(prior, data)
    assert np.allclose(out, _parallel_sum(prior, data), rtol=1e-10)


class _FakeHessianProvider:
    """Minimal stand-in for MajorisingHessianBlockPreconditioner's provider API."""

    hessian_floor = 1e-8
    max_value = np.inf
    safety_scale = 1.0

    def __init__(self, prior_hessian, data_hessian):
        self._prior = prior_hessian
        self._data = data_hessian
        self.shortcut_calls = 0

    def compute_preconditioner(self, _algorithm):
        self.shortcut_calls += 1
        return "provider-shortcut"

    def _compute_prior_hessian_block(self, _image):
        return self._prior

    def _compute_data_hessian_block(self, _image):
        return self._data


def test_block_lehmer_p0_provider_shortcut_disabled_by_finite_cap():
    """The p=0/output_scale=0.5 provider shortcut must yield to a finite cap.

    The shortcut exists for exact majoriser equality; an explicitly finite
    max_relative_contrast asks for a floored blend, which the provider cannot
    supply, so the blend path must run instead.
    """

    class _Algorithm:
        solution = None

    data_scale = 1e-6
    prior_hessian = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    data_hessian = prior_hessian / data_scale  # data precond = 1/data_hessian = 1e-6
    max_contrast = 100.0

    def make(max_relative_contrast):
        blender = _make_blender(
            p=0.0, epsilon=1e-14, output_scale=0.5, max_relative_contrast=max_relative_contrast
        )
        blender.hessian_preconditioner = _FakeHessianProvider(prior_hessian, data_hessian)
        return blender

    uncapped = make(np.inf)
    assert uncapped._compute_from_hessian_provider(_Algorithm()) == "provider-shortcut"
    assert uncapped.hessian_preconditioner.shortcut_calls == 1

    capped = make(max_contrast)
    out = capped._compute_from_hessian_provider(_Algorithm())
    assert capped.hessian_preconditioner.shortcut_calls == 0
    expected_floor = 1.0 / (1.0 + max_contrast)  # 0.5 * L_0(1, 1/M) with prior precond = 1
    assert np.allclose(out[0, 0, 0], expected_floor, rtol=1e-8)
