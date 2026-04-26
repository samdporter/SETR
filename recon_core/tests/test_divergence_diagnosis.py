"""Targeted tests diagnosing potential divergence causes in preconditioned ISTA.

These tests isolate the key mechanisms that can cause the algorithm to diverge:
1. Gradient clipping threshold positive feedback loop
2. Preconditioner magnitude bounds under realistic conditions
3. ISTA update step size bounds
4. Block preconditioner eigenvalue conditioning
5. Shared-space curvature ordering for EM-style data majorisers
"""

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from cil.framework import BlockDataContainer
    from recon_core.cil_extensions.preconditioners.preconditioners import (
        MajorisingHessianBlockPreconditioner,
        MajorisingHessianDiagonalPreconditioner,
    )
    from recon_core.utils.sirf import get_array
except (ImportError, OSError) as exc:
    pytest.skip(f"CIL/SIRF dependencies unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Lightweight array container for mock tests
# ---------------------------------------------------------------------------

class ArrayContainer:
    def __init__(self, data):
        self._arr = np.array(data, dtype=np.float64, copy=True)

    def asarray(self):
        return self._arr

    def as_array(self):
        return self._arr

    def fill(self, data):
        self._arr[...] = np.asarray(data, dtype=self._arr.dtype)

    @property
    def shape(self):
        return self._arr.shape

    def copy(self):
        return ArrayContainer(self._arr.copy())

    def clone(self):
        return self.copy()

    def get_uniform_copy(self, value):
        return ArrayContainer(np.full_like(self._arr, value))

    def abs(self, out=None):
        arr = np.abs(self._arr)
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

    def max(self):
        return float(np.max(self._arr))

    def maximum(self, other, out=None):
        rhs = other.asarray() if isinstance(other, ArrayContainer) else other
        arr = np.maximum(self._arr, rhs)
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

    def minimum(self, other, out=None):
        rhs = other.asarray() if isinstance(other, ArrayContainer) else other
        arr = np.minimum(self._arr, rhs)
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

    def multiply(self, other, out=None):
        rhs = other.asarray() if isinstance(other, ArrayContainer) else other
        arr = self._arr * rhs
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

    def sapyb(self, a, y, b, out=None):
        rhs = y.asarray() if isinstance(y, ArrayContainer) else y
        arr = a * self._arr + b * rhs
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

    def add(self, other, out=None):
        rhs = other.asarray() if isinstance(other, ArrayContainer) else other
        arr = self._arr + rhs
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

    def __add__(self, other):
        rhs = other.asarray() if isinstance(other, ArrayContainer) else other
        return ArrayContainer(self._arr + rhs)

    __radd__ = __add__

    def __mul__(self, other):
        rhs = other.asarray() if isinstance(other, ArrayContainer) else other
        return ArrayContainer(self._arr * rhs)

    __rmul__ = __mul__


class _IdentityOperator:
    def direct(self, x):
        return x

    def adjoint(self, x):
        return x


# ---------------------------------------------------------------------------
# Test 1: Gradient clipping threshold creates positive feedback loop
# ---------------------------------------------------------------------------

class TestGradientClippingFeedback:
    """The ISTA update clips the gradient by M/step_size where M = x.max().

    If x grows (e.g. from a large preconditioned step), M grows, so the clip
    threshold grows, allowing even LARGER steps next time. This is a positive
    feedback loop that can cause divergence.
    """

    @staticmethod
    def _simulate_ista_step(x_arr, grad_arr, step_size, precond_arr=None):
        """Simulate one ISTA gradient step (without proximal)."""
        M = np.max(x_arr)
        if precond_arr is not None:
            grad_arr = precond_arr * grad_arr
        # Clip
        grad_clipped = np.clip(grad_arr, -M / step_size, M / step_size)
        x_new = x_arr - step_size * grad_clipped
        return x_new

    def test_clip_threshold_grows_with_image_max(self):
        """Demonstrate that clip threshold M/step_size grows when x grows."""
        step_size = 1.0
        x = np.array([100.0, 50.0, 10.0])

        # Clip threshold at iter 0
        M0 = x.max()
        clip0 = M0 / step_size

        # Suppose image grows by 2x due to bad step
        x_grown = 2.0 * x
        M1 = x_grown.max()
        clip1 = M1 / step_size

        assert clip1 == 2.0 * clip0, (
            "Clip threshold should double when image max doubles - "
            "this creates a positive feedback loop"
        )

    def test_runaway_growth_without_decay(self):
        """Simulate multiple ISTA steps showing runaway growth when step_size=1, no decay.

        The adversarial case is a NEGATIVE gradient (ascent direction), which
        causes x_new = x - step * (-|g|) = x + step*|g| to grow.
        """
        step_size = 1.0
        x = np.array([100.0, 50.0, 10.0])
        # Negative gradient → update moves x upward
        grad = np.array([-500.0, -300.0, -200.0])

        maxvals = [x.max()]
        for _ in range(20):
            x = self._simulate_ista_step(x, grad, step_size)
            x = np.maximum(x, 0)  # proximal (non-negativity)
            maxvals.append(x.max())

        # With constant step size and the clip = M/step_size growing,
        # the image max should grow unboundedly
        growth_ratio = maxvals[-1] / maxvals[0]
        assert growth_ratio > 1.0, (
            f"Expected growth in image maximum over iterations, got ratio={growth_ratio:.3f}"
        )

    def test_step_size_decay_limits_growth(self):
        """Show that linear step size decay limits growth vs constant step size."""
        x_const = np.array([100.0, 50.0, 10.0])
        x_decay = x_const.copy()
        grad = np.array([-500.0, -300.0, -200.0])  # Negative → causes growth

        for k in range(20):
            # Constant step size
            x_const = self._simulate_ista_step(x_const, grad, 1.0)
            x_const = np.maximum(x_const, 0)

            # Decaying step size
            step_decay = 1.0 / (1.0 + 0.1 * k)
            x_decay = self._simulate_ista_step(x_decay, grad, step_decay)
            x_decay = np.maximum(x_decay, 0)

        # Decaying step size should result in smaller final values
        assert x_decay.max() < x_const.max(), (
            f"Decaying step should limit growth: decay_max={x_decay.max():.1f}, "
            f"const_max={x_const.max():.1f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Preconditioner magnitude analysis
# ---------------------------------------------------------------------------

class TestPreconditionerMagnitude:
    """Test that preconditioner values don't amplify gradients excessively."""

    def test_bsrem_preconditioner_magnitude_scales_with_x(self):
        """BSREM: P = (x + eps) * s_inv. For large x, P grows linearly with x."""
        x_vals = [1.0, 100.0, 10000.0]
        s_inv = 0.01  # Typical inverse sensitivity
        eps = 1e-8

        precond_vals = [(x + eps) * s_inv for x in x_vals]

        # P grows linearly with x
        ratio_10_to_1 = precond_vals[1] / precond_vals[0]
        ratio_100_to_10 = precond_vals[2] / precond_vals[1]
        assert abs(ratio_10_to_1 - 100.0) < 0.1
        assert abs(ratio_100_to_10 - 100.0) < 0.1

    def test_majorising_preconditioner_bounded_by_max_value(self):
        """The majorising preconditioner should respect max_value clamp."""
        precond = MajorisingHessianBlockPreconditioner(
            s_inv=None,
            prior=SimpleNamespace(operator=_IdentityOperator()),
            hessian_floor=1e-8,
            max_value=5.0,
            safety_scale=1.0,
        )

        # Very small Hessian → large inverse → should be clamped
        tiny_h = np.array(
            [[[1e-10, 0.0], [0.0, 1e-10]]],
            dtype=np.float64,
        )
        precond._compute_prior_hessian_block = lambda _: tiny_h.copy()
        precond._compute_data_hessian_block = lambda _: np.zeros_like(tiny_h)

        algo = SimpleNamespace(solution=None)
        out = precond.compute_preconditioner(algo)
        eigvals = np.linalg.eigvalsh(out)
        assert np.all(eigvals <= 5.0 + 1e-12), (
            f"Preconditioner eigenvalues {eigvals} exceed max_value=5.0"
        )

    def test_effective_step_magnitude_with_preconditioner(self):
        """Effective step = step_size * precond * gradient. Check if it's bounded."""
        # Typical scenario: x ~ 1000, s_inv ~ 0.001, gradient ~ 100
        x = 1000.0
        s_inv = 0.001
        eps = 1e-8
        gradient = 100.0
        step_size = 1.0

        # BSREM preconditioner
        bsrem_precond = (x + eps) * s_inv
        effective_step = step_size * bsrem_precond * gradient

        # The effective step should be bounded relative to x
        # For BSREM: effective_step = step * (x + eps) * s_inv * grad
        # = 1.0 * 1000 * 0.001 * 100 = 100
        # That's 10% of x - reasonable

        # But if x grows to 10000:
        x_big = 10000.0
        bsrem_precond_big = (x_big + eps) * s_inv
        effective_step_big = step_size * bsrem_precond_big * gradient

        # effective_step_big = 1.0 * 10000 * 0.001 * 100 = 1000
        # That's 10% of x_big - still same proportion
        ratio_small = effective_step / x
        ratio_big = effective_step_big / x_big
        assert abs(ratio_small - ratio_big) < 1e-10, (
            "BSREM preconditioner maintains constant step-to-image ratio"
        )


# ---------------------------------------------------------------------------
# Test 3: Block Hessian conditioning under realistic scenarios
# ---------------------------------------------------------------------------

class TestBlockHessianConditioning:
    """Test eigenvalue conditioning of 2x2 block Hessians."""

    def test_data_hessian_diagonal_dominance(self):
        """Data Hessian should be well-conditioned when PET and SPECT have similar sensitivities."""
        # s_inv in common space
        x = np.array([100.0, 50.0])  # PET=100, SPECT=50
        s_inv = np.array([0.01, 0.02])
        eps = 1e-8

        denom = (x + eps) * s_inv
        diag = 1.0 / np.maximum(denom, 1e-8)

        # Check condition number
        cond = np.max(diag) / np.min(diag)
        assert cond < 1e4, f"Data Hessian condition number {cond:.1f} is very high"

    def test_block_hessian_condition_number_with_mismatched_modalities(self):
        """When PET >> SPECT in magnitude, block Hessian can be ill-conditioned."""
        # PET dynamic range: ~100, SPECT dynamic range: ~1
        x_pet = 100.0
        x_spect = 1.0
        s_inv_pet = 0.01
        s_inv_spect = 0.5

        data_h = np.zeros((2, 2))
        data_h[0, 0] = 1.0 / ((x_pet + 1e-8) * s_inv_pet)  # = 1.0
        data_h[1, 1] = 1.0 / ((x_spect + 1e-8) * s_inv_spect)  # = 2.0

        # Prior Hessian with strong cross-modal coupling
        prior_h = np.array([[0.1, 0.05], [0.05, 0.1]])

        total_h = data_h + prior_h
        eigvals = np.linalg.eigvalsh(total_h)
        cond = eigvals[-1] / eigvals[0]

        # Document the condition number
        assert cond < 100, f"Total Hessian condition number {cond:.1f}"

    def test_extreme_modality_imbalance_causes_ill_conditioning(self):
        """Extreme imbalance between modalities causes near-singular block Hessian."""
        # PET ~10000, SPECT ~0.1  (dynamic range mismatch ~1e5)
        data_h = np.zeros((2, 2))
        data_h[0, 0] = 0.01  # 1/((10000)*0.01) = 10
        data_h[1, 1] = 200.0  # 1/((0.1)*0.05) = 200

        prior_h = np.array([[0.01, 0.005], [0.005, 0.01]])

        total_h = data_h + prior_h
        eigvals = np.linalg.eigvalsh(total_h)
        cond = eigvals[-1] / eigvals[0]

        # This SHOULD be very large
        assert cond > 10, (
            f"Expected large condition number from modality imbalance, got {cond:.1f}"
        )

        # The inverse eigenvalues will be very different
        inv_eigs = 1.0 / eigvals
        max_inv = np.max(inv_eigs)
        min_inv = np.min(inv_eigs)
        ratio = max_inv / min_inv

        assert ratio > 10, (
            f"Inverse Hessian eigenvalue ratio {ratio:.1f} - "
            "the larger eigenvalue of the preconditioner will amplify one modality's gradient much more"
        )


# ---------------------------------------------------------------------------
# Test 4: shared-space curvature ordering
# ---------------------------------------------------------------------------

class TestSharedSpaceCurvatureOrdering:
    """Shared-grid DTNV should form data curvature in image space after sensitivity pullback."""

    def test_project_then_divide_can_underestimate_curvature(self):
        r = np.array([[0.5, 0.5]], dtype=np.float64)
        x = np.array([100.0, 1.0], dtype=np.float64)
        s_inv = np.array([0.01, 1.0], dtype=np.float64)
        eps = 1e-8

        correct = (r @ (1.0 / ((x + eps) * s_inv))).item()
        incorrect = (1.0 / (((r @ x) + eps) * (r @ s_inv))).item()

        assert correct > 20.0 * incorrect

    def test_shared_space_sensitivity_differs_from_native_sensitivity(self):
        w_t = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        native_sens = np.array([4.0, 1.0], dtype=np.float64)

        pulled = w_t @ native_sens
        assert np.allclose(pulled, np.array([3.4, 1.3]), atol=1e-12, rtol=1e-12)
        assert not np.allclose(pulled, native_sens, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 5: SVRG variance with constant step size
# ---------------------------------------------------------------------------

class TestSVRGStepSize:
    """Test that SVRG + constant step size violates Robbins-Monro conditions."""

    def test_robbins_monro_requires_decay(self):
        """For stochastic gradient methods, convergence requires:
        sum(step_k) = inf  AND  sum(step_k^2) < inf

        With constant step_size = 1.0 and relaxation_eta = 0.0:
        step_k = 1.0 / (1 + 0.0 * k) = 1.0 for all k
        sum(step_k^2) = sum(1.0) = inf  → VIOLATES Robbins-Monro
        """
        N = 1000
        eta = 0.0
        initial_step = 1.0

        steps = [initial_step / (1.0 + eta * k) for k in range(N)]
        sum_steps = sum(steps)
        sum_sq_steps = sum(s**2 for s in steps)

        # With eta=0: both sums diverge
        assert sum_steps == pytest.approx(N * initial_step), "Sum should equal N"
        assert sum_sq_steps == pytest.approx(N * initial_step**2), "Sum of squares should equal N"

        # This means Robbins-Monro is violated
        # For convergence we need sum_sq < inf, but here it grows linearly

    def test_positive_eta_satisfies_robbins_monro(self):
        """With eta > 0, step_k = 1/(1+eta*k) satisfies both conditions."""
        N = 100000
        eta = 0.01
        initial_step = 1.0

        steps = [initial_step / (1.0 + eta * k) for k in range(N)]
        sum_steps = sum(steps)
        sum_sq_steps = sum(s**2 for s in steps)

        # sum(1/(1+eta*k)) ~ (1/eta) * ln(1+eta*N) → inf ✓
        assert sum_steps > 100, f"Sum of steps should be large: {sum_steps}"

        # sum(1/(1+eta*k)^2) ~ (1/eta) * (1 - 1/(1+eta*N)) < 1/eta  (bounded) ✓
        bound = 1.0 / eta
        assert sum_sq_steps < bound * 1.1, (
            f"Sum of squared steps {sum_sq_steps:.1f} should be bounded by ~{bound:.1f}"
        )


# ---------------------------------------------------------------------------
# Test 6: Combined preconditioner + clipping + step size interaction
# ---------------------------------------------------------------------------

class TestCombinedDivergenceMechanisms:
    """Test the interaction of multiple divergence-promoting mechanisms."""

    def test_preconditioned_ista_iteration_bound(self):
        """Simulate preconditioned ISTA steps to check if update is bounded relative to x."""
        rng = np.random.default_rng(42)

        # Initial image (PET, SPECT) at each of 100 voxels
        n_voxels = 100
        x = np.column_stack([
            rng.uniform(10, 1000, n_voxels),   # PET
            rng.uniform(0.1, 10, n_voxels),     # SPECT
        ])

        # Simulate gradient (from subset evaluation)
        gradient = rng.uniform(-50, 50, x.shape)

        # Block preconditioner (diagonal for simplicity)
        s_inv = np.column_stack([
            np.full(n_voxels, 0.01),
            np.full(n_voxels, 0.5),
        ])

        step_size = 1.0

        # BSREM preconditioner
        precond = (x + 1e-8) * s_inv

        # Preconditioned gradient
        precond_grad = precond * gradient

        # Clip by M/step_size (M = max across all containers)
        M = np.max(x)
        clip = M / step_size
        precond_grad_clipped = np.clip(precond_grad, -clip, clip)

        # Update
        x_new = x - step_size * precond_grad_clipped

        # Non-negativity
        x_new = np.maximum(x_new, 0)

        # Check: what is the max relative change?
        delta = np.abs(x_new - x)
        # For SPECT (x ~ 0.1-10), the max change could be huge relative to x
        spect_rel_change = delta[:, 1] / np.maximum(x[:, 1], 1e-10)
        max_spect_rel_change = np.max(spect_rel_change)

        # This documents the problem: SPECT values can change by orders of magnitude
        # because M is dominated by PET and the clip threshold is PET-scale
        assert max_spect_rel_change > 1.0, (
            f"SPECT relative change {max_spect_rel_change:.1f}x - "
            "the PET-dominated clip threshold allows huge SPECT changes"
        )

    def test_per_modality_clipping_prevents_cross_modal_blowup(self):
        """If we clip per-modality by per-modality max, cross-modal blowup is prevented."""
        rng = np.random.default_rng(42)

        n_voxels = 100
        x_pet = rng.uniform(10, 1000, n_voxels)
        x_spect = rng.uniform(0.1, 10, n_voxels)

        gradient_pet = rng.uniform(-50, 50, n_voxels)
        gradient_spect = rng.uniform(-50, 50, n_voxels)

        step_size = 1.0

        # Per-modality clip
        M_pet = np.max(x_pet)
        M_spect = np.max(x_spect)

        clip_pet = M_pet / step_size
        clip_spect = M_spect / step_size

        grad_clipped_pet = np.clip(gradient_pet, -clip_pet, clip_pet)
        grad_clipped_spect = np.clip(gradient_spect, -clip_spect, clip_spect)

        x_new_pet = x_pet - step_size * grad_clipped_pet
        x_new_spect = x_spect - step_size * grad_clipped_spect

        x_new_pet = np.maximum(x_new_pet, 0)
        x_new_spect = np.maximum(x_new_spect, 0)

        # Now SPECT relative change is bounded by O(1)
        spect_rel_change = np.abs(x_new_spect - x_spect) / np.maximum(x_spect, 1e-10)
        max_spect_rel_change = np.max(spect_rel_change)

        # Still could be large for very small SPECT values, but much more controlled
        # At minimum, the update magnitude is bounded by M_spect (not M_pet)
        max_spect_update = np.max(np.abs(x_new_spect - x_spect))
        assert max_spect_update <= M_spect + 1e-10, (
            f"Per-modality clipping bounds SPECT update by M_spect={M_spect:.1f}, "
            f"got {max_spect_update:.1f}"
        )


# ---------------------------------------------------------------------------
# Test 7: MajorisingHessianBlockPreconditioner data Hessian in common space
# ---------------------------------------------------------------------------

class TestDataHessianInCommonSpace:
    """The MajorisingHessianBlockPreconditioner computes data Hessian in common space
    by projecting both x and s_inv through the block operator.

    This means the data Hessian diagonal is computed as:
        diag_m = 1 / ((B_m x + eps) * B_m s_inv)

    for each modality m. The block operator B maps from image space to
    per-modality space. But if B includes resampling, the projected s_inv
    may not correctly represent the inverse of the projected sensitivity.
    """

    def test_data_hessian_common_space_vs_image_space(self):
        """Compare data Hessian computed in common space vs image space."""
        # Image space: 2 voxels
        # Common space: 2 voxels per modality (identity mapping)
        x_img = np.array([100.0, 50.0])  # Image (PET-like)
        s_inv_img = np.array([0.01, 0.02])

        # Common space (via identity operator): same values
        x_common = x_img.copy()
        s_inv_common = s_inv_img.copy()

        eps = 1e-8

        # Image space data Hessian: 1/((x+eps)*s_inv)
        h_image = 1.0 / ((x_img + eps) * s_inv_img)

        # Common space data Hessian (as computed by the preconditioner)
        h_common = 1.0 / np.maximum((x_common + eps) * s_inv_common, 1e-8)

        np.testing.assert_allclose(h_image, h_common, rtol=1e-10)

    def test_data_hessian_with_resampling_operator(self):
        """When B is a resampling operator, B(s_inv) != 1/B(sensitivity).

        sensitivity = 1/s_inv, but B(1/s_inv) != 1/B(s_inv) in general
        (Jensen's inequality for convex 1/x).
        """
        # Simple 2-to-2 resampling (averaging)
        B = np.array([[0.6, 0.4], [0.3, 0.7]])

        s_inv = np.array([0.01, 0.04])  # Inverse sensitivities
        sensitivity = 1.0 / s_inv  # [100, 25]

        # What the code does: project s_inv through B
        s_inv_projected = B @ s_inv  # [0.022, 0.031]

        # What should happen: project sensitivity through B, then invert
        sens_projected = B @ sensitivity  # [70, 47.5]
        s_inv_correct = 1.0 / sens_projected  # [0.0143, 0.0211]

        # These are NOT equal
        assert not np.allclose(s_inv_projected, s_inv_correct, rtol=0.01), (
            "B(s_inv) should differ from 1/B(sensitivity) - "
            "this is a potential source of error in the data Hessian computation"
        )

        # Check which is larger: s_inv_projected vs s_inv_correct
        # By Jensen's inequality (1/x is convex): B(1/x) >= 1/B(x)
        # So s_inv_projected >= s_inv_correct
        assert np.all(s_inv_projected >= s_inv_correct - 1e-10), (
            "Jensen's inequality: B(s_inv) >= 1/B(sensitivity) for convex 1/x"
        )

        # This means data_hessian_common = 1/((x+eps) * B(s_inv))
        # is SMALLER than the correct 1/((x+eps) * s_inv_correct)
        # → data Hessian is UNDERESTIMATED → total Hessian is underestimated
        # → preconditioner (inverse) is OVERESTIMATED → steps too large!
        x = np.array([100.0, 50.0])
        eps = 1e-8

        h_code = 1.0 / ((x + eps) * s_inv_projected)
        h_correct = 1.0 / ((x + eps) * s_inv_correct)

        assert np.all(h_code <= h_correct + 1e-10), (
            "Data Hessian computed with projected s_inv underestimates "
            "the correct Hessian → preconditioner overshoots"
        )
