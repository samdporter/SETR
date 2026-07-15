import logging
from typing import Optional, Sequence

import numpy as np
from cil.framework import BlockDataContainer, DataContainer
from cil.optimisation.functions import ScaledFunction
from cil.optimisation.utilities import Preconditioner
from sirf.STIR import SeparableGaussianImageFilter

from recon_core.cil_extensions.functions import ensure_kl_hessian_support
from recon_core.utils.sirf import get_array

class ConstantPreconditioner(Preconditioner):
    """Constant preconditioner."""

    def __init__(self, value):
        self.value = value

    def apply(self, algorithm, gradient, out=None):
        if out is None:
            return gradient * self.value
        gradient.multiply(self.value, out=out)
        return out


class PreconditionerWithInterval(Preconditioner):
    """Base preconditioner class with update intervals and parameter freezing.

    This abstract class extends CIL's Preconditioner to support:
    - Periodic updates at specified intervals
    - Parameter freezing after a given iteration count
    - Caching of computed preconditioners for efficiency

    Args:
        update_interval: Number of iterations between preconditioner updates.
        freeze_iter: Iteration number after which to freeze preconditioner parameters.
            Set to np.inf to never freeze.
    """

    def __init__(self, update_interval=1, freeze_iter=np.inf):
        self.update_interval = update_interval
        self.freeze_iter = freeze_iter
        self.freeze = None
        self.precond = None

    def apply(self, algorithm, gradient, out=None):
        """
        Apply the preconditioner, managing freezing and update intervals.
        """
        if algorithm.iteration < self.freeze_iter:
            if algorithm.iteration % self.update_interval == 0 or self.precond is None:
                self.precond = self.compute_preconditioner(algorithm)
            if out is not None:
                gradient.multiply(self.precond, out=out)
                return out
            return gradient * self.precond
        else:
            if self.freeze is None:
                self.freeze = self.compute_preconditioner(algorithm)
            if out is not None:
                gradient.multiply(self.freeze, out=out)
                return out
        return gradient * self.freeze

    def compute_preconditioner(self, algorithm, out=None):
        """Compute the preconditioner."""
        raise NotImplementedError


class BSREMPreconditioner(PreconditionerWithInterval):
    """Preconditioner for BSREM."""

    def __init__(
        self, s_inv, update_interval=1, freeze_iter=np.inf, epsilon=None, smooth=False, max_val=None, smoothing_fwhm=(10,10,10)
    ):
        super().__init__(update_interval, freeze_iter)
        self.s_inv = s_inv
        if smooth:
            self.gaussian = SeparableGaussianImageFilter()
            self.gaussian.set_fwhms(smoothing_fwhm)
        else:
            self.gaussian = None
        if epsilon is None:
            epsilon = s_inv.max() * 1e-10
        self.epsilon = epsilon
        self.max_val = max_val

    def compute_preconditioner(self, algorithm, out=None):
        x = algorithm.solution.copy()

        if self.max_val is not None:
            x = x.minimum(self.max_val)

        if isinstance(x, BlockDataContainer):
            for i, xi in enumerate(x.containers):
                if self.gaussian is not None:
                    self.gaussian.apply(xi)
                x.containers[i].fill(xi)
        elif self.gaussian is not None:
            self.gaussian.apply(x)

        if out is None:
            return (x + self.epsilon) * self.s_inv
        x.add(self.epsilon, out=out)
        self.s_inv.multiply(out, out=out)
        return out


class ImageFunctionPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner for the prior, using the inverse Hessian diagonal.
    """

    def __init__(
        self,
        function,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=0,
        max_value=np.inf,
    ):
        super().__init__(update_interval, freeze_iter)
        self.function = function
        self.epsilon = epsilon
        self.max_value = max_value

    def compute_preconditioner(self, algorithm, out=None):
        precond = self.function(algorithm.solution)
        precond = precond.maximum(self.epsilon)
        precond = precond.minimum(self.max_value)

        if out is None:
            return precond
        out.fill(precond)
        return out


class BlockDiagonalPriorPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner that applies a voxel-wise 2x2 inverse-Hessian block in common space.

    This applies a direct prior's voxel-wise 2x2 inverse-Hessian block in shared space.
    A scalar/diagonal preconditioner (e.g. BSREM) can optionally be applied first.
    """

    def __init__(
        self,
        prior,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=1e-8,
        max_value=np.inf,
        base_preconditioner: Optional[Preconditioner] = None,
    ):
        super().__init__(update_interval, freeze_iter)
        self.prior = prior
        self.epsilon = float(epsilon)
        self.max_value = float(max_value)
        self.base_preconditioner = base_preconditioner

    def _compute_block_preconditioner(self, image: DataContainer) -> np.ndarray:
        if not hasattr(self.prior, "inv_preconditioner_block"):
            raise AttributeError(
                f"Prior {type(self.prior)} does not expose inv_preconditioner_block()."
            )

        block = self.prior.inv_preconditioner_block(image, epsilon=self.epsilon)
        block_arr = _to_numpy_array(block)
        if block_arr.shape[-2:] != (2, 2):
            raise ValueError(
                f"Expected block preconditioner shape (..., 2, 2), got {block_arr.shape}."
            )

        block_arr = _symmetrise_blocks(block_arr.astype(np.float64, copy=False))
        eigvals, eigvecs = np.linalg.eigh(block_arr)
        np.maximum(eigvals, self.epsilon, out=eigvals)
        if np.isfinite(self.max_value):
            np.minimum(eigvals, self.max_value, out=eigvals)
        block_arr = (eigvecs * eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
        return _symmetrise_blocks(block_arr)

    def _apply_block_preconditioner(
        self,
        gradient: DataContainer,
        block_arr: np.ndarray,
        out: Optional[DataContainer] = None,
    ):
        grad_arr = _stack_block_container(gradient)
        if grad_arr.shape[-1] != 2:
            raise ValueError(
                f"Block preconditioner requires 2 modalities, got {grad_arr.shape[-1]}."
            )
        if block_arr.shape[:-2] != grad_arr.shape[:-1]:
            raise ValueError(
                f"Gradient/block shape mismatch: gradient {grad_arr.shape}, block {block_arr.shape}."
        )

        precond_common_arr = np.einsum("...ij,...j->...i", block_arr, grad_arr, optimize=True)
        ret = _fill_block_container_from_array(gradient, precond_common_arr)
        if out is None:
            return ret
        out.fill(ret)
        return out

    def compute_preconditioner(self, algorithm, out=None):
        block_arr = self._compute_block_preconditioner(algorithm.solution)
        # Block preconditioners are plain arrays; `out` is ignored intentionally.
        return block_arr

    def apply(self, algorithm, gradient, out=None):
        if algorithm.iteration < self.freeze_iter:
            if algorithm.iteration % self.update_interval == 0 or self.precond is None:
                self.precond = self.compute_preconditioner(algorithm)
            current_block = self.precond
        else:
            if self.freeze is None:
                self.freeze = self.compute_preconditioner(algorithm)
            current_block = self.freeze

        g = (
            self.base_preconditioner.apply(algorithm, gradient)
            if self.base_preconditioner is not None
            else gradient
        )
        return self._apply_block_preconditioner(g, current_block, out=out)


class BlockLehmerMeanPreconditioner(PreconditionerWithInterval):
    """
    Matrix Lehmer mean of two voxel-wise SPD preconditioners.

    For two SPD matrices B and D, define C = D^{-1/2} B D^{-1/2} and

        L_p(B, D) = D^{1/2} f_p(C) D^{1/2},
        f_p(t) = (t^p + 1) / (t^(p-1) + 1).

    This congruence formulation is symmetric and positive definite even when B and D
    do not commute.  It agrees with the scalar Lehmer mean on commuting matrices and
    has the important limits

        L_0(B, D) = 2 (B^{-1} + D^{-1})^{-1},
        L_1(B, D) = (B + D) / 2.

    Thus ``output_scale=0.5`` at ``p=0`` is exactly the parallel sum.  Production
    callers should provide ``hessian_preconditioner`` so both this class and the
    parallel-sum majoriser use identical prior/data Hessian components.  The legacy
    preconditioner operands remain supported for compatibility.
    """

    def __init__(
        self,
        block_preconditioner: Optional[BlockDiagonalPriorPreconditioner] = None,
        scalar_preconditioner: Optional[Preconditioner] = None,
        p: float = 1e-1,
        epsilon: float = 1e-12,
        max_value: float = np.inf,
        update_interval=1,
        freeze_iter=np.inf,
        scalar_reduction: str = "diag",
        output_scale: float = 1.0,
        hessian_preconditioner=None,
    ):
        super().__init__(update_interval, freeze_iter)
        self.block_preconditioner = block_preconditioner
        self.scalar_preconditioner = scalar_preconditioner
        self.hessian_preconditioner = hessian_preconditioner
        self.p = float(p)
        self.epsilon = float(epsilon)
        self.max_value = float(max_value)
        self.scalar_reduction = scalar_reduction
        self.output_scale = float(output_scale)
        if scalar_reduction not in {"mean", "geometric", "diag"}:
            raise ValueError("scalar_reduction must be one of {'mean', 'geometric', 'diag'}.")
        if not np.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("output_scale must be finite and > 0.")
        if self.epsilon <= 0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite and > 0.")
        if self.hessian_preconditioner is None and (
            self.block_preconditioner is None or self.scalar_preconditioner is None
        ):
            raise ValueError(
                "Provide hessian_preconditioner or both block_preconditioner and "
                "scalar_preconditioner."
            )

    @staticmethod
    def _reconstruct_from_eigendecomposition(eigvals, eigvecs):
        return (eigvecs * eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)

    def _project_spd(self, blocks: np.ndarray, floor: Optional[float] = None) -> np.ndarray:
        blocks = _symmetrise_blocks(np.asarray(blocks, dtype=np.float64))
        eigvals, eigvecs = np.linalg.eigh(blocks)
        np.maximum(eigvals, self.epsilon if floor is None else float(floor), out=eigvals)
        projected = self._reconstruct_from_eigendecomposition(eigvals, eigvecs)
        return _symmetrise_blocks(projected)

    def _invert_spd(self, blocks: np.ndarray, floor: Optional[float] = None) -> np.ndarray:
        blocks = _symmetrise_blocks(np.asarray(blocks, dtype=np.float64))
        eigvals, eigvecs = np.linalg.eigh(blocks)
        np.maximum(eigvals, self.epsilon if floor is None else float(floor), out=eigvals)
        inverse = self._reconstruct_from_eigendecomposition(1.0 / eigvals, eigvecs)
        return _symmetrise_blocks(inverse)

    def _finalise_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """Apply Lehmer scaling, final cap, then the provider safety scale."""
        blocks = self.output_scale * _symmetrise_blocks(blocks)
        # ``getattr`` keeps the numerical helper usable in lightweight unit tests
        # that construct an instance without running the dependency-heavy init.
        provider = getattr(self, "hessian_preconditioner", None)
        max_value = self.max_value if provider is None else float(provider.max_value)
        if np.isfinite(max_value):
            eigvals, eigvecs = np.linalg.eigh(blocks)
            np.minimum(eigvals, max_value, out=eigvals)
            blocks = self._reconstruct_from_eigendecomposition(eigvals, eigvecs)
            blocks = _symmetrise_blocks(blocks)
        if provider is not None and provider.safety_scale != 1.0:
            blocks = blocks * provider.safety_scale
        return blocks

    def _blend_blocks(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the scaled two-operand matrix Lehmer mean of SPD block fields."""
        if left.shape != right.shape or left.shape[-2:] != (2, 2):
            raise ValueError(
                f"Lehmer block operands must have matching (..., 2, 2) shapes, "
                f"got {left.shape} and {right.shape}."
            )

        left = self._project_spd(left)
        right = self._project_spd(right)

        if self.p == 0.0:
            # Standard two-operand L_0 is the harmonic mean: twice parallel sum.
            inverse_sum = self._invert_spd(left) + self._invert_spd(right)
            raw_mean = 2.0 * self._invert_spd(inverse_sum)
            return self._finalise_blocks(raw_mean)

        d_eigvals, d_eigvecs = np.linalg.eigh(right)
        np.maximum(d_eigvals, self.epsilon, out=d_eigvals)
        d_sqrt = self._reconstruct_from_eigendecomposition(np.sqrt(d_eigvals), d_eigvecs)
        d_inv_sqrt = self._reconstruct_from_eigendecomposition(
            1.0 / np.sqrt(d_eigvals), d_eigvecs
        )

        relative = _symmetrise_blocks(d_inv_sqrt @ left @ d_inv_sqrt)
        rel_eigvals, rel_eigvecs = np.linalg.eigh(relative)
        # The relative eigenvalues are dimensionless.  Use machine tiny rather than
        # the absolute preconditioner floor so high modality contrasts are retained.
        np.maximum(rel_eigvals, np.finfo(np.float64).tiny, out=rel_eigvals)
        log_rel = np.log(rel_eigvals)
        log_mean_eigvals = np.logaddexp(self.p * log_rel, 0.0) - np.logaddexp(
            (self.p - 1.0) * log_rel, 0.0
        )
        mean_relative = self._reconstruct_from_eigendecomposition(
            np.exp(log_mean_eigvals), rel_eigvecs
        )
        raw_mean = d_sqrt @ mean_relative @ d_sqrt
        return self._finalise_blocks(_symmetrise_blocks(raw_mean))

    def _compute_scalar_field(self, algorithm) -> np.ndarray:
        scalar_precond = self.scalar_preconditioner.compute_preconditioner(algorithm)
        if isinstance(scalar_precond, np.ndarray):
            scalar = scalar_precond
        elif isinstance(scalar_precond, (DataContainer, BlockDataContainer)):
            scalar_stack = _stack_block_container(scalar_precond).astype(np.float64, copy=False)
            if scalar_stack.shape[-1] != 2:
                raise ValueError(
                    f"Block Lehmer blend requires 2 modalities, got {scalar_stack.shape[-1]}."
                )
            if self.scalar_reduction == "diag":
                scalar = np.zeros((*scalar_stack.shape[:-1], 2, 2), dtype=np.float64)
                scalar[..., 0, 0] = scalar_stack[..., 0]
                scalar[..., 1, 1] = scalar_stack[..., 1]
            elif self.scalar_reduction == "mean":
                scalar = 0.5 * (scalar_stack[..., 0] + scalar_stack[..., 1])
            else:
                scalar = np.sqrt(
                    np.maximum(scalar_stack[..., 0], self.epsilon)
                    * np.maximum(scalar_stack[..., 1], self.epsilon)
                )
        else:
            raise TypeError(
                f"Unsupported scalar preconditioner output type {type(scalar_precond)}."
            )
        if scalar.ndim >= 2 and scalar.shape[-2:] == (2, 2):
            scalar = self._project_spd(scalar)
        else:
            scalar = np.maximum(scalar, self.epsilon)
        return scalar

    def _blend_block_and_scalar(self, block_arr: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        block_arr = np.asarray(block_arr, dtype=np.float64)
        if scalar.ndim == block_arr.ndim:
            data_blocks = np.asarray(scalar, dtype=np.float64)
        else:
            lam = np.maximum(np.asarray(scalar, dtype=np.float64), self.epsilon)
            data_blocks = np.zeros_like(block_arr, dtype=np.float64)
            data_blocks[..., 0, 0] = lam
            data_blocks[..., 1, 1] = lam
        return self._blend_blocks(block_arr, data_blocks)

    def _compute_from_hessian_provider(self, algorithm) -> np.ndarray:
        provider = self.hessian_preconditioner
        image = algorithm.solution

        # This is the exact equality contract requested by the experiment: use the
        # identical summed-Hessian inversion, cap, and safety scale as the majoriser.
        if self.p == 0.0 and self.output_scale == 0.5:
            return provider.compute_preconditioner(algorithm)

        prior_h = provider._compute_prior_hessian_block(image)
        data_h = provider._compute_data_hessian_block(image)
        floor = provider.hessian_floor
        prior_precond = self._invert_spd(prior_h, floor=floor)
        data_precond = self._invert_spd(data_h, floor=floor)
        return self._blend_blocks(prior_precond, data_precond)

    def compute_preconditioner(self, algorithm, out=None):
        if self.hessian_preconditioner is not None:
            return self._compute_from_hessian_provider(algorithm)

        block_arr = self.block_preconditioner._compute_block_preconditioner(algorithm.solution)
        scalar = self._compute_scalar_field(algorithm)
        if scalar.ndim == block_arr.ndim:
            if scalar.shape != block_arr.shape:
                raise ValueError(
                    "Scalar/block geometry mismatch: "
                    f"scalar {scalar.shape}, block {block_arr.shape}."
                )
        elif scalar.shape != block_arr.shape[:-2]:
            raise ValueError(
                f"Scalar/block geometry mismatch: scalar {scalar.shape}, block {block_arr.shape}."
            )
        return self._blend_block_and_scalar(block_arr, scalar)

    def apply(self, algorithm, gradient, out=None):
        if algorithm.iteration < self.freeze_iter:
            if algorithm.iteration % self.update_interval == 0 or self.precond is None:
                self.precond = self.compute_preconditioner(algorithm)
            current_block = self.precond
        else:
            if self.freeze is None:
                self.freeze = self.compute_preconditioner(algorithm)
            current_block = self.freeze
        applier = self.block_preconditioner or self.hessian_preconditioner
        return applier._apply_block_preconditioner(gradient, current_block, out=out)


class MajorisingHessianDiagonalPreconditioner(PreconditionerWithInterval):
    """Diagonal preconditioner from inverse of summed data/prior Hessian surrogates."""

    def __init__(
        self,
        s_inv,
        prior,
        update_interval=1,
        freeze_iter=np.inf,
        x_epsilon: float = 1e-8,
        hessian_floor: float = 1e-8,
        max_value: float = np.inf,
        safety_scale: float = 1.0,
    ):
        super().__init__(update_interval, freeze_iter)
        self.s_inv = s_inv
        self.prior = prior
        self.x_epsilon = float(x_epsilon)
        self.hessian_floor = float(hessian_floor)
        self.max_value = float(max_value)
        self.safety_scale = float(safety_scale)
        if self.safety_scale <= 0:
            raise ValueError("safety_scale must be > 0.")

    def _evaluate_prior_diag(self, prior_obj, image):
        if hasattr(prior_obj, "preconditioner_diag"):
            fn = getattr(prior_obj, "preconditioner_diag")
        elif hasattr(prior_obj, "hessian_diag"):
            fn = getattr(prior_obj, "hessian_diag")
        else:
            return None

        try:
            diag = fn(image, epsilon=self.hessian_floor)
        except TypeError as exc:
            if "epsilon" not in str(exc):
                raise
            diag = fn(image)
        return diag.abs()

    def _apply_safety_scale(self, diagonal):
        if self.safety_scale == 1.0:
            return diagonal
        if isinstance(diagonal, BlockDataContainer):
            for con in diagonal.containers:
                arr = get_array(con).astype(np.float64, copy=False)
                arr *= self.safety_scale
                con.fill(arr)
            return diagonal
        arr = _as_array(diagonal).astype(np.float64, copy=False)
        arr *= self.safety_scale
        return _fill_from_array(diagonal, arr)

    def _invert_diagonal_container(self, diagonal, clamp_max: bool):
        if isinstance(diagonal, BlockDataContainer):
            for con in diagonal.containers:
                arr = get_array(con).astype(np.float64, copy=False)
                np.maximum(arr, self.hessian_floor, out=arr)
                np.reciprocal(arr, out=arr)
                if clamp_max and np.isfinite(self.max_value):
                    np.minimum(arr, self.max_value, out=arr)
                con.fill(arr)
            return diagonal

        arr = _as_array(diagonal).astype(np.float64, copy=False)
        np.maximum(arr, self.hessian_floor, out=arr)
        np.reciprocal(arr, out=arr)
        if clamp_max and np.isfinite(self.max_value):
            np.minimum(arr, self.max_value, out=arr)
        return _fill_from_array(diagonal, arr)

    def _compute_data_hessian_diag(self, image: DataContainer) -> DataContainer:
        # EM-type Hessian surrogate: H_data ≈ sensitivity / (x + eps) = 1 / ((x + eps) * s_inv).
        denom = image.copy()
        denom = denom + self.x_epsilon
        denom.multiply(self.s_inv, out=denom)
        return self._invert_diagonal_container(denom, clamp_max=False)

    def _compute_prior_hessian_diag(self, image: DataContainer) -> DataContainer:
        prior_h = self._evaluate_prior_diag(self.prior, image)
        if prior_h is not None:
            return prior_h

        raise AttributeError(
            f"Prior {type(self.prior)} does not expose preconditioner_diag() or hessian_diag()."
        )

    def compute_preconditioner(self, algorithm, out=None):
        image = algorithm.solution
        data_h = self._compute_data_hessian_diag(image)
        prior_h = self._compute_prior_hessian_diag(image)
        total_h = data_h + prior_h
        precond = self._invert_diagonal_container(total_h, clamp_max=True)
        precond = self._apply_safety_scale(precond)

        if out is None:
            return precond
        out.fill(precond)
        return out


class MajorisingHessianBlockPreconditioner(PreconditionerWithInterval):
    """Block preconditioner from inverse of summed data/prior 2x2 Hessian surrogates."""

    def __init__(
        self,
        s_inv: BlockDataContainer,
        prior,
        update_interval=1,
        freeze_iter=np.inf,
        x_epsilon: float = 1e-8,
        hessian_floor: float = 1e-8,
        max_value: float = np.inf,
        safety_scale: float = 1.0,
    ):
        super().__init__(update_interval, freeze_iter)
        self.s_inv = s_inv
        self.prior = prior
        self.x_epsilon = float(x_epsilon)
        self.hessian_floor = float(hessian_floor)
        self.max_value = float(max_value)
        self.safety_scale = float(safety_scale)
        if self.safety_scale <= 0:
            raise ValueError("safety_scale must be > 0.")

    def _compute_prior_hessian_block(self, image: DataContainer) -> np.ndarray:
        if hasattr(self.prior, "preconditioner_block"):
            block = self.prior.preconditioner_block(image, epsilon=self.hessian_floor)
        elif hasattr(self.prior, "hessian_block_diag"):
            block = self.prior.hessian_block_diag(image, epsilon=self.hessian_floor)
        else:
            raise AttributeError(
                f"Prior {type(self.prior)} does not expose block Hessian helpers."
            )

        block_arr = _to_numpy_array(block).astype(np.float64, copy=False)
        if block_arr.shape[-2:] != (2, 2):
            raise ValueError(
                f"Expected prior block Hessian shape (..., 2, 2), got {block_arr.shape}."
            )
        return _symmetrise_blocks(block_arr)

    def _compute_data_hessian_block(self, image: DataContainer) -> np.ndarray:
        if not isinstance(image, BlockDataContainer):
            raise TypeError(
                f"Expected BlockDataContainer image, got {type(image)}."
            )
        if not isinstance(self.s_inv, BlockDataContainer):
            raise TypeError(
                f"Expected BlockDataContainer s_inv, got {type(self.s_inv)}."
            )
        if len(image.containers) != 2:
            raise ValueError(
                f"Block preconditioner requires exactly 2 modalities, got {len(image.containers)}."
            )
        if len(self.s_inv.containers) != len(image.containers):
            raise ValueError(
                f"s_inv/image modality mismatch: x has {len(image.containers)} containers, "
                f"s_inv has {len(self.s_inv.containers)}."
            )

        data_h = np.zeros((*get_array(image.containers[0]).shape, 2, 2), dtype=np.float64)
        for idx, (x_con, s_inv_con) in enumerate(zip(image.containers, self.s_inv.containers)):
            x_arr = get_array(x_con).astype(np.float64, copy=False)
            s_inv_arr = get_array(s_inv_con).astype(np.float64, copy=False)
            if x_arr.shape != s_inv_arr.shape:
                raise ValueError(
                    f"s_inv/image geometry mismatch for modality {idx}: "
                    f"x {x_arr.shape}, s_inv {s_inv_arr.shape}."
                )
            denom = (x_arr + self.x_epsilon) * s_inv_arr
            np.maximum(denom, self.hessian_floor, out=denom)
            data_h[..., idx, idx] = np.reciprocal(denom)
        return data_h

    def _invert_block_hessian(self, hessian_block: np.ndarray) -> np.ndarray:
        hessian_block = _symmetrise_blocks(hessian_block.astype(np.float64, copy=False))
        eigvals, eigvecs = np.linalg.eigh(hessian_block)
        np.maximum(eigvals, self.hessian_floor, out=eigvals)
        inv_eigs = 1.0 / eigvals
        if np.isfinite(self.max_value):
            np.minimum(inv_eigs, self.max_value, out=inv_eigs)
        inv_block = (eigvecs * inv_eigs[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
        return _symmetrise_blocks(inv_block)

    def _apply_block_preconditioner(
        self,
        gradient: DataContainer,
        block_arr: np.ndarray,
        out: Optional[DataContainer] = None,
    ):
        grad_arr = _stack_block_container(gradient)
        if grad_arr.shape[-1] != 2:
            raise ValueError(
                f"Block preconditioner requires 2 modalities, got {grad_arr.shape[-1]}."
            )
        if block_arr.shape[:-2] != grad_arr.shape[:-1]:
            raise ValueError(
                f"Gradient/block shape mismatch: gradient {grad_arr.shape}, block {block_arr.shape}."
        )

        precond_common_arr = np.einsum("...ij,...j->...i", block_arr, grad_arr, optimize=True)
        ret = _fill_block_container_from_array(gradient, precond_common_arr)
        if out is None:
            return ret
        out.fill(ret)
        return out

    def compute_preconditioner(self, algorithm, out=None):
        image = algorithm.solution
        prior_h = self._compute_prior_hessian_block(image)
        data_h = self._compute_data_hessian_block(image)
        total_h = prior_h + data_h
        precond = self._invert_block_hessian(total_h)
        if self.safety_scale != 1.0:
            precond = precond * self.safety_scale
        return precond

    def apply(self, algorithm, gradient, out=None):
        if algorithm.iteration < self.freeze_iter:
            if algorithm.iteration % self.update_interval == 0 or self.precond is None:
                self.precond = self.compute_preconditioner(algorithm)
            current_block = self.precond
        else:
            if self.freeze is None:
                self.freeze = self.compute_preconditioner(algorithm)
            current_block = self.freeze
        return self._apply_block_preconditioner(gradient, current_block, out=out)


def _unwrap_function(obj):
    """Drill through wrappers (e.g. ScaledFunction, OperatorCompositionFunction)."""
    seen = set()
    current = obj
    while hasattr(current, "function") and id(current) not in seen:
        seen.add(id(current))
        current = current.function
    return current


def _extract_data_passes_indices(obj):
    """Return data_passes_indices from possibly wrapped objective functions."""
    stack = [obj]
    seen = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        indices = getattr(current, "data_passes_indices", None)
        if indices:
            return indices

        for attr in ("function", "functions"):
            child = getattr(current, attr, None)
            if child is None:
                continue
            if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                stack.extend(child)
            else:
                stack.append(child)
    return None


def _as_array(dc: DataContainer) -> np.ndarray:
    """Return a NumPy view of a CIL/SIRF DataContainer."""
    return get_array(dc)


def _fill_from_array(dc: DataContainer, arr: np.ndarray) -> DataContainer:
    """Fill a DataContainer with array values and return it."""
    dc.fill(arr)
    return dc


def _to_numpy_array(arr_like) -> np.ndarray:
    """Convert tensor/array-like objects to a NumPy array."""
    if hasattr(arr_like, "detach"):
        return arr_like.detach().cpu().numpy()
    return np.asarray(arr_like)


def _stack_block_container(dc: BlockDataContainer) -> np.ndarray:
    """Stack BlockDataContainer modalities along the last axis."""
    if not isinstance(dc, BlockDataContainer):
        raise TypeError(
            f"Expected BlockDataContainer, got {type(dc)}. "
            "Block preconditioning currently supports two-modality block gradients only."
        )
    return np.stack([get_array(c) for c in dc.containers], axis=-1)


def _fill_block_container_from_array(template: BlockDataContainer, arr: np.ndarray) -> BlockDataContainer:
    """Create a BlockDataContainer like template and fill each modality from arr[..., m]."""
    if arr.shape[-1] != len(template.containers):
        raise ValueError(
            f"Array/modalities mismatch: arr has last dim={arr.shape[-1]}, "
            f"template has {len(template.containers)} containers."
        )
    out = template.copy()
    for m, con in enumerate(out.containers):
        con.fill(arr[..., m])
    return out


def _symmetrise_blocks(blocks: np.ndarray) -> np.ndarray:
    """Return 0.5*(B + B^T) for trailing 2x2 block matrices."""
    return 0.5 * (blocks + np.swapaxes(blocks, -1, -2))


def _accumulate_poisson_hessian_diag(objectives: Sequence, image: DataContainer) -> DataContainer:
    """
    Sum Hessian diagonal contributions across objectives, supporting composed operators.
    Uses the approximation :math:`diag(A^T D[1/\\bar{y}(x)] A 1)` when the objective exposes
    ``_mu_raw_and_mu_eff`` (ResidualAwareKullbackLeibler), falling back to Hessian products
    otherwise.
    """
    ensure_kl_hessian_support()
    diag = image.get_uniform_copy(0)
    ones_image = image.get_uniform_copy(1)
    mu_floor = 1e-12

    for obj in objectives:
        base = _unwrap_function(obj)
        if not hasattr(base, "multiply_with_Hessian"):
            raise AttributeError(
                f"Objective {type(base)} does not expose multiply_with_Hessian()."
            )

        operator = getattr(obj, "operator", None)
        operator = getattr(obj, "operator", None)

        if operator is not None and hasattr(operator, "direct") and hasattr(operator, "adjoint"):
            projected = operator.direct(image)
            mu_eff_arr = None
            if hasattr(base, "_mu_raw_and_mu_eff"):
                _, mu_eff = base._mu_raw_and_mu_eff(projected)
                mu_eff_arr = np.array(mu_eff, copy=False)

            if mu_eff_arr is not None:
                mu_eff_arr = np.maximum(mu_eff_arr, mu_floor)
                inv_mu_arr = 1.0 / mu_eff_arr
                ones_proj = operator.direct(ones_image)
                weights = ones_proj.copy()
                weights_arr = get_array(ones_proj).astype(np.float64, copy=True)
                weights_arr *= inv_mu_arr
                weights.fill(weights_arr)
                contrib = operator.adjoint(weights)
            else:
                ones_proj = projected.get_uniform_copy(1)
                weights = base.multiply_with_Hessian(projected, ones_proj)
                contrib = operator.adjoint(weights)
        else:
            ones = image.get_uniform_copy(1)
            contrib = base.multiply_with_Hessian(image, ones)

        contrib = contrib.abs()
        diag += contrib

    return diag.abs()


class PoissonHessianPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner based on the inverse diagonal of the Poisson log-likelihood Hessian.

    This class evaluates ``H(x) = Σ_i H_i(x)`` where each objective exposes a
    ``multiply_with_Hessian`` method (as provided by SIRF's Poisson log-likelihoods),
    and returns ``H(x)^{-1}`` with configurable flooring/clamping.
    """

    def __init__(
        self,
        objectives: Sequence,
        update_interval: int = 1,
        freeze_iter: float = np.inf,
        epsilon: float = 0.0,
        max_value: float = np.inf,
        hessian_floor: float = 1e-12,
    ):
        if not isinstance(objectives, Sequence) or len(objectives) == 0:
            raise ValueError("PoissonHessianPreconditioner requires a non-empty sequence of objectives.")
        super().__init__(update_interval, freeze_iter)
        self.objectives = list(objectives)
        self.epsilon = float(epsilon)
        self.max_value = float(max_value)
        self.hessian_floor = float(hessian_floor)

    def _compute_hessian_diag(self, image: DataContainer) -> DataContainer:
        return _accumulate_poisson_hessian_diag(self.objectives, image)

    def _finalise(self, diag: DataContainer) -> DataContainer:
        arr = _as_array(diag)
        np.maximum(arr, self.hessian_floor, out=arr)
        np.reciprocal(arr, out=arr)
        if np.isfinite(self.max_value):
            np.minimum(arr, self.max_value, out=arr)
        if self.epsilon > 0:
            np.maximum(arr, self.epsilon, out=arr)
        return _fill_from_array(diag, arr)

    def compute_preconditioner(self, algorithm, out=None):
        diag = self._compute_hessian_diag(algorithm.solution)
        precond = self._finalise(diag)
        if out is None:
            return precond
        out.fill(precond)
        return out


class SubsetPoissonHessianPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner that updates from subset Poisson Hessian diagonals.

    Parameters
    ----------
    objectives:
        Sequence of subset-specific objective functions exposing ``multiply_with_Hessian``.
    update_interval:
        How often the preconditioner is recomputed (default every iteration).
    freeze_iter:
        Iteration after which the preconditioner is frozen.
    epsilon:
        Minimum value for the returned preconditioner.
    max_value:
        Maximum value for the returned preconditioner (np.inf disables clamping).
    hessian_floor:
        Floor applied to Hessian diagonals before inversion to avoid division by zero.
    mode:
        Aggregation strategy across subsets. One of ``{"none","sum","mean","ema"}``.
    ema_decay:
        Exponential moving-average decay factor (only used when mode == "ema").
    reset_interval:
        If provided, the accumulator resets after this many subset contributions.
        For deterministic ordered subsets, set to the number of subsets to keep a
        per-epoch aggregate.
    weights:
        Optional per-subset scale factors.
    scale_to_full:
        When True and mode is "none" or "mean", rescale by the number of subsets to
        recover a full-data Hessian estimate.
    """

    def __init__(
        self,
        objectives: Sequence,
        update_interval: int = 1,
        freeze_iter: float = np.inf,
        epsilon: float = 0.0,
        max_value: float = np.inf,
        hessian_floor: float = 1e-12,
        mode: str = "sum",
        ema_decay: float = 0.9,
        reset_interval: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        scale_to_full: bool = True,
    ):
        if not isinstance(objectives, Sequence) or len(objectives) == 0:
            raise ValueError("SubsetPoissonHessianPreconditioner requires at least one objective.")
        super().__init__(update_interval, freeze_iter)
        self.objectives = list(objectives)
        self.num_subsets = len(self.objectives)
        self.epsilon = float(epsilon)
        self.max_value = float(max_value)
        self.hessian_floor = float(hessian_floor)

        mode = mode.lower()
        if mode not in {"none", "sum", "mean", "ema"}:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from 'none', 'sum', 'mean', 'ema'.")
        if mode == "ema":
            if not (0.0 < ema_decay < 1.0):
                raise ValueError("ema_decay must be in (0, 1) when mode='ema'.")
        self.mode = mode
        self.ema_decay = float(ema_decay)
        self.reset_interval = int(reset_interval) if reset_interval else None

        if weights is not None and len(weights) != self.num_subsets:
            raise ValueError("weights must have the same length as objectives.")
        self.weights = [float(w) for w in weights] if weights is not None else None
        self.total_weight = (
            sum(self.weights) if self.weights is not None else float(self.num_subsets)
        )
        self.scale_to_full = bool(scale_to_full)
        if self.scale_to_full and self.total_weight <= 0:
            raise ValueError("Total subset weight must be > 0 when scale_to_full=True.")

        self._accumulator: Optional[DataContainer] = None
        self._sum_weights: float = 0.0
        self._contributions: int = 0

    def _get_subset_index(self, algorithm) -> int:
        indices = _extract_data_passes_indices(algorithm.f)
        if not indices:
            return None
        last_block = indices[-1]
        if not last_block:
            return None
        if len(last_block) != 1:
            return None  # full gradient or aggregated pass
        subset_idx = last_block[0]
        if subset_idx < 0 or subset_idx >= self.num_subsets:
            raise IndexError(
                f"Subset index {subset_idx} out of range for {self.num_subsets} subsets."
            )
        return subset_idx

    def _compute_subset_hessian(self, image: DataContainer, subset_idx: int):
        diag = _accumulate_poisson_hessian_diag(
            [self.objectives[subset_idx]], image
        )
        weight = self.weights[subset_idx] if self.weights is not None else 1.0
        return diag, weight

    def _aggregate(self, diag: DataContainer, weight: float) -> DataContainer:
        if weight != 1.0:
            diag *= weight
        if self.mode == "none":
            result = diag
        elif self.mode == "sum":
            if self._accumulator is None:
                self._accumulator = diag.copy()
            else:
                self._accumulator += diag
            self._contributions += 1
            result = self._accumulator.copy()
        elif self.mode == "mean":
            if self._accumulator is None:
                self._accumulator = diag.copy()
                self._sum_weights = weight
            else:
                self._accumulator += diag
                self._sum_weights += weight
            self._contributions += 1
            result = self._accumulator.copy()
            if self._sum_weights > 0:
                result /= self._sum_weights
        else:  # EMA
            if self._accumulator is None:
                self._accumulator = diag.copy()
            else:
                self._accumulator *= self.ema_decay
                diag *= (1.0 - self.ema_decay)
                self._accumulator += diag
            self._contributions += 1
            result = self._accumulator.copy()
        return result

    def _post_scale(
        self, diag: DataContainer, weight: float, sum_weights: Optional[float]
    ) -> DataContainer:
        if not self.scale_to_full:
            return diag
        if self.mode in {"none", "mean"}:
            if self.mode == "none":
                if weight <= 0:
                    raise ValueError("Subset weight must be positive when scale_to_full=True.")
                scale = self.total_weight / weight
            else:  # mean
                scale = self.total_weight
            diag *= scale
        return diag

    def _finalise(self, diag: DataContainer) -> DataContainer:
        arr = _as_array(diag)
        np.maximum(arr, self.hessian_floor, out=arr)
        np.reciprocal(arr, out=arr)
        if np.isfinite(self.max_value):
            np.minimum(arr, self.max_value, out=arr)
        if self.epsilon > 0:
            np.maximum(arr, self.epsilon, out=arr)
        return _fill_from_array(diag, arr)

    def _maybe_reset(self):
        if self.reset_interval is None:
            return
        if self._contributions >= self.reset_interval:
            self._accumulator = None
            self._sum_weights = 0.0
            self._contributions = 0

    def compute_preconditioner(self, algorithm, out=None):
        subset_idx = self._get_subset_index(algorithm)
        if subset_idx is None:
            diag_full = _accumulate_poisson_hessian_diag(self.objectives, algorithm.solution)
            precond = self._finalise(diag_full)
        else:
            diag_subset, weight = self._compute_subset_hessian(algorithm.solution, subset_idx)
            aggregated = self._aggregate(diag_subset, weight)
            current_sum_weights = self._sum_weights if self.mode == "mean" else None
            scaled = self._post_scale(aggregated, weight, current_sum_weights)
            precond = self._finalise(scaled)
            self._maybe_reset()
        if out is None:
            return precond
        out.fill(precond)
        return out


class HarmonicMeanPreconditioner(PreconditionerWithInterval):
    """Preconditioner that combines two preconditioners via inverse-sum blend."""

    def __init__(
        self,
        preconds,
        update_interval=np.inf,
        freeze_iter=np.inf,
        epsilon=1e-6,
        scales=None,
    ):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.epsilon = epsilon
        self.scales = scales

    def compute_preconditioner(self, algorithm, out=None):
        a = self.preconds[0].compute_preconditioner(algorithm)
        b = self.preconds[1].compute_preconditioner(algorithm)
        if self.scales is not None:
            a.sapyb(self.scales[0], a, 0, out=a)
            b.sapyb(self.scales[1], b, 0, out=b)
        if out is None:
            return a * b / (a + b + self.epsilon)
        out.fill(a * b / (a + b + self.epsilon))
        return out


class LehmerMeanPreconditioner(PreconditionerWithInterval):
    """Combine multiple preconditioners via a Lehmer mean of order p.

    For positive scalars x_i, the Lehmer mean of order p is

        L_p(x_1, ..., x_n) = (Σ x_i^p) / (Σ x_i^(p-1))

    so that:
        p = 0  -> harmonic mean; for two inputs, 0.5 * L_0 is the parallel sum
        p = 1  -> arithmetic mean
        p > 1  -> biased toward max
        p < 0  -> biased toward min (not recommended here)

    We use an epsilon floor on the inputs to avoid 0^(p-1) when p < 1,
    and to ensure the combined preconditioner never collapses to exactly 0
    when at least one input preconditioner is positive.
    """

    def __init__(
        self,
        preconds,
        p=1e-1,
        epsilon=1e-12,   # floor applied to all inputs before exponentiation
        update_interval=np.inf,
        freeze_iter=np.inf,
        scales=None,
        output_scale=1.0,
    ):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.p = float(p)
        self.output_scale = float(output_scale)
        if not np.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("output_scale must be finite and > 0.")

        if p < 1 and epsilon <= 0:
            epsilon = 1e-12
            logging.warning(
                f"LehmerMeanPreconditioner: epsilon must be > 0 when p={p} < 1 "
                f"to avoid 0^(p-1) singularities. Using epsilon={epsilon}."
            )
        self.epsilon = float(epsilon)

        if scales is not None and len(scales) != len(preconds):
            raise ValueError("Length of scales must match number of preconditioners.")
        self.scales = scales

    def compute_preconditioner(self, algorithm, out=None):
        # 1) Collect individual preconditioners (optionally scaled)
        values = []
        for i, precond in enumerate(self.preconds):
            v = precond.compute_preconditioner(algorithm)
            if self.scales is not None:
                s = self.scales[i]
                if s != 1:
                    v = v * s
            values.append(v)

        p = self.p
        eps = self.epsilon

        # 2) Apply a symmetric floor to all inputs
        #    This guarantees x >= eps everywhere for all subsequent powers.
        clamped = [v.maximum(eps) for v in values]

        # 3) Lehmer numerator and denominator:
        #      num = Σ x^p
        #      den = Σ x^(p-1)
        x0 = clamped[0]
        num = x0.power(p)
        den = x0.power(p - 1)

        for v in clamped[1:]:
            num += v.power(p)
            den += v.power(p - 1)

        # 4) Final safeguard on denominator and division
        den = den.maximum(eps)
        if out is None:
            return (num / den) * self.output_scale
        num.divide(den, out=out)
        if self.output_scale != 1.0:
            out.sapyb(self.output_scale, out, 0, out=out)
        return out


class MaGeZPreconditioner(PreconditionerWithInterval):
    """
    Preconditioner from MaGeZ PETRIC entry.

    Uses scaled harmonic averaging of sensitivity and prior Hessian diagonal.

    P = (x + delta) / (sensitivity + prior_hessian_diag(x + delta))
    """

    def __init__(
        self,
        sensitivity: BlockDataContainer,
        prior_function,
        hessian_scale: float = 1.5,
        delta: float = 1e-8,
        update_interval: int = 1,
        freeze_iter=np.inf,
    ):
        super().__init__(update_interval, freeze_iter)
        self.sensitivity = sensitivity
        self.prior_function = prior_function
        self.hessian_scale = hessian_scale
        self.delta = delta

    def compute_preconditioner(self, algorithm, out=None):
        x = algorithm.solution.copy()
        x_plus_delta = x + self.delta

        if hasattr(self.prior_function, "preconditioner_diag"):
            prior_hessian_diag = self.prior_function.preconditioner_diag(x_plus_delta).abs()
        elif hasattr(self.prior_function, "hessian_diag"):
            prior_hessian_diag = self.prior_function.hessian_diag(x_plus_delta).abs()
        else:
            raise AttributeError(
                f"Prior function {type(self.prior_function)} does not expose "
                "preconditioner_diag() or hessian_diag()."
            )
        denom = self.sensitivity + self.hessian_scale * prior_hessian_diag
        denom = denom.maximum(1e-12)

        if out is None:
            return x_plus_delta / denom

        x_plus_delta.divide(denom, out=out)
        return out


class ArithmeticMeanPreconditioner(PreconditionerWithInterval):
    """Preconditioner that combines two preconditioners using a simple mean."""

    def __init__(self, preconds, update_interval=np.inf, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds

    def compute_preconditioner(self, algorithm, out=None):
        # prepare output buffer
        acc = self.preconds[0].compute_preconditioner(algorithm)

        for p in self.preconds[1:]:
            acc += p.compute_preconditioner(algorithm)

        if out is None:
            return acc / len(self.preconds)

        acc.divide(len(self.preconds), out=out)
        return out


class IdentityPreconditioner(PreconditionerWithInterval):
    """Identity preconditioner."""

    def __init__(self, update_interval=1, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)

    def compute_preconditioner(self, algorithm, out=None):
        if out is None:
            return algorithm.solution.copy()
        out.fill(algorithm.solution)
        return out


class SubsetPreconditioner(PreconditionerWithInterval):
    """Base class for subset preconditioners."""

    def __init__(self, num_subsets, update_interval=1, freeze_iter=np.inf):
        super().__init__(update_interval, freeze_iter)
        self.num_subsets = num_subsets

    def compute_preconditioner(self, algorithm, out=None):
        raise NotImplementedError


class SubsetEMPreconditioner(SubsetPreconditioner):
    """
    Preconditioner for EM with subsets using sensitivities.
    Can be used for OSEM with sequential sampler or for stochastic EM with random sampler.
    """

    def __init__(
        self,
        num_subsets,
        sensitivities,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=1e-6,
    ):
        super().__init__(num_subsets, update_interval, freeze_iter)
        self.counter = 0
        self.sensitivities = sensitivities
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm, out=None):
        if isinstance(algorithm.f, ScaledFunction):
            adj = self.sensitivities[algorithm.f.function.data_passes_indices[-1][0]]
        else:
            adj = self.sensitivities[algorithm.f.data_passes_indices[-1][0]]
        adj += self.epsilon  # avoid division by zero

        if out is None:
            return algorithm.solution / adj

        algorithm.solution.divide(adj, out=out)
        return out


class DualModalitySubsetKernelisedEMPreconditioner(SubsetPreconditioner):
    def __init__(
        self,
        sens,  # list of BlockDataContainer(s1,s2), length=num_subsets
        kernel,  # [K1, K2] kernel operators for each bed
        num_subsets,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=1e-6,
    ):
        super().__init__(num_subsets, update_interval, freeze_iter)
        self.sens = sens
        self.kernel = kernel
        self.epsilon = epsilon
        self.freeze_kernel_iter = freeze_iter

    def apply(self, algorithm, gradient, out=None):
        """
        Apply the preconditioner, managing freezing and update intervals.
        """

        if algorithm.iteration % self.update_interval == 0 or self.precond is None:
            self.precond = self.compute_preconditioner(algorithm).abs()

        if out is None:
            return gradient * self.precond

        gradient.multiply(self.precond, out=out)
        return out

    def compute_preconditioner(self, algorithm, out=None):
        # for the kernelised EM, we need to freeze the alpha after a certain number of iterations
        # rather than freezing the whole preconditioner
        if algorithm.iteration >= self.freeze_kernel_iter and not self.kernel.freeze_alpha:
            logging.info(f"Freezing kernel parameters at iteration {algorithm.iteration}")
            self.kernel.freeze_alpha = True

        if isinstance(algorithm.f, ScaledFunction):
            sg = algorithm.f.function
        else:
            sg = algorithm.f
        try:
            subset_idx = sg.data_passes_indices[-1][0]
        except IndexError:  # can happen if the preconditioner is called before the first iteration
            subset_idx = 0

        sens = self.sens[subset_idx]
        adj = self.kernel.adjoint(sens) + self.epsilon

        if out is None:
            return algorithm.solution / adj

        algorithm.solution.divide(adj, out=out)
        return out


class SubsetKernelisedEMPreconditioner(SubsetPreconditioner):
    """
    Subset preconditioner for (hybrid) kernelised EM.
    Can be used for OS(H)KEM with sequential sampler or for stochastic (H)KEM with random sampler.
    """

    def __init__(
        self,
        num_subsets,
        sensitivities,
        kernel,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=1e-6,
    ):
        super().__init__(num_subsets, update_interval, freeze_iter=np.inf)
        self.sensitivities = sensitivities
        self.kernel = kernel
        self.epsilon = epsilon
        self.frozen_alpha = None
        self.freeze_kernel_iter = freeze_iter

    def apply(self, algorithm, gradient, out=None):
        """
        Apply the preconditioner, managing freezing and update intervals.
        """
        if algorithm.iteration % self.update_interval == 0 or self.precond is None:
            self.precond = self.compute_preconditioner(algorithm).abs()
        if out is None:
            return gradient * self.precond

        gradient.multiply(self.precond, out=out)
        return out

    def compute_preconditioner(self, algorithm, out=None):
        # for the kernelised EM, we need to freeze the alpha after a certain number of iterations
        # rather than freezing the whole preconditioner
        if algorithm.iteration >= self.freeze_kernel_iter:
            self.kernel.freeze_alpha = True

        if isinstance(algorithm.f, ScaledFunction):
            sg = algorithm.f.function
        else:
            sg = algorithm.f
        # if list is empty, return 0
        try:
            subset_idx = sg.data_passes_indices[-1][0]
        except IndexError:  # can happen if the preconditioner is called before the first iteration
            subset_idx = 0
        adj = self.kernel.adjoint(self.sensitivities[subset_idx])
        adj = adj.abs()
        adj += self.epsilon  # avoid division by zero

        if out is None:
            return algorithm.solution.divide(adj)

        algorithm.solution.divide(adj, out=out)
        return out
