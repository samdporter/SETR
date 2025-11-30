import logging
from typing import Optional, Sequence

import numpy as np
from cil.framework import BlockDataContainer, DataContainer
from cil.optimisation.functions import ScaledFunction
from cil.optimisation.utilities import Preconditioner
from sirf.STIR import SeparableGaussianImageFilter

from setr.cil_extensions.functions import ensure_kl_hessian_support
from setr.utils.sirf import get_array

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
    """Preconditioner that combines two preconditioners using a harmonic mean."""

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
            return 2 * a * b / (a + b + self.epsilon)
        out.fill(2 * a * b / (a + b + self.epsilon))
        return out


class LehmerMeanPreconditioner(PreconditionerWithInterval):
    """Combine multiple preconditioners via a Lehmer mean of order p.

    For positive scalars x_i, the Lehmer mean of order p is

        L_p(x_1, ..., x_n) = (Σ x_i^p) / (Σ x_i^(p-1))

    so that:
        p = 0  -> harmonic mean
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
    ):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.p = float(p)

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
            return num / den
        num.divide(den, out=out)
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


class DualModalitySubsetKernelisedEMPreconditioner(SubsetPreconditioner):
    def __init__(
        self,
        sens_bdcs,  # list of BlockDataContainer(s1,s2), length=num_subsets
        kernel,  # [K1, K2] kernel operators for each bed
        uncombine_ops,  # [U1, U2] uncombine (adjoint) operators
        num_subsets,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=1e-6,
    ):
        super().__init__(num_subsets, update_interval, freeze_iter)
        self.sens_bdcs = sens_bdcs
        self.kernel = kernel
        self.uncombine_ops = uncombine_ops
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
        if algorithm.iteration >= self.freeze_kernel_iter:
            for k in self.kernel:
                k.freeze_alpha = True

        if isinstance(algorithm.f, ScaledFunction):
            sg = algorithm.f.function
        else:
            sg = algorithm.f

        k_s = self.kernel[0].adjoint(self.sens_bdc.containers[0])
        total = self.uncombine_ops[0].adjoint(k_s)
        for i in range(1, len(self.sens_bdc.containers)):
            k_s = self.kernel[i].adjoint(self.sens_bdc.containers[i])
            total += self.uncombine_ops[i].adjoint(k_s)
        total += self.epsilon  # to avoid division by zero
        total = total.abs()

        if out is None:
            return algorithm.solution / total

        algorithm.solution.divide(total, out=out)
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
