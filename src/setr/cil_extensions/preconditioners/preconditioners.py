import logging
import numpy as np
from cil.framework import BlockDataContainer
from cil.optimisation.functions import ScaledFunction
from cil.optimisation.utilities import Preconditioner
from sirf.STIR import SeparableGaussianImageFilter


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
        self,
        s_inv,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=None,
        smooth=False,
    ):
        super().__init__(update_interval, freeze_iter)
        self.s_inv = s_inv
        if smooth:
            self.gaussian = SeparableGaussianImageFilter()
            self.gaussian.set_fwhms((10, 10, 10))
        else:
            self.gaussian = None
        if epsilon is None:
            epsilon = s_inv.max() * 1e-10
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm, out=None):
        x = algorithm.solution.copy()

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
    """Combine multiple preconditioners via a Lehmer mean of order p."""

    def __init__(
        self,
        preconds,
        p=1e-1,  # Lehmer order: p=0→harmonic, p=1→arithmetic, p>1→toward max
        epsilon=0,
        update_interval=np.inf,
        freeze_iter=np.inf,
    ):
        super().__init__(update_interval, freeze_iter)
        self.preconds = preconds
        self.p = p
        self.epsilon = epsilon

    def compute_preconditioner(self, algorithm, out=None):

        # Collect (and, if needed, clamp) inputs
        precond_values = [p.compute_preconditioner(algorithm) for p in self.preconds]

        p = self.p
        need_clamp_for_den = p < 1
        eps = self.epsilon

        # First term
        x0 = precond_values[0]
        base_num = x0
        base_den = x0.maximum(eps) if need_clamp_for_den else x0

        num = base_num.power(p)          # Σ x^p
        den = base_den.power(p - 1)      # Σ x^(p-1), safe if p<1

        # Accumulate remaining terms
        for x in precond_values[1:]:
            base_num = x
            base_den = x.maximum(eps) if need_clamp_for_den else x
            num += base_num.power(p)
            den += base_den.power(p - 1)

        # Final guard and division
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
        adj = self.kernel.adjoint(self.sensitivities[sg.data_passes_indices[-1][0]])
        adj += self.epsilon
        adj = adj.abs()

        if out is None:
            return algorithm.solution / adj

        algorithm.solution.divide(adj, out=out)
        return out
