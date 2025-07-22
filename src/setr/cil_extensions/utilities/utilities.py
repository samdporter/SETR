from cil.optimisation.utilities import StepSizeRule


class LinearDecayStepSizeRule(StepSizeRule):
    """
    Linear decay of the step size with iteration.
    """
    def __init__(self, initial_step_size: float, decay: float):
        self.initial_step_size = initial_step_size
        self.decay = decay
        self.step_size = initial_step_size

    def get_step_size(self, algorithm):
        return self.initial_step_size / (1 + self.decay * algorithm.iteration)
    

class ArmijoStepSearchRule(StepSizeRule):
    """
    Armijo rule for step size for initial steps, followed by linear decay.
    This is obsolete and will only work with a modified algorithm that supports it.
    """
    def __init__(self, initial_step_size: float, beta: float, 
                 max_iter: int, tol: float, steps: int,
                 maximiser=False):
        
        self.initial_step_size = initial_step_size
        self.min_step_size = initial_step_size
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.steps = steps
        self.counter = 0
        self.f_x = None
        self.maximiser = maximiser

    def get_step_size(self, algorithm):
        """
        Calculate and return the step size based on the Armijo rule.
        Step size is updated every `update_interval` iterations or during the initial steps.

        After Armijo iterations are exhausted, linear decay is applied.
        """
        # Check if we're within the initial steps or at an update interval
        if self.counter < self.steps: 
            if self.f_x is None:
                self.f_x = algorithm.f(algorithm.solution) + algorithm.g(algorithm.solution)
            precond_grad = algorithm.preconditioner.apply(algorithm, algorithm.gradient_update)

            # if x is zero and the gradient is zero, we should ignore the gradient
            is_zero = algorithm.solution.power(0)
            precond_grad_negs = precond_grad.minimum(0)
            precond_grad_pos = precond_grad.maximum(0)
            precond_grad = precond_grad_pos + precond_grad_negs * is_zero

            g_norm = algorithm.gradient_update.dot(precond_grad)
            print(f"Old Objective value: {self.f_x}")


            
            # Reset step size to initial value for the Armijo search
            step_size = self.initial_step_size
            
            # Armijo step size search
            for _ in range(self.max_iter):
                # Proximal step
                x_new = algorithm.g.proximal(algorithm.solution.copy() - step_size * precond_grad, step_size)
                f_x_new = algorithm.f(x_new) + algorithm.g(x_new)
                print(f"New Objective value: {f_x_new}")
                # Armijo condition check
                print(f"Condition value: {self.f_x - self.tol * step_size * g_norm}")
                if self.maximiser:
                    # negating the condition to turn it into a maximisation problem
                    # we still use a minus sign because step size is negative
                    if f_x_new >= self.f_x - self.tol * step_size * g_norm:
                        self.f_x = f_x_new
                        break
                else:
                    if f_x_new <= self.f_x - self.tol * step_size * g_norm:
                        self.f_x = f_x_new
                        break
                
                # Reduce step size
                step_size *= self.beta

                print(f"Step size: {step_size}")
            
            # Update the internal state with the new step size as the minimum of the current and previous step sizes
            self.min_step_size = min(step_size, self.min_step_size)
            
            if self.counter < self.steps:
                self.counter += 1
            
        return step_size