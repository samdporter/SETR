def ista_update_step(self) -> None:
    r"""Perform a single ISTA update iteration.

    .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))
    """
    self.gradient_update = self.f.gradient(self.x_old, out=self.gradient_update)
    M = self.x.max()
    try:
        step_size = self.step_size_rule.get_step_size(self)
    except NameError:
        raise NameError(
            "`step_size` must be None, a real float or a child class of "
            "cil.optimisation.utilities.StepSizeRule"
        )
    if self.preconditioner is not None:
        grad = self.preconditioner.apply(self, self.gradient_update)
    else:
        grad = self.gradient_update.clone()

    grad=grad.maximum(-M/self.step_size)
    grad=grad.minimum(M/self.step_size)

    self.x_old.sapyb(1.0, grad, -step_size, out=self.x_old)
    self.g.proximal(self.x_old, step_size, out=self.x)
