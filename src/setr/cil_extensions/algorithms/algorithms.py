import logging

def ista_update_step(self) -> None:
    r"""Perform a single ISTA update iteration.

    .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))
    """
    logging.info("Performing ISTA update step")
    self.f.gradient(self.x_old, out=self.gradient_update)
    try:
        step_size = self.step_size_rule.get_step_size(self)
    except NameError:
        raise NameError(
            "`step_size` must be None, a real float or a child class of "
            "cil.optimisation.utilities.StepSizeRule"
        )
    if self.preconditioner is not None:
        self.x_old.sapyb(
            1.0,
            self.preconditioner.apply(self, self.gradient_update),
            -step_size,
            out=self.x_old,
        )
    else:
        self.x_old.sapyb(1.0, self.gradient_update, -step_size, out=self.x_old)
    self.g.proximal(self.x_old, step_size, out=self.x)