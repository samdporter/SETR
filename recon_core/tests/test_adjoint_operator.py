from cil.optimisation.operators import LinearOperator

from recon_core.cil_extensions.operators import AdjointOperator


class _ToyOperator(LinearOperator):
    def __init__(self):
        super().__init__(domain_geometry="native", range_geometry="shared")

    def direct(self, x, out=None):
        return ("direct", x)

    def adjoint(self, x, out=None):
        return ("adjoint", x)


def test_adjoint_operator_swaps_domain_and_range_geometry():
    op = _ToyOperator()

    adjoint = AdjointOperator(op)

    assert adjoint.domain_geometry() == op.range_geometry()
    assert adjoint.range_geometry() == op.domain_geometry()
    assert adjoint.direct("x") == ("adjoint", "x")
    assert adjoint.adjoint("y") == ("direct", "y")
