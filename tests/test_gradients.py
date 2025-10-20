import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch", reason="Gradient operators require PyTorch")

from setr.core.gradients.gradients import Gradient, GradientOptimized, LegacyGradient, check_adjoint


@pytest.mark.parametrize("operator_cls", [Gradient, GradientOptimized])
@pytest.mark.parametrize("bnd_cond", ["Periodic", "Neumann"])
@pytest.mark.parametrize("stencil", ["6", "18", "26"])
@pytest.mark.parametrize("both_directions", [False, True])
@pytest.mark.parametrize("max_step", [1, 3])
def test_gradient_adjointness(operator_cls, bnd_cond, stencil, both_directions, max_step):
    op = operator_cls(
        voxel_sizes=(1.0, 1.0, 1.0),
        stencil=stencil,
        bnd_cond=bnd_cond,
        both_directions=both_directions,
        max_step=max_step,
        normalize=True,
    )
    err = check_adjoint(op, shape=(6, 5, 4), trials=2)
    assert err < 1.0e-5


def test_gradient_dispatch_selects_impl():
    grad_6 = Gradient(voxel_sizes=(1.0, 1.0, 1.0), stencil="6")
    assert isinstance(grad_6._impl, LegacyGradient)

    grad_18 = Gradient(voxel_sizes=(1.0, 1.0, 1.0), stencil="18")
    assert isinstance(grad_18._impl, GradientOptimized)

    grad_26 = Gradient(voxel_sizes=(1.0, 1.0, 1.0), stencil="26", both_directions=True)
    assert isinstance(grad_26._impl, GradientOptimized)
