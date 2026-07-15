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
        BlockLehmerMeanPreconditioner,
        BSREMPreconditioner,
        MajorisingHessianBlockPreconditioner,
        MajorisingHessianDiagonalPreconditioner,
    )
    from recon_core.utils.sirf import _safe_reciprocal
except (ImportError, OSError) as exc:  # pragma: no cover - external dependency
    pytest.skip(f"CIL/SIRF dependencies unavailable: {exc}", allow_module_level=True)


class ArrayContainer:
    """Small array-backed container implementing the subset of DataContainer API we need."""

    def __init__(self, data):
        self._arr = np.array(data, dtype=np.float64, copy=True)

    def asarray(self):
        return self._arr

    def as_array(self):
        return self._arr

    def fill(self, data):
        self._arr[...] = np.asarray(data, dtype=self._arr.dtype)

    def copy(self):
        return ArrayContainer(self._arr.copy())

    def max(self):
        return float(np.max(self._arr))

    def abs(self, out=None):
        arr = np.abs(self._arr)
        if out is None:
            return ArrayContainer(arr)
        out.fill(arr)
        return out

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


class _IdentityOperator:
    def direct(self, x):
        return x

    def adjoint(self, x):
        return x


class _MatrixOperator:
    def __init__(self, matrix):
        self._mat = np.array(matrix, dtype=np.float64, copy=True)

    def direct(self, x):
        return ArrayContainer(self._mat @ x.asarray())

    def adjoint(self, y):
        return ArrayContainer(self._mat.T @ y.asarray())


class _BlockMatrixOperator:
    def __init__(self, matrix):
        self._mat = np.array(matrix, dtype=np.float64, copy=True)

    def direct(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer(
                *[ArrayContainer(self._mat @ con.asarray()) for con in x.containers]
            )
        return ArrayContainer(self._mat @ x.asarray())

    def adjoint(self, y):
        if isinstance(y, BlockDataContainer):
            return BlockDataContainer(
                *[ArrayContainer(self._mat.T @ con.asarray()) for con in y.containers]
            )
        return ArrayContainer(self._mat.T @ y.asarray())


class _MaxContainer:
    def __init__(self, value):
        self._value = float(value)

    def max(self):
        return self._value


def _manual_invert_blocks(hessian_block, floor, max_value, safety_scale):
    hessian_block = 0.5 * (hessian_block + np.swapaxes(hessian_block, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(hessian_block)
    eigvals = np.maximum(eigvals, floor)
    inv_eigs = 1.0 / eigvals
    if np.isfinite(max_value):
        inv_eigs = np.minimum(inv_eigs, max_value)
    inv_block = (eigvecs * inv_eigs[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    inv_block = 0.5 * (inv_block + np.swapaxes(inv_block, -1, -2))
    return inv_block * safety_scale


def test_safe_reciprocal_zero_mask_is_deterministic():
    arr = np.array([0.0, 2.0, -4.0, 0.0], dtype=np.float32)
    out = _safe_reciprocal(arr)
    expected = np.array([0.0, 0.5, -0.25, 0.0], dtype=np.float32)
    assert np.allclose(out, expected)
    assert np.all(np.isfinite(out))


def test_majorising_diagonal_preconditioner_applies_floor_clamp_and_safety_scale():
    precond = MajorisingHessianDiagonalPreconditioner(
        s_inv=None,
        prior=None,
        hessian_floor=0.5,
        max_value=1.5,
        safety_scale=0.5,
    )
    precond._compute_data_hessian_diag = lambda _image: ArrayContainer([1.0, 4.0, 0.0, -2.0])
    precond._compute_prior_hessian_diag = lambda _image: ArrayContainer([3.0, 0.0, 1.0, 2.0])

    algo = SimpleNamespace(solution=ArrayContainer([0.0, 0.0, 0.0, 0.0]))
    out = precond.compute_preconditioner(algo)
    out_arr = out.asarray()

    # Total Hessian: [4, 4, 1, 0] -> floor 0.5 -> inverse [0.25, 0.25, 1, 2]
    # -> clamp max 1.5 -> [0.25, 0.25, 1, 1.5] -> safety_scale 0.5
    expected = np.array([0.125, 0.125, 0.5, 0.75], dtype=np.float64)
    assert np.allclose(out_arr, expected, atol=1e-12, rtol=1e-12)


def test_majorising_diagonal_uses_direct_prior_diag_without_composition():
    class _Prior:
        @staticmethod
        def preconditioner_diag(_x, **_kwargs):
            return ArrayContainer([4.0, 9.0])

    precond = MajorisingHessianDiagonalPreconditioner(
        s_inv=None,
        prior=_Prior(),
    )

    image = ArrayContainer([3.0, 7.0])
    diag = precond._compute_prior_hessian_diag(image).asarray()
    expected = np.array([4.0, 9.0], dtype=np.float64)
    assert np.allclose(diag, expected, atol=1e-12, rtol=1e-12)


def test_majorising_block_preconditioner_matches_manual_inverse_and_scale():
    precond = MajorisingHessianBlockPreconditioner(
        s_inv=None,
        prior=SimpleNamespace(operator=_IdentityOperator()),
        hessian_floor=1e-3,
        max_value=10.0,
        safety_scale=0.25,
    )

    prior_h = np.array(
        [
            [[2.0, 0.3], [0.3, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    data_h = np.array(
        [
            [[1.0, 0.0], [0.0, 4.0]],
            [[-1.0, 0.0], [0.0, 1e-3]],
        ],
        dtype=np.float64,
    )

    precond._compute_prior_hessian_block = lambda _image: prior_h.copy()
    precond._compute_data_hessian_block = lambda _image: data_h.copy()

    algo = SimpleNamespace(solution=None)
    out = precond.compute_preconditioner(algo)
    expected = _manual_invert_blocks(
        prior_h + data_h,
        floor=1e-3,
        max_value=10.0,
        safety_scale=0.25,
    )

    assert np.allclose(out, expected, atol=1e-12, rtol=1e-12)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)
    assert np.max(eigvals) <= 2.5 + 1e-12  # max_value * safety_scale


def test_majorising_block_apply_block_action_in_identity_geometry():
    prior = SimpleNamespace(operator=_IdentityOperator())
    precond = MajorisingHessianBlockPreconditioner(
        s_inv=None,
        prior=prior,
    )

    gradient = BlockDataContainer(
        ArrayContainer([1.0, 2.0]),
        ArrayContainer([3.0, 4.0]),
    )
    block_arr = np.array(
        [
            [[2.0, 0.5], [0.5, 1.0]],
            [[1.0, -0.2], [-0.2, 3.0]],
        ],
        dtype=np.float64,
    )

    out = precond._apply_block_preconditioner(gradient, block_arr)
    out0 = out.containers[0].asarray()
    out1 = out.containers[1].asarray()

    expected0 = np.array([3.5, 1.2], dtype=np.float64)
    expected1 = np.array([3.5, 11.6], dtype=np.float64)
    assert np.allclose(out0, expected0, atol=1e-12, rtol=1e-12)
    assert np.allclose(out1, expected1, atol=1e-12, rtol=1e-12)


def test_majorising_block_data_hessian_stays_diagonal_in_shared_space():
    prior = SimpleNamespace()
    image = BlockDataContainer(
        ArrayContainer([100.0, 1.0]),
        ArrayContainer([20.0, 2.0]),
    )
    s_inv = BlockDataContainer(
        ArrayContainer([0.01, 1.0]),
        ArrayContainer([0.02, 0.5]),
    )

    precond = MajorisingHessianBlockPreconditioner(
        s_inv=s_inv,
        prior=prior,
        x_epsilon=1e-8,
        hessian_floor=1e-8,
    )
    data_h = precond._compute_data_hessian_block(image)

    expected = np.array(
        [
            [[1.0 / ((100.0 + 1e-8) * 0.01), 0.0], [0.0, 1.0 / ((20.0 + 1e-8) * 0.02)]],
            [[1.0 / ((1.0 + 1e-8) * 1.0), 0.0], [0.0, 1.0 / ((2.0 + 1e-8) * 0.5)]],
        ],
        dtype=np.float64,
    )
    assert data_h.shape == (2, 2, 2)
    assert np.allclose(data_h, expected, atol=1e-12, rtol=1e-12)


def test_majorising_block_data_hessian_uses_shared_space_per_voxel_curvature():
    prior = SimpleNamespace()
    image = BlockDataContainer(
        ArrayContainer([100.0, 1.0]),
        ArrayContainer([1.0, 1.0]),
    )
    s_inv = BlockDataContainer(
        ArrayContainer([0.01, 1.0]),
        ArrayContainer([1.0, 1.0]),
    )

    precond = MajorisingHessianBlockPreconditioner(
        s_inv=s_inv,
        prior=prior,
        x_epsilon=1e-8,
        hessian_floor=1e-8,
    )
    data_h = precond._compute_data_hessian_block(image)

    expected_pet_diag = np.array([1.0, 1.0], dtype=np.float64)
    expected_spect_diag = np.array([1.0, 1.0], dtype=np.float64)

    assert np.allclose(data_h[..., 0, 0], expected_pet_diag, atol=1e-8, rtol=1e-8)
    assert np.allclose(data_h[..., 1, 1], expected_spect_diag, atol=1e-8, rtol=1e-8)
    assert np.allclose(data_h[..., 0, 1], 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(data_h[..., 1, 0], 0.0, atol=1e-12, rtol=1e-12)


def test_get_preconditioners_threads_safety_scale_into_block_majoriser():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    args = SimpleNamespace(
        precond_type="mm_diag_block_maj",
        precond_combine="majoriser",
        precond_safety_scale=0.3,
        block_scalar_reduction="diag",
        precond_data_epsilon=1e-8,
        precond_freeze_epochs=None,
    )
    s_inv = SimpleNamespace(containers=[_MaxContainer(2.0), _MaxContainer(3.0)])
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    class _Prior:
        @staticmethod
        def inv_preconditioner_block(*_args, **_kwargs):
            return None

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object(), object(), object()],
        update_interval=7,
        priors_list=[_Prior()],
        initial_estimates=initial_estimates,
    )

    assert isinstance(precond, MajorisingHessianBlockPreconditioner)
    assert precond.safety_scale == pytest.approx(0.3)
    assert precond.update_interval == 7


def test_block_lehmer_p0_half_scale_is_exact_factory_parallel_sum():
    """The production Lehmer path must share the exact majoriser operands."""
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    class _Prior:
        @staticmethod
        def preconditioner_block(*_args, **_kwargs):
            return np.array(
                [
                    [[4.0, 0.7], [0.7, 2.5]],
                    [[1.5, -0.2], [-0.2, 3.0]],
                ],
                dtype=np.float64,
            )

    common_args = dict(
        precond_type="mm_diag_block_maj",
        precond_safety_scale=0.35,
        block_scalar_reduction="diag",
        precond_data_epsilon=1e-8,
        precond_freeze_epochs=None,
        lehmer_p=0.0,
        lehmer_scale=0.5,
        em_precond_smooth=False,
    )
    s_inv = BlockDataContainer(
        ArrayContainer([0.02, 2.0]),
        ArrayContainer([8.0, 0.05]),
    )
    initial_estimates = SimpleNamespace(
        containers=[ArrayContainer([12.0, 3.0]), ArrayContainer([2.0, 20.0])]
    )
    image = BlockDataContainer(
        ArrayContainer([10.0, 1.0]),
        ArrayContainer([1.5, 15.0]),
    )
    algorithm = SimpleNamespace(solution=image, iteration=0)

    lehmer = get_preconditioners(
        args=SimpleNamespace(**common_args, precond_combine="lehmer"),
        s_inv=s_inv,
        all_funs=[object(), object(), object()],
        update_interval=3,
        priors_list=[_Prior()],
        initial_estimates=initial_estimates,
    )
    parallel_sum = get_preconditioners(
        args=SimpleNamespace(**common_args, precond_combine="majoriser"),
        s_inv=s_inv,
        all_funs=[object(), object(), object()],
        update_interval=3,
        priors_list=[_Prior()],
        initial_estimates=initial_estimates,
    )

    assert isinstance(lehmer, BlockLehmerMeanPreconditioner)
    assert isinstance(parallel_sum, MajorisingHessianBlockPreconditioner)
    assert np.array_equal(
        lehmer.compute_preconditioner(algorithm),
        parallel_sum.compute_preconditioner(algorithm),
    )


def test_get_preconditioners_mm_diag_returns_diagonal_majoriser_with_safety_scale():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    class _Prior:
        @staticmethod
        def inv_preconditioner_diag(*_args, **_kwargs):
            return ArrayContainer([1.0, 1.0])

        @staticmethod
        def preconditioner_diag(*_args, **_kwargs):
            return ArrayContainer([1.0, 1.0])

    args = SimpleNamespace(
        precond_type="mm_diag_gershgorin_maj",
        precond_combine="majoriser",
        precond_safety_scale=0.5,
        block_scalar_reduction="diag",
        precond_data_epsilon=1e-8,
        precond_freeze_epochs=None,
    )
    s_inv = SimpleNamespace(containers=[_MaxContainer(2.0), _MaxContainer(3.0)])
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object(), object(), object()],
        update_interval=3,
        priors_list=[_Prior()],
        initial_estimates=initial_estimates,
    )

    assert isinstance(precond, MajorisingHessianDiagonalPreconditioner)
    assert precond.safety_scale == pytest.approx(0.5)


def test_get_preconditioners_mm_diag_sums_multiple_prior_diagonals():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    class _PriorA:
        @staticmethod
        def preconditioner_diag(*_args, **_kwargs):
            return BlockDataContainer(
                ArrayContainer([1.0, 2.0]),
                ArrayContainer([3.0, 4.0]),
            )

    class _PriorB:
        @staticmethod
        def hessian_diag(*_args, **_kwargs):
            return BlockDataContainer(
                ArrayContainer([5.0, 6.0]),
                ArrayContainer([7.0, 8.0]),
            )

    args = SimpleNamespace(
        precond_type="mm_diag_gershgorin_maj",
        precond_combine="majoriser",
        precond_safety_scale=1.0,
        block_scalar_reduction="diag",
        precond_data_epsilon=1e-8,
        precond_freeze_epochs=None,
    )
    s_inv = SimpleNamespace(containers=[_MaxContainer(2.0), _MaxContainer(3.0)])
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object(), object(), object()],
        update_interval=3,
        priors_list=[_PriorA(), _PriorB()],
        initial_estimates=initial_estimates,
    )

    image = BlockDataContainer(
        ArrayContainer([0.0, 0.0]),
        ArrayContainer([0.0, 0.0]),
    )
    prior_diag = precond._compute_prior_hessian_diag(image)
    assert np.allclose(prior_diag.containers[0].asarray(), np.array([6.0, 8.0]))
    assert np.allclose(prior_diag.containers[1].asarray(), np.array([10.0, 12.0]))


def test_get_preconditioners_mm_block_sums_block_and_diagonal_priors():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    class _BlockPrior:
        @staticmethod
        def preconditioner_block(*_args, **_kwargs):
            return np.array(
                [
                    [[1.0, 0.2], [0.2, 2.0]],
                    [[3.0, 0.4], [0.4, 4.0]],
                ],
                dtype=np.float64,
            )

    class _DiagPrior:
        @staticmethod
        def hessian_diag(*_args, **_kwargs):
            return BlockDataContainer(
                ArrayContainer([10.0, 20.0]),
                ArrayContainer([30.0, 40.0]),
            )

    args = SimpleNamespace(
        precond_type="mm_diag_block_maj",
        precond_combine="majoriser",
        precond_safety_scale=1.0,
        block_scalar_reduction="diag",
        precond_data_epsilon=1e-8,
        precond_freeze_epochs=None,
    )
    s_inv = SimpleNamespace(containers=[_MaxContainer(2.0), _MaxContainer(3.0)])
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object(), object(), object()],
        update_interval=3,
        priors_list=[_BlockPrior(), _DiagPrior()],
        initial_estimates=initial_estimates,
    )

    image = BlockDataContainer(
        ArrayContainer([0.0, 0.0]),
        ArrayContainer([0.0, 0.0]),
    )
    prior_block = precond._compute_prior_hessian_block(image)
    expected = np.array(
        [
            [[11.0, 0.2], [0.2, 32.0]],
            [[23.0, 0.4], [0.4, 44.0]],
        ],
        dtype=np.float64,
    )
    assert np.allclose(prior_block, expected, atol=1e-12, rtol=1e-12)


def test_get_preconditioners_bsrem_defaults_to_smoothed_em_preconditioner():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    args = SimpleNamespace(
        precond_type="bsrem",
        precond_combine="majoriser",
    )
    s_inv = _MaxContainer(2.0)
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object()],
        update_interval=1,
        priors_list=None,
        initial_estimates=initial_estimates,
    )

    assert isinstance(precond, BSREMPreconditioner)
    assert precond.gaussian is not None
    assert precond.max_val == pytest.approx(7.0)


def test_get_preconditioners_bsrem_smoothing_can_be_disabled():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    args = SimpleNamespace(
        precond_type="bsrem",
        precond_combine="majoriser",
        em_precond_smooth=False,
    )
    s_inv = _MaxContainer(2.0)
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object()],
        update_interval=1,
        priors_list=None,
        initial_estimates=initial_estimates,
    )

    assert isinstance(precond, BSREMPreconditioner)
    assert precond.gaussian is None


def test_get_preconditioners_bsrem_initial_cap_can_be_disabled():
    exp_src = ROOT.parent / "recon_experiments" / "src"
    if str(exp_src) not in sys.path:
        sys.path.insert(0, str(exp_src))

    try:
        from recon_experiments.runners.dtnv_common import get_preconditioners
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"recon_experiments unavailable: {exc}")

    args = SimpleNamespace(
        precond_type="bsrem",
        precond_combine="majoriser",
        em_precond_cap_to_initial_max=False,
    )
    s_inv = _MaxContainer(2.0)
    initial_estimates = SimpleNamespace(containers=[_MaxContainer(5.0), _MaxContainer(7.0)])

    precond = get_preconditioners(
        args=args,
        s_inv=s_inv,
        all_funs=[object()],
        update_interval=1,
        priors_list=None,
        initial_estimates=initial_estimates,
    )

    assert isinstance(precond, BSREMPreconditioner)
    assert precond.max_val is None
