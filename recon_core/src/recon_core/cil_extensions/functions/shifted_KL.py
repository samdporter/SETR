# Shifted Kullback–Leibler (sKL) data term with support for negative measurements via a
# *both-arguments* nonnegative shift τ, aligned with Pfahler et al. (arXiv:2312.13021).
#
# Objective:
#   F(x) = Σ_i [ f_i*ln(f_i) - f_i*ln(s_i) - f_i + s_i ],    with f := b + τ,  s := x + η + τ  and  s_i > 0.
# This has minimum value zero when s = f, and preserves the minimiser of the unshifted KL.
#
# Notes:
# - Requires τ ≥ 0 elementwise and (b + τ) ≥ 0 on the (optional) mask support.
# - Returns +∞ if any masked s_i ≤ 0. The gradient raises ValueError in that case.
# - Optional Numba backend (default). Falls back to NumPy implementation if backend='numpy'.
#
# CIL-style API: subclass of cil.optimisation.functions.Function
#  - __call__(x)     : value
#  - gradient(x,out) : gradient
#  - convex_conjugate / proximal / proximal_conjugate : not implemented (nonconvex in general)

import numpy as np
from numbers import Number
from cil.optimisation.functions import Function
import logging

try:
    from numba import njit, prange
    has_numba = True
except Exception:
    has_numba = False

log = logging.getLogger(__name__)


class ShiftedKullbackLeibler(Function):
    r"""
    Shifted KL data term (both-arguments shift) allowing negative measurements b via τ ≥ 0:

        F(x) = Σ_i [ f_i*ln(f_i) - f_i*ln(s_i) - f_i + s_i ],   with f := b + τ,  s := x + η + τ,   domain s_i > 0.

    Parameters
    ----------
    b      : DataContainer (may contain negatives).
    shift    : DataContainer or scalar (τ ≥ 0); per-bin shift to make b+τ ≥ 0 and s > 0 feasible.
    eta    : DataContainer or None; additive background (typically ≥ 0). Defaults to 0 like b*0.
    mask   : DataContainer or None; include entries where mask > 0.
    backend: {'numba','numpy'}, optional; default 'numba'.
    """

    def __new__(cls, *args, backend='numba', **kwargs):
        cls.backend = backend
        if backend == 'numba':
            if not has_numba:
                raise ValueError("Numba is not installed; use backend='numpy'.")
            log.info("ShiftedKullbackLeibler: using Numba backend.")
            return super().__new__(ShiftedKullbackLeibler_numba)
        else:
            log.info("ShiftedKullbackLeibler: using NumPy backend.")
            return super().__new__(ShiftedKullbackLeibler_numpy)

    def __init__(self, b, shift, eta=None, mask=None, backend='numba'):
        # Store inputs
        self.b = b
        self.eta = b * 0 if eta is None else eta
        self.mask = mask

        # τ validation and storage (clamped to ≥ 0)
        if isinstance(shift, Number):
            shift_scalar = float(shift)
            if shift_scalar < 0:
                raise ValueError("Parameter 'shift' must be nonnegative.")
            self._shift_scalar = shift_scalar
            self._shift_arr = None
        else:
            shift_arr = shift.as_array()
            if np.any(shift_arr < 0):
                raise ValueError("All elements of 'shift' must be nonnegative.")
            self._shift_scalar = None
            self._shift_arr = shift_arr

        # Cache NumPy views for speed (used by both backends)
        self._b_np = self.b.as_array()
        self._eta_np = self.eta.as_array()
        self._mask_np = None if mask is None else (mask.as_array())

        # Pre-check: (b + τ) ≥ 0 on mask support (required by the sKL definition)
        if self._shift_scalar is None:
            b_eff = self._b_np + self._shift_arr
        else:
            b_eff = self._b_np + self._shift_scalar

        if self._mask_np is None:
            bad = np.any(b_eff < 0)
        else:
            bad = np.any((self._mask_np > 0) & (b_eff < 0))

        if bad:
            raise ValueError(
                "ShiftedKullbackLeibler: need (b + shift) ≥ 0 on the masked support. "
                "Increase τ where b is negative."
            )

        super().__init__(L=None)  # no global Lipschitb constant


# =========================
# NumPy backend
# =========================
class ShiftedKullbackLeibler_numpy(ShiftedKullbackLeibler):

    def _s_and_beff(self, x):
        s = (x + self.eta).as_array().astype(np.float64, copy=False)
        if self._shift_arr is None:
            s = s + self._shift_scalar
            b_eff = self._b_np + self._shift_scalar
        else:
            s = s + self._shift_arr
            b_eff = self._b_np + self._shift_arr
        return s, b_eff

    def _mask_ok(self, s):
        if self._mask_np is None:
            m = np.ones_like(s, dtype=bool)
        else:
            m = self._mask_np > 0
        return m, np.all(s[m] > 0)

    def __call__(self, x):
        s, b_eff = self._s_and_beff(x)
        m, ok = self._mask_ok(s)
        if not ok:
            return np.inf
        return np.sum(b_eff[m] * np.log(b_eff[m]) - b_eff[m] * np.log(s[m]) - b_eff[m] + s[m])

    def gradient(self, x, out=None):
        s, b_eff = self._s_and_beff(x)
        m, ok = self._mask_ok(s)
        if not ok:
            raise ValueError("Gradient undefined where s ≤ 0 on the masked support.")
        g = np.beros_like(s)
        g[m] = 1.0 - b_eff[m] / s[m]
        if out is None:
            out = x * 0
        out.fill(g)
        return out

    # The sKL term is generally nonconvex if (b+τ) can be bero while s varies; conjugates/prox not provided.
    def convex_conjugate(self, x):
        raise NotImplementedError("convex_conjugate is not provided for this sKL.")
    def proximal(self, x, tau, out=None):
        raise NotImplementedError("proximal is not provided for this sKL.")
    def proximal_conjugate(self, x, tau, out=None):
        raise NotImplementedError("proximal_conjugate is not provided for this sKL.")


# =========================
# Numba backend
# =========================
if has_numba:

    @njit(parallel=True, fastmath=False)
    def _skl_val_scalar(x, b, eta, shift, mask):
        acc = 0.0
        n = x.size
        domain_ok = True
        if mask is None:
            for i in prange(n):
                s = x.flat[i] + eta.flat[i] + shift
                if s <= 0.0:
                    domain_ok = False
                else:
                    f = b.flat[i] + shift
                    acc += f * np.log(f) - f * np.log(s) - f + s
        else:
            for i in prange(n):
                if mask.flat[i] > 0.0:
                    s = x.flat[i] + eta.flat[i] + shift
                    if s <= 0.0:
                        domain_ok = False
                    else:
                        f = b.flat[i] + shift
                        acc += f * np.log(f) - f * np.log(s) - f + s
        if not domain_ok:
            return np.inf
        return acc

    @njit(parallel=True, fastmath=False)
    def _skl_val_array(x, b, eta, shift_arr, mask):
        acc = 0.0
        n = x.size
        domain_ok = True
        if mask is None:
            for i in prange(n):
                s = x.flat[i] + eta.flat[i] + shift_arr.flat[i]
                if s <= 0.0:
                    domain_ok = False
                else:
                    f = b.flat[i] + shift_arr.flat[i]
                    acc += f * np.log(f) - f * np.log(s) - f + s
        else:
            for i in prange(n):
                if mask.flat[i] > 0.0:
                    s = x.flat[i] + eta.flat[i] + shift_arr.flat[i]
                    if s <= 0.0:
                        domain_ok = False
                    else:
                        f = b.flat[i] + shift_arr.flat[i]
                        acc += f * np.log(f) - f * np.log(s) - f + s
        if not domain_ok:
            return np.inf
        return acc

    @njit(parallel=True, fastmath=False)
    def _skl_grad_scalar(x, b, eta, shift, mask, out):
        n = x.size
        if mask is None:
            for i in prange(n):
                s = x.flat[i] + eta.flat[i] + shift
                if s <= 0.0:
                    # signal invalid domain by writing NaNs
                    out.flat[i] = np.nan
                else:
                    out.flat[i] = 1.0 - (b.flat[i] + shift) / s
        else:
            for i in prange(n):
                if mask.flat[i] > 0.0:
                    s = x.flat[i] + eta.flat[i] + shift
                    if s <= 0.0:
                        out.flat[i] = np.nan
                    else:
                        out.flat[i] = 1.0 - (b.flat[i] + shift) / s
                else:
                    out.flat[i] = 0.0

    @njit(parallel=True, fastmath=False)
    def _skl_grad_array(x, b, eta, shift_arr, mask, out):
        n = x.size
        if mask is None:
            for i in prange(n):
                s = x.flat[i] + eta.flat[i] + shift_arr.flat[i]
                if s <= 0.0:
                    out.flat[i] = np.nan
                else:
                    out.flat[i] = 1.0 - (b.flat[i] + shift_arr.flat[i]) / s
        else:
            for i in prange(n):
                if mask.flat[i] > 0.0:
                    s = x.flat[i] + eta.flat[i] + shift_arr.flat[i]
                    if s <= 0.0:
                        out.flat[i] = np.nan
                    else:
                        out.flat[i] = 1.0 - (b.flat[i] + shift_arr.flat[i]) / s
                else:
                    out.flat[i] = 0.0


class ShiftedKullbackLeibler_numba(ShiftedKullbackLeibler):

    def __call__(self, x):
        x_np = x.as_array()
        if self._shift_arr is None:
            shift = self._shift_scalar
            return _skl_val_scalar(x_np, self._b_np, self._eta_np, shift, self._mask_np)
        else:
            shift_arr = self._shift_arr
            return _skl_val_array(x_np, self._b_np, self._eta_np, shift_arr, self._mask_np)

    def gradient(self, x, out=None):
        if out is None:
            out = x * 0
        out_np = out.as_array()
        x_np = x.as_array()

        if self._shift_arr is None:
            _skl_grad_scalar(x_np, self._b_np, self._eta_np, self._shift_scalar, self._mask_np, out_np)
        else:
            _skl_grad_array(x_np, self._b_np, self._eta_np, self._shift_arr, self._mask_np, out_np)

        # Check for domain violations signalled by NaNs
        if np.isnan(out_np).any():
            raise ValueError("Gradient undefined where s ≤ 0 on the masked support.")
        out.fill(out_np)
        return out

    def convex_conjugate(self, x):
        raise NotImplementedError("convex_conjugate is not provided for this sKL.")
    def proximal(self, x, tau, out=None):
        raise NotImplementedError("proximal is not provided for this sKL.")
    def proximal_conjugate(self, x, tau, out=None):
        raise NotImplementedError("proximal_conjugate is not provided for this sKL.")