# Shifted Data / Shifted Poisson (SD/SP) KL data term following
# Ahn & Fessler, "Globally convergent image reconstruction for emission tomography
# using relaxed ordered subsets algorithms", IEEE TMI 2003.
#
# The SD/SP reparameterisation shifts both the measured counts and the Poisson
# mean to keep the logarithm arguments strictly positive while preserving the
# minimiser of the original Poisson negative log-likelihood.
#
# Objective (up to an additive constant independent of x):
#   F(x) = Σ_i [ μ̂_i - f̂_i * ln μ̂_i + f̂_i * ln f̂_i - f̂_i ],
#   with μ̂ := x + additive + shift,   f̂ := f + shift.
# The gradient is
#   ∇F = 1 - f̂ / μ̂,
# defined wherever μ̂ > 0 on the (optional) mask support.

import logging
from numbers import Number

import numpy as np
from cil.optimisation.functions import Function

log = logging.getLogger(__name__)

try:
    from numba import njit, prange

    has_numba = True
except Exception:  # pragma: no cover - defensive
    has_numba = False
    njit = None
    prange = range


class ShiftedDataShiftedPoissonKullbackLeibler(Function):
    r"""
    Shifted-data / shifted-Poisson KL data term as in Ahn & Fessler (2003).

    Parameters
    ----------
    f           : DataContainer; measured counts (may be negative).
    additive    : DataContainer or scalar; additive background term.
    shift       : DataContainer or scalar; element-wise non-negative shift ρ such that
                  f + ρ ≥ 0 and (x + additive + ρ) > 0 on the mask.
    mask        : DataContainer or None; include entries where mask > 0.
    backend     : {'numba', 'numpy'}, optional; default 'numba'.
    count_floor : float, optional; strictly positive floor for f̂ inside the logarithm.
    """

    def __new__(cls, *args, backend="numba", **kwargs):
        cls.backend = backend
        if backend == "numba":
            if not has_numba:
                raise ValueError("Numba is not installed; use backend='numpy'.")
            log.info("ShiftedDataShiftedPoissonKullbackLeibler: using Numba backend.")
            return super().__new__(ShiftedDataShiftedPoissonKullbackLeibler_numba)
        else:
            log.info("ShiftedDataShiftedPoissonKullbackLeibler: using NumPy backend.")
            return super().__new__(ShiftedDataShiftedPoissonKullbackLeibler_numpy)

    def __init__(
        self,
        f,
        additive,
        shift,
        mask=None,
        backend="numba",
        count_floor=1e-8,
    ):
        self.f = f
        self.mask = mask

        self._f_np = np.array(self.f.as_array(), dtype=np.float64, copy=True)
        self._mask_np = None if mask is None else np.array(mask.as_array(), copy=False)
        self._mask_bool = None if self._mask_np is None else (self._mask_np > 0.0)

        self.count_floor = float(count_floor) if count_floor is not None else 0.0
        if self.count_floor < 0.0:
            raise ValueError(
                "ShiftedDataShiftedPoissonKullbackLeibler: 'count_floor' must be nonnegative."
            )

        self._add_dc = None
        self._shift_dc = None
        self.additive = additive
        self.shift = shift  # also builds caches

        # Validate f̂ = f + shift ≥ 0 on mask support
        if self._mask_bool is None:
            bad = np.any(self._f_shift_np < 0.0)
        else:
            bad = np.any(self._mask_bool & (self._f_shift_np < 0.0))
        if bad:
            raise ValueError(
                "ShiftedDataShiftedPoissonKullbackLeibler: require (f + shift) ≥ 0 on the masked support."
            )

        super().__init__(L=None)

    # --- properties ---

    @property
    def additive(self):
        return self._add_dc

    @additive.setter
    def additive(self, value):
        dc, arr = self._coerce_datacontainer(value, "additive")
        self._add_dc = dc
        self._add_np = arr

    @property
    def shift(self):
        return self._shift_dc

    @shift.setter
    def shift(self, value):
        dc, arr = self._coerce_datacontainer(value, "shift")
        if np.any(arr < 0.0):
            raise ValueError("ShiftedDataShiftedPoissonKullbackLeibler: 'shift' must be nonnegative.")
        self._shift_dc = dc
        self._shift_np = arr
        self._update_shift_cache()

    # --- helpers ---

    def _coerce_datacontainer(self, value, name):
        if isinstance(value, Number):
            value = self.f * 0 + float(value)
        if hasattr(value, "as_array"):
            arr = np.array(value.as_array(), dtype=np.float64, copy=True)
        else:
            arr = np.array(value, dtype=np.float64, copy=True)
        if arr.shape != self._f_np.shape:
            raise ValueError(
                f"ShiftedDataShiftedPoissonKullbackLeibler: '{name}' must match the shape of measured data."
            )
        return value, arr

    def _update_shift_cache(self):
        self._f_shift_np = self._f_np + self._shift_np
        floor = self.count_floor if self.count_floor > 0.0 else np.finfo(np.float64).tiny
        self._log_f_shift_np = np.log(np.maximum(self._f_shift_np, floor))

    def _mu_raw_and_mu_shift(self, x):
        mu_raw = np.array(x.as_array(), dtype=np.float64, copy=True)
        mu_raw += self._add_np
        mu_shift = mu_raw + self._shift_np
        return mu_raw, mu_shift

    def _mask_ok(self, mu_shift):
        if self._mask_bool is None:
            mask = np.ones_like(mu_shift, dtype=bool)
        else:
            mask = self._mask_bool
        return mask, np.all(mu_shift[mask] > 0.0)

    # --- Hessian support ---

    def multiply_with_Hessian(self, x, direction, out=None):
        mu_raw, mu_shift = self._mu_raw_and_mu_shift(x)
        mask, ok = self._mask_ok(mu_shift)
        if not ok:
            raise ValueError(
                "multiply_with_Hessian undefined where (x + additive + shift) ≤ 0 on the masked support."
            )

        weights = np.zeros_like(mu_shift, dtype=np.float64)
        positive = np.logical_and(mask, mu_shift > 0.0)
        weights[positive] = self._f_shift_np[positive] / (mu_shift[positive] ** 2)

        dir_arr = direction.as_array()
        result = np.array(dir_arr, dtype=np.float64, copy=True)
        result *= weights

        if out is None:
            out = direction.get_uniform_copy(0)
        out.fill(result.astype(dir_arr.dtype, copy=False))
        return out


class ShiftedDataShiftedPoissonKullbackLeibler_numpy(ShiftedDataShiftedPoissonKullbackLeibler):
    """NumPy backend."""

    def _mu_raw_and_mu_shift(self, x):
        mu_raw = x.as_array().astype(np.float64, copy=True)
        mu_raw += self._add_np
        mu_shift = mu_raw + self._shift_np
        return mu_raw, mu_shift

    def _mask_ok(self, mu_shift):
        if self._mask_bool is None:
            m = np.ones_like(mu_shift, dtype=bool)
        else:
            m = self._mask_bool
        return m, np.all(mu_shift[m] > 0.0)

    def __call__(self, x):
        mu_raw, mu_shift = self._mu_raw_and_mu_shift(x)
        m, ok = self._mask_ok(mu_shift)
        if not ok:
            return np.inf
        mu = mu_shift[m]
        f_shift = self._f_shift_np[m]
        return np.sum(mu + f_shift * (self._log_f_shift_np[m] - np.log(mu)) - f_shift)

    def gradient(self, x, out=None):
        mu_raw, mu_shift = self._mu_raw_and_mu_shift(x)
        m, ok = self._mask_ok(mu_shift)
        if not ok:
            raise ValueError(
                "Gradient undefined where (x + additive + shift) ≤ 0 on the masked support."
            )

        g = np.zeros_like(mu_raw)
        g[m] = 1.0 - (self._f_shift_np[m] / mu_shift[m])

        if out is None:
            out = x * 0
        out.fill(g)
        return out

    def convex_conjugate(self, x):
        raise NotImplementedError(
            "convex_conjugate is not provided for ShiftedDataShiftedPoissonKullbackLeibler."
        )

    def proximal(self, x, tau, out=None):
        raise NotImplementedError(
            "proximal is not provided for ShiftedDataShiftedPoissonKullbackLeibler."
        )

    def proximal_conjugate(self, x, tau, out=None):
        raise NotImplementedError(
            "proximal_conjugate is not provided for ShiftedDataShiftedPoissonKullbackLeibler."
        )


if has_numba:

    @njit(parallel=True, fastmath=False)
    def _sdsp_val(x, additive, shift, f_shift, log_f_shift, mask_bool):
        acc = 0.0
        bad = 0
        n = x.size
        if mask_bool is None:
            for i in prange(n):
                mu = x.flat[i] + additive.flat[i] + shift.flat[i]
                if mu <= 0.0:
                    bad += 1
                else:
                    f_s = f_shift.flat[i]
                    acc += mu + f_s * (log_f_shift.flat[i] - np.log(mu)) - f_s
        else:
            for i in prange(n):
                if mask_bool.flat[i]:
                    mu = x.flat[i] + additive.flat[i] + shift.flat[i]
                    if mu <= 0.0:
                        bad += 1
                    else:
                        f_s = f_shift.flat[i]
                        acc += mu + f_s * (log_f_shift.flat[i] - np.log(mu)) - f_s
        if bad > 0:
            return np.inf
        return acc

    @njit(parallel=True, fastmath=False)
    def _sdsp_grad(x, additive, shift, f_shift, mask_bool, out):
        n = x.size
        if mask_bool is None:
            for i in prange(n):
                mu = x.flat[i] + additive.flat[i] + shift.flat[i]
                if mu <= 0.0:
                    out.flat[i] = np.nan
                else:
                    out.flat[i] = 1.0 - f_shift.flat[i] / mu
        else:
            for i in prange(n):
                if mask_bool.flat[i]:
                    mu = x.flat[i] + additive.flat[i] + shift.flat[i]
                    if mu <= 0.0:
                        out.flat[i] = np.nan
                    else:
                        out.flat[i] = 1.0 - f_shift.flat[i] / mu
                else:
                    out.flat[i] = 0.0


class ShiftedDataShiftedPoissonKullbackLeibler_numba(ShiftedDataShiftedPoissonKullbackLeibler):
    """Numba backend."""

    def __call__(self, x):
        x_np = x.as_array()
        return _sdsp_val(
            x_np,
            self._add_np,
            self._shift_np,
            self._f_shift_np,
            self._log_f_shift_np,
            self._mask_bool,
        )

    def gradient(self, x, out=None):
        if out is None:
            out = x * 0
        out_np = out.as_array()
        x_np = x.as_array()

        _sdsp_grad(
            x_np,
            self._add_np,
            self._shift_np,
            self._f_shift_np,
            self._mask_bool,
            out_np,
        )

        if np.isnan(out_np).any():
            raise ValueError(
                "Gradient undefined where (x + additive + shift) ≤ 0 on the masked support."
            )
        out.fill(out_np)
        return out

    def convex_conjugate(self, x):
        raise NotImplementedError(
            "convex_conjugate is not provided for ShiftedDataShiftedPoissonKullbackLeibler."
        )

    def proximal(self, x, tau, out=None):
        raise NotImplementedError(
            "proximal is not provided for ShiftedDataShiftedPoissonKullbackLeibler."
        )

    def proximal_conjugate(self, x, tau, out=None):
        raise NotImplementedError(
            "proximal_conjugate is not provided for ShiftedDataShiftedPoissonKullbackLeibler."
        )

