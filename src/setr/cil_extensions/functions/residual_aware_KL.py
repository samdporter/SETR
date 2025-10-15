# residual_aware_KL.py

import numpy as np
from numbers import Number
import logging

from cil.optimisation.functions import Function

log = logging.getLogger(__name__)

try:
    from numba import njit, prange
    has_numba = True
except Exception:
    has_numba = False
    njit = None
    prange = range
    
class ResidualAwareKullbackLeibler(Function):
    r"""
    Residual-aware KL data term using the Jordan decomposition of the residual.

    Let the raw prediction before residual correction be
        μ_raw(x) := x + additive,
    and split the residual r into its Jordan components
        r = r⁺ - r⁻,     with r⁺ = max(r, 0) and r⁻ = max(-r, 0).

    The corrected Poisson expectation and effective data are then
        μ_eff := μ_raw + r⁺,     f_eff := f + r⁻,
    which ensures both quantities remain non-negative.

    Up to an additive constant independent of x, the objective reads
        F(x) = Σ_i [ μ_eff,i - f_eff,i * ln μ_eff,i + f_eff,i * ln f_eff,i - f_eff,i ].

    The gradient with respect to x (note that dμ_eff/dx = 1 on the mask) is
        ∇F = 1 - f_eff / μ_eff.

    Parameters
    ----------
    f           : DataContainer (nonnegative on mask); measured data.
    additive    : DataContainer or scalar; additive term b combined with x inside the log.
    residual    : DataContainer or scalar; residual correction r (may be negative).
    mask        : DataContainer or None; include entries where mask > 0.
    backend     : {'numba','numpy'}, optional; default 'numba'.
    count_floor : float, optional; minimum positive value used for f inside the logarithm
                  to avoid log(0). Defaults to 1e-8.
    """

    def __new__(cls, *args, backend='numba', **kwargs):
        cls.backend = backend
        if backend == 'numba':
            if not has_numba:
                raise ValueError("Numba is not installed; use backend='numpy'.")
            log.info("ResidualAwareKullbackLeibler: using Numba backend.")
            return super().__new__(ResidualAwareKullbackLeibler_numba)
        else:
            log.info("ResidualAwareKullbackLeibler: using NumPy backend.")
            return super().__new__(ResidualAwareKullbackLeibler_numpy)

    def __init__(self, f, additive, residual=None, mask=None, backend='numba', count_floor=1e-8):
        self.f = f
        self.mask = mask

        self._f_np = np.array(self.f.as_array(), dtype=np.float64, copy=True)
        self._mask_np = None if mask is None else np.array(mask.as_array(), copy=False)
        self._mask_bool = None if self._mask_np is None else (self._mask_np > 0.0)

        self.count_floor = float(count_floor) if count_floor is not None else 0.0
        if self.count_floor < 0.0:
            raise ValueError("ResidualAwareKullbackLeibler: 'count_floor' must be nonnegative.")

        # Validate nonnegativity of f on masked support
        if self._mask_bool is None:
            bad = np.any(self._f_np < 0.0)
        else:
            if self._mask_bool.shape != self._f_np.shape:
                raise ValueError(
                    "ResidualAwareKullbackLeibler: 'mask' must share the same shape as data."
                )
            bad = np.any(self._mask_bool & (self._f_np < 0.0))
        if bad:
            raise ValueError("ResidualAwareKullbackLeibler: require f ≥ 0 on the masked support.")

        if residual is None:
            residual = f * 0  # preserves shape/container type

        self._add_dc = None
        self._res_dc = None
        self.additive = additive
        self.residual = residual  # also builds residual cache

        super().__init__(L=None)

    # --- properties for additive and residual, with coercion to array cache ---

    @property
    def additive(self):
        return self._add_dc

    @additive.setter
    def additive(self, value):
        dc, arr = self._coerce_datacontainer(value, "additive")
        self._add_dc = dc
        self._add_np = arr

    @property
    def residual(self):
        return self._res_dc

    @residual.setter
    def residual(self, value):
        dc, arr = self._coerce_datacontainer(value, "residual")
        self._res_dc = dc
        self._res_np = arr
        self._update_residual_cache()

    # --- helpers ---

    def _update_residual_cache(self):
        # Jordan split
        self._res_plus_np = np.maximum(self._res_np, 0.0)
        self._res_minus_np = np.maximum(-self._res_np, 0.0)
        # Effective data and its log with flooring
        self._f_eff_np = self._f_np + self._res_minus_np
        floor = self.count_floor if self.count_floor > 0.0 else np.finfo(np.float64).tiny
        self._log_f_eff_np = np.log(np.maximum(self._f_eff_np, floor))

    def _coerce_datacontainer(self, value, name):
        if value is None:
            value = self.f * 0
        if isinstance(value, Number):
            value = self.f * 0 + float(value)
        if hasattr(value, "as_array"):
            arr = np.array(value.as_array(), dtype=np.float64, copy=True)
        else:
            arr = np.array(value, dtype=np.float64, copy=True)
        if arr.shape != self._f_np.shape:
            raise ValueError(
                f"ResidualAwareKullbackLeibler: '{name}' must match the shape of measured data."
            )
        return value, arr

    def _mu_raw_and_mu_eff(self, x):
        """Return μ_raw = x + additive and μ_eff = μ_raw + r⁺."""
        mu_raw = np.array(x.as_array(), dtype=np.float64, copy=True)
        mu_raw += self._add_np
        mu_eff = mu_raw + self._res_plus_np
        return mu_raw, mu_eff

    def _mask_ok(self, mu_eff):
        if self._mask_bool is None:
            mask = np.ones_like(mu_eff, dtype=bool)
        else:
            mask = self._mask_bool
        return mask, np.all(mu_eff[mask] > 0.0)

    # --- Hessian support ---

    def multiply_with_Hessian(self, x, direction, out=None):
        """Apply the diagonal Hessian to ``direction`` at current point ``x``."""
        mu_raw, mu_eff = self._mu_raw_and_mu_eff(x)
        mask, ok = self._mask_ok(mu_eff)
        if not ok:
            raise ValueError("multiply_with_Hessian undefined where μ_eff ≤ 0 on the masked support.")

        weights = np.zeros_like(mu_eff, dtype=np.float64)
        positive = np.logical_and(mask, mu_eff > 0.0)
        weights[positive] = self._f_eff_np[positive] / (mu_eff[positive] ** 2)

        dir_arr = direction.as_array()
        result = np.array(dir_arr, dtype=np.float64, copy=True)
        result *= weights

        if out is None:
            out = direction.get_uniform_copy(0)
        out.fill(result.astype(dir_arr.dtype, copy=False))
        return out


class ResidualAwareKullbackLeibler_numpy(ResidualAwareKullbackLeibler):
    """NumPy backend."""

    def _mu_raw_and_mu_eff(self, x):
        # μ_raw(x) = x + additive ; μ_eff = μ_raw + r⁺
        mu_raw = x.as_array().astype(np.float64, copy=True)
        mu_raw += self._add_np
        mu_eff = mu_raw + self._res_plus_np
        return mu_raw, mu_eff

    def _mask_ok(self, mu_eff):
        if self._mask_bool is None:
            m = np.ones_like(mu_eff, dtype=bool)
        else:
            m = self._mask_bool
        return m, np.all(mu_eff[m] > 0.0)

    def __call__(self, x):
        mu_raw, mu_eff = self._mu_raw_and_mu_eff(x)
        m, ok = self._mask_ok(mu_eff)
        if not ok:
            return np.inf
        mu = mu_eff[m]
        f_eff = self._f_eff_np[m]
        # Use precomputed log(f_eff)
        return np.sum(mu + f_eff * (self._log_f_eff_np[m] - np.log(mu)) - f_eff)

    def gradient(self, x, out=None):
        mu_raw, mu_eff = self._mu_raw_and_mu_eff(x)
        m, ok = self._mask_ok(mu_eff)
        if not ok:
            raise ValueError("Gradient undefined where μ_eff ≤ 0 on the masked support.")

        g = np.zeros_like(mu_raw)
        g[m] = 1.0 - (self._f_eff_np[m] / mu_eff[m])

        if out is None:
            out = x * 0
        out.fill(g)
        return out

    def convex_conjugate(self, x):
        raise NotImplementedError("convex_conjugate is not provided for this residual-corrected KL.")

    def proximal(self, x, tau, out=None):
        raise NotImplementedError("proximal is not provided for this residual-corrected KL.")

    def proximal_conjugate(self, x, tau, out=None):
        raise NotImplementedError("proximal_conjugate is not provided for this residual-corrected KL.")


if has_numba:

    @njit(parallel=True, fastmath=False)
    def _rckl_val(x, additive, res_plus, f_eff, log_f_eff, mask_bool):
        """
        Parallel-safe value kernel:
        - Uses an integer reduction 'bad' to avoid races when checking μ_eff>0.
        - Expects mask_bool to be either None or a boolean array; when None, all entries are used.
        """
        acc = 0.0
        bad = 0
        n = x.size

        if mask_bool is None:
            for i in prange(n):
                mu = x.flat[i] + additive.flat[i] + res_plus.flat[i]
                if mu <= 0.0:
                    bad += 1
                else:
                    f_eff_i = f_eff.flat[i]
                    acc += mu + f_eff_i * (log_f_eff.flat[i] - np.log(mu)) - f_eff_i
        else:
            for i in prange(n):
                if mask_bool.flat[i]:
                    mu = x.flat[i] + additive.flat[i] + res_plus.flat[i]
                    if mu <= 0.0:
                        bad += 1
                    else:
                        f_eff_i = f_eff.flat[i]
                        acc += mu + f_eff_i * (log_f_eff.flat[i] - np.log(mu)) - f_eff_i
        if bad > 0:
            return np.inf
        return acc

    @njit(parallel=True, fastmath=False)
    def _rckl_grad(x, additive, res_plus, f_eff, mask_bool, out):
        """
        Parallel-safe gradient kernel.
        Writes NaN where μ_eff ≤ 0 on the (boolean) masked support, 0 off-mask.
        """
        n = x.size
        if mask_bool is None:
            for i in prange(n):
                mu = x.flat[i] + additive.flat[i] + res_plus.flat[i]
                if mu <= 0.0:
                    out.flat[i] = np.nan
                else:
                    out.flat[i] = 1.0 - f_eff.flat[i] / mu
        else:
            for i in prange(n):
                if mask_bool.flat[i]:
                    mu = x.flat[i] + additive.flat[i] + res_plus.flat[i]
                    if mu <= 0.0:
                        out.flat[i] = np.nan
                    else:
                        out.flat[i] = 1.0 - f_eff.flat[i] / mu
                else:
                    out.flat[i] = 0.0


class ResidualAwareKullbackLeibler_numba(ResidualAwareKullbackLeibler):
    """Numba backend."""

    def __call__(self, x):
        x_np = x.as_array()
        return _rckl_val(
            x_np,
            self._add_np,
            self._res_plus_np,
            self._f_eff_np,
            self._log_f_eff_np,
            self._mask_bool,
        )

    def gradient(self, x, out=None):
        if out is None:
            out = x * 0
        out_np = out.as_array()
        x_np = x.as_array()

        _rckl_grad(
            x_np,
            self._add_np,
            self._res_plus_np,
            self._f_eff_np,
            self._mask_bool,
            out_np,
        )

        if np.isnan(out_np).any():
            raise ValueError("Gradient undefined where μ_eff ≤ 0 on the masked support.")
        out.fill(out_np)
        return out

    def convex_conjugate(self, x):
        raise NotImplementedError("convex_conjugate is not provided for this residual-corrected KL.")

    def proximal(self, x, tau, out=None):
        raise NotImplementedError("proximal is not provided for this residual-corrected KL.")

    def proximal_conjugate(self, x, tau, out=None):
        raise NotImplementedError("proximal_conjugate is not provided for this residual-corrected KL.")
