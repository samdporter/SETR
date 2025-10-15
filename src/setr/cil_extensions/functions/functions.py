import numpy as np
from cil.optimisation.functions import Function

class BlockIndicatorBox(Function):
    def __init__(self, lower=0, upper=np.inf):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        # because we're using this as a projection, this should always return 0
        # se we'll be a bit cheeky and return 0.0 to save computation time
        # TODO: change this to work as an actual indicator function
        return 0.0

    def proximal(self, x, tau, out=None):
        if out is None:
            out = x.copy()
        x.maximum(self.lower, out=out)
        out.minimum(self.upper, out=out)
        return out
    
    
def patch_KL_with_nonnegative(eps=1e-8, mode="subgrad", alpha=1e-2, persist_eta=False):
    """
    Monkey-patch CIL's KullbackLeibler *and* ShiftedKullbackLeibler so that the
    log-argument is floored inside value/grad.

        For KL : a(x) := x + η
        For sKL: a(x) := x + η + τ

    We replace a(x) by ã(x) = max(a(x), eps) (on mask>0), and apply a chain factor
    to the gradient:
        - 'subgrad': 1{a>eps} else 0
        - 'leaky'  : 1{a>eps} + α·1{a<=eps}
        - 'ste'    : 1 everywhere (straight–through)

    If persist_eta=True, we add δ := ã - a to η in-place so future calls
    see the corrected mean (s = x+η(+τ) ≥ eps). We save self.eta_orig on first use
    and expose self.reset_eta() to restore it.

    Notes
    -----
    * Works for NumPy and Numba backends of both KL and sKL.
    * For sKL we do **not** change τ; we floor the log-argument s=x+η+τ.
    """
    import numpy as _np
    from numbers import Number as _Number

    # --- locate classes to patch (silently skip those not imported) ---
    _classes = []
    for _name in (
        "KullbackLeibler_numpy", "KullbackLeibler_numba",
        "ShiftedKullbackLeibler_numpy", "ShiftedKullbackLeibler_numba"
    ):
        _cls = globals().get(_name, None)
        if _cls is not None:
            _classes.append(_cls)
    if not _classes:
        raise RuntimeError("Import the KL/sKL classes before patching.")

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if mode not in {"subgrad", "leaky", "ste"}:
        raise ValueError("mode must be one of {'subgrad','leaky','ste'}.")
    if mode == "leaky" and not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1] when mode='leaky'.")

    # ---- helpers shared by KL and sKL ----
    def _mask_arr(self):
        if getattr(self, "mask", None) is None:
            return None
        m = self.mask
        return m if isinstance(m, _np.ndarray) else m.as_array()

    def _ensure_eta_bookkeeping(self):
        if not hasattr(self, "eta_orig"):
            try:
                self.eta_orig = self.eta.copy()
            except AttributeError:
                self.eta_orig = self.eta * 1

            def _reset_eta(_self):
                _self.eta.fill(_self.eta_orig.as_array())
                # keep possible cached numpy views in sync
                for _attr in ("eta_np", "_eta_np"):
                    if hasattr(_self, _attr):
                        setattr(_self, _attr, _self.eta.as_array())
            self.reset_eta = _reset_eta

    def _tau_kind_and_value(self):
        """
        Return ('none', 0) for KL; otherwise ('scalar', t) or ('array', t_arr) for sKL.
        We try several attribute names to be robust across implementations.
        """
        # Preferred attributes (from the provided sKL implementation)
        if hasattr(self, "_tau_arr") and getattr(self, "_tau_arr") is not None:
            return "array", getattr(self, "_tau_arr")
        if hasattr(self, "_tau_scalar") and getattr(self, "_tau_scalar") is not None:
            return "scalar", float(getattr(self, "_tau_scalar"))

        # Fallback to a public 'tau' attribute if present
        if hasattr(self, "tau"):
            t = getattr(self, "tau")
            if isinstance(t, _Number):
                return "scalar", float(t)
            else:
                try:
                    return "array", t.as_array()
                except Exception:
                    pass
        return "none", 0.0  # KL case

    def _log_arg_and_xprime(self, x):
        """
        Compute:
          v      := a(x) = x+η (+τ if present)
          v_clip := max(v, eps) on mask>0
          xprime such that the original call sees its own a(xprime) == v_clip
        """
        x_arr   = x.as_array()
        eta_arr = self.eta.as_array()
        m       = _mask_arr(self)

        tau_kind, tau_val = _tau_kind_and_value(self)
        if tau_kind == "array":
            v = x_arr + eta_arr + tau_val
        elif tau_kind == "scalar":
            v = x_arr + eta_arr + float(tau_val)
        else:
            v = x_arr + eta_arr

        if m is None:
            v_clip = _np.maximum(v, float(eps))
        else:
            v_clip = _np.where(m > 0, _np.maximum(v, float(eps)), v)

        if persist_eta:
            _ensure_eta_bookkeeping(self)
            delta = v_clip - v
            if (delta > 0).any():
                self.eta.fill(eta_arr + delta)
                # keep possible cached numpy views in sync
                for _attr in ("eta_np", "_eta_np"):
                    if hasattr(self, _attr):
                        setattr(self, _attr, self.eta.as_array())
                eta_arr = self.eta.as_array()
                # re-express v_clip exactly with updated eta (no need to max again)
                if tau_kind == "array":
                    v_clip = x_arr + eta_arr + tau_val
                elif tau_kind == "scalar":
                    v_clip = x_arr + eta_arr + float(tau_val)
                else:
                    v_clip = x_arr + eta_arr

        # Build x' so that original backend sees its own a(x') == v_clip
        xprime = x * 0
        if tau_kind == "array":
            xprime.fill(v_clip - eta_arr - tau_val)
        elif tau_kind == "scalar":
            xprime.fill(v_clip - eta_arr - float(tau_val))
        else:
            xprime.fill(v_clip - eta_arr)
        return xprime, v

    def _chain_factor(v):
        if mode == "ste":
            return 1.0
        gt = (v > float(eps)).astype(_np.float32)
        if mode == "subgrad":
            return gt
        return gt + (1.0 - gt) * float(alpha)  # leaky

    # ---- patched methods ----
    def _patched_call(self, x):
        xprime, _ = _log_arg_and_xprime(self, x)
        return self.__class__._orig_call(self, xprime)

    def _patched_grad(self, x, out=None):
        xprime, v = _log_arg_and_xprime(self, x)
        g = self.__class__._orig_grad(self, xprime, out=out)
        cf = _chain_factor(v)
        if isinstance(cf, float):  # STE fast path
            return g
        g_arr = g.as_array()
        g_arr *= cf
        g.fill(g_arr)
        return g

    # ---- apply once per class ----
    for _Cls in _classes:
        if not hasattr(_Cls, "_orig_call"):
            _Cls._orig_call = _Cls.__call__
            _Cls._orig_grad = _Cls.gradient
            _Cls.__call__   = _patched_call
            _Cls.gradient   = _patched_grad


_KL_HESSIAN_PATCHED = False


def ensure_kl_hessian_support():
    """
    Monkey-patch CIL's KullbackLeibler variants and OperatorCompositionFunction with
    ``multiply_with_Hessian`` implementations.

    The patched Hessian corresponds to the Poisson log-likelihood:
        H(z) = diag(b / (z + eta)^2)
    and is propagated through operator compositions as:
        A^T H(Ax) A v
    """
    global _KL_HESSIAN_PATCHED
    if _KL_HESSIAN_PATCHED:
        return

    try:
        from cil.optimisation.functions.KullbackLeibler import (
            KullbackLeibler_numba,
            KullbackLeibler_numpy,
        )
        from cil.optimisation.functions import OperatorCompositionFunction
    except (ImportError, OSError) as exc:  # pragma: no cover - external dependency missing
        raise RuntimeError(
            "Failed to import CIL KullbackLeibler/OperatorCompositionFunction for Hessian patching."
        ) from exc

    def _mask_array(mask):
        if mask is None:
            return None
        if hasattr(mask, "as_array"):
            return mask.as_array()
        return mask

    def _kl_multiply_with_Hessian(self, x, direction, out=None):
        x_arr = x.as_array()
        eta_arr = self.eta.as_array()
        mu = x_arr + eta_arr

        b_arr = self.b.as_array()
        weights = np.zeros_like(mu, dtype=np.float64)

        mask_arr = _mask_array(getattr(self, "mask", None))
        valid = mu > 0.0
        if mask_arr is not None:
            valid = np.logical_and(valid, mask_arr > 0)
        weights[valid] = b_arr[valid] / (mu[valid] ** 2)

        direction_arr = direction.as_array()
        result = np.array(direction_arr, dtype=np.float64, copy=True)
        result *= weights

        if out is None:
            out = direction.get_uniform_copy(0)
        out.fill(result.astype(direction_arr.dtype, copy=False))
        return out

    for _cls in (KullbackLeibler_numpy, KullbackLeibler_numba):
        if not hasattr(_cls, "multiply_with_Hessian"):
            _cls.multiply_with_Hessian = _kl_multiply_with_Hessian

    if not hasattr(OperatorCompositionFunction, "multiply_with_Hessian"):

        def _ocf_multiply_with_Hessian(self, x, direction, out=None):
            if not hasattr(self.function, "multiply_with_Hessian"):
                raise AttributeError(
                    f"{type(self.function).__name__} does not implement multiply_with_Hessian."
                )

            range_geom = self.operator.range_geometry()

            tmp_x = range_geom.allocate()
            self.operator.direct(x, out=tmp_x)

            tmp_v = range_geom.allocate()
            self.operator.direct(direction, out=tmp_v)

            hess_times_v = range_geom.allocate()
            hess_times_v = self.function.multiply_with_Hessian(tmp_x, tmp_v, out=hess_times_v)

            return self.operator.adjoint(hess_times_v, out=out)

        OperatorCompositionFunction.multiply_with_Hessian = _ocf_multiply_with_Hessian

    _KL_HESSIAN_PATCHED = True
