# stabilized_schatten_norm_gpu_stable.py
import math
import numpy as np
import torch
from cil.optimisation.functions import Function

from .common import (
    to_tensor,
    l1_norm,
    l1_norm_prox,
    l2_norm,
    l2_norm_prox,
    charbonnier,
    charbonnier_grad,
    charbonnier_hessian_surrogate,
    charbonnier_hessian_diag,
    fair,
    fair_grad,
    fair_hessian_surrogate,
    fair_hessian_diag,
    perona_malik,
    perona_malik_grad,
    perona_malik_hessian_surrogate,
    perona_malik_hessian_diag,
    nothing,
    nothing_grad,
    nothing_hessian_diag,
    get_mask,  # Use original tail masking function
)
from .small_eig import (
    eigenvalsh_2x2,
    eigenvecsh_2x2,
    eigenvalsh_3x3_cardano,
    eigenvecsh_3x3_cardano
)

from .svd_free_hessian import (
    hessian_components_hybrid_small,
    _kappa_proxy_3x3,
    _log_kappa_proxy_3x3,
    _dtype_cond_threshold,
    _trace_normalize
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_svd_with_fallback(x, condition_threshold=None):
    """
    Per-voxel hybrid SVD: analytic 2x2 always; for 3x3 do Cardano+vectors
    where safe, and SVD elsewhere. Returns (U, S, Vh, used_hybrid_flag).
    """
    try:
        *lead, m, n = x.shape
        r = min(m, n)
        Xb = x.reshape(-1, m, n).contiguous()
        B  = Xb.shape[0]

        Uout = torch.empty(B, m, r, dtype=x.dtype, device=x.device)
        Sout = torch.empty(B, r,    dtype=x.dtype, device=x.device)
        Vhout= torch.empty(B, r, n, dtype=x.dtype, device=x.device)

        if m <= n:
            order = 1
            H = Xb @ Xb.transpose(1, 2)                       # (B, m, m)
        else:
            order = 0
            H = Xb.transpose(1, 2) @ Xb                       # (B, n, n)

        H = 0.5 * (H + H.transpose(1, 2))
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        if r == 2:
            # Analytic 2x2
            S2 = eigenvalsh_2x2(H)
            Bmat = eigenvecsh_2x2(H, S2)                      # eigenvectors (U if order=1, V if order=0)
            S = torch.sqrt(torch.clamp(S2, min=0.0))
            tiny = torch.finfo(S.dtype).eps * 100
            Sinv = torch.where(S > tiny, 1.0/S, torch.zeros_like(S))

            if order == 1:
                Uout = Bmat
                Vhout = (Bmat.transpose(1, 2) @ Xb) * Sinv[..., None]
            else:
                V = Bmat
                Vhout = V.transpose(1, 2)
                Uout  = (Xb @ V) * Sinv[..., None, :]
            Sout = S

        elif r == 3:
            # Hybrid per-voxel
            Hn, alpha = _trace_normalize(H)
            #cond_thr = _dtype_cond_threshold(x.dtype) if condition_threshold is None else condition_threshold
            #proxy = _kappa_proxy_3x3(Hn)
            #ok = (proxy < cond_thr) & torch.isfinite(proxy)   # (B,)
            proxy = _log_kappa_proxy_3x3(Hn)
            thr   = math.log(_dtype_cond_threshold(x.dtype) if condition_threshold is None
                                else float(condition_threshold))
            ok = proxy < thr      # finite by construction; no need for isfinite()


            if ok.all():
                # --- all analytic, no boolean indexing ---
                S2_all = eigenvalsh_3x3_cardano(Hn)                               # (B,3)
                B_all  = eigenvecsh_3x3_cardano(Hn, S2_all)                        # (B,3,3)
                S_all  = torch.sqrt(torch.clamp(S2_all * alpha.unsqueeze(-1), min=0.0))
                tiny   = torch.finfo(S_all.dtype).eps * 100
                Sinv   = torch.where(S_all > tiny, 1.0 / S_all, torch.zeros_like(S_all))

                if order == 1:
                    Uout  = B_all
                    Vhout = (B_all.transpose(1, 2) @ Xb) * Sinv[..., None]
                else:
                    V_all  = B_all
                    Vhout  = V_all.transpose(1, 2)
                    Uout   = (Xb @ V_all) * Sinv[..., None, :]
                Sout = S_all

            elif (~ok).all():
                # --- all SVD, no boolean indexing ---
                U_b, S_b, Vh_b = torch.linalg.svd(Xb, full_matrices=False)
                Uout, Sout, Vhout = U_b, S_b, Vh_b

            else:
                # --- mixed case: boolean index only once per side ---
                # analytic on ok
                S2_ok = eigenvalsh_3x3_cardano(Hn[ok])
                B_ok  = eigenvecsh_3x3_cardano(Hn[ok], S2_ok)
                S_ok  = torch.sqrt(torch.clamp(S2_ok * alpha[ok].unsqueeze(-1), min=0.0))
                tiny  = torch.finfo(S_ok.dtype).eps * 100
                Sinv_ok = torch.where(S_ok > tiny, 1.0 / S_ok, torch.zeros_like(S_ok))

                if order == 1:
                    Uout[ok]   = B_ok
                    Vhout[ok]  = (B_ok.transpose(1, 2) @ Xb[ok]) * Sinv_ok[..., None]
                else:
                    V_ok       = B_ok
                    Vhout[ok]  = V_ok.transpose(1, 2)
                    Uout[ok]   = (Xb[ok] @ V_ok) * Sinv_ok[..., None, :]
                Sout[ok] = S_ok

                # SVD on ~ok
                bad = ~ok
                if bad.any():
                    U_b, S_b, Vh_b = torch.linalg.svd(Xb[bad], full_matrices=False)
                    Uout[bad], Sout[bad], Vhout[bad] = U_b, S_b, Vh_b

        else:
            # r > 3: ordinary SVD
            U_b, S_b, Vh_b = torch.linalg.svd(Xb, full_matrices=False)
            Uout, Sout, Vhout = U_b, S_b, Vh_b

        # reshape back
        U = Uout.view(*lead, m, r)
        S = Sout.view(*lead, r)
        Vh= Vhout.view(*lead, r, n)
        return U, S, Vh, True
    except Exception:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        return U, S, Vh, False



def safe_vals_with_fallback(x, condition_threshold=None):
    """
    Per-voxel hybrid: analytic for 2x2 always; for 3x3 use Cardano in voxels
    that pass a cheap conditioning gate, else SVD.
    """
    try:
        *lead, m, n = x.shape
        r = min(m, n)
        Xb = x.reshape(-1, m, n).contiguous()     # (B, m, n)
        B  = Xb.shape[0]

        # Build Gram per block
        if m <= n:
            order = 1
            H = Xb @ Xb.transpose(1, 2)                       # (B, m, m)
        else:
            order = 0
            H = Xb.transpose(1, 2) @ Xb                       # (B, n, n)

        H = 0.5 * (H + H.transpose(1, 2))
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        S = torch.empty(B, r, dtype=x.dtype, device=x.device)

        if r == 2:
            # 2x2: always analytic
            S2 = eigenvalsh_2x2(H)
            S[:] = torch.sqrt(torch.clamp(S2, min=0.0))
        elif r == 3:
            Hn, alpha = _trace_normalize(H)
            #cond_thr = _dtype_cond_threshold(x.dtype) if condition_threshold is None else condition_threshold
            #proxy = _kappa_proxy_3x3(Hn)
            #ok = (proxy < cond_thr) & torch.isfinite(proxy)
            proxy = _log_kappa_proxy_3x3(Hn)
            thr   = math.log(_dtype_cond_threshold(x.dtype) if condition_threshold is None
                                else float(condition_threshold))
            ok = proxy < thr      # finite by construction; no need for isfinite()


            if ok.all():
                # --- all analytic ---
                S2_all = eigenvalsh_3x3_cardano(Hn)
                S[:] = torch.sqrt(torch.clamp(S2_all * alpha.unsqueeze(-1), min=0.0))

            elif (~ok).all():
                # --- all SVD ---
                S[:] = torch.linalg.svdvals(Xb)

            else:
                # --- mixed ---
                S2_ok = eigenvalsh_3x3_cardano(Hn[ok])
                S_ok  = torch.sqrt(torch.clamp(S2_ok * alpha[ok].unsqueeze(-1), min=0.0))
                S[ok] = S_ok
                bad = ~ok
                if bad.any():
                    S[bad] = torch.linalg.svdvals(Xb[bad])

        else:
            # r > 3 not supported here; fall back
            S[:] = torch.linalg.svdvals(Xb)

        return S.view(*lead, r), True  # True = hybrid was attempted
    except Exception:
        sigma = torch.linalg.svdvals(x)
        return sigma, False


class GPUVectorialTotalVariation(Function):
    """
    GPU implementation of vectorial total variation with stability improvements.
    Maintains exact compatibility with original while adding numerical stability.
    """

    def __init__(
        self,
        eps=None,
        norm="nuclear", 
        smoothing_function=None,
        numpy_out=True,
        tail=None,
        use_stability_improvements=True,  # New flag to enable/disable stability features
    ):
        # Maintain exact original parameter handling
        if eps is not None:
            self.eps = torch.tensor(eps, device=device)
        else:
            self.eps = torch.tensor(0.0, device=device)
        
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out
        self.tail = tail
        self.use_stability_improvements = use_stability_improvements

    def direct(self, x):
        # --- Exact original function selection logic ---
        if self.norm == "nuclear":
            norm_func = l1_norm
        elif self.norm == "frobenius":
            norm_func = l2_norm
        else:
            raise ValueError("Norm not defined")

        if self.smoothing_function == "fair":
            smoothing_func = fair
        elif self.smoothing_function == "charbonnier":
            smoothing_func = charbonnier
        elif self.smoothing_function == "perona_malik":
            smoothing_func = perona_malik
        else:
            smoothing_func = nothing

        # Use stabilized SVD that falls back to eigenvalue decomposition when safe
        if self.use_stability_improvements:
            S, _ = safe_vals_with_fallback(x)
        else:
            S = torch.linalg.svdvals(x)
            
        # --- Exact original tailing logic ---
        mask = get_mask(S, self.tail)       # 1 on smallest `tail` σ
        s_smoothed = smoothing_func(S * mask, self.eps)
        s_to_norm  = s_smoothed + S * (1 - mask)    # <-- pass head unchanged
        out = norm_func(s_to_norm)

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        # Exact original implementation
        x = to_tensor(x)
        val = self.direct(x).sum()
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, eps):  # Keep original parameter name 'eps'
        # Exact original parameter handling
        x = to_tensor(x)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        # Exact original function selection
        if self.norm == "nuclear":
            prox_func = l1_norm_prox
        elif self.norm == "frobenius":
            prox_func = l2_norm_prox
        else:
            raise ValueError("Norm not defined")

        # SVD computation with optional stability improvements
        if self.use_stability_improvements:
            U, S, Vh, _ = safe_svd_with_fallback(x)
        else:
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Exact original proximal logic
        S_prox_values = prox_func(S, eps)

        # Exact original tailing logic
        if self.tail is not None:
            mask = get_mask(S, self.tail)
            S_final = S * (1 - mask) + S_prox_values * mask
        else:
            S_final = S_prox_values

        # Exact original reconstruction
        out = torch.matmul(U, Vh * S_final[..., None])

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        # Exact original parameter handling
        x = to_tensor(x)

        # Exact original function selection
        if self.smoothing_function == "fair":
            grad_func = fair_grad
        elif self.smoothing_function == "charbonnier":
            grad_func = charbonnier_grad
        elif self.smoothing_function == "perona_malik":
            grad_func = perona_malik_grad
        else:
            grad_func = nothing_grad

        # SVD computation with optional stability improvements
        if self.use_stability_improvements:
            U, S, Vh, used_eigen = safe_svd_with_fallback(x)
        else:
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Exact original gradient logic
        S_grad_values = grad_func(S, self.eps)

        # Exact original tailing logic
        mask = torch.ones_like(S) if self.tail is None else get_mask(S, self.tail)
        S_grad_values = S_grad_values * mask

        # Exact original reconstruction
        out = torch.matmul(U, Vh * S_grad_values[..., None])
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    def hessian_surrogate(self, x):
        # Exact original parameter handling
        x = to_tensor(x)

        # Exact original function selection (including ValueError for unknown functions)
        if self.smoothing_function == "fair":
            hessian_func = fair_hessian_surrogate
        elif self.smoothing_function == "charbonnier":
            hessian_func = charbonnier_hessian_surrogate
        elif self.smoothing_function == "perona_malik":
            hessian_func = perona_malik_hessian_surrogate
        else:
            raise ValueError("Unknown smoothing function")  # Keep original behavior

        # Exact original computation
        # Use stabilized SVD that falls back to eigenvalue decomposition when safe
        if self.use_stability_improvements:
            S, _ = safe_vals_with_fallback(x)
        else:
            S = torch.linalg.svdvals(x)
            
        # Exact original tailing logic
        mask = torch.ones_like(S) if self.tail is None else get_mask(S, self.tail)
        
        # Exact original hessian computation
        out = hessian_func(S, self.eps)

        return torch.nan_to_num(out * mask, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_components(self, x):
        """
        Hybrid: small blocks via Cardano when gate passes; SVD fallback otherwise;
        r>3 always SVD. Shares gating with the rest of the stable backend.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        coeffs, rank_one = hessian_components_hybrid_small(
            x,
            tail=self.tail,
            smoothing_function=self.smoothing_function,
            eps_tensor=self.eps,
            condition_threshold=None,                 # or expose arg to this method
            order=1 if x.shape[-2] <= x.shape[-1] else 0,
        )
        return torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0), \
            torch.nan_to_num(rank_one, nan=0.0, posinf=0.0, neginf=0.0)


    def stability_report(self, x, condition_threshold=None, quantiles=(0.5, 0.9, 0.99)):
        """
        Report where the analytic (Cardano) path would be used, per-voxel.
        Recomputes the same gate as safe_*: trace-normalize Gram, build proxy,
        threshold → ok mask.

        Returns a dict with:
        - shape/order/r
        - analytic_fraction (overall % voxels using Cardano)
        - ok_count / bad_count
        - threshold_used
        - proxy_quantiles (on the normalized Gram)
        """
        x = to_tensor(x)
        *lead, m, n = x.shape
        r = min(m, n)
        uses_small = (r <= 3)

        # If blocks aren't small, we won't use the analytic path at all
        if not uses_small:
            return {
                "uses_small_matrix_optimization": False,
                "matrix_shape": (m, n),
                "analytic_fraction": 0.0,
                "reason": "min(m,n) > 3 → always SVD"
            }

        # Flatten to batch
        Xb = x.reshape(-1, m, n)
        B  = Xb.shape[0]

        # Which side for Gram?
        order = 1 if m <= n else 0
        H = Xb @ Xb.transpose(1, 2) if order == 1 else Xb.transpose(1, 2) @ Xb
        H = 0.5 * (H + H.transpose(1, 2))
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        if r == 2:
            # 2×2 is always analytic
            return {
                "uses_small_matrix_optimization": True,
                "matrix_shape": (m, n),
                "order": order,
                "r": r,
                "analytic_fraction": 1.0,
                "ok_count": B,
                "bad_count": 0,
                "threshold_used": None,
                "proxy_quantiles": None,
            }

        # r == 3 → compute per-voxel gate
        Hn, _alpha = _trace_normalize(H)
        
        #thr = _dtype_cond_threshold(x.dtype) if condition_threshold is None else condition_threshold
        #proxy = _kappa_proxy_3x3(Hn)  # larger means more ill-conditioned
        #ok = (proxy < thr) & torch.isfinite(proxy)
        
        proxy = _log_kappa_proxy_3x3(Hn)
        thr   = math.log(_dtype_cond_threshold(x.dtype) if condition_threshold is None
                            else float(condition_threshold))
        ok = proxy < thr      # finite by construction; no need for isfinite()

        ok_count = int(ok.sum().item())
        bad_count = int((~ok).sum().item())
        frac = float(ok_count) / float(max(B, 1))

        # proxy quantiles for quick tuning
        if proxy.numel() > 0:
            p = proxy[torch.isfinite(proxy)]
            if p.numel() > 0:
                qs = torch.quantile(p, torch.tensor(quantiles, device=p.device)).tolist()
            else:
                qs = None
        else:
            qs = None

        return {
            "uses_small_matrix_optimization": True,
            "matrix_shape": (m, n),
            "order": order,
            "r": r,
            "analytic_fraction": frac,
            "ok_count": ok_count,
            "bad_count": bad_count,
            "threshold_used": float(thr),
            "proxy_quantiles": {f"q{int(q*100)}": v for q, v in zip(quantiles, qs or [])},
        }
