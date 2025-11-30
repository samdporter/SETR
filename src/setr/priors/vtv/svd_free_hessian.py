# svd_free_hessian.py
import torch

from .common import (
    charbonnier_hessian_diag,
    fair_hessian_diag,
    get_mask,
    nothing_hessian_diag,
    perona_malik_hessian_diag,
)
from .small_eig import (
    eigenvalsh_2x2,
    eigenvalsh_3x3_cardano,
    eigenvecsh_2x2,
    eigenvecsh_3x3_cardano,
)


# ---------- tiny utilities (shared with your “stable” gate) ----------
def _trace_normalize(H):
    n = H.shape[-1]
    tr = torch.diagonal(H, -2, -1).sum(-1)
    tiny = torch.finfo(H.dtype).tiny
    alpha = torch.clamp(tr / float(n), min=tiny)
    Hn = H / alpha.unsqueeze(-1).unsqueeze(-1)
    return Hn, alpha


def _kappa_proxy_3x3(H):
    trH = torch.diagonal(H, -2, -1).sum(-1)
    trH2 = (H * H).sum(dim=(-2, -1))
    s2 = 0.5 * (trH * trH - trH2)
    detH = torch.linalg.det(H)
    tiny = torch.finfo(H.dtype).tiny
    proxy = (trH / 3.0) * (s2 / torch.clamp(detH, min=tiny))
    return torch.nan_to_num(proxy, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))


def _log_kappa_proxy_3x3(H):
    """
    Log proxy for SPD-ish 3x3 (scale invariant):
    log_proxy = log(tr/3) + log(s2) - log(det)
    All terms clamped to keep them finite in float32.
    """
    trH = torch.diagonal(H, -2, -1).sum(-1)  # (...,)
    # Frobenius^2 == tr(H^2) for symmetric H
    trH2 = (H * H).sum(dim=(-2, -1))  # (...,)
    tiny = torch.finfo(H.dtype).tiny

    s2 = 0.5 * (trH * trH - trH2)
    s2 = torch.clamp(s2, min=tiny)  # keep positive & finite
    detH = torch.clamp(torch.linalg.det(H), min=tiny)  # avoid log(0)

    log_tr_over_3 = torch.log(torch.clamp(trH / 3.0, min=tiny))
    log_s2 = torch.log(s2)
    log_det = torch.log(detH)
    log_proxy = log_tr_over_3 + log_s2 - log_det  # (...,)

    # sanitize NaN -> big positive number so it naturally fails the gate but stays finite
    log_proxy = torch.nan_to_num(log_proxy, nan=1e30)
    return log_proxy


def _dtype_cond_threshold(dtype):
    return 1e7 if dtype == torch.float32 else 1e12


def _pick_hess_diag(smoothing_function):
    if smoothing_function == "fair":
        return fair_hessian_diag
    if smoothing_function == "charbonnier":
        return charbonnier_hessian_diag
    if smoothing_function == "perona_malik":
        return perona_malik_hessian_diag
    return nothing_hessian_diag


# ---------- core projector construction from one-side Gram eigenvectors ----------
def _rank_one_fields_from_gram(M, order, B, S):
    """
    Build slices u_k v_k^T using only one side + σ:
      order=1 (H = M M^T):   (u_k u_k^T M) / σ_k
      order=0 (H = M^T M):   (M v_k v_k^T) / σ_k
    B: (..., n, r) eigenvectors of Gram (U if order=1, V if order=0)
    S: (..., r)     singular values (>=0)
    Returns (..., r, Mdim, Ddim).
    """
    *lead, Mdim, Ddim = M.shape
    r = S.shape[-1]
    tiny = 1e-9 # A more robust epsilon
    Sinv = 1.0 / torch.clamp(S, min=tiny)  # (..., r)

    if order == 1:
        # U = B  (..., M, r)
        U_perm = B.permute(*range(B.ndim - 2), -1, -2)  # (..., r, M)
        P = U_perm.unsqueeze(-1) @ U_perm.unsqueeze(-2)  # (..., r, M, M)
        return (P @ M.unsqueeze(-3)) * Sinv[..., :, None, None]
    else:
        # V = B  (..., D, r)
        V_perm = B.permute(*range(B.ndim - 2), -1, -2)  # (..., r, D)
        Q = V_perm.unsqueeze(-1) @ V_perm.unsqueeze(-2)  # (..., r, D, D)
        return (M.unsqueeze(-3) @ Q) * Sinv[..., :, None, None]


# ---------- small-block eig path (2×2/3×3) ----------
def _gram_eig_small_blocks(M, order):
    """
    Returns (S, B) where:
      S: (..., r) singular values (ascending)
      B: (..., n, r) eigenvectors of Gram (U if order=1, V if order=0)
    Only supports r in {2,3}; caller handles r>3.
    """
    H = (M @ M.transpose(-1, -2)) if order == 1 else (M.transpose(-1, -2) @ M)
    H = 0.5 * (H + H.transpose(-1, -2))
    r = H.shape[-1]

    if r == 2:
        L = eigenvalsh_2x2(H)  # ascending
        B = eigenvecsh_2x2(H, L)
    elif r == 3:
        L = eigenvalsh_3x3_cardano(H)  # ascending
        B = eigenvecsh_3x3_cardano(H, L)
    else:
        raise ValueError(f"_gram_eig_small_blocks: r={r} not supported")

    S = torch.sqrt(torch.clamp(L, min=0.0))
    return S, B


# ---------- public helpers used by both classes ----------
def hessian_components_svd_free_small(x, tail, smoothing_function, order=None):
    """
    SVD-free hessian_components for r∈{2,3}. r>3 → raises (caller decides fallback).
    Returns (hess_coeffs, rank_one_fields).
    """
    *lead, Mdim, Ddim = x.shape
    r = min(Mdim, Ddim)
    if r not in (2, 3):
        raise ValueError("hessian_components_svd_free_small only supports r=2 or r=3")

    if order is None:
        order = 1 if Mdim <= Ddim else 0

    S, B = _gram_eig_small_blocks(x, order)  # ascending S
    hess_diag = _pick_hess_diag(smoothing_function)
    coeffs = hess_diag(
        S, torch.tensor(0.0, dtype=x.dtype, device=x.device)
    )  # tau/eps handled by caller

    if tail is not None:
        coeffs = coeffs * get_mask(S, tail)

    rank_one = _rank_one_fields_from_gram(x, order, B, S)
    return coeffs, rank_one


def hessian_components_hybrid_small(
    x, tail, smoothing_function, eps_tensor, condition_threshold=None, order=None
):
    """
    Hybrid version (used by *stable* backend):
      - r∈{2,3}: Cardano path per-voxel when gate passes; SVD fallback otherwise.
      - r>3: full SVD fallback.
    Returns (hess_coeffs, rank_one_fields).
    """
    *lead, Mdim, Ddim = x.shape
    r = min(Mdim, Ddim)
    if order is None:
        order = 1 if Mdim <= Ddim else 0
    hess_diag = _pick_hess_diag(smoothing_function)

    # r==2 → always analytic
    if r == 2:
        S, B = _gram_eig_small_blocks(x, order)
        coeffs = hess_diag(S, eps_tensor)
        if tail is not None:
            coeffs = coeffs * get_mask(S, tail)
        rank_one = _rank_one_fields_from_gram(x, order, B, S)
        return coeffs, rank_one

    # r==3 → gate
    if r == 3:
        # Gram (for gating only)
        H = (x @ x.transpose(-1, -2)) if order == 1 else (x.transpose(-1, -2) @ x)
        H = 0.5 * (H + H.transpose(-1, -2))

        Hn, alpha = _trace_normalize(H)  # scale-invariant gate
        thr = _dtype_cond_threshold(x.dtype) if condition_threshold is None else condition_threshold
        proxy = _kappa_proxy_3x3(Hn)
        ok = (proxy < thr) & torch.isfinite(proxy)

        # preallocate outputs
        coeffs = torch.empty(*lead, r, dtype=x.dtype, device=x.device)
        rank_one = torch.empty(*lead, r, Mdim, Ddim, dtype=x.dtype, device=x.device)

        if ok.all():
            # all analytic
            L = eigenvalsh_3x3_cardano(Hn)  # (...,3), ascending
            B = eigenvecsh_3x3_cardano(Hn, L)
            S = torch.sqrt(torch.clamp(L * alpha.unsqueeze(-1), min=0.0))
            c = hess_diag(S, eps_tensor)
            if tail is not None:
                c = c * get_mask(S, tail)
            coeffs[:] = c
            rank_one[:] = _rank_one_fields_from_gram(x, order, B, S)

        elif (~ok).all():
            # all fallback
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            c = hess_diag(S, eps_tensor)
            if tail is not None:
                c = c * get_mask(S, tail)
            coeffs[:] = c
            U_perm = U.permute(*range(U.ndim - 2), -1, -2)
            rank_one[:] = U_perm.unsqueeze(-1) @ Vh.unsqueeze(-2)

        else:
            # mixed
            if ok.any():
                L_ok = eigenvalsh_3x3_cardano(Hn[ok])
                B_ok = eigenvecsh_3x3_cardano(Hn[ok], L_ok)
                S_ok = torch.sqrt(torch.clamp(L_ok * alpha[ok].unsqueeze(-1), min=0.0))
                c_ok = hess_diag(S_ok, eps_tensor)
                if tail is not None:
                    c_ok = c_ok * get_mask(S_ok, tail)
                coeffs[ok] = c_ok
                rank_one[ok] = _rank_one_fields_from_gram(x[ok], order, B_ok, S_ok)

            bad = ~ok
            if bad.any():
                U_b, S_b, Vh_b = torch.linalg.svd(x[bad], full_matrices=False)
                c_b = hess_diag(S_b, eps_tensor)
                if tail is not None:
                    c_b = c_b * get_mask(S_b, tail)
                coeffs[bad] = c_b
                U_perm_b = U_b.permute(*range(U_b.ndim - 2), -1, -2)
                rank_one[bad] = U_perm_b.unsqueeze(-1) @ Vh_b.unsqueeze(-2)

        return coeffs, rank_one

    # r>3 → caller uses full SVD
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    coeffs = hess_diag(S, eps_tensor)
    if tail is not None:
        coeffs = coeffs * get_mask(S, tail)
    U_perm = U.permute(*range(U.ndim - 2), -1, -2)
    rank_one = U_perm.unsqueeze(-1) @ Vh.unsqueeze(-2)
    return coeffs, rank_one
