# schatten_norm_gpu.py
import numpy as np
import torch
from cil.optimisation.functions import Function
from torch import vmap

from .common import (
    to_tensor,
    l1_norm_torch,
    l1_norm_prox_torch,
    l2_norm_torch,
    l2_norm_prox_torch,
    charbonnier_torch,
    charbonnier_grad_torch,
    charbonnier_hessian_surrogate,   # w(σ) for Charbonnier
    fair_torch,
    fair_grad_torch,
    fair_hessian_surrogate,          # w(σ) for Fair
    perona_malik_torch,
    perona_malik_grad_torch,
    perona_malik_hessian_surrogate,  # w(σ) for Perona–Malik
    nothing_torch,
    nothing_grad_torch,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))


def add_identity_torch(H, eps=1e-6):
    """
    Batch-safe: add eps*I to the last-2 dims of H (… , n, n).
    """
    I = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
    I = I.expand(*H.shape[:-2], H.shape[-2], H.shape[-1])
    return H + eps * I


# --------- Eigen helpers (SVD-free path) ---------

def eigenvalues_2x2_torch(H):
    """
    H: (..., 2, 2) symmetric
    returns (..., 2) nonnegative eigenvalues (ascending order not guaranteed).
    """
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 1, 0]
    d = H[..., 1, 1]

    half_trace = 0.5 * (a + d)
    det = a * d - b * c
    disc = torch.clamp(half_trace * half_trace - det, min=0.0)
    root = torch.sqrt(disc)

    lam1 = torch.clamp(half_trace + root, min=0.0)
    lam2 = torch.clamp(half_trace - root, min=0.0)
    return torch.stack([lam1, lam2], dim=-1)  # (..., 2)


def eigenvectors_2x2_torch(H, eigenvalues):
    """
    H: (..., 2, 2) symmetric, eigenvalues: (..., 2)
    returns eigenvectors (..., 2, 2) with columns as eigenvectors.
    """
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 1, 0]
    d = H[..., 1, 1]
    lam1 = eigenvalues[..., 0]
    lam2 = eigenvalues[..., 1]

    b_nz = b.abs() > 1e-9
    c_nz = c.abs() > 1e-9

    e1_c1 = torch.stack([b, lam1 - a], dim=-1)
    e1_c2 = torch.stack([lam1 - d, c], dim=-1)
    e1_def = torch.stack([torch.ones_like(a), torch.zeros_like(a)], dim=-1)
    e1 = torch.where(b_nz.unsqueeze(-1), e1_c1,
         torch.where(c_nz.unsqueeze(-1), e1_c2, e1_def))
    e1 = e1 / torch.clamp(torch.linalg.norm(e1, dim=-1, keepdim=True), min=1e-9)

    e2_c1 = torch.stack([b, lam2 - a], dim=-1)
    e2_c2 = torch.stack([lam2 - d, c], dim=-1)
    e2_def = torch.stack([torch.zeros_like(a), torch.ones_like(a)], dim=-1)
    e2 = torch.where(b_nz.unsqueeze(-1), e2_c1,
         torch.where(c_nz.unsqueeze(-1), e2_c2, e2_def))
    e2 = e2 / torch.clamp(torch.linalg.norm(e2, dim=-1, keepdim=True), min=1e-9)

    # stack eigenvectors as columns
    return torch.stack([e1, e2], dim=-1)  # (..., 2, 2)


def eigenvalues_3x3_torch(H):
    """
    H: (..., 3, 3) symmetric
    returns (..., 3) nonnegative eigenvalues (ascending from torch.linalg.eigvalsh).
    """
    vals = torch.linalg.eigvalsh(H)
    return torch.clamp(vals, min=0.0)


def eigenvectors_3x3_torch(H, eigenvalues):
    """
    H: (..., 3, 3) symmetric
    returns (..., 3, 3) eigenvectors (columns), from eigh (SVD-free).
    """
    _, vecs = torch.linalg.eigh(H)
    return vecs


# --------- Value path (sum over smoothed σ), SVD-free ---------

def norm_torch(H, func, smoothing_func, order, eps, tail=None):
    """
    Compute sum_k func( smoothing_func(σ_k) ) per block, SVD-free:
      σ = sqrt(eig(Hsym)), Hsym = M^T M (order=0) or M M^T (order=1).
    Tail semantics: apply smoothing only to the smallest `tail` σ; others pass through.
    """
    M = H
    Hsym = M.transpose(-1, -2) @ M if order == 0 else M @ M.transpose(-1, -2)
    Hsym = add_identity_torch(Hsym, eps / 1e3)

    last = Hsym.shape[-1]
    if last == 2:
        eig = eigenvalues_2x2_torch(Hsym)
    elif last == 3:
        eig = eigenvalues_3x3_torch(Hsym)
    else:
        raise ValueError("Only 2×2 or 3×3 blocks supported")

    sigma = torch.sqrt(eig)  # (..., r)

    if tail is not None:
        sigma_sorted, sort_idx = torch.sort(sigma, dim=-1)
        r = sigma.shape[-1]
        ones_tail = torch.ones((*sigma.shape[:-1], tail), device=sigma.device, dtype=sigma.dtype)
        zeros_head = torch.zeros((*sigma.shape[:-1], r - tail), device=sigma.device, dtype=sigma.dtype)
        mask_sorted = torch.cat([ones_tail, zeros_head], dim=-1)
        inv = torch.argsort(sort_idx, dim=-1)
        mask = torch.gather(mask_sorted, dim=-1, index=inv)
    else:
        mask = torch.ones_like(sigma)

    s_smoothed = smoothing_func(sigma * mask, eps)
    s_to_norm = s_smoothed + sigma * (1 - mask)  # blend head
    return func(s_to_norm)  # sum along σ dim inside func


# --------- Prox/Grad reconstructors (SVD-free via eigenvectors) ---------

def norm_func_torch_xxt(M, func, tau, tail=None, blend_head: bool = True):
    """
    Apply elementwise func to σ, then reconstruct U diag(f(σ)/σ) U^T M
    with H = M M^T (SVD-free via eigenvectors). Tail semantics:
      - blend_head=True  : S_final = σ*(1-mask) + f(σ)*mask
      - blend_head=False : S_final = f(σ)*mask
    """
    H = M @ M.transpose(-1, -2)
    H = add_identity_torch(H, eps=1e-6)

    last = H.shape[-1]
    if last == 2:
        S2 = eigenvalues_2x2_torch(H)
        U = eigenvectors_2x2_torch(H, S2)
    elif last == 3:
        S2 = eigenvalues_3x3_torch(H)
        U = eigenvectors_3x3_torch(H, S2)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S2)
    S_func = func(S, tau)

    if tail is not None:
        S_sorted, sort_idx = torch.sort(S, dim=-1)
        r = S.shape[-1]
        ones_tail = torch.ones((*S.shape[:-1], tail), device=S.device, dtype=S.dtype)
        zeros_head = torch.zeros((*S.shape[:-1], r - tail), device=S.device, dtype=S.dtype)
        mask_sorted = torch.cat([ones_tail, zeros_head], dim=-1)
        inv = torch.argsort(sort_idx, dim=-1)
        mask = torch.gather(mask_sorted, dim=-1, index=inv)
        if blend_head:
            S_final = S * (1 - mask) + S_func * mask
        else:
            S_final = S_func * mask
    else:
        S_final = S_func

    S_inv = torch.where(S > 0, 1.0 / S, torch.zeros_like(S))
    # U diag(S_final * S_inv) U^T M
    UD = U @ torch.diag_embed(S_final * S_inv) @ U.transpose(-1, -2)
    return UD @ M


def norm_func_torch_xtx(M, func, tau, tail=None, blend_head: bool = True):
    """
    Apply elementwise func to σ, then reconstruct M V diag(f(σ)/σ) V^T
    with H = M^T M (SVD-free via eigenvectors).
    """
    H = M.transpose(-1, -2) @ M
    H = add_identity_torch(H, eps=1e-6)

    last = H.shape[-1]
    if last == 2:
        S2 = eigenvalues_2x2_torch(H)
        V = eigenvectors_2x2_torch(H, S2)
    elif last == 3:
        S2 = eigenvalues_3x3_torch(H)
        V = eigenvectors_3x3_torch(H, S2)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S2)
    S_func = func(S, tau)

    if tail is not None:
        S_sorted, sort_idx = torch.sort(S, dim=-1)
        r = S.shape[-1]
        ones_tail = torch.ones((*S.shape[:-1], tail), device=S.device, dtype=S.dtype)
        zeros_head = torch.zeros((*S.shape[:-1], r - tail), device=S.device, dtype=S.dtype)
        mask_sorted = torch.cat([ones_tail, zeros_head], dim=-1)
        inv = torch.argsort(sort_idx, dim=-1)
        mask = torch.gather(mask_sorted, dim=-1, index=inv)
        if blend_head:
            S_final = S * (1 - mask) + S_func * mask
        else:
            S_final = S_func * mask
    else:
        S_final = S_func

    S_inv = torch.where(S > 0, 1.0 / S, torch.zeros_like(S))
    # M V diag(S_final * S_inv) V^T
    VD = V @ torch.diag_embed(S_final * S_inv) @ V.transpose(-1, -2)
    return M @ VD


def norm_func_torch(M, func, tau, order=0, tail=None, blend_head: bool = True):
    if order == 0:
        return norm_func_torch_xtx(M, func, tau, tail, blend_head)
    elif order == 1:
        return norm_func_torch_xxt(M, func, tau, tail, blend_head)
    else:
        raise ValueError("Invalid order")


# --------- Vmap wrappers ---------

def vectorised_norm(A, func, smoothing_func, order=0, eps=0, tail=None):
    def single_block(block):
        return norm_torch(block, func, smoothing_func, order, eps, tail)
    return vmap(vmap(vmap(single_block, in_dims=0), in_dims=0), in_dims=0)(A)


def vectorised_norm_func(A, func, tau, order=0, tail=None, blend_head: bool = True):
    def single_block(block):
        return norm_func_torch(block, func, tau, order, tail, blend_head)
    return vmap(vmap(vmap(single_block, in_dims=0), in_dims=0), in_dims=0)(A)


# --------- Per-σ map (no reconstruction), SVD-free ---------

def sigma_map_torch(M, elem_func, tau, order=0, tail=None, masked_only=True):
    """
    Return elementwise mapping of singular values σ of M without SVD:
      σ = sqrt(eig(Hsym)), Hsym = M M^T (order=1) or M^T M (order=0).
    Returns (..., r). If tail is set:
      - masked_only=True  : keep ONLY smallest `tail` σ (others → 0)
      - masked_only=False : blend head through (σ on head, elem on tail)
    """
    H = M @ M.transpose(-1, -2) if order == 1 else M.transpose(-1, -2) @ M
    H = add_identity_torch(H, eps=1e-6)

    last = H.shape[-1]
    if last == 2:
        S2 = eigenvalues_2x2_torch(H)
    elif last == 3:
        S2 = eigenvalues_3x3_torch(H)
    else:
        raise ValueError(f"Only 2×2 or 3×3 supported, got {last}×{last}.")

    S = torch.sqrt(S2)  # (..., r)

    if tail is not None:
        S_sorted, sort_idx = torch.sort(S, dim=-1)
        r = S.shape[-1]
        ones_tail = torch.ones((*S.shape[:-1], tail), device=S.device, dtype=S.dtype)
        zeros_head = torch.zeros((*S.shape[:-1], r - tail), device=S.device, dtype=S.dtype)
        mask_sorted = torch.cat([ones_tail, zeros_head], dim=-1)
        inv = torch.argsort(sort_idx, dim=-1)
        mask = torch.gather(mask_sorted, dim=-1, index=inv)
    else:
        mask = torch.ones_like(S)

    mapped = elem_func(S, tau)  # elementwise on σ

    if tail is not None and masked_only:
        return mapped * mask
    elif tail is not None and not masked_only:
        return mapped * mask + S * (1 - mask)
    else:
        return mapped


def vectorised_sigma_map(A, elem_func, tau, order=0, tail=None, masked_only=True):
    def single_block(block):
        return sigma_map_torch(block, elem_func, tau, order, tail, masked_only)
    return vmap(vmap(vmap(single_block, in_dims=0), in_dims=0), in_dims=0)(A)


# --------- Main class ---------

class GPUVectorialTotalVariation(Function):
    """
    GPU implementation of the vectorial total variation function.
    Fully SVD-free (uses eig of 2×2/3×3 blocks).
    """

    def __init__(
        self,
        eps=None,
        norm="nuclear",
        smoothing_function=None,
        numpy_out=True,
        tail=None,
    ):
        super(GPUVectorialTotalVariation, self).__init__()
        self.eps = torch.tensor(eps if eps is not None else 0.0, device=device)
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out
        self.tail = tail

    def direct(self, x):
        # value path: sum over smoothed σ (blend head semantics)
        order = 1 if x.shape[-2] <= x.shape[-1] else 0
        if self.norm == "nuclear":
            norm_func = l1_norm_torch
        elif self.norm == "frobenius":
            norm_func = l2_norm_torch
        else:
            raise ValueError("Norm not defined")

        if self.smoothing_function == "fair":
            smoothing_func = fair_torch
        elif self.smoothing_function == "charbonnier":
            smoothing_func = charbonnier_torch
        elif self.smoothing_function == "perona_malik":
            smoothing_func = perona_malik_torch
        else:
            smoothing_func = nothing_torch

        out = vectorised_norm(x, norm_func, smoothing_func, order, self.eps, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        x = to_tensor(x)
        val = self.direct(x).sum()
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, tau):
        # prox path: process tail, pass head through (blend_head=True)
        x = to_tensor(x)
        order = 1 if x.shape[-2] <= x.shape[-1] else 0
        if self.norm == "nuclear":
            prox_func = l1_norm_prox_torch
        elif self.norm == "frobenius":
            prox_func = l2_norm_prox_torch
        else:
            raise ValueError("Proximal for this norm not defined")
        out = vectorised_norm_func(x, prox_func, tau, order, self.tail, blend_head=True)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        # grad path: ONLY the tail contributes (blend_head=False)
        x = to_tensor(x)
        order = 1 if x.shape[-2] <= x.shape[-1] else 0
        if self.smoothing_function == "fair":
            grad_func = fair_grad_torch
        elif self.smoothing_function == "charbonnier":
            grad_func = charbonnier_grad_torch
        elif self.smoothing_function == "perona_malik":
            grad_func = perona_malik_grad_torch
        else:
            raise ValueError("Smoothing function not defined for gradient")
        out = vectorised_norm_func(x, grad_func, self.eps, order, self.tail, blend_head=False)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_surrogate(self, x):
        """
        Return per-σ IRLS weights w(σ)=g'(σ)/(2σ), SVD-free, for MM/IRLS.
        Tail semantics: keep ONLY the smallest `tail` σ (others → 0).
        """
        x = to_tensor(x)
        order = 1 if x.shape[-2] <= x.shape[-1] else 0

        if self.smoothing_function == "fair":
            elem = fair_hessian_surrogate
        elif self.smoothing_function == "charbonnier":
            elem = charbonnier_hessian_surrogate
        elif self.smoothing_function == "perona_malik":
            elem = perona_malik_hessian_surrogate
        else:
            # unsmoothed TNV: w(σ)=1/(2σ) with safe floor
            def elem(S, _tau):
                tiny = torch.finfo(S.dtype).eps
                return 0.5 / torch.clamp(S, min=tiny)

        out = vectorised_sigma_map(x, elem, self.eps, order, self.tail, masked_only=True)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
