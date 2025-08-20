# schatten_norm_gpu.py
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
    charbonnier_hessian_surrogate,   # w(σ) for Charbonnier
    fair,
    fair_grad,
    fair_hessian_surrogate,          # w(σ) for Fair
    perona_malik,
    perona_malik_grad,
    perona_malik_hessian_surrogate,  # w(σ) for Perona–Malik
    nothing,
    fair_hessian_diag,
    charbonnier_hessian_diag,
    perona_malik_hessian_diag,
    nothing_hessian_diag, 
    get_mask, 
    add_identity,
)

from .small_eig import (
    eigenvalsh_2x2,
    eigenvecsh_2x2,
    eigenvalsh_3x3_cardano,
    eigenvecsh_3x3_cardano
)

from .svd_free_hessian import hessian_components_svd_free_small

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- Helpers for tail masking and order selection ---------

def choose_order(M):
    """
    Decide which Gram matrix is smaller:
      order=1 -> H = M M^T  (use left eig basis U)
      order=0 -> H = M^T M  (use right eig basis V)
    """
    return 1 if M.shape[-2] <= M.shape[-1] else 0


# --------- Core value/grad/prox maps (batched) ---------

def norm(M, func, smoothing_func, order, eps, tail=None):
    """
    Compute sum_k func( smoothing_func(σ_k) ) per block, in batch.
    σ are extracted via eig of the appropriate Gram matrix.
    Returns tensor of shape (...,) (one value per block).
    """
    Hsym = M.transpose(-1, -2) @ M if order == 0 else M @ M.transpose(-1, -2)
    Hsym = Hsym.contiguous()
    Hsym = add_identity(Hsym)

    last = Hsym.shape[-1]
    if last == 2:
        eig = eigenvalsh_2x2(Hsym)
    elif last == 3:
        eig = eigenvalsh_3x3_cardano(Hsym)
    else:
        raise ValueError("Only 2×2 or 3×3 blocks supported")

    sigma = torch.sqrt(eig)  # (..., r)

    mask = get_mask(sigma, tail) if tail is not None else torch.ones_like(sigma)
    s_smoothed = smoothing_func(sigma * mask, eps)
    s_to_norm = s_smoothed + sigma * (1 - mask)  # blend head through
    return func(s_to_norm)  # sum along σ dim inside func


def norm_func(M, func, tau, order=0, tail=None, blend_head: bool = True):
    """
    Apply elementwise map φ on σ via eig of Gram matrix, then reconstruct:
      order=1:  U diag(φ(σ)/σ) U^T M   with H = M M^T
      order=0:  M V diag(φ(σ)/σ) V^T   with H = M^T M
    Tail semantics:
      - blend_head=True  : S_final = σ*(1-mask) + φ(σ)*mask
      - blend_head=False : S_final =           φ(σ)*mask
    Works batched on inputs (..., M, d).
    """
    H = M @ M.transpose(-1, -2) if order == 1 else M.transpose(-1, -2) @ M
    H = add_identity(H)

    n = H.shape[-1]
    if n == 2:
        S2 = eigenvalsh_2x2(H)
        B  = eigenvecsh_2x2(H, S2)   # U if order=1, V if order=0
    elif n == 3:
        S2 = eigenvalsh_3x3_cardano(H)
        B  = eigenvecsh_3x3_cardano(H, S2)
    else:
        raise ValueError(f"Only 2×2 or 3×3 blocks supported, got {n}.")

    S = torch.sqrt(S2)               # (..., r)
    S_map = func(S, tau)             # φ(σ)

    if tail is not None:
        mask = get_mask(S, tail)     # 1 on smallest tail
        S_final = S*(1 - mask) + S_map*mask if blend_head else S_map*mask
    else:
        S_final = S_map

    tiny = torch.finfo(S.dtype).eps
    scale = torch.where(S > 0, S_final / torch.clamp(S, min=tiny), torch.zeros_like(S))  # (..., r)

    D = (B * scale[..., None, :]) @ B.transpose(-1, -2)  # (..., n, n)
    return D @ M if order == 1 else M @ D


def sigma_map(M, elem_func, tau, order=0, tail=None, masked_only=True):
    """
    Return elementwise mapping of singular values σ of M without SVD:
      σ = sqrt(eig(Hsym)), Hsym = M M^T (order=1) or M^T M (order=0).
    Returns (..., r). If tail is set:
      - masked_only=True  : keep ONLY smallest `tail` σ (others → 0)
      - masked_only=False : blend head through (σ on head, elem on tail)
    """
    H = M @ M.transpose(-1, -2) if order == 1 else M.transpose(-1, -2) @ M
    H = add_identity(H)

    last = H.shape[-1]
    if last == 2:
        S2 = eigenvalsh_2x2(H)
    elif last == 3:
        S2 = eigenvalsh_3x3_cardano(H)
    else:
        raise ValueError(f"Only 2×2 or 3×3 supported, got {last}×{last}.")

    S = torch.sqrt(S2)  # (..., r)

    mask = get_mask(S, tail) if tail is not None else torch.ones_like(S)
    mapped = elem_func(S, tau)  # elementwise on σ

    if tail is not None and masked_only:
        return mapped * mask
    elif tail is not None:
        return mapped * mask + S * (1 - mask)
    else:
        return mapped


# --------- Main class ---------

class GPUVectorialTotalVariation(Function):
    """
    GPU implementation of the vectorial total variation function.
    SVD-free for value/prox/grad (via eig of 2×2/3×3 blocks).
    `hessian_components` performs a full SVD (U, V needed).
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
        order = choose_order(x)
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

        out = norm(x, norm_func, smoothing_func, order, self.eps, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        x = to_tensor(x)
        val = self.direct(x).sum()
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, tau):
        # prox path: process tail, pass head through (blend_head=True)
        x = to_tensor(x)
        order = choose_order(x)
        if self.norm == "nuclear":
            prox_func = l1_norm_prox
        elif self.norm == "frobenius":
            prox_func = l2_norm_prox
        else:
            raise ValueError("Proximal for this norm not defined")
        out = norm_func(x, prox_func, tau, order, self.tail, blend_head=True)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        # grad path: ONLY the tail contributes (blend_head=False)
        x = to_tensor(x)
        order = choose_order(x)
        if self.smoothing_function == "fair":
            grad_func = fair_grad
        elif self.smoothing_function == "charbonnier":
            grad_func = charbonnier_grad
        elif self.smoothing_function == "perona_malik":
            grad_func = perona_malik_grad
        else:
            raise ValueError("Smoothing function not defined for gradient")
        out = norm_func(x, grad_func, self.eps, order, self.tail, blend_head=False)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_surrogate(self, x):
        """
        Return per-σ IRLS weights w(σ)=g'(σ)/(2σ), SVD-free, for MM/IRLS.
        Tail semantics: keep ONLY the smallest `tail` σ (others → 0).
        """
        x = to_tensor(x)
        order = choose_order(x)

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

        out = sigma_map(x, elem, self.eps, order, self.tail, masked_only=True)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_components(self, x):
        """
        SVD-free for r∈{2,3} using Gram eig projectors; r>3 → full SVD.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        *lead, Mdim, Ddim = x.shape
        r = min(Mdim, Ddim)

        # SVD-free small blocks
        if r in (2, 3):
            coeffs, rank_one = hessian_components_svd_free_small(
                x, tail=self.tail, smoothing_function=self.smoothing_function,
                order=1 if Mdim <= Ddim else 0,
            )
            # smoothing epsilon (self.eps) is only needed inside hessian_diag;
            # in the helper we passed 0.0; if you want it used, swap the call to
            # route self.eps there (or keep as-is if your diag funcs read it differently).
            return torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0), \
                torch.nan_to_num(rank_one, nan=0.0, posinf=0.0, neginf=0.0)

        # Fallback for larger r
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        if self.smoothing_function == "fair":
            hdiag = fair_hessian_diag
        elif self.smoothing_function == "charbonnier":
            hdiag = charbonnier_hessian_diag
        elif self.smoothing_function == "perona_malik":
            hdiag = perona_malik_hessian_diag
        else:
            hdiag = nothing_hessian_diag

        coeffs = hdiag(S, self.eps)
        if self.tail is not None:
            coeffs = coeffs * get_mask(S, self.tail)

        U_perm = U.permute(*range(U.ndim - 2), -1, -2)
        rank_one = U_perm.unsqueeze(-1) @ Vh.unsqueeze(-2)
        return torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0), \
            torch.nan_to_num(rank_one, nan=0.0, posinf=0.0, neginf=0.0)