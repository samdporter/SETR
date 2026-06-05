# schatten_norm_gpu.py
import numpy as np
import torch
from cil.optimisation.functions import Function

from .common import (
    charbonnier,
    charbonnier_grad,
    charbonnier_hessian_diag,
    charbonnier_hessian_surrogate,  # w(σ) for Charbonnier
    fair,
    fair_grad,
    fair_hessian_diag,
    fair_hessian_surrogate,  # w(σ) for Fair
    get_mask,
    l1_norm,
    l1_norm_prox,
    l2_norm,
    l2_norm_prox,
    nothing,
    nothing_hessian_diag,
    perona_malik,
    perona_malik_grad,
    perona_malik_hessian_diag,
    perona_malik_hessian_surrogate,  # w(σ) for Perona–Malik
    to_tensor,
)
from .small_eig import (
    adaptive_gram_regularization,
    eigenvalsh_2x2,
    eigenvalsh_3x3_cardano,
    eigenvecsh_2x2,
    eigenvecsh_3x3_cardano,
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


def norm(M, norm_type, smoothing_func, order, eps, tail=None):
    """
    Enhanced norm computation with adaptive regularization.

    Args:
        norm_type: "nuclear" or "frobenius"
        smoothing_func: smoothing function to apply
    """
    # Get adaptively regularized Gram matrix
    H_reg, reg_scale, kappa = adaptive_gram_regularization(M, order)

    # Extract eigenvalues
    n = H_reg.shape[-1]
    H_reg_safe = torch.nan_to_num(H_reg, nan=0.0, posinf=0.0, neginf=0.0)
    if n == 2:
        eig = eigenvalsh_2x2(H_reg_safe)
    elif n == 3:
        eig = eigenvalsh_3x3_cardano(H_reg_safe)
    else:
        eig = torch.linalg.eigvalsh(H_reg_safe)

    # Compensate for regularization in singular values
    # Mathematical justification: σ²(A) + ε = λ(H̃)
    # Therefore: σ(A) = √(max(λ(H̃) - ε, 0))
    sigma_squared = torch.clamp(eig - reg_scale[..., None], min=0)
    sigma = torch.sqrt(sigma_squared)

    # Apply smoothing and norm based on type
    mask = get_mask(sigma, tail) if tail is not None else torch.ones_like(sigma)

    if norm_type == "nuclear":
        # Nuclear norm: sum_i h(sigma_i)
        s_smoothed = smoothing_func(sigma * mask, eps)
        s_to_sum = s_smoothed + sigma * (1 - mask)
        return torch.sum(s_to_sum, dim=-1)

    elif norm_type == "frobenius":
        # Frobenius norm: h(||sigma||_2)
        frobenius_norm = torch.sqrt(torch.sum(sigma**2, dim=-1))
        return smoothing_func(frobenius_norm, eps)

    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def norm_func(M, func, tau, order=0, tail=None, blend_head: bool = True):
    """
    Apply elementwise map φ on σ via eig of Gram matrix.
    Mathematical correction: Account for regularization in singular value extraction.
    """
    # Use adaptive regularization function from earlier
    H, epsilon_used, kappa = adaptive_gram_regularization(M, order)

    n = H.shape[-1]
    if n == 2:
        S2_reg = eigenvalsh_2x2(H)
        B = eigenvecsh_2x2(H, S2_reg)
    elif n == 3:
        S2_reg = eigenvalsh_3x3_cardano(H)
        B = eigenvecsh_3x3_cardano(H, S2_reg)
    else:
        raise ValueError(f"Only 2×2 or 3×3 blocks supported, got {n}.")

    # Mathematical correction: λ(H̃) = σ²(M) + ε
    S2_true = torch.clamp(S2_reg - epsilon_used[..., None], min=0)
    S = torch.sqrt(S2_true)

    S_map = func(S, tau)

    if tail is not None:
        mask = get_mask(S, tail)
        S_final = S * (1 - mask) + S_map * mask if blend_head else S_map * mask
    else:
        S_final = S_map

    tiny = torch.finfo(S.dtype).eps
    scale = torch.where(S > 0, S_final / torch.clamp(S, min=tiny), torch.zeros_like(S))

    D = (B * scale[..., None, :]) @ B.transpose(-1, -2)
    return D @ M if order == 1 else M @ D


def sigma_map(M, elem_func, tau, order=0, tail=None, masked_only=True):
    """
    Return elementwise mapping of singular values σ of M.
    Mathematical correction: Account for Gram matrix regularization.

    Definition: σᵢ(M) = √(λᵢ(H̃) - ε) where H̃ = Gram(M) + εI
    """
    # Apply adaptive regularization using previously defined function
    H_reg, epsilon_used, kappa = adaptive_gram_regularization(M, order)

    last = H_reg.shape[-1]
    if last == 2:
        S2_reg = eigenvalsh_2x2(H_reg)
    elif last == 3:
        S2_reg = eigenvalsh_3x3_cardano(H_reg)
    else:
        raise ValueError(f"Only 2×2 or 3×3 supported, got {last}×{last}.")

    # Mathematical correction: λ(H̃) = σ²(M) + ε
    # Therefore: σ²(M) = λ(H̃) - ε
    S2_true = torch.clamp(S2_reg - epsilon_used[..., None], min=0)
    S = torch.sqrt(S2_true)

    mask = get_mask(S, tail) if tail is not None else torch.ones_like(S)
    mapped = elem_func(S, tau)

    if tail is not None and masked_only:
        return mapped * mask
    elif tail is not None:
        return mapped * mask + S * (1 - mask)
    else:
        return mapped


def gradient_norm(M, norm_type, grad_func, eps, order, tail=None):
    """
    Compute gradient with respect to nuclear or Frobenius norm.

    Args:
        norm_type: "nuclear" or "frobenius"
        grad_func: smoothing gradient function (e.g., perona_malik_grad)
    """
    # Get eigendecomposition
    H_reg, reg_scale, kappa = adaptive_gram_regularization(M, order)

    n = H_reg.shape[-1]
    if n == 2:
        S2_reg = eigenvalsh_2x2(H_reg)
        B = eigenvecsh_2x2(H_reg, S2_reg)
    elif n == 3:
        S2_reg = eigenvalsh_3x3_cardano(H_reg)
        B = eigenvecsh_3x3_cardano(H_reg, S2_reg)
    else:
        raise ValueError(f"Only 2×2 or 3×3 blocks supported, got {n}.")

    # Correct for regularization
    S2_true = torch.clamp(S2_reg - reg_scale[..., None], min=0)
    S = torch.sqrt(S2_true)

    tiny = torch.finfo(S.dtype).eps

    if norm_type == "nuclear":
        # Nuclear norm: gradient element-wise on singular values
        S_grad = grad_func(S, eps)

        # Apply tailing
        if tail is not None:
            mask = get_mask(S, tail)
            S_grad = S_grad * mask

        # Scale for reconstruction
        scale = torch.where(S > tiny, S_grad / torch.clamp(S, min=tiny), torch.zeros_like(S))

    elif norm_type == "frobenius":
        # Frobenius norm: chain rule with ||sigma||_2
        frobenius_norm = torch.sqrt(torch.sum(S**2, dim=-1, keepdim=True))
        frobenius_norm = torch.maximum(frobenius_norm, torch.tensor(tiny * 100, device=S.device))

        # h'(||sigma||_2)
        h_prime = grad_func(frobenius_norm.squeeze(-1), eps)

        # Chain rule: h'(||sigma||_2) * sigma / ||sigma||_2
        grad_frob = S / frobenius_norm
        S_grad_final = h_prime.unsqueeze(-1) * grad_frob

        # Scale for reconstruction
        scale = torch.where(S > tiny, S_grad_final / torch.clamp(S, min=tiny), torch.zeros_like(S))

    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

    # Reconstruct gradient matrix
    D = (B * scale[..., None, :]) @ B.transpose(-1, -2)
    return D @ M if order == 1 else M @ D


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

        if self.smoothing_function == "fair":
            smoothing_func = fair
        elif self.smoothing_function == "charbonnier":
            smoothing_func = charbonnier
        elif self.smoothing_function == "perona_malik":
            smoothing_func = perona_malik
        else:
            smoothing_func = nothing

        out = norm(x, self.norm, smoothing_func, order, self.eps, self.tail)
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
        # Compute gradient for nuclear or Frobenius norm
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
        out = gradient_norm(x, self.norm, grad_func, self.eps, order, self.tail)
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
                x,
                tail=self.tail,
                smoothing_function=self.smoothing_function,
                order=1 if Mdim <= Ddim else 0,
            )
            # smoothing epsilon (self.eps) is only needed inside hessian_diag;
            # in the helper we passed 0.0; if you want it used, swap the call to
            # route self.eps there (or keep as-is if your diag funcs read it differently).
            return torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(
                rank_one, nan=0.0, posinf=0.0, neginf=0.0
            )

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
        return torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(
            rank_one, nan=0.0, posinf=0.0, neginf=0.0
        )
