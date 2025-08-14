# schatten_norm_gpu_slow.py

import numpy as np
import torch
from cil.optimisation.functions import Function

from .common import (
    to_tensor,
    l1_norm_torch,
    l1_norm_prox_torch,
    l2_norm_torch,
    l2_norm_prox_torch,
    charbonnier_torch,
    charbonnier_grad_torch,
    charbonnier_hessian_surrogate,
    fair_torch,
    fair_grad_torch,
    fair_hessian_surrogate,
    perona_malik_torch,
    perona_malik_grad_torch,
    perona_malik_hessian_surrogate,
    nothing_torch,
    nothing_grad_torch,
    get_sv_tail_mask,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUVectorialTotalVariation(Function):
    """
    GPU implementation of the vectorial total variation function.
    """

    def __init__(
        self,
        eps=None,
        norm="nuclear",
        smoothing_function=None,
        numpy_out=True,
        tail=None,
    ):
        if eps is not None:
            self.eps = torch.tensor(eps, device=device)
        else:
            self.eps = torch.tensor(0.0, device=device)
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out
        self.tail = tail

    def direct(self, x):
        # --- Select appropriate functions ---
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

        # --- Efficient Batched Calculation ---
        # 1. Compute singular values for all voxels
        S = torch.linalg.svdvals(x)  # S shape: (nx, ny, nz, min(M,d))

        # 2. Apply tailing if specified
        S_tailed = S * get_sv_tail_mask(S, self.tail) if self.tail is not None else S
        # 3. Apply smoothing function h(s)
        s_smoothed = smoothing_func(S_tailed, self.eps)

        # 4. Apply norm function (e.g., sum for nuclear norm)
        out = norm_func(s_smoothed)

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        x = to_tensor(x)
        val = self.direct(x).sum()
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, eps):
        x = to_tensor(x)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.norm == "nuclear":
            prox_func = l1_norm_prox_torch
        elif self.norm == "frobenius":
            prox_func = l2_norm_prox_torch
        else:
            raise ValueError("Norm not defined")

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Apply proximal operator h_prox(s)
        S_prox_values = prox_func(S, eps)

        # If tailing, only apply prox to the tail values.
        if self.tail is not None:
            mask = get_sv_tail_mask(S, self.tail)
            # Combine original head with processed tail
            S_final = S * (1 - mask) + S_prox_values * mask
        else:
            S_final = S_prox_values

        out = torch.matmul(U, torch.matmul(torch.diag_embed(S_final), Vh))

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        x = to_tensor(x)

        if self.smoothing_function == "fair":
            grad_func = fair_grad_torch
        elif self.smoothing_function == "charbonnier":
            grad_func = charbonnier_grad_torch
        elif self.smoothing_function == "perona_malik":
            grad_func = perona_malik_grad_torch
        else:  # Default to non-smoothed
            # The gradient of h(s)=s is h'(s)=1
            grad_func = nothing_grad_torch

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Apply gradient function h'(s)
        S_grad_values = grad_func(S, self.eps)

        # If tailing, the gradient for non-tailed values is 0
        mask = torch.ones_like(S) if self.tail is None else get_sv_tail_mask(S, self.tail)
        S_grad_values = S_grad_values * mask

        # Reconstruct the gradient matrix: U diag(h'(s)) V^T
        out = torch.matmul(U, torch.matmul(torch.diag_embed(S_grad_values), Vh))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    def hessian_surrogate(self, x):
        x = to_tensor(x)

        if self.smoothing_function == "fair":
            hessian_func = fair_hessian_surrogate
        elif self.smoothing_function == "charbonnier":
            hessian_func = charbonnier_hessian_surrogate
        elif self.smoothing_function == "perona_malik":
            hessian_func = perona_malik_hessian_surrogate
        else:
            raise ValueError("Unknown smoothing function")

        S = torch.linalg.svdvals(x)

        mask = torch.ones_like(S) if self.tail is None else get_sv_tail_mask(S, self.tail)
        # Compute the Hessian surrogate
        out = hessian_func(S, self.eps)

        return out * mask
