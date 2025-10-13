# schatten_norm_gpu_slow.py

import numpy as np
import torch
from cil.optimisation.functions import Function

from .common import (
    charbonnier,
    charbonnier_grad,
    charbonnier_hessian_diag,
    charbonnier_hessian_surrogate,
    fair,
    fair_grad,
    fair_hessian_diag,
    fair_hessian_surrogate,
    get_mask,
    l1_norm,
    l1_norm_prox,
    l2_norm,
    l2_norm_prox,
    nothing,
    nothing_grad,
    nothing_hessian_diag,
    perona_malik,
    perona_malik_grad,
    perona_malik_hessian_diag,
    perona_malik_hessian_surrogate,
    to_tensor,
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
        if self.smoothing_function == "fair":
            smoothing_func = fair
        elif self.smoothing_function == "charbonnier":
            smoothing_func = charbonnier
        elif self.smoothing_function == "perona_malik":
            smoothing_func = perona_malik
        else:
            smoothing_func = nothing

        S = torch.linalg.svdvals(x)

        if self.norm == "nuclear":
            # Nuclear norm: sum_i h(sigma_i)
            if self.tail is not None:
                mask = get_mask(S, self.tail)
                s_smoothed = smoothing_func(S * mask, self.eps)
                s_to_sum = s_smoothed + S * (1 - mask)
            else:
                s_to_sum = smoothing_func(S, self.eps)
            out = torch.sum(s_to_sum, dim=-1)

        elif self.norm == "frobenius":
            # Frobenius norm: h(||sigma||_2) = h(sqrt(sum_i sigma_i^2))
            frobenius_norm = torch.sqrt(torch.sum(S**2, dim=-1))
            if self.tail is not None:
                # For tailing with Frobenius, we need to handle this differently
                # Apply smoothing to the Frobenius norm itself
                out = smoothing_func(frobenius_norm, self.eps)
            else:
                out = smoothing_func(frobenius_norm, self.eps)
        else:
            raise ValueError("Norm not defined")

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
            prox_func = l1_norm_prox
        elif self.norm == "frobenius":
            prox_func = l2_norm_prox
        else:
            raise ValueError("Norm not defined")

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Apply proximal operator h_prox(s)
        S_prox_values = prox_func(S, eps)

        # If tailing, only apply prox to the tail values.
        if self.tail is not None:
            mask = get_mask(S, self.tail)
            # Combine original head with processed tail
            S_final = S * (1 - mask) + S_prox_values * mask
        else:
            S_final = S_prox_values

        out = torch.matmul(U, Vh * S_final[..., None])

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        x = to_tensor(x)

        if self.smoothing_function == "fair":
            grad_func = fair_grad
        elif self.smoothing_function == "charbonnier":
            grad_func = charbonnier_grad
        elif self.smoothing_function == "perona_malik":
            grad_func = perona_malik_grad
        else:  # Default to non-smoothed
            # The gradient of h(s)=s is h'(s)=1
            grad_func = nothing_grad

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        if self.norm == "nuclear":
            # Nuclear norm: gradient of sum_i h(sigma_i)
            # Gradient: U diag(h'(sigma_i)) V^T
            S_grad_values = grad_func(S, self.eps)

            # If tailing, the gradient for non-tailed values is 0
            mask = torch.ones_like(S) if self.tail is None else get_mask(S, self.tail)
            S_grad_values = S_grad_values * mask

            # Reconstruct the gradient matrix: U diag(h'(s)) V^T
            out = torch.matmul(U, Vh * S_grad_values[..., None])

        elif self.norm == "frobenius":
            # Frobenius norm: gradient of h(||sigma||_2)
            # Chain rule: h'(||sigma||_2) * grad(||sigma||_2)
            #           = h'(||sigma||_2) * sigma / ||sigma||_2
            frobenius_norm = torch.sqrt(torch.sum(S**2, dim=-1, keepdim=True))
            # Avoid division by zero
            frobenius_norm = torch.maximum(frobenius_norm, torch.tensor(1e-10, device=S.device))

            # h'(||sigma||_2) - scalar for each voxel
            h_prime = grad_func(frobenius_norm.squeeze(-1), self.eps)

            # grad(||sigma||_2) w.r.t sigma = sigma / ||sigma||_2
            grad_frob_norm = S / frobenius_norm

            # Chain rule: h'(||sigma||_2) * sigma / ||sigma||_2
            S_grad_values = h_prime.unsqueeze(-1) * grad_frob_norm

            # Reconstruct the gradient matrix
            out = torch.matmul(U, Vh * S_grad_values[..., None])

        else:
            raise ValueError("Norm not defined")

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

        mask = torch.ones_like(S) if self.tail is None else get_mask(S, self.tail)
        # Compute the Hessian surrogate
        out = hessian_func(S, self.eps)

        return torch.nan_to_num(out * mask, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_components(self, x):
        """
        Calculates the components needed for the diagonal preconditioner.
        Returns the building blocks for the calling class to use.

        Returns:
            tuple: A tuple containing:
                - hess_coeffs (torch.Tensor): h''(s_k), shape (..., num_singular_values).
                - rank_one_fields (torch.Tensor): u_k v_k^T, shape (..., num_singular_values, M, d).
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.smoothing_function == "fair":
            hessian_diag_func = fair_hessian_diag
        elif self.smoothing_function == "charbonnier":
            hessian_diag_func = charbonnier_hessian_diag
        elif self.smoothing_function == "perona_malik":
            hessian_diag_func = perona_malik_hessian_diag
        else:
            hessian_diag_func = nothing_hessian_diag

        # --- Step 1: Perform SVD on the entire field of matrices ---
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        # U shape: (..., M, r), S shape: (..., r), Vh shape: (..., r, d)

        # --- Step 2: Calculate Hessian coefficients h''(s_k) ---
        hess_coeffs = hessian_diag_func(S, self.eps)  # Shape: (..., r)

        mask = get_mask(S, self.tail)
        hess_coeffs = hess_coeffs * mask

        # --- Step 3: Construct the field of rank-1 basis matrices u_k v_k^T ---
        # Target shape: (..., r, M, d)

        # U shape is (..., M, r). We need k to be an outer dimension.
        U_perm = U.permute(*range(U.ndim - 2), -1, -2)  # Swap last two dims -> (..., r, M)

        # Unsqueeze to prepare for batched matrix multiplication (outer product)
        # U_perm becomes (..., r, M, 1)
        # Vh becomes     (..., r, 1, d)
        U_unsqueezed = U_perm.unsqueeze(-1)
        Vh_unsqueezed = Vh.unsqueeze(-2)

        # This performs a batch of r outer products for each voxel
        rank_one_fields = U_unsqueezed @ Vh_unsqueezed  # Shape: (..., r, M, d)

        return hess_coeffs, rank_one_fields
