# gpu_total_variation.py

from cil.optimisation.functions import Function

import torch
from functorch import vmap
import numpy as np

from .common import (
    l1_norm,
    l2_norm,
    l1_norm_prox,
    l2_norm_prox,
    fair,
    charbonnier,
    perona_malik,
    nothing,
    fair_grad,
    charbonnier_grad,
    perona_malik_grad,
    nothing_grad,
    fair_hessian_diag,
    charbonnier_hessian_diag,
    perona_malik_hessian_diag,
    nothing_hessian_diag,
    fair_hessian_surrogate,
    charbonnier_hessian_surrogate,
    perona_malik_hessian_surrogate,
    to_tensor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUVectorNorm(Function):
    """
    GPU implementation of smoothed total variation applied separately to each modality.

    Input: Jacobian tensor of shape (nx, ny, nz, M, d) where:
    - (nx, ny, nz) are spatial dimensions
    - M is number of modalities
    - d is number of finite difference directions
    """

    def __init__(self, eps=None, norm="l2", smoothing_function=None, numpy_out=True):
        if eps is not None:
            self.eps = torch.tensor(eps, device=device)
        else:
            self.eps = torch.tensor(1e-6, device=device)

        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out

    def direct(self, x):
        """
        Compute the total variation functional value.

        Args:
            x: Input Jacobian tensor of shape (nx, ny, nz, M, d)

        Returns:
            Total variation value
        """
        # Select appropriate functions
        if self.norm == "l1":
            norm_func = l1_norm
        elif self.norm == "l2":
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

        # x shape: (nx, ny, nz, M, d)
        # We want to process each modality separately

        total_tv = 0.0

        # Process each modality separately
        for m in range(x.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (nx, ny, nz, d)
            modality_gradients = x[..., m, :]

            # Compute gradient magnitude at each voxel: shape (nx, ny, nz)
            grad_magnitudes = norm_func(modality_gradients)

            # Apply smoothing function
            smoothed_grad = smoothing_func(grad_magnitudes, self.eps)

            # Sum over all voxels
            total_tv += torch.sum(smoothed_grad)

        return torch.nan_to_num(total_tv, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        val = self.direct(x)
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, eps):
        """
        Compute the proximal operator of the total variation functional.

        Args:
            x: Input Jacobian tensor of shape (nx, ny, nz, M, d)
            eps: Proximal parameter

        Returns:
            Proximal result of same shape as x
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.norm == "l1":
            prox_func = l1_norm_prox
        elif self.norm == "l2":
            prox_func = l2_norm_prox
        else:
            raise ValueError("Norm not defined")

        out = torch.zeros_like(x)

        # Process each modality separately
        for m in range(x.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (nx, ny, nz, d)
            modality_gradients = x[..., m, :]

            # Apply proximal operator to each voxel's gradient vector
            out[..., m, :] = prox_func(modality_gradients, eps)

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        """
        Compute the gradient of the total variation functional.

        Args:
            x: Input Jacobian tensor of shape (nx, ny, nz, M, d)

        Returns:
            Gradient tensor of same shape as x
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.norm == "l1":
            norm_func = l1_norm
        elif self.norm == "l2":
            norm_func = l2_norm
        else:
            raise ValueError("Norm not defined")

        if self.smoothing_function == "fair":
            grad_func = fair_grad
        elif self.smoothing_function == "charbonnier":
            grad_func = charbonnier_grad
        elif self.smoothing_function == "perona_malik":
            grad_func = perona_malik_grad
        else:
            grad_func = nothing_grad

        gradient = torch.zeros_like(x)

        # Process each modality separately
        for m in range(x.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (nx, ny, nz, d)
            modality_gradients = x[..., m, :]

            # Compute gradient magnitude at each voxel
            grad_magnitudes = norm_func(modality_gradients)

            # Avoid division by zero
            grad_mag_safe = torch.maximum(grad_magnitudes, torch.tensor(1e-9, device=x.device))

            # Compute derivative of smoothing function
            smooth_deriv = grad_func(grad_magnitudes, self.eps)

            # Chain rule: derivative w.r.t. original gradients
            if self.norm == "l1":
                # For L1 norm: ∂||g||₁/∂g = sign(g)
                gradient[..., m, :] = smooth_deriv.unsqueeze(-1) * torch.sign(modality_gradients)
            elif self.norm == "l2":
                # For L2 norm: ∂||g||₂/∂g = g/||g||₂
                gradient[..., m, :] = (smooth_deriv / grad_mag_safe).unsqueeze(
                    -1
                ) * modality_gradients

        return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    def radial_derivatives(self, U, stabiliser: float = 1e-9, positive: bool = True):
        """
        For U[..., d], return (r2, r, alpha, beta) where
        r2 = ||U||^2, r = sqrt(r2 + stabiliser),
        alpha = phi'(r)/r, beta = phi''(r) - alpha.
        """
        U = to_tensor(U)
        r2 = torch.sum(U * U, dim=-1)                  # (...,)
        r  = torch.sqrt(r2 + stabiliser)               # (...,)

        eps = self.eps
        if self.smoothing_function == "charbonnier":
            phi1 = charbonnier_grad
            phi2 = charbonnier_hessian_surrogate if positive else charbonnier_hessian_diag
        elif self.smoothing_function == "fair":
            phi1 = fair_grad
            phi2 = fair_hessian_surrogate if positive else fair_hessian_diag
        elif self.smoothing_function == "perona_malik":
            phi1 = perona_malik_grad
            phi2 = perona_malik_hessian_surrogate if positive else perona_malik_hessian_diag
        else:
            phi1 = nothing_grad
            phi2 = nothing_hessian_diag

        phi1_r = phi1(r, eps)                          # (...,)
        phi2_r = phi2(r, eps)                          # (...,)
        alpha  = phi1_r / r                            # (...,)
        beta   = phi2_r - alpha                        # (...,)
        return r2, r, alpha, beta

    def hessian_dir_diag(self, U, stabiliser: float = 1e-9, positive: bool = True):
        """
        Return per-direction diagonal in U-space:
            h_j = alpha + beta * (u_j^2 / (||u||^2 + stabiliser))
        Shape: input U[..., d] -> output (..., d)
        """
        U = to_tensor(U)
        r2, r, alpha, beta = self.radial_derivatives(U, stabiliser=stabiliser, positive=positive)
        frac = (U * U) / (r2.unsqueeze(-1) + stabiliser)   # (..., d)
        h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac
        return torch.nan_to_num(h_dir, nan=0.0, posinf=0.0, neginf=0.0)

