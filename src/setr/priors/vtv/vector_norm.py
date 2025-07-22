# gpu_total_variation.py

from cil.optimisation.functions import Function

try:
    import torch
    from torch import vmap
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    device = 'cpu'
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))


def l1_norm_torch(x):
    return torch.sum(torch.abs(x), dim=-1)


def l1_norm_prox_torch(x, eps):
    return torch.sign(x) * torch.clamp(torch.abs(x) - eps, min=0)


def l2_norm_torch(x):
    return torch.sqrt(torch.sum(x ** 2, dim=-1))


def l2_norm_prox_torch(x, eps):
    # Unsqueeze eps for broadcasting
    eps_unsqueezed = eps.unsqueeze(-1)
    
    # Calculate norms along the last dimension
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    # Avoid division by zero
    norms = torch.maximum(norms, torch.tensor(1e-9, device=x.device))
    
    # Calculate scaling factor
    factor = torch.clamp(norms - eps_unsqueezed, min=0.0) / norms
    return x * factor


def charbonnier_torch(x, eps):
    return torch.sqrt(x ** 2 + eps ** 2) - eps


def charbonnier_grad_torch(x, eps):
    # Add small epsilon to denominator for stability
    return x / torch.sqrt(x ** 2 + eps ** 2)


def charbonnier_hessian_diag_torch(x, eps):
    # Returns g''(x) for Charbonnier: eps²/(x² + eps²)^(3/2)
    return eps ** 2 / (x ** 2 + eps ** 2) ** 1.5


def fair_torch(x, eps):
    return eps * (torch.abs(x) / eps - torch.log1p(torch.abs(x) / eps))


def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x))


def fair_hessian_diag_torch(x, eps):
    # Returns g''(x) for Fair: eps/(eps + |x|)^2
    return eps / (eps + torch.abs(x)) ** 2


def perona_malik_torch(x, eps):
    return (eps / 2) * (1 - torch.exp(-x ** 2 / (eps ** 2)))


def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-x ** 2 / (eps ** 2)) / (eps ** 2)


def perona_malik_hessian_diag_torch(x, eps):
    # Returns g''(x) for Perona-Malik: (eps² − 2x²) e^(−x²/eps²)/eps³
    return (eps ** 2 - 2 * x ** 2) * torch.exp(-x ** 2 / (eps ** 2)) / (eps ** 3)


def nothing_torch(x, eps=0):
    return x


def nothing_grad_torch(x, eps=0):
    return torch.ones_like(x)


def nothing_hessian_diag_torch(x, eps=0):
    return torch.zeros_like(x)


class GPUVectorNorm(Function):
    """
    GPU implementation of smoothed total variation applied separately to each modality.
    
    Input: Jacobian tensor of shape (nx, ny, nz, M, d) where:
    - (nx, ny, nz) are spatial dimensions
    - M is number of modalities
    - d is number of finite difference directions
    """
    def __init__(self, eps=None, norm='l2', smoothing_function=None, numpy_out=True):
        
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
        if self.norm == 'l1':
            norm_func = l1_norm_torch
        elif self.norm == 'l2':
            norm_func = l2_norm_torch
        else:
            raise ValueError('Norm not defined')

        if self.smoothing_function == 'fair':
            smoothing_func = fair_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbonnier_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_torch
        else:
            smoothing_func = nothing_torch

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

        if self.norm == 'l1':
            prox_func = l1_norm_prox_torch
        elif self.norm == 'l2':
            prox_func = l2_norm_prox_torch
        else:
            raise ValueError('Norm not defined')

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

        if self.norm == 'l1':
            norm_func = l1_norm_torch
        elif self.norm == 'l2':
            norm_func = l2_norm_torch
        else:
            raise ValueError('Norm not defined')

        if self.smoothing_function == 'fair':
            grad_func = fair_grad_torch
        elif self.smoothing_function == 'charbonnier':
            grad_func = charbonnier_grad_torch
        elif self.smoothing_function == 'perona_malik':
            grad_func = perona_malik_grad_torch
        else:
            grad_func = nothing_grad_torch

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
            if self.norm == 'l1':
                # For L1 norm: ∂||g||₁/∂g = sign(g)
                gradient[..., m, :] = smooth_deriv.unsqueeze(-1) * torch.sign(modality_gradients)
            elif self.norm == 'l2':
                # For L2 norm: ∂||g||₂/∂g = g/||g||₂
                gradient[..., m, :] = (smooth_deriv / grad_mag_safe).unsqueeze(-1) * modality_gradients
        
        return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    def phi_hessian(self, U):
        """
        Compute φ''(‖u‖) for each vector u in U.
        U: tensor of shape (..., d)
        returns: tensor of shape (...)
        """
        # flatten all but last dim
        orig_shape, d = U.shape[:-1], U.shape[-1]
        U_flat = U.reshape(-1, d)

        # per‐vector φ'' using your existing torch kernels
        def phi2_vec(u):
            r = torch.linalg.norm(u)
            if   self.smoothing_function == 'charbonnier':
                return charbonnier_hessian_diag_torch(r, self.eps)
            elif self.smoothing_function == 'fair':
                return fair_hessian_diag_torch(r, self.eps)
            elif self.smoothing_function == 'perona_malik':
                return perona_malik_hessian_diag_torch(r, self.eps)
            else:
                return nothing_hessian_diag_torch(r, self.eps)

        # vmap over the batch axis
        phi2_flat = vmap(phi2_vec)(U_flat)   # shape (prod(orig_shape),)
        return phi2_flat.reshape(*orig_shape)