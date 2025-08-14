import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=torch.float32)
    else:
        return x.to(device, dtype=torch.float32)

def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))


def l1_norm_torch(x):
    return torch.sum(torch.abs(x), dim=-1)


def l1_norm_prox_torch(x, eps):
    return torch.sign(x) * torch.clamp(torch.abs(x) - eps, min=0)


def l2_norm_torch(x):
    return torch.sqrt(torch.sum(x**2, dim=-1))

def l2_norm_prox_torch(x, eps):

    eps_unsqueezed = eps.unsqueeze(-1)
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    norms = torch.maximum(norms, torch.tensor(1e-9, device=x.device))
    factor = torch.clamp(norms - eps_unsqueezed, min=0.0) / norms
    return x * factor


def charbonnier_torch(x, eps):
    return torch.sqrt(x**2 + eps**2) - eps


def charbonnier_grad_torch(x, eps):
    # Add small epsilon to denominator for stability
    return x / torch.sqrt(x**2 + eps**2)


def charbonnier_hessian_surrogate(x, eps):
    return 0.5 / torch.sqrt(x**2 + eps**2)


def charbonnier_hessian_diag_torch(x, eps):
    return eps ** 2 / (x ** 2 + eps ** 2) ** 1.5


def charbonnier_inv_hessian_diag_torch(x, eps):
    return (x ** 2 + eps ** 2) ** 1.5 / (eps**2)


def fair_torch(x, eps):
    return eps * (torch.abs(x) / (eps) - torch.log1p(torch.abs(x) / (eps)))


def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x))


def fair_hessian_surrogate(x, eps):
    return 0.5 / (eps + torch.abs(x))


def fair_hessian_diag_torch(x, eps):
    return eps / (eps + torch.abs(x)) ** 2


def fair_inv_hessian_diag_torch(x, eps):
    return (eps + torch.abs(x)) ** 2 / (eps)


def perona_malik_torch(x, eps):
    return (eps / 2) * (1 - torch.exp(-(x**2) / (eps**2)))


def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-(x**2) / (eps**2)) / (eps**2)

def perona_malik_hessian_surrogate(x, eps):
    return  0.5 * torch.exp(-(x**2) / (eps**2)) / (eps**2)


def perona_malik_hessian_diag_torch(x, eps):
    return (eps ** 2 - 2 * x ** 2) * torch.exp(-x ** 2 / (eps ** 2)) / (eps ** 3)


def perona_malik_inv_hessian_diag_torch(x, eps):
    return (eps ** 3) * torch.exp(x ** 2 / (eps ** 2)) / (eps ** 2 - 2 * x ** 2)


def nothing_torch(x, eps=0):
    return x


def nothing_grad_torch(x, eps=0):
    return torch.ones_like(x)

def get_sv_tail_mask(S, tail):
    result = torch.zeros_like(S)
    num_singular_values = S.shape[-1]
    start_index = max(0, num_singular_values - tail)
    result[..., start_index:] = 1.0
    return result
