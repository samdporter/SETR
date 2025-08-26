import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=torch.float32)
    else:
        return x.to(device, dtype=torch.float32)

def pseudo_inverse(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))


def l1_norm(x):
    return torch.sum(torch.abs(x), dim=-1)


def l1_norm_prox(x, eps):
    return torch.sign(x) * torch.clamp(torch.abs(x) - eps, min=0)


def l2_norm(x):
    return torch.sqrt(torch.sum(x**2, dim=-1))

def l2_norm_prox(x, eps):

    eps_unsqueezed = eps.unsqueeze(-1)
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    norms = torch.maximum(norms, torch.tensor(1e-9, device=x.device))
    factor = torch.clamp(norms - eps_unsqueezed, min=0.0) / norms
    return x * factor


def charbonnier(x, eps):
    return torch.sqrt(x**2 + eps**2) - eps


def charbonnier_grad(x, eps):
    # Add small epsilon to denominator for stability
    return x / torch.sqrt(x**2 + eps**2)


def charbonnier_hessian_surrogate(x, eps):
    return 0.5 / torch.sqrt(x**2 + eps**2)


def charbonnier_hessian_diag(x, eps):
    return eps ** 2 / (x ** 2 + eps ** 2) ** 1.5


def charbonnier_inv_hessian_diag(x, eps):
    return (x ** 2 + eps ** 2) ** 1.5 / (eps**2)


def fair(x, eps):
    return eps * (torch.abs(x) / (eps) - torch.log1p(torch.abs(x) / (eps)))


def fair_grad(x, eps):
    return x / (eps + torch.abs(x))


def fair_hessian_surrogate(x, eps):
    return 0.5 / (eps + torch.abs(x))


def fair_hessian_diag(x, eps):
    return eps / (eps + torch.abs(x)) ** 2


def fair_inv_hessian_diag(x, eps):
    return (eps + torch.abs(x)) ** 2 / (eps)


def perona_malik(x, eps):
    return (eps / 2) * (1 - torch.exp(-(x**2) / (eps**2)))


def perona_malik_grad(x, eps):
    return x * torch.exp(-(x**2) / (eps**2)) / (eps**2)

def perona_malik_hessian_surrogate(x, eps):
    return  0.5 * torch.exp(-(x**2) / (eps**2)) / (eps**2)


def perona_malik_hessian_diag(x, eps):
    return (eps ** 2 - 2 * x ** 2) * torch.exp(-x ** 2 / (eps ** 2)) / (eps ** 3)


def perona_malik_inv_hessian_diag(x, eps):
    return (eps ** 3) * torch.exp(x ** 2 / (eps ** 2)) / (eps ** 2 - 2 * x ** 2)


def nothing(x, eps=0):
    return x


def nothing_grad(x, eps=0):
    return torch.ones_like(x)

def nothing_hessian_diag(x, eps=0):
    return torch.zeros_like(x)

def get_mask(S, tail: int):
    """
    Returns a mask (..., r) with 1's on the smallest `tail` singular values of S (..., r).
    """
    if tail is None or tail == S.shape[-1]:
        return torch.ones_like(S)
    S_sorted, sort_idx = torch.sort(S, dim=-1)  # ascending
    r = S.shape[-1]
    ones_tail = torch.ones((*S.shape[:-1], tail), device=S.device, dtype=S.dtype)
    zeros_head = torch.zeros((*S.shape[:-1], r - tail), device=S.device, dtype=S.dtype)
    mask_sorted = torch.cat([ones_tail, zeros_head], dim=-1)
    inv = torch.argsort(sort_idx, dim=-1)
    return torch.gather(mask_sorted, dim=-1, index=inv)


def add_identity(H, rel=1e-7):
    # rel is relative to the average diagonal scale
    n = H.shape[-1]
    tr = torch.clamp(torch.diagonal(H, dim1=-2, dim2=-1).sum(-1), min=torch.finfo(H.dtype).tiny)
    jitter = rel * (tr / n)                        # scale with matrix size
    I = torch.eye(n, dtype=H.dtype, device=H.device).expand_as(H)
    return (H + jitter[..., None, None] * I).contiguous()
