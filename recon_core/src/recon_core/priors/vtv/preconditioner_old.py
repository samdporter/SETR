import torch

from .numerical_constants import get_sqrt_epsilon
from .vtv import (
    _accumulate_over_neighborhood_block,
    _direction_valid_mask,
    _directional_projector_stats_from_jacobian,
    _smoothing_derivatives,
)


def preconditioner_weights_core_fastest_exact(wvtv, x_arr, epsilon: float = 1e-8):
    """
    Legacy per-modality radial preconditioner (exact, no SVD).
    """
    J = wvtv.jacobian.direct(x_arr)
    A = wvtv.weights.unsqueeze(-1) * J

    r2 = torch.sum(A * A, dim=-1)
    eps_sqrt = get_sqrt_epsilon(A.dtype)
    r = torch.sqrt(r2 + eps_sqrt)

    phi1, phi2 = _smoothing_derivatives(wvtv.smoothing)
    phi1_r = phi1(r, wvtv.vtv.eps)
    phi2_r = phi2(r, wvtv.vtv.eps)
    eps_floor = torch.tensor(float(wvtv.vtv.eps), device=A.device, dtype=A.dtype)
    r_safe = torch.maximum(r, eps_floor)
    alpha = phi1_r / r_safe
    beta = phi2_r - alpha
    frac = (A * A) / (r_safe.unsqueeze(-1) ** 2)
    h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac

    S = wvtv.jacobian.sensitivity(x_arr)
    S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
    if S.ndim < A.ndim:
        S = S.expand_as(A)
    S2 = S * S
    w2 = (wvtv.weights * wvtv.weights).unsqueeze(-1)

    P = torch.sum(w2 * S2 * h_dir, dim=-1)
    return torch.clamp(P, min=epsilon)


def preconditioner_weights_core_slow(wvtv, x_arr):
    """
    Legacy diagonal Hessian via full SVD (principal + isotropic terms).
    """
    J = wvtv.jacobian.direct(x_arr)
    w = wvtv.weights.unsqueeze(-1)
    A_field = w * J

    hess_coeffs, rank_one_fields = wvtv.vtv.hessian_components(A_field)

    P_diag = torch.zeros_like(
        x_arr, device=rank_one_fields.device, dtype=rank_one_fields.dtype
    )
    num_singular_values = rank_one_fields.shape[-3]

    for k in range(num_singular_values):
        C_k_field = rank_one_fields[..., k, :, :]
        w_dev = w.to(C_k_field.device, dtype=C_k_field.dtype)
        influence_image = wvtv.jacobian.adjoint(w_dev * C_k_field)
        influence_image = influence_image.to(P_diag.device, dtype=P_diag.dtype)
        h_double_prime_k = hess_coeffs[..., k].unsqueeze(-1).to(P_diag.device)
        P_diag += h_double_prime_k * (influence_image**2)

    try:
        sigma_weights_half = wvtv.vtv.hessian_surrogate(A_field)
    except Exception:
        sigma_weights_half = None

    if sigma_weights_half is not None:
        alpha_total = (2.0 * torch.sum(sigma_weights_half, dim=-1)).to(P_diag.device)
        S = wvtv.jacobian.sensitivity(x_arr)
        S = torch.as_tensor(S, device=P_diag.device, dtype=P_diag.dtype)
        S2 = S * S
        S_jm = torch.sum(S2, dim=-1)
        w2 = (wvtv.weights * wvtv.weights).to(P_diag.device, dtype=P_diag.dtype)
        P_diag = P_diag + alpha_total.unsqueeze(-1) * w2 * S_jm

    return torch.clamp(P_diag, min=1e-8)


def preconditioner_weights_core_bd_synergy(wvtv, x_arr, epsilon: float = 1e-8):
    """
    Legacy block-diagonal synergy-aware preconditioner (M=2 only).
    """
    j = wvtv.jacobian.direct(x_arr)
    weights = wvtv.weights.to(j.device, dtype=j.dtype)
    if j.shape[-2] != 2:
        raise ValueError("bd_synergy requires exactly 2 modalities.")

    y = weights.unsqueeze(-1) * j
    y1 = y[..., 0, :]
    y2 = y[..., 1, :]

    a = torch.sum(y1 * y1, dim=-1)
    b = torch.sum(y2 * y2, dim=-1)
    c = torch.sum(y1 * y2, dim=-1)

    eps2 = float(wvtv.vtv.eps) ** 2
    a11 = a + eps2
    a22 = b + eps2
    a12 = c
    eps_t = torch.as_tensor(epsilon, device=a11.device, dtype=a11.dtype)
    eps_sq = eps_t * eps_t
    det = a11 * a22 - a12 * a12
    det = torch.clamp(det, min=eps_sq)
    s = torch.sqrt(det)
    t = a11 + a22
    den = torch.sqrt(torch.clamp(t + 2.0 * s, min=eps_t))

    ap11 = a11 + s
    ap22 = a22 + s
    ap12 = a12
    det_ap = ap11 * ap22 - ap12 * ap12
    det_ap = torch.clamp(det_ap, min=eps_sq)
    inv_ap11 = ap22 / det_ap
    inv_ap22 = ap11 / det_ap
    inv_ap12 = -ap12 / det_ap

    w11 = den * inv_ap11
    w22 = den * inv_ap22
    w12 = den * inv_ap12
    w = torch.stack(
        [torch.stack([w11, w12], dim=-1), torch.stack([w12, w22], dim=-1)], dim=-2
    )

    nx, ny, nz, _, d = j.shape
    base_grad = wvtv.jacobian.grad[0] if isinstance(wvtv.jacobian.grad, list) else wvtv.jacobian.grad
    base_grad = getattr(base_grad, "gradient", base_grad)
    directions = list(getattr(base_grad, "directions", []))
    bnd_cond = getattr(base_grad, "bnd_cond", "Neumann")

    if len(directions) == 0:
        out = torch.zeros((*j.shape[:-2], 2, 2), device=j.device, dtype=j.dtype)
        eye = torch.eye(2, device=out.device, dtype=out.dtype)
        return out + epsilon * eye

    valid_mask = _direction_valid_mask((nx, ny, nz), directions, j.device, j.dtype, bnd_cond)
    q_scale = torch.as_tensor(wvtv.jacobian.sensitivity(x_arr), device=j.device, dtype=j.dtype)
    if valid_mask is not None:
        q_scale = q_scale * valid_mask.unsqueeze(-2)

    d_diag, _ = _directional_projector_stats_from_jacobian(wvtv.jacobian, j, 2, d)

    w0 = weights[..., 0]
    w1 = weights[..., 1]
    s1 = q_scale[..., 0, :]
    s2 = q_scale[..., 1, :]
    d1 = d_diag[..., 0, :]
    d2 = d_diag[..., 1, :]

    k1 = (w0 * w0).unsqueeze(-1) * (s1 * s1) * d1
    k2 = (w1 * w1).unsqueeze(-1) * (s2 * s2) * d2
    k12 = (w0 * w1).unsqueeze(-1) * (s1 * s2) * torch.sqrt(torch.clamp(d1 * d2, min=0.0))

    k = torch.stack(
        [torch.stack([k1, k12], dim=-1), torch.stack([k12, k2], dim=-1)], dim=-2
    )

    t = torch.matmul(k, w.unsqueeze(-3))
    t = 0.5 * (t + t.transpose(-1, -2))

    p_block = _accumulate_over_neighborhood_block(t, directions, bnd_cond)
    p_block = 0.5 * (p_block + p_block.transpose(-1, -2))

    eye = torch.eye(2, device=p_block.device, dtype=p_block.dtype)
    eps_t = torch.as_tensor(epsilon, device=p_block.device, dtype=p_block.dtype)
    p11 = p_block[..., 0, 0]
    p22 = p_block[..., 1, 1]
    p12 = p_block[..., 0, 1]
    trace = p11 + p22
    diff = p11 - p22
    disc = torch.sqrt(torch.clamp(0.25 * diff * diff + p12 * p12, min=0.0))
    lambda_min = 0.5 * trace - disc
    shift = torch.clamp(eps_t - lambda_min, min=0.0)
    return p_block + shift.unsqueeze(-1).unsqueeze(-1) * eye


__all__ = [
    "preconditioner_weights_core_fastest_exact",
    "preconditioner_weights_core_slow",
    "preconditioner_weights_core_bd_synergy",
]
