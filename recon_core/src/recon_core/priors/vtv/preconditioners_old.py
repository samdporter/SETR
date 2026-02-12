"""
Deprecated VTV preconditioner implementations preserved for reference.

This module contains legacy MM/LS variants and older method names that are
no longer active in the main preconditioner dispatch. Do not import from
production code; use recon_core.priors.vtv.preconditioners instead.
"""

import torch

from .common import (
    charbonnier_grad,
    fair_grad,
    nothing_grad,
    perona_malik_grad,
)
from .numerical_constants import get_division_epsilon, get_sqrt_epsilon
from .vtv import (
    _accumulate_over_neighborhood_block,
    _direction_valid_mask,
    _directional_projector_stats_from_jacobian,
)


_DIAG_METHODS = {"frob_diag_pd", "mm_diag_jensen", "mm_diag_gershgorin"}
_BLOCK_METHODS = {"mm_block_2x2", "ls_block_gershgorin", "ls_block_diag"}
_ALL_METHODS = _DIAG_METHODS | _BLOCK_METHODS


def _smoothing_grad(smoothing: str):
    if smoothing == "charbonnier":
        return charbonnier_grad
    if smoothing == "fair":
        return fair_grad
    if smoothing == "perona_malik":
        return perona_malik_grad
    return nothing_grad


def _aggregate_directional_sensitivity(wvtv, x_arr, like):
    S = wvtv.jacobian.sensitivity(x_arr)
    S = torch.as_tensor(S, device=like.device, dtype=like.dtype)
    if S.ndim < like.ndim:
        S = S.expand_as(like)
    S_squared = S * S

    nx, ny, nz, M, d = like.shape
    participation = wvtv._compute_directional_participation_counts(
        (nx, ny, nz, d), like.device, like.dtype
    )
    participation = participation.unsqueeze(-2).expand(nx, ny, nz, M, d)

    d_diag, _ = _directional_projector_stats_from_jacobian(wvtv.jacobian, like, M, d)
    return (S_squared * participation * d_diag).sum(dim=-1)


def _mm_block_weight(A, smoothing: str, eps):
    """Compute W = h(S) for 2x2 Gram S = A A^T with h(lambda)=phi'(sqrt(lambda))/(2 sqrt(lambda))."""
    if A.shape[-2] != 2:
        raise ValueError("MM block weight requires exactly 2 modalities.")

    S = torch.matmul(A, A.transpose(-2, -1))
    a = S[..., 0, 0]
    b = S[..., 0, 1]
    c = S[..., 1, 1]

    tau = a + c
    delta = torch.sqrt(torch.clamp((a - c) ** 2 + 4.0 * b * b, min=0.0))
    lam1 = 0.5 * (tau + delta)
    lam2 = 0.5 * (tau - delta)
    lam1 = torch.clamp(lam1, min=0.0)
    lam2 = torch.clamp(lam2, min=0.0)

    eps_sqrt = get_sqrt_epsilon(A.dtype)
    sigma1 = torch.sqrt(lam1 + eps_sqrt)
    sigma2 = torch.sqrt(lam2 + eps_sqrt)
    phi1 = _smoothing_grad(smoothing)
    eps_t = torch.as_tensor(eps, device=A.device, dtype=A.dtype)
    eps_div = get_division_epsilon(A.dtype)
    w1 = phi1(sigma1, eps_t) / (2.0 * sigma1 + eps_div)
    w2 = phi1(sigma2, eps_t) / (2.0 * sigma2 + eps_div)

    diff = lam1 - lam2
    tol = get_division_epsilon(A.dtype)
    v = (w1 - w2) / diff
    u = (lam1 * w2 - lam2 * w1) / diff

    v = torch.where(diff.abs() > tol, v, torch.zeros_like(v))
    u = torch.where(diff.abs() > tol, u, w1)

    w11 = u + v * a
    w22 = u + v * c
    w12 = v * b
    return torch.stack(
        [torch.stack([w11, w12], dim=-1), torch.stack([w12, w22], dim=-1)], dim=-2
    )


def _spd_floor_blocks(blocks, epsilon: float):
    eye = torch.eye(2, device=blocks.device, dtype=blocks.dtype)
    eps_t = torch.as_tensor(epsilon, device=blocks.device, dtype=blocks.dtype)
    p11 = blocks[..., 0, 0]
    p22 = blocks[..., 1, 1]
    p12 = blocks[..., 0, 1]
    trace = p11 + p22
    diff = p11 - p22
    disc = torch.sqrt(torch.clamp(0.25 * diff * diff + p12 * p12, min=0.0))
    lambda_min = 0.5 * trace - disc
    shift = torch.clamp(eps_t - lambda_min, min=0.0)
    return blocks + shift.unsqueeze(-1).unsqueeze(-1) * eye


def _directional_scaling_blocks(wvtv, x_arr, like):
    base_grad = wvtv.jacobian.grad[0] if isinstance(wvtv.jacobian.grad, list) else wvtv.jacobian.grad
    base_grad = getattr(base_grad, "gradient", base_grad)
    directions = list(getattr(base_grad, "directions", []))
    bnd_cond = getattr(base_grad, "bnd_cond", "Neumann")

    if len(directions) == 0:
        return None, directions, bnd_cond

    nx, ny, nz, M, d = like.shape
    valid_mask = _direction_valid_mask((nx, ny, nz), directions, like.device, like.dtype, bnd_cond)
    q_scale = torch.as_tensor(wvtv.jacobian.sensitivity(x_arr), device=like.device, dtype=like.dtype)
    if valid_mask is not None:
        q_scale = q_scale * valid_mask.unsqueeze(-2)

    d_diag, _ = _directional_projector_stats_from_jacobian(wvtv.jacobian, like, M, d)

    weights = wvtv.weights.to(like.device, dtype=like.dtype)
    w0 = weights[..., 0]
    w1 = weights[..., 1]
    s1 = q_scale[..., 0, :]
    s2 = q_scale[..., 1, :]
    d1 = d_diag[..., 0, :]
    d2 = d_diag[..., 1, :]

    k1 = (w0 * w0).unsqueeze(-1) * (s1 * s1) * d1
    k2 = (w1 * w1).unsqueeze(-1) * (s2 * s2) * d2
    k12 = (w0 * w1).unsqueeze(-1) * (s1 * s2) * torch.sqrt(
        torch.clamp(d1 * d2, min=0.0)
    )

    k = torch.stack(
        [torch.stack([k1, k12], dim=-1), torch.stack([k12, k2], dim=-1)], dim=-2
    )
    return k, directions, bnd_cond


def _map_edge_blocks_to_voxels(wvtv, x_arr, like, edge_blocks, epsilon: float):
    k, directions, bnd_cond = _directional_scaling_blocks(wvtv, x_arr, like)
    if k is None:
        out = torch.zeros((*like.shape[:-2], 2, 2), device=like.device, dtype=like.dtype)
        eye = torch.eye(2, device=out.device, dtype=out.dtype)
        return out + epsilon * eye

    if edge_blocks.ndim == k.ndim - 1:
        edge_blocks = edge_blocks.unsqueeze(-3).expand_as(k)

    t = torch.matmul(k, edge_blocks)
    t = 0.5 * (t + t.transpose(-1, -2))
    p_block = _accumulate_over_neighborhood_block(t, directions, bnd_cond)
    p_block = 0.5 * (p_block + p_block.transpose(-1, -2))
    p_block = torch.nan_to_num(p_block, nan=0.0, posinf=0.0, neginf=0.0)
    return _spd_floor_blocks(p_block, epsilon)


def compute_precond_diag(wvtv, x_arr, method: str, epsilon: float = 1e-8):
    if method not in _DIAG_METHODS:
        raise ValueError(f"Unknown diagonal preconditioner method: {method}.")

    J = wvtv.jacobian.direct(x_arr)
    weights = wvtv.weights.to(J.device, dtype=J.dtype)
    A = weights.unsqueeze(-1) * J

    if method == "mm_diag_jensen":
        A_frob_sq = torch.sum(A * A, dim=(-2, -1))
        r = min(A.shape[-2], A.shape[-1])
        sigma_avg_sq = A_frob_sq / r
        sigma_avg = torch.sqrt(sigma_avg_sq + get_sqrt_epsilon(A.dtype))

        phi1 = _smoothing_grad(wvtv.smoothing)
        phi_prime = phi1(sigma_avg, wvtv.vtv.eps)

        eps_div = get_division_epsilon(A.dtype)
        omega = phi_prime / (2.0 * sigma_avg + eps_div)

        S_jm = _aggregate_directional_sensitivity(wvtv, x_arr, A)
        P_diag = omega.unsqueeze(-1) * S_jm * (weights * weights)
        P_diag = torch.nan_to_num(P_diag, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(P_diag, min=epsilon)

    if method == "frob_diag_pd":
        A_frob_sq = torch.sum(A * A, dim=(-2, -1))
        A_frob = torch.sqrt(A_frob_sq + get_sqrt_epsilon(A.dtype))
        phi1 = _smoothing_grad(wvtv.smoothing)
        phi_prime = phi1(A_frob, wvtv.vtv.eps)

        eps_floor = torch.as_tensor(float(wvtv.vtv.eps), device=A.device, dtype=A.dtype)
        A_frob_safe = torch.maximum(A_frob, eps_floor)
        M = A.shape[-2]
        omega = M * phi_prime / A_frob_safe

        S_jm = _aggregate_directional_sensitivity(wvtv, x_arr, A)
        P_diag = omega.unsqueeze(-1) * S_jm * (weights * weights)
        P_diag = torch.nan_to_num(P_diag, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(P_diag, min=epsilon)

    if method == "mm_diag_gershgorin":
        W = _mm_block_weight(A, wvtv.smoothing, wvtv.vtv.eps)
        w12_abs = torch.abs(W[..., 0, 1])
        D = torch.stack([W[..., 0, 0] + w12_abs, W[..., 1, 1] + w12_abs], dim=-1)
        S_jm = _aggregate_directional_sensitivity(wvtv, x_arr, A)
        P_diag = D * S_jm * (weights * weights)
        P_diag = torch.nan_to_num(P_diag, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(P_diag, min=epsilon)

    raise ValueError(f"Unhandled diagonal preconditioner method: {method}.")


def _build_dilation_eig(J):
    d = J.shape[-2]
    U, S, Vh = torch.linalg.svd(J, full_matrices=True)
    V = Vh.transpose(-2, -1)

    inv_sqrt2 = torch.tensor(1.0 / (2.0 ** 0.5), device=J.device, dtype=J.dtype)
    r = S.shape[-1]
    q_parts = []
    lambda_parts = []

    for k in range(r):
        u_k = U[..., :, k]
        v_k = V[..., :, k]
        q_parts.append(torch.cat([u_k, v_k], dim=-1) * inv_sqrt2)
        lambda_parts.append(S[..., k])

    for k in range(r):
        u_k = U[..., :, k]
        v_k = V[..., :, k]
        q_parts.append(torch.cat([u_k, -v_k], dim=-1) * inv_sqrt2)
        lambda_parts.append(-S[..., k])

    if d > r:
        u_extra = U[..., :, r:]
        zeros = torch.zeros(
            (*u_extra.shape[:-2], 2, u_extra.shape[-1]), device=J.device, dtype=J.dtype
        )
        q_parts.append(torch.cat([u_extra, zeros], dim=-2))
        lambda_parts.append(
            torch.zeros((*S.shape[:-1], u_extra.shape[-1]), device=J.device, dtype=J.dtype)
        )

    if V.shape[-1] > r:
        v_extra = V[..., :, r:]
        zeros = torch.zeros(
            (*v_extra.shape[:-2], d, v_extra.shape[-1]), device=J.device, dtype=J.dtype
        )
        q_parts.append(torch.cat([zeros, v_extra], dim=-2))
        lambda_parts.append(
            torch.zeros((*S.shape[:-1], v_extra.shape[-1]), device=J.device, dtype=J.dtype)
        )

    q_blocks = [q.unsqueeze(-1) if q.ndim == J.ndim - 1 else q for q in q_parts]
    Q = torch.cat(q_blocks, dim=-1)

    lambda_blocks = [
        lam.unsqueeze(-1) if lam.ndim == J.ndim - 2 else lam for lam in lambda_parts
    ]
    lambdas = torch.cat(lambda_blocks, dim=-1)
    return Q, lambdas


def _ls_hessian_matrix(J, eps):
    d = J.shape[-2]
    n = d + 2
    Q, lambdas = _build_dilation_eig(J)
    Q_t = Q.transpose(-1, -2)

    eps_t = torch.as_tensor(float(eps), device=J.device, dtype=J.dtype)
    denom = torch.sqrt(lambdas * lambdas + eps_t * eps_t)
    psi_prime = lambdas / denom
    psi_double = (eps_t * eps_t) / (denom * denom * denom)

    li = lambdas.unsqueeze(-1)
    lj = lambdas.unsqueeze(-2)
    diff = li - lj

    eps_div = get_division_epsilon(J.dtype)
    psi_prime_i = psi_prime.unsqueeze(-1)
    psi_prime_j = psi_prime.unsqueeze(-2)
    psi_double_i = psi_double.unsqueeze(-1)
    C = torch.where(diff.abs() > eps_div, (psi_prime_i - psi_prime_j) / diff, psi_double_i)
    eye = torch.eye(n, device=J.device, dtype=torch.bool)
    C = C.masked_fill(eye, 0.0)

    H = torch.zeros((*J.shape[:-2], 2 * d, 2 * d), device=J.device, dtype=J.dtype)
    for q in range(d):
        for beta in range(2):
            H_basis = torch.zeros((n, n), device=J.device, dtype=J.dtype)
            H_basis[q, d + beta] = 1.0
            H_basis[d + beta, q] = 1.0

            tilde = Q_t @ H_basis @ Q
            diag_tilde = torch.diagonal(tilde, dim1=-2, dim2=-1)
            M = torch.diag_embed(psi_double * diag_tilde) + C * tilde
            delta_G = Q @ M @ Q_t
            delta_Y = delta_G[..., :d, d:]

            col = q * 2 + beta
            H[..., :, col] = delta_Y.reshape(*J.shape[:-2], 2 * d)

    H = 0.5 * (H + H.transpose(-1, -2))
    return H


def _ls_blocks_from_hessian(H, d, method: str):
    H_blocks = H.reshape(*H.shape[:-2], d, 2, d, 2)
    B = torch.zeros((*H.shape[:-2], d, 2, 2), device=H.device, dtype=H.dtype)
    eye = torch.eye(2, device=H.device, dtype=H.dtype)

    for p in range(d):
        H_pp = H_blocks[..., p, :, p, :]
        H_pp = 0.5 * (H_pp + H_pp.transpose(-1, -2))
        if method == "ls_block_diag":
            B[..., p, :, :] = H_pp
            continue

        if method == "ls_block_gershgorin":
            off_sum = torch.zeros(H_pp.shape[:-2], device=H.device, dtype=H.dtype)
            for q in range(d):
                if q == p:
                    continue
                H_pq = H_blocks[..., p, :, q, :]
                sig = torch.linalg.svdvals(H_pq)
                off_sum = off_sum + sig[..., 0]
            B[..., p, :, :] = H_pp + off_sum.unsqueeze(-1).unsqueeze(-1) * eye
            continue

        raise ValueError(f"Unknown LS block method: {method}.")

    return B


def _ls_blocks_lowmem(J, eps, method: str):
    if method not in {"ls_block_diag", "ls_block_gershgorin"}:
        raise ValueError(f"Unknown LS block method: {method}.")

    d = J.shape[-2]
    n = d + 2
    Q, lambdas = _build_dilation_eig(J)
    Q_t = Q.transpose(-1, -2)

    eps_t = torch.as_tensor(float(eps), device=J.device, dtype=J.dtype)
    denom = torch.sqrt(lambdas * lambdas + eps_t * eps_t)
    psi_prime = lambdas / denom
    psi_double = (eps_t * eps_t) / (denom * denom * denom)

    li = lambdas.unsqueeze(-1)
    lj = lambdas.unsqueeze(-2)
    diff = li - lj

    eps_div = get_division_epsilon(J.dtype)
    psi_prime_i = psi_prime.unsqueeze(-1)
    psi_prime_j = psi_prime.unsqueeze(-2)
    psi_double_i = psi_double.unsqueeze(-1)
    C = torch.where(diff.abs() > eps_div, (psi_prime_i - psi_prime_j) / diff, psi_double_i)
    eye = torch.eye(n, device=J.device, dtype=torch.bool)
    C = C.masked_fill(eye, 0.0)

    B = torch.zeros((*J.shape[:-2], d, 2, 2), device=J.device, dtype=J.dtype)
    off_sum = None
    if method == "ls_block_gershgorin":
        off_sum = torch.zeros((*J.shape[:-2], d), device=J.device, dtype=J.dtype)

    for q in range(d):
        delta_cols = []
        for beta in range(2):
            H_basis = torch.zeros((n, n), device=J.device, dtype=J.dtype)
            H_basis[q, d + beta] = 1.0
            H_basis[d + beta, q] = 1.0

            tilde = Q_t @ H_basis @ Q
            diag_tilde = torch.diagonal(tilde, dim1=-2, dim2=-1)
            M = torch.diag_embed(psi_double * diag_tilde) + C * tilde
            delta_G = Q @ M @ Q_t
            delta_Y = delta_G[..., :d, d:]
            delta_cols.append(delta_Y)

        blocks_q = torch.stack(delta_cols, dim=-1)
        B[..., q, :, :] = blocks_q[..., q, :, :]

        if off_sum is not None:
            sig = torch.linalg.svdvals(blocks_q)
            sigma_max = sig[..., 0]
            mask = torch.ones((d,), device=J.device, dtype=J.dtype)
            mask[q] = 0.0
            off_sum = off_sum + sigma_max * mask

    B = 0.5 * (B + B.transpose(-1, -2))
    if off_sum is not None:
        eye2 = torch.eye(2, device=J.device, dtype=J.dtype)
        B = B + off_sum.unsqueeze(-1).unsqueeze(-1) * eye2
    return B


def compute_precond_block(wvtv, x_arr, method: str, epsilon: float = 1e-8):
    if method not in _BLOCK_METHODS:
        raise ValueError(f"Unknown block preconditioner method: {method}.")

    J = wvtv.jacobian.direct(x_arr)
    if J.shape[-2] != 2:
        raise ValueError("Block preconditioners require exactly 2 modalities.")

    weights = wvtv.weights.to(J.device, dtype=J.dtype)
    A = weights.unsqueeze(-1) * J
    d = A.shape[-1]

    if method == "mm_block_2x2":
        W = _mm_block_weight(A, wvtv.smoothing, wvtv.vtv.eps)
        return _map_edge_blocks_to_voxels(wvtv, x_arr, A, W, epsilon)

    if method in {"ls_block_diag", "ls_block_gershgorin"}:
        J_mat = A.transpose(-2, -1)
        B_dir = _ls_blocks_lowmem(J_mat, wvtv.vtv.eps, method)
        return _map_edge_blocks_to_voxels(wvtv, x_arr, A, B_dir, epsilon)

    raise ValueError(f"Unhandled block preconditioner method: {method}.")


__all__ = [
    "compute_precond_diag",
    "compute_precond_block",
    "_ls_hessian_matrix",
    "_ls_blocks_from_hessian",
    "_ALL_METHODS",
    "_DIAG_METHODS",
    "_BLOCK_METHODS",
]
