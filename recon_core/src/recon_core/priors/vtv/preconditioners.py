"""
VTV preconditioners (active implementations).

Notes on math (MM weights):
- For smoothed nuclear norm phi(Y)=tr((Y^T Y + eps^2 I)^{1/2}) (Charbonnier),
  the MM/IRLS weight matrix is W=(Y Y^T + eps^2 I)^{-1/2} in modality space.
- The quadratic surrogate uses tr(W Y Y^T); no extra 1/2 factor is folded into W.
- Gershgorin inflation is applied to W (row-sum) to obtain a diagonal majoriser.

If you need legacy/obsolete methods, see preconditioners_old.py and preconditioner_old.py.
"""

from __future__ import annotations

import torch

from .numerical_constants import get_division_epsilon
from .vtv import (
    _direction_valid_mask,
    _directional_projector_stats_from_jacobian,
    _shift_with_zeros,
)


_CANONICAL_DIAG_METHODS = {"mm_diag_tight", "mm_diag_gershgorin_maj"}
_CANONICAL_BLOCK_METHODS = {"mm_diag_block_maj", "mm_diag_block_tight"}

_DIAG_METHODS = _CANONICAL_DIAG_METHODS
_BLOCK_METHODS = _CANONICAL_BLOCK_METHODS
_ALL_METHODS = _DIAG_METHODS | _BLOCK_METHODS


def _canonical_method_name(method: str) -> str:
    return method


def _smoothing_grad(smoothing: str):
    from .common import (
        charbonnier_grad,
        fair_grad,
        nothing_grad,
        perona_malik_grad,
    )

    if smoothing == "charbonnier":
        return charbonnier_grad
    if smoothing == "fair":
        return fair_grad
    if smoothing == "perona_malik":
        return perona_malik_grad
    return nothing_grad


def _aggregate_directional_sensitivity(wvtv, x_arr, like):
    """Aggregate per-direction sensitivity for diagonal preconditioners."""
    S = wvtv.jacobian.sensitivity(x_arr)
    S = torch.as_tensor(S, device=like.device, dtype=like.dtype)
    if S.ndim < like.ndim:
        S = S.expand_as(like)
    S_squared = S * S

    nx, ny, nz, M, d = like.shape

    # Periodic boundary conditions should not underweight boundary voxels.
    base_grad = wvtv.jacobian.grad[0] if isinstance(wvtv.jacobian.grad, list) else wvtv.jacobian.grad
    base_grad = getattr(base_grad, "gradient", base_grad)
    bnd_cond = getattr(base_grad, "bnd_cond", "Neumann")
    if bnd_cond == "Periodic":
        participation = torch.full((nx, ny, nz, d), 2.0, device=like.device, dtype=like.dtype)
    else:
        participation = wvtv._compute_directional_participation_counts(
            (nx, ny, nz, d), like.device, like.dtype
        )
    participation = participation.unsqueeze(-2).expand(nx, ny, nz, M, d)

    d_diag, _ = _directional_projector_stats_from_jacobian(wvtv.jacobian, like, M, d)
    return (S_squared * participation * d_diag).sum(dim=-1)


def _mm_block_weight(A, smoothing: str, eps):
    """Return W = (A A^T + eps^2 I)^{-1/2} (2x2 blocks) via eigen decomposition."""
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

    eps_t = torch.as_tensor(float(eps), device=A.device, dtype=A.dtype)
    sigma1 = torch.sqrt(lam1)
    sigma2 = torch.sqrt(lam2)

    if smoothing == "charbonnier":
        w1 = torch.reciprocal(torch.sqrt(lam1 + eps_t * eps_t))
        w2 = torch.reciprocal(torch.sqrt(lam2 + eps_t * eps_t))
    else:
        phi1 = _smoothing_grad(smoothing)
        eps_div = get_division_epsilon(A.dtype)
        w1 = phi1(sigma1, eps_t) / (sigma1 + eps_div)
        w2 = phi1(sigma2, eps_t) / (sigma2 + eps_div)

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


def _directional_projector_vectors_from_jacobian(jacobian, like, n_params: int, n_dirs: int):
    """Return per-modality projector vectors and gammas for D_m = I - gamma_m xi_m xi_m^T."""
    jac_grad = getattr(jacobian, "grad", None)
    if jac_grad is None:
        jac_grad = [None]
    elif isinstance(jac_grad, list) and len(jac_grad) == 0:
        jac_grad = [None]
    grads = jac_grad if isinstance(jac_grad, list) else [jac_grad]

    def _vectors_for_grad(g):
        if g is None or not hasattr(g, "anatomical_grad"):
            xi = torch.zeros((*like.shape[:-2], n_dirs), device=like.device, dtype=like.dtype)
            gamma = torch.zeros((*like.shape[:-2],), device=like.device, dtype=like.dtype)
            return xi, gamma

        anat = torch.as_tensor(g.anatomical_grad, device=like.device, dtype=like.dtype)
        den = torch.norm(anat, p=2, dim=-1, keepdim=True)
        eta = g.eta if hasattr(g, "eta") else torch.tensor(0.0, device=like.device)
        eta = torch.as_tensor(eta, device=like.device, dtype=like.dtype)
        xi = anat / torch.sqrt(den * den + eta * eta)
        gamma = g.gamma if hasattr(g, "gamma") else torch.tensor(1.0, device=like.device)
        gamma = torch.as_tensor(gamma, device=like.device, dtype=like.dtype)
        return xi, gamma

    xi_list = []
    gamma_list = []
    for idx in range(n_params):
        g = grads[idx] if len(grads) > 1 else grads[0]
        xi, gamma = _vectors_for_grad(g)
        xi_list.append(xi)
        gamma_list.append(gamma)

    xi_all = torch.stack(xi_list, dim=-2)  # (..., M, d)
    gamma_all = torch.stack(gamma_list, dim=-1)  # (..., M)
    return xi_all, gamma_all


def _pair_alpha_from_c(c, xi, gamma):
    """
    Compute alpha^{mm'} = c^T D_m D_{m'} c with D_m = I - gamma_m xi_m xi_m^T.

    This is exact for the implemented directional projector model and supports
    modality-specific projector vectors xi_m and gammas.
    """
    c_norm_sq = torch.sum(c * c, dim=-1)  # (...)
    xi_dot_c = torch.einsum("...md,...d->...m", xi, c)  # (..., M)
    xi_gram = torch.einsum("...md,...nd->...mn", xi, xi)  # (..., M, M)

    m_dim = xi.shape[-2]
    rows = []
    for m in range(m_dim):
        gm = gamma[..., m]
        xm = xi_dot_c[..., m]
        cols = []
        for n in range(m_dim):
            gn = gamma[..., n]
            xn = xi_dot_c[..., n]
            xmn = xi_gram[..., m, n]
            alpha = c_norm_sq - gm * (xm * xm) - gn * (xn * xn) + gm * gn * xmn * xm * xn
            cols.append(alpha)
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


def _map_mm_block_projector_to_voxels(wvtv, x_arr, like, W, epsilon: float):
    """
    Method-2 style tight block-Jacobi pullback in image space.

    Uses alpha^{mm'} = c^T D_m D_{m'} c for:
      - centre stencil vector c_{l,l}
      - forward-neighbour vectors c_{l,l+e_r}
    """
    nx, ny, nz, M, d = like.shape
    if M != 2:
        raise ValueError("mm_diag_block_tight requires exactly 2 modalities.")

    base_grad = wvtv.jacobian.grad[0] if isinstance(wvtv.jacobian.grad, list) else wvtv.jacobian.grad
    base_grad = getattr(base_grad, "gradient", base_grad)
    directions = list(getattr(base_grad, "directions", []))
    bnd_cond = getattr(base_grad, "bnd_cond", "Neumann")

    if len(directions) == 0:
        out = torch.zeros((*like.shape[:-2], 2, 2), device=like.device, dtype=like.dtype)
        eye = torch.eye(2, device=out.device, dtype=out.dtype)
        return out + epsilon * eye

    q_scale = torch.as_tensor(wvtv.jacobian.sensitivity(x_arr), device=like.device, dtype=like.dtype)
    if q_scale.ndim < like.ndim:
        q_scale = q_scale.expand_as(like)

    valid_mask = _direction_valid_mask((nx, ny, nz), directions, like.device, like.dtype, bnd_cond)
    if valid_mask is not None:
        q_scale = q_scale * valid_mask.unsqueeze(-2)

    xi, gamma = _directional_projector_vectors_from_jacobian(wvtv.jacobian, like, M, d)

    # Sensitivity should be modality-independent for a shared gradient stencil.
    # We use the mean across modalities to stay robust to small numeric discrepancies.
    c_scale = torch.mean(q_scale, dim=-2)  # (..., d)
    weights = wvtv.weights.to(like.device, dtype=like.dtype)
    weight_outer = weights.unsqueeze(-1) * weights.unsqueeze(-2)  # (..., 2, 2)
    bwb = weight_outer * W

    # Anchor contribution: c_{l,l} = -q for valid forward stencil channels.
    c_anchor = -c_scale
    alpha_anchor = _pair_alpha_from_c(c_anchor, xi, gamma)  # (..., 2, 2)
    p_block = bwb * alpha_anchor

    for ch, sh in enumerate(directions):
        c_edge = torch.zeros_like(c_scale)
        c_edge[..., ch] = c_scale[..., ch]
        alpha_edge = _pair_alpha_from_c(c_edge, xi, gamma)  # (..., 2, 2)
        edge_block = bwb * alpha_edge
        if bnd_cond == "Periodic":
            p_block = p_block + torch.roll(edge_block, shifts=sh, dims=(0, 1, 2))
        else:
            p_block = p_block + _shift_with_zeros(edge_block, sh)

    p_block = 0.5 * (p_block + p_block.transpose(-1, -2))
    p_block = torch.nan_to_num(p_block, nan=0.0, posinf=0.0, neginf=0.0)
    return _spd_floor_blocks(p_block, epsilon)


def _map_mm_block_majoriser_to_voxels(wvtv, x_arr, like, W, epsilon: float):
    """
    Method-1 fully-majorising block pullback from dtnv_preconditioner_corrected.tex.

    Builds Khat_{l,r} then applies voxelwise splitting:
        M_j = 2 * sum_r (S_{j,r} + S_{j-e_r,r}),  S_{l,r} = q_{l,r}^2 * Khat_{l,r}.
    """
    nx, ny, nz, M, d = like.shape
    if M != 2:
        raise ValueError("mm_diag_block_maj requires exactly 2 modalities.")

    base_grad = wvtv.jacobian.grad[0] if isinstance(wvtv.jacobian.grad, list) else wvtv.jacobian.grad
    base_grad = getattr(base_grad, "gradient", base_grad)
    directions = list(getattr(base_grad, "directions", []))
    bnd_cond = getattr(base_grad, "bnd_cond", "Neumann")

    if len(directions) == 0:
        out = torch.zeros((*like.shape[:-2], 2, 2), device=like.device, dtype=like.dtype)
        eye = torch.eye(2, device=out.device, dtype=out.dtype)
        return out + epsilon * eye

    q_scale = torch.as_tensor(wvtv.jacobian.sensitivity(x_arr), device=like.device, dtype=like.dtype)
    if q_scale.ndim < like.ndim:
        q_scale = q_scale.expand_as(like)

    valid_mask = _direction_valid_mask((nx, ny, nz), directions, like.device, like.dtype, bnd_cond)
    if valid_mask is not None:
        q_scale = q_scale * valid_mask.unsqueeze(-2)

    xi, gamma = _directional_projector_vectors_from_jacobian(wvtv.jacobian, like, M, d)
    # Method-1 derivation assumes a shared guidance direction across modalities.
    # In standard DTNV runs this holds (single anatomical guidance input).
    xi_common = xi[..., 0, :]
    # Keep xi unmasked at boundaries: stencil validity is already encoded in q/c
    # coefficients, while D_m D_m' in the derivation uses the full xi vector.

    # Use mean sensitivity per direction as the shared stencil coefficient magnitude.
    q_common = torch.mean(q_scale, dim=-2)  # (..., d)

    weights = wvtv.weights.to(like.device, dtype=like.dtype)
    weight_outer = weights.unsqueeze(-1) * weights.unsqueeze(-2)  # (..., 2, 2)
    omega = weight_outer * W  # Omega = B W B with spatially varying modality weights

    xi_norm_sq = torch.sum(xi_common * xi_common, dim=-1)  # (...)
    g0 = gamma[..., 0]
    g1 = gamma[..., 1]
    mu00 = g0 + g0 - g0 * g0 * xi_norm_sq
    mu11 = g1 + g1 - g1 * g1 * xi_norm_sq
    mu01 = g0 + g1 - g0 * g1 * xi_norm_sq

    phi = torch.zeros_like(omega)
    phi[..., 0, 0] = omega[..., 0, 0] * mu00
    phi[..., 1, 1] = omega[..., 1, 1] * mu11
    phi01 = omega[..., 0, 1] * mu01
    phi[..., 0, 1] = phi01
    phi[..., 1, 0] = phi01
    phi = 0.5 * (phi + phi.transpose(-1, -2))
    phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    phi_eigs = torch.linalg.eigvalsh(phi)
    phi_norm = torch.maximum(torch.abs(phi_eigs[..., 0]), torch.abs(phi_eigs[..., 1]))

    xi_abs = torch.abs(xi_common)
    xi_abs_sum = torch.sum(xi_abs, dim=-1)  # (...)

    s_dir = torch.zeros((*like.shape[:-2], d, 2, 2), device=like.device, dtype=like.dtype)
    eye = torch.eye(2, device=like.device, dtype=like.dtype)
    for ch in range(d):
        xi_r = xi_common[..., ch]
        xi_r_abs = xi_abs[..., ch]
        lambda_r = xi_abs_sum - xi_r_abs
        infl = xi_r_abs * lambda_r * phi_norm

        khat = omega - (xi_r * xi_r).unsqueeze(-1).unsqueeze(-1) * phi
        khat = khat + infl.unsqueeze(-1).unsqueeze(-1) * eye
        s_dir[..., ch, :, :] = (q_common[..., ch] * q_common[..., ch]).unsqueeze(-1).unsqueeze(-1) * khat

    p_block = torch.zeros((*like.shape[:-2], 2, 2), device=like.device, dtype=like.dtype)
    for ch, sh in enumerate(directions):
        t = s_dir[..., ch, :, :]
        p_block = p_block + t
        if bnd_cond == "Periodic":
            p_block = p_block + torch.roll(t, shifts=sh, dims=(0, 1, 2))
        else:
            p_block = p_block + _shift_with_zeros(t, sh)
    p_block = 2.0 * p_block
    p_block = 0.5 * (p_block + p_block.transpose(-1, -2))
    p_block = torch.nan_to_num(p_block, nan=0.0, posinf=0.0, neginf=0.0)
    return _spd_floor_blocks(p_block, epsilon)


def compute_precond_diag(wvtv, x_arr, method: str, epsilon: float = 1e-8):
    """
    Diagonal preconditioners in image space.

    Canonical methods from dtnv_preconditioner_corrected.tex:
    - mm_diag_tight: diagonal-only pullback (tight, not a Loewner majoriser in general).
    - mm_diag_gershgorin_maj: Gershgorin-inflated diagonal majoriser.
    """
    method = _canonical_method_name(method)
    if method not in _DIAG_METHODS:
        raise ValueError(f"Unknown diagonal preconditioner method: {method}.")

    J = wvtv.jacobian.direct(x_arr)
    weights = wvtv.weights.to(J.device, dtype=J.dtype)
    A = weights.unsqueeze(-1) * J

    S_jm = _aggregate_directional_sensitivity(wvtv, x_arr, A)
    w2 = weights * weights

    if method in {"mm_diag_tight", "mm_diag_gershgorin_maj"}:
        W = _mm_block_weight(A, wvtv.smoothing, wvtv.vtv.eps)
        w11 = W[..., 0, 0]
        w22 = W[..., 1, 1]

        if method == "mm_diag_tight":
            D = torch.stack([w11, w22], dim=-1)
        else:
            w12 = torch.abs(W[..., 0, 1])
            D = torch.stack([w11 + w12, w22 + w12], dim=-1)

        P_diag = D * S_jm * w2
        P_diag = torch.nan_to_num(P_diag, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(P_diag, min=epsilon)

    raise ValueError(f"Unhandled diagonal preconditioner method: {method}.")


def _ls_hessian_matrix(J, eps):
    """Legacy LS helper forwarded to preconditioners_old."""
    from .preconditioners_old import _ls_hessian_matrix as _old_ls_hessian_matrix

    return _old_ls_hessian_matrix(J, eps)


def _ls_blocks_from_hessian(H, d):
    """Legacy LS helper forwarded to preconditioners_old (diag extraction)."""
    from .preconditioners_old import _ls_blocks_from_hessian as _old_ls_blocks_from_hessian

    return _old_ls_blocks_from_hessian(H, d, method="ls_block_diag")


def _ls_blocks_lowmem(J, eps):
    """Legacy LS helper forwarded to preconditioners_old (diag extraction)."""
    from .preconditioners_old import _ls_blocks_lowmem as _old_ls_blocks_lowmem

    return _old_ls_blocks_lowmem(J, eps, method="ls_block_diag")


def compute_precond_block(wvtv, x_arr, method: str, epsilon: float = 1e-8):
    """
    Block (2x2) preconditioners in image space.

    Canonical methods from dtnv_preconditioner_corrected.tex:
    - mm_diag_block_maj: fully-majorising block surrogate (Method 1 chain).
    - mm_diag_block_tight: tight block-Jacobi pullback (Method 2 style).
    """
    method = _canonical_method_name(method)
    if method not in _BLOCK_METHODS:
        raise ValueError(f"Unknown block preconditioner method: {method}.")

    J = wvtv.jacobian.direct(x_arr)
    if J.shape[-2] != 2:
        raise ValueError("Block preconditioners require exactly 2 modalities.")

    weights = wvtv.weights.to(J.device, dtype=J.dtype)
    A = weights.unsqueeze(-1) * J

    if method == "mm_diag_block_maj":
        W = _mm_block_weight(A, wvtv.smoothing, wvtv.vtv.eps)
        return _map_mm_block_majoriser_to_voxels(wvtv, x_arr, A, W, epsilon)

    if method == "mm_diag_block_tight":
        W = _mm_block_weight(A, wvtv.smoothing, wvtv.vtv.eps)
        return _map_mm_block_projector_to_voxels(wvtv, x_arr, A, W, epsilon)

    raise ValueError(f"Unhandled block preconditioner method: {method}.")


__all__ = [
    "compute_precond_diag",
    "compute_precond_block",
    "_ALL_METHODS",
    "_DIAG_METHODS",
    "_BLOCK_METHODS",
]
