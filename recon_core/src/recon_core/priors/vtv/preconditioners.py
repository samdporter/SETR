"""
VTV preconditioners (active implementations).

Notes on math (MM weights):
- For smoothed nuclear norm phi(Y)=tr((Y^T Y + eps^2 I)^{1/2}) (Charbonnier),
  the MM/IRLS weight matrix is W=(Y Y^T + eps^2 I)^{-1/2} in modality space.
- The quadratic surrogate uses tr(W Y Y^T); no extra 1/2 factor is folded into W.
- Gershgorin inflation is applied to W (row-sum) to obtain a diagonal majoriser.

Notes on math (Lewis-Sendov block methods):
- "ls_block_diag" and "ls_block_gershgorin" build the full Lewis-Sendov Hessian
  of the smoothed singular-value density via the Hermitian dilation of the
  per-voxel Jacobian (see tnv_preconditioners_1_.md, Sections E-H), then
  extract per-direction 2x2 diagonal blocks. "ls_block_diag" is a curvature
  estimate (not a majoriser); "ls_block_gershgorin" adds spectral-norm
  Gershgorin inflation from the off-diagonal blocks to obtain a provable
  Loewner majoriser (Sec. G.4).

If you need legacy/obsolete methods, see preconditioners_old.py and preconditioner_old.py.
"""

from __future__ import annotations

import torch

from .numerical_constants import get_division_epsilon
from .vtv import (
    _accumulate_over_neighborhood_block,
    _direction_valid_mask,
    _directional_projector_stats_from_jacobian,
    _shift_with_zeros,
)


_CANONICAL_DIAG_METHODS = {"mm_diag_tight", "mm_diag_gershgorin_maj"}
_CANONICAL_BLOCK_METHODS = {
    "mm_diag_block_maj",
    "mm_diag_block_tight",
    "ls_block_diag",
    "ls_block_gershgorin",
}

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

    _a = phi[..., 0, 0]; _c = phi[..., 1, 1]; _b = phi[..., 0, 1]
    _mean = 0.5 * (_a + _c)
    _disc = torch.sqrt((0.5 * (_a - _c)) ** 2 + _b ** 2)
    phi_eigs = torch.stack([_mean - _disc, _mean + _disc], dim=-1)
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


def _build_dilation_eig(J):
    """
    Eigendecomposition of the Hermitian dilation of J (per-voxel d x 2 Jacobian).

    Returns (Q, lambdas) with Q orthogonal ((d+2) x (d+2)) and lambdas the
    dilation eigenvalues (+-singular values of J, zero-padded).
    """
    J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
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
    """Full Lewis-Sendov Hessian of rho(J) = sum_k sqrt(sigma_k(J)^2 + eps^2)."""
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


def _ls_blocks_from_hessian(H, d, method: str = "ls_block_diag"):
    """Extract per-direction 2x2 diagonal blocks H_pp from the full LS Hessian."""
    if method not in {"ls_block_diag", "ls_block_gershgorin"}:
        raise ValueError(f"Unknown LS block method: {method}.")

    H_blocks = H.reshape(*H.shape[:-2], d, 2, d, 2)
    B = torch.zeros((*H.shape[:-2], d, 2, 2), device=H.device, dtype=H.dtype)
    eye = torch.eye(2, device=H.device, dtype=H.dtype)

    for p in range(d):
        H_pp = H_blocks[..., p, :, p, :]
        H_pp = 0.5 * (H_pp + H_pp.transpose(-1, -2))
        if method == "ls_block_diag":
            B[..., p, :, :] = H_pp
            continue

        off_sum = torch.zeros(H_pp.shape[:-2], device=H.device, dtype=H.dtype)
        for q in range(d):
            if q == p:
                continue
            H_pq = H_blocks[..., p, :, q, :]
            sig = torch.linalg.svdvals(H_pq)
            off_sum = off_sum + sig[..., 0]
        B[..., p, :, :] = H_pp + off_sum.unsqueeze(-1).unsqueeze(-1) * eye

    return B


def _ls_blocks_lowmem(J, eps, method: str = "ls_block_diag"):
    """
    Per-direction 2x2 LS Hessian diagonal blocks, computed without materialising
    the full (2d x 2d) Hessian.

    "ls_block_gershgorin" additionally inflates each block by the spectral norm
    (largest singular value) of the off-diagonal H_pq blocks (Gershgorin, Sec. G.4).
    """
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


def _directional_scaling_blocks(wvtv, x_arr, like):
    """Per-direction 2x2 scaling blocks k^{(r)} used to pull edge blocks to voxels."""
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


def _map_ls_edge_blocks_to_voxels(wvtv, x_arr, like, edge_blocks, epsilon: float):
    """Pull per-direction LS edge blocks back to per-voxel SPD blocks."""
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


def compute_precond_block(wvtv, x_arr, method: str, epsilon: float = 1e-8):
    """
    Block (2x2) preconditioners in image space.

    Canonical methods from dtnv_preconditioner_corrected.tex:
    - mm_diag_block_maj: fully-majorising block surrogate (Method 1 chain).
    - mm_diag_block_tight: tight block-Jacobi pullback (Method 2 style).

    Lewis-Sendov voxel-block methods (tnv_preconditioners_1_.md, Sections E-H):
    - ls_block_diag: per-direction diagonal blocks of the full LS Hessian
      (curvature estimate, not a majoriser).
    - ls_block_gershgorin: ls_block_diag with Gershgorin spectral-norm
      inflation from off-diagonal blocks (provable Loewner majoriser).
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

    if method in {"ls_block_diag", "ls_block_gershgorin"}:
        J_mat = A.transpose(-2, -1)
        B_dir = _ls_blocks_lowmem(J_mat, wvtv.vtv.eps, method)
        return _map_ls_edge_blocks_to_voxels(wvtv, x_arr, A, B_dir, epsilon)

    raise ValueError(f"Unhandled block preconditioner method: {method}.")


__all__ = [
    "compute_precond_diag",
    "compute_precond_block",
    "_ls_hessian_matrix",
    "_ls_blocks_from_hessian",
    "_ls_blocks_lowmem",
    "_ALL_METHODS",
    "_DIAG_METHODS",
    "_BLOCK_METHODS",
]
