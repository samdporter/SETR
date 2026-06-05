import torch

SQRT3 = 1.7320508075688772


def eigenvalsh_2x2(H, eps=1e-12):
    """Stable 2x2 eigenvalue computation."""
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 1, 0]
    d = H[..., 1, 1]

    # Enforce symmetry
    bc = 0.5 * (b + c)

    half_trace = 0.5 * (a + d)
    det = a * d - bc**2
    disc = torch.clamp(half_trace * half_trace - det, min=0.0)
    root = torch.sqrt(disc)

    lam1 = torch.clamp(half_trace + root, min=0.0)
    lam2 = torch.clamp(half_trace - root, min=0.0)

    return torch.sort(torch.stack([lam1, lam2], dim=-1), dim=-1)[0]


def eigenvecsh_2x2(H, eigenvalues, eps=1e-9):
    """Robust 2x2 symmetric eigenvectors with ascending-eigenvalue ordering."""
    H = 0.5 * (H + H.transpose(-1, -2))
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 1, 0]
    d = H[..., 1, 1]
    bc = 0.5 * (b + c)

    lam_lo = eigenvalues[..., 0]
    lam_hi = eigenvalues[..., 1]

    # Adaptive tolerance avoids near-degenerate float32 instabilities.
    tol0 = torch.as_tensor(eps, device=H.device, dtype=H.dtype)
    dyn = 32.0 * torch.finfo(H.dtype).eps * (torch.abs(a) + torch.abs(d) + 2.0 * torch.abs(bc) + 1.0)
    tol = torch.maximum(tol0, dyn)
    repeated = torch.abs(lam_hi - lam_lo) <= tol

    # Compute eigenvector for the largest eigenvalue from the better-conditioned row.
    cand1 = torch.stack([bc, lam_hi - a], dim=-1)
    cand2 = torch.stack([lam_hi - d, bc], dim=-1)
    n1 = torch.linalg.norm(cand1, dim=-1)
    n2 = torch.linalg.norm(cand2, dim=-1)
    v_hi = torch.where((n1 >= n2).unsqueeze(-1), cand1, cand2)

    # Degenerate fallback: choose the dominant diagonal axis.
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    fallback_hi = torch.stack(
        [torch.where(a >= d, ones, zeros), torch.where(a >= d, zeros, ones)],
        dim=-1,
    )
    bad = torch.linalg.norm(v_hi, dim=-1) <= tol
    v_hi = torch.where(bad.unsqueeze(-1), fallback_hi, v_hi)
    v_hi = v_hi / torch.clamp(torch.linalg.norm(v_hi, dim=-1, keepdim=True), min=torch.finfo(H.dtype).tiny)

    # Enforce exact orthogonality for the second eigenvector.
    v_lo = torch.stack([-v_hi[..., 1], v_hi[..., 0]], dim=-1)
    vecs = torch.stack([v_lo, v_hi], dim=-1)  # columns correspond to [lam_lo, lam_hi]

    identity = torch.eye(2, device=H.device, dtype=H.dtype).expand(*H.shape)
    return torch.where(repeated.unsqueeze(-1).unsqueeze(-1), identity, vecs)


def _safe_norm(v, eps=1e-30):
    return torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def eigenvalsh_3x3_cardano(H):
    """
    Batched analytic eigenvalues for real-symmetric 3x3.
    Returns ascending eigenvalues (..., 3).
    """
    H = 0.5 * (H + H.transpose(-1, -2))  # enforce symmetry
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    n = H.shape[-1]
    tr = torch.clamp(torch.diagonal(H, dim1=-2, dim2=-1).sum(-1), min=torch.finfo(H.dtype).tiny)
    alpha = (tr / n)[..., None, None]
    H = (H / alpha).contiguous()

    a = H[..., 0, 0]
    d = H[..., 1, 1]
    f = H[..., 2, 2]
    b = H[..., 0, 1]
    c = H[..., 0, 2]
    e = H[..., 1, 2]

    c2 = -(a + d + f)
    c1 = a * d + a * f + d * f - (b * b + c * c + e * e)
    c0 = a * e * e + d * c * c + f * b * b - a * d * f - 2.0 * b * c * e

    p = c2 * c2 - 3.0 * c1
    q = -13.5 * c0 - 0.5 * c2 * c2 * c2 + 4.5 * c2 * c1

    p_clamp = torch.clamp(p, min=0.0)
    disc = torch.clamp(p_clamp * p_clamp * p_clamp - q * q, min=0.0)
    phi = (1.0 / 3.0) * torch.atan2(torch.sqrt(disc), q)

    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    x1 = 2.0 * cphi
    x2 = -cphi - SQRT3 * sphi
    x3 = -cphi + SQRT3 * sphi

    scale = torch.sqrt(p_clamp) * (1.0 / 3.0)
    shift = -c2 * (1.0 / 3.0)
    L = torch.stack([scale * x1 + shift, scale * x2 + shift, scale * x3 + shift], dim=-1)
    L_sorted, _ = torch.sort(L, dim=-1)

    # scale back and clamp small negatives
    L_sorted = torch.clamp(L_sorted * (tr / n)[..., None], min=0.0)
    return L_sorted


@torch.no_grad()
def eigenvecsh_3x3_cardano(H, L_sorted):
    I = torch.eye(3, dtype=H.dtype, device=H.device)
    I = I.view(*((1,) * (H.ndim - 2)), 3, 3)

    H = 0.5 * (H + H.transpose(-1, -2))  # enforce symmetry
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    def eigenvec_for_lambda(lam):
        B = H - lam[..., None, None] * I
        c0 = B[..., :, 0]
        c1 = B[..., :, 1]
        c2v = B[..., :, 2]

        v01 = torch.cross(c0, c1, dim=-1)
        v02 = torch.cross(c0, c2v, dim=-1)
        v12 = torch.cross(c1, c2v, dim=-1)

        cand = torch.stack([v01, v02, v12], dim=-2)  # (...,3,3)
        norms = torch.linalg.norm(cand, dim=-1)  # (...,3)

        imax = norms.argmax(dim=-1)
        v = torch.gather(
            cand, dim=-2, index=imax[..., None, None].expand(*cand.shape[:-2], 1, cand.shape[-1])
        ).squeeze(-2)

        tiny = 1e-30
        bad = norms.max(dim=-1).values < 1e-18
        if bad.any():
            cols = torch.stack([c0, c1, c2v], dim=-1)  # (...,3,3)
            col_norms = torch.linalg.norm(cols, dim=-2)
            j = col_norms.argmax(dim=-1)  # (...,)
            basis = torch.eye(3, dtype=H.dtype, device=H.device)
            e = basis[j]  # (...,3)
            col = torch.gather(cols, -1, j[..., None, None].expand(*cols.shape[:-2], 3, 1)).squeeze(
                -1
            )
            fallback = e - (e * col).sum(dim=-1, keepdim=True) * col / _safe_norm(col)
            v = torch.where(bad[..., None], fallback, v)

        return v / _safe_norm(v, eps=tiny)

    v1 = eigenvec_for_lambda(L_sorted[..., 0])
    v2 = eigenvec_for_lambda(L_sorted[..., 1])
    v2 = (v2 - (v2 * v1).sum(dim=-1, keepdim=True) * v1) / _safe_norm(
        v2 - (v2 * v1).sum(dim=-1, keepdim=True) * v1
    )
    v3 = torch.cross(v1, v2, dim=-1)
    v3 = v3 / _safe_norm(v3)

    V = torch.stack([v1, v2, v3], dim=-1)
    V = orthonormalize_columns_chol(V, tol=1e-6)
    return V


def orthonormalize_columns_chol(V, tol=1e-6):
    """
    Make columns of V (...,3,3) exactly orthonormal using
    V <- V @ R^{-1}, where R = chol(V^T V) (upper).
    Only applies the fix where ||V^T V - I||_inf > tol.
    """
    # Gram matrix
    G = V.transpose(-2, -1) @ V  # (...,3,3)

    # Deviation from orthonormality
    I = torch.eye(3, dtype=V.dtype, device=V.device).expand_as(G)
    dev = (G - I).abs().amax(dim=(-2, -1))  # (...,)

    need = dev > tol
    if not need.any():
        return V  # already orthonormal enough

    Vfix = V[need]
    Gfix = G[need]

    # Robust Cholesky (no throws); info>0 means not SPD
    R, info = torch.linalg.cholesky_ex(Gfix, upper=True)

    # Fallback to QR only for genuinely bad cases
    bad = info > 0
    if bad.any():
        Vbad = Vfix[bad]
        Vbad, _ = torch.linalg.qr(Vbad, mode="reduced")
        Vfix = Vfix.clone()
        Vfix[bad] = Vbad
        # Recompute G/R on the good subset
        good = ~bad
        if good.any():
            R = torch.linalg.cholesky(Gfix[good], upper=True)
            # Solve (R^T) X^T = V^T  -> X = V @ R^{-1}
            Xt = torch.linalg.solve_triangular(
                R.transpose(-2, -1), Vfix[good].transpose(-2, -1), upper=False
            )
            Vfix[good] = Xt.transpose(-2, -1)
        V = V.clone()
        V[need] = Vfix
        return V

    # Fast path: all SPD → one triangular solve
    Xt = torch.linalg.solve_triangular(
        R.transpose(-2, -1), Vfix.transpose(-2, -1), upper=False
    )  # solves (R^T) X^T = V^T
    Vfix = Xt.transpose(-2, -1)
    V = V.clone()
    V[need] = Vfix
    return V


def adaptive_regularization(H, base_eps=1e-7):
    n = H.shape[-1]

    H_safe = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    # Eigenvalue computation (unchanged)
    if n == 2:
        eigs = eigenvalsh_2x2(H_safe)
    elif n == 3:
        eigs = eigenvalsh_3x3_cardano(H_safe)
    else:
        eigs = torch.linalg.eigvalsh(H_safe)

    finfo = torch.finfo(H.dtype)
    lam_min = eigs[..., 0].clamp_min(finfo.tiny)
    lam_max = eigs[..., -1].clamp_min(0.0)
    kappa = lam_max / lam_min

    eps_geometric = base_eps * torch.sqrt(lam_min * lam_max)
    eps_tikhonov = torch.sqrt(lam_min * finfo.eps)
    eps_val = torch.maximum(eps_geometric, eps_tikhonov)

    eps_min = finfo.eps * n
    eps_val = eps_val.clamp_min(eps_min)

    return eps_val, kappa


def adaptive_gram_regularization(M, order=None):
    """
    Constructs regularized Gram matrix with adaptive scaling.

    Mathematical guarantee: κ(H̃) ≤ κ_max with minimal perturbation.
    """
    if order is None:
        order = 1 if M.shape[-2] <= M.shape[-1] else 0

    # Compute Gram matrix
    H = M @ M.transpose(-1, -2) if order == 1 else M.transpose(-1, -2) @ M
    H = 0.5 * (H + H.transpose(-1, -2))  # Enforce symmetry

    # Adaptive regularization
    ε, κ = adaptive_regularization(H)

    # Apply scaled identity
    n = H.shape[-1]
    I = torch.eye(n, dtype=H.dtype, device=H.device)
    H_reg = H + ε[..., None, None] * I

    return H_reg, ε, κ
