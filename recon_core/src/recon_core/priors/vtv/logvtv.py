class WeightedLogVectorialTotalVariation(Function):
    """
    GPU vectorial total variation in log-domain with optional gradient normalization.

    Computes R_log(u) = dV * sum_j || ∇(log(u + log_eps))(j) ||_*
    where u is the multi-modality image and log_eps prevents log(0).
    """

    # Canonical names for Hessian diagonal variants and their aliases
    _HESSIAN_ALIASES = {
        # legacy → canonical
        "slow": "svd_principal_alpha",
        "fast": "mm_jensen",
        "fastest_positive": "frobenius_surrogate_pd",
        "fastest_exact": "vector_tv_per_modality",
        # new aliases
        "mm_diag": "mm_jensen",
        "frob_diag": "frobenius_surrogate_pd",
    }

    _HESSIAN_CANONICAL = {
        "svd_principal_alpha",
        "mm_jensen",
        "frobenius_surrogate_pd",
        "vector_tv_per_modality",
        "bd_synergy",
    }

    def __init__(
        self,
        geometry,
        weights,
        delta,
        log_eps_values,
        smoothing="charbonnier",
        norm="nuclear",
        anatomical=None,
        stable=True,
        stencil="6",
        both_directions=False,
        max_step=1,
        tail_singular_values=None,
        hessian="svd_principal_alpha",
        bnd_cond="Periodic",
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        self._dV = float(np.prod(voxel_sizes))
        if isinstance(anatomical, ImageData):
            anatomical = get_array(anatomical)
        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            stencil=stencil,
            both_directions=both_directions,
            max_step=max_step,
            bnd_cond=bnd_cond,
        )

        self.smoothing = smoothing
        # Canonicalize hessian name (accept legacy aliases for backwards compatibility)
        if hessian in self._HESSIAN_CANONICAL:
            self.hessian = hessian
        else:
            self.hessian = self._HESSIAN_ALIASES.get(hessian, hessian)
        self.bdc2a = BlockDataContainerToArray(geometry)

        self.weights = self.bdc2a.direct(weights)

        self.inv_weights = torch.reciprocal(self.weights)
        self.inv_weights = torch.nan_to_num(self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0)

        # Convert per-modality log_eps values to torch tensor
        # Shape: (M,) where M is number of modalities (typically 2 for PET+SPECT)
        self.log_eps = torch.tensor(
            log_eps_values,
            device=self.weights.device,
            dtype=self.weights.dtype
        )

        if tail_singular_values is not None:
            print(f"tail_singular_values = {tail_singular_values}")
        if stable:
            from .schatten_norm_gpu_slow import GPUVectorialTotalVariation as GpuVTV
        else:
            from .schatten_norm_gpu_stable import GPUVectorialTotalVariation as GpuVTV
        self.vtv = GpuVTV(
            eps=delta,
            norm=norm,
            smoothing_function=smoothing,
            tail=tail_singular_values,
        )

    def __call__(self, x):
        x_arr = self.bdc2a.direct(x)

        # Clamp to non-negative and apply log transform with per-modality epsilon
        x_clamped = torch.clamp(x_arr, min=0.0)
        # Reshape log_eps from (M,) to (1,1,1,M) for broadcasting with (nx,ny,nz,M)
        log_eps_broadcast = self.log_eps.view(1, 1, 1, -1)
        v = torch.log(x_clamped + log_eps_broadcast)

        # Compute Jacobian of v
        J = self.jacobian.direct(v)

        w = self.weights.unsqueeze(-1)
        U = w * J

        return self._dV*self.vtv(U)

    def gradient(self, x, out=None):
        x_arr = self.bdc2a.direct(x)

        # Clamp to non-negative and apply log transform with per-modality epsilon
        x_clamped = torch.clamp(x_arr, min=0.0)
        # Reshape log_eps from (M,) to (1,1,1,M) for broadcasting
        log_eps_broadcast = self.log_eps.view(1, 1, 1, -1)
        v = torch.log(x_clamped + log_eps_broadcast)

        # Compute Jacobian of v
        J = self.jacobian.direct(v)

        w = self.weights.unsqueeze(-1)
        U = w * J

        inner = w * self.vtv.gradient(U)

        # Apply Jacobian adjoint to get gradient w.r.t. v
        g_v = self.jacobian.adjoint(inner)

        # Chain rule: multiply by 1/(u + log_eps) with per-modality epsilon
        chain_multiplier = torch.reciprocal(x_clamped + log_eps_broadcast)
        g_u = g_v * chain_multiplier

        return self.bdc2a.adjoint(self._dV*g_u, out=out)

    def _compute_directional_participation_counts(self, shape, device, dtype):
        """
        Compute per-voxel directional participation counts for boundary handling.

        Voxels participate in 1 direction at boundaries, 2 in interior.

        Args:
            shape: (nx, ny, nz, d) spatial + directional dimensions
            device: torch device
            dtype: torch dtype

        Returns:
            Tensor of shape (nx, ny, nz, d) with participation counts
        """
        nx, ny, nz, d = shape

        def _counts_1d(n: int):
            """Boundary=1, interior=2 for dimension of size n"""
            c = torch.full((n,), 2.0, device=device, dtype=dtype)
            if n > 0:
                c[0] = 1.0
                if n > 1:
                    c[-1] = 1.0
            return c

        # Start with interior assumption (2.0 everywhere)
        C = torch.full((nx, ny, nz, d), 2.0, device=device, dtype=dtype)

        # Override with boundary counts per direction
        if d >= 1:
            C[..., 0] = _counts_1d(nx).view(nx, 1, 1).expand(nx, ny, nz)
        if d >= 2:
            C[..., 1] = _counts_1d(ny).view(1, ny, 1).expand(nx, ny, nz)
        if d >= 3:
            C[..., 2] = _counts_1d(nz).view(1, 1, nz).expand(nx, ny, nz)

        return C

    def _preconditioner_weights_core_fast(self, v_arr, eta: float = 0.7, epsilon: float = 1e-8):
        """
        MM/IRLS preconditioner using Jensen's inequality (SVD-free).

        Operates on log-transformed image v_arr = log(u + log_eps).

            P_{j,m} = η · [ω_j · S_jm] · b_{j,m}²

        where:
        - ω_j = φ'(σ_avg) / (2·σ_avg), with σ_avg = √(||A_j||²_F / r)
        - r = min(M, d) = rank of gradient matrix
        - S_jm = Σ_dirs (scale²_{j,m,dir} · n_dir(j)), Jacobian sensitivity
        - b_{j,m} = spatial weight

        Uses Jensen's inequality to avoid SVD:
        Σ_ℓ φ'(σ_ℓ)/σ_ℓ ≥ r · φ'(σ_avg) / σ_avg where σ_avg = √(Σ_ℓ σ²_ℓ / r)

        This is an MM (Majorization-Minimization) surrogate that guarantees P > 0.
        """
        # Compute weighted gradient field (already in log-domain)
        J = self.jacobian.direct(v_arr)  # (nx, ny, nz, M, d)
        weights = self.weights.to(J.device, dtype=J.dtype)
        A = weights.unsqueeze(-1) * J

        # Compute Frobenius norm per voxel: ||A_j||²_F = Σ_{m,d} A²_{j,m,d}
        A_frob_sq = torch.sum(A * A, dim=(-2, -1))  # (nx, ny, nz)

        # Compute rank (min of M, d dimensions)
        nx, ny, nz, M, d = A.shape
        r = min(M, d)

        # Average singular value via Jensen: σ_avg = √(||A||²_F / r)
        sigma_avg_sq = A_frob_sq / r
        eps_sqrt = get_sqrt_epsilon(A.dtype)
        sigma_avg = torch.sqrt(sigma_avg_sq + eps_sqrt)  # stabilize

        phi1, _ = _smoothing_derivatives(self.smoothing)
        phi_prime = phi1(sigma_avg, self.vtv.eps)

        eps_div = get_division_epsilon(A.dtype)
        omega = phi_prime / (2.0 * sigma_avg + eps_div)

        # Get Jacobian sensitivity: per-direction scaling factors
        S = self.jacobian.sensitivity(v_arr)  # (nx, ny, nz, M, d)
        S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
        if S.ndim < A.ndim:
            S = S.expand_as(A)
        S_squared = S * S

        # Compute directional participation counts (boundary vs interior)
        nx, ny, nz, M, d = A.shape
        participation = self._compute_directional_participation_counts(
            (nx, ny, nz, d), A.device, A.dtype
        )  # (nx, ny, nz, d)
        participation = participation.unsqueeze(-2).expand(nx, ny, nz, M, d)  # (nx, ny, nz, M, d)

        d_diag, _ = _directional_projector_stats_from_jacobian(self.jacobian, A, M, d)
        # Aggregate sensitivity: S_jm = Σ_dirs (S²_{j,m,dir} · n_dir · diag(D^T D))
        S_jm = (S_squared * participation * d_diag).sum(dim=-1)  # (nx, ny, nz, M)

        # Assemble preconditioner: P_{j,m} = η · ω_j · S_jm · b²_{j,m}
        weights_squared = weights * weights  # (nx, ny, nz, M)
        P_diag = eta * omega.unsqueeze(-1) * S_jm * weights_squared
        P_diag = torch.clamp(P_diag, min=epsilon)

        return P_diag

    def _hessian_diag_fast(self, v_arr, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns a BlockDataContainer holding a diagonal positive surrogate H ≈ ∇²V(v).
        Shape matches v (nx,ny,nz,M). Guaranteed H >= epsilon.

        Operates on log-transformed image v_arr.
        """
        H = self._preconditioner_weights_core_fast(v_arr, eta, epsilon)
        return H

    def _inv_hessian_diag_fast(self, v_arr, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns inverse of hessian_diag(v).
        Since H is floored by epsilon, inv(H) is bounded above by 1/epsilon.

        Operates on log-transformed image v_arr.
        """
        H = self._preconditioner_weights_core_fast(v_arr, eta, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return Hinv

    def _preconditioner_weights_core_fastest_positive(
        self, v_arr, eta: float = 0.7, epsilon: float = 1e-8
    ):
        """
        Ultra-fast positive-definite preconditioner using rank estimate.

        Operates on log-transformed image v_arr = log(u + log_eps).

            P_{j,m} = η · [M · φ'(||A_j||_F) / ||A_j||_F] · S_jm · b_{j,m}²

        where:
        - ||A_j||_F = Frobenius norm of gradient matrix at voxel j (cheap to compute)
        - φ'(·) = smoothing function derivative (e.g., Charbonnier)
        - M = number of modalities (rank upper bound)
        - S_jm = Jacobian sensitivity (same as fast method)
        - b_{j,m} = spatial weight

        This avoids SVD entirely by using Frobenius norm as a surrogate for singular values.
        Approximation: Σ_ℓ φ'(σ_ℓ)/σ_ℓ ≈ M · φ'(||A||_F)/||A||_F
        """
        # Compute weighted gradient field (already in log-domain)
        J = self.jacobian.direct(v_arr)  # (nx, ny, nz, M, d)
        weights = self.weights.to(J.device, dtype=J.dtype)
        A = weights.unsqueeze(-1) * J

        # Compute Frobenius norm per voxel: ||A_j||_F = sqrt(Σ_{m,d} A²_{j,m,d})
        A_frob_sq = torch.sum(A * A, dim=(-2, -1))  # (nx, ny, nz)
        eps_sqrt = get_sqrt_epsilon(A.dtype)
        A_frob = torch.sqrt(A_frob_sq + eps_sqrt)  # stabilize

        eps_floor = torch.tensor(float(self.vtv.eps), device=A.device, dtype=A.dtype)
        A_frob_safe = torch.maximum(A_frob, eps_floor)

        phi1, _ = _smoothing_derivatives(self.smoothing)
        phi_prime = phi1(A_frob, self.vtv.eps)

        # Hessian surrogate weight: M · φ'(||A||_F) / ||A||_F
        M = A.shape[-2]  # number of modalities
        omega = M * phi_prime / A_frob_safe  # (nx, ny, nz)

        # Get Jacobian sensitivity (same as fast method)
        S = self.jacobian.sensitivity(v_arr)  # (nx, ny, nz, M, d)
        S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
        if S.ndim < A.ndim:
            S = S.expand_as(A)
        S_squared = S * S

        # Compute directional participation counts
        nx, ny, nz, M, d = A.shape
        participation = self._compute_directional_participation_counts(
            (nx, ny, nz, d), A.device, A.dtype
        )
        participation = participation.unsqueeze(-2).expand(nx, ny, nz, M, d)

        d_diag, _ = _directional_projector_stats_from_jacobian(self.jacobian, A, M, d)
        # Aggregate sensitivity: S_jm = Σ_dirs (S²_{j,m,dir} · n_dir · diag(D^T D))
        S_jm = (S_squared * participation * d_diag).sum(dim=-1)  # (nx, ny, nz, M)

        # Assemble preconditioner
        weights_squared = weights * weights
        P_diag = eta * omega.unsqueeze(-1) * S_jm * weights_squared
        P_diag = torch.clamp(P_diag, min=epsilon)

        return P_diag

    def _preconditioner_weights_core_fastest_exact(self, v_arr, epsilon: float = 1e-8):
        """
        Decoupled Vectorial TV preconditioner with exact per-modality radial
        structure (no SVD).

        Operates on log-transformed image v_arr = log(u + log_eps).

            For U = b ⊙ (Jv) with per-modality vectors U_{j,m,·}, let r = ||U||.
            α = φ'(r)/r, β = φ''(r) − α, and per-direction

                h_dir = α + β · (U_d^2 / (r^2 + tiny)).

            The image-space diagonal preconditioner is

                P_{j,m} = Σ_d [ b_{j,m}^2 · S_{j,m,d}^2 · h_dir ].

        This reduces exactly to the single-modality vector-norm Hessian diagonal.
        """
        # Weighted gradient field (already in log-domain)
        J = self.jacobian.direct(v_arr)  # (nx, ny, nz, M, d)
        A = self.weights.unsqueeze(-1) * J

        # Radial terms per modality
        r2 = torch.sum(A * A, dim=-1)  # (nx, ny, nz, M)
        eps_sqrt = get_sqrt_epsilon(A.dtype)
        r = torch.sqrt(r2 + eps_sqrt)

        if self.smoothing == "charbonnier":
            from .common import charbonnier_grad as phi1, charbonnier_hessian_diag as phi2
        elif self.smoothing == "fair":
            from .common import fair_grad as phi1, fair_hessian_diag as phi2
        elif self.smoothing == "perona_malik":
            from .common import perona_malik_grad as phi1, perona_malik_hessian_diag as phi2
        else:
            from .common import nothing_grad as phi1, nothing_hessian_diag as phi2

        phi1_r = phi1(r, self.vtv.eps)
        phi2_r = phi2(r, self.vtv.eps)
        eps_floor = torch.tensor(float(self.vtv.eps), device=A.device, dtype=A.dtype)
        r_safe = torch.maximum(r, eps_floor)
        alpha = phi1_r / r_safe
        beta = phi2_r - alpha
        frac = (A * A) / (r_safe.unsqueeze(-1) ** 2)  # (..., M, d)
        h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac

        # Sensitivity mapping
        S = self.jacobian.sensitivity(v_arr)  # (nx, ny, nz, M, d)
        S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
        if S.ndim < A.ndim:
            S = S.expand_as(A)
        S2 = S * S
        w2 = (self.weights * self.weights).unsqueeze(-1)

        P = torch.sum(w2 * S2 * h_dir, dim=-1)
        P = torch.clamp(P, min=epsilon)
        return P

    def _preconditioner_weights_core_slow(self, v_arr):
        """
        Diagonal Hessian preconditioner via SVD decomposition (principal + isotropic terms).

        Operates on log-transformed image v_arr = log(u + log_eps).

            P_i = Σ_k φ''(σ_k) · [(J^T w u_k v_k^T)_i]²

        where:
        - φ''(σ_k) = exact second derivative of smoothing function at σ_k
        - u_k, v_k = left/right singular vectors from SVD(w·J·v)
        - J = Jacobian operator, J^T = adjoint
        - w = spatial weighting

        This builds a consistent diagonal approximation by:
        1. SVD of weighted gradient field A = w·J·v
        2. For each singular value σ_k:
           - Extract rank-1 component C_k = u_k v_k^T
           - Backproject through J^T: z_k = J^T(w·C_k)
           - Accumulate: P += φ''(σ_k) · z_k²
        3. Add the isotropic component α = Σ_k φ'(σ_k)/σ_k, mapped diagonally via
           P += α · (w^2 · Σ_dir sensitivity^2), which matches the vector-norm limit (M=1).
        """
        # Compute weighted gradient field (already in log-domain)
        J = self.jacobian.direct(v_arr)  # (nx, ny, nz, M, d)
        w = self.weights.unsqueeze(-1)  # (nx, ny, nz, M, 1)
        A_field = w * J  # (nx, ny, nz, M, d)

        # SVD decomposition: get φ''(σ_k) and u_k v_k^T for all k
        hess_coeffs, rank_one_fields = self.vtv.hessian_components(A_field)
        # hess_coeffs: (nx, ny, nz, r) - φ''(σ_k) values
        # rank_one_fields: (nx, ny, nz, r, M, d) - u_k v_k^T fields

        # Accumulate diagonal contributions (principal φ'' terms)
        P_diag = torch.zeros_like(
            v_arr, device=rank_one_fields.device, dtype=rank_one_fields.dtype
        )  # (nx, ny, nz, M)
        num_singular_values = rank_one_fields.shape[-3]  # r

        for k in range(num_singular_values):
            # Extract k-th rank-1 field: u_k v_k^T
            C_k_field = rank_one_fields[..., k, :, :]  # (nx, ny, nz, M, d)

            # Backproject through weighted Jacobian adjoint: z_k = J^T(w·C_k)
            w_dev = w.to(C_k_field.device, dtype=C_k_field.dtype)
            influence_image = self.jacobian.adjoint(w_dev * C_k_field)  # (nx, ny, nz, M)
            influence_image = influence_image.to(P_diag.device, dtype=P_diag.dtype)

            # Weight by φ''(σ_k) and accumulate squared influence
            h_double_prime_k = (
                hess_coeffs[..., k].unsqueeze(-1).to(P_diag.device)
            )  # (nx, ny, nz, 1)
            P_diag += h_double_prime_k * (influence_image**2)

        # --- Add isotropic α-term: α = Σ_k φ'(σ_k)/σ_k ---
        # Use hessian_surrogate which gives 0.5·φ'(σ)/σ per singular value
        try:
            sigma_weights_half = self.vtv.hessian_surrogate(A_field)  # (..., r)
        except Exception:
            # Fallback if backend lacks hessian_surrogate for some reason
            sigma_weights_half = None

        if sigma_weights_half is not None:
            # α_total per voxel
            alpha_total = (2.0 * torch.sum(sigma_weights_half, dim=-1)).to(
                P_diag.device
            )  # (nx, ny, nz)

            # Map α diagonally through sensitivity (no participation counts for consistency
            # with single-modality vector-norm mapping)
            S = self.jacobian.sensitivity(v_arr)  # (nx, ny, nz, M, d)
            S = torch.as_tensor(S, device=P_diag.device, dtype=P_diag.dtype)
            S2 = S * S
            S_jm = torch.sum(S2, dim=-1)  # (nx, ny, nz, M)

            w2 = (self.weights * self.weights).to(
                P_diag.device, dtype=P_diag.dtype
            )  # (nx, ny, nz, M)
            P_diag = P_diag + alpha_total.unsqueeze(-1) * w2 * S_jm

        # Floor at epsilon to prevent numerical issues (voxels with zero Hessian)
        # This can occur at isolated points or boundaries with no gradient participation
        P_diag = torch.clamp(P_diag, min=1e-8)

        return P_diag

    def _hessian_diag_fastest_positive(
        self, v_arr, eta: float = 0.7, epsilon: float = 1e-8, out=None
    ):
        """Returns diagonal positive preconditioner using Frobenius approximation (fastest)."""
        H = self._preconditioner_weights_core_fastest_positive(v_arr, eta, epsilon)
        return H

    def _inv_hessian_diag_fastest_positive(
        self, v_arr, eta: float = 0.7, epsilon: float = 1e-8, out=None
    ):
        """Returns inverse diagonal preconditioner (Frobenius, positive-definite)."""
        H = self._preconditioner_weights_core_fastest_positive(v_arr, eta, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return Hinv

    def _hessian_diag_fastest_exact(self, v_arr, epsilon: float = 1e-8, out=None):
        """Returns diagonal Hessian using Frobenius approximation with exact φ''."""
        H = self._preconditioner_weights_core_fastest_exact(v_arr, epsilon)
        return H

    def _inv_hessian_diag_fastest_exact(self, v_arr, epsilon: float = 1e-8, out=None):
        """Returns inverse diagonal Hessian (Frobenius, exact φ'')."""
        H = self._preconditioner_weights_core_fastest_exact(v_arr, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return Hinv

    def _hessian_diag_slow(self, v_arr, out=None):
        """Exact diagonal Hessian via full SVD (slowest, most accurate)."""
        diag_arr = self._preconditioner_weights_core_slow(v_arr)
        return diag_arr

    def _inv_hessian_diag_slow(self, v_arr, out=None):
        """Inverse of exact diagonal Hessian via full SVD."""
        diag_arr = self._preconditioner_weights_core_slow(v_arr)
        inv_arr = torch.reciprocal(diag_arr)
        inv_arr = torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return inv_arr

    def hessian_diag(self, x, out=None, eta: float = 0.7, epsilon: float = 1e-8):
        """
        Compute diagonal Hessian approximation in log-domain.

        Args:
            x: Input BlockDataContainer (in intensity space)
            out: Optional output container
            eta: Damping factor for MM methods (default 0.7)
            epsilon: Floor value for stability (default 1e-8)

        Method selection (via self.hessian):
            - "bd_synergy": not supported in log domain
            - "svd_principal_alpha": SVD rank‑1 principal plus isotropic α
            - "mm_jensen"/"mm_diag": SVD‑free MM/Jensen surrogate (positive)
            - "frobenius_surrogate_pd"/"frob_diag": Frobenius surrogate (positive, ultra-fast)
            - "vector_tv_per_modality": Per‑modality vector‑norm exact radial
        """
        # Transform to log-domain with per-modality epsilon
        x_arr = self.bdc2a.direct(x)
        x_clamped = torch.clamp(x_arr, min=0.0)
        # Reshape log_eps from (M,) to (1,1,1,M) for broadcasting
        log_eps_broadcast = self.log_eps.view(1, 1, 1, -1)
        v = torch.log(x_clamped + log_eps_broadcast)

        # Compute TNV Hessian diagonal in log-domain (dispatches to appropriate method)
        if self.hessian == "bd_synergy":
            raise ValueError("bd_synergy is block-diagonal and not supported in WeightedLogVectorialTotalVariation.")
        elif self.hessian == "svd_principal_alpha":
            H_v = self._hessian_diag_slow(v)
        elif self.hessian in {"mm_jensen", "mm_diag"}:
            H_v = self._hessian_diag_fast(v, eta=eta, epsilon=epsilon)
        elif self.hessian in {"frobenius_surrogate_pd", "frob_diag"}:
            H_v = self._hessian_diag_fastest_positive(v, eta=eta, epsilon=epsilon)
        elif self.hessian == "vector_tv_per_modality":
            H_v = self._hessian_diag_fastest_exact(v, epsilon=epsilon)
        else:
            raise ValueError(
                f"Unknown Hessian type: {self.hessian}. "
                "Options: 'bd_synergy', 'svd_principal_alpha', "
                "'mm_jensen'/'mm_diag', 'frobenius_surrogate_pd'/'frob_diag', "
                "'vector_tv_per_modality'"
            )

        # Chain rule for second derivative: H_u ≈ H_v / (u + log_eps)^2 with per-modality epsilon
        chain_multiplier = torch.reciprocal((x_clamped + log_eps_broadcast) ** 2)
        H_u = H_v * chain_multiplier
        H_u = torch.clamp(H_u, min=epsilon)

        return self.bdc2a.adjoint(self._dV*H_u, out=out)

    def inv_hessian_diag(self, x, out=None, eta: float = 0.7):
        """
        Compute inverse diagonal Hessian approximation in log-domain.

        Args:
            x: Input BlockDataContainer (in intensity space)
            out: Optional output container
            eta: Damping factor for MM methods (default 0.7)
            epsilon: Regularization for inversion (default 1e-8)

        Method selection (via self.hessian):
            - "bd_synergy": not supported in log domain
            - "svd_principal_alpha": SVD rank‑1 principal plus isotropic α
            - "mm_jensen"/"mm_diag": SVD‑free MM/Jensen surrogate (positive)
            - "frobenius_surrogate_pd"/"frob_diag": Frobenius surrogate (positive)
            - "vector_tv_per_modality": Per‑modality vector‑norm exact radial
        """
        # Compute hessian_diag (which already applies chain rule)
        H = self.hessian_diag(x, out=None, eta=eta)
        H_arr = self.bdc2a.direct(H)
        Hinv = torch.reciprocal(H_arr)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return self.bdc2a.adjoint(Hinv/self._dV, out=out)
