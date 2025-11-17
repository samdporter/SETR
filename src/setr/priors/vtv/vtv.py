# VTV.py

import numpy as np
from cil.optimisation.functions import Function

try:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = "cpu"
from sirf.STIR import ImageData

from setr.core.gradients import Jacobian
from setr.utils import BlockDataContainerToArray
from setr.utils.sirf import get_array


class WeightedVectorialTotalVariation(Function):
    """
    GPU vectorial total variation with optional gradient normalization.
    """

    # Canonical names for Hessian diagonal variants and their aliases
    _HESSIAN_ALIASES = {
        # legacy → canonical
        "slow": "svd_principal_alpha",
        "fast": "mm_jensen",
        "fastest_positive": "frobenius_surrogate_pd",
        "fastest_exact": "vector_tv_per_modality",
    }

    _HESSIAN_CANONICAL = {
        "svd_principal_alpha",
        "mm_jensen",
        "frobenius_surrogate_pd",
        "vector_tv_per_modality",
    }

    def __init__(
        self,
        geometry,
        weights,
        delta,
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
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J

        return self._dV*self.vtv(U)

    def gradient(self, x, out=None):
        x_arr = self.bdc2a.direct(x)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J

        inner = w * self.vtv.gradient(U)

        ret = self.jacobian.adjoint(inner)

        return self.bdc2a.adjoint(self._dV*ret, out=out)

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

    def _preconditioner_weights_core_fast(self, x_arr, eta: float = 0.7, epsilon: float = 1e-8):
        """
        MM/IRLS preconditioner using Jensen's inequality (SVD-free).

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
        # Compute weighted gradient field
        J = self.jacobian.direct(x_arr)  # (nx, ny, nz, M, d)
        A = self.weights.unsqueeze(-1) * J

        # Compute Frobenius norm per voxel: ||A_j||²_F = Σ_{m,d} A²_{j,m,d}
        A_frob_sq = torch.sum(A * A, dim=(-2, -1))  # (nx, ny, nz)

        # Compute rank (min of M, d dimensions)
        nx, ny, nz, M, d = A.shape
        r = min(M, d)

        # Average singular value via Jensen: σ_avg = √(||A||²_F / r)
        sigma_avg_sq = A_frob_sq / r
        sigma_avg = torch.sqrt(sigma_avg_sq + 1e-12)  # stabilize

        # Select smoothing function derivative
        if self.smoothing == "charbonnier":
            from .common import charbonnier_grad

            phi_prime = charbonnier_grad(sigma_avg, self.vtv.eps)
        elif self.smoothing == "fair":
            from .common import fair_grad

            phi_prime = fair_grad(sigma_avg, self.vtv.eps)
        elif self.smoothing == "perona_malik":
            from .common import perona_malik_grad

            phi_prime = perona_malik_grad(sigma_avg, self.vtv.eps)
        else:
            phi_prime = torch.ones_like(sigma_avg)

        # Jensen bound: ω_j = φ'(σ_avg) / (2·σ_avg)
        omega = phi_prime / (2.0 * sigma_avg + 1e-12)  # (nx, ny, nz)

        # Get Jacobian sensitivity: per-direction scaling factors
        S = self.jacobian.sensitivity(x_arr)  # (nx, ny, nz, M, d)
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

        # Aggregate sensitivity: S_jm = Σ_dirs (S²_{j,m,dir} · n_dir)
        S_jm = (S_squared * participation).sum(dim=-1)  # (nx, ny, nz, M)

        # Assemble preconditioner: P_{j,m} = η · ω_j · S_jm · b²_{j,m}
        weights_squared = self.weights * self.weights  # (nx, ny, nz, M)
        P_diag = eta * omega.unsqueeze(-1) * S_jm * weights_squared
        P_diag = torch.clamp(P_diag, min=epsilon)

        return P_diag

    def _hessian_diag_fast(self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns a BlockDataContainer holding a diagonal positive surrogate H ≈ ∇²V(x).
        Shape matches x (nx,ny,nz,M). Guaranteed H >= epsilon.
        """
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fast(x_arr, eta, epsilon)
        return self.bdc2a.adjoint(self._dV*H, out=out)

    def _inv_hessian_diag_fast(self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns a BlockDataContainer with the elementwise inverse of hessian_diag(x).
        Since H is floored by epsilon, inv(H) is bounded above by 1/epsilon.
        """
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fast(x_arr, eta, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return self.bdc2a.adjoint(Hinv/self._dV, out=out)

    def _preconditioner_weights_core_fastest_positive(
        self, x_arr, eta: float = 0.7, epsilon: float = 1e-8
    ):
        """
        Ultra-fast positive-definite preconditioner using rank estimate.

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
        # Compute weighted gradient field
        J = self.jacobian.direct(x_arr)  # (nx, ny, nz, M, d)
        A = self.weights.unsqueeze(-1) * J

        # Compute Frobenius norm per voxel: ||A_j||_F = sqrt(Σ_{m,d} A²_{j,m,d})
        A_frob_sq = torch.sum(A * A, dim=(-2, -1))  # (nx, ny, nz)
        A_frob = torch.sqrt(A_frob_sq + 1e-12)  # stabilize

        # Select smoothing function derivative
        if self.smoothing == "charbonnier":
            from .common import charbonnier_grad

            phi_prime = charbonnier_grad(A_frob, self.vtv.eps)
        elif self.smoothing == "fair":
            from .common import fair_grad

            phi_prime = fair_grad(A_frob, self.vtv.eps)
        elif self.smoothing == "perona_malik":
            from .common import perona_malik_grad

            phi_prime = perona_malik_grad(A_frob, self.vtv.eps)
        else:
            phi_prime = torch.ones_like(A_frob)

        # Hessian surrogate weight: M · φ'(||A||_F) / ||A||_F
        M = A.shape[-2]  # number of modalities
        omega = M * phi_prime / (A_frob + 1e-12)  # (nx, ny, nz)

        # Get Jacobian sensitivity (same as fast method)
        S = self.jacobian.sensitivity(x_arr)  # (nx, ny, nz, M, d)
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

        # Aggregate sensitivity: S_jm = Σ_dirs (S²_{j,m,dir} · n_dir)
        S_jm = (S_squared * participation).sum(dim=-1)  # (nx, ny, nz, M)

        # Assemble preconditioner
        weights_squared = self.weights * self.weights
        P_diag = eta * omega.unsqueeze(-1) * S_jm * weights_squared
        P_diag = torch.clamp(P_diag, min=epsilon)

        return P_diag

    def _preconditioner_weights_core_fastest_exact(self, x_arr, epsilon: float = 1e-8):
        """
        Decoupled Vectorial TV preconditioner with exact per-modality radial
        structure (no SVD):

            For U = b ⊙ (Jx) with per-modality vectors U_{j,m,·}, let r = ||U||.
            α = φ'(r)/r, β = φ''(r) − α, and per-direction

                h_dir = α + β · (U_d^2 / (r^2 + tiny)).

            The image-space diagonal preconditioner is

                P_{j,m} = Σ_d [ b_{j,m}^2 · S_{j,m,d}^2 · h_dir ].

        This reduces exactly to the single-modality vector-norm Hessian diagonal.
        """
        # Weighted gradient field
        J = self.jacobian.direct(x_arr)  # (nx, ny, nz, M, d)
        A = self.weights.unsqueeze(-1) * J

        # Radial terms per modality
        r2 = torch.sum(A * A, dim=-1)  # (nx, ny, nz, M)
        r = torch.sqrt(r2 + 1e-12)

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
        alpha = phi1_r / (r + 1e-12)
        beta = phi2_r - alpha

        frac = (A * A) / (r2.unsqueeze(-1) + 1e-12)  # (..., M, d)
        h_dir = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * frac

        # Sensitivity mapping
        S = self.jacobian.sensitivity(x_arr)  # (nx, ny, nz, M, d)
        S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
        if S.ndim < A.ndim:
            S = S.expand_as(A)
        S2 = S * S
        w2 = (self.weights * self.weights).unsqueeze(-1)

        P = torch.sum(w2 * S2 * h_dir, dim=-1)
        P = torch.clamp(P, min=epsilon)
        return P

    def _preconditioner_weights_core_slow(self, x_arr):
        """
        Diagonal Hessian preconditioner via SVD decomposition (principal + isotropic terms).

            P_i = Σ_k φ''(σ_k) · [(J^T w u_k v_k^T)_i]²

        where:
        - φ''(σ_k) = exact second derivative of smoothing function at σ_k
        - u_k, v_k = left/right singular vectors from SVD(w·J·x)
        - J = Jacobian operator, J^T = adjoint
        - w = spatial weighting

        This builds a consistent diagonal approximation by:
        1. SVD of weighted gradient field A = w·J·x
        2. For each singular value σ_k:
           - Extract rank-1 component C_k = u_k v_k^T
           - Backproject through J^T: z_k = J^T(w·C_k)
           - Accumulate: P += φ''(σ_k) · z_k²
        3. Add the isotropic component α = Σ_k φ'(σ_k)/σ_k, mapped diagonally via
           P += α · (w^2 · Σ_dir sensitivity^2), which matches the vector-norm limit (M=1).
        """
        # Compute weighted gradient field
        J = self.jacobian.direct(x_arr)  # (nx, ny, nz, M, d)
        w = self.weights.unsqueeze(-1)  # (nx, ny, nz, M, 1)
        A_field = w * J  # (nx, ny, nz, M, d)

        # SVD decomposition: get φ''(σ_k) and u_k v_k^T for all k
        hess_coeffs, rank_one_fields = self.vtv.hessian_components(A_field)
        # hess_coeffs: (nx, ny, nz, r) - φ''(σ_k) values
        # rank_one_fields: (nx, ny, nz, r, M, d) - u_k v_k^T fields

        # Accumulate diagonal contributions (principal φ'' terms)
        P_diag = torch.zeros_like(
            x_arr, device=rank_one_fields.device, dtype=rank_one_fields.dtype
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
            S = self.jacobian.sensitivity(x_arr)  # (nx, ny, nz, M, d)
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

    def _hessian_diag_fastest_positive(self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """Returns diagonal positive preconditioner using Frobenius approximation (fastest)."""
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fastest_positive(x_arr, eta, epsilon)
        return self.bdc2a.adjoint(self._dV*H, out=out)

    def _inv_hessian_diag_fastest_positive(
        self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None
    ):
        """Returns inverse diagonal preconditioner (Frobenius, positive-definite)."""
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fastest_positive(x_arr, eta, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return self.bdc2a.adjoint(Hinv/self._dV, out=out)

    def _hessian_diag_fastest_exact(self, x, epsilon: float = 1e-8, out=None):
        """Returns diagonal Hessian using Frobenius approximation with exact φ''."""
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fastest_exact(x_arr, epsilon)
        return self.bdc2a.adjoint(self._dV*H, out=out)

    def _inv_hessian_diag_fastest_exact(self, x, epsilon: float = 1e-8, out=None):
        """Returns inverse diagonal Hessian (Frobenius, exact φ'')."""
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fastest_exact(x_arr, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)
        return self.bdc2a.adjoint(Hinv/self._dV, out=out)

    def _hessian_diag_slow(self, x, out=None):
        """Exact diagonal Hessian via full SVD (slowest, most accurate)."""
        x_arr = self.bdc2a.direct(x)
        diag_arr = self._preconditioner_weights_core_slow(x_arr)
        return self.bdc2a.adjoint(self._dV*diag_arr, out=out)

    def _inv_hessian_diag_slow(self, x, out=None):
        """Inverse of exact diagonal Hessian via full SVD."""
        diag_arr = self._preconditioner_weights_core_slow(self.bdc2a.direct(x))
        inv_arr = torch.reciprocal(diag_arr)
        torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0, out=inv_arr)
        return self.bdc2a.adjoint(inv_arr/self._dV, out=out)

    def hessian_diag(self, x, out=None, eta: float = 0.7, epsilon: float = 1e-8):
        """
        Compute diagonal Hessian approximation.

        Args:
            x: Input BlockDataContainer
            out: Optional output container
            eta: Damping factor for MM methods (default 0.7)
            epsilon: Floor value for stability (default 1e-8)

        Method selection (via self.hessian):
            - "svd_principal_alpha": SVD rank‑1 principal plus isotropic α
            - "mm_jensen": SVD‑free MM/Jensen surrogate (positive)
            - "frobenius_surrogate_pd": Frobenius surrogate (positive, ultra-fast)
            - "vector_tv_per_modality": Per‑modality vector‑norm exact radial
        """
        if self.hessian == "svd_principal_alpha":
            return self._hessian_diag_slow(x, out=out)
        elif self.hessian == "mm_jensen":
            return self._hessian_diag_fast(x, eta=eta, epsilon=epsilon, out=out)
        elif self.hessian == "frobenius_surrogate_pd":
            return self._hessian_diag_fastest_positive(x, eta=eta, epsilon=epsilon, out=out)
        elif self.hessian == "vector_tv_per_modality":
            return self._hessian_diag_fastest_exact(x, epsilon=epsilon, out=out)
        else:
            raise ValueError(
                f"Unknown Hessian type: {self.hessian}. "
                f"Options: 'svd_principal_alpha', 'mm_jensen', 'frobenius_surrogate_pd', 'vector_tv_per_modality'"
            )

    def inv_hessian_diag(self, x, out=None, eta: float = 0.7):
        """
        Compute inverse diagonal Hessian approximation.

        Args:
            x: Input BlockDataContainer
            out: Optional output container
            eta: Damping factor for MM methods (default 0.7)
            epsilon: Regularization for inversion (default 1e-8)

        Method selection (via self.hessian):
            - "svd_principal_alpha": SVD rank‑1 principal plus isotropic α
            - "mm_jensen": SVD‑free MM/Jensen surrogate (positive)
            - "frobenius_surrogate_pd": Frobenius surrogate (positive)
            - "vector_tv_per_modality": Per‑modality vector‑norm exact radial
        """
        if self.hessian == "svd_principal_alpha":
            return self._inv_hessian_diag_slow(x, out=out)
        elif self.hessian == "mm_jensen":
            return self._inv_hessian_diag_fast(x, eta=eta, out=out)
        elif self.hessian == "frobenius_surrogate_pd":
            return self._inv_hessian_diag_fastest_positive(x, eta=eta, out=out)
        elif self.hessian == "vector_tv_per_modality":
            return self._inv_hessian_diag_fastest_exact(x, out=out)
        else:
            raise ValueError(
                f"Unknown Hessian type: {self.hessian}. "
                f"Options: 'svd_principal_alpha', 'mm_jensen', 'frobenius_surrogate_pd', 'vector_tv_per_modality'"
            )


class WeightedTotalVariation(Function):
    """
    GPU total variation with optional gradient normalization.
    Applies smoothed total variation separately to each modality.
    """

    def __init__(
        self,
        geometry,
        weights,
        delta,
        smoothing="charbonnier",
        norm="l2",
        anatomical=None,
        diagonal=False,
        both_directions=False,
        hessian="slow",
        stencil="6",
        max_step=1,
        bnd_cond="Periodic",
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if hasattr(anatomical, "as_array"):  # ImageData
            anatomical = get_array(anatomical)
            
        self._dV = float(np.prod(voxel_sizes))

        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            both_directions=both_directions,
            stencil=stencil,
            max_step=max_step,
            bnd_cond=bnd_cond,
        )

        self.smoothing = smoothing
        self.hessian = hessian
        self.bdc2a = BlockDataContainerToArray(geometry)

        self.weights = self.bdc2a.direct(weights)

        self.inv_weights = torch.reciprocal(self.weights)
        self.inv_weights = torch.nan_to_num(self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0)

        from .vector_norm import GPUVectorNorm

        self.tv = GPUVectorNorm(eps=delta, norm=norm, smoothing_function=smoothing)

    def __call__(self, x):
        x_arr = self.bdc2a.direct(x)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J
        total_tv = 0.0
        for m in range(U.shape[-2]):
            modality_gradients = U[..., m, :]
            total_tv += self.tv(modality_gradients)

        return self._dV*total_tv

    def gradient(self, x, out=None):
        x_arr = self.bdc2a.direct(x)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J

        inner = torch.zeros_like(U)
        for m in range(U.shape[-2]):
            modality_gradients = U[..., m, :]
            inner[..., m, :] = w[..., m, :] * self.tv.gradient(modality_gradients)

        ret = self.jacobian.adjoint(inner)

        return self.bdc2a.adjoint(self._dV*ret, out=out)

    def proximal(self, x, tau, out=None):
        """
        Proximal operator for total variation.
        """
        x_arr = self.bdc2a.direct(x)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J

        proxU = torch.zeros_like(U)
        for m in range(U.shape[-2]):
            modality_gradients = U[..., m, :]
            proxU[..., m, :] = self.tv.proximal(modality_gradients, tau*self._dV)

        ret = self.jacobian.adjoint(proxU * w)
        return self.bdc2a.adjoint(ret, out=out)

    def hessian_diag(self, x, out=None, stabiliser: float = 1e-9, positive: bool = True):
        """
        Diagonal approximation in image space:
        sum_j (w^2 * s_j^2) * h_j(U), with U = w * (Jx).
        Requires jacobian.sensitivity(...) -> (..., M, d) per-direction sensitivities.
        """
        x_arr = self.bdc2a.direct(x)
        J = self.jacobian.direct(x_arr)
        w = self.weights.unsqueeze(-1)
        U = w * J

        h_dir = torch.zeros_like(U)
        for m in range(U.shape[-2]):
            modality_gradients = U[..., m, :]
            h_dir[..., m, :] = self.tv.hessian_dir_diag(
                modality_gradients, stabiliser=stabiliser, positive=positive
            )

        S = torch.as_tensor(self.jacobian.sensitivity(x_arr), device=U.device, dtype=U.dtype)
        S2 = S * S
        w2 = (self.weights**2).unsqueeze(-1)

        h_img = torch.sum(w2 * S2 * h_dir, dim=-1)

        return self.bdc2a.adjoint(self._dV*h_img, out=out)

    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        hess_arr = self.hessian_diag(x, out=None)
        H = torch.as_tensor(
            hess_arr if isinstance(hess_arr, torch.Tensor) else self.bdc2a.direct(hess_arr),
            device=device,
        )
        inv_arr = torch.reciprocal(H + epsilon)
        return self.bdc2a.adjoint(inv_arr, out=out)


class TotalVariation(Function):
    """
    GPU total variation for single ImageData objects with optional anatomical guidance.
    Supports directional TV through anatomical image guidance.
    """

    def __init__(
        self,
        geometry,
        weight=1.0,
        delta=1e-6,
        smoothing="charbonnier",
        norm="l2",
        anatomical=None,
        both_directions=False,
        stencil="6",
        hessian="slow",
    ):
        """
        Initialize single-modality Total Variation prior.

        Args:
            geometry: ImageData template defining the image space
            weight: Scalar weighting factor for the TV prior
            delta: Smoothing parameter for the TV function
            smoothing: Smoothing function type ('charbonnier', 'fair', etc.)
            norm: Vector norm type ('l2', 'l1', etc.)
            anatomical: Optional anatomical image for directional guidance
            both_directions: Use bidirectional gradients
            stencil: Stencil connectivity ('6', '18', '26')
            hessian: Hessian approximation method
        """
        from setr.core.gradients import DirectionalGradient, Gradient

        self.geometry = geometry
        self.weight = float(weight)
        self.delta = float(delta)
        self.smoothing = smoothing
        self.hessian = hessian

        voxel_sizes = geometry.voxel_sizes()
        self._dV = float(np.prod(voxel_sizes))

        # Choose gradient operator based on anatomical guidance
        if anatomical is not None:
            if hasattr(anatomical, "as_array"):  # ImageData
                anatomical_arr = get_array(anatomical)
            else:
                anatomical_arr = anatomical

            self.gradient_op = DirectionalGradient(
                anatomical=anatomical_arr,
                voxel_sizes=voxel_sizes,
                both_directions=both_directions,
                stencil=stencil,
                normalize=True,
            )
            self.directional = True
        else:
            self.gradient_op = Gradient(
                voxel_sizes=voxel_sizes,
                both_directions=both_directions,
                stencil=stencil,
                normalize=True,
            )
            self.directional = False

        # Create the GPU total variation backend
        from .vector_norm import GPUVectorNorm

        self.tv = GPUVectorNorm(eps=delta, norm=norm, smoothing_function=smoothing)

    def __call__(self, x):
        """Evaluate the TV functional on ImageData x."""
        if not isinstance(x, ImageData):
            raise TypeError("TotalVariation expects ImageData input")

        x_arr = get_array(x)
        grad = self.gradient_op.direct(x_arr)  # Shape: (nz, ny, nx, d)

        # Apply weight and compute TV
        weighted_grad = self.weight * grad
        return self._dV*self.tv(weighted_grad)

    def gradient(self, x, out=None):
        """Compute gradient of TV functional."""
        if not isinstance(x, ImageData):
            raise TypeError("TotalVariation expects ImageData input")

        x_arr = get_array(x)
        grad = self.gradient_op.direct(x_arr)  # (nz, ny, nx, d)

        # Apply weight
        weighted_grad = self.weight * grad

        # Get TV gradient
        tv_grad = self.tv.gradient(weighted_grad)  # (nz, ny, nx, d)

        # Apply weight again and compute adjoint
        weighted_tv_grad = self.weight * tv_grad
        result_arr = self.gradient_op.adjoint(weighted_tv_grad)

        # Convert back to ImageData
        if out is None:
            out = x.clone()

        if hasattr(result_arr, "detach"):  # torch tensor
            result_arr = result_arr.detach().cpu().numpy()

        out.fill(self._dV*result_arr)
        return out

    def proximal(self, x, tau, out=None):
        """Proximal operator for TV."""
        if not isinstance(x, ImageData):
            raise TypeError("TotalVariation expects ImageData input")

        x_arr = get_array(x)
        grad = self.gradient_op.direct(x_arr)  # (nz, ny, nx, d)

        # Apply weight
        weighted_grad = self.weight * grad

        # Apply TV proximal operator
        prox_grad = self.tv.proximal(weighted_grad, tau*self._dV)

        # Apply weight and compute adjoint
        weighted_prox = self.weight * prox_grad
        result_arr = self.gradient_op.adjoint(weighted_prox)

        # Convert back to ImageData
        if out is None:
            out = x.clone()

        if hasattr(result_arr, "detach"):  # torch tensor
            result_arr = result_arr.detach().cpu().numpy()

        out.fill(result_arr)
        return out

    def hessian_diag(self, x, out=None, stabiliser=1e-9, positive=True):
        """Diagonal Hessian approximation."""
        if not isinstance(x, ImageData):
            raise TypeError("TotalVariation expects ImageData input")

        x_arr = get_array(x)
        grad = self.gradient_op.direct(x_arr)  # (nz, ny, nx, d)

        # Apply weight
        weighted_grad = self.weight * grad

        # Get TV Hessian diagonal
        h_dir = self.tv.hessian_dir_diag(weighted_grad, stabiliser=stabiliser, positive=positive)

        # For single modality, sum over gradient directions with weight^2
        h_img = self.weight**2 * torch.sum(h_dir, dim=-1)

        # Convert back to ImageData
        if out is None:
            out = x.clone()

        if hasattr(h_img, "detach"):  # torch tensor
            h_img = h_img.detach().cpu().numpy()

        out.fill(self._dV*h_img)
        return out

    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        """Inverse diagonal Hessian approximation."""
        h = self.hessian_diag(x, out=None)
        h_arr = get_array(h)

        # Convert to torch if needed
        if not isinstance(h_arr, torch.Tensor):
            h_arr = torch.as_tensor(h_arr, device=device)

        inv_arr = torch.reciprocal(h_arr + epsilon)

        # Convert back to ImageData
        if out is None:
            out = x.clone()

        if hasattr(inv_arr, "detach"):  # torch tensor
            inv_arr = inv_arr.detach().cpu().numpy()

        out.fill(inv_arr)
        return out
