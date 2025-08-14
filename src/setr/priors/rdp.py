"""
Relative Difference Prior (RDP) for SETR — tensor-first, with optional *directional* differences,
plus a multi-modality WeightedRDP using a Jacobian.

Reference:
- Nuyts et al., “A concave prior penalizing relative differences for MAP reconstruction
  in emission tomography,” IEEE TNS, 49(1):56–60, 2002.  DOI:10.1109/TNS.2002.998681.  :contentReference[oaicite:0]{index=0}

Per directed neighbour edge and per direction d:
    Δ = G λ        (edge-wise differences, stacked over directions)
    Σ = S λ        (edge-wise sums,       stacked over directions)
    (optional) Δ̃ = D_grad.direct(λ)  where D_grad is a DirectionalGradient operator
    D = Σ + γ |Δ̃| + ε
    φ(Δ̃,Σ) = Δ̃² / D
    M(λ) = -β Σ_edges φ

Exact derivatives (with Δ̃ from the operator; Σ unchanged):
    ∂φ/∂Δ̃ = (2 Δ̃ D - γ Δ̃² sign(Δ̃)) / D²
    ∂φ/∂Σ  = - Δ̃² / D²
Gradient:
    ∇M(λ) = -β [ Gᵀ( ∂φ/∂Δ̃ ) + Sᵀ( ∂φ/∂Σ ) ],
with Gᵀ implicitly containing the adjoint of the (directional) Δ operator.

Hessian–vector product (exact, a.e.):
    Let Gv be (directional) gradient applied to v as well. Define (a.e.):
        φ_dd =  2/D - 4γ|Δ̃|/D² + 2γ² Δ̃²/D³
        φ_ds =  2(γ Δ̃² sign(Δ̃) - Δ̃ D)/D³
        φ_ss =  2 Δ̃²/D³
    Then
        H(λ)v = -β [ Gᵀ( φ_dd ⊙ Gv + φ_ds ⊙ S v )
                    + Sᵀ( φ_ds ⊙ Gv + φ_ss ⊙ S v ) ].

Exact diagonal (per voxel):
    For each directed edge, endpoint contributions are
        H_jj += -β [ φ_dd + φ_ss + 2 φ_ds ]
        H_kk += -β [ φ_dd + φ_ss - 2 φ_ds ]
If both_directions=True, multiply edge terms by 1/2 (de-dup).
"""

from __future__ import annotations
import torch
from cil.optimisation.functions import Function

# Project operators
from setr.core.gradients import Gradient, DirectionalGradient, Sum, Jacobian
from setr.utils import BlockDataContainerToArray

# -------------------------------------------------------------------------
# Device / dtype helpers
# -------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE  = torch.float32

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(_DEVICE, dtype=_DTYPE)
    if hasattr(x, "as_array"):
        x = x.as_array()
    return torch.as_tensor(x, device=_DEVICE, dtype=_DTYPE)

def _from_tensor_like(template, t: torch.Tensor):
    arr = t.detach().to("cpu").numpy()
    if hasattr(template, "clone"):
        out = template.clone()
        out.fill(arr)
        return out
    return arr


# -------------------------------------------------------------------------
# Scalar (single-modality) RDP with optional directional differences
# -------------------------------------------------------------------------
class RelativeDifferencePrior(Function):
    def __init__(
        self,
        domain_geometry,
        beta: float = 1.0,
        gamma: float = 0.1,
        stencil: str = "6",
        both_directions: bool = False,
        epsilon: float = 1e-12,
        anatomical=None,         # if not None → use DirectionalGradient
        gamma_dir: float = 1.0,  # projector strength for DirectionalGradient
        eta_dir: float = 1e-6,
        bnd_cond: str = "Neumann",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.stencil = stencil
        self.both_directions = both_directions
        self.epsilon = epsilon
        self.edge_factor = 0.5 if self.both_directions else 1.0

        # voxel sizes (for API parity; no scaling used here)
        if hasattr(domain_geometry, "containers"):
            voxel_sizes = domain_geometry.containers[0].voxel_sizes()
        else:
            voxel_sizes = domain_geometry.voxel_sizes()

        # Δ-operator: plain Gradient OR DirectionalGradient (projected)
        if anatomical is not None:
            if hasattr(anatomical, "as_array"):
                anatomical = anatomical.as_array()
            self.use_dir = True
            self.gradient_op = DirectionalGradient(
                anatomical=anatomical,
                voxel_sizes=voxel_sizes,
                gamma=gamma_dir,
                eta=eta_dir,
                bnd_cond=bnd_cond,
                numpy_out=False,
                stencil=stencil,
                both_directions=both_directions,
                normalize=False
            )
        else:
            self.use_dir = False
            self.gradient_op = Gradient(
                voxel_sizes=voxel_sizes,
                bnd_cond=bnd_cond,
                numpy_out=False,
                stencil=stencil,
                both_directions=both_directions,
                normalize=False
            )

        # Σ-operator: always the standard Sum
        self.sum_op = Sum(
            voxel_sizes=voxel_sizes,
            bnd_cond=bnd_cond,
            numpy_out=False,
            stencil=stencil,
            both_directions=both_directions,
        )

    # ========================== CORE TENSOR METHODS ==========================
    def _edges(self, x_t: torch.Tensor):
        Δ_t = self.gradient_op.direct(x_t)   # (..., n_dir)
        Σ_t = self.sum_op.direct(x_t)        # (..., n_dir)
        if Δ_t.shape != Σ_t.shape:
            raise RuntimeError(
                f"Gradient and Sum produce different edge shapes: {Δ_t.shape} vs {Σ_t.shape}."
            )
        D_t = Σ_t + self.gamma * torch.abs(Δ_t) + self.epsilon
        return Δ_t, Σ_t, D_t

    @staticmethod
    def _partials_phi(Δ_t: torch.Tensor, D_t: torch.Tensor, gamma: float):
        Δ2 = Δ_t * Δ_t
        D2 = D_t * D_t
        dφ_dΔ = (2.0 * Δ_t * D_t - gamma * Δ2 * torch.sign(Δ_t)) / D2
        dφ_dΣ = -Δ2 / D2
        # second partials (a.e.)
        absΔ = torch.abs(Δ_t)
        D3 = D2 * D_t
        φ_dd = 2.0 / D_t - 4.0 * gamma * absΔ / D2 + 2.0 * (gamma ** 2) * Δ2 / D3
        φ_ds = 2.0 * (gamma * Δ2 * torch.sign(Δ_t) - Δ_t * D_t) / D3
        φ_ss = 2.0 * Δ2 / D3
        return dφ_dΔ, dφ_dΣ, φ_dd, φ_ds, φ_ss

    def _value_tensor(self, x_t: torch.Tensor) -> torch.Tensor:
        Δ_t, _, D_t = self._edges(x_t)
        return -self.beta * self.edge_factor * torch.sum((Δ_t * Δ_t) / D_t)

    def _grad_tensor(self, x_t: torch.Tensor) -> torch.Tensor:
        Δ_t, _, D_t = self._edges(x_t)
        dφ_dΔ, dφ_dΣ, _, _, _ = self._partials_phi(Δ_t, D_t, self.gamma)
        cG = self.edge_factor * dφ_dΔ
        cS = self.edge_factor * dφ_dΣ
        return -self.beta * (self.gradient_op.adjoint(cG) + self.sum_op.adjoint(cS))

    def _hess_vec_tensor(self, x_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        Δ_t, _, D_t = self._edges(x_t)
        _, _, φ_dd, φ_ds, φ_ss = self._partials_phi(Δ_t, D_t, self.gamma)
        Gv = self.gradient_op.direct(v_t)
        Sv = self.sum_op.direct(v_t)
        φ_dd *= self.edge_factor
        φ_ds *= self.edge_factor
        φ_ss *= self.edge_factor
        y_edges = φ_dd * Gv + φ_ds * Sv
        z_edges = φ_ds * Gv + φ_ss * Sv
        return -self.beta * (self.gradient_op.adjoint(y_edges) + self.sum_op.adjoint(z_edges))

    def _hess_diag_tensor(self, x_t: torch.Tensor) -> torch.Tensor:
        Δ_t, _, D_t = self._edges(x_t)
        _, _, φ_dd, φ_ds, φ_ss = self._partials_phi(Δ_t, D_t, self.gamma)
        # per-edge endpoint contributions (forward orientation)
        plus  = -(self.beta) * self.edge_factor * (φ_dd + φ_ss + 2.0 * φ_ds)  # source j
        minus = -(self.beta) * self.edge_factor * (φ_dd + φ_ss - 2.0 * φ_ds)  # sink   k
        # Scatter distinct src/sink values via combination of Dᵀ and Sᵀ:
        # For arrays s,t on edges: Dᵀ s + Sᵀ t gives (src: s+t, sink: -s+t).
        # Choose s=(plus-minus)/2, t=(plus+minus)/2 to realise (src: plus, sink: minus).
        s = 0.5 * (plus - minus)
        t = 0.5 * (plus + minus)
        return self.gradient_op.adjoint(s) + self.sum_op.adjoint(t)

    # ============================ PUBLIC API ============================
    def __call__(self, x) -> float:
        if hasattr(x, "containers"):
            total = 0.0
            for c in x.containers:
                xt = _to_tensor(c)
                total += float(self._value_tensor(xt).detach().item())
            return total
        xt = _to_tensor(x)
        return float(self._value_tensor(xt).detach().item())

    def gradient(self, x, out=None):
        def one(img):
            xt = _to_tensor(img)
            gt = self._grad_tensor(xt)
            return _from_tensor_like(img, gt)

        if hasattr(x, "containers"):
            if out is None:
                out = x.copy()
            for i, c in enumerate(x.containers):
                out.containers[i].fill(one(c).as_array())
            return out
        g = one(x)
        if out is None:
            return g
        out.fill(g.as_array())
        return out

    def hessian(self, x, v, out=None):
        xt = _to_tensor(x if hasattr(x, "as_array") else x)
        vt = _to_tensor(v if hasattr(v, "as_array") else v)
        Hv = self._hess_vec_tensor(xt, vt)
        mapped = _from_tensor_like(v, Hv)
        if out is None:
            return mapped
        out.fill(mapped.as_array() if hasattr(mapped, "as_array") else mapped)
        return out

    def hessian_diag(self, x):
        xt = _to_tensor(x if hasattr(x, "as_array") else x)
        Hii = self._hess_diag_tensor(xt)
        return _from_tensor_like(x, Hii)

    def inv_hessian_diag(self, x, damping: float = 1e-8):
        """
        Elementwise inverse of H_ii(λ). Keeps sign. Damps tiny magnitudes.
        """
        xt = _to_tensor(x if hasattr(x, "as_array") else x)
        Hii_t = self._hess_diag_tensor(xt)
        safe  = torch.where(torch.abs(Hii_t) < damping, torch.sign(Hii_t) * damping, Hii_t)
        inv_t = 1.0 / safe
        return _from_tensor_like(x, inv_t)

    # Nonconvex prior — no proximal/convex conjugate
    def convex_conjugate(self, x):
        raise NotImplementedError("RDP is not convex; convex conjugate undefined.")

    def proximal(self, x, tau, out=None):
        raise NotImplementedError("RDP proximal operator has no closed form.")


# -------------------------------------------------------------------------
# Multi-modality WeightedRDP using a Jacobian (acts per modality)
# -------------------------------------------------------------------------
class WeightedRDP(Function):
    """
    Multi-modality RDP with per-modality weights and a Jacobian operator.

    Let X ∈ R^{Nx×Ny×Nz×M}. The Jacobian maps X → G = J X ∈ R^{...×M×d}.
    Define weighted edge differences U = w ⊙ G (broadcast along directions),
    and per-modality neighbour sums Σ = S X (applied independently per modality).
    With D = Σ + γ |U| + ε and φ(U,Σ) = U² / D, the objective is

        M(X) = -β Σ_edges φ.

    Gradient:
        ∇M(X) = -β [ Jᵀ( w ⊙ ∂φ/∂U ) + Sᵀ( ∂φ/∂Σ ) ].

    Hessian–vector product (exact, a.e.):
        Let Gv = J v, Sv = S v, and Uv = w ⊙ Gv. Then
        H(X)v = -β [ Jᵀ( w ⊙ ( φ_dd ⊙ Uv + φ_ds ⊙ Sv ) )
                    + Sᵀ(        φ_ds ⊙ Uv + φ_ss ⊙ Sv ) ].

    Diagonal (per voxel, exact via src/sink scattering):
        For each directed edge (per modality), the endpoint contributions are
            H_jj += -β [ φ_dd + φ_ss + 2 φ_ds ]
            H_kk += -β [ φ_dd + φ_ss - 2 φ_ds ],
        with φ_* evaluated at U and D. We realise distinct src/sink adds
        using Dᵀ and Sᵀ combinations as in the scalar class.

    If both_directions=True, all edge terms are multiplied by 1/2 to de-duplicate.
    """

    def __init__(
        self,
        geometry,                 # BlockDataContainer geometry
        weights,                  # BlockDataContainer of per-modality weights
        beta: float,
        gamma: float,
        stencil: str = "6",
        both_directions: bool = False,
        epsilon: float = 1e-12,
        anatomical=None,          # optional, forwarded to Jacobian
        bnd_cond: str = "Neumann",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.stencil = stencil
        self.both_directions = both_directions
        self.edge_factor = 0.5 if self.both_directions else 1.0
        self.epsilon = epsilon

        # Mapping between BlockDataContainer <-> array
        self.bdc2a = BlockDataContainerToArray(geometry)

        # voxel sizes (for operator init)
        voxel_sizes = geometry.containers[0].voxel_sizes()

        # Jacobian over modalities
        if hasattr(anatomical, "as_array"):
            anatomical = anatomical.as_array()
        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            stencil=stencil,
            both_directions=both_directions,
            bnd_cond=bnd_cond,
        )

        # Sum operator (single-modality); we will batch over modalities
        self.sum_op = Sum(
            voxel_sizes=voxel_sizes,
            bnd_cond=bnd_cond,
            numpy_out=False,
            stencil=stencil,
            both_directions=both_directions,
        )

        # Cache weights on device: shape (..., M)
        w_arr = self.bdc2a.direct(weights)        # numpy or torch
        self.weights = torch.as_tensor(w_arr, device=_DEVICE, dtype=_DTYPE)

    # ---------- helpers to apply Sum per modality ----------
    def _sum_direct_multi(self, X_t: torch.Tensor) -> torch.Tensor:
        # X_t: (..., M)
        M = X_t.shape[-1]
        outs = []
        outs.extend(self.sum_op.direct(X_t[..., m]) for m in range(M))
        return torch.stack(outs, dim=-2)  # (..., M, d)

    def _sum_adjoint_multi(self, E_t: torch.Tensor) -> torch.Tensor:
        # E_t: (..., M, d)
        M = E_t.shape[-2]
        outs = []
        outs.extend(self.sum_op.adjoint(E_t[..., m, :]) for m in range(M))
        return torch.stack(outs, dim=-1)  # (..., M)

    # ========================== CORE TENSOR METHODS ==========================
    def _edges(self, X_t: torch.Tensor):
        # G: (..., M, d)
        G = torch.as_tensor(self.jacobian.direct(X_t), device=_DEVICE, dtype=_DTYPE)
        Σ = self._sum_direct_multi(X_t)                                 # (..., M, d)
        w = self.weights.to(device=X_t.device, dtype=X_t.dtype)         # (..., M)
        U = w.unsqueeze(-1) * G                                         # (..., M, d)
        D = Σ + self.gamma * torch.abs(U) + self.epsilon
        return U, Σ, D, w, G

    @staticmethod
    def _partials_phi(U: torch.Tensor, D: torch.Tensor, gamma: float):
        U2 = U * U
        D2 = D * D
        dφ_dU = (2.0 * U * D - gamma * U2 * torch.sign(U)) / D2
        dφ_dΣ = -U2 / D2
        # seconds (a.e.)
        absU = torch.abs(U)
        D3 = D2 * D
        φ_dd = 2.0 / D - 4.0 * gamma * absU / D2 + 2.0 * (gamma ** 2) * U2 / D3
        φ_ds = 2.0 * (gamma * U2 * torch.sign(U) - U * D) / D3
        φ_ss = 2.0 * U2 / D3
        return dφ_dU, dφ_dΣ, φ_dd, φ_ds, φ_ss

    def _value_tensor(self, X_t: torch.Tensor) -> torch.Tensor:
        U, _, D, _, _ = self._edges(X_t)
        return -self.beta * self.edge_factor * torch.sum((U * U) / D)

    def _grad_tensor(self, X_t: torch.Tensor) -> torch.Tensor:
        U, Σ, D, w, _ = self._edges(X_t)
        dφ_dU, dφ_dΣ, _, _, _ = self._partials_phi(U, D, self.gamma)
        cJ = self.edge_factor * (w.unsqueeze(-1) * dφ_dU)  # (..., M, d)
        cS = self.edge_factor * dφ_dΣ                      # (..., M, d)
        # adjoints:
        Jt = torch.as_tensor(self.jacobian.adjoint(cJ), device=_DEVICE, dtype=_DTYPE)  # (..., M)
        St = self._sum_adjoint_multi(cS)                                                       # (..., M)
        return -self.beta * (Jt + St)

    def _hess_vec_tensor(self, X_t: torch.Tensor, V_t: torch.Tensor) -> torch.Tensor:
        U, Σ, D, w, G = self._edges(X_t)
        _, _, φ_dd, φ_ds, φ_ss = self._partials_phi(U, D, self.gamma)

        Gv = torch.as_tensor(self.jacobian.direct(V_t), device=_DEVICE, dtype=_DTYPE)  # (..., M, d)
        Sv = self._sum_direct_multi(V_t)                                              # (..., M, d)
        Uv = w.unsqueeze(-1) * Gv

        # Jᵀ term gets an extra factor w from gradient chain rule
        Y = φ_dd * Uv + φ_ds * Sv                             # (..., M, d)
        Z = φ_ds * Uv + φ_ss * Sv                             # (..., M, d)

        part_J = torch.as_tensor(self.jacobian.adjoint(w.unsqueeze(-1) * Y),
                                device=_DEVICE, dtype=_DTYPE)
        part_S = self._sum_adjoint_multi(Z)

        return -self.beta * self.edge_factor * (part_J + part_S)

    def _hess_diag_tensor(self, X_t: torch.Tensor) -> torch.Tensor:
        U, Σ, D, w, _ = self._edges(X_t)
        _, _, φ_dd, φ_ds, φ_ss = self._partials_phi(U, D, self.gamma)

        plus  = -(self.beta) * self.edge_factor * (φ_dd + φ_ss + 2.0 * φ_ds)
        minus = -(self.beta) * self.edge_factor * (φ_dd + φ_ss - 2.0 * φ_ds)

        # Scatter distinct src/sink adds via combination of Jᵀ and Sᵀ:
        s = 0.5 * (plus - minus)    # for Jᵀ (difference)
        t = 0.5 * (plus + minus)    # for Sᵀ (sum)

        # Jᵀ needs the same per-edge weighting factor w as in the gradient path
        diag_J = torch.as_tensor(self.jacobian.adjoint(w.unsqueeze(-1) * s),
                                device=_DEVICE, dtype=_DTYPE)  # (..., M)
        diag_S = self._sum_adjoint_multi(t)                                     # (..., M)
        return diag_J + diag_S

    # ============================ PUBLIC API ============================
    def __call__(self, x) -> float:
        X_arr = self.bdc2a.direct(x)                     # (nx,ny,nz,M)
        X_t   = torch.as_tensor(X_arr, device=_DEVICE, dtype=_DTYPE)
        return float(self._value_tensor(X_t).detach().item())

    def gradient(self, x, out=None):
        X_arr = self.bdc2a.direct(x)
        X_t   = torch.as_tensor(X_arr, device=_DEVICE, dtype=_DTYPE)
        g_t   = self._grad_tensor(X_t)                  # (..., M)
        return self.get_arr_and_fill(g_t, out)

    def hessian(self, x, v, out=None):
        X_arr = self.bdc2a.direct(x)
        V_arr = self.bdc2a.direct(v)
        X_t   = torch.as_tensor(X_arr, device=_DEVICE, dtype=_DTYPE)
        V_t   = torch.as_tensor(V_arr, device=_DEVICE, dtype=_DTYPE)
        Hv_t  = self._hess_vec_tensor(X_t, V_t)
        return self.get_arr_and_fill(Hv_t, out)

    def get_arr_and_fill(self, arg0, out):
        g_arr = arg0.detach().to("cpu").numpy()
        result = self.bdc2a.adjoint(g_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def hessian_diag(self, x):
        X_arr = self.bdc2a.direct(x)
        X_t   = torch.as_tensor(X_arr, device=_DEVICE, dtype=_DTYPE)
        Hii_t = self._hess_diag_tensor(X_t)
        Hii_arr = Hii_t.detach().to("cpu").numpy()
        return self.bdc2a.adjoint(Hii_arr)

    def inv_hessian_diag(self, x, damping: float = 1e-8):
        X_arr = self.bdc2a.direct(x)
        X_t   = torch.as_tensor(X_arr, device=_DEVICE, dtype=_DTYPE)
        Hii_t = self._hess_diag_tensor(X_t)
        safe  = torch.where(torch.abs(Hii_t) < damping, torch.sign(Hii_t) * damping, Hii_t)
        inv_t = 1.0 / safe
        inv_arr = inv_t.detach().to("cpu").numpy()
        return self.bdc2a.adjoint(inv_arr)

    # Nonconvex prior — no proximal/convex conjugate
    def convex_conjugate(self, x):
        raise NotImplementedError("RDP is not convex; convex conjugate undefined.")

    def proximal(self, x, tau, out=None):
        raise NotImplementedError("RDP proximal operator has no closed form.")
