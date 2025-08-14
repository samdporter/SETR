# VTV.py

from cil.optimisation.functions import Function

try:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = "cpu"
import numpy as np
from sirf.STIR import ImageData

from setr.core.gradients import Jacobian
from setr.utils import BlockDataContainerToArray


class WeightedVectorialTotalVariation(Function):
    """
    GPU vectorial total variation with optional gradient normalization.
    """

    def __init__(
        self,
        geometry,
        weights,
        delta,
        smoothing="fair",
        norm="nuclear",
        anatomical=None,
        stable=True,
        stencil="6",
        both_directions=False,
        tail_singular_values=None,
        hessian="diagonal",
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = anatomical.as_array()

        # Jacobian operator: maps N×M images → N×M×d (stack of finite diffs)
        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            stencil=stencil,
            both_directions=both_directions,
        )

        self.smoothing = smoothing
        self.hessian = hessian
        self.bdc2a = BlockDataContainerToArray(geometry)

        # Pull out the weights as an array/tensor of shape (Nx,Ny,Nz,M)
        self.weights = self.bdc2a.direct(weights)  # shape (..., M)

        # Inverse‐weight is used in inv_hessian_diag
        self.inv_weights = torch.reciprocal(self.weights)
        self.inv_weights = torch.nan_to_num(self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0)

        if tail_singular_values is not None:
            print(f"tail_singular_values = {tail_singular_values}")
        if stable:
            from .schatten_norm_gpu_slow import GPUVectorialTotalVariation as GpuVTV
        else:
            from .schatten_norm_gpu import GPUVectorialTotalVariation as GpuVTV
        self.vtv = GpuVTV(
            eps=delta,
            norm=norm,
            smoothing_function=smoothing,
            tail=tail_singular_values,
        )

    def __call__(self, x):
        x_arr = self.bdc2a.direct(x)  # shape (nx, ny, nz, M)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)  # (..., M, 1)
        U = w * J  # (..., M, d)

        return self.vtv(U)

    def gradient(self, x, out=None):
        
        x_arr = self.bdc2a.direct(x)  # (nx, ny, nz, M)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)  # (..., M, 1)
        U = w * J  # (..., M, d)

        inner = w * self.vtv.gradient(U)  # the vtv.gradient already accounts for smoothing etc.

        ret = self.jacobian.adjoint(inner)  # shape (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

    def _preconditioner_weights_core(self, x_arr, eta: float = 0.7, eps_P: float = 1e-8):
        """
        Diagonal MM/IRLS preconditioner for (directional) TNV with a directional Jacobian:

            P_{j,m} = eta * [ omega_j * S_jm ] * b_{j,m}^2 , floored by eps_P,

        where:
        - omega_j = sum_ell w(sigma_{j,ell}) from self.vtv.hessian_surrogate(A),
                    computed SVD-free inside your schatten backend (tail handled there).
        - S_jm    = sum_over_dirs ( scale_{j,m,dir}^2 * n_dir(j) ),
                    with scale_{·} coming from `Jacobian.sensitivity` and
                    n_dir(j) ∈ {1,2} the voxelwise participation count (boundary vs interior).

        Assumptions:
        * `self.jacobian.direct` already includes any directional projector; same for `adjoint`.
        * `self.jacobian.sensitivity(images)` returns per-direction scales (broadcast spatially),
            consistent with the `direct/adjoint` scaling (Δ, bank, both_directions).
        """
        # ----- 1) Build A = w ⊙ (J x)
        J = self.jacobian.direct(x_arr)                 # shape: (nx, ny, nz, M, d)
        A = self.weights.unsqueeze(-1) * J              # shape: (nx, ny, nz, M, d)

        # ----- 2) Spectral IRLS weights (SVD-free inside your schatten backend)
        #         omega_sigma: (..., r)  -> sum over singular values -> omega: (...,)
        omega_sigma = self.vtv.hessian_surrogate(A)     # per-σ weights w(σ)
        omega = omega_sigma.sum(dim=-1)                 # shape: (nx, ny, nz)

        # ----- 3) Per-direction scales from operator (already Δ/bank-consistent)
        #         S has shape (nx, ny, nz, M, d) or is broadcastable to it
        S = self.jacobian.sensitivity(x_arr)            # numpy or torch
        S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
        if S.ndim < A.ndim:
            # make sure it broadcasts to (nx, ny, nz, M, d)
            S = S.expand_as(A)
        # square scales
        S2 = S * S                                      # (..., M, d)

        # ----- 4) Voxelwise participation counts n_dir(j) ∈ {1,2}
        #         For forward differences along x/y/z: interior=2, boundary=1.
        nx, ny, nz, M, d = A.shape

        def _counts_1d(n: int):
            c = torch.full((n,), 2.0, device=A.device, dtype=A.dtype)
            if n > 0:
                c[0] = 1.0
                if n > 1:
                    c[-1] = 1.0
            return c  # (n,)

        # start with all-2 (safe upper bound), then overwrite first min(3,d) axes with boundary-aware counts
        C = torch.full((nx, ny, nz, d), 2.0, device=A.device, dtype=A.dtype)

        # x-axis counts in dir slot 0 (if present)
        if d >= 1:
            cx = _counts_1d(nx).view(nx, 1, 1).expand(nx, ny, nz)
            C[..., 0] = cx
        # y-axis counts in dir slot 1 (if present)
        if d >= 2:
            cy = _counts_1d(ny).view(1, ny, 1).expand(nx, ny, nz)
            C[..., 1] = cy
        # z-axis counts in dir slot 2 (if present)
        if d >= 3:
            cz = _counts_1d(nz).view(1, 1, nz).expand(nx, ny, nz)
            C[..., 2] = cz

        # broadcast counts across modalities: (nx,ny,nz,1,d) -> (nx,ny,nz,M,d)
        C = C.unsqueeze(-2).expand(nx, ny, nz, M, d)

        # ----- 5) Assemble operator-consistent diagonal energy S_jm
        #         S_jm = sum_dir ( S^2 * counts )
        S_jm = (S2 * C).sum(dim=-1)                     # shape: (nx, ny, nz, M)

        # ----- 6) Final diagonal with damping and floor
        P_diag = (omega.unsqueeze(-1) * S_jm) * (self.weights * self.weights)  # (..., M)
        P_diag = eta * P_diag
        P_diag = torch.clamp(P_diag, min=eps_P)

        return P_diag



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
        smoothing="fair",
        norm="l2",
        anatomical=None,
        diagonal=False,
        both_directions=False,
        hessian="diagonal",
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if hasattr(anatomical, "as_array"):  # ImageData
            anatomical = anatomical.as_array()

        # Jacobian operator: maps N×M images → N×M×d (stack of finite diffs)
        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            diagonal=diagonal,
            both_directions=both_directions,
        )

        self.smoothing = smoothing
        self.hessian = hessian
        self.bdc2a = BlockDataContainerToArray(geometry)

        # Pull out the weights as an array/tensor of shape (Nx,Ny,Nz,M)
        self.weights = self.bdc2a.direct(weights)  # shape (..., M)

        # Inverse‐weight is used in inv_hessian_diag
        self.inv_weights = torch.reciprocal(self.weights)
        self.inv_weights = torch.nan_to_num(self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0)

        # Create the GPU total variation backend
        from .vector_norm import GPUVectorNorm

        self.tv = GPUVectorNorm(eps=delta, norm=norm, smoothing_function=smoothing)

    def __call__(self, x):
        
        x_arr = self.bdc2a.direct(x)  # shape (nx, ny, nz, M)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J  # (..., M, d)

        return self.tv(U)

    def gradient(self, x, out=None):

        x_arr = self.bdc2a.direct(x)  # (nx, ny, nz, M)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J  # (..., M, d)

        inner = w * self.tv.gradient(U)  # the tv.gradient already accounts for smoothing etc.

        ret = self.jacobian.adjoint(inner)  # shape (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

    def proximal(self, x, tau, out=None):
        """
        Proximal operator for total variation.
        """
        x_arr = self.bdc2a.direct(x)  # (nx,ny,nz,M)
        J = self.jacobian.direct(x_arr)  # (nx,ny,nz,M,d)

        w = self.weights.unsqueeze(-1)
        U = w * J  # (...,M,d)

        proxU = self.tv.proximal(U, tau)  # (...,M,d)

        # Push back to image space:
        ret = self.jacobian.adjoint(proxU * w)  # (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

    def hessian_diag(self, x, out=None):

        x_arr = self.bdc2a.direct(x)  # (nx,ny,nz,M)
        J = self.jacobian.direct(x_arr)  # (nx,ny,nz,M,d)

        w = self.weights.unsqueeze(-1)  # (nx,ny,nz,M,1)
        U = w * J  # (nx,ny,nz,M,d)

        S_np = self.jacobian.sensitivity(x_arr)  # numpy or torch
        S = torch.as_tensor(S_np, device=U.device)

        batch_shape = U.shape[:-1]  # (...,M)
        U_flat = U.reshape(-1, U.shape[-1])  # (B, d)
        phi2 = self.tv.phi_hessian(U_flat)  # (B,)
        phi2 = phi2.reshape(*batch_shape)  # (...,M)

        S2_sum = torch.sum(S * S, dim=-1)  # (...,M)
        hess_arr = phi2 * (self.weights**2) * S2_sum

        result = self.bdc2a.adjoint(hess_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        # reuse hessian_diag code
        hess_arr = self.hessian_diag(x, out=None)  # BDC or array
        # if it’s a numpy array, convert to torch:
        H = torch.as_tensor(
            hess_arr if isinstance(hess_arr, torch.Tensor) else self.bdc2a.direct(hess_arr),
            device=device,
        )
        inv_arr = torch.reciprocal(H + epsilon)
        result = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result)
            return out
        return result
