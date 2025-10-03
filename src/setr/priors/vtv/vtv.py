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
from setr.utils.sirf import get_array


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
        hessian="slow",
        bnd_cond="Periodic",
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = get_array(anatomical)
        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            stencil=stencil,
            both_directions=both_directions,
            bnd_cond=bnd_cond,
        )

        self.smoothing = smoothing
        self.hessian = hessian
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

        return self.vtv(U)

    def gradient(self, x, out=None):
        
        x_arr = self.bdc2a.direct(x) 
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J

        inner = w * self.vtv.gradient(U)

        ret = self.jacobian.adjoint(inner)
        
        return self.bdc2a.adjoint(ret, out=out)
    
    def _preconditioner_weights_core_fast(self, x_arr, eta: float = 0.7, epsilon: float = 1e-8):
        """
        Diagonal MM/IRLS preconditioner for (directional) TNV with a directional Jacobian:

            P_{j,m} = eta * [ omega_j * S_jm ] * b_{j,m}^2 , floored by epsilon,

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
        J = self.jacobian.direct(x_arr) 
        A = self.weights.unsqueeze(-1) * J

        omega_sigma = self.vtv.hessian_surrogate(A)
        omega = omega_sigma.sum(dim=-1)

        S = self.jacobian.sensitivity(x_arr) 
        S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
        if S.ndim < A.ndim:
            S = S.expand_as(A)
        S2 = S * S 
        
        nx, ny, nz, M, d = A.shape

        def _counts_1d(n: int):
            c = torch.full((n,), 2.0, device=A.device, dtype=A.dtype)
            if n > 0:
                c[0] = 1.0
                if n > 1:
                    c[-1] = 1.0
            return c 

        C = torch.full((nx, ny, nz, d), 2.0, device=A.device, dtype=A.dtype)

        if d >= 1:
            cx = _counts_1d(nx).view(nx, 1, 1).expand(nx, ny, nz)
            C[..., 0] = cx
        if d >= 2:
            cy = _counts_1d(ny).view(1, ny, 1).expand(nx, ny, nz)
            C[..., 1] = cy
        if d >= 3:
            cz = _counts_1d(nz).view(1, 1, nz).expand(nx, ny, nz)
            C[..., 2] = cz

        C = C.unsqueeze(-2).expand(nx, ny, nz, M, d)

        S_jm = (S2 * C).sum(dim=-1)

        P_diag = (omega.unsqueeze(-1) * S_jm) * (self.weights * self.weights)  # (..., M)
        P_diag = eta * P_diag
        P_diag = torch.clamp(P_diag, min=epsilon)

        return P_diag
    
    def _hessian_diag_fast(self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns a BlockDataContainer holding a diagonal positive surrogate H ≈ ∇²V(x).
        Shape matches x (nx,ny,nz,M). Guaranteed H >= epsilon.
        """
        x_arr = self.bdc2a.direct(x)   
        H = self._preconditioner_weights_core_fast(x_arr, eta, epsilon)  
        return self.bdc2a.adjoint(H, out=out)

    def _inv_hessian_diag_fast(self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns a BlockDataContainer with the elementwise inverse of hessian_diag(x).
        Since H is floored by epsilon, inv(H) is bounded above by 1/epsilon.
        """
        x_arr = self.bdc2a.direct(x)
        H = self._preconditioner_weights_core_fast(x_arr, eta, epsilon)
        Hinv = torch.reciprocal(H)
        Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=1.0/epsilon, neginf=0.0)
        return self.bdc2a.adjoint(Hinv, out=out)

    def _preconditioner_weights_core_slow(self, x_arr):
        """
        Core implementation to calculate the diagonal preconditioner weights
        based on the corrected Hessian derivation.
        """
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        A_field = w * J

        hess_coeffs, rank_one_fields = self.vtv.hessian_components(A_field)

        P_diag = torch.zeros_like(x_arr)

        num_singular_values = rank_one_fields.shape[-3]
        for k in range(num_singular_values):
            C_k_field = rank_one_fields[..., k, :, :] 

            influence_image = self.jacobian.adjoint(
                w * C_k_field
            ) 
            h_double_prime_k = hess_coeffs[..., k]

            h_double_prime_k = h_double_prime_k.unsqueeze(-1)
            
            P_diag += h_double_prime_k * (influence_image**2)

        return P_diag

    def _hessian_diag_slow(self, x, out=None):
        """
        Computes a diagonal approximation of the Hessian, suitable for preconditioning.
        This method implements the formula:
        p_i = sum_k h''(s_k) * ( (J^T u_k v_k^T)_i )^2
        """
        x_arr = self.bdc2a.direct(x)
        diag_arr = self._preconditioner_weights_core_slow(x_arr)

        return self.bdc2a.adjoint(diag_arr, out=out)

    def _inv_hessian_diag_slow(self, x, out=None, epsilon=1e-9):
        """
        Computes the action of the inverse of the diagonal Hessian approximation.
        This is a simple element-wise division by the preconditioner weights.
        """
        diag_arr = self._preconditioner_weights_core_slow(self.bdc2a.direct(x))

        inv_arr = torch.reciprocal(diag_arr + epsilon)
        torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0, out=inv_arr)

        return self.bdc2a.adjoint(inv_arr, out=out)

    def hessian_diag(self, x, out=None):
        if self.hessian == "slow":
            return self._hessian_diag_slow(x, out=out)
        elif self.hessian == "fast":
            return self._hessian_diag_fast(x, out=out)
        else:
            raise ValueError("Unknown Hessian type")
        
    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        if self.hessian == "slow":
            return self._inv_hessian_diag_slow(x, out=out, epsilon=epsilon)
        elif self.hessian == "fast":
            return self._inv_hessian_diag_fast(x, out=out, epsilon=epsilon)
        else:
            raise ValueError("Unknown Hessian type")


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
        hessian="slow",
        stencil='6',
        bnd_cond="Periodic",
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if hasattr(anatomical, "as_array"):  # ImageData
            anatomical = get_array(anatomical)

        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            both_directions=both_directions,
            stencil=stencil,
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
        
        return total_tv

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
        
        return self.bdc2a.adjoint(ret, out=out)

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
            proxU[..., m, :] = self.tv.proximal(modality_gradients, tau)


        ret = self.jacobian.adjoint(proxU * w)  
        return self.bdc2a.adjoint(ret, out=out)

    def hessian_diag(self, x, out=None, stabiliser: float = 1e-9, positive: bool = True):
        """
        Diagonal approximation in image space:
        sum_j (w^2 * s_j^2) * h_j(U), with U = w * (Jx).
        Requires jacobian.sensitivity(...) -> (..., M, d) per-direction sensitivities.
        """
        x_arr = self.bdc2a.direct(x)          
        J     = self.jacobian.direct(x_arr)    
        w     = self.weights.unsqueeze(-1)    
        U     = w * J                         


        h_dir = torch.zeros_like(U)
        for m in range(U.shape[-2]):  
            modality_gradients = U[..., m, :]
            h_dir[..., m, :] = self.tv.hessian_dir_diag(modality_gradients, stabiliser=stabiliser, positive=positive)


        S = torch.as_tensor(self.jacobian.sensitivity(x_arr), device=U.device, dtype=U.dtype)  
        S2 = S * S
        w2 = (self.weights ** 2).unsqueeze(-1)  

        h_img = torch.sum(w2 * S2 * h_dir, dim=-1) 

        return self.bdc2a.adjoint(h_img, out=out)


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
        smoothing="fair",
        norm="l2",
        anatomical=None,
        both_directions=False,
        stencil='6',
        hessian="slow"
    ):
        """
        Initialize single-modality Total Variation prior.
        
        Args:
            geometry: ImageData template defining the image space
            weight: Scalar weighting factor for the TV prior
            delta: Smoothing parameter for the TV function
            smoothing: Smoothing function type ('fair', 'huber', etc.)
            norm: Vector norm type ('l2', 'l1', etc.) 
            anatomical: Optional anatomical image for directional guidance
            both_directions: Use bidirectional gradients
            stencil: Stencil connectivity ('6', '18', '26')
            hessian: Hessian approximation method
        """
        from setr.core.gradients import Gradient, DirectionalGradient
        
        self.geometry = geometry
        self.weight = float(weight)
        self.delta = float(delta)
        self.smoothing = smoothing
        self.hessian = hessian
        
        voxel_sizes = geometry.voxel_sizes()
        
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
        return self.tv(weighted_grad)

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
        
        if hasattr(result_arr, 'detach'):  # torch tensor
            result_arr = result_arr.detach().cpu().numpy()
        
        out.fill(result_arr)
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
        prox_grad = self.tv.proximal(weighted_grad, tau)
        
        # Apply weight and compute adjoint
        weighted_prox = self.weight * prox_grad
        result_arr = self.gradient_op.adjoint(weighted_prox)
        
        # Convert back to ImageData
        if out is None:
            out = x.clone()
        
        if hasattr(result_arr, 'detach'):  # torch tensor
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
        
        if hasattr(h_img, 'detach'):  # torch tensor
            h_img = h_img.detach().cpu().numpy()
        
        out.fill(h_img)
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
        
        if hasattr(inv_arr, 'detach'):  # torch tensor
            inv_arr = inv_arr.detach().cpu().numpy()
        
        out.fill(inv_arr)
        return out

