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
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = get_array(anatomical)

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
            from .schatten_norm_gpu_stable import GPUVectorialTotalVariation as GpuVTV
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
        P_diag = torch.clamp(P_diag, min=epsilon)

        return P_diag
    
    def _hessian_diag_fast(self, x, eta: float = 0.7, epsilon: float = 1e-8, out=None):
        """
        Returns a BlockDataContainer holding a diagonal positive surrogate H ≈ ∇²V(x).
        Shape matches x (nx,ny,nz,M). Guaranteed H >= epsilon.
        """
        x_arr = self.bdc2a.direct(x)                               # (nx,ny,nz,M)
        H = self._preconditioner_weights_core_fast(x_arr, eta, epsilon)   # torch tensor (...,M)
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
        # 1. Compute the Jacobian field, Jx.
        J = self.jacobian.direct(x_arr)

        # 2. Apply the data-fidelity weights. This becomes the input 'A' for the VTV function.
        w = self.weights.unsqueeze(-1)
        A_field = w * J

        # 3. Call the backend to get the Hessian components from the SVD of A_field.
        hess_coeffs, rank_one_fields = self.vtv.hessian_components(A_field)

        # 4. Initialize the final diagonal preconditioner tensor P.
        P_diag = torch.zeros_like(x_arr)

        # 5. Loop over each singular mode k, calculate its contribution, and accumulate.
        num_singular_values = rank_one_fields.shape[-3]
        for k in range(num_singular_values):
            # a) Get the field of rank-1 matrices for this mode
            C_k_field = rank_one_fields[..., k, :, :]  # Shape: (nx, ny, nz, M, d)

            # b) The formula is p_i = sum_k h''(s_k) * ( (J^T u_k v_k^T)_i )^2
            # The rank-one fields are u_k v_k^T from A=wJx. We need to compute J^T(w * u_k v_k^T).
            influence_image = self.jacobian.adjoint(
                w * C_k_field
            ) 
            # c) Get the corresponding h''(s_k) coefficients for this mode.
            h_double_prime_k = hess_coeffs[..., k]

            # d) Unsqueeze the coefficient to broadcast over the M modalities.
            h_double_prime_k = h_double_prime_k.unsqueeze(-1)
            
            # e) Accumulate the contribution for this mode: h''(s_k) * (J^T u_k v_k^T)^2
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
        # 1. Get the preconditioner weights
        diag_arr = self._preconditioner_weights_core_slow(self.bdc2a.direct(x))

        # 2. Invert the weights, adding epsilon for stability
        inv_arr = torch.reciprocal(diag_arr + epsilon)
        torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0, out=inv_arr)

        # 3. Convert back to BlockDataContainer
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
        stencil='6'
    ):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if hasattr(anatomical, "as_array"):  # ImageData
            anatomical = get_array(anatomical)

        # Jacobian operator: maps N×M images → N×M×d (stack of finite diffs)
        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            both_directions=both_directions,
            stencil=stencil,
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

        # Apply GPUVectorNorm to each modality separately
        total_tv = 0.0
        for m in range(U.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (..., d)
            modality_gradients = U[..., m, :]
            total_tv += self.tv(modality_gradients)
        
        return total_tv

    def gradient(self, x, out=None):

        x_arr = self.bdc2a.direct(x)  # (nx, ny, nz, M)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J  # (..., M, d)

        # Apply GPUVectorNorm gradient to each modality separately
        inner = torch.zeros_like(U)
        for m in range(U.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (..., d)
            modality_gradients = U[..., m, :]
            inner[..., m, :] = w[..., m, :] * self.tv.gradient(modality_gradients)

        ret = self.jacobian.adjoint(inner)  # shape (nx,ny,nz,M)
        
        return self.bdc2a.adjoint(ret, out=out)

    def proximal(self, x, tau, out=None):
        """
        Proximal operator for total variation.
        """
        x_arr = self.bdc2a.direct(x)  # (nx,ny,nz,M)
        J = self.jacobian.direct(x_arr)  # (nx,ny,nz,M,d)

        w = self.weights.unsqueeze(-1)
        U = w * J  # (...,M,d)

        # Apply GPUVectorNorm proximal to each modality separately
        proxU = torch.zeros_like(U)
        for m in range(U.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (..., d)
            modality_gradients = U[..., m, :]
            proxU[..., m, :] = self.tv.proximal(modality_gradients, tau)

        # Push back to image space:
        ret = self.jacobian.adjoint(proxU * w)  # (nx,ny,nz,M)
        return self.bdc2a.adjoint(ret, out=out)

    def hessian_diag(self, x, out=None, stabiliser: float = 1e-9, positive: bool = True):
        """
        Diagonal approximation in image space:
        sum_j (w^2 * s_j^2) * h_j(U), with U = w * (Jx).
        Requires jacobian.sensitivity(...) -> (..., M, d) per-direction sensitivities.
        """
        x_arr = self.bdc2a.direct(x)             # (..., M)
        J     = self.jacobian.direct(x_arr)      # (..., M, d)
        w     = self.weights.unsqueeze(-1)       # (..., M, 1)
        U     = w * J                            # (..., M, d)

        # Apply GPUVectorNorm hessian_dir_diag to each modality separately
        h_dir = torch.zeros_like(U)
        for m in range(U.shape[-2]):  # Loop over M modalities
            # Extract gradients for modality m: shape (..., d)
            modality_gradients = U[..., m, :]
            h_dir[..., m, :] = self.tv.hessian_dir_diag(modality_gradients, stabiliser=stabiliser, positive=positive)

        # Finite-difference sensitivities per direction
        S = torch.as_tensor(self.jacobian.sensitivity(x_arr), device=U.device, dtype=U.dtype)  # (..., M, d)
        S2 = S * S
        w2 = (self.weights ** 2).unsqueeze(-1)   # (..., M, 1)

        h_img = torch.sum(w2 * S2 * h_dir, dim=-1)  # (..., M)

        return self.bdc2a.adjoint(h_img, out=out)


    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        # reuse hessian_diag code
        hess_arr = self.hessian_diag(x, out=None)  # BDC or array
        # if it’s a numpy array, convert to torch:
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

