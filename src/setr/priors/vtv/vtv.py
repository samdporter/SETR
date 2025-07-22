# VTV.py

from cil.optimisation.functions import Function
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    device = 'cpu'
import numpy as np
from sirf.STIR import ImageData
from setr.core.gradients import Jacobian
from setr.utils import BlockDataContainerToArray

class WeightedVectorialTotalVariation(Function):
    """
    GPU/CPU vectorial total variation with optional gradient normalization.
    """

    def __init__(
            self, geometry, weights,
            delta, smoothing='fair', norm='nuclear',
            gpu=True, anatomical=None, stable=True,
            diagonal = False, both_directions=False,
            tail_singular_values=None,
            hessian='diagonal',):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = anatomical.as_array()

        # Jacobian operator: maps N×M images → N×M×d (stack of finite diffs)
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical,
            gpu=gpu, numpy_out=not gpu,
            method="forward",
            diagonal=diagonal,
            both_directions=both_directions,
        )

        self.smoothing = smoothing
        self.hessian = hessian
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)

        # Pull out the weights as an array/tensor of shape (Nx,Ny,Nz,M)
        self.weights = self.bdc2a.direct(weights)  # shape (..., M)
        self.gpu = gpu

        # Inverse‐weight is used in inv_hessian_diag
        if gpu:
            self.inv_weights = torch.reciprocal(self.weights)
            self.inv_weights = torch.nan_to_num(
                self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0
            )
        else:
            self.inv_weights = np.reciprocal(self.weights, where=self.weights != 0)

        # Choose GPU or CPU backend for the actual vectorial TV
        if gpu:
            if tail_singular_values is not None:
                print(f"tail_singular_values = {tail_singular_values}")
            if stable:
                from .schatten_norm_gpu_slow import GPUVectorialTotalVariation as GpuVTV
            else:
                from .schatten_norm_gpu import GPUVectorialTotalVariation as GpuVTV
            self.vtv = GpuVTV(eps=delta, norm=norm, smoothing_function=smoothing, tail=tail_singular_values)
        else:
            if tail_singular_values is not None:
                raise ValueError("tail_singular_values is only implemented for GPU")
            from .schatten_norm_cpu import CPUVectorialTotalVariation
            self.vtv = CPUVectorialTotalVariation(delta, smoothing_function=smoothing)

    def __call__(self, x):
        # 1) Pull x (BDC) → array/tensor of shape (..., M)
        x_arr = self.bdc2a.direct(x)                        # shape (nx, ny, nz, M)
        # 2) Compute the Jacobian: J_raw shape (nx, ny, nz, M, d)
        J = self.jacobian.direct(x_arr)

        # 4) Multiply by weights (shape (..., M) → expand to (..., M, 1))
        w = self.weights.unsqueeze(-1)                      # (..., M, 1)
        U = w * J                                            # (..., M, d)

        # 5) Call GPU/CPU VTV on U (returns per‐voxel TV components)
        return self.vtv(U)

    def call_no_sum(self, x):
        # Same as __call__, but returns the un-summed per‐voxel quantities
        x_arr = self.bdc2a.direct(x)
        J = self.jacobian.direct(x_arr)

        w = self.weights.unsqueeze(-1)
        U = w * J
        out_U = self.vtv.call_no_sum(U)   # shape: (nx,ny,nz,M)
        return self.bdc2a.adjoint(out_U)

    def gradient(self, x, out=None):
        # 1) Pull x → array/tensor
        x_arr = self.bdc2a.direct(x)                        # (nx, ny, nz, M)
        # 2) Compute J_raw (nx, ny, nz, M, d)
        J = self.jacobian.direct(x_arr)

        # 4) Multiply by weights
        w = self.weights.unsqueeze(-1)          # (..., M, 1)
        U = w * J                                # (..., M, d)

        # 5) Compute inner gradient: shape (..., M, d)
        inner = w * self.vtv.gradient(U)        # the vtv.gradient already accounts for smoothing etc.

        # 7) Push back to image‐space: J^* (−divergence)
        # In our code, Jacobian.adjoint performs exactly −divg
        ret = self.jacobian.adjoint(inner)      # shape (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

    def _preconditioner_weights_core(self, x_arr):
        """
        Core implementation to calculate the diagonal preconditioner weights
        based on the corrected Hessian derivation.
        """
        # 1. Compute the Jacobian field, Jx.
        #    We do not apply normalization here as the Hessian is for the unnormalized functional.
        J = self.jacobian.direct(x_arr)

        # 2. Apply the data-fidelity weights. This becomes the input 'A' for the VTV function.
        #    A = w * Jx
        w = self.weights.unsqueeze(-1) if self.gpu else np.expand_dims(self.weights, -1)
        A_field = w * J

        # 3. Call the backend to get the Hessian components from the SVD of A_field.
        #    - hess_coeffs is h''(s_k)
        #    - rank_one_fields is the collection of u_k v_k^T matrices for each k
        hess_coeffs, rank_one_fields = self.vtv.hessian_components(A_field)
        # hess_coeffs shape: (nx, ny, nz, r)
        # rank_one_fields shape: (nx, ny, nz, r, M, d)

        num_singular_values = rank_one_fields.shape[-3]

        # 4. Initialize the final diagonal preconditioner tensor P.
        #    The result should have the same shape as the input image array.
        P_diag = torch.zeros_like(x_arr) if self.gpu else np.zeros_like(x_arr)

        # 5. Loop over each singular mode k, calculate its contribution, and accumulate.
        for k in range(num_singular_values):
            # a) Get the field of rank-1 matrices for this mode
            C_k_field = rank_one_fields[..., k, :, :]  # Shape: (nx, ny, nz, M, d)

            # b) The formula is p_i = sum_k h''(s_k) * ( (J^T u_k v_k^T)_i )^2
            #    The weights `w` are already baked into the SVD components (u_k, v_k, s_k).
            #    The operator is effectively `wJ`. The adjoint is `(wJ)^T = J^T w`.
            #    So we need to compute J^T (w * C_k).
            
            # The rank-one fields are u_k v_k^T from A=wJx. We need to compute J^T(w * u_k v_k^T).
            # Since J is the gradient operator and w is a per-modality weight, J^T(w*...) is correct.
            influence_image = self.jacobian.adjoint(w * C_k_field) # Shape: (nx, ny, nz, M)

            # c) Get the corresponding h''(s_k) coefficients for this mode.
            #    Shape: (nx, ny, nz)
            h_double_prime_k = hess_coeffs[..., k]

            # d) Unsqueeze the coefficient to broadcast over the M modalities.
            #    Shape becomes (nx, ny, nz, 1)
            h_double_prime_k = h_double_prime_k.unsqueeze(-1) if self.gpu else np.expand_dims(h_double_prime_k, -1)

            # e) Accumulate the contribution for this mode: h''(s_k) * (J^T u_k v_k^T)^2
            P_diag += h_double_prime_k * (influence_image ** 2)

        return P_diag

    def hessian_diag(self, x, out=None):
        """
        Computes a diagonal approximation of the Hessian, suitable for preconditioning.
        This method implements the formula:
        p_i = sum_k h''(s_k) * ( (J^T u_k v_k^T)_i )^2
        """
        x_arr = self.bdc2a.direct(x)
        diag_arr = self._preconditioner_weights_core(x_arr)
        
        # According to the derivation, no extra weighting is needed here.
        # The weights `w` were correctly applied to the input of the SVD.
        
        result = self.bdc2a.adjoint(diag_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        """
        Computes the action of the inverse of the diagonal Hessian approximation.
        This is a simple element-wise division by the preconditioner weights.
        """
        # 1. Get the preconditioner weights
        diag_arr = self._preconditioner_weights_core(self.bdc2a.direct(x))
        
        # 2. Invert the weights, adding epsilon for stability
        if self.gpu:
            inv_arr = torch.reciprocal(diag_arr + epsilon)
            torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0, out=inv_arr)
        else:
            inv_arr = np.reciprocal(diag_arr + epsilon)

        # 3. Convert back to BlockDataContainer
        result = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            # Note: This operation does not apply to the input 'x', but returns a scaling array.
            # The typical use is P^-1 * g. So here we return the scaling array.
            # The calling function should multiply this by the gradient.
            # To match the expected 'Function' API, perhaps it should act on x?
            # Assuming the goal is to return the inverted diagonal P^-1 itself.
            out.fill(result)
            return out
        return result
    

    def proximal(self, x, tau, out=None):
        x_arr = self.bdc2a.direct(x)                  # (nx,ny,nz,M)
        J = self.jacobian.direct(x_arr)           # (nx,ny,nz,M,d)

        w = self.weights.unsqueeze(-1)               # (...,M,1)
        U = w * J                                     # (...,M,d)

        proxU = self.vtv.proximal(U, tau)             # (...,M,d)

        # Push back to image space:
        ret = self.jacobian.adjoint(proxU * w)        # (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

class WeightedTotalVariation(Function):
    """
    GPU/CPU total variation with optional gradient normalization.
    Applies smoothed total variation separately to each modality.
    """

    def __init__(
            self, geometry, weights,
            delta, smoothing='fair', norm='l2',
            gpu=True, anatomical=None, stable=True,
            diagonal=False, both_directions=False,
            hessian='diagonal'):
        
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if hasattr(anatomical, 'as_array'):  # ImageData
            anatomical = anatomical.as_array()

        # Jacobian operator: maps N×M images → N×M×d (stack of finite diffs)
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical,
            gpu=gpu, numpy_out=not gpu,
            method="forward",
            diagonal=diagonal,
            both_directions=both_directions,
        )

        self.smoothing = smoothing
        self.hessian = hessian
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)

        # Pull out the weights as an array/tensor of shape (Nx,Ny,Nz,M)
        self.weights = self.bdc2a.direct(weights)  # shape (..., M)
        self.gpu = gpu

        # Inverse‐weight is used in inv_hessian_diag
        if gpu:
            self.inv_weights = torch.reciprocal(self.weights)
            self.inv_weights = torch.nan_to_num(
                self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0
            )
        else:
            self.inv_weights = np.reciprocal(self.weights, where=self.weights != 0)

        # Create the GPU total variation backend
        from .vector_norm import GPUVectorNorm
        self.tv = GPUVectorNorm(eps=delta, norm=norm, smoothing_function=smoothing)

    def __call__(self, x):
        # 1) Pull x (BDC) → array/tensor of shape (..., M)
        x_arr = self.bdc2a.direct(x)                        # shape (nx, ny, nz, M)
        # 2) Compute the Jacobian: J_raw shape (nx, ny, nz, M, d)
        J = self.jacobian.direct(x_arr)

        # 3) Multiply by weights (shape (..., M) → expand to (..., M, 1))
        w = self.weights.unsqueeze(-1) if self.gpu else np.expand_dims(self.weights, -1)
        U = w * J                                            # (..., M, d)

        # 4) Call GPU TV on U
        return self.tv(U)

    def gradient(self, x, out=None):
        # 1) Pull x → array/tensor
        x_arr = self.bdc2a.direct(x)                        # (nx, ny, nz, M)
        # 2) Compute J_raw (nx, ny, nz, M, d)
        J = self.jacobian.direct(x_arr)

        # 3) Multiply by weights
        w = self.weights.unsqueeze(-1) if self.gpu else np.expand_dims(self.weights, -1)
        U = w * J                                # (..., M, d)

        # 4) Compute inner gradient: shape (..., M, d)
        inner = w * self.tv.gradient(U)        # the tv.gradient already accounts for smoothing etc.

        # 5) Push back to image‐space: J^* (−divergence)
        ret = self.jacobian.adjoint(inner)      # shape (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

    def _preconditioner_weights_core(self, x_arr):
        """
        Core implementation to calculate the diagonal preconditioner weights
        for the total variation case.
        """
        # 1. Compute the Jacobian field
        J = self.jacobian.direct(x_arr)

        # 2. Apply the data-fidelity weights
        w = self.weights.unsqueeze(-1) if self.gpu else np.expand_dims(self.weights, -1)
        A_field = w * J

        # 3. Get the Hessian components from the TV backend
        hess_coeffs, gradient_fields = self.tv.hessian_components(A_field)
        # hess_coeffs shape: (nx, ny, nz, M)
        # gradient_fields shape: (nx, ny, nz, M, d)

        # 4. For diagonal preconditioner in TV case, we approximate as:
        # P_diag[i,m] ≈ h''(||∇_i x_m||) * ||J^T (normalized_gradient)||^2
        
        P_diag = torch.zeros_like(x_arr) if self.gpu else np.zeros_like(x_arr)
        
        # Process each modality separately
        for m in range(x_arr.shape[-1]):
            # Get the normalized gradients for this modality
            grad_field_m = gradient_fields[..., m, :]  # (nx, ny, nz, d)
            
            # Compute the influence in image space
            influence_image_m = self.jacobian.adjoint(w[..., m:m+1] * grad_field_m.unsqueeze(-2))
            # Shape: (nx, ny, nz, 1) -> squeeze to (nx, ny, nz)
            influence_image_m = influence_image_m.squeeze(-1)
            
            # Apply the Hessian coefficient
            P_diag[..., m] = hess_coeffs[..., m] * (influence_image_m ** 2)

        return P_diag

    def hessian_diag(self, x, out=None):
        """
        Computes a diagonal approximation of the Hessian for total variation.
        """
        x_arr = self.bdc2a.direct(x)
        diag_arr = self._preconditioner_weights_core(x_arr)
        
        result = self.bdc2a.adjoint(diag_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        """
        Computes the action of the inverse of the diagonal Hessian approximation.
        """
        # 1. Get the preconditioner weights
        diag_arr = self._preconditioner_weights_core(self.bdc2a.direct(x))
        
        # 2. Invert the weights, adding epsilon for stability
        if self.gpu:
            inv_arr = torch.reciprocal(diag_arr + epsilon)
            torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0, out=inv_arr)
        else:
            inv_arr = np.reciprocal(diag_arr + epsilon)

        # 3. Convert back to BlockDataContainer
        result = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def proximal(self, x, tau, out=None):
        """
        Proximal operator for total variation.
        """
        x_arr = self.bdc2a.direct(x)                  # (nx,ny,nz,M)
        J = self.jacobian.direct(x_arr)           # (nx,ny,nz,M,d)

        w = self.weights.unsqueeze(-1) if self.gpu else np.expand_dims(self.weights, -1)
        U = w * J                                     # (...,M,d)

        proxU = self.tv.proximal(U, tau)             # (...,M,d)

        # Push back to image space:
        ret = self.jacobian.adjoint(proxU * w)        # (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret
    
    def hessian_diag(self, x, out=None):
        # 1) pull to array/tensor
        x_arr = self.bdc2a.direct(x)                 # (nx,ny,nz,M)
        J     = self.jacobian.direct(x_arr)          # (nx,ny,nz,M,d)

        # 2) modality weights & form U_jm,k = w_jm * J_jm,k
        w     = self.weights.unsqueeze(-1)           # (nx,ny,nz,M,1)
        U     = w * J                                # (nx,ny,nz,M,d)

        # 3) sensitivity map S_jm,k = 1/Δℓ_k
        S_np  = self.jacobian.sensitivity(x_arr)     # numpy or torch
        S     = torch.as_tensor(S_np, device=U.device)

        # 4) φ''(r) via vmap inside GPUVectorNorm
        #    flatten voxel+modality dims into batch:
        batch_shape = U.shape[:-1]                   # (...,M)
        U_flat = U.reshape(-1, U.shape[-1])         # (B, d)
        phi2   = self.tv.phi_hessian(U_flat)        # (B,)
        phi2   = phi2.reshape(*batch_shape)         # (...,M)

        # 5) assemble H_jm = φ''(r) * w_jm² * Σ_k S_jm,k²
        S2_sum = torch.sum(S*S, dim=-1)             # (...,M)
        hess_arr = phi2 * (self.weights**2) * S2_sum

        # 6) push back
        result = self.bdc2a.adjoint(hess_arr)
        if out is not None:
            out.fill(result); return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=1e-9):
        # reuse hessian_diag code
        hess_arr = self.hessian_diag(x, out=None)     # BDC or array
        # if it’s a numpy array, convert to torch:
        H = torch.as_tensor(
            hess_arr if isinstance(hess_arr, torch.Tensor) else self.bdc2a.direct(hess_arr),
            device=device
        )
        inv_arr = torch.reciprocal(H + epsilon)
        result  = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result); return out
        return result