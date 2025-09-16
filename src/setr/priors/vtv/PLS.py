# PLS.py
#
# Symmetric synergistic Parallel Level Sets (PLS) prior with customisable φ and ψ.
# Definition (per voxel, for each modality pair i<j):
#   S_ij = φ( ψ(‖∇x_i‖ · ‖∇x_j‖) − ψ(|⟨∇x_i, ∇x_j⟩|) )
# We implement the weighted variant with U_i = w_i ∇x_i. For M>2, sum over all i<j.
#
# Valid choices:
#   φ, ψ ∈ { id(s)=s, square(s)=s**2, sqrt(s)=√s, log(s)=log s, exp(s)=e^s }
# ψ is always applied to nonnegative inputs (product of norms and |dot|).
# φ sees a real input; for sqrt/log we safely rectify: φ(z)=f(max(z,0)+ε).
#
# API:
#   - __call__(x)    -> scalar value
#   - gradient(x)    -> BlockDataContainer shaped like x
#   - proximal(...)  -> NotImplementedError (non-separable)
#
import numpy as np
from cil.optimisation.functions import Function

from setr.core.gradients import Jacobian
from setr.utils import BlockDataContainerToArray
from setr.utils.sirf import get_array

try:
    import torch

    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = "cpu"


# ----------------------- scalar φ/ψ factories -----------------------
def _scalar_fun(mode: str, eps: float, gpu: bool, allow_negative_input: bool):
    """
    Return (val, dval) functions mapping tensor/ndarray -> same shape,
    with numerically safe definitions.

    mode ∈ {'id','square','sqrt','log','exp'}
    eps  : small positive constant
    allow_negative_input:
        - True  for φ (argument z can be negative) -> use relu for sqrt/log
        - False for ψ (argument ≥0 by construction)
    """
    if gpu:
        xp = torch

        def _relu(x):
            return xp.clamp(x, min=0.0)

        def _clip_exp(x):
            return xp.clamp(x, min=-60.0, max=60.0)  # avoids overflow

        if mode == "id":

            def val(x):
                return x

            def der(x):
                return xp.ones_like(x)

            return val, der

        if mode == "square":

            def val(x):
                return x * x

            def der(x):
                return 2.0 * x

            return val, der

        if mode == "sqrt":

            def val(x):
                y = _relu(x) if allow_negative_input else x
                return xp.sqrt(y + eps)

            def der(x):
                y = _relu(x) if allow_negative_input else x
                return 0.5 / xp.sqrt(y + eps)

            return val, der

        if mode == "log":

            def val(x):
                y = _relu(x) if allow_negative_input else x
                return xp.log(y + eps)

            def der(x):
                y = _relu(x) if allow_negative_input else x
                return 1.0 / (y + eps)

            return val, der

        if mode == "exp":

            def val(x):
                return xp.exp(_clip_exp(x))

            def der(x):
                return xp.exp(_clip_exp(x))

            return val, der

    else:
        xp = np

        def _relu(x):
            return xp.maximum(x, 0.0)

        def _clip_exp(x):
            return xp.clip(x, -60.0, 60.0)

        if mode == "id":

            def val(x):
                return x

            def der(x):
                return xp.ones_like(x)

            return val, der

        if mode == "square":

            def val(x):
                return x * x

            def der(x):
                return 2.0 * x

            return val, der

        if mode == "sqrt":

            def val(x):
                y = _relu(x) if allow_negative_input else x
                return xp.sqrt(y + eps)

            def der(x):
                y = _relu(x) if allow_negative_input else x
                return 0.5 / xp.sqrt(y + eps)

            return val, der

        if mode == "log":

            def val(x):
                y = _relu(x) if allow_negative_input else x
                return xp.log(y + eps)

            def der(x):
                y = _relu(x) if allow_negative_input else x
                return 1.0 / (y + eps)

            return val, der

        if mode == "exp":

            def val(x):
                return xp.exp(_clip_exp(x))

            def der(x):
                return xp.exp(_clip_exp(x))

            return val, der

    raise ValueError(f"Unknown mode {mode!r} for scalar function.")


# ----------------------- PLS prior -----------------------
class WeightedParallelLevelSets(Function):
    """
    Symmetric synergistic PLS prior with weights and GPU/CPU backends.

    Energy:
    S(x) = sum_{i<j} C_{ij} φ( ψ(‖U_i‖‖U_j‖) − ψ(|⟨U_i, U_j⟩|) ),  U_i = w_i ∇x_i

    Parameters
    ----------
    geometry : CIL BlockGeometry (M modalities)
    weights  : BlockDataContainer, per-modality weights, broadcast along spatial dims
    gpu      : bool (torch if True else numpy)
    anatomical : optional array for Jacobian (as in your VTV)
    diagonal, both_directions : finite-difference options for Jacobian
    coupling : (M×M) symmetric, zero diagonal; default ones off-diagonal
    phi_mode : {'id','square','sqrt','log','exp'}    (applied to z)
    psi_mode : {'id','square','sqrt','log','exp'}    (applied to nonnegative inputs)
    phi_eps  : small ε for φ (used in sqrt/log safety)
    psi_eps  : small ε for ψ (used in sqrt/log safety)
    """

    def __init__(
        self,
        geometry,
        weights,
        gpu=True,
        anatomical=None,
        diagonal=False,
        both_directions=False,
        coupling=None,
        phi_mode="id",
        psi_mode="id",
        phi_eps=1e-12,
        psi_eps=1e-12,
    ):
        self.gpu = bool(gpu and _HAS_TORCH)
        self.phi_eps = float(phi_eps)
        self.psi_eps = float(psi_eps)

        voxel_sizes = geometry.containers[0].voxel_sizes()
        if hasattr(anatomical, "as_array"):
            anatomical = get_array(anatomical)

        self.jacobian = Jacobian(
            voxel_sizes,
            anatomical=anatomical,
            gpu=self.gpu,
            numpy_out=not self.gpu,
            method="forward",
            diagonal=diagonal,
            both_directions=both_directions,
        )
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=self.gpu)

        # weights (..., M)
        self.weights = self.bdc2a.direct(weights)

        # modalities count
        self.M = int(self.weights.shape[-1])

        # coupling
        if coupling is None:
            C = np.ones((self.M, self.M), dtype=np.float32) - np.eye(self.M, dtype=np.float32)
        else:
            C = np.asarray(coupling, dtype=np.float32)
            if C.shape != (self.M, self.M):
                raise ValueError(f"coupling must be ({self.M},{self.M}), got {C.shape}")
            if not np.allclose(C, C.T, atol=1e-8):
                raise ValueError("coupling must be symmetric")
            np.fill_diagonal(C, 0.0)
        self.C = torch.as_tensor(C, device=_DEVICE) if self.gpu else C

        # upper/lower masks (zero diag)
        if self.gpu:
            ones = torch.ones((self.M, self.M), device=_DEVICE)
            self.UTR = torch.triu(ones, diagonal=1)
            self.LTR = torch.tril(ones, diagonal=-1)
        else:
            ones = np.ones((self.M, self.M), dtype=np.float32)
            self.UTR = np.triu(ones, k=1)
            self.LTR = np.tril(ones, k=-1)

        # scalar maps
        self.phi_val, self.phi_der = _scalar_fun(
            phi_mode, self.phi_eps, self.gpu, allow_negative_input=True
        )
        self.psi_val, self.psi_der = _scalar_fun(
            psi_mode, self.psi_eps, self.gpu, allow_negative_input=False
        )

        self._tiny = 1e-12

    # -------- internal helpers --------
    def _einsum(self, subs, *ops):
        return torch.einsum(subs, *ops) if self.gpu else np.einsum(subs, *ops)

    def _prepare_U(self, x):
        """
        Returns
        -------
        U : (..., M, d)  with U_m = w_m ∇x_m
        p : (..., M)     with p_m = ‖U_m‖_2
        C : (..., M, M)  inner products C_{mn} = <U_m, U_n>
        """
        x_arr = self.bdc2a.direct(x)  # (..., M)
        J = self.jacobian.direct(x_arr)  # (..., M, d)
        w = self.weights.unsqueeze(-1) if self.gpu else self.weights[..., None]
        U = w * J  # (..., M, d)

        if self.gpu:
            p = torch.linalg.norm(U, dim=-1).clamp_min(self._tiny)  # (..., M)
            C = torch.einsum("...md,...nd->...mn", U, U)  # (..., M, M)
        else:
            p = np.linalg.norm(U, axis=-1)
            p = np.maximum(p, self._tiny)
            C = np.einsum("...md,...nd->...mn", U, U)
        return U, p, C

    # -------- energy --------
    def __call__(self, x):
        U, p, C = self._prepare_U(x)
        # Pairwise quantities
        pp = self._einsum("...m,...n->...mn", p, p)  # (..., M, M)  products of norms
        abs_dot = torch.abs(C) if self.gpu else np.abs(C)  # (..., M, M)
        # z_ij = ψ(pp_ij) − ψ(|C_ij|)
        z = self.psi_val(pp) - self.psi_val(abs_dot)  # (..., M, M)
        # Only sum i<j
        pair_val = self.phi_val(z) * (self.C[None, ...])  # (..., M, M)
        return (pair_val * self.UTR).sum()

    # -------- gradient --------
    def gradient(self, x, out=None):
        """
        For each pair (i<j):
            z = ψ(p_i p_j) − ψ(|c_ij|)
            ∂/∂U_i: φ'(z) [ ψ'(p_i p_j) p_j * U_i/p_i  −  ψ'(|c_ij|) sign(c_ij) U_j ]
            ∂/∂U_j: φ'(z) [ ψ'(p_i p_j) p_i * U_j/p_j  −  ψ'(|c_ij|) sign(c_ij) U_i ]
        Sum contributions over all pairs.
        """
        U, p, C = self._prepare_U(x)
        pp = self._einsum("...m,...n->...mn", p, p)  # (..., M, M)
        abs_dot = torch.abs(C) if self.gpu else np.abs(C)  # (..., M, M)
        sign_dot = torch.sign(C) if self.gpu else np.sign(C)  # (..., M, M)

        psi_pp = self.psi_val(pp)  # (..., M, M)
        psi_absC = self.psi_val(abs_dot)  # (..., M, M)
        z = psi_pp - psi_absC  # (..., M, M)

        phi_p = self.phi_der(z)  # φ'(z)        (..., M, M)
        dpsi_pp = self.psi_der(pp)  # ψ'(pp)       (..., M, M)
        dpsi_absC = self.psi_der(abs_dot)  # ψ'(|C|)      (..., M, M)

        # a_base and b_base as in analysis
        a_base = (phi_p * dpsi_pp) * (self.C[None, ...])  # (..., M, M)
        b_base = (phi_p * dpsi_absC) * (self.C[None, ...])  # (..., M, M)

        # Symmetrise over i<j: b_sym_{mn} = {b_{mn} if m<n; b_{nm} if m>n; 0 on diag}
        if self.gpu:
            a_sym = a_base * self.UTR + a_base.transpose(-1, -2) * self.LTR
            b_sym = b_base * self.UTR + b_base.transpose(-1, -2) * self.LTR
        else:
            a_sym = a_base * self.UTR + np.swapaxes(a_base, -1, -2) * self.LTR
            b_sym = b_base * self.UTR + np.swapaxes(b_base, -1, -2) * self.LTR

        # --- Term 1: unit directions times Σ_n a_sym_{mn} * p_n
        unitU = U / (p.unsqueeze(-1) if self.gpu else p[..., None])  # (..., M, d)
        s1_coeff = self._einsum("...mn,...n->...m", a_sym, p)  # (..., M)
        term1 = unitU * (s1_coeff[..., None])  # (..., M, d)

        # --- Term 2: alignment cross-talk: Σ_n b_sym_{mn} * sign(C_{mn}) * U_n
        coef = b_sym * sign_dot  # (..., M, M)
        term2 = self._einsum("...mn,...nd->...md", coef, U)  # (..., M, d)

        # Gradient in U-space
        G = term1 - term2  # (..., M, d)

        # Chain back: J^T (w * G)
        w = self.weights.unsqueeze(-1) if self.gpu else self.weights[..., None]
        inner = w * G  # (..., M, d)
        img_grad = self.jacobian.adjoint(inner)  # (..., M)
        img_grad = self.bdc2a.adjoint(img_grad)

        if out is not None:
            out.fill(img_grad)
            return out
        return img_grad

    def proximal(self, x, tau, out=None):
        raise NotImplementedError("PLS proximal is non-separable; use a first-order/PDHG scheme.")
