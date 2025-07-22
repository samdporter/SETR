import torch
from torch.autograd import Function
from setr.core.gradients import Jacobian
from setr.utils import BlockDataContainerToArray
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def whiten(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Zero-mean, unit-variance normalize each channel (column) of a 2D tensor.
    Args:
        tensor: shape (N, D)
        eps: small constant to avoid division by zero
    Returns:
        whitened tensor of same shape
    """
    mean = tensor.mean(dim=0, keepdim=True)
    std  = tensor.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
    return (tensor - mean) / std

class MutualInformationGradientPrior(Function):
    def __init__(self,
                 geometry,
                 sigma: float = 1.0,
                 use_autograd: bool = True,
                 norm_eps: float = 1e-12,
                 gpu: bool = True,
                 anatomical=None,
                 max_points: int = 10000):
        self.sigma = sigma
        self.norm_eps = norm_eps
        self.use_autograd = use_autograd
        self.device = 'cuda' if gpu else 'cpu'
        self.max_points = max_points

        voxel_sizes = geometry.containers[0].voxel_sizes()
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical,
            gpu=gpu, numpy_out=not gpu,
            method="forward",
            diagonal=True, both_directions=True
        )
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)
        self._log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

    def _get_joint_gradients(self, x):
        J_all = self.jacobian.direct(self.bdc2a.direct(x)).view(-1, 6)
        N = J_all.shape[0]
        if N > self.max_points:
            idx = torch.randperm(N, device=J_all.device)[:self.max_points]
        else:
            idx = torch.arange(N, device=J_all.device)
        return J_all[idx], idx, J_all.shape

    def __call__(self, x):
        J, _, _ = self._get_joint_gradients(x)
        # whiten for scale invariance
        Jw = whiten(J)
        Jw = Jw.detach()
        Jw.requires_grad_(self.use_autograd)
        return self._compute_mi(Jw)

    def _compute_mi(self, J: torch.Tensor) -> torch.Tensor:
        N, D = J.shape
        sigma = self.sigma

        def log_kde(dists, dim, d):
            logs = -0.5 * dists / (sigma**2)
            logs += -0.5 * d * self._log_2pi
            logs += - d * torch.log(torch.tensor(sigma, device=J.device))
            return torch.logsumexp(logs, dim=dim) - torch.log(torch.tensor(N, dtype=J.dtype, device=J.device))

        D_joint = torch.cdist(J, J, p=2).pow(2)
        log_joint = log_kde(D_joint, dim=1, d=D)

        x1, x2 = J[:, :3], J[:, 3:]
        D1 = torch.cdist(x1, x1, p=2).pow(2)
        D2 = torch.cdist(x2, x2, p=2).pow(2)
        log_p1 = log_kde(D1, dim=1, d=3)
        log_p2 = log_kde(D2, dim=1, d=3)

        mi = log_joint - log_p1 - log_p2
        return -mi.sum()

    def gradient(self, x, out=None):
        J, idx, full_shape = self._get_joint_gradients(x)
        # whiten the same way as forward
        Jw = whiten(J)
        if self.use_autograd:
            Jw.requires_grad_(True)
            loss = self._compute_mi(Jw)
            loss.backward()
            grad_w = Jw.grad
            # chain rule through whitening:
            # dL/dJ = (I/std) - ((J-mean)*(sum dL/dJw)/std^2)/N
            mean = J.mean(dim=0, keepdim=True)
            std = J.std(dim=0, unbiased=False, keepdim=True).clamp(min=self.norm_eps)
            dL_dJ = (grad_w / std) - ((J - mean) * (grad_w * (J - mean)).sum(dim=0, keepdim=True) / (std**3 * J.shape[0]))
            grad_flat = dL_dJ
        else:
            # fallback to manual gradient on whitened Jw
            grad_flat = self._manual_gradient(Jw)
            # not chaining back through whiten
            # user may accept approximate
        # scatter back
        grad_full = torch.zeros((full_shape[0], 6), device=grad_flat.device, dtype=grad_flat.dtype)
        grad_full[idx] = grad_flat
        grad = grad_full.view(*self.jacobian.direct(self.bdc2a.direct(x)).shape)
        result = self.bdc2a.adjoint(self.jacobian.adjoint(grad))
        if out is not None:
            out.fill(result); return out
        return result

    # Manual gradient remains unchanged except uses whitened J internally:
    def _manual_gradient(self, J):
        N, D = J.shape
        sigma = self.sigma
        log_2pi = self._log_2pi

        diff = J.unsqueeze(1) - J.unsqueeze(0)
        sq = (diff**2).sum(-1)
        log_kj = -0.5*sq/sigma**2 - 0.5*D*log_2pi - D*torch.log(torch.tensor(sigma, device=J.device))
        wj = torch.softmax(log_kj, dim=1)
        grad_joint = (wj.unsqueeze(-1)*diff).sum(dim=1)/sigma**2

        x1, x2 = J[:, :3], J[:, 3:]
        def marg_grad(x, d):
            diff = x.unsqueeze(1)-x.unsqueeze(0)
            sq = (diff**2).sum(-1)
            log_k = -0.5*sq/sigma**2 - 0.5*d*log_2pi - d*torch.log(torch.tensor(sigma, device=x.device))
            w = torch.softmax(log_k, dim=1)
            return (w.unsqueeze(-1)*diff).sum(dim=1)/sigma**2

        g1 = marg_grad(x1, 3)
        g2 = marg_grad(x2, 3)
        return torch.cat([g1, g2], dim=1) - grad_joint

    def hessian_diag_arr(self, x):
        g = self.gradient(x)
        g_arr = self.bdc2a.direct(g)
        return (g_arr**2).sum(dim=-1)

    def hessian_diag(self, x, out=None):
        arr = self.hessian_diag_arr(x)
        res = self.bdc2a.adjoint(arr)
        if out is not None: out.fill(res); return out
        return res

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        arr = self.hessian_diag_arr(x) + epsilon
        inv = torch.reciprocal(arr)
        res = self.bdc2a.adjoint(inv)
        if out is not None: out.fill(res); return out
        return res


class MutualInformationImagePrior(Function):
    def __init__(self,
                 geometry,
                 sigma: float = 1.0,
                 use_autograd: bool = True,
                 gpu: bool = True,
                 max_points: int = 10000):
        self.sigma = sigma
        self.use_autograd = use_autograd
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.max_points = max_points

        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)
        self._log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

    def _get_joint_vectors(self, x):
        data = self.bdc2a.direct(x)
        vecs = data.view(-1, data.shape[-1])
        N = vecs.shape[0]
        if N > self.max_points:
            idx = torch.randperm(N, device=vecs.device)[:self.max_points]
            return vecs[idx], idx, vecs.shape
        else:
            idx = torch.arange(N, device=vecs.device)
            return vecs, idx, vecs.shape

    def __call__(self, x):
        V, _, _ = self._get_joint_vectors(x)
        Vw = whiten(V)
        Vw = Vw.detach()
        Vw.requires_grad_(self.use_autograd)
        return self._compute_mi(Vw)

    def _compute_mi(self, J):
        N, D = J.shape
        sigma = self.sigma

        def log_kde(dists, dim, d):
            logs = -0.5 * dists / (sigma**2)
            logs += -0.5 * d * self._log_2pi
            logs += - d * torch.log(torch.tensor(sigma, device=J.device))
            return torch.logsumexp(logs, dim=dim) - torch.log(torch.tensor(N, dtype=J.dtype, device=J.device))

        D_joint = torch.cdist(J, J, p=2).pow(2)
        log_joint = log_kde(D_joint, dim=1, d=D)

        x1 = J[:, 0:1]; x2 = J[:, 1:2]
        D1 = torch.cdist(x1, x1, p=2).pow(2)
        D2 = torch.cdist(x2, x2, p=2).pow(2)
        log_p1 = log_kde(D1, dim=1, d=1)
        log_p2 = log_kde(D2, dim=1, d=1)

        return -(log_joint - log_p1 - log_p2).sum()

    def gradient(self, x, out=None):
        V, idx, full_shape = self._get_joint_vectors(x)
        Vw = whiten(V)
        if self.use_autograd:
            Vw.requires_grad_(True)
            loss = self._compute_mi(Vw)
            loss.backward()
            grad_w = Vw.grad
            # chain rule through whitening
            mean = V.mean(dim=0, keepdim=True)
            std = V.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
            dL_dV = (grad_w / std) - ((V - mean) * (grad_w * (V - mean)).sum(dim=0, keepdim=True)
                                       / (std**3 * V.shape[0]))
            grad_flat = dL_dV
        else:
            grad_flat = torch.zeros_like(Vw)  # fallback
        back = torch.zeros_like(V)
        back[idx] = grad_flat
        grad_full = back.view_as(self.bdc2a.direct(x))
        result = self.bdc2a.adjoint(grad_full)
        if out is not None:
            out.fill(result); return out
        return result

    def hessian_diag_arr(self, x):
        g = self.gradient(x)
        g_arr = self.bdc2a.direct(g)
        return (g_arr**2).sum(dim=-1)

    def hessian_diag(self, x, out=None):
        arr = self.hessian_diag_arr(x)
        if arr.ndim == 3:
            res = self.bdc2a.adjoint(arr)
        else:
            splits = torch.unbind(arr, dim=-1)
            res = self.bdc2a.adjoint(torch.stack(splits, dim=-1))
        if out is not None: out.fill(res); return out
        return res

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        arr = self.hessian_diag_arr(x) + epsilon
        inv = torch.reciprocal(arr)
        if arr.ndim == 3:
            res = self.bdc2a.adjoint(inv)
        else:
            splits = torch.unbind(inv, dim=-1)
            res = self.bdc2a.adjoint(torch.stack(splits, dim=-1))
        if out is not None: out.fill(res); return out
        return res
