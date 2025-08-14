import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor(x, *, like_dtype=torch.float32, device=device):
    """Cheap, no-copy when possible; keeps everything on `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=like_dtype, copy=False)
    return torch.as_tensor(x, device=device, dtype=like_dtype)

def _to_numpy(x):
    """Safe PyTorch → NumPy: detach, move to CPU, then numpy()."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Jacobian:
    def __init__(
        self,
        voxel_sizes: tuple | list = (1.0, 1.0, 1.0),
        bnd_cond="Neumann",
        anatomical=None,          # None → plain Gradient3D; tensor/ndarray → DirectionalGradient
        stencil="6",              # '6' | '18' | '26'
        both_directions=False,    # False → 3/9/13 channels; True → 6/18/26
        normalize=True,           # must match the gradient operator
        numpy_out=False
    ):
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes)
        self.stencil = stencil
        self.both_directions = both_directions
        self.normalize = normalize
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out

        def _init_grad(anat):
            if anat is None:
                return Gradient(
                    voxel_sizes=self.voxel_sizes,
                    stencil=self.stencil,
                    bnd_cond=self.bnd_cond,
                    both_directions=self.both_directions,
                    normalize=self.normalize,
                )
            anat_t = torch.as_tensor(anat, device=device, dtype=torch.float32)
            return DirectionalGradient(
                anatomical=anat_t,
                voxel_sizes=self.voxel_sizes,
                stencil=self.stencil,
                bnd_cond=self.bnd_cond,
                both_directions=self.both_directions,
                normalize=self.normalize,
            )

        if isinstance(anatomical, list):
            self.grad = [_init_grad(a) for a in anatomical]
            self.multi_anatomy = True
            self.n_params_expected = len(self.grad)
        else:
            self.grad = _init_grad(anatomical)
            self.multi_anatomy = False
            self.n_params_expected = None

    def direct(self, images):
        X = _to_tensor(images, like_dtype=torch.float32, device=device)
        n_params = X.shape[-1]
        if self.multi_anatomy and (n_params != self.n_params_expected):
            raise ValueError(f"Expected {self.n_params_expected} parameter images, got {n_params}.")

        outs = []
        if isinstance(self.grad, list):
            outs.extend(self.grad[i].direct(X[..., i]) for i in range(n_params))
        else:
            outs.extend(self.grad.direct(X[..., i]) for i in range(n_params))

        Y = torch.stack(outs, dim=-2)  # always a Tensor internally
        return _to_numpy(Y) if self.numpy_out else Y

    def adjoint(self, jacobians):
        Y = _to_tensor(jacobians, like_dtype=torch.float32, device=device)
        n_params = Y.shape[-2]
        if self.multi_anatomy and (n_params != self.n_params_expected):
            raise ValueError(f"Expected {self.n_params_expected} parameter fields, got {n_params}.")

        outs = []
        if isinstance(self.grad, list):
            outs.extend(self.grad[i].adjoint(Y[..., i, :]) for i in range(n_params))
        else:
            outs.extend(self.grad.adjoint(Y[..., i, :]) for i in range(n_params))

        out = torch.stack(outs, dim=-1)
        return _to_numpy(out) if self.numpy_out else out

    def sensitivity(self, images):
        grads = self.grad if isinstance(self.grad, list) else [self.grad]
        proto = grads[0]
        n_dirs = len(proto.directions)

        # per-direction scale from physical step (already correct)
        if self.normalize:
            scale_per_dir = (1.0 / proto._step)
        else:
            scale_per_dir = torch.ones(n_dirs, device=device, dtype=torch.float32)

        bank = getattr(proto, "_bank_scale", 1.0)  # fallback if not present
        scale_per_dir = scale_per_dir / bank

        X = _to_tensor(images, like_dtype=torch.float32, device=device)
        leading = X.shape[:-1]
        n_params = X.shape[-1]
        S = scale_per_dir.view(*((1,) * len(leading)), 1, n_dirs).expand(*leading, n_params, n_dirs)
        return _to_numpy(S) if self.numpy_out else S

    def calculate_norm(self):
        grads = self.grad if isinstance(self.grad, list) else [self.grad]
        proto = grads[0]
        bank = getattr(proto, "_bank_scale", 1.0)
        if self.normalize:
            n_dirs = len(proto.directions)
            # include bank scaling so L is ~ invariant across 6/18/26 stencils
            return 2.0 * np.sqrt(n_dirs) / float(bank)
        step = proto._step.detach().cpu().numpy()
        # if not normalising by step: include bank in the norm as well
        return float((np.sqrt(np.sum((2.0 / step) ** 2))) / float(bank))



def _shift_neumann_3d(x: torch.Tensor, sh: tuple[int,int,int]) -> torch.Tensor:
    dx, dy, dz = map(int, sh)
    X, Y, Z = x.shape
    x5 = x.unsqueeze(0).unsqueeze(0)
    xpad = F.pad(x5, (1,1,1,1,1,1), mode="replicate")
    out = xpad[:, :, 1+dx:1+dx+X, 1+dy:1+dy+Y, 1+dz:1+dz+Z]
    return out.squeeze(0).squeeze(0)

def _in_halfspace(dx: int, dy: int, dz: int) -> bool:
    if dx != 0: return dx > 0
    return dy > 0 if dy != 0 else dz > 0

def _stencil_halfspace(stencil: str):
    allowed = { "6": {1}, "18": {1,2}, "26": {1,2,3} }[stencil]
    dirs = []
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                if dx==dy==dz==0: continue
                l1 = abs(dx)+abs(dy)+abs(dz)
                if l1 in allowed and _in_halfspace(dx,dy,dz):
                    dirs.append((dx,dy,dz))
    return dirs

class Gradient:
    def __init__(
        self,
        voxel_sizes,
        stencil: str = "6",
        bnd_cond: str = "Neumann",
        both_directions: bool = False,
        normalize: bool = True,
        numpy_out: bool = False
    ):
        self.vx, self.vy, self.vz = map(float, voxel_sizes)
        self.bnd_cond = bnd_cond
        self.both_directions = both_directions
        self.normalize = normalize
        self.numpy_out = numpy_out

        half = _stencil_halfspace(stencil)
        self.directions = half + [(-dx,-dy,-dz) for dx,dy,dz in half] if both_directions else half

        steps = []
        for dx,dy,dz in self.directions:
            sx, sy, sz = abs(dx)*self.vx, abs(dy)*self.vy, abs(dz)*self.vz
            steps.append(np.sqrt(sx*sx + sy*sy + sz*sz))
        self._step = torch.tensor(steps, device=device, dtype=torch.float32)
        self._bank_scale = np.sqrt(len(self.directions)/3.0)

    def _shift(self, x: torch.Tensor, sh: tuple[int,int,int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            return torch.roll(x, shifts=sh, dims=(0,1,2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_3d(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def _diff_fwd(self, x: torch.Tensor, sh: tuple[int,int,int]) -> torch.Tensor:
        return self._shift(x, sh) - x

    def _diff_bwd(self, x: torch.Tensor, sh: tuple[int,int,int]) -> torch.Tensor:
        inv = (-sh[0], -sh[1], -sh[2])
        return x - self._shift(x, inv)

    def direct(self, x):
        X = _to_tensor(x, like_dtype=torch.float32, device=device)
        outs = [self._diff_fwd(X, sh) for sh in self.directions]
        Y = torch.stack(outs, dim=-1)
        if self.normalize:
            Y = Y / self._step.view(*(1,)*(Y.ndim-1), -1)
        Y = Y / self._bank_scale
        return _to_numpy(Y) if self.numpy_out else Y

    def adjoint(self, x):
        Y = _to_tensor(x, like_dtype=torch.float32, device=device)
        if self.normalize:
            Y = Y / self._step.view(*(1,)*(Y.ndim-1), -1)
        out = torch.zeros_like(Y[..., 0])
        for ch, sh in enumerate(self.directions):
            out -= self._diff_bwd(Y[..., ch], sh)
        out = out / self._bank_scale
        return _to_numpy(out) if self.numpy_out else out

    def calculate_norm(self):
        # include scaling so L is ~constant across stencils (≈ 2*sqrt(3))
        return 2.0 * np.sqrt(len(self.directions)) / self._bank_scale


class Sum:
    """
    Edge-sum operator S with the same stencil and boundary rules as Gradient.

    For each direction sh in `directions`:
        (S x)[..., ch] = x + shift(x, sh)

    Adjoint S^T distributes edge-channel values to both endpoints:
        (S^T y) = Σ_ch [ y_ch + shift(y_ch, sh) ].
    """

    def __init__(
        self,
        voxel_sizes,
        stencil: str = "6",
        bnd_cond: str = "Neumann",
        both_directions: bool = False,
        numpy_out: bool = False
    ):
        # voxel_sizes kept for parity/API symmetry (not used for scaling)
        self.bnd_cond = bnd_cond
        self.both_directions = both_directions
        self.numpy_out = numpy_out

        half = _stencil_halfspace(stencil)
        self.directions = half + [(-dx,-dy,-dz) for dx,dy,dz in half] if both_directions else half

    def _shift(self, x: torch.Tensor, sh: tuple[int,int,int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            return torch.roll(x, shifts=sh, dims=(0,1,2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_3d(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def direct(self, x):
        X = _to_tensor(x, like_dtype=torch.float32, device=device)
        outs = [X + self._shift(X, sh) for sh in self.directions]
        out = torch.stack(outs, dim=-1)
        return _to_numpy(out) if self.numpy_out else out

    def adjoint(self, x):
        Y = _to_tensor(x, like_dtype=torch.float32, device=device)
        out = torch.zeros_like(Y[..., 0])
        for ch, sh in enumerate(self.directions):
            edge = Y[..., ch]
            out = out + edge
            out = out + self._shift(edge, sh)
        return _to_numpy(out) if self.numpy_out else out

    def calculate_norm(self):
        # Operator norm upper bound (each channel sums two samples):
        # ||S|| ≤ 2 * sqrt(#channels). Often sufficient for step-size tuning.
        return 2.0 * np.sqrt(len(self.directions))


class DirectionalGradient:
    def __init__(
        self,
        anatomical,
        voxel_sizes,
        gamma=1,
        eta=None,
        bnd_cond="Neumann",
        both_directions=False,
        stencil="6",
        normalize=True,
        numpy_out=False
    ):
        self.anatomical = anatomical
        self.voxel_size = voxel_sizes
        self.gamma = gamma
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out
        self.gradient = Gradient(
            voxel_sizes=self.voxel_size,
            stencil=stencil,
            bnd_cond=self.bnd_cond,
            both_directions=both_directions,
            normalize=normalize
        )
        self.anatomical_grad = self.gradient.direct(self.anatomical)
        if eta is None:
            max_val = self.anatomical_grad.max().item()
            min_val = self.anatomical_grad.min().item()
            self.eta = (max_val - min_val) / 100000
        else:
            self.eta = eta

        self.directional_op = gpu_directional_op
        self.eta = torch.tensor(self.eta, device=device)
        self.gamma = torch.tensor(self.gamma, device=device)
        if not isinstance(self.anatomical_grad, torch.Tensor):
            self.anatomical_grad = torch.tensor(self.anatomical_grad, device=device)
        else:
            self.anatomical_grad = self.anatomical_grad.to(device)

    def direct(self, x):
        X = _to_tensor(x, like_dtype=torch.float32, device=device)
        gradient = self.gradient.direct(X)
        out = self.directional_op(gradient, self.anatomical_grad, self.gamma, self.eta)
        return _to_numpy(out) if self.numpy_out else out

    def adjoint(self, x):
        X = _to_tensor(x, like_dtype=torch.float32, device=device)
        x = self.directional_op(X, self.anatomical_grad, self.gamma, self.eta)
        out = self.gradient.adjoint(x)
        return _to_numpy(out) if self.numpy_out else out


def gpu_directional_op(image_gradient, anatomical_gradient, gamma=1, eta=1e-6):
        den = torch.norm(anatomical_gradient, p=2, dim=-1, keepdim=True)
        xi  = anatomical_gradient / torch.sqrt(den**2 + eta**2)  # or (den + eta)
        return image_gradient - gamma * torch.sum(image_gradient * xi, dim=-1, keepdim=True) * xi


def check_adjoint(
    op, shape, *, input_is_vector=False, n_params=1, trials=3,
    seed=0, dtype=torch.float32, device=device
):
    """
    Checks <op x, y> = <x, op^* y> with random x,y.
    shape: (X, Y, Z) spatial size
    input_is_vector: set True for Jacobian-like operators that expect a last param-dim
    n_params: number of parameter images when input_is_vector=True
    Returns: max relative mismatch over `trials`
    """
    g = torch.Generator(device=device).manual_seed(seed)
    max_rel = 0.0
    for _ in range(trials):
        xshape = (*shape, n_params) if input_is_vector else shape
        x = torch.randn(xshape, generator=g, device=device, dtype=dtype)
        Y = op.direct(x)
        if not isinstance(Y, torch.Tensor):
            Y = torch.as_tensor(Y, device=device)
        y = torch.empty_like(Y).normal_(generator=g)
        lhs = torch.sum(Y * y)                        # <Gx, y>
        Ya = op.adjoint(y)
        if not isinstance(Ya, torch.Tensor):
            Ya = torch.as_tensor(Ya, device=device)
        rhs = torch.sum(x * Ya)                        # <x, G^T y>
        num = (lhs - rhs).abs().item()
        den = max(1.0, float(max(lhs.abs().item(), rhs.abs().item())))
        max_rel = max(max_rel, num / den)
    return max_rel
