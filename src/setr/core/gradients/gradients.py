import numpy as np
import torch

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
        anatomical=None,  # None → plain Gradient3D; tensor/ndarray → DirectionalGradient
        stencil="6",  # '6' | '18' | '26'
        both_directions=False,  # False → 3/9/13 channels; True → 6/18/26
        max_step: int = 1,
        normalize=True,  # must match the gradient operator
        numpy_out=False,
    ):
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes)
        self.stencil = stencil
        self.both_directions = both_directions
        self.max_step = int(max_step)
        self.normalize = normalize
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1.")

        def _init_grad(anat):
            if anat is None:
                return Gradient(
                    voxel_sizes=self.voxel_sizes,
                    stencil=self.stencil,
                    bnd_cond=self.bnd_cond,
                    both_directions=self.both_directions,
                    max_step=self.max_step,
                    normalize=self.normalize,
                )
            anat_t = torch.as_tensor(anat, device=device, dtype=torch.float32)
            return DirectionalGradient(
                anatomical=anat_t,
                voxel_sizes=self.voxel_sizes,
                stencil=self.stencil,
                bnd_cond=self.bnd_cond,
                both_directions=self.both_directions,
                max_step=self.max_step,
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

        # Pre-allocate to avoid stacking
        if isinstance(self.grad, list):
            first_grad = self.grad[0].direct(X[..., 0])
        else:
            first_grad = self.grad.direct(X[..., 0])

        # Ensure it's a tensor
        if not isinstance(first_grad, torch.Tensor):
            first_grad = torch.as_tensor(first_grad, device=device, dtype=torch.float32)

        n_dirs = first_grad.shape[-1]
        Y = torch.empty(
            (*X.shape[:-1], n_params, n_dirs), dtype=first_grad.dtype, device=first_grad.device
        )
        Y[..., 0, :] = first_grad

        # Fill remaining parameters
        for i in range(1, n_params):
            if isinstance(self.grad, list):
                grad_result = self.grad[i].direct(X[..., i])
            else:
                grad_result = self.grad.direct(X[..., i])

            if not isinstance(grad_result, torch.Tensor):
                grad_result = torch.as_tensor(grad_result, device=device, dtype=torch.float32)
            Y[..., i, :] = grad_result

        return _to_numpy(Y) if self.numpy_out else Y

    def adjoint(self, jacobians):
        Y = _to_tensor(jacobians, like_dtype=torch.float32, device=device)
        n_params = Y.shape[-2]
        if self.multi_anatomy and (n_params != self.n_params_expected):
            raise ValueError(f"Expected {self.n_params_expected} parameter fields, got {n_params}.")

        # Pre-allocate to avoid stacking
        if isinstance(self.grad, list):
            first_adj = self.grad[0].adjoint(Y[..., 0, :])
        else:
            first_adj = self.grad.adjoint(Y[..., 0, :])

        # Ensure it's a tensor
        if not isinstance(first_adj, torch.Tensor):
            first_adj = torch.as_tensor(first_adj, device=device, dtype=torch.float32)

        out = torch.empty((*Y.shape[:-2], n_params), dtype=first_adj.dtype, device=first_adj.device)
        out[..., 0] = first_adj

        # Fill remaining parameters
        for i in range(1, n_params):
            if isinstance(self.grad, list):
                adj_result = self.grad[i].adjoint(Y[..., i, :])
            else:
                adj_result = self.grad.adjoint(Y[..., i, :])

            if not isinstance(adj_result, torch.Tensor):
                adj_result = torch.as_tensor(adj_result, device=device, dtype=torch.float32)
            out[..., i] = adj_result

        return _to_numpy(out) if self.numpy_out else out

    def sensitivity(self, images):
        """
        Returns per-parameter, per-stencil-channel scale factors S so that
        diag(G^T G) ~ sum_k S[..., k]^2, matching the channels produced by .direct().
        Works for both Gradient and DirectionalGradient backends.
        """
        # Pick a prototype gradient operator
        grads = self.grad if isinstance(self.grad, list) else [self.grad]
        proto = grads[0]

        # Reach the underlying finite-difference engine
        g = getattr(proto, "gradient", proto)  # DirectionalGradient -> inner Gradient

        # Per-channel physical scaling (already includes both_directions & stencil)
        step = g._step  # shape (d,)
        bank = getattr(g, "_bank_scale", 1.0)

        per = 1.0 / step if self.normalize else torch.ones_like(step)
        per = per / bank  # make L ~ invariant across 6/18/26

        # Broadcast to (..., n_params, d)
        X = _to_tensor(images, like_dtype=torch.float32, device=device)
        leading, n_params = X.shape[:-1], X.shape[-1]
        S = per.view(*((1,) * len(leading)), 1, -1).expand(*leading, n_params, per.numel())

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


def _shift_neumann_symmetric(x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
    """Shift using zero-flux (symmetric) Neumann boundaries for arbitrary |Δ|."""
    deltas = tuple(int(v) for v in sh)

    result = x
    for dim, delta in enumerate(deltas):
        if delta == 0:
            continue
        size = result.shape[dim]
        if size == 0:
            continue
        idx = torch.arange(size, device=result.device, dtype=torch.long)
        if delta > 0:
            src = torch.clamp(idx + delta, max=size - 1)
        else:
            src = torch.clamp(idx + delta, min=0)
        result = torch.index_select(result, dim, src)
    return result


def _shift_neumann_adjoint(x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
    """Adjoint of `_shift_neumann_symmetric` for zero-flux boundaries."""
    deltas = tuple(int(v) for v in sh)

    result = x
    for dim, delta in reversed(list(enumerate(deltas))):
        if delta == 0:
            continue
        size = result.shape[dim]
        if size == 0:
            continue
        idx = torch.arange(size, device=result.device, dtype=torch.long)
        if delta > 0:
            src = torch.clamp(idx + delta, max=size - 1)
        else:
            src = torch.clamp(idx + delta, min=0)
        tmp = torch.zeros_like(result)
        tmp.index_add_(dim, src, result)
        result = tmp
    return result


def _in_halfspace(dx: int, dy: int, dz: int) -> bool:
    if dx != 0:
        return dx > 0
    return dy > 0 if dy != 0 else dz > 0


def _stencil_halfspace(stencil: str):
    allowed = {"6": {1}, "18": {1, 2}, "26": {1, 2, 3}}[stencil]
    dirs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                l1 = abs(dx) + abs(dy) + abs(dz)
                if l1 in allowed and _in_halfspace(dx, dy, dz):
                    dirs.append((dx, dy, dz))
    return dirs


class _GradientLegacy:
    def __init__(
        self,
        voxel_sizes,
        stencil: str = "6",
        bnd_cond: str = "Neumann",
        both_directions: bool = False,
        max_step: int = 1,
        normalize: bool = True,
        numpy_out: bool = False,
    ):
        self.vx, self.vy, self.vz = map(float, voxel_sizes)
        self.bnd_cond = bnd_cond
        self.both_directions = both_directions
        self.max_step = int(max_step)
        self.normalize = normalize
        self.numpy_out = numpy_out
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1.")

        half = _stencil_halfspace(stencil)
        base_dirs = half + [(-dx, -dy, -dz) for dx, dy, dz in half] if both_directions else half
        self.directions = [
            (dx * step, dy * step, dz * step)
            for dx, dy, dz in base_dirs
            for step in range(1, self.max_step + 1)
        ]

        steps = []
        for dx, dy, dz in self.directions:
            sx, sy, sz = abs(dx) * self.vx, abs(dy) * self.vy, abs(dz) * self.vz
            steps.append(np.sqrt(sx * sx + sy * sy + sz * sz))
        self._step = torch.tensor(steps, device=device, dtype=torch.float32)
        self._bank_scale = np.sqrt(len(self.directions) / 3.0)

    def _shift(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            return torch.roll(x, shifts=sh, dims=(0, 1, 2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_symmetric(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def _shift_adjoint(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            inv = (-sh[0], -sh[1], -sh[2])
            return torch.roll(x, shifts=inv, dims=(0, 1, 2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_adjoint(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def _diff_fwd(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        return self._shift(x, sh) - x

    def _diff_bwd(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        inv = (-sh[0], -sh[1], -sh[2])
        return x - self._shift(x, inv)

    def direct(self, x):
        X = _to_tensor(x, like_dtype=torch.float32, device=device)

        outs = [self._diff_fwd(X, sh) for sh in self.directions]
        Y = torch.stack(outs, dim=-1)

        if self.normalize:
            Y = Y / self._step.view(*(1,) * (Y.ndim - 1), -1)
        Y = Y / self._bank_scale
        return _to_numpy(Y) if self.numpy_out else Y

    def adjoint(self, x):
        Y = _to_tensor(x, like_dtype=torch.float32, device=device)
        if self.normalize:
            Y = Y / self._step.view(*(1,) * (Y.ndim - 1), -1)

        out = torch.zeros_like(Y[..., 0])

        if self.bnd_cond == "Neumann":
            for ch, sh in enumerate(self.directions):
                y_ch = Y[..., ch]
                out = out + self._shift_adjoint(y_ch, sh)
                out = out - y_ch
        else:
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
        max_step: int = 1,
        numpy_out: bool = False,
    ):
        # voxel_sizes kept for parity/API symmetry (not used for scaling)
        self.bnd_cond = bnd_cond
        self.both_directions = both_directions
        self.max_step = int(max_step)
        self.numpy_out = numpy_out
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1.")

        half = _stencil_halfspace(stencil)
        base_dirs = half + [(-dx, -dy, -dz) for dx, dy, dz in half] if both_directions else half
        self.directions = [
            (dx * step, dy * step, dz * step)
            for dx, dy, dz in base_dirs
            for step in range(1, self.max_step + 1)
        ]

    def _shift(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            return torch.roll(x, shifts=sh, dims=(0, 1, 2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_symmetric(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def _shift_adjoint(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            inv = (-sh[0], -sh[1], -sh[2])
            return torch.roll(x, shifts=inv, dims=(0, 1, 2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_adjoint(x, sh)
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
            out = out + self._shift_adjoint(edge, sh)
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
        max_step=1,
        normalize=True,
        numpy_out=False,
    ):
        self.anatomical = anatomical
        self.voxel_size = voxel_sizes
        self.gamma = gamma
        self.bnd_cond = bnd_cond
        self.max_step = int(max_step)
        self.numpy_out = numpy_out
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1.")

        self.gradient = Gradient(
            voxel_sizes=self.voxel_size,
            stencil=stencil,
            bnd_cond=self.bnd_cond,
            both_directions=both_directions,
            max_step=self.max_step,
            normalize=normalize,
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
    xi = anatomical_gradient / torch.sqrt(den**2 + eta**2)  # or (den + eta)
    return image_gradient - gamma * torch.sum(image_gradient * xi, dim=-1, keepdim=True) * xi


class GradientOptimized:
    """
    Optimized version of Gradient with pre-allocation and reduced padding overhead.
    Should be mathematically identical to Gradient but faster.
    """

    def __init__(
        self,
        voxel_sizes,
        stencil: str = "6",
        bnd_cond: str = "Neumann",
        both_directions: bool = False,
        max_step: int = 1,
        normalize: bool = True,
        numpy_out: bool = False,
    ):
        self.vx, self.vy, self.vz = map(float, voxel_sizes)
        self.bnd_cond = bnd_cond
        self.both_directions = both_directions
        self.max_step = int(max_step)
        self.normalize = normalize
        self.numpy_out = numpy_out
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1.")

        half = _stencil_halfspace(stencil)
        base_dirs = half + [(-dx, -dy, -dz) for dx, dy, dz in half] if both_directions else half
        self.directions = [
            (dx * step, dy * step, dz * step)
            for dx, dy, dz in base_dirs
            for step in range(1, self.max_step + 1)
        ]

        steps = []
        for dx, dy, dz in self.directions:
            sx, sy, sz = abs(dx) * self.vx, abs(dy) * self.vy, abs(dz) * self.vz
            steps.append(np.sqrt(sx * sx + sy * sy + sz * sz))
        self._step = torch.tensor(steps, device=device, dtype=torch.float32)
        self._bank_scale = np.sqrt(len(self.directions) / 3.0)

    def _shift(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            return torch.roll(x, shifts=sh, dims=(0, 1, 2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_symmetric(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def _shift_adjoint(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        if self.bnd_cond == "Periodic":
            inv = (-sh[0], -sh[1], -sh[2])
            return torch.roll(x, shifts=inv, dims=(0, 1, 2))
        elif self.bnd_cond == "Neumann":
            return _shift_neumann_adjoint(x, sh)
        else:
            raise ValueError("Unsupported boundary condition")

    def _diff_fwd(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        return self._shift(x, sh) - x

    def _diff_bwd(self, x: torch.Tensor, sh: tuple[int, int, int]) -> torch.Tensor:
        inv = (-sh[0], -sh[1], -sh[2])
        return x - self._shift(x, inv)

    def direct(self, x):
        X = _to_tensor(x, like_dtype=torch.float32, device=device)

        # Pre-allocate output to avoid list + stack
        n_dirs = len(self.directions)
        Y = torch.zeros((*X.shape, n_dirs), dtype=X.dtype, device=X.device)

        for i, sh in enumerate(self.directions):
            Y[..., i] = self._diff_fwd(X, sh)

        if self.normalize:
            Y = Y / self._step.view(*(1,) * (Y.ndim - 1), -1)
        Y = Y / self._bank_scale
        return _to_numpy(Y) if self.numpy_out else Y

    def adjoint(self, x):
        Y = _to_tensor(x, like_dtype=torch.float32, device=device)
        if self.normalize:
            Y = Y / self._step.view(*(1,) * (Y.ndim - 1), -1)

        out = torch.zeros_like(Y[..., 0])

        if self.bnd_cond == "Neumann":
            for ch, sh in enumerate(self.directions):
                y_ch = Y[..., ch]
                out = out + self._shift_adjoint(y_ch, sh)
                out = out - y_ch
        else:
            for ch, sh in enumerate(self.directions):
                out -= self._diff_bwd(Y[..., ch], sh)

        out = out / self._bank_scale
        return _to_numpy(out) if self.numpy_out else out

    def calculate_norm(self):
        return 2.0 * np.sqrt(len(self.directions)) / self._bank_scale


class Gradient:
    """Dispatcher that selects the appropriate gradient implementation."""

    def __init__(
        self,
        voxel_sizes,
        stencil: str = "6",
        bnd_cond: str = "Neumann",
        both_directions: bool = False,
        max_step: int = 1,
        normalize: bool = True,
        numpy_out: bool = False,
    ):
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes)
        self.stencil = stencil
        self.bnd_cond = bnd_cond
        self.both_directions = both_directions
        self.max_step = int(max_step)
        self.normalize = normalize
        self.numpy_out = numpy_out
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1.")

        impl_cls = GradientOptimized if stencil != "6" else _GradientLegacy
        self._impl = impl_cls(
            voxel_sizes=self.voxel_sizes,
            stencil=stencil,
            bnd_cond=bnd_cond,
            both_directions=both_directions,
            max_step=self.max_step,
            normalize=normalize,
            numpy_out=numpy_out,
        )

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def direct(self, x):
        return self._impl.direct(x)

    def adjoint(self, x):
        return self._impl.adjoint(x)

    def calculate_norm(self):
        return self._impl.calculate_norm()


LegacyGradient = _GradientLegacy


def check_adjoint(
    op,
    shape,
    *,
    input_is_vector=False,
    n_params=1,
    trials=3,
    seed=0,
    dtype=torch.float32,
    device=device,
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
        lhs = torch.sum(Y * y)  # <Gx, y>
        Ya = op.adjoint(y)
        if not isinstance(Ya, torch.Tensor):
            Ya = torch.as_tensor(Ya, device=device)
        rhs = torch.sum(x * Ya)  # <x, G^T y>
        num = (lhs - rhs).abs().item()
        den = max(1.0, float(max(lhs.abs().item(), rhs.abs().item())))
        max_rel = max(max_rel, num / den)
    return max_rel
