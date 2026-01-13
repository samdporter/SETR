import numpy as np
import torch

from recon_core.utils.sirf import get_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BlockDataContainerToArray:
    def __init__(self, domain_geometry, gpu=True):
        self.domain_geometry = domain_geometry
        self.gpu = gpu

    def direct(self, x):
        if not hasattr(x, "containers"):
            raise ValueError(
                "Input x must be a block data container with a 'containers' attribute."
            )
        arrays = [get_array(d) for d in x.containers]
        if not self.gpu:
            return np.stack(arrays, axis=-1)
        tens = [torch.tensor(arr, device=device) for arr in arrays]
        return torch.stack(tens, dim=-1)

    def adjoint(self, x, out=None):
        if self.gpu and isinstance(x, torch.Tensor):
            x_arr = x.cpu().numpy()
        else:
            x_arr = np.asarray(x).clone()
        if out is not None:
            for i, r in enumerate(out):
                out[i].fill(x_arr[..., i])
            return out
        for i, r in enumerate(self.domain_geometry.containers):
            self.domain_geometry.containers[i].fill(x_arr[..., i])
        return self.domain_geometry
