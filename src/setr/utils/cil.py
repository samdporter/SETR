import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlockDataContainerToArray:
    def __init__(self, domain_geometry, gpu=True):
        self.domain_geometry = domain_geometry
        self.gpu = gpu

    def direct(self, x, out=None):
        if not hasattr(x, "containers"):
            raise ValueError("Input x must be a block data container with a 'containers' attribute.")
        arrays = [d.as_array() for d in x.containers]
        if self.gpu:
            tens = [torch.tensor(arr, device=device) for arr in arrays]
            ret = torch.stack(tens, dim=-1)
        else:
            ret = np.stack(arrays, axis=-1)
        if out is not None:
            out.fill(ret)
        return ret

    def adjoint(self, x, out=None):
        if self.gpu and isinstance(x, torch.Tensor):
            x_arr = x.cpu().numpy()
        else:
            x_arr = np.asarray(x)
        res = self.domain_geometry.clone()
        for i, r in enumerate(res.containers):
            r.fill(x_arr[..., i])
        if out is not None:
            out.fill(res)
        return res