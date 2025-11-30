import numpy as np
from cil.optimisation.operators import LinearOperator

# --- Numba implementations for CPU acceleration ---
import numba
@numba.jit(nopython=True, parallel=True)
def _numba_convolve_3d(x, psf):
    D, H, W = x.shape
    pd, ph, pw = psf.shape
    out = np.zeros_like(x)
    for i in numba.prange(D):
        for j in range(H):
            for k in range(W):
                acc = 0.0
                for di in range(pd):
                    for dj in range(ph):
                        for dk in range(pw):
                            xi = i + di - pd//2
                            yj = j + dj - ph//2
                            zk = k + dk - pw//2
                            if 0 <= xi < D and 0 <= yj < H and 0 <= zk < W:
                                acc += x[xi, yj, zk] * psf[di, dj, dk]
                out[i, j, k] = acc
    return out

def get_fwhm_from_sigma(sigma):
    """Convert sigma to FWHM for Gaussian."""
    return [s * 2.3548 for s in sigma]

def get_sigma_from_fwhm(fwhm):
    """Convert FWHM to sigma for Gaussian."""
    return [f / 2.3548 for f in fwhm]


class GaussianBlurringOperator(LinearOperator):
    def __init__(self, fwhm, domain_geometry, backend='auto'):
        super().__init__(domain_geometry=domain_geometry,
                         range_geometry=domain_geometry)
        voxel_sizes = np.array(domain_geometry.voxel_sizes())
        self.sigma = np.array(get_sigma_from_fwhm(fwhm)) / voxel_sizes
        self.psf = self._make_psf(self.sigma)
        # choose backend
        if backend == 'auto':
            for b in ('torch', 'numba', 'scipy'):
                try:
                    if b == 'torch':
                        import torch
                        if not torch.cuda.is_available():
                            continue  # Skip torch if CUDA not available
                    else:
                        __import__(b)
                    backend = b
                    break
                except ImportError:
                    continue
        self.backend = backend
        if backend == 'torch':
            import torch
            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Torch backend selected but no CUDA GPUs available. "
                    "Use backend='auto', 'numba', or 'scipy' instead."
                )
            self.psf_t = torch.tensor(self.psf,
                                      dtype=torch.float32
                                    ).unsqueeze(0).unsqueeze(0).cuda()
        # numba & scipy need no extra storage

    @staticmethod
    def _make_psf(sigma, sd=3):
        rng = [int(np.ceil(s*sd)) for s in sigma]
        grids = np.meshgrid(*[np.arange(-r, r+1) for r in rng], indexing='ij')
        d2 = sum((g/s)**2 for g,s in zip(grids, sigma))
        psf = np.exp(-0.5*d2)
        return psf/psf.sum()

    def direct(self, x, out=None):
        arr = x.as_array()
        if   self.backend == 'torch':
            F = self.torch.nn.functional
            t = self.torch.tensor(arr, dtype=self.torch.float32
                                  ).unsqueeze(0).unsqueeze(0).cuda()
            kd,kh,kw = self.psf.shape
            blurred = F.conv3d(t, self.psf_t,
                               padding=(kd//2,kh//2,kw//2)
                              ).squeeze().cpu().numpy()
            del t
            self.clear_gpu()
        elif self.backend == 'numba':
            blurred = _numba_convolve_3d(arr, self.psf)
        else:  # scipy
            from scipy.ndimage import convolve
            blurred = convolve(arr, self.psf, mode='reflect')
        if out is None:
            out = x.clone()
        out.fill(blurred)
        return out

    def adjoint(self, x, out=None):
        arr = x.as_array()
        if   self.backend == 'torch':
            F = self.torch.nn.functional
            t = self.torch.tensor(arr, dtype=self.torch.float32
                                  ).unsqueeze(0).unsqueeze(0).cuda()
            psf_flip = self.psf_t.flip(-1).flip(-2).flip(-3)
            kd,kh,kw = self.psf.shape
            result = F.conv3d(t, psf_flip,
                              padding=(kd//2,kh//2,kw//2)
                             ).squeeze().cpu().numpy()
            del t
            self.clear_gpu()
        elif self.backend == 'numba':
            # Flip the PSF for cross-correlation (adjoint)
            psf_flipped = self.psf[::-1, ::-1, ::-1]
            result = _numba_convolve_3d(arr, psf_flipped)
        else:
            from scipy.ndimage import correlate
            result = correlate(arr, self.psf, mode='reflect')
        if out is None:
            out = x.clone()
        out.fill(result)
        return out
    
    def clear_gpu(self):
        """Release any cached GPU memory."""
        if self.backend == 'torch':
            # free PyTorch’s CUDA cache
            self.torch.cuda.empty_cache()


def create_gaussian_blur_operator(gauss_fwhm, template_image):
    """
    Create GaussianBlurringOperator from FWHM and template image.

    Args:
        gauss_fwhm: FWHM values in mm [z, y, x] or None
        template_image: SIRF ImageData for geometry information

    Returns:
        GaussianBlurringOperator or None if gauss_fwhm is None

    Note:
        GaussianBlurringOperator automatically converts FWHM (mm) to sigma
        and then to voxel units using the geometry's voxel sizes.
    """
    if gauss_fwhm is None:
        return None

    # GaussianBlurringOperator expects FWHM in mm and handles conversion internally
    return GaussianBlurringOperator(gauss_fwhm, template_image, backend='auto')
