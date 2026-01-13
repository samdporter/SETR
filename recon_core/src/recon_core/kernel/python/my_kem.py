import numpy as np

try:
    from cil.optimisation.operators import LinearOperator
except (ImportError, OSError):  # pragma: no cover - optional dependency
    class LinearOperator:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            self.__dict__.update(kwargs)

# Prefer the fully featured SIRF helper, but fall back to a minimal implementation
try:  # pragma: no cover - optional dependency
    from recon_core.utils.sirf import get_array  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def get_array(x):
        """Lightweight fallback that extracts NumPy arrays from common data containers."""
        if hasattr(x, "asarray"):
            return x.asarray()
        if hasattr(x, "as_array"):
            return x.as_array()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

# Try importing numba
try:
    import numba

    NUMBA_AVAIL = True
except (ImportError, OSError):  # pragma: no cover - optional dependency
    NUMBA_AVAIL = False
    numba = None  # type: ignore


DEFAULT_PARAMETERS = {
    "num_neighbours": 5,
    "sigma_anat": 0.1,
    "sigma_dist": 10000,
    "sigma_emission": 0.1,
    "normalize_features": True,
    "normalize_kernel": True,
    "use_mask": True,
    "mask_k": 20,
    "recalc_mask": False,
    "distance_weighting": False,
    "hybrid": False,
}


def get_kernel_operator(domain_geometry, backend="auto", **kwargs):
    """
    Returns the kernel operator with automatic backend selection.

    Parameters
    ----------
    domain_geometry : ImageGeometry
        Domain geometry for the operator
    backend : str, optional
        Backend to use: 'auto', 'torch', or 'numba'
        - 'auto': Try torch (GPU) first, fall back to numba (CPU)
        - 'torch': Use PyTorch GPU backend (requires torch + CUDA)
        - 'numba': Use Numba CPU backend
    **kwargs
        Additional parameters passed to the operator

    Returns
    -------
    BaseKernelOperator
        Kernel operator instance (TorchKernelOperator or KernelOperator)

    Examples
    --------
    Auto-select backend (prefers GPU):
    >>> op = get_kernel_operator(geometry, backend='auto')

    Force GPU:
    >>> op = get_kernel_operator(geometry, backend='torch', dtype='float32')

    Force CPU:
    >>> op = get_kernel_operator(geometry, backend='numba')
    """
    if backend == "auto":
        # Try torch (GPU) first, fall back to numba (CPU)
        for b in ('torch', 'numba'):
            try:
                if b == 'torch':
                    import torch
                    if torch.cuda.is_available():
                        backend = 'torch'
                        break
                elif b == 'numba':
                    import numba
                    backend = 'numba'
                    break
            except ImportError:
                continue

        # If nothing worked, default to numba and let it raise error if not available
        if backend == 'auto':
            backend = 'numba'

    if backend == 'torch':
        try:
            from .gpu_kernel_operator import TorchKernelOperator
            return TorchKernelOperator(domain_geometry, **kwargs)
        except ImportError as e:
            raise RuntimeError(
                f"PyTorch backend not available: {e}\n"
                "Install pytorch with: pip install torch torchvision"
            )
    elif backend == 'numba':
        if not NUMBA_AVAIL:
            raise RuntimeError(
                "Numba backend not available. Please install numba to use the kernel operator."
            )
        return KernelOperator(domain_geometry, **kwargs)
    else:
        raise ValueError(
            f"Backend '{backend}' not supported. "
            "Use 'auto', 'torch', or 'numba'."
        )


class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry, range_geometry=domain_geometry)
        default_parameters = DEFAULT_PARAMETERS.copy()
        self.parameters = default_parameters | kwargs
        self.anatomical_image = None
        self.mask = None
        self.backend = "numba"
        self.freeze_emission_kernel = False
        self.frozen_emission_kernel = None
        self._normalisation_map = None
        self._full_mask_cache: np.ndarray | None = None
        self._anatomical_weights: np.ndarray | None = None

    def set_parameters(self, parameters):
        self.parameters.update(parameters)
        self.mask = None
        self._anatomical_weights = None

    def set_anatomical_image(self, image):
        if self.parameters["normalize_features"]:
            arr = get_array(image)
            std = arr.std()
            norm = arr / std if std > 1e-12 else arr
            tmp = image.clone()
            tmp.fill(norm)
            self.anatomical_image = tmp
        else:
            self.anatomical_image = image
        self.mask = None
        self._anatomical_weights = None

    def precompute_mask(self):
        if not NUMBA_AVAIL:
            raise RuntimeError("Numba backend required for mask precomputation.")
        if self.anatomical_image is None:
            raise RuntimeError(
                "An anatomical image must be set before precomputing a mask."
            )
        n = int(self.parameters["num_neighbours"])
        total = n**3
        mask_k = self.parameters["mask_k"]
        k = mask_k if mask_k is not None else total
        k = max(1, min(int(k), total))
        arr = np.ascontiguousarray(get_array(self.anatomical_image), dtype=np.float64)
        return _nb_precompute_mask(arr, n, k)

    def precompute_anatomical_weights(self):
        """
        Pre-compute the anatomical kernel weights that remain constant across iterations.
        This includes:
        - Anatomical intensity-based Gaussian weights
        - Distance-based weights (if enabled)
        - Combined weights stored per voxel and neighbor

        Returns:
            np.ndarray: Pre-computed weights of shape (s0, s1, s2, total_neighbors)
                       where total_neighbors is n³ (without mask) or k (with mask)
        """
        if not NUMBA_AVAIL:
            raise RuntimeError("Numba backend required for weight precomputation.")
        if self.anatomical_image is None:
            raise RuntimeError(
                "An anatomical image must be set before precomputing weights."
            )

        n = int(self.parameters["num_neighbours"])
        sigma_anat = self.parameters["sigma_anat"]
        sigma_dist = self.parameters["sigma_dist"]
        distance_weighting = self.parameters["distance_weighting"]
        use_mask = self.parameters["use_mask"]

        arr = np.ascontiguousarray(get_array(self.anatomical_image), dtype=np.float64)

        if use_mask:
            if self.mask is None:
                self.mask = self.precompute_mask()
            return _nb_precompute_anatomical_weights_mask(
                arr, self.mask, n, sigma_anat, sigma_dist, distance_weighting
            )
        else:
            return _nb_precompute_anatomical_weights(
                arr, n, sigma_anat, sigma_dist, distance_weighting
            )

    def _get_full_mask(self, shape: tuple[int, int, int], n: int) -> np.ndarray:
        """Return a cached all-True mask for the current geometry."""
        expected_shape = (shape[0], shape[1], shape[2], n**3)
        mask = self._full_mask_cache
        if mask is None or mask.shape != expected_shape:
            mask = np.ones(expected_shape, dtype=np.bool_)
            self._full_mask_cache = mask
        return mask

    def _update_hybrid_reference(self, emission_array: np.ndarray) -> np.ndarray:
        """Store (or reuse) the emission image that defines the hybrid weights."""
        if not self.parameters["hybrid"]:
            return emission_array

        # If frozen, always return the frozen reference (don't update)
        if self.freeze_emission_kernel:
            if self.frozen_emission_kernel is None:
                # First call after freezing: initialize and freeze
                self.frozen_emission_kernel = np.array(emission_array, copy=True)
            return self.frozen_emission_kernel

        # Not frozen: update the reference
        if emission_array is None:
            raise ValueError("Hybrid emission reference requires an emission array.")

        self.frozen_emission_kernel = np.array(emission_array, copy=True)
        return self.frozen_emission_kernel

    def _get_hybrid_reference(self) -> np.ndarray | None:
        """Return the emission image used for the hybrid weights."""
        if not self.parameters["hybrid"]:
            return None

        if self.frozen_emission_kernel is None:
            raise RuntimeError(
                "Hybrid emission reference has not been initialised. "
                "Call direct() (or explicitly freeze a reference) before adjoint()."
            )

        return self.frozen_emission_kernel

    def apply(self, x):
        p = self.parameters
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            p["num_neighbours"],
            p["sigma_anat"],
            p["sigma_dist"],
            p["sigma_emission"],
            p["normalize_kernel"],
            p["use_mask"],
            p["recalc_mask"],
            p["distance_weighting"],
            p["hybrid"],
        )

    def direct(self, x, out=None):
        res = self.apply(x)
        if out is None:
            return res
        out.fill(get_array(res))
        return out

    def adjoint(self, x, out=None):
        # default: same as forward (kernel remains self-adjoint without mask/hybrid)
        res = self.direct(x)
        if out is None:
            return res
        out.fill(get_array(res))
        return out


if NUMBA_AVAIL:
    # --- existing numba kernels (_nb_kernel, _nb_kernel_mask, _nb_adjoint) ---
    # (unchanged, already include hybrid in the mask‐kernel version)

    class KernelOperator(BaseKernelOperator):
        def __init__(self, domain_geometry, **kwargs):
            super().__init__(domain_geometry, **kwargs)
            self.backend = "numba"

        def neighbourhood_kernel(
            self,
            x,
            image,
            num_neighbours,
            sigma_anat,
            sigma_dist,
            sigma_emission,
            normalize_kernel,
            use_mask,
            recalc_mask,
            distance_weighting,
            hybrid,
        ):
            arr = get_array(image)
            x_arr = get_array(x)
            ref_arr = self._update_hybrid_reference(x_arr) if hybrid else x_arr
            norm_arr = (
                np.zeros_like(arr, dtype=np.float64)
                if normalize_kernel
                else np.zeros((1, 1, 1), dtype=np.float64)
            )
            n = num_neighbours

            # Pre-compute or retrieve cached anatomical weights
            if self._anatomical_weights is None:
                self._anatomical_weights = self.precompute_anatomical_weights()

            # Use sparse or dense pre-computed kernel based on masking
            if use_mask:
                res = _nb_kernel_precomputed_sparse(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    self.mask,
                    norm_arr,
                    n,
                    sigma_emission,
                    normalize_kernel,
                    hybrid,
                )
            else:
                res = _nb_kernel_precomputed(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    norm_arr,
                    n,
                    sigma_emission,
                    normalize_kernel,
                    hybrid,
                )

            out = image.clone()
            out.fill(res)
            self._normalisation_map = norm_arr if normalize_kernel else None
            return out

        def adjoint(self, x, out=None):
            arr = get_array(self.anatomical_image)
            x_arr = get_array(x)
            p = self.parameters
            norm_arr = (
                self._normalisation_map
                if p["normalize_kernel"]
                else np.zeros((1, 1, 1), dtype=np.float64)
            )
            if p["normalize_kernel"] and norm_arr is None:
                raise RuntimeError(
                    "Normalization map has not been initialised. "
                    "Call direct() before adjoint() when using normalize_kernel=True."
                )

            n = p["num_neighbours"]

            # Pre-compute or retrieve cached anatomical weights
            if self._anatomical_weights is None:
                self._anatomical_weights = self.precompute_anatomical_weights()

            ref_arr = self._get_hybrid_reference() if p["hybrid"] else x_arr

            # Use sparse or dense pre-computed adjoint kernel based on masking
            if p["use_mask"]:
                res = _nb_adjoint_precomputed_sparse(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    self.mask,
                    norm_arr,
                    n,
                    p["sigma_emission"],
                    p["hybrid"],
                )
            else:
                res = _nb_adjoint_precomputed(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    norm_arr,
                    n,
                    p["sigma_emission"],
                    p["hybrid"],
                )

            img = x.clone()
            img.fill(res)
            if out is None:
                return img
            out.fill(res)
            return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_precompute_mask(anat_arr, n, k_keep):
        """
        Precompute sparse mask as integer indices of the k_keep most similar neighbors.
        Returns shape (s0, s1, s2, k_keep) with integer indices in [0, n³).
        """
        s0, s1, s2 = anat_arr.shape
        total = n ** 3
        half = n // 2
        # Return sparse indices instead of boolean mask
        mask_indices = np.zeros((s0, s1, s2, k_keep), dtype=np.int32)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    diffs = np.empty(total, dtype=np.float64)
                    center = anat_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                diffs[idx] = abs(anat_arr[ii, jj, kk] - center)
                                idx += 1

                    # Find indices of k_keep smallest differences
                    # Using argsort (partial sort would be better but numba doesn't support it well)
                    sorted_indices = np.argsort(diffs)
                    mask_indices[i, j, k, :] = sorted_indices[:k_keep]

        return mask_indices


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_precompute_anatomical_weights(anat_arr, n, sigma_anat, sigma_dist, distance_weighting):
        """
        Pre-compute anatomical weights for all voxels and all n³ neighbors.
        Returns shape: (s0, s1, s2, n³)
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        total = n ** 3
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist

        # Pre-compute distance weights
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        weights = np.zeros((s0, s1, s2, total), dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = anat_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                diff_an = anat_arr[ii, jj, kk] - ca
                                wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                weights[i, j, k, idx] = wi_an * wd_an[di + half, dj + half, dk + half]
                                idx += 1

        return weights


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_precompute_anatomical_weights_mask(anat_arr, mask_indices, n, sigma_anat, sigma_dist, distance_weighting):
        """
        Pre-compute anatomical weights for all voxels using sparse mask indices.
        mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
        Returns: shape (s0, s1, s2, k) with weights for valid neighbors only.
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        k = mask_indices.shape[3]  # Number of kept neighbors
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist

        # Pre-compute distance weights for all possible offsets
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        # Sparse weights - only store k neighbors per voxel
        weights = np.zeros((s0, s1, s2, k), dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k_vox in range(s2):
                    ca = anat_arr[i, j, k_vox]

                    # Iterate only over the k valid neighbors
                    for k_idx in range(k):
                        flat_idx = mask_indices[i, j, k_vox, k_idx]

                        # Convert flat index back to (di, dj, dk) offset
                        dk = (flat_idx % n) - half
                        dj = ((flat_idx // n) % n) - half
                        di = (flat_idx // (n * n)) - half

                        # Apply boundary conditions
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        kk = k_vox + dk
                        if kk < 0:
                            kk = -kk - 1
                        elif kk >= s2:
                            kk = 2 * s2 - kk - 1

                        # Compute anatomical weight
                        diff_an = anat_arr[ii, jj, kk] - ca
                        wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                        weights[i, j, k_vox, k_idx] = wi_an * wd_an[di + half, dj + half, dk + half]

        return weights


    @numba.njit(cache=True, parallel=True)
    def _nb_kernel(
        x_arr, ref_arr, anat_arr, norm_arr, n,
        sigma_anat, sigma_dist, sigma_emission,
        normalize, distance_weighting, hybrid,
    ):
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        out = np.empty_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = anat_arr[i, j, k]
                    c_ref = ref_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                # weights & accumulation MUST be inside dk-loop
                                diff_an = anat_arr[ii, jj, kk] - ca
                                wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                w = wi_an * wd_an[di + half, dj + half, dk + half]
                                if hybrid:
                                    diff_em = ref_arr[ii, jj, kk] - c_ref
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                sumv += x_arr[ii, jj, kk] * w
                                wsum += w

                    if normalize:
                        if wsum > 1e-12:
                            sumv /= wsum
                            norm_arr[i, j, k] = wsum
                        else:
                            norm_arr[i, j, k] = 1.0
                    out[i, j, k] = sumv

        return out


    @numba.njit(cache=True, parallel=True)
    def _nb_kernel_mask(
        x_arr,
        ref_arr,
        anat_arr,
        norm_arr,
        mask,
        n,
        sigma_anat,
        sigma_dist,
        sigma_emission,
        normalize,
        distance_weighting,
        hybrid,
    ):
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        # precompute spatial weights
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)
        out = np.empty_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = anat_arr[i, j, k]
                    c_ref = ref_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0
                    idx = 0

                    for di in range(-half, half + 1):
                        for dj in range(-half, half + 1):
                            for dk in range(-half, half + 1):
                                if mask[i, j, k, idx]:
                                    ii = i + di
                                    if ii < 0:
                                        ii = -ii - 1
                                    elif ii >= s0:
                                        ii = 2 * s0 - ii - 1
                                    jj = j + dj
                                    if jj < 0:
                                        jj = -jj - 1
                                    elif jj >= s1:
                                        jj = 2 * s1 - jj - 1
                                    kk = k + dk
                                    if kk < 0:
                                        kk = -kk - 1
                                    elif kk >= s2:
                                        kk = 2 * s2 - kk - 1

                                    # anat weight
                                    diff_an = anat_arr[ii, jj, kk] - ca
                                    wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                    w = wi_an * wd_an[di + half, dj + half, dk + half]

                                    # hybrid emission
                                    if hybrid:
                                        diff_em = ref_arr[ii, jj, kk] - c_ref
                                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                        w *= wi_em

                                    sumv += x_arr[ii, jj, kk] * w
                                    wsum += w
                                idx += 1

                    if normalize:
                        norm = wsum if wsum > 1e-12 else 1.0
                        if wsum > 1e-12:
                            sumv /= wsum
                        norm_arr[i, j, k] = norm
                    out[i, j, k] = sumv

        return out


    @numba.njit(cache=True, parallel=True)
    def _nb_adjoint(
        x_arr,
        ref_arr,
        anat_arr,
        norm_arr,
        mask,
        use_mask,
        n,
        sigma_anat,
        sigma_dist,
        sigma_emission,
        distance_weighting,
        hybrid,
    ):
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        out = np.zeros_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    cv = anat_arr[i, j, k]
                    val = x_arr[i, j, k]
                    if norm_arr.shape[0] > 1:
                        norm = norm_arr[i, j, k]
                        val = val / norm if norm > 1e-12 else 0.0
                    c_ref = ref_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                do_weight = (not use_mask) or mask[i, j, k, idx]
                                if do_weight:
                                    diff_an = anat_arr[ii, jj, kk] - cv
                                    wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                    w = wi_an * wd_an[di + half, dj + half, dk + half]

                                    if hybrid:
                                        diff_em = ref_arr[ii, jj, kk] - c_ref
                                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                        w *= wi_em

                                    out[ii, jj, kk] += val * w
                                idx += 1

        return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_kernel_precomputed(
        x_arr,
        ref_arr,
        anat_weights,
        norm_arr,
        n,
        sigma_emission,
        normalize,
        hybrid,
    ):
        """
        Forward kernel using pre-computed anatomical weights (dense version, no mask).
        Only calculates emission weights (if hybrid) and applies to data.
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        total = n ** 3
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.empty_like(x_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    c_ref = ref_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                # Get pre-computed anatomical weight
                                w = anat_weights[i, j, k, idx]

                                # Apply emission weight if hybrid
                                if hybrid and w > 0.0:
                                    diff_em = ref_arr[ii, jj, kk] - c_ref
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                sumv += x_arr[ii, jj, kk] * w
                                wsum += w
                                idx += 1

                    if normalize:
                        if wsum > 1e-12:
                            sumv /= wsum
                            norm_arr[i, j, k] = wsum
                        else:
                            norm_arr[i, j, k] = 1.0
                    out[i, j, k] = sumv

        return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_kernel_precomputed_sparse(
        x_arr,
        ref_arr,
        anat_weights,
        mask_indices,
        norm_arr,
        n,
        sigma_emission,
        normalize,
        hybrid,
    ):
        """
        Forward kernel using pre-computed anatomical weights (sparse version with mask).
        Only iterates over k masked neighbors per voxel.
        anat_weights: shape (s0, s1, s2, k)
        mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        k = anat_weights.shape[3]
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.empty_like(x_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k_vox in range(s2):
                    c_ref = ref_arr[i, j, k_vox]
                    sumv = 0.0
                    wsum = 0.0

                    # Iterate only over k valid neighbors
                    for k_idx in range(k):
                        flat_idx = mask_indices[i, j, k_vox, k_idx]

                        # Convert flat index to (di, dj, dk) offset
                        dk = (flat_idx % n) - half
                        dj = ((flat_idx // n) % n) - half
                        di = (flat_idx // (n * n)) - half

                        # Apply boundary conditions
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        kk = k_vox + dk
                        if kk < 0:
                            kk = -kk - 1
                        elif kk >= s2:
                            kk = 2 * s2 - kk - 1

                        # Get pre-computed anatomical weight
                        w = anat_weights[i, j, k_vox, k_idx]

                        # Apply emission weight if hybrid
                        if hybrid and w > 0.0:
                            diff_em = ref_arr[ii, jj, kk] - c_ref
                            wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                            w *= wi_em

                        sumv += x_arr[ii, jj, kk] * w
                        wsum += w

                    if normalize:
                        if wsum > 1e-12:
                            sumv /= wsum
                            norm_arr[i, j, k_vox] = wsum
                        else:
                            norm_arr[i, j, k_vox] = 1.0
                    out[i, j, k_vox] = sumv

        return out


    # Scatter updates cause races when parallelized, so keep serial execution.
    @numba.njit(cache=True, parallel=False, fastmath=True)
    def _nb_adjoint_precomputed(
        x_arr,
        ref_arr,
        anat_weights,
        norm_arr,
        n,
        sigma_emission,
        hybrid,
    ):
        """
        Adjoint kernel using pre-computed anatomical weights (dense version, no mask).
        Only calculates emission weights (if hybrid) and applies to data.
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.zeros_like(x_arr, dtype=np.float64)

        for i in range(s0):
            for j in range(s1):
                for k in range(s2):
                    val = x_arr[i, j, k]
                    if norm_arr.shape[0] > 1:
                        norm = norm_arr[i, j, k]
                        val = val / norm if norm > 1e-12 else 0.0
                    c_ref = ref_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                # Get pre-computed anatomical weight
                                w = anat_weights[i, j, k, idx]

                                # Apply emission weight if hybrid
                                if hybrid and w > 0.0:
                                    diff_em = ref_arr[ii, jj, kk] - c_ref
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                out[ii, jj, kk] += val * w
                                idx += 1

        return out


    # Scatter updates cause races when parallelized, so keep serial execution.
    @numba.njit(cache=True, parallel=False, fastmath=True)
    def _nb_adjoint_precomputed_sparse(
        x_arr,
        ref_arr,
        anat_weights,
        mask_indices,
        norm_arr,
        n,
        sigma_emission,
        hybrid,
    ):
        """
        Adjoint kernel using pre-computed anatomical weights (sparse version with mask).
        Only iterates over k masked neighbors per voxel.
        anat_weights: shape (s0, s1, s2, k)
        mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        k = anat_weights.shape[3]
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.zeros_like(x_arr, dtype=np.float64)

        for i in range(s0):
            for j in range(s1):
                for k_vox in range(s2):
                    val = x_arr[i, j, k_vox]
                    if norm_arr.shape[0] > 1:
                        norm = norm_arr[i, j, k_vox]
                        val = val / norm if norm > 1e-12 else 0.0
                    c_ref = ref_arr[i, j, k_vox]

                    # Iterate only over k valid neighbors
                    for k_idx in range(k):
                        flat_idx = mask_indices[i, j, k_vox, k_idx]

                        # Convert flat index to (di, dj, dk) offset
                        dk = (flat_idx % n) - half
                        dj = ((flat_idx // n) % n) - half
                        di = (flat_idx // (n * n)) - half

                        # Apply boundary conditions
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        kk = k_vox + dk
                        if kk < 0:
                            kk = -kk - 1
                        elif kk >= s2:
                            kk = 2 * s2 - kk - 1

                        # Get pre-computed anatomical weight
                        w = anat_weights[i, j, k_vox, k_idx]

                        # Apply emission weight if hybrid
                        if hybrid and w > 0.0:
                            diff_em = ref_arr[ii, jj, kk] - c_ref
                            wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                            w *= wi_em

                        out[ii, jj, kk] += val * w

        return out


else:

    class KernelOperator(BaseKernelOperator):  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Numba backend not available. Please install numba to use KernelOperator."
            )
