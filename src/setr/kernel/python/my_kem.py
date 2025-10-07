import itertools

import numpy as np

np.seterr(over="raise", invalid="raise")

from cil.optimisation.operators import LinearOperator

from setr.utils.sirf import get_array

# try importing sliding_window_view from numpy
try:
    from numpy.lib.stride_tricks import sliding_window_view

    SLIDING_WINDOW_AVAIL = True
except ImportError:
    SLIDING_WINDOW_AVAIL = False

# Try importing numba
try:
    import numba

    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False
    NUMBA_AVAIL = False


def get_kernel_operator(domain_geometry, backend="auto", **kwargs):
    """
    Returns the best available kernel operator.
    backend: 'auto'|'numba'|'python'
    auto order: numba → python
    """
    if backend == "auto":
        backend = "numba" if NUMBA_AVAIL else "python"
    if backend == "numba" and NUMBA_AVAIL:
        return NumbaKernelOperator(domain_geometry, **kwargs)
    elif backend == "python" and SLIDING_WINDOW_AVAIL:
        return KernelOperator(domain_geometry, **kwargs)
    else:
        raise ValueError(
            f"Backend '{backend}' not available. "
            "Please install numba or numpy with sliding_window_view."
        )


class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry, range_geometry=domain_geometry)
        default_parameters = {
            "num_neighbours": 5,
            "sigma_anat": 0.1,
            "sigma_dist": 0.1,
            "sigma_emission": 0.1,
            "normalize_features": False,
            "normalize_kernel": False,
            "use_mask": False,
            "mask_k": None,
            "recalc_mask": False,
            "distance_weighting": False,
            "hybrid": False,
        }
        self.parameters = default_parameters | kwargs
        self.anatomical_image = None
        self.mask = None
        self.backend = "python"
        self.freeze_emission_kernel = False
        self.frozen_emission_kernel = None

    def set_parameters(self, parameters):
        self.parameters.update(parameters)
        self.mask = None

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

    def precompute_mask(self):
        n = self.parameters["num_neighbours"]
        K = n**3
        k = self.parameters["mask_k"] or K
        arr = get_array(self.anatomical_image)
        pad = n // 2

        arr_p = np.pad(arr, pad, mode="reflect")
        neigh = sliding_window_view(arr_p, (n, n, n))  # → (S0,S1,S2,n,n,n)
        S0, S1, S2, _, _, _ = neigh.shape
        flat = neigh.reshape(S0, S1, S2, K)  # → (S0,S1,S2,K)
        center = arr[..., None]  # → (S0,S1,S2,1)
        diff = np.abs(flat - center)  # → (S0,S1,S2,K)

        thresh = np.partition(diff, k - 1, axis=-1)[..., k - 1 : k]  # (S0,S1,S2,1)
        return diff <= thresh

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
        # default: same as forward for python backend
        res = self.direct(x)
        if out is None:
            return res
        out.fill(get_array(res))
        return out


class KernelOperator(BaseKernelOperator):
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
        """Pure Python implementation using explicit loops (slow but correct)."""
        arr = get_array(image)
        x_arr = get_array(x)
        n = num_neighbours
        half = n // 2
        s0, s1, s2 = arr.shape

        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        # Precompute distance weights
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        # Precompute mask if needed
        if use_mask:
            if self.mask is None or recalc_mask:
                self.mask = self.precompute_mask()
            mask = self.mask
        else:
            mask = None

        out = np.zeros_like(arr, dtype=np.float64)

        for i in range(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = arr[i, j, k]
                    cex = x_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0

                    if use_mask:
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

                                # Check mask
                                if use_mask:
                                    if not mask[i, j, k, idx]:
                                        idx += 1
                                        continue
                                    idx += 1

                                # Anatomical weight
                                diff_an = arr[ii, jj, kk] - ca
                                wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                w = wi_an * wd_an[di + half, dj + half, dk + half]

                                # Hybrid emission weight
                                if hybrid:
                                    diff_em = x_arr[ii, jj, kk] - cex
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                sumv += x_arr[ii, jj, kk] * w
                                wsum += w

                    if normalize_kernel and wsum > 1e-12:
                        sumv /= wsum
                    out[i, j, k] = sumv

        result = image.clone()
        result.fill(out)
        return result

    def adjoint(self, x, out=None):
        """Adjoint for Python backend using explicit loops (slow but correct)."""
        p = self.parameters

        # Only use non-symmetric adjoint when mask or hybrid is used
        if not (p["use_mask"] or p["hybrid"]):
            # For pure anatomical kernel without mask, it's self-adjoint
            res = self.direct(x)
            if out is None:
                return res
            out.fill(get_array(res))
            return out

        # Non-symmetric adjoint implementation
        arr = get_array(self.anatomical_image)
        x_arr = get_array(x)
        n = p["num_neighbours"]
        half = n // 2
        s0, s1, s2 = arr.shape

        sig2_an = 2.0 * p["sigma_anat"] * p["sigma_anat"]
        dist2_an = 2.0 * p["sigma_dist"] * p["sigma_dist"]
        sig2_em = 2.0 * p["sigma_emission"] * p["sigma_emission"]

        # Precompute distance weights
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if p["distance_weighting"]:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        # Precompute mask if needed
        if p["use_mask"]:
            if self.mask is None or p["recalc_mask"]:
                self.mask = self.precompute_mask()
            mask = self.mask
        else:
            mask = None

        result = np.zeros((s0, s1, s2), dtype=np.float64)

        for i in range(s0):
            for j in range(s1):
                for k in range(s2):
                    cv = arr[i, j, k]
                    val = x_arr[i, j, k]

                    if p["use_mask"]:
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

                                # Check mask
                                do_weight = True
                                if p["use_mask"]:
                                    do_weight = mask[i, j, k, idx]
                                    idx += 1

                                if do_weight:
                                    # Anatomical weight
                                    diff_an = arr[ii, jj, kk] - cv
                                    wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                    w = wi_an * wd_an[di + half, dj + half, dk + half]

                                    # Hybrid emission weight
                                    if p["hybrid"]:
                                        diff_em = x_arr[ii, jj, kk] - x_arr[i, j, k]
                                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                        w *= wi_em

                                    result[ii, jj, kk] += val * w

        img = x.clone()
        img.fill(result)

        if out is None:
            return img
        out.fill(result)
        return out


if NUMBA_AVAIL:
    # --- existing numba kernels (_nb_kernel, _nb_kernel_mask, _nb_adjoint) ---
    # (unchanged, already include hybrid in the mask‐kernel version)

    class NumbaKernelOperator(BaseKernelOperator):
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
            n = num_neighbours

            if use_mask:
                if self.mask is None or recalc_mask:
                    self.mask = self.precompute_mask()
                mask_int = self.mask.astype(np.int8)
                res = _nb_kernel_mask(
                    x_arr,
                    arr,
                    mask_int,
                    n,
                    sigma_anat,
                    sigma_dist,
                    sigma_emission,
                    normalize_kernel,
                    distance_weighting,
                    hybrid,
                )
            elif hybrid:
                full_mask = np.ones((arr.shape[0], arr.shape[1], arr.shape[2], n**3), dtype=np.int8)
                res = _nb_kernel_mask(
                    x_arr,
                    arr,
                    full_mask,
                    n,
                    sigma_anat,
                    sigma_dist,
                    sigma_emission,
                    normalize_kernel,
                    distance_weighting,
                    hybrid,
                )
            else:
                res = _nb_kernel(
                    x_arr,
                    arr,
                    n,
                    sigma_anat,
                    sigma_dist,
                    sigma_emission,
                    normalize_kernel,
                    distance_weighting,
                    hybrid,
                )

            out = image.clone()
            out.fill(res)
            return out

        def adjoint(self, x, out=None):
            arr = get_array(self.anatomical_image)
            x_arr = get_array(x)
            p = self.parameters

            if p["use_mask"] or p["hybrid"]:
                if self.mask is None or p["recalc_mask"]:
                    self.mask = self.precompute_mask()
                mask_int = self.mask.astype(np.int8)

                res = _nb_adjoint(
                    x_arr,
                    arr,
                    mask_int,
                    p["use_mask"],
                    p["num_neighbours"],
                    p["sigma_anat"],
                    p["sigma_dist"],
                    p["sigma_emission"],
                    p["distance_weighting"],
                    p["hybrid"],
                )
            else:
                # self-adjoint for pure anatomical
                res = _nb_kernel(
                    x_arr,
                    arr,
                    p["num_neighbours"],
                    p["sigma_anat"],
                    p["sigma_dist"],
                    p["sigma_emission"],
                    p["normalize_kernel"],
                    p["distance_weighting"],
                    p["hybrid"],
                )

            img = x.clone()
            img.fill(res)
            if out is None:
                return img
            out.fill(res)
            return out


@numba.njit(cache=True, parallel=True)
def _nb_kernel(
    x_arr,
    anat_arr,
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
                cex = x_arr[i, j, k]
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

                            # anat weight
                            diff_an = anat_arr[ii, jj, kk] - ca
                            wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                            w = wi_an * wd_an[di + half, dj + half, dk + half]

                            # hybrid emission
                            if hybrid:
                                diff_em = x_arr[ii, jj, kk] - cex
                                wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                w *= wi_em

                            sumv += x_arr[ii, jj, kk] * w
                            wsum += w

                if normalize and wsum > 1e-12:
                    sumv /= wsum
                out[i, j, k] = sumv

    return out


@numba.njit(cache=True, parallel=True)
def _nb_kernel_mask(
    x_arr,
    anat_arr,
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
                cex = x_arr[i, j, k]
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
                                    diff_em = x_arr[ii, jj, kk] - cex
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                sumv += x_arr[ii, jj, kk] * w
                                wsum += w
                            idx += 1

                if normalize and wsum > 1e-12:
                    sumv /= wsum
                out[i, j, k] = sumv

    return out


@numba.njit(cache=True, parallel=True)
def _nb_adjoint(
    x_arr,
    anat_arr,
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
                                    diff_em = x_arr[ii, jj, kk] - x_arr[i, j, k]
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                out[ii, jj, kk] += val * w
                            idx += 1

    return out
