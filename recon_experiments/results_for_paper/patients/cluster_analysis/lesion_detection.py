"""
Automatic lesion detection and masking for medical imaging.

This module provides intelligent functions to automatically identify and create
masks for hot lesions based on uptake patterns in PET/SPECT images.
"""

import logging
import numpy as np
from scipy import ndimage
from typing import List, Dict, Optional, Tuple, Union, Literal
from sirf.STIR import ImageData


def find_hottest_lesions(
    image: Union[ImageData, np.ndarray],
    num_lesions: int = 5,
    min_separation: float = 30.0,
    background_percentile: float = 50.0,
    voxel_sizes: Optional[Tuple[float, float, float]] = None,
    background_value: Optional[float] = None,
) -> List[Tuple[int, int, int]]:
    """
    Find the M hottest lesions in an image, ensuring spatial separation.

    This function identifies local maxima that stand out from the background
    and are spatially separated from each other.

    Parameters
    ----------
    image : ImageData or ndarray
        3D image to analyze
    num_lesions : int
        Number of lesions to identify (M)
    min_separation : float
        Minimum distance (in mm) between lesion centers (default 30 mm)
    background_percentile : float
        Percentile of image values to use as background threshold (0-100)
    voxel_sizes : tuple of float, optional
        Voxel sizes in mm (z, y, x). Required if image is ndarray.

    Returns
    -------
    lesion_centers : list of tuple
        List of (z, y, x) coordinates for each lesion center
    """
    # Convert to numpy array if needed
    if hasattr(image, 'as_array'):
        img_arr = image.as_array()
        if voxel_sizes is None:
            # STIR returns (z, y, x)
            voxel_sizes = image.voxel_sizes()
    else:
        img_arr = np.asarray(image)
        if voxel_sizes is None:
            raise ValueError("voxel_sizes must be provided for numpy arrays")

    # Threshold based on background (prefer user-supplied mean, fall back to percentile)
    if background_value is not None:
        threshold = float(background_value)
    else:
        threshold = np.percentile(img_arr[img_arr > 0], background_percentile)

    # Find local maxima using maximum filter
    # Use a structuring element sized for ~5mm radius
    footprint_size = tuple(max(1, int(5.0 / vs)) for vs in voxel_sizes)
    local_max = ndimage.maximum_filter(img_arr, size=footprint_size) == img_arr
    local_max &= img_arr > threshold

    # Get coordinates and values of all local maxima
    coords = np.argwhere(local_max)
    values = img_arr[local_max]

    # Sort by intensity (descending)
    sorted_indices = np.argsort(values)[::-1]
    sorted_coords = coords[sorted_indices]
    sorted_values = values[sorted_indices]

    # Select lesions with minimum physical separation
    selected_lesions = []
    # Explicit axis order: coords are (z, y, x)
    vz, vy, vx = voxel_sizes
    # Prefer the grow_lesions logger if configured, fall back to module logger
    logger = logging.getLogger("grow_lesions") if logging.getLogger("grow_lesions").handlers else logging.getLogger(__name__)
    logger.warning(f"Hotspot selection: min_separation={min_separation} mm, voxel_sizes=(vz={vz}, vy={vy}, vx={vx}), candidates={len(sorted_coords)}, threshold={threshold:.3g}")

    for coord, value in zip(sorted_coords, sorted_values):
        if len(selected_lesions) >= num_lesions:
            break

        # Check separation from already selected lesions
        if len(selected_lesions) == 0:
            selected_lesions.append(tuple(coord))
            logger.warning(f"Hotspot seed {tuple(coord)} accepted (first seed)")
        else:
            # Calculate minimum physical distance (mm) to any selected lesion
            distances_mm = []
            for sel in selected_lesions:
                dz, dy, dx = coord - np.array(sel)
                dist_mm = np.sqrt((dz * vz) ** 2 + (dy * vy) ** 2 + (dx * vx) ** 2)
                distances_mm.append(dist_mm)
            if distances_mm:
                min_dist = float(min(distances_mm))
                if min_dist >= min_separation:
                    selected_lesions.append(tuple(coord))
                    logger.warning(
                        f"Hotspot seed {tuple(coord)} accepted: min distance {min_dist:.2f} mm ≥ {min_separation} mm"
                    )
                else:
                    logger.warning(
                        f"Hotspot seed {tuple(coord)} rejected: min distance {min_dist:.2f} mm < {min_separation} mm"
                    )

    return selected_lesions


import numpy as np
from typing import Literal, Optional, Tuple, Union, List
from scipy import ndimage


def generate_fibonacci_sphere_directions(n_samples: int = 162) -> np.ndarray:
    """
    Generate evenly distributed unit vectors on sphere using Fibonacci spiral.

    The Fibonacci sphere algorithm provides near-uniform distribution of points
    on a sphere, avoiding clustering at poles.

    Parameters
    ----------
    n_samples : int
        Number of ray directions (default 162 for good spherical coverage)

    Returns
    -------
    directions : np.ndarray, shape (n_samples, 3)
        Unit vectors in (dz, dy, dx) order matching image coordinate system

    References
    ----------
    Swinbank & Purser (2006). Fibonacci grids: A novel approach to global modelling.
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_samples)
    theta = golden_angle * indices
    z = np.linspace(1 - 1.0/n_samples, -1 + 1.0/n_samples, n_samples)
    radius = np.sqrt(1 - z*z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    # Return as (n_samples, 3) with order (z, y, x)
    return np.column_stack([z, y, x])


def cast_ray_and_find_max_gradient(
    image: np.ndarray,
    gradient_z: np.ndarray,
    gradient_y: np.ndarray,
    gradient_x: np.ndarray,
    seed_point: Tuple[int, int, int],
    direction: np.ndarray,
    max_ray_length: float,
    min_intensity_fraction: float,
    seed_value: float,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    background_mean: Optional[float] = None
) -> Optional[Tuple[int, int, int]]:
    """
    Cast ray from seed point and find voxel with maximum NEGATIVE gradient.

    Traverses along the ray direction from the seed point, searching for the
    point of steepest NEGATIVE gradient (intensity decreasing outward - the
    lesion boundary). Only accepts negative gradients to avoid confusion with
    nearby hot lesions (which would have positive gradients in the ray direction).

    Parameters
    ----------
    image : np.ndarray
        3D intensity image
    gradient_z : np.ndarray
        3D gradient in z direction
    gradient_y : np.ndarray
        3D gradient in y direction
    gradient_x : np.ndarray
        3D gradient in x direction
    seed_point : tuple of int
        Starting point (z, y, x) in voxel coordinates
    direction : np.ndarray
        Unit direction vector (dz, dy, dx)
    max_ray_length : float
        Maximum distance to search along ray (in mm)
    min_intensity_fraction : float
        Fraction for threshold calculation (typically 0.42 = 42% for PET).
        Uses proper PET formula: threshold = background + fraction * (peak - background)
    seed_value : float
        Intensity at seed point
    voxel_sizes : tuple of float
        Voxel dimensions in mm as (vz, vy, vx)
    background_mean : float, optional
        Mean background intensity. If provided, uses proper PET threshold formula:
        threshold = background + fraction * (peak - background).
        If None, falls back to: threshold = fraction * peak

    Returns
    -------
    boundary_point : tuple of int or None
        (z, y, x) coordinates of max negative gradient point, or None if no valid point found

    Notes
    -----
    The directional gradient is computed as the dot product of the gradient vector
    with the ray direction. A negative value indicates intensity is decreasing in
    the ray direction (moving away from lesion center toward background).
    """
    z0, y0, x0 = seed_point
    Z, Y, X = image.shape

    # Convert max_ray_length from mm to voxels
    mean_voxel_size = np.mean(voxel_sizes)
    step_size_voxels = 0.5  # Subvoxel stepping for accuracy
    max_steps = int(max_ray_length / (mean_voxel_size * step_size_voxels))

    # Normalize direction and scale by anisotropic voxel sizes
    direction = direction / np.linalg.norm(direction)
    scaled_direction = direction / np.array(voxel_sizes)
    scaled_direction = scaled_direction / np.linalg.norm(scaled_direction)

    max_grad = -np.inf
    best_point = None

    # Use proper PET threshold formula: background + fraction * (peak - background)
    # If background_mean is not provided, fall back to simple fraction of peak
    if background_mean is not None:
        min_intensity = background_mean + min_intensity_fraction * (seed_value - background_mean)
    else:
        min_intensity = seed_value * min_intensity_fraction

    for step in range(1, max_steps + 1):
        # Position along ray in voxel coordinates
        pos = np.array([z0, y0, x0], dtype=float) + scaled_direction * step * step_size_voxels
        zi, yi, xi = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))

        # Check bounds
        if not (0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X):
            break

        # Check intensity threshold (stop if too dark)
        if image[zi, yi, xi] < min_intensity:
            break

        # Compute directional gradient (dot product of gradient vector with ray direction)
        # Positive = intensity increasing in ray direction (approaching nearby hot lesion)
        # Negative = intensity decreasing in ray direction (lesion boundary - what we want)
        grad_vec = np.array([gradient_z[zi, yi, xi], gradient_y[zi, yi, xi], gradient_x[zi, yi, xi]])
        directional_grad = np.dot(grad_vec, direction)

        # CRITICAL: Only accept NEGATIVE gradients (intensity decreasing outward)
        # This prevents confusion with nearby hot lesions
        if directional_grad >= 0:
            continue  # Skip positive gradients

        # Track maximum negative gradient (most negative = steepest drop)
        grad_magnitude = abs(directional_grad)
        if grad_magnitude > max_grad:
            max_grad = grad_magnitude
            best_point = (zi, yi, xi)

    return best_point


def construct_mask_delaunay(
    boundary_points: List[Tuple[int, int, int]],
    image_shape: Tuple[int, int, int],
    seed_point: Tuple[int, int, int]
) -> np.ndarray:
    """
    Construct lesion mask using Delaunay triangulation from sparse boundary points.

    CRITICAL: This method is required because binary_fill_holes would fail with
    sparse boundary points (162 rays → ~14mm spacing with 2-4mm voxels = porous
    boundary → fill would leak). Delaunay triangulation properly reconstructs
    the mesh from sparse points.

    Parameters
    ----------
    boundary_points : list of tuple
        List of (z, y, x) boundary voxel coordinates
    image_shape : tuple of int
        Shape of the image volume (Z, Y, X)
    seed_point : tuple of int
        Seed voxel (z, y, x) - included in mask as fallback

    Returns
    -------
    mask : np.ndarray, dtype=bool
        Binary mask of lesion

    Notes
    -----
    - Assumes lesion is star-convex from seed point
    - Performance: O(N_voxels × log(N_boundary)) ≈ 50-200ms per lesion
    - Fallback: Returns seed-only mask if Delaunay fails (e.g., coplanar points)

    References
    ----------
    Mikell et al. (2018). Impact of 90Y PET gradient-based tumor segmentation
    on voxel-level dosimetry in liver radioembolization.
    """
    from scipy.spatial import Delaunay

    mask = np.zeros(image_shape, dtype=bool)

    if len(boundary_points) < 4:
        # Degenerate case: not enough points for 3D triangulation
        mask[seed_point] = True
        return mask

    # Convert to numpy array
    points = np.array(boundary_points)

    try:
        # Build Delaunay triangulation
        delaunay = Delaunay(points)
    except Exception as e:
        # Fallback to seed-only if Delaunay fails (e.g., coplanar points)
        import logging
        logger = logging.getLogger("grow_lesions") if logging.getLogger("grow_lesions").handlers else logging.getLogger(__name__)
        logger.warning(f"Delaunay triangulation failed: {e}. Using seed-only mask.")
        mask[seed_point] = True
        return mask

    # Find bounding box to limit voxel testing (performance optimization)
    z_min, y_min, x_min = points.min(axis=0).astype(int)
    z_max, y_max, x_max = points.max(axis=0).astype(int)

    Z, Y, X = image_shape
    z_min = max(0, z_min)
    z_max = min(Z, z_max + 1)
    y_min = max(0, y_min)
    y_max = min(Y, y_max + 1)
    x_min = max(0, x_min)
    x_max = min(X, x_max + 1)

    # Test all voxels in bounding box
    # Note: find_simplex returns -1 if point is outside convex hull
    for z in range(z_min, z_max):
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if delaunay.find_simplex([z, y, x]) >= 0:
                    mask[z, y, x] = True

    return mask


def grow_lesion_mask(
    image: Union["ImageData", np.ndarray],
    seed_point: Tuple[int, int, int],
    method: Literal["intensity", "radial_gradient"] = "intensity",
    intensity_tolerance: float = 0.20,
    background_mean: Optional[float] = None,
    max_size: Optional[int] = None,
    connectivity: int = 3,
    # --- Radial-gradient options (new method - Mikell et al. 2018) ---
    gradient_sigma: float = 0.1,
    max_ray_length: float = 25.0,
    min_fraction_of_seed: float = 0.42,
) -> np.ndarray:
    """
    Grow a lesion mask from a seed point.

    Methods:
      - method='intensity' : Fixed threshold connected component within (1±tolerance)*seed_value,
                             with optional max_size truncation by closest-to-seed intensity.
                             Simple and fast, but uses arbitrary threshold.

      - method='radial_gradient' : Radial profile/ray-casting method (Mikell et al., 2018).
                                   Casts rays from seed point, finds max gradient along each ray
                                   to define lesion boundary, then fills using Delaunay triangulation.
                                   More anatomically accurate but assumes star-convex lesions.
                                   FAILS for U-shaped, necrotic, or complex geometries.

    Parameters
    ----------
    image : ImageData or ndarray
        3D intensity image
    seed_point : tuple of int
        Seed voxel coordinates (z, y, x)
    method : {'intensity', 'radial_gradient'}
        Segmentation method to use
    intensity_tolerance : float
        For 'intensity' method: fractional tolerance (e.g., 0.20 = ±20%)
    background_mean : float, optional
        For 'intensity' method: background intensity for relative thresholding
    max_size : int, optional
        Maximum number of voxels (truncates to closest voxels if exceeded)
    connectivity : int
        Connectivity for 'intensity' method (1=6-neighbor, 2=18, 3=26)
    gradient_sigma : float
        For 'radial_gradient': Gaussian smoothing sigma (default 0.1 for minimal smoothing).
        WARNING: Excessive smoothing causes gradients to shift outward, producing oversized masks.
        Use minimal smoothing for pre-smoothed reconstructions (Q.Clear, BSREM, etc.)
    max_ray_length : float
        For 'radial_gradient': Maximum ray search distance in mm (safety limit, default 50.0)
    min_fraction_of_seed : float
        For 'radial_gradient': Stop ray if intensity < fraction * seed_value (default 0.42 = 42%, PET literature standard)

    Returns
    -------
    mask : ndarray (bool)
        Binary lesion mask

    References
    ----------
    Mikell et al. (2018). Impact of 90Y PET gradient-based tumor segmentation
    on voxel-level dosimetry in liver radioembolization.
    """
    # Convert to numpy array if needed
    img_arr = image.as_array() if hasattr(image, "as_array") else np.asarray(image)
    if img_arr.ndim != 3:
        raise ValueError("image must be a 3D array")

    z0, y0, x0 = map(int, seed_point)
    Z, Y, X = img_arr.shape
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        raise ValueError("seed_point is out of bounds")

    seed_value = float(img_arr[z0, y0, x0])
    if not np.isfinite(seed_value):
        raise ValueError("Seed voxel value is not finite")

    # ---- Old behaviour (unchanged) -----------------------------------------
    if method == "intensity":
        if background_mean is not None:
            # Threshold relative to background: keep voxels above background + (1 - drop_frac)*(seed - background)
            # e.g. tol=0.10 -> 90% of (seed - background); tol=0.30 -> 70% of (seed - background)
            drop_frac = float(intensity_tolerance)
            lower_bound = float(background_mean) + (1.0 - drop_frac) * (seed_value - float(background_mean))
        else:
            lower_bound = seed_value * (1.0 - intensity_tolerance)
        upper_bound = seed_value * (1.0 + intensity_tolerance)

        in_range = (img_arr >= lower_bound) & (img_arr <= upper_bound)

        struct = ndimage.generate_binary_structure(3, connectivity)
        labeled, _ = ndimage.label(in_range, structure=struct)

        seed_label = labeled[z0, y0, x0]
        mask = (labeled == seed_label)

        if max_size is not None and int(mask.sum()) > max_size:
            mask_coords = np.argwhere(mask)
            mask_values = img_arr[mask]
            intensity_diff = np.abs(mask_values - seed_value)
            sorted_indices = np.argsort(intensity_diff)[:max_size]

            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[tuple(mask_coords[sorted_indices].T)] = True
            mask = new_mask

        return mask

    # ---- Radial gradient method (Mikell et al. 2018) ----------------------
    # CRITICAL ASSUMPTION: Lesion is star-convex from seed point.
    # Will FAIL for U-shaped, necrotic, or complex geometries.
    # Use local_background method for unknown/irregular shapes.
    if method == "radial_gradient":
        from scipy.ndimage import gaussian_filter

        # Apply Gaussian smoothing (minimal for pre-smoothed data like Q.Clear)
        # WARNING: Excessive smoothing blurs boundaries and shifts gradient maxima outward
        sigma = float(gradient_sigma)
        img_smoothed = gaussian_filter(img_arr.astype(np.float32), sigma=sigma)

        # Compute gradient components on smoothed image
        # gz, gy, gx are the gradient vectors needed to determine gradient DIRECTION
        gz, gy, gx = np.gradient(img_smoothed, edge_order=1)

        # Get voxel sizes for anisotropic handling
        voxel_sizes = (1.0, 1.0, 1.0)
        if hasattr(image, 'voxel_sizes'):
            voxel_sizes = image.voxel_sizes()

        # Generate ray directions using Fibonacci sphere
        directions = generate_fibonacci_sphere_directions(n_samples=162)

        # Cast rays and collect boundary points
        boundary_points = []
        max_length = float(max_ray_length)
        min_frac = float(min_fraction_of_seed)

        for direction in directions:
            # Pass gradient components (gz, gy, gx) to enable directional gradient checking
            # This ensures we only accept NEGATIVE gradients (intensity decreasing outward)
            pt = cast_ray_and_find_max_gradient(
                img_arr, gz, gy, gx, (z0, y0, x0), direction,
                max_length, min_frac, seed_value, voxel_sizes, background_mean
            )
            if pt is not None:
                boundary_points.append(pt)

        # Check if we have enough valid rays
        logger = logging.getLogger("grow_lesions") if logging.getLogger("grow_lesions").handlers else logging.getLogger(__name__)
        if len(boundary_points) < 10:
            logger.warning(
                f"Radial gradient found only {len(boundary_points)} valid rays (< 10); "
                f"using seed-only mask. Lesion may be too small, irregular, or seed point may be suboptimal."
            )
            mask = np.zeros_like(img_arr, dtype=bool)
            mask[z0, y0, x0] = True
            return mask

        # Construct mask using Delaunay triangulation
        mask = construct_mask_delaunay(boundary_points, img_arr.shape, (z0, y0, x0))

        # Optional: respect max_size constraint
        if max_size is not None and int(mask.sum()) > max_size:
            # Truncate to voxels closest to seed point
            mask_coords = np.argwhere(mask)
            distances = np.sum((mask_coords - np.array([z0, y0, x0])) ** 2, axis=1)
            sorted_indices = np.argsort(distances)[:max_size]
            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[tuple(mask_coords[sorted_indices].T)] = True
            mask = new_mask

        return mask

    raise ValueError(f"Unknown method: {method}. Supported methods: 'intensity', 'radial_gradient'")



from typing import Dict, Literal, Optional, Tuple, Union
import numpy as np

def detect_lesions_and_create_masks(
    image: Union["ImageData", np.ndarray],
    num_lesions: int = 5,
    method: Literal["intensity", "radial_gradient"] = "intensity",
    intensity_tolerance: float = 0.20,
    background_value: Optional[float] = None,
    max_size: Optional[int] = None,
    min_separation: float = 30.0,
    background_percentile: float = 50.0,
    voxel_sizes: Optional[Tuple[float, float, float]] = None,
    connectivity: int = 1,
    return_as_imagedata: bool = True,
    # --- radial_gradient options ---
    gradient_sigma: float = 1.0,
    max_ray_length: float = 25.0,
    min_fraction_of_seed: float = 0.42,
) -> Dict[str, Union["ImageData", np.ndarray]]:
    """
    Detect the M hottest lesions (by local maxima) and create one mask per lesion.

    This function combines:
      1) lesion centre detection via `find_hottest_lesions`, and
      2) per-lesion region growing via `grow_lesion_mask`.

    Parameters
    ----------
    image : ImageData or ndarray
        3D image to analyse.
    num_lesions : int
        Number of lesion seeds to detect.
    method : {'intensity', 'size', 'gradient', 'local_background'}
        Mask-growing strategy (passed to `grow_lesion_mask`):

        - 'intensity' :
            Connected component of voxels within
            [(1 - intensity_tolerance)*I_seed, (1 + intensity_tolerance)*I_seed],
            optionally truncated to `lesion_size` voxels closest in intensity to I_seed.
        - 'size' :
            Within the connected component above 0.5*I_seed, select `lesion_size` voxels
            whose intensities are closest to I_seed (requires `lesion_size`).
        - 'gradient' :
            Breadth-first region growing stopped by a local edge criterion based on the
            gradient magnitude |∇I|, with an additional minimum intensity fraction of the seed.
        - 'local_background' :
            Iterative shell-based growth: at each iteration include shell voxels with
            I >= μ_shell + k_sigma·σ_shell (after outlier exclusion), plus a minimum seed fraction.

    intensity_tolerance : float
        Used only when method='intensity'. Fractional tolerance around the seed value.
        Example: 0.20 corresponds to ±20%.
    lesion_size : int, optional
        Used as:
          - method='size'        : required; number of voxels to select.
          - method='intensity'   : optional maximum size cap (keeps closest-to-seed intensities).
          - method in {'gradient','local_background'} : optional hard cap on region size.
    min_separation : float
        Minimum distance (mm) between lesion centres (passed to `find_hottest_lesions`).
    background_percentile : float
        Percentile threshold used to suppress background when detecting hotspots
        (passed to `find_hottest_lesions`).
    voxel_sizes : tuple of float, optional
        Voxel sizes in mm as (z, y, x). If `image` is ImageData, this may be inferred
        by `find_hottest_lesions`.
    connectivity : int
        Connectivity for connected components / neighbourhoods: 1, 2, or 3 for
        6-, 18-, or 26-connected neighbourhoods.
    return_as_imagedata : bool
        If True and `image` is ImageData-like, return masks as ImageData objects
        with identical geometry to `image`. Otherwise return numpy boolean arrays.

    Gradient options (method='gradient')
    -----------------------------------
    edge_percentile : float
        Edge threshold defined as the given percentile of |∇I| within a local ROI
        around the seed. Larger values are more permissive (larger regions).
    min_fraction_of_seed : float
        Exclude candidates with I < min_fraction_of_seed * I_seed.
    local_roi_radius : int
        Radius (voxels) of the local ROI used to compute the edge_percentile threshold.

    Local-background options (method='local_background')
    ----------------------------------------------------
    k_sigma : float
        Shell inclusion threshold: I >= μ_shell + k_sigma·σ_shell.
        Larger values are more conservative (smaller regions).
    shell_max_iters : int
        Maximum number of shell-expansion iterations.
    shell_exclude_hot_percentile : float
        Robustness parameter: exclude shell voxels above this percentile when computing
        μ_shell and σ_shell (reduces influence of nearby hotspots).
    min_fraction_of_seed_bg : float
        Additional guard: exclude candidates with I < min_fraction_of_seed_bg * I_seed.

    Returns
    -------
    lesion_masks : dict
        Dictionary mapping lesion labels ('lesion_1', ..., 'lesion_M') to masks
        (ImageData or ndarray). Each mask corresponds to one detected hotspot seed.

    Notes
    -----
    - This function always returns up to `num_lesions` masks, one per detected seed.
      If fewer than `num_lesions` seeds are found by `find_hottest_lesions`, fewer masks
      will be returned.
    - For post-therapy 90Y PET, 'local_background' is typically the most robust default.
    """
    lesion_centres = find_hottest_lesions(
        image=image,
        num_lesions=num_lesions,
        min_separation=min_separation,
        background_percentile=background_percentile,
        voxel_sizes=voxel_sizes,
        background_value=background_value,
    )

    lesion_masks: Dict[str, Union["ImageData", np.ndarray]] = {}

    for i, centre in enumerate(lesion_centres, start=1):
        mask = grow_lesion_mask(
            image=image,
            seed_point=centre,
            method=method,
            intensity_tolerance=intensity_tolerance,
            background_mean=background_value,
            max_size=lesion_size,
            connectivity=connectivity,
            edge_percentile=edge_percentile,
            min_fraction_of_seed=min_fraction_of_seed,
            local_roi_radius=local_roi_radius,
            k_sigma=k_sigma,
            shell_max_iters=shell_max_iters,
            shell_exclude_hot_percentile=shell_exclude_hot_percentile,
            min_fraction_of_seed_bg=min_fraction_of_seed_bg,
        )

        if return_as_imagedata and hasattr(image, "as_array"):
            mask_img = image.clone()
            mask_img.fill(mask.astype(np.float32))
            lesion_masks[f"lesion_{i}"] = mask_img
        else:
            lesion_masks[f"lesion_{i}"] = mask

    return lesion_masks


def compute_lesion_statistics(
    image: Union[ImageData, np.ndarray],
    lesion_masks: Dict[str, Union[ImageData, np.ndarray]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each detected lesion.

    Parameters
    ----------
    image : ImageData or ndarray
        Original image
    lesion_masks : dict
        Dictionary of lesion masks from detect_lesions_and_create_masks

    Returns
    -------
    statistics : dict
        Nested dictionary with statistics for each lesion:
        - 'max': Maximum value in lesion
        - 'mean': Mean value in lesion
        - 'std': Standard deviation in lesion
        - 'volume': Number of voxels in lesion
        - 'center': Center of mass (z, y, x)
    """
    # Convert to numpy array if needed
    if hasattr(image, 'as_array'):
        img_arr = image.as_array()
    else:
        img_arr = np.asarray(image)

    statistics = {}

    for name, mask in lesion_masks.items():
        # Convert mask to numpy if needed
        if hasattr(mask, 'as_array'):
            mask_arr = mask.as_array().astype(bool)
        else:
            mask_arr = np.asarray(mask).astype(bool)

        # Extract lesion values
        lesion_values = img_arr[mask_arr]

        # Compute statistics
        stats = {
            'max': float(np.max(lesion_values)),
            'mean': float(np.mean(lesion_values)),
            'std': float(np.std(lesion_values)),
            'volume': int(np.sum(mask_arr)),
            'center': tuple(ndimage.center_of_mass(mask_arr))
        }

        statistics[name] = stats

    return statistics


def extract_lesion_metadata(
    image: Union[ImageData, np.ndarray],
    lesion_masks: Dict[str, Union[ImageData, np.ndarray]]
) -> Dict[str, Dict[str, Union[Tuple[int, int, int], Tuple[float, float, float], float, int]]]:
    """
    Capture reusable lesion characteristics from a reference image and masks.

    Parameters
    ----------
    image : ImageData or ndarray
        Reference image used to generate the masks.
    lesion_masks : dict
        Lesion masks produced on the reference image.

    Returns
    -------
    metadata : dict
        Dictionary keyed by lesion name with:
        - 'seed_point': integer voxel coordinates of peak intensity
        - 'voxel_count': number of voxels in the mask
        - 'center_of_mass': center of mass of the mask (floats)
        - 'peak_value': peak intensity value at the seed
    """
    if hasattr(image, 'as_array'):
        img_arr = image.as_array()
    else:
        img_arr = np.asarray(image)

    metadata: Dict[str, Dict[str, Union[Tuple[int, int, int], Tuple[float, float, float], float, int]]] = {}

    for name, mask in lesion_masks.items():
        if hasattr(mask, 'as_array'):
            mask_arr = mask.as_array().astype(bool)
        else:
            mask_arr = np.asarray(mask).astype(bool)

        if not np.any(mask_arr):
            continue

        coords = np.argwhere(mask_arr)
        lesion_values = img_arr[mask_arr]
        peak_idx = int(np.argmax(lesion_values))
        peak_coord = tuple(int(c) for c in coords[peak_idx])

        metadata[name] = {
            'seed_point': peak_coord,
            'voxel_count': int(mask_arr.sum()),
            'center_of_mass': tuple(ndimage.center_of_mass(mask_arr)),
            'peak_value': float(lesion_values[peak_idx])
        }

    return metadata


def regrow_lesion_masks(
    image: Union[ImageData, np.ndarray],
    lesion_metadata: Dict[str, Dict[str, Union[Tuple[int, int, int], int]]],
    method: Literal['intensity', 'size'] = 'size',
    intensity_tolerance: float = 0.20,
    background_mean: Optional[float] = None,
    connectivity: int = 3,
    return_as_imagedata: bool = True
) -> Dict[str, Union[ImageData, np.ndarray]]:
    """
    Regrow lesion masks on a target image using metadata captured from a reference.

    Parameters
    ----------
    image : ImageData or ndarray
        Target image on which masks should be regrown.
    lesion_metadata : dict
        Metadata obtained from :func:`extract_lesion_metadata`.
    method : {'intensity', 'size'}
        Region growing strategy (defaults to 'size' for fixed-voxel propagation).
    intensity_tolerance : float
        Fractional tolerance for 'intensity' method.
    connectivity : int
        Connectivity setting passed to :func:`grow_lesion_mask`.
    return_as_imagedata : bool
        If True and target is ImageData, return ImageData masks.

    Returns
    -------
    regrown_masks : dict
        Dictionary of lesion names to regrown masks in the target geometry.
    """
    regrown_masks: Dict[str, Union[ImageData, np.ndarray]] = {}

    for name, info in lesion_metadata.items():
        seed_point = info.get('seed_point')
        if seed_point is None:
            continue

        seed_point_int = tuple(int(coord) for coord in seed_point)
        voxel_count = int(info.get('voxel_count', 0))

        mask_arr = grow_lesion_mask(
            image=image,
            seed_point=seed_point_int,
            method=method,
            intensity_tolerance=intensity_tolerance,
            background_mean=background_mean,
            max_size=voxel_count if voxel_count > 0 else None,
            connectivity=connectivity
        )

        if return_as_imagedata and hasattr(image, 'clone'):
            mask_img = image.clone()
            mask_img.fill(mask_arr.astype(np.float32))
            regrown_masks[name] = mask_img
        else:
            regrown_masks[name] = mask_arr

    return regrown_masks
