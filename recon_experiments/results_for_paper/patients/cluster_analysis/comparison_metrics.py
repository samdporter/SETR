"""
Metrics for comparing lesion segmentation methods.

This module provides comprehensive metrics for quantitative evaluation
of different lesion segmentation approaches, with special emphasis on
gradient-based methods which require surface accuracy metrics.

Key features:
- Dice coefficient and Jaccard index (volume overlap)
- Hausdorff distance (CRITICAL for detecting boundary irregularities)
- TBR and CNR (contrast metrics)

References
----------
Mikell et al. (2018). Impact of 90Y PET gradient-based tumor segmentation
on voxel-level dosimetry in liver radioembolization.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient (DSC) between two binary masks.

    The Dice coefficient measures volumetric overlap between two masks.
    DSC = 2 * |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask

    Returns
    -------
    float
        Dice coefficient in [0, 1], where 1 = perfect overlap

    Notes
    -----
    Dice is sensitive to volume differences but insensitive to surface accuracy.
    Should be complemented with Hausdorff distance for gradient-based methods.
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * float(intersection) / float(union)


def compute_volume_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Jaccard index (Intersection over Union).

    IoU = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask

    Returns
    -------
    float
        Jaccard index in [0, 1]

    Notes
    -----
    More conservative than Dice (lower values for same overlap).
    Jaccard = Dice / (2 - Dice)
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def compute_hausdorff_distance(
    mask1: np.ndarray,
    mask2: np.ndarray,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    percentile: float = 95.0
) -> float:
    """
    Compute Hausdorff distance between mask boundaries (in mm).

    *** CRITICAL METRIC FOR GRADIENT-BASED METHODS ***

    Gradient-based segmentation is prone to "spiking" (boundary irregularities)
    due to noise amplification. Volume metrics (Dice/Jaccard) average out these
    surface errors. Hausdorff distance exposes them by measuring the maximum
    distance from any boundary point to the nearest point in the other mask.

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask
    voxel_sizes : tuple of float
        Voxel dimensions in mm as (vz, vy, vx)
    percentile : float
        Use percentile HD to reduce sensitivity to outliers (default 95.0)
        95% HD is more robust than max HD (100%)

    Returns
    -------
    float
        Hausdorff distance in mm (lower is better)
        Returns inf if masks don't have valid boundaries

    Notes
    -----
    - 95% HD < 5mm: excellent boundary agreement
    - 95% HD < 10mm: acceptable for most applications
    - 95% HD > 15mm: significant boundary discrepancy, investigate cause

    This metric is essential for validating radial_gradient method, which
    uses sparse boundary points (162 rays) that could introduce irregularities.

    References
    ----------
    Huttenlocher et al. (1993). Comparing images using the Hausdorff distance.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Extract boundaries via erosion
    boundary1 = mask1 & ~binary_erosion(mask1)
    boundary2 = mask2 & ~binary_erosion(mask2)

    if not np.any(boundary1) or not np.any(boundary2):
        return np.inf

    # Distance transforms with anisotropic voxel sizes
    dist1 = distance_transform_edt(~boundary1, sampling=voxel_sizes)
    dist2 = distance_transform_edt(~boundary2, sampling=voxel_sizes)

    # Get distances from each boundary to nearest point in other mask
    distances_1_to_2 = dist1[boundary2]
    distances_2_to_1 = dist2[boundary1]

    # Compute percentile Hausdorff (more robust than max)
    hd1 = np.percentile(distances_1_to_2, percentile)
    hd2 = np.percentile(distances_2_to_1, percentile)

    return float(max(hd1, hd2))


def compute_tbr(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    background_mask: np.ndarray
) -> float:
    """
    Compute tumor-to-background ratio (TBR).

    TBR = mean(lesion intensity) / mean(background intensity)

    Parameters
    ----------
    image : np.ndarray
        3D intensity image
    lesion_mask : np.ndarray
        Binary lesion mask
    background_mask : np.ndarray
        Binary background region mask

    Returns
    -------
    float
        TBR value (higher = better contrast)
        Returns nan if masks are empty

    Notes
    -----
    Higher TBR indicates better lesion detectability.
    Typical values for Y-90 PET: 5-20 for hot lesions.
    """
    lesion_vals = image[lesion_mask.astype(bool)]
    bg_vals = image[background_mask.astype(bool)]

    if len(lesion_vals) == 0 or len(bg_vals) == 0:
        return np.nan

    return float(lesion_vals.mean()) / float(bg_vals.mean())


def compute_contrast_to_noise_ratio(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    background_mask: np.ndarray
) -> float:
    """
    Compute contrast-to-noise ratio (CNR).

    CNR = |mean(lesion) - mean(background)| / std(background)

    Parameters
    ----------
    image : np.ndarray
        3D intensity image
    lesion_mask : np.ndarray
        Binary lesion mask
    background_mask : np.ndarray
        Binary background region mask

    Returns
    -------
    float
        CNR value (higher = better detectability)
        Returns nan if masks are empty or background has zero variance

    Notes
    -----
    CNR incorporates noise (via background std) unlike TBR.
    Rose criterion: CNR > 5 for reliable detection.
    """
    lesion_vals = image[lesion_mask.astype(bool)]
    bg_vals = image[background_mask.astype(bool)]

    if len(lesion_vals) == 0 or len(bg_vals) == 0:
        return np.nan

    bg_std = bg_vals.std()
    if bg_std == 0:
        return np.nan

    return float(abs(lesion_vals.mean() - bg_vals.mean())) / bg_std


def compare_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    image: Optional[np.ndarray] = None,
    background_mask: Optional[np.ndarray] = None,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    label1: str = "method1",
    label2: str = "method2"
) -> Dict[str, float]:
    """
    Comprehensive comparison of two segmentation masks.

    Computes all available metrics between two masks. If image and background
    mask are provided, also computes contrast metrics (TBR, CNR).

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask
    image : np.ndarray, optional
        Original intensity image (required for TBR/CNR)
    background_mask : np.ndarray, optional
        Background region mask (required for TBR/CNR)
    voxel_sizes : tuple of float
        Voxel dimensions in mm as (vz, vy, vx)
    label1 : str
        Label for first method (used in output keys)
    label2 : str
        Label for second method (used in output keys)

    Returns
    -------
    dict
        Dictionary of comparison metrics:
        - '{label1}_volume': number of voxels in mask1
        - '{label2}_volume': number of voxels in mask2
        - 'dice': Dice coefficient
        - 'jaccard': Jaccard index
        - 'hausdorff_95_mm': 95% Hausdorff distance (mm)
        - 'volume_ratio': ratio of mask1/mask2 volumes
        - '{label1}_tbr': TBR for mask1 (if image provided)
        - '{label2}_tbr': TBR for mask2 (if image provided)
        - '{label1}_cnr': CNR for mask1 (if image provided)
        - '{label2}_cnr': CNR for mask2 (if image provided)

    Examples
    --------
    >>> metrics = compare_masks(
    ...     intensity_mask, radial_gradient_mask,
    ...     image=vendor_image,
    ...     background_mask=bg_mask,
    ...     voxel_sizes=(3.27, 2.13, 2.13),
    ...     label1='intensity',
    ...     label2='radial_gradient'
    ... )
    >>> print(f"Dice: {metrics['dice']:.3f}")
    >>> print(f"95% HD: {metrics['hausdorff_95_mm']:.1f} mm")
    """
    metrics = {
        f'{label1}_volume': int(mask1.sum()),
        f'{label2}_volume': int(mask2.sum()),
        'dice': compute_dice_coefficient(mask1, mask2),
        'jaccard': compute_volume_overlap(mask1, mask2),
        'hausdorff_95_mm': compute_hausdorff_distance(mask1, mask2, voxel_sizes, percentile=95.0),
        'volume_ratio': float(mask1.sum()) / float(mask2.sum()) if mask2.sum() > 0 else np.inf,
    }

    if image is not None and background_mask is not None:
        metrics.update({
            f'{label1}_tbr': compute_tbr(image, mask1, background_mask),
            f'{label2}_tbr': compute_tbr(image, mask2, background_mask),
            f'{label1}_cnr': compute_contrast_to_noise_ratio(image, mask1, background_mask),
            f'{label2}_cnr': compute_contrast_to_noise_ratio(image, mask2, background_mask),
        })

    return metrics
