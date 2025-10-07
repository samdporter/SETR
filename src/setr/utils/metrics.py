"""
Image quality metrics for reconstruction evaluation.

Provides functions to compute various error metrics (MSE, RMSE, NRMSE, MAE, etc.)
between reconstructed images and reference images, with optional masking support.
"""

import numpy as np
from sirf.STIR import ImageData

from setr.utils.sirf import get_array


def compute_mse(image, reference, mask=None):
    """
    Compute Mean Squared Error between image and reference.

    Args:
        image: ImageData or numpy array
        reference: ImageData or numpy array (same shape as image)
        mask: Optional ImageData or numpy array (boolean or 0/1)

    Returns:
        float: MSE value
    """
    img_arr = get_array(image) if isinstance(image, ImageData) else np.asarray(image)
    ref_arr = get_array(reference) if isinstance(reference, ImageData) else np.asarray(reference)

    if img_arr.shape != ref_arr.shape:
        raise ValueError(f"Image shapes do not match: {img_arr.shape} vs {ref_arr.shape}")

    diff = img_arr - ref_arr
    squared_diff = diff**2

    if mask is not None:
        mask_arr = get_array(mask) if isinstance(mask, ImageData) else np.asarray(mask)
        mask_arr = mask_arr.astype(bool)

        if mask_arr.shape != img_arr.shape:
            raise ValueError(
                f"Mask shape {mask_arr.shape} does not match image shape {img_arr.shape}"
            )

        if not np.any(mask_arr):
            return np.nan  # Empty mask

        mse = np.mean(squared_diff[mask_arr])
    else:
        mse = np.mean(squared_diff)

    return float(mse)


def compute_rmse(image, reference, mask=None):
    """
    Compute Root Mean Squared Error between image and reference.

    Args:
        image: ImageData or numpy array
        reference: ImageData or numpy array (same shape as image)
        mask: Optional ImageData or numpy array (boolean or 0/1)

    Returns:
        float: RMSE value
    """
    mse = compute_mse(image, reference, mask)
    return float(np.sqrt(mse))


def compute_nrmse(image, reference, mask=None, normalization="range"):
    """
    Compute Normalized Root Mean Squared Error between image and reference.

    Args:
        image: ImageData or numpy array
        reference: ImageData or numpy array (same shape as image)
        mask: Optional ImageData or numpy array (boolean or 0/1)
        normalization: How to normalize RMSE. Options:
            - 'range': Divide by (max - min) of reference
            - 'max': Divide by max of reference
            - 'mean': Divide by mean of reference
            - 'euclidean': Divide by L2 norm of reference

    Returns:
        float: NRMSE value
    """
    rmse = compute_rmse(image, reference, mask)

    ref_arr = get_array(reference) if isinstance(reference, ImageData) else np.asarray(reference)

    if mask is not None:
        mask_arr = get_array(mask) if isinstance(mask, ImageData) else np.asarray(mask)
        mask_arr = mask_arr.astype(bool)
        ref_arr = ref_arr[mask_arr]

    if normalization == "range":
        norm_factor = ref_arr.max() - ref_arr.min()
    elif normalization == "max":
        norm_factor = ref_arr.max()
    elif normalization == "mean":
        norm_factor = ref_arr.mean()
    elif normalization == "euclidean":
        norm_factor = np.linalg.norm(ref_arr)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return np.nan if norm_factor == 0 else float(rmse / norm_factor)


def compute_mae(image, reference, mask=None):
    """
    Compute Mean Absolute Error between image and reference.

    Args:
        image: ImageData or numpy array
        reference: ImageData or numpy array (same shape as image)
        mask: Optional ImageData or numpy array (boolean or 0/1)

    Returns:
        float: MAE value
    """
    img_arr = get_array(image) if isinstance(image, ImageData) else np.asarray(image)
    ref_arr = get_array(reference) if isinstance(reference, ImageData) else np.asarray(reference)

    if img_arr.shape != ref_arr.shape:
        raise ValueError(f"Image shapes do not match: {img_arr.shape} vs {ref_arr.shape}")

    diff = np.abs(img_arr - ref_arr)

    if mask is not None:
        mask_arr = get_array(mask) if isinstance(mask, ImageData) else np.asarray(mask)
        mask_arr = mask_arr.astype(bool)

        if mask_arr.shape != img_arr.shape:
            raise ValueError(
                f"Mask shape {mask_arr.shape} does not match image shape {img_arr.shape}"
            )

        if not np.any(mask_arr):
            return np.nan

        mae = np.mean(diff[mask_arr])
    else:
        mae = np.mean(diff)

    return float(mae)


def compute_nmae(image, reference, mask=None, normalization="range"):
    """
    Compute Normalized Mean Absolute Error between image and reference.

    Args:
        image: ImageData or numpy array
        reference: ImageData or numpy array (same shape as image)
        mask: Optional ImageData or numpy array (boolean or 0/1)
        normalization: How to normalize MAE (see compute_nrmse for options)

    Returns:
        float: NMAE value
    """
    mae = compute_mae(image, reference, mask)

    ref_arr = get_array(reference) if isinstance(reference, ImageData) else np.asarray(reference)

    if mask is not None:
        mask_arr = get_array(mask) if isinstance(mask, ImageData) else np.asarray(mask)
        mask_arr = mask_arr.astype(bool)
        ref_arr = ref_arr[mask_arr]

    if normalization == "range":
        norm_factor = ref_arr.max() - ref_arr.min()
    elif normalization == "max":
        norm_factor = ref_arr.max()
    elif normalization == "mean":
        norm_factor = ref_arr.mean()
    elif normalization == "euclidean":
        norm_factor = np.linalg.norm(ref_arr)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return np.nan if norm_factor == 0 else float(mae / norm_factor)


def compute_all_metrics(image, reference, mask=None, normalization="range"):
    """
    Compute all common metrics between image and reference.

    Args:
        image: ImageData or numpy array
        reference: ImageData or numpy array (same shape as image)
        mask: Optional ImageData or numpy array (boolean or 0/1)
        normalization: How to normalize metrics (see compute_nrmse for options)

    Returns:
        dict: Dictionary with keys 'mse', 'rmse', 'nrmse', 'mae', 'nmae'
    """
    return {
        "mse": compute_mse(image, reference, mask),
        "rmse": compute_rmse(image, reference, mask),
        "nrmse": compute_nrmse(image, reference, mask, normalization),
        "mae": compute_mae(image, reference, mask),
        "nmae": compute_nmae(image, reference, mask, normalization),
    }


def compute_block_metrics(block_image, block_reference, mask=None, normalization="range"):
    """
    Compute metrics for BlockDataContainer (multi-modal images).

    Computes metrics separately for each container (modality).

    Args:
        block_image: BlockDataContainer with reconstructed images
        block_reference: BlockDataContainer with reference images (same structure)
        mask: Optional mask (single image or BlockDataContainer matching structure)
        normalization: How to normalize metrics

    Returns:
        dict: Nested dictionary with structure:
            {
                'modality_0': {'mse': ..., 'rmse': ..., 'nrmse': ..., 'mae': ..., 'nmae': ...},
                'modality_1': {...},
                ...
            }
    """
    if len(block_image.containers) != len(block_reference.containers):
        raise ValueError(
            f"Block containers must have same length: "
            f"{len(block_image.containers)} vs {len(block_reference.containers)}"
        )

    metrics = {}

    for i, (img, ref) in enumerate(zip(block_image.containers, block_reference.containers)):
        # Handle mask
        if mask is None:
            mask_i = None
        elif hasattr(mask, "containers"):  # BlockDataContainer
            mask_i = mask.containers[i]
        else:  # Single mask for all modalities
            mask_i = mask

        metrics[f"modality_{i}"] = compute_all_metrics(img, ref, mask_i, normalization)

    return metrics


def create_mask_from_threshold(image, threshold=0.0, mode="greater"):
    """
    Create binary mask from image based on threshold.

    Args:
        image: ImageData or numpy array
        threshold: Threshold value
        mode: 'greater', 'less', 'greater_equal', 'less_equal'

    Returns:
        ImageData or numpy array (same type as input) with boolean mask
    """
    is_imagedata = isinstance(image, ImageData)
    arr = get_array(image) if is_imagedata else np.asarray(image)

    if mode == "greater":
        mask_arr = arr > threshold
    elif mode == "less":
        mask_arr = arr < threshold
    elif mode == "greater_equal":
        mask_arr = arr >= threshold
    elif mode == "less_equal":
        mask_arr = arr <= threshold
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not is_imagedata:
        return mask_arr
    mask_image = image.clone()
    mask_image.fill(mask_arr.astype(np.float32))
    return mask_image
