"""
Image loading using SIRF.STIR ImageData.

This module provides a wrapper around SIRF.STIR's ImageData class for loading
and working with Interfile images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

try:
    import sirf.STIR as pet
    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False
    pet = None


def load_image(path: Union[str, Path]):
    """
    Load an image using SIRF.STIR ImageData.

    Args:
        path: Path to the image file (.hv header or compatible format)

    Returns:
        SIRF.STIR ImageData object

    Raises:
        ImportError: If SIRF is not installed
        RuntimeError: If the image cannot be loaded
    """
    if not HAS_SIRF:
        raise ImportError(
            "SIRF.STIR is required for image loading. "
            "Install SIRF following the instructions at: "
            "https://github.com/SyneRBI/SIRF/wiki"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        image = pet.ImageData(str(path))
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {path}: {e}") from e


def get_image_array(image):
    """
    Get the underlying numpy array from a SIRF ImageData object.

    Args:
        image: SIRF.STIR ImageData object

    Returns:
        numpy array with image data
    """
    if not HAS_SIRF:
        raise ImportError("SIRF.STIR is required")

    return image.as_array()


def get_image_shape(image) -> tuple:
    """
    Get the shape of a SIRF ImageData object.

    Args:
        image: SIRF.STIR ImageData object

    Returns:
        Tuple of dimensions
    """
    return get_image_array(image).shape


def get_voxel_sizes(image) -> tuple:
    """
    Get the voxel sizes (in mm) from a SIRF ImageData object.

    Args:
        image: SIRF.STIR ImageData object

    Returns:
        Tuple of voxel sizes (z, y, x) in mm
    """
    if not HAS_SIRF:
        raise ImportError("SIRF.STIR is required")

    voxel_sizes = image.voxel_sizes()
    shape = get_image_shape(image)
    return tuple(voxel_sizes[dim] for dim in range(len(shape)))


__all__ = ["load_image", "get_image_array", "get_image_shape", "get_voxel_sizes"]
