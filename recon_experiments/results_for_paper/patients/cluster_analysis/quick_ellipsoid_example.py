"""
Quick example: Create ellipsoid ROIs and compute CoV/Mean

This shows the correct way to:
1. Create an ellipsoid
2. Get it as a SIRF ImageData mask (1s and 0s)
3. Apply it to an image
4. Compute statistics (mean, CoV)
"""

import sys
sys.path.append("/home/storage")

from cluster_analysis.jupyter_utils import SpatialMask
from cluster_analysis.interfile import load_image
import numpy as np

# Load your image
image = load_image("path/to/your/image.hv")
image_array = image.as_array()

print(f"Image shape: {image_array.shape}")

# Method 1: Create ellipsoid programmatically
# ============================================

ellipsoid_mask = SpatialMask(
    shape=image_array.shape,
    rois=[{
        "type": "ellipsoid",  # 3D ellipsoid
        "center_x": 64,       # Center coordinates
        "center_y": 64,
        "center_z": 32,
        "radius_x": 20,       # Radii in each direction
        "radius_y": 15,
        "radius_z": 10
    }],
    name="tumor_region"
)

# Convert to SIRF ImageData mask (THIS IS THE KEY STEP!)
mask_image = ellipsoid_mask.to_image_mask(image)

# Now mask_image is a SIRF ImageData with:
# - 1.0 inside the ellipsoid
# - 0.0 outside the ellipsoid

print(f"Mask created: {mask_image}")
print(f"Mask shape: {mask_image.as_array().shape}")

# Apply the mask
masked_image = image * mask_image

# Compute statistics
masked_array = masked_image.as_array()
mask_array = mask_image.as_array()

# Get values inside ROI (where mask == 1)
roi_values = masked_array[mask_array > 0]

mean = roi_values.mean()
std = roi_values.std()
cov = std / mean

print("\nStatistics in ellipsoid ROI:")
print(f"  Mean: {mean:.6f}")
print(f"  Std:  {std:.6f}")
print(f"  CoV:  {cov:.6f}")

# Save the mask for reuse
ellipsoid_mask.save("masks/tumor_ellipsoid.json")
print("\nMask saved to masks/tumor_ellipsoid.json")

# You can also save the SIRF mask image itself
mask_image.write("masks/tumor_ellipsoid_mask.hv")
print("SIRF mask image saved to masks/tumor_ellipsoid_mask.hv")


# Method 2: Load and reuse mask
# ==============================

print("\n" + "="*50)
print("Reusing saved mask on another image")
print("="*50)

# Load the mask
reloaded_mask = SpatialMask.load("masks/tumor_ellipsoid.json")

# Apply to the same or different image
another_image = load_image("path/to/another/image.hv")
another_mask_image = reloaded_mask.to_image_mask(another_image)
another_masked = another_image * another_mask_image

# Compute stats
arr = another_masked.as_array()
mask_arr = another_mask_image.as_array()
values = arr[mask_arr > 0]

print(f"Mean: {values.mean():.6f}")
print(f"CoV:  {values.std() / values.mean():.6f}")


# Method 3: Interactive in Jupyter
# =================================

"""
In a Jupyter notebook with %matplotlib widget:

from cluster_analysis.jupyter_utils import view_image

# View and draw interactively
viewer = view_image("image.hv", figsize=(14, 10))

# Draw ellipse with buttons, then:
mask_image = viewer.get_mask_image()  # Get SIRF ImageData mask!
stats = viewer.get_stats()             # Get statistics dict

print(f"Mean: {stats['mean']}")
print(f"CoV: {stats['coefficient_of_variation']}")

# Apply to current image
masked = viewer.get_masked_image()

# Save for reuse
spatial_mask = viewer.get_spatial_mask(name="my_ellipsoid")
spatial_mask.save("masks/my_ellipsoid.json")
"""
