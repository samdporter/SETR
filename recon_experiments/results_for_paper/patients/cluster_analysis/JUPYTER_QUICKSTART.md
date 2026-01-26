# Jupyter Quick Start Guide

This guide will get you up and running with interactive image analysis in Jupyter notebooks.

## Setup

### 1. Install Jupyter Dependencies

```bash
cd cluster_analysis
pip install -r requirements-jupyter.txt
```

### 2. Launch JupyterLab

```bash
jupyter lab
```

### 3. Enable Interactive Widgets

In your first notebook cell, run:

```python
%matplotlib widget
```

This enables interactive matplotlib features.

## Basic Workflow

### Step 1: Load and View an Image

```python
from cluster_analysis.jupyter_utils import view_image

# View an Interfile image
viewer = view_image("path/to/image.hv", figsize=(14, 10))
```

### Step 2: Create ROIs Interactively

**Rectangle ROI:**
1. Click the "Rectangle ROI" button
2. Click once at one corner
3. Click again at the opposite corner
4. Rectangle is automatically created

**Polygon ROI:**
1. Click the "Polygon ROI" button
2. Click multiple times to add vertices
3. Press **Enter** to complete the polygon
4. Press **Esc** to cancel

**Navigate Slices (3D only):**
- Use the slice slider to move through the volume
- ROIs are associated with specific slices

### Step 3: Save Your Mask

```python
# Get the mask from the viewer
mask = viewer.get_spatial_mask(name="tumor_region")

# Save to file
mask.save("masks/tumor_region.json")
```

### Step 4: Compute Statistics

```python
from cluster_analysis.stats import compute_basic_stats

# Get flat mask
flat_mask = viewer.get_flat_mask()

# Compute statistics
stats = compute_basic_stats(viewer.image.values, flat_mask)

print(f"Mean: {stats['mean']:.4f}")
print(f"Std Dev: {stats['std']:.4f}")
print(f"Coefficient of Variation: {stats['coefficient_of_variation']:.4f}")
```

### Step 5: Apply to Multiple Images

```python
from cluster_analysis.jupyter_utils import apply_mask_batch, SpatialMask
from pathlib import Path

# Load saved mask
mask = SpatialMask.load("masks/tumor_region.json")

# Find all images
image_paths = list(Path("cluster_sweeps_new").rglob("final_image_0.hv"))

# Define processing function
def analyze(image, flat_mask):
    stats = compute_basic_stats(image.values, flat_mask)
    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "cv": stats["coefficient_of_variation"]
    }

# Process all images
results = apply_mask_batch(image_paths[:10], mask, callback=analyze)

# View as DataFrame
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

## Working with SIRF

If you have SIRF installed:

```python
from cluster_analysis.jupyter_utils import view_sirf_image
import sirf.STIR as pet

# Load SIRF ImageData
sirf_image = pet.ImageData("path/to/image.hv")

# View interactively (same controls as before)
viewer = view_sirf_image(sirf_image, figsize=(14, 10))

# After creating ROIs, apply mask to SIRF image
masked_sirf = viewer.apply_mask_to_sirf()

# The masked SIRF image can be used in SIRF workflows
masked_sirf.write("masked_image.hv")
```

## Advanced Usage

### Loading Existing Masks

```python
from cluster_analysis.jupyter_utils import SpatialMask

# Load a mask
mask = SpatialMask.load("masks/my_roi.json")

# Inspect ROIs
print(f"Mask contains {len(mask.rois)} ROIs")
for i, roi in enumerate(mask.rois):
    print(f"  ROI {i}: {roi['type']}")
```

### Manual Mask Creation

```python
# Create a mask programmatically
mask = SpatialMask(
    shape=(128, 128, 64),  # image dimensions
    rois=[
        {
            "type": "rectangle",
            "x_min": 40, "y_min": 40,
            "x_max": 80, "y_max": 80,
            "z_min": 0, "z_max": 64
        },
        {
            "type": "sphere",
            "center_x": 64, "center_y": 64, "center_z": 32,
            "radius": 20
        }
    ],
    name="manual_mask"
)

# Save for later use
mask.save("masks/manual_mask.json")
```

### Integrating with Parameter Sweeps

```python
from cluster_analysis import analyse_parameter_grid, AnalysisConfig
from cluster_analysis.masking import MaskConfig

# Use automatic masking for full sweep
config = AnalysisConfig(
    root=Path("cluster_sweeps_new/my_sweep"),
    mask=MaskConfig(method="fraction", fraction=0.5)
)

image_results, aggregate_results = analyse_parameter_grid(config)

# Then use custom spatial mask for detailed analysis on best parameters
best_alpha, best_beta = 0.5, 0.01
best_image_path = f"cluster_sweeps_new/my_sweep/alpha_{best_alpha}_beta_{best_beta}/final_image_0.hv"

viewer = view_image(best_image_path)
# ... create custom ROI ...
```

## Tips

1. **Save masks frequently**: Don't lose your carefully drawn ROIs!

2. **Use descriptive names**: `tumor_region.json` is better than `mask1.json`

3. **Test on one image first**: Before batch processing, verify your mask works correctly

4. **Adjust figure size**: Use `figsize=(width, height)` to make the viewer larger or smaller

5. **Use keyboard shortcuts**:
   - **Enter**: Complete polygon
   - **Esc**: Cancel drawing

6. **Multiple ROIs**: You can create multiple ROIs on different slices or combine them

7. **Export for publications**: Masks are saved as JSON and can be version controlled

## Troubleshooting

**Problem**: Widgets not showing up
- **Solution**: Make sure you ran `%matplotlib widget` at the start
- Try: `pip install ipympl`

**Problem**: "Module not found" errors
- **Solution**: Install Jupyter dependencies: `pip install -r requirements-jupyter.txt`

**Problem**: SIRF import fails
- **Solution**: SIRF must be installed separately. It's optional - you can still use Interfile images

**Problem**: Mask doesn't align with image
- **Solution**: Ensure the mask shape matches the image shape. Check with `print(mask.shape)`

## Next Steps

- See [example_notebook.ipynb](example_notebook.ipynb) for a complete tutorial
- Read [README.md](README.md) for full documentation
- Explore the CLI for batch processing: `python -m cluster_analysis --help`
