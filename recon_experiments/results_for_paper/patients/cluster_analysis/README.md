# Cluster Analysis Helpers

This package provides a comprehensive toolkit for analyzing PET reconstruction
parameter sweeps using SIRF.STIR ImageData. It offers both command-line tools
and interactive Jupyter notebook capabilities for viewing images and creating custom masks.

## Features

### Core Analysis (Requires SIRF)
- Image loading using SIRF.STIR ImageData
- Several mask strategies (fractional, percentile, absolute, non-zero)
- Numerically stable voxel statistics (mean, variance, standard deviation,
  coefficient of variation, ℓ2 norm)
- Simple SVG L-curve plotter relying only on the standard library
- Command line interface for batch processing

### Interactive Jupyter Support (Optional)
- **Interactive image viewer** with slice navigation for 3D volumes
- **Interactive mask creation** with polygon and rectangle drawing tools
- **Spatial mask persistence** - save and load masks as JSON
- **Batch processing** - apply custom masks across multiple images
- **Live statistics** - compute metrics on custom ROIs interactively

## Installation

### Prerequisites

**SIRF must be installed first.** Follow the installation instructions at:
https://github.com/SyneRBI/SIRF/wiki

The package requires `sirf.STIR` for image loading.

### Jupyter Support (Optional)

For interactive Jupyter notebook features:

```bash
pip install -r requirements-jupyter.txt
```

This installs: `numpy`, `matplotlib`, `ipympl`, `pandas`, and `jupyterlab`

## Quick Start

### Command Line Interface

```bash
python -m cluster_analysis \
  --root cluster_sweeps_new/2bpos_alpha_beta_sirt2 \
  --images final_image_0.hv final_image_1.hv \
  --mask-method fraction \
  --fraction 0.5 \
  --output analysis_results/sirt2
```

This will generate:

- `analysis_results/sirt2/voxel_metrics.csv` – per-image stats
- `analysis_results/sirt2/parameter_metrics.csv` – aggregated stats per (α, β)
- `analysis_results/sirt2/l_curve.svg` – L-curve-like scatter plot

Use `--mask-method percentile` (with `--percentile`) or `--mask-method absolute`
(with `--absolute-threshold`) to switch the ROI definition.

Disable logarithmic axes via `--no-log` if you prefer linear scaling.

### Interactive Jupyter Notebooks

See [example_notebook.ipynb](example_notebook.ipynb) for a complete walkthrough.

#### Basic Usage

```python
from cluster_analysis.jupyter_utils import view_image, SpatialMask
from pathlib import Path

# View an image interactively
viewer = view_image("path/to/image.hv", figsize=(14, 10))

# Draw ROIs using the interactive tools:
# - Click "Rectangle ROI" and click twice to define corners
# - Click "Polygon ROI" and click to add points, press Enter to finish

# Save your mask
mask = viewer.get_spatial_mask(name="my_roi")
mask.save("masks/my_roi.json")

# Apply mask to get statistics
from cluster_analysis.stats import compute_basic_stats
flat_mask = mask.to_flat_mask()
stats = compute_basic_stats(viewer.data.ravel().tolist(), flat_mask)
print(f"Mean in ROI: {stats['mean']:.4f}")
```

#### Working with SIRF.STIR ImageData

```python
from cluster_analysis.jupyter_utils import view_image
import sirf.STIR as pet

# Load and view a SIRF ImageData object
sirf_image = pet.ImageData("path/to/image.hv")
viewer = view_image(sirf_image, figsize=(14, 10))

# Create mask and apply to SIRF image
masked_sirf_image = viewer.get_masked_image()
masked_sirf_image.write("masked_image.hv")
```

#### Batch Processing with Custom Masks

```python
from cluster_analysis.jupyter_utils import apply_mask_batch, SpatialMask
from cluster_analysis.stats import compute_basic_stats
from cluster_analysis.interfile import get_image_array

# Load a saved mask
mask = SpatialMask.load("masks/my_roi.json")

# Apply to multiple images
image_paths = list(Path("cluster_sweeps_new").rglob("final_image_0.hv"))

def analyze_image(image, flat_mask):
    values = get_image_array(image).ravel().tolist()
    stats = compute_basic_stats(values, flat_mask)
    return {"mean": stats["mean"], "std": stats["std"]}

results = apply_mask_batch(image_paths, mask, callback=analyze_image)
```

## Interactive Controls

When using the Jupyter viewer:

- **Slice slider** (3D only): Navigate through slices
- **Rectangle ROI**: Click twice to define opposite corners
- **Polygon ROI**: Click to add points, press **Enter** to finish
- **Clear ROIs**: Remove all ROIs
- **Undo**: Remove the last ROI
- **Esc key**: Cancel current drawing operation

## Project Structure

```
cluster_analysis/
├── __init__.py          # Main package exports
├── interfile.py         # SIRF.STIR ImageData wrapper
├── masking.py           # Automatic mask generation
├── stats.py             # Statistical computations
├── analysis.py          # High-level analysis orchestration
├── plotting.py          # SVG plotting utilities
├── jupyter_utils.py     # Interactive Jupyter tools
├── __main__.py          # CLI entry point
├── example_notebook.ipynb  # Tutorial notebook
├── requirements.txt     # Dependency information
└── requirements-jupyter.txt  # Jupyter dependencies
```

## Use Cases

1. **Quick parameter sweep analysis**: Use CLI with automatic masking
2. **Custom ROI analysis**: Use Jupyter to define precise regions interactively
3. **Batch processing**: Define a mask once, apply to many images
4. **Integration with SIRF workflows**: Seamlessly work with SIRF.STIR ImageData
5. **Publication figures**: Export masks and statistics for reproducibility

## API Examples

### Loading Images

```python
from cluster_analysis.interfile import load_image, get_image_array, get_voxel_sizes

# Load a SIRF ImageData object
image = load_image("path/to/image.hv")

# Get numpy array
array = get_image_array(image)
print(f"Shape: {array.shape}")

# Get voxel sizes
voxel_sizes = get_voxel_sizes(image)
print(f"Voxel sizes (mm): {voxel_sizes}")
```

### Creating Masks Programmatically

```python
from cluster_analysis.jupyter_utils import SpatialMask

# Create a mask with multiple ROIs
mask = SpatialMask(
    shape=(128, 128, 64),  # image dimensions
    rois=[
        {
            "type": "rectangle",
            "x_min": 40, "y_min": 40,
            "x_max": 80, "y_max": 80,
            "z_min": 20, "z_max": 44,
            "slice": 32  # Display on slice 32
        },
        {
            "type": "polygon",
            "points": [(50, 50), (70, 50), (60, 70)],
            "slice": 32
        }
    ],
    name="tumor_region"
)

# Save for later use
mask.save("masks/tumor_region.json")

# Convert to boolean array
bool_mask = mask.to_array_mask()
```

### Computing Statistics

```python
from cluster_analysis.stats import compute_basic_stats
from cluster_analysis.interfile import load_image, get_image_array

image = load_image("image.hv")
array = get_image_array(image)
values = array.ravel().tolist()

# Compute stats on all voxels
stats = compute_basic_stats(values)

# Or with a mask
mask = [True] * len(values)  # or from SpatialMask.to_flat_mask()
stats = compute_basic_stats(values, mask)

print(f"Mean: {stats['mean']}")
print(f"Std: {stats['std']}")
print(f"CV: {stats['coefficient_of_variation']}")
```

## Dependencies

- **SIRF** (required): Image loading
- **NumPy** (optional): For Jupyter features
- **Matplotlib** (optional): For interactive viewing
- **ipympl** (optional): For widget support in Jupyter
- **Pandas** (optional): For convenient data handling

## Notes

- This package uses SIRF.STIR ImageData as the core image representation
- All images must be in a format readable by SIRF (e.g., Interfile .hv/.v)
- The interactive Jupyter features require `numpy`, `matplotlib`, and `ipympl`
- Masks are saved as JSON and are human-readable and version-control friendly
