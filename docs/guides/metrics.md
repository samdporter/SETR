# Image Quality Metrics Usage Guide

This guide explains how to use the image quality metrics functionality for reconstruction evaluation.

## Overview

The metrics module (`setr.utils.metrics`) provides functions to compute various error metrics between reconstructed images and reference (ground truth) images:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **NRMSE**: Normalized Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **NMAE**: Normalized Mean Absolute Error

All metrics support optional masking to restrict evaluation to regions of interest.

## Basic Usage

### Computing Individual Metrics

```python
from sirf.STIR import ImageData
from setr.utils.metrics import compute_mse, compute_rmse, compute_nrmse

# Load images
reconstructed = ImageData("output/image_0_100.hv")
reference = ImageData("reference/ground_truth.hv")

# Compute metrics
mse = compute_mse(reconstructed, reference)
rmse = compute_rmse(reconstructed, reference)
nrmse = compute_nrmse(reconstructed, reference, normalization='range')

print(f"MSE: {mse:.6e}")
print(f"RMSE: {rmse:.6e}")
print(f"NRMSE: {nrmse:.6f}")
```

### Computing All Metrics at Once

```python
from setr.utils.metrics import compute_all_metrics

metrics = compute_all_metrics(reconstructed, reference)
# Returns: {'mse': ..., 'rmse': ..., 'nrmse': ..., 'mae': ..., 'nmae': ...}

for name, value in metrics.items():
    print(f"{name.upper()}: {value:.6e}")
```

### Using Masks

Restrict metrics to regions of interest:

```python
from setr.utils.metrics import create_mask_from_threshold, compute_rmse

# Create mask from reference (e.g., only voxels > 1% of max)
mask = create_mask_from_threshold(
    reference,
    threshold=0.01 * reference.max(),
    mode='greater'
)

# Compute metrics only within mask
rmse_masked = compute_rmse(reconstructed, reference, mask=mask)
```

### Multi-Modal (BlockDataContainer)

For multi-modal reconstructions (PET + SPECT):

```python
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.utils.metrics import compute_block_metrics

# Load multi-modal images
ref_pet = ImageData("reference/image_0_final.hv")
ref_spect = ImageData("reference/image_1_final.hv")
reference = EnhancedBlockDataContainer(ref_pet, ref_spect)

recon_pet = ImageData("output/image_0_100.hv")
recon_spect = ImageData("output/image_1_100.hv")
reconstructed = EnhancedBlockDataContainer(recon_pet, recon_spect)

# Compute metrics per modality
metrics = compute_block_metrics(reconstructed, reference)
# Returns:
# {
#   'modality_0': {'mse': ..., 'rmse': ..., 'nrmse': ..., 'mae': ..., 'nmae': ...},
#   'modality_1': {'mse': ..., 'rmse': ..., 'nrmse': ..., 'mae': ..., 'nmae': ...}
# }

print(f"PET RMSE: {metrics['modality_0']['rmse']:.6e}")
print(f"SPECT RMSE: {metrics['modality_1']['rmse']:.6e}")
```

## CIL Callbacks for Reconstruction

### Using ComputeMetricsCallback

Automatically compute and save metrics during reconstruction:

```python
from setr.cil_extensions.callbacks import ComputeMetricsCallback
from setr.utils.metrics import create_mask_from_threshold

# Load reference
reference = ImageData("ground_truth.hv")

# Optional: create mask
mask = create_mask_from_threshold(reference, threshold=0.01 * reference.max())

# Create callback
metrics_callback = ComputeMetricsCallback(
    reference=reference,
    filename="output/metrics",  # Will create metrics.csv
    interval=10,  # Compute every 10 iterations
    mask=mask,  # Optional
    normalization='range',  # For NRMSE/NMAE
    verbose=True  # Log metrics to console
)

# Run reconstruction with callback
algo.run(100, callbacks=[metrics_callback])
```

**Output**: Creates `output/metrics.csv` with columns:
```
iteration,mse,rmse,nrmse,mae,nmae
10,1.234e-05,0.003512,0.0234,0.002891,0.0192
20,8.765e-06,0.002961,0.0197,0.002341,0.0156
...
```

### Using PrintMetricsCallback

Lighter version that only prints to console (no CSV):

```python
from setr.cil_extensions.callbacks import PrintMetricsCallback

metrics_callback = PrintMetricsCallback(
    reference=reference,
    interval=10,
    mask=mask,
    metrics=['rmse', 'nrmse']  # Only print these metrics
)

algo.run(100, callbacks=[metrics_callback])
```

### Multi-Modal Callback Example

```python
# Load reference (multi-modal)
ref_pet = ImageData("ref/image_0_final.hv")
ref_spect = ImageData("ref/image_1_final.hv")
reference = EnhancedBlockDataContainer(ref_pet, ref_spect)

# Create masks per modality
mask_pet = create_mask_from_threshold(ref_pet, threshold=0.01 * ref_pet.max())
mask_spect = create_mask_from_threshold(ref_spect, threshold=0.01 * ref_spect.max())
mask = EnhancedBlockDataContainer(mask_pet, mask_spect)

# Create callback
metrics_callback = ComputeMetricsCallback(
    reference=reference,
    filename="output/metrics",
    interval=18,  # Every epoch for 18 subsets
    mask=mask,
    normalization='range',
    verbose=True
)

# Run reconstruction
algo.run(1800, callbacks=[metrics_callback])  # 100 epochs × 18 subsets
```

**Output**: Creates `output/metrics.csv` with columns:
```
iteration,modality_0_mse,modality_0_rmse,modality_0_nrmse,...,modality_1_mse,modality_1_rmse,modality_1_nrmse,...
18,1.2e-05,0.0035,0.023,...,3.4e-06,0.0018,0.012,...
36,8.7e-06,0.0030,0.020,...,2.1e-06,0.0015,0.010,...
...
```

## Normalization Options

For NRMSE and NMAE, choose normalization method:

- **'range'**: Divide by (max - min) of reference [default]
- **'max'**: Divide by max of reference
- **'mean'**: Divide by mean of reference
- **'euclidean'**: Divide by L2 norm of reference

```python
nrmse_range = compute_nrmse(recon, ref, normalization='range')
nrmse_max = compute_nrmse(recon, ref, normalization='max')
```

## Subset Selection Experiments Integration

For the subset selection experiments, metrics are computed automatically if configured:

### Configuration

In `base_config_anthro.yaml`:

```yaml
# Metrics computation
compute_metrics: true
metrics_interval: null  # Use update_interval
reference_path: "path/to/convergence_reference/output"
mask_threshold: 0.01  # Voxels > 1% of max
metrics_normalization: "range"
```

### Reference Path Structure

The reference path should point to a completed convergence run output directory containing:
```
reference_path/
├── image_0_final.hv  # PET convergence reference
└── image_1_final.hv  # SPECT convergence reference
```

### Running with Metrics

```bash
python run_subset_selection.py \
    --config base_config_anthro.yaml \
    --override \
        reference_path=/path/to/convergence/gamma_100 \
        gamma_tnv=100
```

This will:
1. Load convergence reference for gamma=100
2. Create mask from reference (voxels > 1% of max)
3. Compute metrics every `update_interval` iterations
4. Save to `output/metrics.csv`
5. Log metrics to console during reconstruction

## Analysis Workflow

### 1. Run Convergence Reference

```bash
# 10,000 epochs to convergence
./launch_sweep.sh sweep_convergence_ref.yaml full
```

### 2. Run Main Experiments with Metrics

Update configs to point to convergence references:

```yaml
reference_path: "output/subset_selection_convergence/subset_separate_prior_always_precond_vtv_svd_principal_alpha_gamma_100"
```

Then run:

```bash
./launch_sweep.sh sweep_main_experiments.yaml full
```

### 3. Compare Convergence

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics from different experiments
metrics_config1 = pd.read_csv("output/config1/metrics.csv", index_col='iteration')
metrics_config2 = pd.read_csv("output/config2/metrics.csv", index_col='iteration')

# Plot convergence comparison
plt.figure(figsize=(10, 6))
plt.semilogy(metrics_config1.index, metrics_config1['modality_0_nrmse'],
             label='Config 1 (PET)')
plt.semilogy(metrics_config2.index, metrics_config2['modality_0_nrmse'],
             label='Config 2 (PET)')
plt.xlabel('Iteration')
plt.ylabel('NRMSE')
plt.legend()
plt.title('Convergence Comparison')
plt.grid(True, alpha=0.3)
plt.savefig('convergence_comparison.png', dpi=150)
```

## Tips and Best Practices

1. **Masking**: Always use masks to exclude background and focus on anatomical regions
2. **Normalization**: Use 'range' normalization for comparison across different reconstructions
3. **Interval**: Set `metrics_interval` = `update_interval` to track convergence smoothly
4. **Storage**: Metrics CSV files are small (~1 KB per 1000 iterations) vs images (~10 MB each)
5. **Convergence**: Look for NRMSE < 0.01 (1%) as indication of convergence
6. **Multi-modal**: Compare modalities separately - they may converge at different rates

## Error Handling

The metrics callback gracefully handles errors:

```python
# If reference path is invalid or files missing
# -> Warning logged, reconstruction continues without metrics

# If mask is empty
# -> Returns NaN for masked metrics

# If shapes don't match
# -> ValueError raised with clear message
```

## Performance Notes

- Metrics computation adds ~0.5-1% overhead per iteration
- Mask creation is one-time at callback initialization
- CSV writing is buffered (minimal I/O overhead)
- For 10,000 iteration runs, metrics computation adds ~5-10 minutes total
