# MANC Phantom Bootstrap Reconstruction Workflows

This document describes the workflows for running DTNV and HKEM reconstructions on MANC NEMA phantom bootstrap datasets.

## Overview

Two reconstruction workflows are available for the MANC phantom bootstrap data located at:
```
/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/PET/bootstraps/nonparametric/
```

1. **DTNV Reconstruction**: Dual-modality (PET+SPECT) reconstruction with specified alpha/beta values
2. **HKEM Reconstruction**: Hybrid Kernelized EM reconstruction with SPECT emission guidance

Both workflows support:
- Cluster execution via SGE array jobs (30 bootstraps in parallel)
- Local testing with reduced parameters

## Data Structure

- **PET Bootstrap Data**: 30 bootstrap datasets (000-029)
  - Path: `prompts_nonparam_s73_sf0.05_XXX`
- **SPECT Data**: Shared across all bootstraps
  - Path: `/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/SPECT`
- **IMPORTANT**: SPECT data requires flipping (`flip: true`) for MANC phantom

## DTNV Bootstrap Reconstruction

### Description

Runs DTNV reconstruction with a specified alpha/beta pair across all bootstrap datasets.

### Configuration Files

- Base config: [config_manc_bootstrap_dtnv.yaml](../configs/config_manc_bootstrap_dtnv.yaml)
- Key parameters:
  - `num_epochs: 50` (default)
  - `flip: true` (SPECT flipping for MANC data)
  - `gamma_tnv: 50.0` (can be overridden)
  - `use_kappa: true`

### Cluster Execution

**Launch reconstruction for all 30 bootstraps:**

```bash
cd /home/sam/working/synergistic_recon/scripts
./launch_manc_bootstrap_dtnv.sh <alpha> <beta> [num_bootstraps]
```

**Examples:**

```bash
# Run with alpha=0.5, beta=1.0 for all 30 bootstraps
./launch_manc_bootstrap_dtnv.sh 0.5 1.0 30

# Run with alpha=1.0, beta=1.0 for first 10 bootstraps
./launch_manc_bootstrap_dtnv.sh 1.0 1.0 10
```

**Monitor job progress:**

```bash
# Check job status
qstat -u $USER

# View logs
tail -f ~/setr_logs/manc_dtnv_a0.5_b1.0_*.log
```

**Output location:**

```
/home/sam/working/synergistic_recon/results/manc_bootstraps/dtnv/alpha_X_beta_Y/bootstrap_XXX/
```

### Local Testing

**Quick test on a single bootstrap (5 epochs):**

```bash
cd /home/sam/working/synergistic_recon/scripts
./test_manc_bootstrap_dtnv_local.sh [alpha] [beta] [bootstrap_num]
```

**Example:**

```bash
# Test bootstrap 0 with alpha=0.5, beta=1.0
./test_manc_bootstrap_dtnv_local.sh 0.5 1.0 0
```

**Output location:**

```
/home/sam/working/synergistic_recon/test_output/dtnv_local/alpha_X_beta_Y/bootstrap_XXX/
```

## HKEM Bootstrap Reconstruction

### Description

Runs complete HKEM pipeline for bootstrap analysis:
1. SPECT HKEM reconstruction (once, shared)
2. For each bootstrap:
   - Resample SPECT to PET space
   - PET HKEM reconstruction with SPECT emission guidance

### Configuration Files

- SPECT config: [config_manc_hkem_spect.yaml](../configs/config_manc_hkem_spect.yaml)
- PET config: [config_manc_hkem_pet.yaml](../configs/config_manc_hkem_pet.yaml)
- Resample config: [config_manc_resample.yaml](../configs/config_manc_resample.yaml)

**Key parameters:**
- SPECT: `num_epochs: 15`, `num_subsets: 12`, `flip: true`
- PET: `num_epochs: 15`, `num_subsets: 9`, `guidance: emission`

### Cluster Execution

**Run complete pipeline locally (sequential):**

```bash
cd /home/sam/working/synergistic_recon/scripts
./run_manc_bootstrap_hkem.sh [num_bootstraps]
```

**Example:**

```bash
# Process all 30 bootstraps
./run_manc_bootstrap_hkem.sh 30

# Process first 5 bootstraps
./run_manc_bootstrap_hkem.sh 5
```

**Control flags** (edit in script):
- `DO_SPECT=true`: Run SPECT HKEM
- `DO_RESAMPLE=true`: Resample SPECT to PET
- `DO_PET_HKEM=true`: Run PET HKEM

**Output location:**

```
/home/sam/working/synergistic_recon/results/manc_bootstraps/hkem/
├── spect_hkem/              # Shared SPECT reconstruction
├── bootstrap_000/           # PET HKEM for bootstrap 0
├── bootstrap_001/           # PET HKEM for bootstrap 1
└── ...
```

### Local Testing

**Quick test on a single bootstrap (5 epochs):**

```bash
cd /home/sam/working/synergistic_recon/scripts
./test_manc_bootstrap_hkem_local.sh [bootstrap_num]
```

**Example:**

```bash
# Test bootstrap 0
./test_manc_bootstrap_hkem_local.sh 0
```

**Output location:**

```
/home/sam/working/synergistic_recon/test_output/hkem_local/
├── spect_hkem/              # SPECT reconstruction
└── bootstrap_XXX/           # PET reconstruction
```

## Key Implementation Details

### SPECT Flipping

MANC phantom data requires SPECT images to be flipped for proper co-registration with PET:
- Set `flip: true` in all MANC configs
- Applied during reconstruction (DTNV) or resampling (HKEM)

### Resampling for MANC Data

MANC phantom uses simplified resampling (no zoom operation):
- Script: [resample_spect_to_pet_simple.py](../scripts/resample_spect_to_pet_simple.py)
- Transform: `spect2pet.nii` (not `spect2pet_zoom_nonrigid.nii`)
- Supports flip operation

### Cluster Resource Requirements

**DTNV (per bootstrap):**
- Time: 4 hours (`h_rt=04:00:00`)
- Memory: 16GB (`h_vmem=16G`)
- GPU: 1 (`gpu=1`)

**HKEM (sequential pipeline):**
- Runs locally on compute node
- SPECT HKEM: ~15-30 minutes
- Per bootstrap: ~15-30 minutes
- Total for 30 bootstraps: ~8-15 hours

## Troubleshooting

### Common Issues

1. **Bootstrap directory not found**
   - Check path: `/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/PET/bootstraps/nonparametric/`
   - Verify bootstrap naming: `prompts_nonparam_s73_sf0.05_XXX`

2. **SPECT resampling fails**
   - Ensure `spect2pet.nii` transform exists in SPECT directory
   - Check `flip: true` is set in config

3. **GPU errors**
   - Check GPU availability: `nvidia-smi`
   - Set `no_gpu: true` to force CPU mode (slower)

4. **Memory errors**
   - Reduce `keep_all_views_in_cache` to `false`
   - Request more memory in cluster job

### Checking Results

**Verify DTNV reconstruction:**

```bash
# Check output files exist
ls results/manc_bootstraps/dtnv/alpha_0.5_beta_1.0/bootstrap_000/

# Expected files:
# - reconstruction_x.hv (final reconstruction)
# - objective.csv (objective function values)
# - job_completion.txt (job status)
```

**Verify HKEM reconstruction:**

```bash
# Check SPECT reconstruction
ls results/manc_bootstraps/hkem/spect_hkem/reconstruction_x.hv

# Check PET reconstruction for bootstrap 0
ls results/manc_bootstraps/hkem/bootstrap_000/reconstruction_x.hv

# Check resampled SPECT in bootstrap directory
ls /home/storage/prepared_data/phantom_data/manc_nema_phantom_data/PET/bootstraps/nonparametric/prompts_nonparam_s73_sf0.05_000/spect.hv
```

## Files Created

### Scripts

- [launch_manc_bootstrap_dtnv.sh](../scripts/launch_manc_bootstrap_dtnv.sh) - Launcher for DTNV cluster jobs
- [run_manc_bootstrap_dtnv.qsub.sh](../scripts/run_manc_bootstrap_dtnv.qsub.sh) - SGE submission script for DTNV
- [run_manc_bootstrap_hkem.sh](../scripts/run_manc_bootstrap_hkem.sh) - HKEM pipeline script
- [resample_spect_to_pet_simple.py](../scripts/resample_spect_to_pet_simple.py) - Simplified resampling with flip
- [test_manc_bootstrap_dtnv_local.sh](../scripts/test_manc_bootstrap_dtnv_local.sh) - Local DTNV test
- [test_manc_bootstrap_hkem_local.sh](../scripts/test_manc_bootstrap_hkem_local.sh) - Local HKEM test

### Configs

- [config_manc_bootstrap_dtnv.yaml](../configs/config_manc_bootstrap_dtnv.yaml) - DTNV base config
- [config_manc_hkem_spect.yaml](../configs/config_manc_hkem_spect.yaml) - HKEM SPECT config
- [config_manc_hkem_pet.yaml](../configs/config_manc_hkem_pet.yaml) - HKEM PET config
- [config_manc_resample.yaml](../configs/config_manc_resample.yaml) - Resampling config

## Next Steps

1. **Test locally** before submitting cluster jobs:
   ```bash
   ./test_manc_bootstrap_dtnv_local.sh 0.5 1.0 0
   ./test_manc_bootstrap_hkem_local.sh 0
   ```

2. **Submit small test job** (e.g., 3 bootstraps):
   ```bash
   ./launch_manc_bootstrap_dtnv.sh 0.5 1.0 3
   ```

3. **Monitor and verify** results before full 30-bootstrap run

4. **Full production run**:
   ```bash
   ./launch_manc_bootstrap_dtnv.sh 0.5 1.0 30
   ./run_manc_bootstrap_hkem.sh 30
   ```
