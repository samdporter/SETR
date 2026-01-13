# Phantom Reconstruction Experiments

This directory contains configuration and documentation for standardized phantom reconstruction experiments across five algorithms (HKEM, dTNV, TNV, Log-dTNV, Log-TNV) and four phantom datasets (NEMA, Manchester NEMA, Manchester NEMA short, Anthropomorphic).

## Overview

**Total Experiments**: 20 (4 phantoms × 5 algorithms)

| Phantom | HKEM | dTNV | TNV | Log-dTNV | Log-TNV |
|---------|------|------|-----|----------|---------|
| Manchester NEMA | ✓ | ✓ | ✓ | ✓ | ✓ |
| Manchester NEMA (short) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anthropomorphic | ✓ | ✓ | ✓ | ✓ | ✓ |
| NEMA | ✓ | ✓ | ✓ | ✓ | ✓ |

## Algorithm Descriptions

### HKEM (Hybrid Kernel EM)
**Type**: Two-stage sequential reconstruction

**Workflow**:
1. Stage 1: SPECT reconstruction with CT anatomical guidance
2. Resample: Transform SPECT to PET space
3. Stage 2: PET reconstruction with SPECT emission guidance

**Key Features**:
- Uses kernel-based smoothing guided by anatomical and emission features
- Two-stage approach enables sequential refinement
- Hybrid kernel combines anatomical (CT) and emission (SPECT) information

**Output Structure**:
```
output/{phantom}/hkem/
├── spect/               # Stage 1 outputs
│   ├── reconstruction_x.hv
│   ├── reconstruction_alpha.hv
│   └── objective.csv
├── spect_resampled/     # Resampled to PET space
│   └── spect_in_pet_space.hv
└── pet/                 # Stage 2 outputs
    ├── reconstruction_x.hv
    ├── reconstruction_alpha.hv
    └── objective.csv
```

### dTNV (Directional Total Nuclear Variation)
**Type**: Joint PET+SPECT reconstruction with CT guidance

**Key Settings**:
- `directional_tnv: true` - Enables CT anatomical gradient projection
- Joint reconstruction of both modalities simultaneously
- Uses vectorial total variation regularization

**How Directional Guidance Works**:
- Computes gradients of CT anatomical image
- Projects PET/SPECT gradients orthogonal to CT structure
- Preserves edges aligned with anatomy, smooths others

**Output Structure**:
```
output/{phantom}/dtnv/
├── image_0_*.hv        # PET reconstructions at different iterations
├── image_1_*.hv        # SPECT reconstructions at different iterations
├── objective.csv
└── ...
```

### TNV (Isotropic Total Nuclear Variation)
**Type**: Joint PET+SPECT reconstruction without CT guidance

**Key Settings**:
- `directional_tnv: false` - Isotropic (non-directional) TNV
- No anatomical guidance from CT
- Uses only data-driven cross-modal weights (alpha, beta)

**Difference from dTNV**:
- dTNV uses CT anatomical edges to guide regularization
- TNV uses purely isotropic smoothing
- Both use vectorial TV across PET+SPECT channels

**Output Structure**:
```
output/{phantom}/tnv/
├── image_0_*.hv        # PET reconstructions
├── image_1_*.hv        # SPECT reconstructions
├── objective.csv
└── ...
```

### Log-dTNV (Log-domain Directional TNV)
**Type**: Joint PET+SPECT reconstruction with CT guidance in log domain

**Key Settings**:
- `directional_tnv: true` - Enables CT anatomical gradient projection
- `use_log_tnv: true` - **KEY**: Log-domain transformation
- Joint reconstruction with log-transformed intensities

**Difference from dTNV**:
- Log transformation normalizes dynamic range between modalities
- Delta parameter becomes scale-invariant
- Better handling of large intensity differences
- Same directional CT guidance mechanism as dTNV

**Output Structure**:
```
output/{phantom}/log_dtnv/
├── image_0_*.hv        # PET reconstructions (log-domain)
├── image_1_*.hv        # SPECT reconstructions (log-domain)
├── objective.csv
└── ...
```

### Log-TNV (Log-domain Isotropic TNV)
**Type**: Joint PET+SPECT reconstruction without CT guidance in log domain

**Key Settings**:
- `directional_tnv: false` - Isotropic TNV
- `use_log_tnv: true` - **KEY**: Log-domain transformation
- No anatomical guidance

**Difference from TNV**:
- Log transformation for better dynamic range handling
- Otherwise identical to TNV (isotropic, no CT guidance)

**Output Structure**:
```
output/{phantom}/log_tnv/
├── image_0_*.hv        # PET reconstructions (log-domain)
├── image_1_*.hv        # SPECT reconstructions (log-domain)
├── objective.csv
└── ...
```

## Phantom Datasets

### Manchester NEMA
- **Location**: `/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/`
- **Subsets**: [14 PET, 18 SPECT]
- **Resolution**:
  - PET: [7, 7, 7] mm FWHM
  - SPECT: [6.78, 6.78, 6.78] mm FWHM
  - Collimator: [1.21, 0.027, false]
- **Special**: `flip: true`

### Manchester NEMA (short bootstrap)
- **Location**:
  - PET: `/home/storage/bootstraps/prompts_nonparam_s73_sf0.0167_000`
  - SPECT: `/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/SPECT`
- **Subsets**: [18 PET, 14 SPECT]
- **Resolution**:
  - PET: [5.61, 4.83, 4.93] mm FWHM
  - SPECT: [6.78, 6.78, 6.78] mm FWHM
  - Collimator: [1.21, 0.027, false]
- **Special**: Same transform file as full Manchester (`spect2pet_nozoom.nii`), `flip: true`

### Anthropomorphic
- **Location**: `/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/`
  - PET: `PET/phantom_short`
  - SPECT: `SPECT/phantom_140`
- **Subsets**: [18 PET, 18 SPECT]
- **Resolution**:
  - PET: [5.61, 4.83, 4.93] mm FWHM (UCL 710 acceptance)
  - SPECT: [6.8, 6.8, 6.8] mm FWHM
  - Collimator: [1.31, 0.027, false]

### NEMA
- **Location**: `/home/storage/prepared_data/phantom_data/nema_phantom_data/`
- **Subsets**: [18 PET, 18 SPECT]
- **Resolution**:
  - PET: [7.3, 7.3, 7.3] mm FWHM
  - SPECT: [6.7, 6.7, 6.7] mm FWHM
  - Collimator: [0.81, 0.03, false]

## Configuration System

### Hierarchical Config Composition

Configs are composed from three layers:

```
Final Config = Base + Phantom + Algorithm + CLI Overrides
```

**Layer 1: Base** (algorithm-dependent)
- **HKEM**: `base_hkem.yaml`
  - 10 epochs, ordered subsets (sequential sampling)
  - No variance reduction (uses SGFunction)
  - Kernel operator defaults
- **TNV variants**: `base_tnv.yaml`
  - 100 epochs, SVRG variance reduction
  - Shared across dTNV, TNV, log-dTNV, log-TNV

**Layer 2: Phantom** (`phantom_1bpos_{phantom}.yaml`)
- Data paths (pet_data_path, spect_data_path)
- Resolution modeling (gauss_fwhm, spect_res)
- Phantom-specific subsets (for TNV: [9 PET, 12 SPECT] or [36 PET, 18 SPECT])

**Layer 3: Algorithm** (`algo_{algorithm}.yaml`)
- **HKEM**: Modality, guidance type, sigma values, num_subsets
  - SPECT: 12 subsets, sigma_anatomical=1.0
  - PET: 9 subsets, sigma_anatomical=3.0
- **dTNV/TNV**: Prior settings, directional_tnv, use_log_tnv, alpha/beta

**Layer 4: CLI Overrides** (optional)
- Runtime parameter changes via `--override key=value`

### Config Files

All experiment configs are self-contained in `experiments/configs/`:

```
experiments/configs/
├── base_hkem.yaml               # HKEM base (10 epochs, ordered subsets)
├── base_tnv.yaml                # TNV base (100 epochs, SVRG)
├── phantom_1bpos_manc.yaml      # Manchester NEMA
├── phantom_1bpos_manc_short.yaml# Manchester NEMA (bootstrap)
├── phantom_1bpos_anthro.yaml    # Anthropomorphic
├── phantom_1bpos_nema.yaml      # NEMA
├── algo_hkem_spect.yaml         # HKEM Stage 1 (12 subsets)
├── algo_hkem_pet.yaml           # HKEM Stage 2 (9 subsets)
├── algo_dtnv.yaml               # dTNV (CT guidance, standard domain)
├── algo_tnv.yaml                # TNV (no CT, standard domain)
├── algo_log_dtnv.yaml           # Log-dTNV (CT guidance, log domain)
└── algo_log_tnv.yaml            # Log-TNV (no CT, log domain)
```

**Note**: These configs are separate from the main `configs/` directory, making it easy to modify experiment parameters without affecting other workflows.

## Usage

### Running Experiments

The master script `experiments/scripts/run_phantom_experiments.py` orchestrates all experiments.

#### Single Experiment

```bash
# Run Manchester phantom with HKEM
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm hkem

# Run Anthropomorphic phantom with dTNV
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm dtnv

# Run NEMA phantom with TNV
python experiments/scripts/run_phantom_experiments.py --phantom nema --algorithm tnv
```

#### All Algorithms for One Phantom

```bash
# Run all five algorithms for Manchester phantom (including log variants)
python experiments/scripts/run_phantom_experiments.py --phantom manc --all-algorithms
```

#### Batch Mode - All Combinations

```bash
# Run all experiments (current default: 4 phantoms × 5 algorithms = 20)
python experiments/scripts/run_phantom_experiments.py --batch-all
```

#### Parameter Overrides

```bash
# Override regularization parameter
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv \
    --override alpha=0.05

# Override multiple parameters
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm tnv \
    --override num_epochs=200 --override alpha=0.1
```

#### Dry Run (Preview Configs)

```bash
# Preview what would be run without executing
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm hkem --dry-run

# Dry run for all combinations
python experiments/scripts/run_phantom_experiments.py --batch-all --dry-run
```

## Parameter Consistency

### Shared Across All Experiments
- `num_epochs: 100`
- `initial_step_size: 1.0`
- `relaxation_eta: 0.01`
- `use_scatter: true`
- `variance_reduction: "svrg"`
- `save_images: true`

### Phantom-Specific
- `num_subsets` - Varies by phantom
- `pet_gauss_fwhm` - Scanner-specific resolution
- `spect_gauss_fwhm` - Scanner-specific resolution
- `spect_res` - Collimator parameters
- Data paths

### Algorithm-Specific

**HKEM**:
- `num_neighbours: 5`
- `sigma_anatomical`: 1.0 (SPECT), 3.0 (PET)
- `sigma_emission`: 0.1 (SPECT), 1.0 (PET)
- `freeze_iter: 90`

**dTNV**:
- `directional_tnv: true`
- `alpha: 0.02`, `beta: 0.02`
- `gamma_tnv: 1.0`
- `tnv_stencil: "26"`

**TNV**:
- `directional_tnv: false` (KEY DIFFERENCE)
- `alpha: 0.02`, `beta: 0.02`
- `gamma_tnv: 1.0`
- `tnv_stencil: "26"`

## Output Organization

All results are saved to `output/{phantom}/{algorithm}/`:

```
output/
├── manc/
│   ├── hkem/
│   │   ├── spect/
│   │   ├── spect_resampled/
│   │   └── pet/
│   ├── dtnv/        # + objective.csv, image_*.hv
│   ├── tnv/
│   ├── log_dtnv/
│   └── log_tnv/
├── manc_short/
│   ├── hkem/        # Same substructure as manc
│   ├── dtnv/
│   ├── tnv/
│   ├── log_dtnv/
│   └── log_tnv/
├── anthro/
│   ├── hkem/
│   │   ├── spect/
│   │   ├── spect_resampled/
│   │   └── pet/
│   ├── dtnv/
│   ├── tnv/
│   ├── log_dtnv/
│   └── log_tnv/
└── nema/
    ├── hkem/
    │   ├── spect/
    │   ├── spect_resampled/
    │   └── pet/
    ├── dtnv/
    ├── tnv/
    ├── log_dtnv/
    └── log_tnv/
```

## Experiment Matrix

| # | Phantom | Algorithm | Output Path | Description |
|---|---------|-----------|-------------|-------------|
| 1 | manc | hkem | output/manc/hkem/ | SPECT→PET sequential |
| 2 | manc | dtnv | output/manc/dtnv/ | Joint with CT guidance |
| 3 | manc | tnv | output/manc/tnv/ | Joint isotropic |
| 4 | manc | log_dtnv | output/manc/log_dtnv/ | Log-domain joint with CT guidance |
| 5 | manc | log_tnv | output/manc/log_tnv/ | Log-domain joint isotropic |
| 6 | manc_short | hkem | output/manc_short/hkem/ | SPECT→PET sequential (bootstrap PET) |
| 7 | manc_short | dtnv | output/manc_short/dtnv/ | Joint with CT guidance |
| 8 | manc_short | tnv | output/manc_short/tnv/ | Joint isotropic |
| 9 | manc_short | log_dtnv | output/manc_short/log_dtnv/ | Log-domain joint with CT guidance |
| 10 | manc_short | log_tnv | output/manc_short/log_tnv/ | Log-domain joint isotropic |
| 11 | anthro | hkem | output/anthro/hkem/ | SPECT→PET sequential |
| 12 | anthro | dtnv | output/anthro/dtnv/ | Joint with CT guidance |
| 13 | anthro | tnv | output/anthro/tnv/ | Joint isotropic |
| 14 | anthro | log_dtnv | output/anthro/log_dtnv/ | Log-domain joint with CT guidance |
| 15 | anthro | log_tnv | output/anthro/log_tnv/ | Log-domain joint isotropic |
| 16 | nema | hkem | output/nema/hkem/ | SPECT→PET sequential |
| 17 | nema | dtnv | output/nema/dtnv/ | Joint with CT guidance |
| 18 | nema | tnv | output/nema/tnv/ | Joint isotropic |
| 19 | nema | log_dtnv | output/nema/log_dtnv/ | Log-domain joint with CT guidance |
| 20 | nema | log_tnv | output/nema/log_tnv/ | Log-domain joint isotropic |

## Validation

### Pre-Flight Checks

Before running experiments, validate that:

1. **Data paths exist**:
   ```bash
   ls /home/storage/prepared_data/phantom_data/manc_nema_phantom_data/
   ls /home/storage/bootstraps/prompts_nonparam_s73_sf0.0167_000
   ls /home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/
   ls /home/storage/prepared_data/phantom_data/nema_phantom_data/
   ```

2. **Config composition works**:
   ```bash
   python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv --dry-run
   ```

3. **Output directories are writable**:
   ```bash
   mkdir -p output/test && rm -rf output/test
   ```

### Testing

Test with a single short experiment first:

```bash
# Run one experiment with reduced epochs
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm tnv \
    --override num_epochs=2
```

## Troubleshooting

### Common Issues

1. **Missing data paths**
   - Verify phantom data locations exist
   - Check paths in `configs/experiments/phantom_1bpos_*.yaml`

2. **Config composition errors**
   - Use `--dry-run` to preview merged config
   - Check YAML syntax in experiment config files

3. **HKEM resampling fails**
   - Ensure `scripts/resample_spect_to_pet.py` exists
   - Check SPECT reconstruction completed successfully

4. **GPU memory errors**
   - Reduce `num_subsets` or batch size
   - Set `no_gpu: true` in overrides

## Notes

- All experiments use 100 epochs for consistency
- HKEM takes longest (3 stages: SPECT → resample → PET)
- dTNV and TNV are joint reconstructions (faster)
- Delta parameter is auto-computed if not specified
- Configs are saved to `tmp/experiment_configs/` for reference

## References

- **HKEM workflow**: See `scripts/run_hkem_comparison_1bpos.sh`
- **dTNV implementation**: See `scripts/run_dtnv_1bpos.py`
- **Directional gradients**: See `src/setr/core/gradients/gradients.py`
- **Prior setup**: See `src/setr/scripts/dtnv_common.py`
