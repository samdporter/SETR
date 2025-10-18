# MANC Bootstrap Reconstruction - Quick Start

## TL;DR

### DTNV Reconstruction (with specific alpha/beta)

```bash
# Test locally first (5 epochs, bootstrap 0)
./scripts/test_manc_bootstrap_dtnv_local.sh 0.5 1.0 0

# Submit cluster job for all 30 bootstraps
./scripts/launch_manc_bootstrap_dtnv.sh 0.5 1.0 30

# Monitor
qstat -u $USER
tail -f ~/setr_logs/manc_dtnv_a0.5_b1.0_*.log
```

### HKEM Reconstruction (SPECT guidance for PET)

```bash
# Test locally first (5 epochs, bootstrap 0)
./scripts/test_manc_bootstrap_hkem_local.sh 0

# Run full pipeline (sequential, 30 bootstraps)
./scripts/run_manc_bootstrap_hkem.sh 30
```

## What Gets Created

### DTNV
```
results/manc_bootstraps/dtnv/
└── alpha_0.5_beta_1.0/
    ├── bootstrap_000/
    │   ├── reconstruction_x.hv
    │   ├── objective.csv
    │   └── job_completion.txt
    ├── bootstrap_001/
    └── ...
```

### HKEM
```
results/manc_bootstraps/hkem/
├── spect_hkem/
│   └── reconstruction_x.hv      # Shared SPECT recon
├── bootstrap_000/
│   └── reconstruction_x.hv      # PET recon with SPECT guidance
├── bootstrap_001/
└── ...
```

## Key Parameters

### DTNV
- **Epochs**: 50 (configurable)
- **Alpha/Beta**: Set via command line
- **Flip**: Enabled (required for MANC)
- **Gamma_TNV**: 50.0 (can override)

### HKEM
- **SPECT**: 15 epochs, 12 subsets
- **PET**: 15 epochs, 9 subsets
- **Guidance**: SPECT emission
- **Flip**: Enabled (required for MANC)

## Files Reference

| Purpose | Script/Config |
|---------|--------------|
| DTNV cluster launcher | `scripts/launch_manc_bootstrap_dtnv.sh` |
| DTNV local test | `scripts/test_manc_bootstrap_dtnv_local.sh` |
| HKEM pipeline | `scripts/run_manc_bootstrap_hkem.sh` |
| HKEM local test | `scripts/test_manc_bootstrap_hkem_local.sh` |
| DTNV config | `configs/config_manc_bootstrap_dtnv.yaml` |
| HKEM SPECT config | `configs/config_manc_hkem_spect.yaml` |
| HKEM PET config | `configs/config_manc_hkem_pet.yaml` |

See [manc_bootstrap_workflows.md](manc_bootstrap_workflows.md) for full documentation.
