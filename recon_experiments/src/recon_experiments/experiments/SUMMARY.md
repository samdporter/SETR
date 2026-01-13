# Phantom Experiments - Final Setup Summary

## Complete Framework: 20 Experiments Ready to Run

**Total**: 4 phantoms × 5 algorithms = **20 experiments**

### Algorithms Configured

| Algorithm | Epochs | Subsets | Variance Reduction | CT Guidance | Domain |
|-----------|--------|---------|-------------------|-------------|--------|
| **HKEM** | 10 | SPECT: 12, PET: 9 | None (ordered) | Kernel + Emission | Standard |
| **dTNV** | 100 | Phantom-specific | SVRG | ✓ Directional | Standard |
| **TNV** | 100 | Phantom-specific | SVRG | ✗ Isotropic | Standard |
| **Log-dTNV** | 100 | Phantom-specific | SVRG | ✓ Directional | Log |
| **Log-TNV** | 100 | Phantom-specific | SVRG | ✗ Isotropic | Log |

### Key Differences

**HKEM vs TNV Variants:**
- **HKEM**: 10 epochs, ordered subsets (sequential), no variance reduction
  - Uses `base_hkem.yaml`
  - Two-stage: SPECT (12 subsets) → PET (9 subsets)
- **TNV variants**: 100 epochs, SVRG variance reduction
  - Uses `base_tnv.yaml`
  - Joint PET+SPECT reconstruction

**dTNV vs TNV:**
- **dTNV**: `directional_tnv: true` - CT anatomical edges guide regularization
- **TNV**: `directional_tnv: false` - Isotropic (no CT guidance)

**Log variants:**
- **Log-domain**: `use_log_tnv: true` - Better dynamic range handling
- **Standard**: `use_log_tnv: false` - Standard intensity domain

### Configuration Architecture

**Two-tier base configs:**
```
base_hkem.yaml  → HKEM experiments (10 epochs, ordered subsets)
base_tnv.yaml   → TNV experiments (100 epochs, SVRG)
```

**Composition order:**
```
Final Config = Base (HKEM or TNV) + Phantom + Algorithm + Overrides
```

**Example for HKEM:**
```
base_hkem.yaml (10 epochs, sequential)
  + phantom_1bpos_manc.yaml (data paths, resolution)
  + algo_hkem_spect.yaml (12 subsets, sigma values)
  = SPECT config
```

**Example for dTNV:**
```
base_tnv.yaml (100 epochs, SVRG)
  + phantom_1bpos_manc.yaml (data paths, resolution)
  + algo_dtnv.yaml (directional_tnv: true, alpha/beta)
  = dTNV config
```

### Directory Structure

```
experiments/
├── configs/
│   ├── base_hkem.yaml           # HKEM: 10 epochs, ordered subsets
│   ├── base_tnv.yaml            # TNV: 100 epochs, SVRG
│   ├── phantom_1bpos_manc.yaml  # Manchester NEMA
│   ├── phantom_1bpos_manc_short.yaml  # Manchester NEMA (bootstrap)
│   ├── phantom_1bpos_anthro.yaml
│   ├── phantom_1bpos_nema.yaml
│   ├── algo_hkem_spect.yaml     # 12 subsets
│   ├── algo_hkem_pet.yaml       # 9 subsets
│   ├── algo_dtnv.yaml
│   ├── algo_tnv.yaml
│   ├── algo_log_dtnv.yaml
│   └── algo_log_tnv.yaml
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick reference
├── STRUCTURE.md                  # Directory explanation
└── SUMMARY.md                    # This file
```

### Quick Start

```bash
# Test single experiment (dry-run)
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm hkem --dry-run

# Run single experiment
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv

# Run all algorithms for one phantom
python experiments/scripts/run_phantom_experiments.py --phantom manc --all-algorithms

# Run all 20 experiments
python experiments/scripts/run_phantom_experiments.py --batch-all
```

### Validation Results

✅ All configurations tested:
- HKEM: 10 epochs, sequential sampling, SPECT (12 subsets) + PET (9 subsets)
- dTNV: 100 epochs, SVRG, directional_tnv=true
- TNV: 100 epochs, SVRG, directional_tnv=false
- Log-dTNV: 100 epochs, SVRG, use_log_tnv=true, directional_tnv=true
- Log-TNV: 100 epochs, SVRG, use_log_tnv=true, directional_tnv=false

✅ Config composition works correctly:
- HKEM uses `base_hkem.yaml`
- All TNV variants use `base_tnv.yaml`

✅ Output organization:
```
output/
├── manc/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
├── manc_short/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
├── anthro/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
└── nema/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
```

### Parameter Summary

**HKEM (per stage):**
- Epochs: 10
- SPECT subsets: 12
- PET subsets: 9
- Sampling: sequential (ordered)
- Variance reduction: None (SGFunction)

**dTNV/TNV/Log variants:**
- Epochs: 100
- Subsets: phantom-specific (manc [14, 18], manc_short [18, 14], anthro [18, 18], nema [18, 18])
- Variance reduction: SVRG
- Initial step size: 1.0
- Relaxation eta: 0.01

### Ready to Run!

All configs are independent and self-contained in `experiments/`. Modify any parameter without affecting the main codebase.

See [README.md](README.md) for full documentation.
