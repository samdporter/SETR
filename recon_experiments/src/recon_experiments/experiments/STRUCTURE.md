# Experiments Directory Structure

This directory contains everything needed for the phantom reconstruction experiments.

## Directory Layout

```
experiments/
├── configs/                          # Self-contained experiment configs
│   ├── base_hkem.yaml                # Shared HKEM parameters
│   ├── base_tnv.yaml                 # Shared TNV parameters
│   ├── phantom_1bpos_manc.yaml       # Manchester NEMA settings
│   ├── phantom_1bpos_manc_short.yaml # Manchester NEMA bootstrap settings
│   ├── phantom_1bpos_anthro.yaml     # Anthropomorphic settings
│   ├── phantom_1bpos_nema.yaml       # NEMA settings
│   ├── algo_hkem_spect.yaml          # HKEM Stage 1 (SPECT)
│   ├── algo_hkem_pet.yaml            # HKEM Stage 2 (PET)
│   ├── algo_dtnv.yaml                # dTNV (directional_tnv: true)
│   ├── algo_tnv.yaml                 # TNV (directional_tnv: false)
│   ├── algo_log_dtnv.yaml            # Log-domain dTNV
│   └── algo_log_tnv.yaml             # Log-domain TNV
├── README.md                         # Full documentation
├── QUICKSTART.md                     # Quick reference
└── STRUCTURE.md                      # This file
```

## Key Differences from Main `configs/`

The configs in `experiments/configs/` are:
- **Self-contained**: All experiment parameters in one place
- **Independent**: Won't affect other workflows using `configs/`
- **Modifiable**: Easy to adjust for this specific experiment set
- **Versioned**: Can be tracked separately in git

## Config Composition

Configs are merged in this order:

```
base (hkem/tnv) → phantom → algorithm → CLI overrides
```

Example for "Manchester + dTNV":
1. Load `base_tnv.yaml` (num_epochs=100, SVRG, shared defaults)
2. Merge `phantom_1bpos_manc.yaml` (data paths, subsets, resolution)
3. Merge `algo_dtnv.yaml` (directional_tnv=true, alpha/beta)
4. Apply any `--override` flags

## Editing Configs

To modify parameters for all experiments:
- Edit `configs/base_hkem.yaml` (HKEM) or `configs/base_tnv.yaml` (TNV variants)

To modify phantom-specific settings:
- Edit `configs/phantom_1bpos_{phantom}.yaml`

To modify algorithm settings:
- Edit `configs/algo_{algorithm}.yaml`

## Running Experiments

All experiments are run via:
```bash
python experiments/scripts/run_phantom_experiments.py [options]
```

The script automatically:
1. Reads configs from `experiments/configs/`
2. Merges them appropriately
3. Saves temporary composed configs to `tmp/experiment_configs/`
4. Executes the appropriate reconstruction scripts

## Example Workflow

```bash
# 1. Review config files in experiments/configs/
ls experiments/configs/

# 2. Preview what would run (dry-run)
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv --dry-run

# 3. Run the experiment
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv

# 4. Check results
ls output/manc/dtnv/
```

## See Also

- [README.md](README.md) - Full documentation with algorithm details
- [QUICKSTART.md](QUICKSTART.md) - Quick command reference
