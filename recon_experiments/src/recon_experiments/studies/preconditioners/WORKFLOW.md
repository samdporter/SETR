# Preconditioner Experiment Workflow

This directory contains the complete workflow for comparing preconditioner performance in synergistic PET/SPECT reconstruction.

## Overview

The experiment has **two stages**:

1. **Baseline reconstructions**: Run long reconstructions (default `mm_block_diag`) to establish reference solutions for each alpha value
2. **Preconditioner sweep**: Test different preconditioners and step sizes, comparing convergence to the baseline

## Stage 1: Baseline Reconstructions

### Purpose
Establish reference solutions by running long reconstructions for each alpha value using the tested `run_dtnv_1bpos.py` script.

### Scripts
- Uses `runners/scripts/run_dtnv_1bpos.py` (tested production script)
- `launch_baseline_recons.sh` - Launch baselines for all alpha values
- `scripts/baseline_recon.qsub.sh` - SGE job script for cluster

### Usage

**Test locally (one alpha):**
```bash
./launch_baseline_recons.sh config_1bpos_anthro.yaml 20 local
```

**Submit to cluster (all alphas):**
```bash
./launch_baseline_recons.sh config_1bpos_anthro.yaml 200 full
```

### Parameters
- **Config**: `config_1bpos_anthro.yaml` (uses full phantom data)
- **Epochs**: 200 (longer than sweep to ensure convergence)
- **Alpha values**: Read from `parameters/alphas.csv`
- **Script**: Uses the proven `run_dtnv_1bpos.py` with config overrides
- **Preconditioner**: `mm_block_diag` by default (override via `PRECOND_TYPE`)
- **Combine**: `harmonic` by default (override via `PRECOND_COMBINE`)
- **Block scalar reduction**: `diag` by default (override via `PRECOND_SCALAR_REDUCTION`)

### Output
```
output/baselines_1bpos/
├── baseline_alpha_0.005/
│   ├── baseline_metrics.json     # Quick reference metrics
│   ├── result.csv                # Full result summary
│   ├── objective.csv             # Convergence history
│   ├── image_*.hv                # Reconstructed images
│   ├── preconditioner_*.hv       # Preconditioner snapshots
│   └── preconditioner_test.log   # Detailed log
├── baseline_alpha_0.05/
│   └── ...
└── baseline_alpha_0.5/
    └── ...
```

### Resource Requirements
- **Runtime**: 72 hours per baseline
- **Memory**: 95GB
- **GPU**: Required
- **Total**: 3 baselines × 72h = 216 CPU-hours

## Stage 2: Preconditioner Sweep

### Purpose
Compare different preconditioners and step sizes by measuring convergence speed to the baseline solutions.

### Scripts
- `scripts/run_precond_sweep_single.py` - Wrapper that calls `run_dtnv_1bpos.py` / `run_dtnv_2bpos.py`
- `scripts/precond_sweep.qsub.sh` - SGE job script for each test
- `launch_precond_sweep.sh` - Launch all sweep jobs

### Usage

**Already configured!** The existing sweep setup is correct:
```bash
./launch_precond_sweep.sh precond_sweep_1bpos.yaml full
```

### Current Configuration
- **Preconditioners**: Defined in `parameters/precond_types.csv` (includes combine modes)
- **Alphas**: Defined in `parameters/alphas.csv`
- **Step sizes**: Defined in `parameters/step_sizes.csv`
- **Epochs per job**: 50 (shorter than baseline)

### Why This Works

1. **Same objective function**: All tests use identical objective settings
2. **Only preconditioner differs**: Each test uses its specified preconditioner and combine mode
3. **Fair comparison**: Same alpha, same objective, only convergence speed varies
4. **Alpha range**: Tests stability to prior weighting (your goal!)

### Output
```
output/precond_1bpos/
├── precond_mm_block_diag_combine_harmonic_alpha_0.005_step_0.05/
├── precond_mm_block_diag_combine_harmonic_alpha_0.005_step_0.1/
├── ...
└── precond_ls_block_diag_combine_harmonic_alpha_0.5_step_5.0/
```

## Stage 3: Analysis

### Purpose
Compare preconditioner performance against baselines to identify optimal configurations.

### Script
- `scripts/analyze_precond_sweep.py` - Comprehensive analysis

### Usage

**After baselines and sweep complete:**
```bash
cd recon_experiments/src/recon_experiments/studies/preconditioners

# Run analysis
python scripts/analyze_precond_sweep.py \
    --sweep precond_1bpos \
    --baseline baselines_1bpos \
    --convergence-threshold 0.01

# Or monitor in real-time during sweep
python scripts/analyze_precond_sweep.py \
    --sweep precond_1bpos \
    --baseline baselines_1bpos \
    --watch \
    --watch-interval 300
```

### Analysis Outputs

```
output/precond_1bpos_analysis/
├── analysis_results.csv         # All metrics for each test
├── summary_report.md            # Human-readable summary
├── convergence_alpha_0.005.png  # Convergence curves per alpha
├── convergence_alpha_0.05.png
└── convergence_alpha_0.5.png
```

### Metrics Computed

For each test:
- **Convergence speed**: Iterations to reach baseline objective (within 1%)
- **Final accuracy**: Objective gap vs baseline
- **Image error**: RMSE vs baseline reconstruction
- **Speedup**: Runtime ratio vs baseline
- **Stability**: Success rate across step sizes

## Complete Workflow

### 1. Run Baselines (Do This First!)
```bash
# Submit baseline jobs (3 alphas × 200 epochs each)
./launch_baseline_recons.sh config_1bpos_anthro.yaml 200 full

# Monitor
qstat -u $USER | grep baseline
```

### 2. Wait for Baselines to Complete
Check for completion files:
```bash
ls -la output/baselines_1bpos/*/baseline_metrics.json
```

### 3. Run Preconditioner Sweep
```bash
# Submit sweep jobs (60 jobs × 48h each)
./launch_precond_sweep.sh precond_sweep_1bpos.yaml full

# Monitor
qstat -u $USER | grep precond
```

### 4. Analyze Results
```bash
# Watch mode (updates every 5 minutes)
python scripts/analyze_precond_sweep.py \
    --sweep precond_1bpos \
    --baseline baselines_1bpos \
    --watch \
    --watch-interval 300
```

### 5. Review Results
```bash
# View summary
cat output/precond_1bpos_analysis/summary_report.md

# View detailed results
less output/precond_1bpos_analysis/analysis_results.csv

# View convergence plots
xdg-open output/precond_1bpos_analysis/convergence_alpha_0.005.png
```

## Parameter Files

All parameters are in `parameters/`:

- **alphas.csv**: Alpha values to test (currently: 0.005, 0.05, 0.5)
- **step_sizes.csv**: Step sizes to test (currently: 0.05, 0.1, 0.5, 1.0, 5.0)
- **precond_types.csv**: Preconditioner types (currently: 5 types)

**DO NOT modify these** without re-running baselines!

## Configuration Files

- **configs/precond_sweep_1bpos.yaml**: Sweep configuration (epochs, resources, etc.)
- **../../configs/config_1bpos_anthro.yaml**: Base reconstruction config (uses full phantom data)

## Questions the Analysis Answers

1. **Which preconditioner is fastest?** → Minimum `iterations_to_convergence`
2. **Which is most robust to step size?** → Success rate across step sizes
3. **Which is most robust to alpha?** → Consistent performance across alphas
4. **What's the optimal step size per preconditioner?** → Best `iterations_to_convergence` for each
5. **Is accuracy compromised?** → `final_obj_gap` and `image_rmse` vs baseline

## Notes

- **Baseline must complete first**: Sweep results can't be analyzed without baselines
- **Same alpha for comparison**: Each sweep test is compared to its corresponding baseline
- **Objective is identical**: Only preconditioner affects convergence speed, not solution
- **Alpha tests stability**: Different alphas test robustness to prior weighting

## Troubleshooting

**Baseline jobs failing?**
```bash
# Check logs
tail -f output/baselines_1bpos/_logs/*.o*

# Test locally first
PRECOND_TYPE=mm_block_diag ./launch_baseline_recons.sh config_1bpos_anthro.yaml 200 local
```

**Analysis script errors?**
```bash
# Make sure baselines exist
ls output/baselines_1bpos/*/baseline_metrics.json

# Check sweep results
ls output/precond_1bpos/*/result.csv | wc -l
```

**Missing dependencies?**
```bash
pip install pandas matplotlib tabulate
```
