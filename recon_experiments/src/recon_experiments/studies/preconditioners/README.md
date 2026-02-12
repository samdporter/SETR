# Preconditioner Experiments

This directory contains the complete workflow for comparing preconditioner performance
in synergistic PET/SPECT reconstruction.

## Quick Start

**See [WORKFLOW.md](WORKFLOW.md) for detailed instructions.**

### 1. Run Baseline Reconstructions (Stage 1)
```bash
./launch_baseline_recons.sh config_1bpos_anthro.yaml 200 full
```
Baseline runs default to `mm_block_diag` with `precond_combine=harmonic` and `block_scalar_reduction=diag`
(set `PRECOND_TYPE` / `PRECOND_COMBINE` / `PRECOND_SCALAR_REDUCTION` to override).

### 2. Run Preconditioner Sweep (Stage 2)
```bash
./launch_precond_sweep.sh precond_sweep_1bpos.yaml full
```

### 3. Analyze Results (Stage 3)
```bash
python scripts/analyze_precond_sweep.py --sweep precond_1bpos --baseline baselines_1bpos --watch
```

### 4. Check Status Anytime
```bash
./check_status.sh
```

## Why Two Stages?

1. **Baselines**: Establish reference solutions for each alpha value using long, accurate reconstructions
2. **Sweep**: Test different preconditioners and step sizes, measuring convergence speed to baseline
3. **Analysis**: Compare performance, identify optimal configurations

This design allows you to test which preconditioners are:
- Fastest to converge
- Most robust to step size selection
- Most stable across different alpha values (prior weighting)

## Documentation

- **[WORKFLOW.md](WORKFLOW.md)** - Complete workflow guide
- **[docs/guides/preconditioners_cluster.md](../../docs/guides/preconditioners_cluster.md)** - Additional cluster information
