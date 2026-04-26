# Preconditioner Experiments

This directory contains the complete workflow for comparing preconditioner performance
in synergistic PET/SPECT reconstruction.

## Quick Start

**See [WORKFLOW.md](WORKFLOW.md) for detailed instructions.**

### 1. Run Baseline Reconstructions (Stage 1)
```bash
./launch_baseline_recons.sh config_1bpos_anthro.yaml 1000 full
```
For 2 bed positions:
```bash
./launch_baseline_recons.sh config_2bpos.yaml 1000 full
```
Baseline runs default to `mm_diag_block_maj` with `precond_combine=majoriser` and `block_scalar_reduction=diag`
(set `PRECOND_TYPE` / `PRECOND_COMBINE` / `PRECOND_SCALAR_REDUCTION` to override).
`precond_combine=harmonic` is accepted for compatibility and mapped to `majoriser`.

### 2. Run Preconditioner Sweep (Stage 2)
```bash
./launch_precond_sweep.sh precond_sweep_1bpos.yaml full
```
For 2 bed positions:
```bash
./launch_precond_sweep.sh precond_sweep_2bpos.yaml full
```
For repeated stochastic trials per parameter setting:
```bash
SWEEP_REPEATS=5 ./launch_precond_sweep.sh precond_sweep_1bpos.yaml full
```

### Single Command: Baseline then Sweep
```bash
./run_baseline_then_sweep.sh config_1bpos_anthro.yaml 1000 full precond_sweep_1bpos.yaml full 300
```
For fully local sequential execution:
```bash
./run_baseline_then_sweep.sh config_1bpos_anthro.yaml 1000 local_all precond_sweep_1bpos.yaml local_all
```
When `SWEEP_REPEATS` is unset, this orchestrator defaults to `SWEEP_REPEATS=5` for `local_all`, `test`, and `full` sweep modes.
Arguments are:
- baseline config
- baseline epochs
- baseline mode (`local|local_all|test|full`)
- sweep config
- sweep mode (`local|local_all|test|full`)
- poll interval seconds (optional, default `300`) for waiting on baseline completion

### 3. Analyze Results (Stage 3)
```bash
python scripts/analyze_precond_sweep.py --sweep precond_1bpos --baseline baselines_1bpos --watch
```
For 2 bed positions, use the corresponding `precond_2bpos` / `baselines_2bpos` directories.

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
