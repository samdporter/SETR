# Preconditioner Comparison Tests

The `functionality/preconditioners/` directory contains scripts for testing different VTV preconditioner methods on the cluster or locally. This guide centralises the instructions that were previously scattered across the repository.

## Quick Start

### 1. Local Test (Single Run)

Test one configuration locally before submitting to the cluster:

```bash
cd functionality/preconditioners
./launch_precond_sweep.sh precond_sweep_2bpos.yaml local
```

This runs the first parameter combination (bsrem, alpha=50, step=0.05) on your local machine.

### 2. Cluster Test (Single Job)

Submit one test job to the cluster to verify SGE configuration:

```bash
./launch_precond_sweep.sh precond_sweep_2bpos.yaml test
```

### 3. Full Sweep

Submit all parameter combinations to the cluster:

```bash
./launch_precond_sweep.sh precond_sweep_2bpos.yaml full
```

**Total jobs**: 5 precond types × 7 alphas × 4 step sizes = **140 jobs**

## Directory Structure

```
functionality/preconditioners/
├── configs/
│   └── precond_sweep_2bpos.yaml    # Sweep configuration
├── parameters/
│   ├── precond_types.csv           # Preconditioner types to test
│   ├── alphas.csv                  # Alpha values
│   └── step_sizes.csv              # Step sizes
├── scripts/
│   └── precond_sweep.qsub.sh       # SGE job script
├── output/                         # Results (created during runs)
│   └── precond_2bpos/
│       └── precond_*/              # Individual test outputs
├── launch_precond_sweep.sh         # Main launcher
└── README.md                       # This file
```

## Preconditioner Methods

The sweep tests 5 preconditioner methods:

1. **bsrem** - Baseline (no VTV preconditioning, data-fidelity only)
2. **vtv_svd_principal_alpha** *(legacy name `vtv_slow`)* — SVD principal rank-one terms + isotropic α
3. **vtv_mm_jensen** *(legacy `vtv_fast`)* — MM/Jensen surrogate, SVD-free, ~3× faster
4. **vtv_frobenius_surrogate_pd** *(legacy `vtv_fastest_positive`)* — Frobenius surrogate, guaranteed positive
5. **vtv_vector_tv_per_modality** *(legacy `vtv_fastest_exact`)* — Per-modality vector TV (exact radial)

## Parameter Files

Edit CSV files in `parameters/` to customize the sweep:

### precond_types.csv
```csv
precond_type
bsrem
vtv_svd_principal_alpha
vtv_mm_jensen
vtv_frobenius_surrogate_pd
vtv_vector_tv_per_modality
```

### alphas.csv
```csv
alpha
50.0
100.0
500.0
1000.0
5000.0
10000.0
50000.0
```

### step_sizes.csv
```csv
step_size
0.05
0.1
0.5
1.0
```

## Configuration

The sweep configuration is in `configs/precond_sweep_2bpos.yaml`:

```yaml
sweep_name: "precond_2bpos"
base_config: "config_2bpos.yaml"
script: "test_preconditioner_single.py"

sge:
  runtime: "48:00:00"
  memory: "95G"
  cores: 4
  gpu: true

parameters:
  precond_types_file: "precond_types.csv"
  alphas_file: "alphas.csv"
  step_sizes_file: "step_sizes.csv"

fixed_params:
  num_epochs: 50
```

## Output Structure

Each test creates an output directory:

```
output/precond_2bpos/precond_vtv_mm_jensen_alpha_500.0_step_0.1/
├── image_*.hv                  # Reconstructed images at different iterations
├── objective.csv               # Objective function values
├── preconditioner_*.hv         # Preconditioner images
├── result.csv                  # Test summary (status, time, final objective)
├── preconditioner_test.log     # Detailed log
├── job_completion.txt          # Job completion status
└── kappas/                     # Kappa squared images
    ├── pet_kappa_sq.hv
    └── spect_kappa_sq.hv
```

## Monitoring Jobs

```bash
# Check SGE queue
qstat -u $USER

# Count running/pending jobs
qstat -u $USER | grep precond | wc -l

# View specific job details
qstat -j <job_id>

# Check logs
tail -f output/precond_2bpos/_logs/precond_*.o*
```

## Collecting Results

After jobs complete, collect results into a summary:

```bash
cd functionality/preconditioners
python ../../scripts/collect_precond_results.py --sweep precond_2bpos
```

This will create `output/precond_2bpos_summary.csv` with all test results.

## Analysis

Compare preconditioner performance:

```bash
# View summary sorted by final objective
python ../../scripts/analyze_preconditioner_tests.py \
    --results output/precond_2bpos_summary.csv
```

## Resource Requirements

- **Memory**: 95GB (for 2 bed position data)
- **Runtime**: 48 hours per job
- **GPU**: Required
- **Disk**: ~10GB per job

Total sweep requirements:
- 140 jobs × 95GB = ~13.3TB peak memory (across cluster)
- 140 jobs × 48h = 6720 CPU-hours (if sequential)

## Customization

### Quick Test Subset

For faster testing, create a subset in `parameters/`:

```csv
# alphas_quick.csv
alpha
500.0
5000.0

# step_sizes_quick.csv
step_size
0.1
0.5

# precond_types_quick.csv
precond_type
bsrem
vtv_mm_jensen
```

Then create `configs/precond_sweep_2bpos_quick.yaml` pointing to these files.

This reduces to: 2 precond × 2 alphas × 2 steps = **8 jobs** (e.g., `bsrem` and `vtv_mm_jensen`).

### Single Bed Position

For faster tests, create `configs/precond_sweep_1bpos.yaml`:

```yaml
sweep_name: "precond_1bpos"
base_config: "config_1bpos.yaml"
script: "test_preconditioner_single.py"

sge:
  runtime: "24:00:00"
  memory: "60G"
  cores: 4
  gpu: true
# ... rest same as 2bpos
```

## Troubleshooting

### Local Test Fails

If `./launch_precond_sweep.sh local` fails:
1. Check you have the correct Python environment activated
2. Verify SIRF is properly installed
3. Check data paths in `configs/config_2bpos.yaml`

### Jobs Fail on Cluster

Check the log files in `output/precond_2bpos/_logs/`:

```bash
# Find failed jobs
grep -l "FAILURE" output/precond_2bpos/_logs/*.o*

# Check error patterns
grep "Error\|error\|FAILED" output/precond_2bpos/_logs/*.o*
```

Common issues:
- Out of memory → Increase `sge.memory` in config
- Timeout → Increase `sge.runtime` in config
- GPU errors → Check GPU health on nodes

### Results Missing

If `result.csv` is missing from output directories:
- Check if reconstruction completed successfully
- Look for errors in `preconditioner_test.log`
- Verify `job_completion.txt` shows `status=completed`

## Comparison with Original Test Script

This cluster-friendly version differs from `scripts/test_preconditioners.py`:

| Aspect | test_preconditioners.py | test_preconditioner_single.py |
|--------|------------------------|------------------------------|
| Setup | Shared once for all tests | Fresh for each test |
| Execution | Sequential batch | Parallel on cluster |
| Parameters | Loops internally | Single combination per run |
| Use case | Local development | Production cluster sweeps |
