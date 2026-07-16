# Subset Selection Experiments

Experimental framework for testing different subset organization and prior update strategies in DTNV reconstruction with SVRG optimization.

The study runner now supports both 1-bed and 2-bed PET reconstruction spaces.
The 2-bed paired mode requires equal PET and SPECT subset counts; the provided 2-bed base config enforces that.

## Overview

This experiment investigates how different subset organization schemes and prior update frequencies affect convergence efficiency in stochastic variance-reduced gradient (SVRG) reconstruction. The key hypothesis is that all configurations should converge to the same solution (due to SVRG's variance reduction), but with different convergence rates.

### Experimental Factors

**1. Subset Organization (2 variants)**
- `separate`: 18 PET subsets + 18 SPECT subsets sampled independently (36 total)
- `paired`: 18 SumFunction(PET, SPECT) pairs sampled together (18 total)

**2. Prior Update Strategy (3 variants)**

All three keep the SVRG/SAGA sampler UNIFORM — the only regime in which CIL's
`num_functions` gradient scaling is unbiased — and encode the prior-update
frequency in the stochastic function list, which always sums exactly to
(data + prior). With N data functions:

- `folded`: prior/N folded into each data function (n = N). Full prior
  gradient difference enters every update.
- `half`: N copies of prior/N appended (n = 2N). Prior drawn with probability
  1/2 (every other update), each draw applying 2x the prior gradient difference.
- `epoch`: full prior appended as one extra function (n = N+1). Prior drawn
  with probability 1/(N+1) (about once per data pass), each draw applying
  (N+1)x the prior gradient difference.

Legacy modes (kept for reproducibility): `always` (prior outside the sampler,
evaluated every iteration) and `subset` (prior sampled with a NON-uniform
probability — BIASED with CIL's uniform gradient scaling, overweights the
prior by ~prior_updates_per_epoch; do not use for new experiments).

**3. Preconditioner Type**
- Defined in `parameters/precond_types.csv`

**4. Prior Weight**
- Defined in `parameters/gammas.csv`
- Current baseline-aligned setting: `gamma_tnv=0.01` with `alpha=beta=1`

### Preconditioner types

`parameters/precond_types.csv` lines may be `type` or `type,combine`; the
optional second column overrides `precond_combine` for that run (e.g.
`ls_block_diag,lehmer` with `lehmer_p`/`lehmer_scale` from the base config).

### Total Experiments

**Main experiments**: `subset_modes × prior_modes × precond_types × gammas` @ 100 epochs each

**Convergence references**: one run per gamma @ 1,000 epochs each
- Configuration: separate, always, mm_diag_block_maj

With the current parameter files this is 18 main runs (2 subset modes x 3 prior modes x 3 preconditioners) and 1 convergence reference per dataset.

## Directory Structure

```
recon_experiments/src/recon_experiments/studies/subset_selection/
├── configs/
│   ├── base_config_anthro.yaml          # 1bpos anthropomorphic phantom
│   ├── base_config_manc.yaml            # 1bpos bootstrap phantom
│   ├── base_config_2bpos.yaml           # 2bpos patient study
│   ├── sweep_main_experiments.yaml      # 1bpos main 36-run sweep
│   ├── sweep_main_experiments_2bpos.yaml # 2bpos main 36-run sweep
│   ├── sweep_convergence_ref.yaml       # 1bpos convergence reference sweep
│   └── sweep_convergence_ref_2bpos.yaml # 2bpos convergence reference sweep
├── parameters/
│   ├── subset_modes.csv                 # separate, paired
│   ├── prior_modes.csv                  # always, subset
│   ├── precond_types.csv                # Preconditioners to test
│   └── gammas.csv                       # Prior weights to test
├── scripts/
│   ├── run_subset_selection.py          # Main reconstruction script
│   ├── launch_sweep.sh                  # Cluster submission launcher
│   ├── subset_sweep.qsub.sh            # SGE array job script
│   └── test_local.sh                    # Quick local testing
├── output/                               # Results directory (created automatically)
└── README.md                             # This file
```

## Usage

### 1. Local Testing

Quick test with minimal epochs:

```bash
cd recon_experiments/src/recon_experiments/studies/subset_selection/scripts
./test_local.sh
```

Or run specific configuration:

```bash
python recon_experiments/src/recon_experiments/studies/subset_selection/scripts/run_subset_selection.py \
    --config recon_experiments/src/recon_experiments/studies/subset_selection/configs/base_config_anthro.yaml \
    --override \
        output_path=recon_experiments/src/recon_experiments/studies/subset_selection/output/test \
        num_epochs=2 \
        subset_mode=separate \
        prior_mode=always \
        precond_type=mm_diag_gershgorin_maj \
        gamma_tnv=0.01
```

Two-bed local example:

```bash
python recon_experiments/src/recon_experiments/studies/subset_selection/scripts/run_subset_selection.py \
    --config recon_experiments/src/recon_experiments/studies/subset_selection/configs/base_config_2bpos.yaml \
    --override \
        output_path=recon_experiments/src/recon_experiments/studies/subset_selection/output/test_2bpos \
        num_epochs=2 \
        subset_mode=paired \
        prior_mode=subset \
        precond_type=mm_diag_block_maj \
        gamma_tnv=0.01
```

### 2. Cluster Submission

**Main experiments (36 runs @ 100 epochs):**

```bash
cd recon_experiments/src/recon_experiments/studies/subset_selection/scripts
./launch_sweep.sh sweep_main_experiments.yaml full
```

Two-bed main experiments:

```bash
./launch_sweep.sh sweep_main_experiments_2bpos.yaml full
```

**Convergence references (3 runs @ 1,000 epochs):**

```bash
./launch_sweep.sh sweep_convergence_ref.yaml full
```

Two-bed convergence references:

```bash
./launch_sweep.sh sweep_convergence_ref_2bpos.yaml full
```

**Test modes:**

```bash
# Submit one job to cluster
./launch_sweep.sh sweep_main_experiments.yaml test

# Run one job locally
./launch_sweep.sh sweep_main_experiments.yaml local
```

### 3. Monitoring Progress

Check job status:
```bash
qstat -u $USER
```

View logs:
```bash
tail -f recon_experiments/src/recon_experiments/studies/subset_selection/output/subset_selection_main/_logs/*.o*
```

Check completion status:
```bash
find recon_experiments/src/recon_experiments/studies/subset_selection/output/subset_selection_main -name "job_completion.txt" | xargs cat
```

## Configuration Details

### Data

- **1bpos anthropomorphic config**:
  PET: `/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/PET/phantom_short`
  SPECT: `/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140`
- **2bpos config**:
  PET: `/home/storage/prepared_data/oxford_patient_data/sirt3/PET`
  SPECT: `/home/storage/prepared_data/oxford_patient_data/sirt3/SPECT`

### Reconstruction Parameters

- **1bpos subsets**: [18, 18] for PET and SPECT
- **2bpos subsets**: [9, 9] by default so paired mode is well-defined
- **Alpha = Beta = 1**, with `gamma_tnv=0.01` to align the subset references with the preconditioner baselines
- **Kappa weighting**: Disabled in the anthropomorphic base config
- **Prior**: DTNV (vectorial TV) only, no modality-specific priors
- **Resolution modeling**:
  - PET: [5.61, 4.83, 4.93] mm FWHM
  - SPECT: [6.8, 6.8, 6.8] mm FWHM + collimator model
- **Step size**: Initial 1.0, decay 0.01
- **SVRG snapshot interval**: 2 × num_subsets

### Resource Requirements

**Main experiments:**
- Runtime: 24 hours
- Memory: 32GB
- GPU: Required
- Cores: 1

**Convergence references:**
- Runtime: 168 hours (7 days)
- Memory: 60GB
- GPU: Required
- Cores: 1

## Expected Outputs

Each reconstruction produces:

```
output/<sweep_name>/<config_name>/
├── initial_image_0.hv              # Initial PET image
├── initial_image_1.hv              # Initial SPECT image
├── s_inv_0.hv                      # PET sensitivity^-1
├── s_inv_1.hv                      # SPECT sensitivity^-1
├── kappa_sq_0.hv                   # PET kappa-squared weights
├── kappa_sq_1.hv                   # SPECT kappa-squared weights
├── image_*_*.hv                    # Reconstructed images at intervals
├── objective.csv                   # Objective function values
├── args.csv                        # Full parameter record
├── job_completion.txt              # Job status info
└── tmp/                            # Working directory
```

## Analysis Workflow

1. **Convergence verification**: Compare main experiments vs convergence references
2. **Efficiency comparison**: Iterations to reach threshold vs convergence reference
3. **Subset organization**: paired vs separate convergence rates
4. **Prior frequency**: always vs subset update convergence rates
5. **Preconditioner impact**: How preconditioner type affects convergence with different subset schemes

## Implementation Notes

### SVRG Gradient Equivalence

All configurations see approximately the same gradient through SVRG's variance reduction:

```
g = g_sampled_subset - snapshot_sampled_subset + snapshot_full
```

The snapshot computes full gradient (all data + prior), so configurations differ only in:
- Variance of stochastic gradient
- Correlation between PET/SPECT updates
- Preconditioner-subset interaction

### Prior Probability Calculations

**Paired + Prior as subset:**
- 18 paired data functions + 1 prior = 19 stochastic functions
- Target: 1 prior eval per 2 data subset evals
- prob(prior) = 1/2, prob(each pair) = 1/36
- Check: 0.5 + 18×(1/36) = 0.5 + 0.5 = 1.0 ✓

**Separate + Prior as subset:**
- 18 PET + 18 SPECT + 1 prior = 37 functions
- Target: 1 prior eval per 3 data subset evals
- prob(prior) = 1/3, prob(each data) = 1/54
- Check: (1/3) + 36×(1/54) = (1/3) + (2/3) = 1.0 ✓

## Troubleshooting

**Import errors:**
- Check SIRF environment is activated: `source ~/sirf_venv/bin/activate`
- Verify SIRF installation: `source $INSTALLDIR/bin/env_sirf.sh`

**GPU errors:**
- Check GPU availability: `nvidia-smi`
- Review logs for ECC errors or OOM kills

**Job failures:**
- Check `job_completion.txt` for failure reason
- Review SGE logs in `output/<sweep_name>/_logs/`
- Verify data paths are accessible

**Convergence issues:**
- Check objective.csv for trends
- Verify kappa images are reasonable
- Review preconditioner choice for prior configuration

## Documentation

- **Metrics Usage**: See [docs/guides/metrics.md](../../docs/guides/metrics.md) for detailed guide on using image quality metrics
- **Subset Selection**: This README covers the subset selection experimental framework
- **SETR Main Docs**: See [docs/](../../docs/) for general SETR documentation

## Contact

For questions or issues, check the main SETR documentation or contact the development team.
