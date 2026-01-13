# Subset Selection Experiments

Experimental framework for testing different subset organization and prior update strategies in DTNV reconstruction with SVRG optimization.

## Overview

This experiment investigates how different subset organization schemes and prior update frequencies affect convergence efficiency in stochastic variance-reduced gradient (SVRG) reconstruction. The key hypothesis is that all configurations should converge to the same solution (due to SVRG's variance reduction), but with different convergence rates.

### Experimental Factors

**1. Subset Organization (2 variants)**
- `separate`: 18 PET subsets + 18 SPECT subsets sampled independently (36 total)
- `paired`: 18 SumFunction(PET, SPECT) pairs sampled together (18 total)

**2. Prior Update Strategy (2 variants)**
- `always`: Prior in outer SumFunction, evaluated every iteration
- `subset`: Prior as separate function in SVRG sampler with adjusted probability

**3. Preconditioner Type (3 variants)**
- `bsrem`: Basic BSREM preconditioner only
- `vtv_svd_principal_alpha`: VTV preconditioner with SVD principal eigenvalue
- `vtv_frobenius_surrogate_pd`: VTV preconditioner with Frobenius surrogate

**4. Prior Weight (3 values)**
- gamma_tnv: 10, 100, 1000

### Prior Sampling Probabilities

To maintain approximately equal prior gradient evaluations across configurations:

- **separate + always**: 1 prior eval per 1 data subset eval
- **paired + subset**: prob(prior) = 1/2, ratio 1:2 data
- **separate + subset**: prob(prior) = 1/3, ratio 1:3 data

### Total Experiments

**Main experiments**: 2 × 2 × 3 × 3 = **36 runs** @ 100 epochs each

**Convergence references**: 3 runs @ 10,000 epochs each
- Configuration: separate, always, vtv_svd_principal_alpha
- Gamma values: 10, 100, 1000

**Grand total**: **39 reconstruction runs**

## Directory Structure

```
functionality/subset_selection/
├── configs/
│   ├── base_config_anthro.yaml          # Base config for anthropomorphic phantom
│   ├── sweep_main_experiments.yaml      # Main 36-run sweep
│   └── sweep_convergence_ref.yaml       # Convergence reference sweep
├── parameters/
│   ├── subset_modes.csv                 # separate, paired
│   ├── prior_modes.csv                  # always, subset
│   ├── precond_types.csv                # bsrem, vtv_svd_principal_alpha, vtv_frobenius_surrogate_pd
│   └── gammas.csv                       # 10, 100, 1000
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
cd functionality/subset_selection/scripts
./test_local.sh
```

Or run specific configuration:

```bash
python scripts/run_subset_selection.py \
    --config functionality/subset_selection/configs/base_config_anthro.yaml \
    --override \
        output_path=output/test \
        num_epochs=2 \
        subset_mode=separate \
        prior_mode=always \
        precond_type=bsrem \
        gamma_tnv=10
```

### 2. Cluster Submission

**Main experiments (36 runs @ 100 epochs):**

```bash
cd functionality/subset_selection/scripts
./launch_sweep.sh sweep_main_experiments.yaml full
```

**Convergence references (3 runs @ 10,000 epochs):**

```bash
./launch_sweep.sh sweep_convergence_ref.yaml full
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
tail -f functionality/subset_selection/output/subset_selection_main/_logs/*.o*
```

Check completion status:
```bash
find functionality/subset_selection/output/subset_selection_main -name "job_completion.txt" | xargs cat
```

## Configuration Details

### Data

- **PET**: `/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/PET/phantom_short`
- **SPECT**: `/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140`
- Single bed position reconstruction

### Reconstruction Parameters

- **Subsets**: [18, 18] for PET and SPECT
- **Alpha = Beta = 1** (with gradient energy scaling applied to alpha)
- **Kappa weighting**: Enabled
- **Prior**: DTNV (vectorial TV) only, no modality-specific priors
- **Resolution modeling**:
  - PET: [5.1, 4.9, 4.9] mm FWHM
  - SPECT: [6.7, 6.7, 6.7] mm FWHM + collimator model
- **Step size**: Initial 0.1, decay 0.02
- **SVRG snapshot interval**: 2 × num_subsets

### Resource Requirements

**Main experiments:**
- Runtime: 24 hours
- Memory: 32GB
- GPU: Required
- Cores: 1

**Convergence references:**
- Runtime: 168 hours (7 days)
- Memory: 32GB
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
- 18 pairs + 1 prior = 19 functions
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
