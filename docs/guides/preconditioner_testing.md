# VTV Preconditioner Testing

## Overview

The scripts under `scripts/` and `functionality/preconditioners/` provide tooling to comprehensively test VTV preconditioning strategies:

1. **svd_principal_alpha** *(legacy `SLOW`)*: SVD principal rank-one terms + isotropic α (baseline)
2. **mm_jensen** *(legacy `FAST`)*: MM/Jensen surrogate, SVD-free (default, good tradeoff)
3. **frobenius_surrogate_pd** *(legacy `FASTEST_POSITIVE`)*: Frobenius surrogate, guaranteed PD
4. **vector_tv_per_modality** *(legacy `FASTEST_EXACT`)*: Per-modality vector TV (exact radial, may not be PD)

## Scripts

### `test_vtv_preconditioners_synthetic.py` ⭐ NEW

Standalone test on synthetic 3D geometric data. Tests all 4 methods without requiring real data.

**Usage:**
```bash
# Basic usage (default: 64×64×32 voxels, Charbonnier)
python scripts/test_vtv_preconditioners_synthetic.py

# Custom parameters
python scripts/test_vtv_preconditioners_synthetic.py \
    --shape 128 128 64 \
    --voxel-size 1.5 1.5 2.0 \
    --delta 0.01 \
    --smoothing charbonnier \
    --output results/my_test
```

**Output:**
- `results.csv`: Comparison table with all metrics
- `preconditioner_comparison.png`: Visual comparison of all methods
- `timing_comparison.png`: Computation time bar charts

**What it validates:**
1. Objective/gradient values are identical (✓)
2. Hessian diagonal approximation accuracy
3. Positive definiteness of preconditioners
4. Computation speed (speedup vs `slow`)
5. Visual comparison of preconditioner fields

**Runtime:** ~1-5 minutes (depends on image size)

### `test_preconditioners.py`

Full reconstruction test on real PET/SPECT data. Main testing script that:
- Loads data and sets up objectives **once** (expensive operations)
- Tests multiple preconditioner configurations with varying:
  - Penalty strengths: α = β ∈ {1, 10, 100, 500}
  - Initial step sizes: {0.01, 0.05, 0.1, 0.3, 0.5, 1.0}
- Runs 50 epochs per configuration
- Saves results to CSV with timing and convergence metrics

**Usage:**
```bash
# Default settings (uses config_2bpos.yaml)
python scripts/test_preconditioners.py

# Custom settings
python scripts/test_preconditioners.py \
    --config configs/config_2bpos.yaml \
    --output results/my_precond_test \
    --alphas 1 10 100 500 \
    --step-sizes 0.01 0.1 0.5 1.0 \
    --precond-types bsrem vtv_mm_jensen vtv_svd_principal_alpha
```

**Output:**
- `results/preconditioner_tests/results.csv`: All test results
- `results/preconditioner_tests/run_XXXX/`: Individual run outputs
- `results/preconditioner_tests/preconditioner_tests.log`: Full log

### `analyze_preconditioner_tests.py`

Analysis and visualization script that generates:
- Summary table of best configurations
- Objective vs parameters plots
- Heatmaps for each preconditioner
- Convergence and runtime comparisons
- Detailed comparison of optimal settings

**Usage:**
```bash
# Analyze default results
python scripts/analyze_preconditioner_tests.py

# Analyze custom results
python scripts/analyze_preconditioner_tests.py \
    --results results/my_precond_test/results.csv \
    --output results/my_precond_test/analysis
```

**Output Plots:**
- `objective_vs_params.png`: Best objective vs α and step size
- `objective_heatmaps.png`: 2D heatmap for each preconditioner
- `convergence_runtime.png`: Convergence metrics and runtime distributions
- `precond_setup_time.png`: Preconditioner computation cost
- `detailed_comparison.png`: 4-panel comparison of key metrics
- `summary_table.txt`: Text summary of best configurations

## Preconditioner Types

### `bsrem` (Baseline)
BSREM preconditioner only:
```
P = x / (A^T 1 + ε)
```
where A is the acquisition operator.

### `vtv_svd_principal_alpha` (SVD principal + isotropic α)
BSREM + SVD-based VTV diagonal approximation:
```
P_inv = Lehmer_mean(P_bsrem, P_vtv_svd_principal_alpha)

P_vtv_svd_principal_alpha[i] = Σ_k φ''(σ_k) · [(J^T w u_k v_k^T)_i]^2
                              + α_j · b_{j,i}^2 · Σ_dir S_{j,i,dir}^2

where:
  σ_k, u_k, v_k from SVD: A = U Σ V^T
  α_j = Σ_k φ'(σ_k)/σ_k (isotropic component)
```

**Pros:**
- Captures principal spectral curvature exactly
- Adds isotropic α term for radial curvature
- Best accuracy for validation

**Cons:**
- Slow: O(n·M^3·d^3) - requires SVD per voxel
- Memory intensive for large M
- Still diagonal (omits cross terms)

### `vtv_mm_jensen` (MM/Jensen surrogate)
BSREM + MM/Jensen VTV diagonal approximation:
```
P_inv = Lehmer_mean(P_bsrem, P_vtv_mm_jensen)

P_vtv_mm_jensen[j,m] = η · ω_j · S_{j,m} · b_{j,m}^2

where:
  ω_j = Σ_ℓ φ'(σ_ℓ)/σ_ℓ        (Hessian surrogate, SVD-free)
  S_{j,m} = Σ_d scale_d^2 · n_dir  (Jacobian sensitivity)
```

**Pros:**
- Fast: No SVD required
- Guaranteed positive-definite (MM surrogate)
- Good accuracy for most problems

**Cons:**
- Approximate: aggregates singular value information
- Less accurate for highly ill-conditioned problems

### `vtv_frobenius_surrogate_pd` (Frobenius surrogate)
BSREM + Frobenius-based positive-definite approximation:
```
P_inv = Lehmer_mean(P_bsrem, P_vtv_frobenius_surrogate_pd)

P_vtv[j,m] = η · [M · φ'(||A_j||_F) / ||A_j||_F] · S_{j,m} · b_{j,m}^2

where:
  ||A_j||_F = Frobenius norm at voxel j (cheap)
  M = number of modalities
```

**Pros:**
- Fastest: No SVD, uses Frobenius norm
- Guaranteed positive-definite
- Best for large-scale problems

**Cons:**
- Cruder approximation than `fast`
- Uses rank estimate M

### `vtv_vector_tv_per_modality` (Per-modality vector TV)
BSREM + Per-modality vector-norm exact preconditioner:
```
P_inv = Lehmer_mean(P_bsrem, P_vtv_vector_tv_per_modality)

P_vtv[j,m] = Σ_dir [ b_{j,m}^2 · S_{j,m,dir}^2 · (α_{j,m} + β_{j,m} · U_{j,m,dir}^2 / (r_{j,m}^2 + ε)) ]

where:
  r_{j,m} = ||A_{j,m,·}||_2,
  α_{j,m} = φ'(r_{j,m}) / r_{j,m},
  β_{j,m} = φ''(r_{j,m}) − α_{j,m}
```

**Pros:**
- Fast: No SVD (per-modality vector norms)
- Uses exact radial structure of vector TV
- Good for research/comparison

**Cons:**
- Not guaranteed positive-definite
- Ignores cross-modal coupling

## Test Configuration

From `config_2bpos.yaml`:
- **Data**: Oxford patient (SIRT3), 2 bed positions
- **Subsets**: PET=9, SPECT=12
- **Epochs**: 50 (test) vs 10 (default)
- **Prior**: TNV with directional guidance (26-stencil)
- **Smoothing**: Fair function with δ = max(image)/1e4 · α

## Expected Runtime

Per test run (50 epochs):
- BSREM: ~30-45 min
- VTV mm_jensen: ~40-60 min
- VTV svd_principal_alpha: ~90-120 min

Full grid (3 precond × 4 α × 6 step = 72 runs): **~3-5 days**

## Recommendations

### Quick Test
```bash
python scripts/test_preconditioners.py \
    --alphas 10 100 \
    --step-sizes 0.1 0.5 \
    --precond-types bsrem vtv_mm_jensen
```
Runtime: ~8 runs × 45 min = ~6 hours

### Full Test (Recommended)
```bash
# Run overnight on GPU cluster
python scripts/test_preconditioners.py
```

### Analysis
```bash
# After tests complete
python scripts/analyze_preconditioner_tests.py
```

## Key Metrics

1. **Final Objective**: Lower is better (negative log-likelihood + prior)
2. **Convergence Metric**: Relative change in last 10 epochs (lower = more converged)
3. **Runtime**: Total reconstruction time (faster is better)
4. **Precond Setup Time**: Time to compute preconditioner each iteration

## Mathematical Details

See full analysis in the codebase review notes. Key points:

**svd_principal_alpha is exact-ish**: Principal spectral curvature + isotropic α of the Gauss-Newton Hessian.

**mm_jensen is approximate** but reasonable:
- Aggregates h''(σ_k) → loses directional information
- Uses sensitivity approximation → crude estimate of J^T J
- Works well when singular values are similar
- η=0.7 provides damping

**Which to use?**
- **mm_jensen**: Default for most problems (good speed/accuracy tradeoff)
- **svd_principal_alpha**: Use if convergence issues or highly ill-conditioned
- **BSREM**: Baseline; no prior preconditioning

## Notes

- Both VTV preconditioners use Lehmer mean (p=0.1) to combine with BSREM
- Preconditioner frozen after 10 epochs (10 × num_subsets iterations)
- All tests use same initial estimate (OSEM with Gaussian smoothing)
- Results depend on data - Oxford patient data has good conditioning
