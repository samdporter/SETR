# Objective-Consistency Gating

_Purpose: the whole chapter rests on every run minimising the **same** penalised
objective, so that a "converged" reference is a shared ground truth and only the
optimisation path differs between methods. This note gates every sweep against
its preconditioner baseline on the objective-defining parameter set and records
which runs are safe to compare._

Gate script: `scripts/check_objective_consistency.py`
Baselines (ground truth): `baselines_1bpos_gersh_eta0001_gamma002`,
`baselines_2bpos_gersh_eta0001_gamma002` (alpha 1.0).

Objective-defining set gated: effective (scaled) `alpha`/`beta`, prior family and
all TNV geometry (`gamma_tnv`, `delta`, `directional_tnv`, `tnv_stencil`,
`tnv_bnd_cond`, `smoothing`, `use_kappa`, …), the acquisition/forward model
(`pet_gauss_fwhm`, `spect_gauss_fwhm`, `spect_res`, `use_scatter`, `use_tof`,
`use_zoom_registration`), the feasible set (`support_mask_*`), and the data paths.
Optimisation knobs (`num_epochs`, `initial_step_size`, `relaxation_eta`,
`precond_type`, `precond_combine`, `num_subsets`, seed, snapshot cadence, initial
image) are reported but **not** gated — for a convex objective they change only
the path, not the minimiser.

## Result

| Sweep | Baseline | Runs | Obj-defining verdict |
|---|---|---|---|
| `precond_1bpos_step1_2` (main, legend5) | 1bpos gersh | 60 | **PASS** (0 mismatches; 3 dirs missing `args.csv`) |
| `precond_2bpos_step1_2` (main, legend5) | 2bpos gersh | 60 | **PASS** (0 mismatches) |
| `precond_1bpos` (LS Lehmer + majoriser) | 1bpos gersh | 30 | **PASS** (0 mismatches) |
| `precond_2bpos` (LS Lehmer + majoriser) | 2bpos gersh | 30 | **PASS** (0 mismatches) |
| `subset_selection_main` (1bpos) | 1bpos gersh | 8 | **FAIL** — `delta` |
| `subset_selection_main_2bpos` | 2bpos gersh | 8 | **FAIL** — `delta` + `support_mask_from_sensitivity` |

The preconditioner and Lehmer sweeps share the baseline objective exactly. The
LS-Lehmer `combine=lehmer` runs are objective-clean — their divergence (see the
Lehmer note) is an optimisation failure, not an objective mismatch.

## The subset-selection objective mismatch (rogue runs)

The subset-selection runs do **not** share the baseline objective:

| | 1bpos | 2bpos |
|---|---|---|
| baseline `delta` | 0.0500061 | 0.2005958 |
| subset `delta` | 0.0018977 | 0.0030113 |
| ratio | **26.3×** | **66.6×** |
| baseline `support_mask_from_sensitivity` | True | **False** |
| subset `support_mask_from_sensitivity` | True | **True** |

`delta` is the Charbonnier smoothing of the TNV singular values,
`phi(sigma) = sqrt(sigma^2 + delta^2)`; it sets the edge scale below which the
nuclear norm is rounded, so a 26–67× change is a materially different (much less
smoothed) prior and a different minimiser. On 2bpos the feasible set differs too.

**It is not an initial-image effect.** The saved `initial_image_0/1` are byte-for-byte
the same as the preconditioner runs, and re-applying the standard rule
(`set_auto_delta_from_scaled_images`: `min(p99(alpha·x_PET), p99(beta·x_SPECT)) / 20`)
to those images reproduces the baseline `delta` (0.0500061 on 1bpos) — **not** the
`delta` the subset run actually recorded and used (0.0018977). Both runners import
the same delta function, so the subset value comes from a different image state at
delta-computation time (a SPECT scaling/warp ordering in the subset path, before
the initial images are written out). The subset study is nonetheless internally
self-consistent: its own `subset_selection_convergence` reference also uses
`delta = 0.0018977`, so subset runs converge to the subset reference — just not to
the preconditioner ground truth.

### Consequence
Subset-selection results **cannot** be pooled against the preconditioner baseline.
Either (a) re-run the subset sweeps with `delta` pinned to the baseline value (and,
on 2bpos, `support_mask_from_sensitivity: false`) to unify the objective, or
(b) present subset selection as a self-contained sub-study analysed against its own
converged reference, with this discrepancy stated explicitly.

### Reliable fix for a re-run
Pin the objective explicitly rather than trusting auto-estimation (which drifts):
```yaml
# subset base config
delta: 0.05000608682632448          # 1bpos  (0.20059575915336608 for 2bpos)
support_mask_from_sensitivity: false # 2bpos only, to match the 2bpos baseline
```
`set_auto_delta_from_scaled_images` early-returns when `delta` is already set, so an
explicit value is honoured and removes the drift entirely.

_Per-run reports: `output/<sweep>_analysis/objective_consistency_report.csv`._
