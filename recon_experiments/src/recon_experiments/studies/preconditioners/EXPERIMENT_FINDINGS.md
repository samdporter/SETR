# Preconditioner Chapter — Findings

Working notes, one section per experiment. Ground truth = the long preconditioner
baselines (`baselines_{1,2}bpos_gersh_eta0001_gamma002`, ~900–1000 epochs).
All comparisons are objective-gated (see `OBJECTIVE_CONSISTENCY_GATING.md`): every
run analysed here minimises the *same* penalised objective as its baseline, so the
only thing differing is the optimisation path. Headline metric = **NRMSE-to-baseline
inside PET-space VOIs** (per the convex objective, all methods share one minimiser;
the question is how fast each gets there). x-axis = epochs.

Status key: ✅ done · ⏳ awaiting cluster rerun.

---

## Exp 5 — Are the baselines converged? ✅

`output/exp5_baseline_convergence/` (`plot_baseline_convergence.py`)

**Yes, and it is genuine convergence, not step-starvation.** The distinction the
chapter must defend is "converged" vs "the relaxed step just decayed to zero".

| baseline | epochs | J0 → J* | total decrease | final gap / total | final PET velocity ‖Δx‖/‖x‖ per epoch | **final step** |
|---|---|---|---|---|---|---|
| 1bpos (phantom) | 1000 | 2.6445e6 → 2.6303e6 | 14 269 (0.54% of J0) | **8.8e-9** | **7.3e-8** | **0.22** |
| 2bpos (patient) | 900 | 4.2869e6 → 3.5728e6 | 714 061 (17% of J0) | **2.3e-8** | **4.8e-5** | **0.27** |

- The final objective gap is ~10⁻⁸ of the total decrease → the baselines sit on the
  minimiser to ~8 significant figures. They are legitimate ground truth.
- The **image itself has stopped moving** (velocity 10⁻⁷–10⁻⁵ per epoch), while the
  **relaxed step is still O(0.2–0.3)** — not tiny. So the plateau is real convergence:
  if we un-decayed the step the iterate would not move. (Panel 3 of the figure makes
  this explicit: velocity ⟶ floor while the dashed step is still ~0.25.)
- 99.9% of the total objective decrease is reached by **epoch 45 (1bpos) / 22 (2bpos)**,
  with the step still ~0.86–0.94 — i.e. the objective is essentially solved well
  inside the 25-epoch sweep window. (Caveat for the reader: the *image* NRMSE keeps
  refining after the objective flattens — see exp 3 — because the objective is very
  flat near the minimiser, especially on 1bpos where it only moves 0.54%.)

Figure: `baseline_convergence.png` (3 panels: objective gap + step; per-epoch
relative objective change; solution velocity vs step). Thesis-usable as-is.

---

## Exp 3 — Best preconditioner (incl. BSREM; excl. block-majoriser & Lehmer) ⏳/✅

Methods = `precond_types_legend5.csv`: bsrem, mm_diag_tight, mm_diag_gershgorin_maj,
mm_diag_block_tight, ls_block_diag (mm_diag_block_maj and ls_block_gershgorin
excluded by default). Step 1.0; 5 reps; median over reps, p90 in parens.

### 1bpos (phantom) ✅ — `output/exp3_best_precond_1bpos_step1/`

PET whole-image NRMSE-to-baseline:

| method | ep1 | ep5 | ep10 | ep25 | p90 @ ep25 |
|---|---|---|---|---|---|
| **mm_diag_block_tight** | 0.53 | 0.14 | 0.067 | **0.019** | 0.020 |
| mm_diag_tight | 0.53 | 0.18 | 0.076 | 0.021 | 0.025 |
| mm_diag_gershgorin_maj | 0.53 | 0.16 | 0.34 | 0.020 | 2.68 (big mid-run excursions) |
| ls_block_diag | 0.53 | 0.20 | 0.095 | 0.030 | 0.43 (one rep unstable; 4/5 ran) |
| bsrem | — | — | — | **diverges** (J → 10¹²) | — |

- **BSREM (data-only) diverges** at step 1: the EM preconditioner `(x+ε)·s⁻¹` overshoots
  where sensitivity is low; objective blows up to ~10¹². Prior curvature *in the
  preconditioner* is required. (In the high-count `hot_sphere` VOI bsrem stays bounded
  at ~0.05 — the divergence is a background/low-sensitivity effect.)
- **mm_diag_block_tight wins**: fastest to baseline and tightest rep spread.
  `mm_diag_tight` is a close, robust second. The Gershgorin majoriser reaches a good
  endpoint but has large mid-run excursions (conservative but noisy). `ls_block_diag`
  is competitive but had one unstable rep.

### 2bpos (patient) ✅ — `output/exp3_best_precond_2bpos_step1/`

**Opposite of 1bpos: preconditioner choice barely matters and BSREM is fine.** The
four prior-curvature preconditioners are indistinguishable (curves overlap to 3 sig
figs) and data-only BSREM ties them. Final objective gap to J* per method (epoch ~22):

| method | final gap / J* |
|---|---|
| ls_block_diag | 2.40e-4 |
| mm_diag_block_tight | 2.50e-4 |
| mm_diag_tight | 2.72e-4 |
| mm_diag_gershgorin_maj | 2.95e-4 |
| bsrem | 2.97e-4 |

- All within ~25% of each other — effectively a tie. In the **lesion** VOIs BSREM is
  marginally *fastest*; in whole-image/background it is marginally slowest.
- Interpretation (flag as such): the patient reconstruction is **data-curvature
  dominated** — the (scaled) prior curvature is small relative to the data Hessian, so
  `(P_D⁻¹+P_prior⁻¹)⁻¹ ≈ P_D` for every prior variant, collapsing them onto BSREM. No
  divergence because the EM preconditioner is well-conditioned on this data.
- Contrast with 1bpos, where the prior curvature is significant: there the choice
  matters, block-tight wins, and BSREM diverges.

**Chapter takeaway (exp 3):** the *value* of prior-curvature preconditioning is
problem-dependent. When the prior is significant (phantom) it is essential — block-tight
is best and data-only BSREM diverges; when the problem is data-dominated (this patient)
all preconditioners, BSREM included, converge together. Recommended default:
`mm_diag_block_tight` (never worse; decisive when it matters).

Figures: `output/figures/exp3_best_precond_1bpos.png`, `..._2bpos.png` (data note: one
truncated snapshot in the synced copy — `mm_diag_block_tight` rep5 `image_1_570.v` —
was restored from the comic source; the analysis also now skips unreadable snapshots).

---

## Exp 1 — Lehmer mean vs harmonic mean ⏳

Harmonic combine (= inverse-sum majoriser, `combine=majoriser`) converges cleanly on
both bpos. Lehmer p=0.1 as first run **diverged** (J 2.6e6 → 8.9e12) because the Lehmer
general branch ≈ **2× the parallel-sum majoriser** as p→0 (the missing 1/n between
"parallel sum" and "harmonic mean"). Confirmed numerically: `0.5·L₀.₁ / majoriser` is
1.00 (equal curvatures) → 1.10 (10:1) → 1.29 (100:1). The committed `lehmer_scale=0.5`
therefore lands Lehmer p=0.1 essentially on the harmonic majoriser. **Awaiting the
scaled reruns** (comic was 2 commits behind; the deployed runs were unscaled).

---

## Exp 2 — Optimised subset selection ⏳

The existing subset runs are on a **different objective** (delta 26–67× smaller, +a
support-mask mismatch on 2bpos) — they do NOT share the baseline solution
(see `OBJECTIVE_CONSISTENCY_GATING.md`). Delta is now pinned in the subset base configs
so a rerun matches the ground-truth objective. **Awaiting the delta-pinned reruns**,
which will be analysed against the existing `baselines_{1,2}bpos_gersh` (no new long
baseline needed). Will re-gate the reruns to confirm 0 objective mismatches first.

---

## Exp 4 — 100 vs 25 epochs ⏳

Runs not yet performed (trimmed config prepped: 5 methods × step 1 × 3 reps, 100 epochs,
2bpos). Prior expectation from exp 5: the *objective* is ~99.9% solved by ~epoch 22, but
the *image* NRMSE keeps refining past 25 epochs (exp 3), so 100 epochs may still move the
image — the point of the experiment. **Awaiting the 100-epoch reruns.**
