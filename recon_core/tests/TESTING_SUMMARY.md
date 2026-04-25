# VTV Preconditioners — Testing Summary

This suite separates *function-value majorisation* from *matrix (Loewner) dominance* and documents which properties are expected per method.

## Expected majorisation behavior

- **mm_diag_block_maj**
  - Expected: **function-value MM majorisation** of the smoothed nuclear norm potential in Jacobian space.
  - Tested by: `test_mm_surrogate_majorizes_local_nuclear` (tangent majoriser of the concave trace–sqrt).
  - Not required: Loewner dominance vs the full Hessian in image space.

- **mm_diag_gershgorin_maj**
  - Expected: **Loewner (PSD) dominance** of the *Gershgorin diagonal* over the 2×2 MM weight matrix `W`.
  - Tested by: `test_mm_diag_gershgorin_loewner_majorizer`.

- **mm_diag_tight**
  - Expected: **no Loewner dominance guarantee** (diagonal-only heuristic).
  - Documented by: `test_mm_diag_is_not_loewner_majorizer` (searches a counterexample).

- Deprecated methods such as `ls_block_diag` and `frob_diag` have been removed from the active preconditioner surface and are no longer part of the supported test matrix.

## Other sanity/robustness coverage

- **Edge→voxel aggregation**: `test_edge_accumulation_matches_reference` compares `_accumulate_over_neighborhood_block` to a reference loop for periodic and non-periodic modes.
- **Boundary consistency**: `test_periodic_participation_is_uniform` confirms uniform participation under periodic BC.
- **Scaling sanity**: `test_scaling_weights_monotone` confirms the MM diagonal increases when spatial weights are scaled up.
- **Directional projector identity**: `test_directional_projector_identity_case` validates reduction when directional projector is identity.
- **Inversion correctness**: `test_diag_quadratic_minimiser` and `test_block_inverse_and_minimiser` validate symmetry, identity, and quadratic minimiser properties.
- **Finite/NaN safety**: `test_extreme_values_are_finite`.
- **Dtype consistency**: `test_mm_weight_dtype_spd`.

These tests are designed to be **CPU-friendly** with small tensor sizes and fixed seeds.
