# Weighted Vectorial TV: Diagonal Hessian Approximations

This note documents the mathematics behind the diagonal Hessian approximations used for the weighted vectorial total variation (VTV) prior in this repository, and explains the unit tests added to validate key properties of the implementations.

The implementations live in `src/setr/priors/vtv/vtv.py` and use spectral (singular-value–based) smoothing functions defined in `src/setr/priors/vtv/common.py`. The “vector-norm” single-modality baseline is in `src/setr/priors/vtv/vector_norm.py`.

## Setup and Notation

- Let `x` be the block image with `M` modalities. At each voxel `j` we form a local gradient matrix
  `A_j = W_j ⊙ (J x)_j ∈ R^{M×d}`, where `d` is the number of finite-difference directions, `J` the (possibly directional) gradient, and `W_j` the spatial weights per modality (broadcast over directions). We write `b_{j,m}` for the weight of modality `m` at voxel `j` and use `w_j` for the whole vector.
- The VTV functional is a sum of smoothed nuclear norms over voxels:
  `F(x) = Σ_j G(A_j)`, with `G(A) = Σ_k φ(σ_k(A))`, a spectral function of the singular values `σ_k`.
- Typical smoothers φ are Charbonnier, Fair, and Perona–Malik; see `common.py` for `φ`, `φ'`, `φ''`, and the surrogate weight `w(σ) = φ'(σ)/(2σ)`.

Two objects matter for building diagonal preconditioners in image space:

1) The spectral derivatives of `G` (gradient and Hessian) with respect to `A`.
2) The mapping back to image space through the Jacobian `J` and the spatial weights `W`.

## Spectral Derivatives (per voxel)

Let the SVD of `A` be `A = U diag(σ) V^T`. For `G(A) = Σ_k φ(σ_k)`, standard spectral calculus gives:

- Gradient: `∇_A G(A) = U diag(φ'(σ)) V^T`.
- Hessian (bilinear form): for a perturbation `H` of `A`,
  
  `D²G(A)[H, H] = Σ_i φ''(σ_i) ⟨u_i v_i^T, H⟩²`  (principal terms)
  `+ Σ_{i≠j} ½ · (φ'(σ_i) − φ'(σ_j)) / (σ_i − σ_j) · (⟨u_i v_j^T, H⟩² + ⟨u_j v_i^T, H⟩²)`  (cross terms)
  `+ Σ_i (φ'(σ_i)/σ_i) · ||H_{⊥,i}||²`  (isotropic components in the nullspaces),

where `H_{⊥,i}` denotes components orthogonal to the principal rank‑1 directions. The exact form of the last line depends on how one decomposes `H` in the `U/V` frames, but its effect is an “isotropic” contribution weighted by `φ'(σ_i)/σ_i` (see the vector‐norm special case below).

In practice, computing the full exact diagonal in image space would require applying the backprojection (`J^T (W ⊙ ·)`) to all rank‑1 slices `u_i v_j^T` for all `(i, j)`, which is too expensive (O(r²) backprojections per voxel).

## Image-Space Diagonals via Sensitivities

For a diagonal preconditioner in image space, we approximate quadratic forms through `J` by per-direction “sensitivities.” Let `S_{j,m,dir}` denote the magnitude mapping from a unit perturbation at image voxel `(j,m)` to gradient direction `dir` via the local Jacobian entry; in code this is returned by `Jacobian.sensitivity(...)` as a tensor of shape `(..., M, d)`.

Given an image perturbation `δx`, its contribution in the gradient domain is approximately
`δA ≈ W ⊙ (J δx)`, so the preconditioner diagonal at `(j,m)` accumulates as

`P_{j,m} ≈ (b_{j,m})² · Σ_dir [ S_{j,m,dir}² · h_{j,m,dir} ]`,

where `h_{j,m,dir}` is the per-direction curvature factor delivered by the chosen approximation in the U‑space (see below). Some variants fold directional factors into a scalar weight times `Σ_dir S²` to avoid dependence on the current direction distribution.

Boundary participation counts (1 on boundaries, 2 interior) are optionally included in the sum over directions to reflect how many finite differences a voxel participates in. These are used consistently in the “mm_jensen” and “frobenius_surrogate_pd” variants.

## Approximations Implemented

All methods compute a diagonal `P` with `(j,m)` entries. We denote `r = min(M, d)`.

### 1) svd_principal_alpha (principal + isotropic)

File: `src/setr/priors/vtv/vtv.py`, `_preconditioner_weights_core_slow`.

Steps per voxel:
- Compute `A = W ⊙ (Jx)` and its SVD to get `(U, σ, V)`.
- Principal term: for each singular value `σ_k`, form the rank‑1 slice `C_k = u_k v_k^T` and backproject `z_k = J^T(W ⊙ C_k)`. Accumulate `P += φ''(σ_k) · z_k²` (elementwise square in image space).
- Isotropic term: add
  `α · (W² · Σ_dir S²)`, where `α = Σ_k φ'(σ_k)/σ_k`.

Rationale: the principal term captures curvature along the rank‑1 directions explicitly; the isotropic term collects the remaining “radial” curvature present in spectral Hessians (this reduces exactly to the known vector‑norm result when `M=1`, see below). This is still a diagonal approximation: cross terms with `i≠j` are omitted to keep cost reasonable.

Special case `M=1` (vector‐norm): if `A` is a single row vector `u ∈ R^d`, `G(A)=φ(||u||₂)` and the exact Hessian is

`∇²_u φ(||u||) = α I + β · (u u^T)/(||u||²)`,

with `α = φ'(r)/r` and `β = φ''(r) − α`, `r = ||u||`. The image‐space diagonal produced by the above mapping coincides with this structure once folded through sensitivities and weights.

Limitations: omits cross‐pair terms `(φ'(σ_i)−φ'(σ_j))/(σ_i−σ_j)`, which are O(r²) to realize exactly.

### 2) mm_jensen (Jensen / MM IRLS surrogate; SVD‑free)

File: `vtv.py`, `_preconditioner_weights_core_fast`.

- Use Jensen on the spectral weights to avoid SVD: define `σ_avg = sqrt(||A||_F² / r)`.
- Surrogate per voxel: `ω = φ'(σ_avg) / (2 σ_avg)`.
- Assemble: `P_{j,m} = η · ω_j · S_jm · b_{j,m}²`, where `S_jm = Σ_dir (S² · participation)` and `η ∈ (0,1]` is a damping factor.

Properties: conservative MM majorizer, positive definite by construction, cheap to compute.

### 3) frobenius_surrogate_pd (Frobenius surrogate; SVD‑free)

File: `vtv.py`, `_preconditioner_weights_core_fastest_positive` (alias for `frobenius_surrogate_pd`).

- Use Frobenius norm directly: `ω = M · φ'(||A||_F) / ||A||_F`.
- Assemble: `P_{j,m} = η · ω_j · S_jm · b_{j,m}²` as above.

Properties: ultra-cheap, positive, more conservative than “fast”.

### 4) vector_tv_per_modality (per-modality vector TV; SVD‑free)

File: `vtv.py`, `_preconditioner_weights_core_fastest_exact` (alias for `vector_tv_per_modality`).

- For each modality, compute vector-norm radial coefficients `α = φ'(r)/r` and `β = φ''(r) − α` with `r = ||A_{j,m,·}||₂`.
- Per-direction curvature in U‑space: `h_dir = α + β · (U_d² / (r² + tiny))` with `U = A_{j,m,·}`.
- Assemble: `P_{j,m} = Σ_d [ b_{j,m}² · S_{j,m,d}² · h_dir ]`.

This reduces exactly to the single-modality vector-norm Hessian diagonal; it’s still cheap and SVD‑free.

Legacy aliases
- slow → svd_principal_alpha
- fast → mm_jensen
- fastest_positive → frobenius_surrogate_pd
- fastest_exact → vector_tv_per_modality

## Smoothers and Radial Coefficients

For vector norms (`M=1`) we explicitly work with radial coefficients (see `vector_norm.py`):

- `r = ||u||₂`, `α = φ'(r)/r`, `β = φ''(r) − α`.
- Per-direction curvature in U‑space: `h_dir = α + β · (u_dir² / (r² + stabiliser))`.

For spectral norms, the “isotropic” addition `Σ_k φ'(σ_k)/σ_k` mirrors the vector norm’s `α` contribution aggregated over singular directions. In code we exploit the helper `hessian_surrogate` that returns `w(σ) = φ'(σ)/(2σ)`, so `Σ_k φ'(σ_k)/σ_k = 2 Σ_k w(σ_k)`.

## Tests Added

File: `tests/test_vtv_hessian_diag.py`.

Both tests use a minimal harness that bypasses the full constructor of `WeightedVectorialTotalVariation` and injects a `MockJacobian` exposing deterministic `direct`, `adjoint`, and `sensitivity` tensors. This isolates the diagonal formulas.

1) `test_m1_fastest_exact_matches_vector_norm`
- Setup: constant unit `J` and unit sensitivities `S`, with distinct weights per modality (e.g., `b=[2.0, 0.5]`).
- Expected formula: `P_{j,m} = b_{j,m}² · φ''(ρ_{j,m}) · Σ_dir S²`, where `ρ_{j,m} = sqrt(Σ_dir (b_{j,m} · J)²)`.
- Checks: verifies the implemented `_preconditioner_weights_core_fastest_exact` (vector_tv_per_modality) matches the vector-norm α,β diagonal.

2) `test_slow_includes_isotropic_alpha_term_when_J_zero`
- Setup: zero gradients `J=0` so all principal backprojections vanish (`z_k=0`), and unit sensitivities.
- For Fair smoothing with `eps`, `φ'(0)/(0)` is interpreted via the helper as `1/eps`, and with `r=min(M,d)` singular values we get `α_total = r/eps`.
- Expected diagonal: `H = α_total · (b² · Σ_dir S²)`.
- Checks: compares `_preconditioner_weights_core_slow` output with this expected expression. This validates that the isotropic `α` term is included.

Both tests skip gracefully if PyTorch (or other dependencies) aren’t importable in the test process.

## Practical Notes

- The “slow” variant is now a mathematically consistent diagonal approximation (principal `φ''` slices + isotropic `α`), not a full exact diagonal of the spectral Hessian — implementing the cross terms would require O(r²) backprojections per voxel.
- The “mm_jensen” and “frobenius_surrogate_pd” variants are positive MM surrogates built to be SVD-free and conservative, using per-voxel Frobenius aggregates and boundary participation counts.
- The “vector_tv_per_modality” variant is per-modality exact in the vector‑norm sense and maps back with the correct `b²` scaling.
- For `M=1`, the “svd_principal_alpha” and “vector_tv_per_modality” variants reduce to the familiar vector-norm structure with radial `(α, β)` coefficients in `vector_norm.py`.

## References

- Lewis, A. S., Sendov, H. S. (2001). Twice differentiable spectral functions. SIAM Journal on Matrix Analysis and Applications, 23(2), 368–386.
- Sun, W., Sun, Y. (2002). Matrix perturbation analysis of spectral functions. (for general spectral Hessians)
- Standard TV smoothing identities: Charbonnier, Fair, and Perona–Malik smoothers as implemented in `common.py`.
