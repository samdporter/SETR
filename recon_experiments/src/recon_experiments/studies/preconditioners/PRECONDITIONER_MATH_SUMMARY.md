# Preconditioner Maths Summary (TNV / VTV)

This note summarizes the **Hessian / curvature approximations** and **combination choices** used in the DTNV/VTV preconditioner experiments.

Notation:
- Let the weighted Jacobian at a voxel be `A = w ⊙ Jx` with shape `(M, d)` where `M=2` modalities.
- The TNV penalty at a voxel is based on the (smoothed) nuclear norm of `A`.
- For Charbonnier smoothing, the singular-value potential is `g(σ) = √(σ² + ε²)`.

## 1) MM / IRLS weights (nuclear norm surrogate)

For the smoothed nuclear norm

```
φ(A) = tr((Aᵀ A + ε² I)^{1/2})
```

the standard MM/IRLS surrogate at `A0` uses

```
W = (A0 A0ᵀ + ε² I)^{-1/2}
```

and the quadratic majoriser

```
φ(A) ≤ c(A0) + 0.5 * tr(W A Aᵀ)
```

with `c(A0)` chosen to match equality at `A=A0`.

### mm_block_diag
- Uses the **full 2×2** MM weight `W` (per voxel) and pulls it back into image space with the stencil sensitivities.
- This is the most faithful MM/IRLS **block** surrogate.

### mm_diag
- Uses only the **diagonal** of `W`.
- Cheap but **not** a Loewner majoriser of `W`; it can underestimate coupling.

### mm_diag_gershgorin
- Uses **Gershgorin row‑sum inflation** of `W` to make a **diagonal Loewner majoriser**:

```
D = diag(W₁₁ + |W₁₂|, W₂₂ + |W₁₂|)
```

This is conservative and PSD‑dominates `W`.

## 2) LS / dilation block Hessian (ls_block_diag)

Uses the **symmetric dilation** (Low‑rank / LS) construction to compute a 2×2 block approximation to the Hessian of `φ(A)`.

- This is **not guaranteed** to dominate the full Hessian once truncated to block‑diagonal form.
- It is still SPD and typically more informative than a purely diagonal surrogate.

## 3) Frobenius diagonal surrogate (frob_diag)

A very cheap isotropic proxy based on the Frobenius norm:

```
ω = 1 / √(0.5 * ||A||_F² + ε²)
```

The image‑space diagonal is then scaled by sensitivities and weights.

- Fast, always positive, **not** a nuclear‑norm majoriser.

## 4) Directional weighting and boundary handling

- Directional projectors from anatomical guidance are applied per direction.
- **Neumann** boundary accumulation uses **clamped shifts** (symmetric extension), consistent with the Jacobian implementation.
- **Periodic** uses roll; **Other** uses zero‑padding.

## 5) Combining preconditioners

We blend the **data‑fidelity (EM/BSREM)** scalar preconditioner with the **prior (TNV)** preconditioner.

### Diagonal blend (scalar + scalar)
Uses the Lehmer mean

```
L_p(x, y) = (x^p + y^p) / (x^{p-1} + y^{p-1})
```

with **p=0** giving the **harmonic mean**:

```
H(x, y) = 1 / (1/x + 1/y)
```

### Block blend (block + scalar)
We form a block blend of a **2×2 SPD block** `B` with a scalar preconditioner.

#### Scalar reduction options
The scalar preconditioner comes per modality `(λ_pet, λ_spect)` and must be mapped into block form:

- **mean**: `λ = 0.5(λ_pet + λ_spect)` → use `λI`
- **geometric**: `λ = √(λ_pet λ_spect)` → use `λI`
- **diag** (current default): `D = diag(λ_pet, λ_spect)` (no collapse)

#### Harmonic block blend (p=0)
With `diag` reduction, the only valid blend is the **harmonic mean**:

```
H(B, D) = (B^{-1} + D^{-1})^{-1}
```

This preserves modality scales and yields a **block SPD** preconditioner.

### Majoriser blend (Hessian-sum inverse)
An explicit baseline option is available via `precond_combine=majoriser`:

```
H_total = H_data + H_prior
P = H_total^{-1}
```

where:
- `H_data` uses EM-type curvature `sensitivity / (x + eps)` (equivalently inverse of BSREM preconditioner),
- `H_prior` uses the chosen TNV Hessian approximation (`mm_block_diag` for block mode, or the selected diagonal method).

## 6) Defaults in current experiments

- `precond_combine = majoriser` (default; `harmonic` is treated as alias)
- `block_scalar_reduction = diag`
- `precond_type = mm_block_diag`

These defaults are intentionally conservative and preserve PET/SPECT scale differences while keeping the block structure.

---

### Practical notes
- Smaller preconditioner values lead to **larger effective step sizes** when the inverse is applied; floors/epsilons must be chosen carefully.
- The **diag** reduction avoids collapsing PET/SPECT scales, which is important when the modalities differ significantly.
