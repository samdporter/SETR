# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Vectorial Total Variation regularization
- GPU-accelerated gradients and Jacobians
- Python wrapper for STIR Kernel EM
- Numba-optimized kernel operators
- Extended CIL functionality (preconditioners, callbacks, step size rules)
- SIRF data loading utilities
- Initial project structure and build system
- **Four TNV preconditioner methods** (2025-01-04):
  - Method A (`svd_principal_alpha`, legacy `slow`): Lewis-Sendov full SVD + isotropic α
  - Method B (`mm_jensen`, legacy `fast`): MM-Jensen inequality, SVD-free, 2.8× faster
  - Method C (`vector_tv_per_modality`, legacy `fastest_exact`): Decoupled vectorial TV per modality
  - Method D (`frobenius_surrogate_pd`, legacy `fastest_positive`): Scalar Frobenius approximation
- Comprehensive preconditioner test infrastructure:
  - `test_vtv_preconditioners_synthetic.py`: Synthetic 3D geometric validation
  - `test_preconditioners.py`: Full reconstruction testing (5 variants)
  - `run_preconditioner_tests.sh`: Quick and full test modes
  - `PRECONDITIONER_METHODS.md`: Complete method documentation
- **Cluster-friendly preconditioner testing framework** (2025-10-05):
  - New directory: `functionality/preconditioners/` for cluster sweeps
  - `test_preconditioner_single.py`: Standalone single-run test script
  - `precond_sweep.qsub.sh`: SGE job script for 3-way parameter sweep
  - `launch_precond_sweep.sh`: Launcher with local/test/full modes
  - Parameter files for precond types, alphas, and step sizes
  - Comprehensive README with usage examples
  - Support for 5 precond types × 7 alphas × 4 step sizes = 140 job combinations
- Synthetic diagnostics and visualization updates (2025-10-06):
  - Added log-scale preconditioner plot and slow-term diagnostics in `test_vtv_preconditioners_synthetic.py`
- Documented spectral-to-diagonal derivations in `docs/reference/vtv_hessian_diagonals.md`

### Changed
- Updated `test_preconditioners.py` to test all 5 preconditioner variants
- Updated `run_preconditioner_tests.sh` with new test configurations (18/80 runs)
- **Modified `test_preconditioners.py`** (2025-10-05):
  - Added `attach_prior_hessian()` to wrap inv_hessian_diag methods
  - Priors for preconditioners now correctly apply `bo` operator transformations
  - Fixed preconditioner to use wrapped hessian methods via `p.inv_hessian_diag`
- Updated `run_preconditioner_tests.sh` to reference new cluster framework
- Consolidated Markdown docs into `docs/` with guides and reference index
- Renamed Hessian diagonal variants to canonical names (`svd_principal_alpha`, `mm_jensen`,
  `frobenius_surrogate_pd`, `vector_tv_per_modality`) and updated scripts/configs; legacy aliases retained

### Fixed
- **Critical bug in `fast` preconditioner** (2025-01-04):
  - Was incorrectly using SVD via `hessian_surrogate`
  - Now correctly implements Jensen's inequality without SVD
  - Achieved true 2.8× speedup over baseline
- **Zero Hessian values in `slow` method** (2025-01-04):
  - Added epsilon floor (1e-8) to prevent division by zero
  - Now guaranteed positive-definite for all methods
- **State interference bug in VTV test suite** (2025-01-04):
  - `BlockDataContainerToArray.adjoint()` was modifying shared geometry in-place
  - Tests now clone geometry/weights for each VTV instance
  - Objectives and gradients now correctly identical across all methods
- **Critical operator composition bug in preconditioner tests** (2025-10-05):
  - VTV preconditioner was receiving untransformed images (mixed geometry)
  - VTV expected transformed images (common PET space via `bo` operator)
  - Issue: `ImageFunctionPreconditioner` called `vtv.inv_hessian_diag(x)` directly
  - Fix: `attach_prior_hessian()` wraps method to apply `bo.direct(x)` before computation
  - Error was: "RuntimeError: stack expects each tensor to be equal size, but got [83, 255, 255] at entry 0 and [128, 128, 128] at entry 1"
  - Now correctly transforms: original space → `bo.direct()` → VTV computation → `bo.adjoint()` → original space

### Removed

## [0.1.0] - 2025-07-22

### Added
- Initial release
- Basic package structure with pyproject.toml
- Core module with dependency checking
- Utils module with image processing utilities
- Test suite with pytest
- MIT license
- README with installation instructions

[unreleased]: https://github.com/samdporter/setr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/samdporter/setr/releases/tag/v0.1.0

### Fixed
- README.md title
