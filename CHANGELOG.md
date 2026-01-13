# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-13

### Changed - Major Repository Restructure

**Repository Split into Core Library and Experiments Package**

This release represents a major restructuring of the repository to separate stable library code from research experiments.

#### Package Structure
- **Split monolithic `setr` package into two independent packages**:
  - `recon_core` (v0.2.0) - Stable reconstruction library with minimal dependencies
  - `recon_experiments` (v0.1.0) - Experiment runners, sweeps, and analysis tools
- **Adopted src-layout** for both packages with modern `pyproject.toml` configuration
- **Established clean API boundaries** with comprehensive public API in `recon_core.__init__`
- **Enabled independent versioning** and release cycles for each package

#### Import Changes
- **Core library imports**: `setr.*` → `recon_core.*` (e.g., `from setr.priors import WeightedVectorialTotalVariation` → `from recon_core.priors import WeightedVectorialTotalVariation`)
- **Experiment helpers**: `setr.scripts.*` → `recon_experiments.runners.*` (e.g., `from setr.scripts.common import init_run_env` → `from recon_experiments.runners.common import init_run_env`)
- **Updated 500+ import statements** across 100+ Python files
- **Preserved numerical behaviour** - all algorithms produce identical results

#### Directory Migration
- **Core library** (`recon_core/src/recon_core/`):
  - `src/setr/cil_extensions/` → Extensions to CIL framework
  - `src/setr/core/` → Core gradient computations
  - `src/setr/kernel/` → Low-level EM kernels
  - `src/setr/priors/` → Regularisation functions (VTV, RDP, MI)
  - `src/setr/utils/` → Utility functions (SIRF, CIL, IO, NiftyReg)
  - `tests/` → Test suite (17 test files, 169 tests)
  - `data/` → Test data files

- **Experiments package** (`recon_experiments/src/recon_experiments/`):
  - `src/setr/scripts/` + `scripts/` → `runners/scripts/` (40 reconstruction scripts)
  - `experiments/` → `experiments/` (22 files, experiment orchestration)
  - `functionality/` → `studies/` (81 files, organised research studies)
  - `sweeps/` → `sweeps/` (7 files, parameter sweep framework)
  - `configs/` → Top-level config files (25 YAML files)
  - `*.ipynb` → `notebooks/` (analysis notebooks)

#### Dependency Management
- **Core library dependencies** (minimal):
  - torch >= 2.0.0
  - numba >= 0.58.0
  - numpy >= 1.21.0, < 2.0.0
  - matplotlib >= 3.5.0
  - pandas >= 1.3.0
  - pyyaml >= 5.4.0
  - External: SIRF, CIL, STIR (documented separately)

- **Experiments package dependencies**:
  - Depends on `recon-core`
  - Added: seaborn, tqdm for analysis and progress tracking
  - Optional: jupyter for notebooks

- **Removed from core**: wandb, seaborn, jupyter (moved to experiments)

### Added

#### Documentation
- **Comprehensive documentation reorganisation**:
  - New `docs/index.md` - Central documentation hub with organised sections
  - New `docs/cluster-usage.md` - 400+ line comprehensive HPC/cluster guide covering:
    - Job submission basics (SGE templates)
    - Bootstrap reconstruction workflows (DTNV, HKEM)
    - Parameter sweep framework
    - Preconditioner comparison studies
    - Monitoring and debugging procedures
    - Resource requirements and troubleshooting
  - `docs/migration.md` - Complete migration guide with import mapping tables
  - `docs/architecture/repository-split.md` - Detailed refactoring documentation
  - `docs/architecture/vtv_hessian_diagonals.md` - Technical reference (relocated)

#### Package Documentation
- `recon_core/README.md` - Complete library documentation with API reference
- `recon_core/CHANGELOG.md` - Core library version history
- `recon_experiments/README.md` - Experiments framework guide
- Root `README.md` - Updated monorepo overview with quick start
- Multiple README files in experiment subdirectories for specific workflows

#### Tooling
- `validate_split.sh` - Automated validation script to verify repository structure
- Shell scripts for cluster job submission (13 .sh and .qsub.sh files)

### Changed

#### Repository Structure
- **Removed duplicate directories** after verification:
  - Old `src/setr/` package (merged into recon_core and recon_experiments)
  - Root `tests/` and `data/` (moved to recon_core/)
  - Root `scripts/`, `experiments/`, `functionality/`, `sweeps/`, `configs/` (moved to recon_experiments/)
  - Build artifacts in `dist/`

- **Cleaned up documentation**:
  - `README.md` - Now main entry point (was ROOT_README.md)
  - Removed duplicate/outdated READMEs
  - Fixed broken documentation references
  - Consolidated cluster guides into comprehensive `docs/cluster-usage.md`
  - Created `docs/architecture/` for design documentation

#### Configuration
- **Updated all config files** to reference new package structure
- **Updated .gitignore** for new package structure (dist/, build/ for both packages)

### Removed

- Old `setr` package from `src/`
- Duplicate `ROOT_README.md` (merged into README.md)
- Temporary `README.md.old` backup file
- Build artifacts directory `dist/`
- Old `docs/reference/` directory (consolidated into `docs/architecture/`)

### Migration Notes

**For existing users**, see `docs/migration.md` for:
- Complete import mapping table (old → new)
- Step-by-step migration instructions
- Example code updates
- Testing and validation procedures

**Key changes**:
1. Install both packages: `pip install -e recon_core/ && pip install -e recon_experiments/`
2. Update imports from `setr.*` to `recon_core.*` for library code
3. Update imports from `setr.scripts.*` to `recon_experiments.runners.*` for experiment helpers
4. Update config file paths if referencing old locations
5. Run tests to verify numerical consistency

**Numerical behaviour preserved**: All reconstruction algorithms produce identical results to the previous version.

## [0.1.0] - Legacy

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
- Per-modality dynamic-range scaling utilities and regression tests (2025-11-20):
  - `setr.utils.dynamic_range` exposes `dynamic_range_scale_sirf` plus helpers that capture and persist the pre-scaled prior weights before adjusting `alpha`, `beta`, `gamma_pet`, and `gamma_spect`.
  - `tests/test_dynamic_range_scaling.py` validates masking, absolute-intensity handling, fallback behaviour, and repeated scaling so every DTNV script can rely on consistent modalities.
- Resampling diagnostics and tooling for PET/SPECT guidance (2025-11-20):
  - Added `scripts/resample_spect_to_pet.py`/`scripts/resample_spect_to_pet_simple.py` to warp SPECT reconstructions directly into PET space using the stored no-zoom displacement.
  - Added `scripts/nifty_resample_diagnostics.py` and `scripts/test_spatial_ops.py` to inspect forward/backward `NiftyResample` consistency, SPECT→PET resampling, couch shifts, combine/uncombine operators, and block wiring with optional viewer dumps.

### Changed
- Updated `test_preconditioners.py` to test all 5 preconditioner variants
- Updated `run_preconditioner_tests.sh` with new test configurations (18/80 runs)
- Dynamic-range scaling now divides each modality’s weights (`alpha`, `beta`, `gamma_pet`, `gamma_spect`) by its own robust intensity range so `alpha=beta=1` yields balanced PET/SPECT priors
- **Modified `test_preconditioners.py`** (2025-10-05):
  - Added `attach_prior_hessian()` to wrap inv_hessian_diag methods
  - Priors for preconditioners now correctly apply `bo` operator transformations
  - Fixed preconditioner to use wrapped hessian methods via `p.inv_hessian_diag`
- Updated `run_preconditioner_tests.sh` to reference new cluster framework
- Consolidated Markdown docs into `docs/` with guides and reference index
- Renamed Hessian diagonal variants to canonical names (`svd_principal_alpha`, `mm_jensen`,
  `frobenius_surrogate_pd`, `vector_tv_per_modality`) and updated scripts/configs; legacy aliases retained
- SPECT→PET resampling now relies solely on the direct `NiftyResampleOperator` (see `setr.scripts.common.get_resampling_operators` and the new scripts) so the previous enlarge-then-zoom pipeline has been dropped in favour of an adjoint-consistent warp informed by the stored no-zoom displacement.

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

[unreleased]: https://github.com/samdporter/setr/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/samdporter/setr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/samdporter/setr/releases/tag/v0.1.0

### Fixed
- README.md title
