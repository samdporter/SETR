# Changelog - recon_core

All notable changes to the core reconstruction library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Stabilised `2x2` symmetric eigenvector construction in `vtv` (`small_eig.eigenvecsh_2x2`) for near-degenerate singular values, preventing gradient sign/magnitude failures in the analytic (`stable=False`) backend.

### Added
- Regression finite-difference tests for weighted VTV (`tests/test_vtv_finite_difference.py`), including a pathological near-degenerate `2x3` case that previously exposed backend divergence.
- Optional end-to-end real-data full-pipeline objective FD check scaffold (`tests/test_full_pipeline_objective_fd.py`) behind `SETR_RUN_FULL_PIPELINE_FD=1`.

## [0.2.0] - 2024-01-13

### Changed
- **BREAKING**: Package renamed from `setr` to `recon_core`
- **BREAKING**: All imports now use `recon_core` namespace
- Reorganised as standalone package with clean public API
- Updated all internal imports to use new package name
- Minimal dependency set (removed experiment-specific dependencies)

### Added
- Comprehensive public API exposed at top level
- Explicit `__all__` exports for clean namespace
- Dedicated README.md with installation and usage instructions
- This CHANGELOG.md file

### Removed
- Experiment runner code (moved to `recon_experiments` package)
- `src/setr/scripts/` directory (now in experiments package)

### Fixed
- No functional changes - algorithms preserve numerical behaviour
- All existing functionality maintained

### Migration
- See [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) for import changes
- All imports: `setr.*` → `recon_core.*`
- Numerical results unchanged

## [0.1.1] - Previous Version

Previous monolithic package version (deprecated).

## [0.1.0] - Initial Release

Initial version of monolithic `setr` package.
