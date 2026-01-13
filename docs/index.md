# SETR Documentation

Welcome to the Synergistic Emission Tomographic Reconstruction (SETR) documentation.

## Getting Started

- **[Installation Guide](../recon_core/README.md#installation)** - Install core library and dependencies
- **[Quick Start](../README.md#quick-start)** - Run your first reconstruction
- **[Migration Guide](migration.md)** - Migrating from old `setr` package structure

## Package Documentation

### Core Library (recon_core)

The stable reconstruction library providing GPU-accelerated algorithms and regularisation.

- **[Core Library README](../recon_core/README.md)** - Complete library documentation
- **[API Reference](../recon_core/README.md#api-overview)** - Public API surface
- **[Changelog](../recon_core/CHANGELOG.md)** - Version history

### Experiments Package (recon_experiments)

Research infrastructure for running experiments, parameter sweeps, and HPC workflows.

- **[Experiments README](../recon_experiments/README.md)** - Experiments framework guide
- **[Sweeps Framework](../recon_experiments/src/recon_experiments/sweeps/README.md)** - Parameter sweeps
- **[Studies](../recon_experiments/src/recon_experiments/studies/)** - Organised research studies
- **[Experiment Scripts](../recon_experiments/src/recon_experiments/experiments/README.md)** - Experiment runners

## User Guides

### Cluster and HPC

- **[Cluster Usage Guide](cluster-usage.md)** - Running SETR on HPC clusters (comprehensive)
- **[Bootstrap Workflows](manc_bootstrap_workflows.md)** - Bootstrap reconstruction workflows
- **[Bootstrap Quick Start](manc_bootstrap_quickstart.md)** - Quick reference for bootstrap runs

### Advanced Topics

- **[Preconditioner Testing](guides/preconditioner_testing.md)** - Testing VTV preconditioners
- **[Preconditioner Sweeps](guides/preconditioners_cluster.md)** - Cluster-based preconditioner comparison
- **[Parameter Sweeps](guides/sweeps.md)** - General sweep framework usage
- **[Metrics and Analysis](guides/metrics.md)** - Computing and analysing reconstruction metrics

## Technical Reference

### Architecture and Design

- **[Repository Split](architecture/repository-split.md)** - Monorepo structure and refactoring history
- **[VTV Hessian Diagonals](architecture/vtv_hessian_diagonals.md)** - Mathematical derivations for VTV preconditioners

### Core Concepts

#### Regularisation

- **Vectorial Total Variation (VTV)** - GPU-accelerated multimodal regularisation with Schatten norms
- **Relative Difference Prior (RDP)** - Anatomically-guided regularisation
- **Mutual Information** - Cross-modal similarity priors

#### Algorithms

- **DTNV** - Dual Total Nuclear Variation (PET + SPECT joint reconstruction)
- **HKEM** - Hybrid Kernel EM with anatomical guidance
- **BSREM** - Block Sequential Regularised Expectation Maximisation

## Development

- **[Development Setup](../README.md#development)** - Set up development environment
- **[Running Tests](../recon_core/README.md#testing)** - Test suite and coverage
- **[Code Quality](../README.md#code-quality)** - Formatting and linting tools

## Support and Resources

- **Issues**: [GitHub Issues](https://github.com/samdporter/setr/issues)
- **Repository**: [github.com/samdporter/setr](https://github.com/samdporter/setr)
- **External Dependencies**: [SIRF](https://github.com/SyneRBI/SIRF), [CIL](https://github.com/TomographicImaging/CIL), [STIR](https://github.com/UCL/STIR)

## License

MIT License - see [LICENSE](../LICENSE) file for details.
