# Synergistic Emission Tomographic Reconstruction (SETR)

Monorepo for multimodal PET/SPECT/CT reconstruction research.

## Repository Structure

This repository contains two packages:

### 📦 [recon_core](recon_core/) - Core Reconstruction Library

Stable, versioned library providing:
- GPU-accelerated regularisation (Vectorial Total Variation, RDP, MI)
- CIL framework extensions (operators, preconditioners, callbacks)
- Low-level EM kernels
- SIRF/CIL utilities

**Status**: Stable API, minimal dependencies
**Version**: 0.2.0
**Dependencies**: torch, numba, numpy, matplotlib, pandas, pyyaml + CIL, SIRF (external)

### 🧪 [recon_experiments](recon_experiments/) - Experiments & Analysis

Research infrastructure providing:
- Pre-configured experiment runners (DTNV, HKEM, RDP, etc.)
- Parameter sweep framework
- HPC/cluster integration
- Analysis tools and notebooks
- Organised research studies

**Status**: Active development
**Version**: 0.1.0
**Dependencies**: recon-core + seaborn, tqdm, jupyter (optional)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/samdporter/setr.git
cd synergistic_recon

# Install core library
cd recon_core
pip install -e .

# Install experiments package
cd ../recon_experiments
pip install -e .
```

### External Dependencies

Before using SETR, install these external packages:
- [SIRF](https://github.com/SyneRBI/SIRF) - Synergistic Image Reconstruction Framework
- [CIL](https://github.com/TomographicImaging/CIL) - Core Imaging Library
- [STIR](https://github.com/UCL/STIR) - Software for Tomographic Image Reconstruction

See [recon_core/README.md](recon_core/README.md) for detailed installation instructions.

### Running Reconstructions

```bash
# Quick test reconstruction
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config recon_experiments/configs/config_1bpos_manc.yaml \
    --override num_epochs=10

# Full reconstruction
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config recon_experiments/configs/config_1bpos_manc.yaml
```

## Migration from Old Structure

If you're updating from the previous monolithic structure (`setr` package), see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for complete migration instructions.

**Summary of changes:**
- `setr.*` → `recon_core.*` (core library imports)
- `setr.scripts.*` → `recon_experiments.runners.*` (experiment helpers)
- Two separate packages with independent versioning
- Preserved numerical behaviour (results unchanged)

## Documentation

- **Core Library**: [recon_core/README.md](recon_core/README.md)
- **Experiments**: [recon_experiments/README.md](recon_experiments/README.md)
- **Migration Guide**: [docs/migration.md](docs/migration.md)
- **Cluster Usage**: [docs/cluster-usage.md](docs/cluster-usage.md)
- **Full Documentation Index**: [docs/index.md](docs/index.md)

## Key Features

### Advanced Regularisation
- Vectorial Total Variation (GPU-accelerated with Schatten norms)
- Relative Difference Prior (RDP) with directional variants
- Mutual Information priors for anatomical guidance
- Kappa-squared weighting for cross-modal coupling

### Multimodal Support
- Joint PET/SPECT/CT reconstruction
- Multiple bed position handling
- Flexible resampling between modalities
- Anatomically-guided regularisation

### High Performance
- CUDA acceleration for priors and gradients
- Subset-based methods (OSEM, BSREM)
- Advanced preconditioners (BSREM, Lehmer mean)
- Variance reduction strategies

### Experiment Infrastructure
- Pre-configured reconstruction pipelines
- Parameter sweep framework for HPC
- Comprehensive metrics and visualisation
- Organised study workflows

## Development

### Running Tests

```bash
# Core library tests
cd recon_core
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=recon_core --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint
ruff check src/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Project History

**2026-01-13**: Repository split into recon_core and recon_experiments
- Established clean API boundaries
- Separated stable library from research code
- Reduced core dependencies
- Enabled independent versioning

**Previous versions**: Monolithic `setr` package (now deprecated)

## Citation

If you use SETR in your research, please cite:

```bibtex
@software{setr2024,
  author = {Porter, Sam},
  title = {SETR: Synergistic Emission Tomographic Reconstruction},
  year = {2024},
  url = {https://github.com/samdporter/setr}
}
```

## License

MIT License - see LICENSE file for details.

## Contributors

- Sam Porter (UCL) - Primary author and maintainer

## Support

- **Issues**: https://github.com/samdporter/setr/issues
- **Documentation**: https://setr.readthedocs.io
- **Discussions**: https://github.com/samdporter/setr/discussions

## Acknowledgements

This work builds upon:
- **SIRF**: Software for Synergistic Image Reconstruction Framework
- **CIL**: Core Imaging Library (CCPi/TomographicImaging)
- **STIR**: Software for Tomographic Image Reconstruction

Developed at University College London (UCL).
