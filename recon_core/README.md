# recon_core

Core reconstruction library for multimodal PET/SPECT/CT imaging.

## Overview

`recon_core` provides the foundational algorithms, operators, and utilities for synergistic emission tomographic reconstruction. It combines PET, SPECT, and CT anatomical information using GPU-accelerated algorithms with advanced regularisation techniques.

### Key Features

- **CIL Extensions**: Custom algorithms, callbacks, operators, and preconditioners extending the Core Imaging Library (CIL)
- **Advanced Regularisation**: Vectorial Total Variation (VTV), Relative Difference Prior (RDP), and Mutual Information priors
- **GPU Acceleration**: CUDA-accelerated implementations for performance-critical operations
- **Multimodal Support**: Unified framework for PET, SPECT, and CT data
- **Flexible Architecture**: Modular design with clear separation of concerns

## Installation

### Prerequisites

**External dependencies (must be installed separately):**
- [SIRF](https://github.com/SyneRBI/SIRF) - Software for Synergistic Image Reconstruction Framework
- [CIL](https://github.com/TomographicImaging/CIL) - Core Imaging Library
- [STIR](https://github.com/UCL/STIR) - Software for Tomographic Image Reconstruction (SPECT_subsets branch)

### Install from source

```bash
cd recon_core
pip install -e .

# For development with testing and linting tools:
pip install -e .[dev]

# For running tests:
pip install -e .[test]
```

## Quick Start

```python
import recon_core as rc

# Use regularisation priors
prior = rc.WeightedVectorialTotalVariation(
    weights=[1.0, 1.0, 1.0],
    epsilon=1e-4,
    no_gpu=False
)

# Set up preconditioners
precond = rc.BSREMPreconditioner(kappa=1.0, eta=1e-10)

# Use CIL extensions
container = rc.EnhancedBlockDataContainer(image_list)

# Access SIRF utilities
pet_am = rc.get_pet_am(template, data_path)
```

## Package Structure

```
recon_core/
├── cil_extensions/      # CIL framework extensions
│   ├── algorithms/      # Custom ISTA updates
│   ├── callbacks/       # Progress tracking, image saving
│   ├── operators/       # Resampling, transforms
│   ├── preconditioners/ # BSREM, image function, Lehmer mean
│   ├── functions/       # Block indicators, KL variants
│   └── framework/       # EnhancedBlockDataContainer
├── priors/              # Regularisation functions
│   └── vtv/             # Vectorial Total Variation (GPU/CPU)
├── kernel/              # Low-level EM kernels
├── core/                # Core computational components
│   └── gradients/       # Gradient operators (Jacobian, etc.)
└── utils/               # Utility functions
    ├── sirf.py          # SIRF data loading, acquisition models
    ├── cil.py           # CIL framework utilities
    ├── io.py            # Configuration and I/O
    └── nifty.py         # NiftyReg integration
```

## Public API

The public API is carefully curated and exported from the top-level `recon_core` module. See the [API Reference](https://setr.readthedocs.io) for complete documentation.

### Core Components

- **Regularisation**: `WeightedVectorialTotalVariation`, `RelativeDifferencePrior`, `WeightedRDP`
- **Operators**: `NiftyResampleOperator`, `ScalingOperator`, `FlipOperator`
- **Preconditioners**: `BSREMPreconditioner`, `ImageFunctionPreconditioner`, `LehmerMeanPreconditioner`
- **Callbacks**: `SaveImageCallback`, `SaveObjectiveCallback`, `ComputeMetricsCallback`
- **Gradients**: `Jacobian`, `Gradient`, `DirectionalGradient`
- **Utilities**: `get_pet_am`, `get_spect_am`, `load_config`, `BlockDataContainerToArray`

## Configuration

Configuration files use YAML format. Example:

```yaml
alpha: 1.0  # PET data fidelity weight
beta: 0.25  # SPECT data fidelity weight
num_subsets: [7, 7]  # [PET, SPECT]
num_epochs: 100
pet_gauss_fwhm: 4.0  # mm
spect_gauss_fwhm: 8.0  # mm
use_kappa: true  # Enable kappa-squared weighting
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Exclude slow tests
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/test_schatten.py -v

# Run with coverage
pytest tests/ --cov=recon_core --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

## Dependencies

**Core dependencies:**
- `torch>=2.0.0` - GPU acceleration
- `numba>=0.58.0` - JIT compilation
- `numpy>=1.21.0,<2.0.0` - Numerical computing
- `matplotlib>=3.5.0` - Visualisation
- `pandas>=1.3.0` - Data handling (CSV I/O)
- `pyyaml>=5.4.0` - Configuration parsing

**External (install separately):**
- SIRF, CIL, STIR - See Prerequisites above

## Citation

If you use this software in your research, please cite:

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

## Support

- **Issues**: [GitHub Issues](https://github.com/samdporter/setr/issues)
- **Documentation**: [Read the Docs](https://setr.readthedocs.io)
- **Repository**: [GitHub](https://github.com/samdporter/setr)
