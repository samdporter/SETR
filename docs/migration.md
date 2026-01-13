# Migration Guide: setr → recon_core + recon_experiments

This guide documents all changes required when migrating from the old monolithic `setr` package to the new split structure with `recon_core` and `recon_experiments`.

## Overview

The repository has been split into two packages:

- **recon_core**: Core reconstruction library (stable, minimal dependencies)
- **recon_experiments**: Experiment runners, sweeps, and analysis tools

## Installation

### Old Structure
```bash
pip install -e .
```

### New Structure
```bash
# Install both packages
cd recon_core
pip install -e .

cd ../recon_experiments
pip install -e .
```

## Import Changes

### Core Library Imports

All core functionality now imports from `recon_core`:

| Old Import | New Import |
|------------|------------|
| `from setr.cil_extensions import *` | `from recon_core.cil_extensions import *` |
| `from setr.priors import *` | `from recon_core.priors import *` |
| `from setr.kernel import *` | `from recon_core.kernel import *` |
| `from setr.utils import *` | `from recon_core.utils import *` |
| `from setr.core import *` | `from recon_core.core import *` |

### Experiment Runner Imports

Experiment runner helpers now import from `recon_experiments.runners`:

| Old Import | New Import |
|------------|------------|
| `from setr.scripts.common import init_run_env` | `from recon_experiments.runners.common import init_run_env` |
| `from setr.scripts.dtnv_common import get_algorithm` | `from recon_experiments.runners.dtnv_common import get_algorithm` |
| `from setr.scripts.hkem_common import setup_kernel` | `from recon_experiments.runners.hkem_common import setup_kernel` |

## Detailed Migration Examples

### Example 1: Simple Script Using Core Library

**Old code:**
```python
from setr.priors import WeightedVectorialTotalVariation
from setr.cil_extensions.operators import NiftyResampleOperator
from setr.utils import get_pet_am, load_config

prior = WeightedVectorialTotalVariation(weights=[1, 1, 1])
resampler = NiftyResampleOperator(...)
am = get_pet_am(...)
config = load_config("config.yaml")
```

**New code:**
```python
from recon_core.priors import WeightedVectorialTotalVariation
from recon_core.cil_extensions.operators import NiftyResampleOperator
from recon_core.utils import get_pet_am, load_config

prior = WeightedVectorialTotalVariation(weights=[1, 1, 1])
resampler = NiftyResampleOperator(...)
am = get_pet_am(...)
config = load_config("config.yaml")
```

### Example 2: Experiment Runner Script

**Old code:**
```python
from setr.cil_extensions.framework import EnhancedBlockDataContainer
from setr.scripts.common import init_run_env, save_results
from setr.scripts.dtnv_common import get_algorithm, get_callbacks
from setr.utils.io import load_config

config = load_config(args.config)
msg_red = init_run_env(config)
algo = get_algorithm(...)
callbacks = get_callbacks(...)
```

**New code:**
```python
from recon_core.cil_extensions.framework import EnhancedBlockDataContainer
from recon_experiments.runners.common import init_run_env, save_results
from recon_experiments.runners.dtnv_common import get_algorithm, get_callbacks
from recon_core.utils.io import load_config

config = load_config(args.config)
msg_red = init_run_env(config)
algo = get_algorithm(...)
callbacks = get_callbacks(...)
```

### Example 3: Test File

**Old code:**
```python
import pytest
from setr.priors.vtv.schatten_norm_gpu import GPUVectorialTotalVariation
from setr.core.gradients import Jacobian
from setr.utils import BlockDataContainerToArray

def test_vtv_gradient():
    jacobian = Jacobian(...)
    vtv = GPUVectorialTotalVariation(...)
    # test code
```

**New code:**
```python
import pytest
from recon_core.priors.vtv.schatten_norm_gpu import GPUVectorialTotalVariation
from recon_core.core.gradients import Jacobian
from recon_core.utils import BlockDataContainerToArray

def test_vtv_gradient():
    jacobian = Jacobian(...)
    vtv = GPUVectorialTotalVariation(...)
    # test code
```

## Running Experiments

### Old Structure
```bash
# From repository root
python scripts/run_dtnv_1bpos.py --config configs/config_1bpos.yaml
```

### New Structure

**Option 1: As module (recommended)**
```bash
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config recon_experiments/configs/config_1bpos.yaml
```

**Option 2: Direct execution**
```bash
cd recon_experiments/src/recon_experiments/runners/scripts
python run_dtnv_1bpos.py --config ../../../../configs/config_1bpos.yaml
```

## Configuration Files

Config files remain largely the same, but paths may need updating:

### Old Structure
```yaml
output_path: output/experiment_name
working_path: tmp/experiment_name
```

### New Structure
```yaml
# Use absolute paths or paths relative to execution directory
output_path: /home/user/recon_experiments/output/experiment_name
working_path: /tmp/experiment_name
```

## Public API Changes

The `recon_core` package now exposes a curated public API. All previously accessible classes are still available, but now through a cleaner interface:

### Direct Top-Level Imports (Recommended)

```python
# Old: required full module path
from setr.priors.vtv.vtv import WeightedVectorialTotalVariation

# New: available from top level
from recon_core import WeightedVectorialTotalVariation
```

### Module-Level Imports (Still Supported)

```python
# Old
from setr.priors import WeightedVectorialTotalVariation

# New
from recon_core.priors import WeightedVectorialTotalVariation
```

Both styles work, but top-level imports from `recon_core` are preferred for the public API.

## Directory Structure Changes

### Old Structure
```
synergistic_recon/
├── src/setr/              # Everything mixed together
│   ├── cil_extensions/
│   ├── priors/
│   ├── utils/
│   └── scripts/           # Experiment helpers (problematic!)
├── scripts/               # Experiment runners
├── configs/               # Experiment configs
└── tests/                 # Tests
```

### New Structure
```
synergistic_recon/
├── recon_core/            # Core library package
│   ├── src/recon_core/
│   │   ├── cil_extensions/
│   │   ├── priors/
│   │   ├── utils/
│   │   ├── kernel/
│   │   └── core/
│   ├── tests/
│   └── data/
│
└── recon_experiments/     # Experiments package
    ├── src/recon_experiments/
    │   ├── runners/       # From src/setr/scripts + scripts/
    │   ├── sweeps/
    │   ├── studies/
    │   └── experiments/
    └── configs/
```

## Breaking Changes

### 1. Package Name
- `setr` → `recon_core` (for core library)
- No direct equivalent for experiments (use `recon_experiments.runners`)

### 2. Experiment Runners Location
- **Old**: `src/setr/scripts/common.py`
- **New**: `recon_experiments/src/recon_experiments/runners/common.py`

### 3. Test Imports
- All test files updated to import from `recon_core`
- No changes to test logic or assertions

### 4. CLI Entrypoints
- **Old**: `setr` command (if installed)
- **New**: `run-recon` command from experiments package

## Backward Compatibility

**There is NO backward compatibility** - all imports must be updated. However:

- **Numerical results are preserved** - algorithms are unchanged
- **Config file format is unchanged** - YAML structure is the same
- **API surface is unchanged** - all classes/functions still available

## Testing After Migration

### Quick Validation

```bash
# Test core imports
python -c "from recon_core import WeightedVectorialTotalVariation; print('✓ Core imports work')"

# Test experiments imports
python -c "from recon_experiments.runners.common import init_run_env; print('✓ Experiments imports work')"

# Run core tests
cd recon_core
pytest tests/ -m "not slow"

# Run quick experiment
cd ../recon_experiments
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config configs/config_test_debug.yaml \
    --override num_epochs=1
```

### Numerical Verification

To verify results are unchanged:

1. Run same config with old code, save output
2. Run same config with new code, save output
3. Compare:
   ```python
   import numpy as np
   from sirf.STIR import ImageData

   old = ImageData("old_output/final_image.hv")
   new = ImageData("new_output/final_image.hv")

   diff = old.as_array() - new.as_array()
   print(f"Max absolute difference: {np.abs(diff).max()}")
   print(f"Relative error: {np.linalg.norm(diff) / np.linalg.norm(old.as_array())}")
   ```

Results should match to machine precision.

## Common Migration Issues

### Issue 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'setr'
```

**Solution:**
Update all imports from `setr` to either `recon_core` or `recon_experiments.runners`

### Issue 2: Path Issues in Configs

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'output/...'
```

**Solution:**
Use absolute paths in config files or ensure you run from the correct directory

### Issue 3: Circular Imports

**Error:**
```
ImportError: cannot import name 'X' from partially initialized module
```

**Solution:**
This shouldn't occur with the new structure. If it does, check for accidental mutual dependencies.

### Issue 4: Missing SIRF/CIL

**Error:**
```
ModuleNotFoundError: No module named 'sirf'
```

**Solution:**
Install SIRF and CIL from source (see recon_core/README.md)

## Automation Script

For bulk migration of import statements:

```bash
#!/bin/bash
# migrate_imports.sh

# Update core imports
find . -name "*.py" -type f -exec sed -i \
    's/from setr\.\([^s]\)/from recon_core.\1/g' {} \;

# Update script imports
find . -name "*.py" -type f -exec sed -i \
    's/from setr\.scripts\./from recon_experiments.runners./g' {} \;

# Update import statements
find . -name "*.py" -type f -exec sed -i \
    's/import setr\./import recon_core./g' {} \;

echo "Migration complete. Review changes and test!"
```

**⚠️ Always review automated changes before committing!**

## Getting Help

If you encounter issues during migration:

1. Check this guide for your specific use case
2. Review the [recon_core README](recon_core/README.md)
3. Review the [recon_experiments README](recon_experiments/README.md)
4. Open an issue on [GitHub](https://github.com/samdporter/setr/issues)

## Timeline

- **Old structure**: Deprecated as of 2024-01-13
- **New structure**: Active development
- **Support**: Old imports will not be maintained; migrate as soon as possible
