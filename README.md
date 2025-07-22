# SETR
## Synergistic Emission Tomographic Reconstruction

Tools for nuclear medicine image reconstruction combining PET/SPECT with CT anatomical information.

## Contents

- **Vectorial Total Variation** regularization
- **GPU-accelerated gradients and Jacobians**
- **Python wrapper for STIR Kernel EM**
- **Numba-optimized kernel operators**
- **Extended CIL functionality:**
  - Preconditioners
  - Callbacks  
  - Step size rules
- **SIRF data loading utilities**

## Installation

```bash
# Install dependencies via conda / pip or from source
- SIRF
- CIL

# Install SETR
pip install py-setr
```

## Note
In order to use the partitioner with SPECT, you will need to build stir with
https://github.com/samdporter/STIR/tree/SPECT_subsets