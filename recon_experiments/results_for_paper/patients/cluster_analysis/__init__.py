"""
Tools for analysing reconstruction parameter sweeps using SIRF.STIR ImageData.

The package provides the following main components:

Core Analysis:
* ``interfile`` – image loading using SIRF.STIR ImageData
* ``masking`` – mask generation strategies operating on 1-D voxel buffers.
* ``stats`` – numerically stable summary statistics for masked voxels.
* ``analysis`` – high-level orchestration for parameter grid analysis.
* ``plotting`` – SVG-based visualization utilities.

Interactive Jupyter Support:
* ``jupyter_utils`` – interactive image viewing and mask creation for Jupyter
  notebooks with matplotlib widgets. Works directly with SIRF.STIR ImageData objects.

Command Line:
* See :mod:`cluster_analysis.__main__` for a command line entry point.
"""

from .analysis import (
    AnalysisConfig,
    ImageResult,
    AggregateResult,
    analyse_parameter_grid,
    write_results_csv,
)
from .plotting import plot_l_curve_svg

# Jupyter utilities are optional (require numpy and matplotlib)
try:
    from .jupyter_utils import (
        SpatialMask,
        ImageViewer,
        view_image,
        apply_mask_batch,
    )
    _JUPYTER_AVAILABLE = True
except ImportError:
    _JUPYTER_AVAILABLE = False

__all__ = [
    "AnalysisConfig",
    "ImageResult",
    "AggregateResult",
    "analyse_parameter_grid",
    "write_results_csv",
    "plot_l_curve_svg",
]

# Add jupyter utilities to exports if available
if _JUPYTER_AVAILABLE:
    __all__.extend([
        "SpatialMask",
        "ImageViewer",
        "view_image",
        "apply_mask_batch",
    ])
