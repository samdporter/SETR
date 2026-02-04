"""
RECON_CORE: Core reconstruction library for multimodal PET/SPECT/CT imaging.

This package provides:
- CIL extensions for algorithms, callbacks, operators, preconditioners
- Regularisation priors (Vectorial Total Variation with Schatten norms)
- Low-level kernels for EM-based methods
- Core gradient operators for directional TV
- SIRF and CIL utility functions

Public API organised by functionality.
"""

__version__ = "0.2.0"
__author__ = "Sam Porter"
__email__ = "sam.porter.18@ucl.ac.uk"

# ============================================================================
# CIL Extensions - Algorithms
# ============================================================================
from recon_core.cil_extensions.algorithms import ista_update_step

# ============================================================================
# CIL Extensions - Callbacks
# ============================================================================
from recon_core.cil_extensions.callbacks import (
    Callback,
    ComputeMetricsCallback,
    PrintMetricsCallback,
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveKernelisedImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
    SaveStepSizeCallback,
    SubsetValueCallback,
)

# ============================================================================
# CIL Extensions - Framework
# ============================================================================
from recon_core.cil_extensions.framework import EnhancedBlockDataContainer

# ============================================================================
# CIL Extensions - Functions
# ============================================================================
from recon_core.cil_extensions.functions import (
    BlockIndicatorBox,
    ResidualAwareKullbackLeibler,
    ShiftedDataShiftedPoissonKullbackLeibler,
    ShiftedKullbackLeibler,
    ensure_kl_hessian_support,
    patch_KL_with_nonnegative,
)

# ============================================================================
# CIL Extensions - Operators
# ============================================================================
from recon_core.cil_extensions.operators import (
    AdjointOperator,
    CouchShiftOperator,
    DirectionalOperator,
    EnlargementOperator,
    FlipOperator,
    ImageCombineOperator,
    ImageResampleOperator,
    ImageSummationOperator,
    NaNToZeroOperator,
    NiftyResampleOperator,
    ScalingOperator,
    TruncationOperator,
    ZeroEndSlicesOperator,
    ZoomOperator,
)

# ============================================================================
# CIL Extensions - Preconditioners
# ============================================================================
from recon_core.cil_extensions.preconditioners import (
    ArithmeticMeanPreconditioner,
    BSREMPreconditioner,
    ConstantPreconditioner,
    DualModalitySubsetKernelisedEMPreconditioner,
    HarmonicMeanPreconditioner,
    IdentityPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
    PoissonHessianPreconditioner,
    PreconditionerWithInterval,
    SubsetEMPreconditioner,
    SubsetKernelisedEMPreconditioner,
    SubsetPoissonHessianPreconditioner,
    SubsetPreconditioner,
)

# ============================================================================
# CIL Extensions - Utilities
# ============================================================================
from recon_core.cil_extensions.utilities import (
    ArmijoStepSearchRule,
    LinearDecayStepSizeRule,
)

# ============================================================================
# Core Gradients
# ============================================================================
from recon_core.core.gradients import (
    DirectionalGradient,
    Gradient,
    GradientOptimized,
    Jacobian,
    LegacyGradient,
    Sum,
    check_adjoint,
    gpu_directional_op,
)

# ============================================================================
# Kernels
# ============================================================================
from recon_core.kernel import (
    STIRKernelOperator,
)

# ============================================================================
# Priors / Regularisers
# ============================================================================
from recon_core.priors import (
    TotalVariation,
    WeightedLogVectorialTotalVariation,
    WeightedTotalVariation,
    WeightedVectorialTotalVariation,
    schatten_norm_gpu,
    schatten_norm_gpu_slow,
    schatten_norm_gpu_stable,
)

# ============================================================================
# Utilities
# ============================================================================
from recon_core import utils
from recon_core.utils import (
    BlockDataContainerToArray,
    apply_overrides,
    create_spect_uniform_image,
    get_pet_am,
    get_pet_data,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
    load_config,
    parse_cli,
    save_args,
)

# ============================================================================
# Public API Exports
# ============================================================================
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # CIL Extensions - Algorithms
    "ista_update_step",
    # CIL Extensions - Callbacks
    "Callback",
    "SaveImageCallback",
    "SaveKernelisedImageCallback",
    "SaveGradientUpdateCallback",
    "PrintObjectiveCallback",
    "SaveObjectiveCallback",
    "SavePreconditionerCallback",
    "SubsetValueCallback",
    "ComputeMetricsCallback",
    "PrintMetricsCallback",
    "SaveStepSizeCallback",
    # CIL Extensions - Framework
    "EnhancedBlockDataContainer",
    # CIL Extensions - Functions
    "BlockIndicatorBox",
    "ShiftedKullbackLeibler",
    "ShiftedDataShiftedPoissonKullbackLeibler",
    "ResidualAwareKullbackLeibler",
    "ensure_kl_hessian_support",
    "patch_KL_with_nonnegative",
    # CIL Extensions - Operators
    "AdjointOperator",
    "ScalingOperator",
    "ZeroEndSlicesOperator",
    "NaNToZeroOperator",
    "TruncationOperator",
    "DirectionalOperator",
    "NiftyResampleOperator",
    "ZoomOperator",
    "EnlargementOperator",
    "CouchShiftOperator",
    "ImageCombineOperator",
    "ImageResampleOperator",
    "FlipOperator",
    "ImageSummationOperator",
    # CIL Extensions - Preconditioners
    "ConstantPreconditioner",
    "PreconditionerWithInterval",
    "BSREMPreconditioner",
    "ImageFunctionPreconditioner",
    "HarmonicMeanPreconditioner",
    "LehmerMeanPreconditioner",
    "PoissonHessianPreconditioner",
    "ArithmeticMeanPreconditioner",
    "IdentityPreconditioner",
    "SubsetPreconditioner",
    "SubsetEMPreconditioner",
    "DualModalitySubsetKernelisedEMPreconditioner",
    "SubsetKernelisedEMPreconditioner",
    "SubsetPoissonHessianPreconditioner",
    # CIL Extensions - Utilities
    "LinearDecayStepSizeRule",
    "ArmijoStepSearchRule",
    # Core Gradients
    "DirectionalGradient",
    "Gradient",
    "GradientOptimized",
    "LegacyGradient",
    "Sum",
    "Jacobian",
    "gpu_directional_op",
    "check_adjoint",
    # Kernels
    "STIRKernelOperator",
    # Priors
    "TotalVariation",
    "WeightedTotalVariation",
    "WeightedVectorialTotalVariation",
    "WeightedLogVectorialTotalVariation",
    "schatten_norm_gpu_slow",
    "schatten_norm_gpu_stable",
    "schatten_norm_gpu",
    # Utilities
    "utils",
    "BlockDataContainerToArray",
    "apply_overrides",
    "create_spect_uniform_image",
    "get_pet_am",
    "get_pet_data",
    "get_pet_data_multiple_bed_pos",
    "get_spect_am",
    "get_spect_data",
    "load_config",
    "parse_cli",
    "save_args",
]
