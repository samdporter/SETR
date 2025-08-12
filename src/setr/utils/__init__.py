"""
setr.utils
Convenience umbrella for all utils subpackages.
"""

from .cil import BlockDataContainerToArray
from .io import apply_overrides, load_config, parse_cli, save_args

# from .nifty import NothingAsYet
from .sirf import (
    create_spect_uniform_image,
    get_pet_am,
    get_pet_data,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
)

__all__ = [
    "BlockDataContainerToArray",
    # "NothingAsYet",
    "get_pet_am",
    "get_pet_data",
    "get_pet_data_multiple_bed_pos",
    "get_spect_am",
    "get_spect_data",
    "create_spect_uniform_imageparse_cli",
    "load_config",
    "apply_overrides",
]
