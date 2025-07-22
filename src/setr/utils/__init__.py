"""
setr.utils
Convenience umbrella for all utils subpackages.
"""

from .cil import BlockDataContainerToArray
#from .nifty import NothingAsYet
from .sirf import (
    get_pet_am, get_pet_data, get_pet_data_multiple_bed_pos,
    get_spect_am, get_spect_data, create_spect_uniform_image

)
from .io import (
    parse_cli,
    load_config,
    apply_overrides,
    save_args
)

__all__ = [
    "BlockDataContainerToArray",
    # "NothingAsYet",
    "get_pet_am", "get_pet_data", "get_pet_data_multiple_bed_pos",
    "get_spect_am", "get_spect_data", "create_spect_uniform_image"
    "parse_cli", "load_config", "apply_overrides"
]