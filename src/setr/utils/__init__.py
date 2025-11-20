"""
Lazy-export utilities to avoid importing heavy dependencies unless needed.
"""

from importlib import import_module
from typing import Dict, Tuple

__all__ = [
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

_IMPORT_MAP: Dict[str, Tuple[str, str]] = {
    "BlockDataContainerToArray": ("setr.utils.cil", "BlockDataContainerToArray"),
    "apply_overrides": ("setr.utils.io", "apply_overrides"),
    "create_spect_uniform_image": ("setr.utils.sirf", "create_spect_uniform_image"),
    "get_pet_am": ("setr.utils.sirf", "get_pet_am"),
    "get_pet_data": ("setr.utils.sirf", "get_pet_data"),
    "get_pet_data_multiple_bed_pos": ("setr.utils.sirf", "get_pet_data_multiple_bed_pos"),
    "get_spect_am": ("setr.utils.sirf", "get_spect_am"),
    "get_spect_data": ("setr.utils.sirf", "get_spect_data"),
    "load_config": ("setr.utils.io", "load_config"),
    "parse_cli": ("setr.utils.io", "parse_cli"),
    "save_args": ("setr.utils.io", "save_args"),
}


def __getattr__(name):
    if name not in _IMPORT_MAP:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module_name, attr_name = _IMPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
