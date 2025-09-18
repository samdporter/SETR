import argparse
import ast
import contextlib
import logging
import os

import pandas as pd
import yaml


def parse_cli():
    p = argparse.ArgumentParser(description="BSREM (YAML‑driven)")
    p.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    p.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="*",
        help="Override YAML keys, e.g. alpha=0.5 beta=2",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    """Load YAML into a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config_with_inheritance(path: str) -> dict:
    """
    Load YAML config with inheritance support.

    Supports 'inherit_from' key to inherit from base configs.
    Base configs are resolved recursively and merged with override precedence.

    Args:
        path: Path to the config file

    Returns:
        Merged configuration dictionary
    """
    config = load_config(path)

    # If no inheritance, return as-is
    if 'inherit_from' not in config:
        return config

    # Get base config path(s)
    inherit_from = config.pop('inherit_from')
    if isinstance(inherit_from, str):
        inherit_from = [inherit_from]

    # Load and merge base configs
    base_config = {}
    config_dir = os.path.dirname(path)

    for base_path in inherit_from:
        # Resolve relative paths from config directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(config_dir, base_path)

        # Recursively load base config (may have its own inheritance)
        base = load_config_with_inheritance(base_path)
        base_config = _deep_merge(base_config, base)

    # Merge current config over base configs
    return _deep_merge(base_config, config)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Given overrides like ["alpha=0.5", "spect.gauss_fwhm=[1,2,3]"],
    apply them into cfg (supports nested keys via dots).
    """
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        with contextlib.suppress(Exception):
            val = ast.literal_eval(val)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def save_args(args, output_filename):
    # Save command-line arguments.
    df_args = pd.DataFrame([vars(args)])
    df_args.to_csv(os.path.join(args.output_path, output_filename), index=False)
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info(f"Arguments saved to {os.path.join(args.output_path, output_filename)}")
