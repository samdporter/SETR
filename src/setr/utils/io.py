import argparse
import ast
import yaml
import os
import pandas as pd
import logging

def parse_cli():
    p = argparse.ArgumentParser(description="BSREM (YAML‑driven)")
    p.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to YAML config file"
    )
    p.add_argument(
        "--override", "-o", type=str, nargs="*",
        help="Override YAML keys, e.g. alpha=0.5 beta=2"
    )
    return p.parse_args()

def load_config(path: str) -> dict:
    """Load YAML into a nested dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Given overrides like ["alpha=0.5", "spect.gauss_fwhm=[1,2,3]"],
    apply them into cfg (supports nested keys via dots).
    """
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        try:
            val = ast.literal_eval(val)
        except Exception:
            pass
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