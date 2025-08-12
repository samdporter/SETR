#!/usr/bin/env python3
"""
Resample SPECT reconstruction to PET space for use as guidance in PET reconstruction.
"""

import argparse
import logging
import os

from sirf.Reg import NiftiImageData3DDisplacement
from sirf.STIR import ImageData

from setr.cil_extensions.operators import NiftyResampleOperator
from setr.utils.io import apply_overrides, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample SPECT reconstruction to PET space"
    )

    # Config-based approach
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="*",
        help="Override config values, e.g. pet_template=path/to/template.hv",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Handle config-based approach (only mode supported now)
    if not args.config:
        raise ValueError("--config is required. Please provide a YAML config file.")

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # Expand environment variables in paths
    for key in [
        "pet_template",
        "spect_reconstruction",
        "transform_file",
        "output_file",
    ]:
        if key in cfg:
            cfg[key] = os.path.expandvars(cfg[key])

    pet_template_path = cfg["pet_template"]
    spect_recon_path = cfg["spect_reconstruction"]
    transform_path = cfg["transform_file"]
    output_path = cfg["output_file"]

    # Load data
    logging.info(f"Loading PET template: {pet_template_path}")
    pet_template = ImageData(pet_template_path)

    logging.info(f"Loading SPECT reconstruction: {spect_recon_path}")
    spect_recon = ImageData(spect_recon_path)

    logging.info(f"Loading transformation: {transform_path}")
    transform = NiftiImageData3DDisplacement(transform_path)

    # Create resampler
    resampler = NiftyResampleOperator(
        reference=pet_template, floating=spect_recon, transform=transform
    )

    # Resample
    logging.info("Resampling SPECT to PET space...")
    spect_recon2pet = resampler.direct(spect_recon)

    # Save result
    logging.info(f"Saving resampled image: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    spect_recon2pet.write(output_path)

    logging.info("Resampling completed successfully")


if __name__ == "__main__":
    main()
