#!/usr/bin/env python3
"""
Simple resample SPECT reconstruction to PET space for MANC phantom data.
This version supports flipping and uses simpler resampling (no zoom operation).
"""

import argparse
import logging
import os
import string

from cil.optimisation.operators import CompositionOperator
from sirf.Reg import NiftiImageData3DDisplacement
from sirf.STIR import ImageData

from setr.cil_extensions.operators import (
    EnlargementOperator,
    FlipOperator,
    NiftyResampleOperator,
)
from setr.utils import get_pet_data, get_pet_data_multiple_bed_pos, get_spect_data
from setr.utils.io import apply_overrides, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample SPECT reconstruction to PET space (simple version for MANC data)"
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

    template = string.Template
    cfg = {k: template(v).safe_substitute(cfg) if isinstance(v, str) else v for k, v in cfg.items()}

    # Expand environment variables in paths
    for key in [
        "pet_template",
        "spect_reconstruction",
        "transform_file",
        "output_file",
    ]:
        if key in cfg:
            cfg[key] = os.path.expandvars(cfg[key])

    # Get flip parameter (default: False)
    flip = cfg.get("flip", False)

    # Get PET template
    if cfg.get("use_2bpos", False):
        pet_data = get_pet_data_multiple_bed_pos(
            cfg["pet_dir"], tof=cfg.get("use_tof", False), suffixes=["_f1b1", "_f2b1"]
        )
    else:
        pet_data = get_pet_data(cfg["pet_dir"], tof=cfg.get("use_tof", False))
    pet_template = pet_data["template_image"]

    spect_recon_path = cfg["spect_reconstruction"]
    transform_path = cfg["transform_file"]
    output_path = cfg["output_file"]

    spect_data = get_spect_data(cfg["spect_dir"])

    logging.info(f"Loading SPECT reconstruction: {spect_recon_path}")
    spect_recon = ImageData(spect_recon_path)

    logging.info(f"Loading transformation: {transform_path}")
    transform = NiftiImageData3DDisplacement(transform_path)

    # Build operator chain
    operators = []

    # 1. Flip if needed
    if flip:
        logging.info("Adding flip operator for MANC data")
        flip_op = FlipOperator(spect_recon)
        operators.append(flip_op)
        current_image = flip_op.direct(spect_recon)
    else:
        current_image = spect_recon

    # 2. Enlargement (always needed)
    logging.info("Adding enlargement operator")
    enlarger = EnlargementOperator(
        enlarged_shape=(128, 256, 256),
        enlargement_sino=spect_data["acquisition_data"],
        original_floating=current_image,
    )
    operators.append(enlarger)
    current_image = enlarger.direct(current_image)

    # 3. Resampling (no zoom for MANC data)
    logging.info("Adding resampling operator")
    resampler = NiftyResampleOperator(
        reference=pet_template,
        floating=current_image,
        transform=transform,
    )
    operators.append(resampler)

    # Create composition
    if len(operators) > 1:
        composition = CompositionOperator(*reversed(operators))
    else:
        composition = operators[0]

    # Resample
    logging.info("Resampling SPECT to PET space...")
    if flip:
        spect_recon2pet = composition.direct(spect_recon)
    else:
        spect_recon2pet = composition.direct(spect_recon)

    # Save result
    logging.info(f"Saving resampled image: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    spect_recon2pet.write(output_path)

    logging.info("Resampling completed successfully")


if __name__ == "__main__":
    main()
