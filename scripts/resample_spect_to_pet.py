#!/usr/bin/env python3
"""
Resample SPECT reconstruction to PET space for use as guidance in PET reconstruction.
"""

import argparse
import logging
import os
import string

from sirf.Reg import NiftiImageData3DDisplacement
from sirf.STIR import ImageData

from setr.cil_extensions.operators import NiftyResampleOperator
from setr.utils.io import apply_overrides, load_config
from setr.utils import get_pet_data_multiple_bed_pos, get_pet_data


def parse_args():
    parser = argparse.ArgumentParser(description="Resample SPECT reconstruction to PET space")

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
    cfg = {
        k: template(v).safe_substitute(cfg) if isinstance(v, str) else v
        for k, v in cfg.items()
    }

    # Expand environment variables in paths
    for key in [
        "pet_template",
        "spect_reconstruction",
        "transform_file",
        "output_file",
    ]:
        if key in cfg:
            cfg[key] = os.path.expandvars(cfg[key])

    if cfg["use_2bpos"]:
        pet_data = get_pet_data_multiple_bed_pos(
            cfg["pet_dir"], tof=True, suffixes=["_f1b1", "_f2b1"]
        )
    else:
        pet_data = get_pet_data(cfg["pet_dir"], tof=True)
    pet_template = pet_data["template_image"]
    spect_recon_path = cfg["spect_reconstruction"]
    transform_path = cfg["transform_file"]
    output_path = cfg["output_file"]

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

    # Now we need to get the individual bed positions aligned correctly
    if cfg["use_2bpos"]:
        resampler_f1b1 = NiftyResampleOperator(
            reference=pet_data["bed_positions"]["_f1b1"]["template_image"],
            floating=spect_recon,
            transform=NiftiImageData3DDisplacement(
                transform_path.replace(".nii", "_f1b1.nii")
            ),
        )
        resampler_f2b1 = NiftyResampleOperator(
            reference=pet_data["bed_positions"]["_f2b1"]["template_image"],
            floating=spect_recon,
            transform=NiftiImageData3DDisplacement(
                transform_path.replace(".nii", "_f2b1.nii")
            ),
        )
        spect_recon2pet_f1b1 = resampler_f1b1.direct(spect_recon)
        spect_recon2pet_f2b1 = resampler_f2b1.direct(spect_recon)
        
        # Save individual bed position images
        output_path_f1b1 = output_path.replace(".hv", "_f1b1.hv")
        output_path_f2b1 = output_path.replace(".hv", "_f2b1.hv")
        logging.info(f"Saving resampled image for bed position 1: {output_path_f1b1}")
        spect_recon2pet_f1b1.write(output_path_f1b1)
        logging.info(f"Saving resampled image for bed position 2: {output_path_f2b1}")
        spect_recon2pet_f2b1.write(output_path_f2b1)

    logging.info("Resampling completed successfully")


if __name__ == "__main__":
    main()
