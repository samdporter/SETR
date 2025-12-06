#!/usr/bin/env python3
"""
Master script for running phantom reconstruction experiments.

This script orchestrates HKEM, dTNV, and TNV reconstructions across
multiple phantom datasets (NEMA, Manchester NEMA, Anthropomorphic).

It automatically composes configs from base + phantom + algorithm
and manages the two-stage HKEM workflow.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


# Script paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CONFIG_DIR = EXPERIMENTS_DIR / "configs"

# Available phantoms and algorithms
PHANTOMS = ["manc", "anthro", "nema"]
ALGORITHMS = ["hkem", "dtnv", "tnv", "log_dtnv", "log_tnv"]


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(filepath: Path) -> dict:
    """Load YAML configuration file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, filepath: Path):
    """Save dictionary to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def compose_config(phantom: str, algorithm: str, extra_overrides: Dict = None) -> dict:
    """
    Compose configuration from base + phantom + algorithm configs.

    Args:
        phantom: Phantom name (manc, anthro, nema)
        algorithm: Algorithm name (hkem_spect, hkem_pet, dtnv, tnv, log_dtnv, log_tnv)
        extra_overrides: Additional overrides to apply

    Returns:
        Composed configuration dictionary
    """
    # Choose base config based on algorithm type
    # HKEM uses ordered subsets (no variance reduction), different epochs
    # TNV variants use SVRG variance reduction
    if algorithm.startswith("hkem"):
        base_config = load_yaml(CONFIG_DIR / "base_hkem.yaml")
    else:
        base_config = load_yaml(CONFIG_DIR / "base_tnv.yaml")

    # Load phantom-specific config
    phantom_config = load_yaml(CONFIG_DIR / f"phantom_1bpos_{phantom}.yaml")

    # Load algorithm-specific config
    algo_config = load_yaml(CONFIG_DIR / f"algo_{algorithm}.yaml")

    # Merge: base <- phantom <- algorithm <- extra_overrides
    config = deep_merge(base_config, phantom_config)
    config = deep_merge(config, algo_config)

    if extra_overrides:
        config = deep_merge(config, extra_overrides)

    return config


def parse_override(override_str: str) -> Dict:
    """
    Parse override string like 'alpha=0.05' or 'num_epochs=200'.

    Args:
        override_str: Override string in format key=value

    Returns:
        Dictionary with override
    """
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Use key=value")

    key, value = override_str.split('=', 1)

    # Try to convert to appropriate type
    try:
        # Try int first
        value = int(value)
    except ValueError:
        try:
            # Then float
            value = float(value)
        except ValueError:
            # Keep as string, but handle booleans
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False

    return {key: value}


def run_command(cmd: List[str], description: str):
    """
    Run a shell command and handle errors.

    Args:
        cmd: Command as list of strings
        description: Description for logging
    """
    logging.info(f"Running: {description}")
    logging.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)


def run_dtnv_tnv(phantom: str, algorithm: str, config_path: Path, dry_run: bool = False):
    """
    Run dTNV or TNV reconstruction.

    Args:
        phantom: Phantom name
        algorithm: Algorithm name (dtnv or tnv)
        config_path: Path to composed config file
        dry_run: If True, only print what would be done
    """
    output_dir = REPO_ROOT / "output" / phantom / algorithm

    if dry_run:
        logging.info(f"[DRY RUN] Would run {algorithm.upper()} for {phantom}")
        logging.info(f"[DRY RUN] Output directory: {output_dir}")
        logging.info(f"[DRY RUN] Config: {config_path}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run dTNV script
    script_path = SCRIPT_DIR / "run_dtnv_1bpos.py"
    cmd = ["python", str(script_path), "--config", str(config_path)]

    run_command(cmd, f"{algorithm.upper()} reconstruction for {phantom}")


def run_hkem(phantom: str, config_spect_path: Path, config_pet_path: Path, dry_run: bool = False):
    """
    Run HKEM two-stage reconstruction workflow.

    Stage 1: SPECT reconstruction with CT guidance
    Stage 2: Resample SPECT to PET space
    Stage 3: PET reconstruction with SPECT emission guidance

    Args:
        phantom: Phantom name
        config_spect_path: Path to SPECT config
        config_pet_path: Path to PET config
        dry_run: If True, only print what would be done
    """
    base_output_dir = REPO_ROOT / "output" / phantom / "hkem"
    spect_output_dir = base_output_dir / "spect"
    resample_output_dir = base_output_dir / "spect_resampled"
    pet_output_dir = base_output_dir / "pet"

    if dry_run:
        logging.info(f"[DRY RUN] Would run HKEM workflow for {phantom}")
        logging.info(f"[DRY RUN] Stage 1: SPECT -> {spect_output_dir}")
        logging.info(f"[DRY RUN] Stage 2: Resample -> {resample_output_dir}")
        logging.info(f"[DRY RUN] Stage 3: PET -> {pet_output_dir}")
        return

    # Create output directories
    spect_output_dir.mkdir(parents=True, exist_ok=True)
    resample_output_dir.mkdir(parents=True, exist_ok=True)
    pet_output_dir.mkdir(parents=True, exist_ok=True)

    hkem_script = SCRIPT_DIR / "run_hkem_1bpos.py"
    resample_script = SCRIPT_DIR / "resample_spect_to_pet.py"

    # Stage 1: SPECT reconstruction
    logging.info("=" * 60)
    logging.info("HKEM Stage 1: SPECT reconstruction with CT guidance")
    logging.info("=" * 60)

    cmd = ["python", str(hkem_script), "--config", str(config_spect_path)]
    run_command(cmd, "SPECT reconstruction")

    # Stage 2: Resample SPECT to PET space
    logging.info("=" * 60)
    logging.info("HKEM Stage 2: Resampling SPECT to PET space")
    logging.info("=" * 60)

    # Create resample config
    resample_config = {
        "spect_reconstruction": str(spect_output_dir / "reconstruction_x.hv"),
        "output_path": str(resample_output_dir),
        "phantom": phantom,
    }

    resample_config_path = base_output_dir / "resample_config.yaml"
    save_yaml(resample_config, resample_config_path)

    if resample_script.exists():
        cmd = ["python", str(resample_script), "--config", str(resample_config_path)]
        run_command(cmd, "SPECT resampling")
    else:
        logging.warning(f"Resample script not found: {resample_script}")
        logging.warning("Skipping resampling stage")

    # Stage 3: PET reconstruction with SPECT emission guidance
    logging.info("=" * 60)
    logging.info("HKEM Stage 3: PET reconstruction with SPECT emission guidance")
    logging.info("=" * 60)

    cmd = ["python", str(hkem_script), "--config", str(config_pet_path)]
    run_command(cmd, "PET reconstruction")


def run_experiment(phantom: str, algorithm: str, overrides: List[str] = None, dry_run: bool = False):
    """
    Run a single experiment for phantom + algorithm combination.

    Args:
        phantom: Phantom name
        algorithm: Algorithm name
        overrides: List of override strings (key=value)
        dry_run: If True, only print what would be done
    """
    logging.info("=" * 70)
    logging.info(f"Experiment: {phantom.upper()} + {algorithm.upper()}")
    logging.info("=" * 70)

    # Parse overrides
    extra_overrides = {}
    if overrides:
        for override_str in overrides:
            extra_overrides.update(parse_override(override_str))

    # Handle HKEM (two-stage workflow)
    if algorithm == "hkem":
        # Compose SPECT config
        spect_config = compose_config(phantom, "hkem_spect", extra_overrides)
        spect_config["output_path"] = str(REPO_ROOT / "output" / phantom / "hkem" / "spect")

        # Compose PET config
        pet_config = compose_config(phantom, "hkem_pet", extra_overrides)
        pet_config["output_path"] = str(REPO_ROOT / "output" / phantom / "hkem" / "pet")

        # Save temporary configs
        temp_dir = REPO_ROOT / "tmp" / "experiment_configs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        spect_config_path = temp_dir / f"{phantom}_hkem_spect.yaml"
        pet_config_path = temp_dir / f"{phantom}_hkem_pet.yaml"

        if not dry_run:
            save_yaml(spect_config, spect_config_path)
            save_yaml(pet_config, pet_config_path)

        if dry_run:
            logging.info(f"[DRY RUN] SPECT config preview:")
            print(yaml.dump(spect_config, default_flow_style=False, sort_keys=False))
            logging.info(f"[DRY RUN] PET config preview:")
            print(yaml.dump(pet_config, default_flow_style=False, sort_keys=False))

        # Run HKEM workflow
        run_hkem(phantom, spect_config_path, pet_config_path, dry_run)

    else:
        # dTNV or TNV
        config = compose_config(phantom, algorithm, extra_overrides)
        config["output_path"] = str(REPO_ROOT / "output" / phantom / algorithm)

        # Save temporary config
        temp_dir = REPO_ROOT / "tmp" / "experiment_configs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        config_path = temp_dir / f"{phantom}_{algorithm}.yaml"

        if not dry_run:
            save_yaml(config, config_path)

        if dry_run:
            logging.info(f"[DRY RUN] Config preview:")
            print(yaml.dump(config, default_flow_style=False, sort_keys=False))

        # Run dTNV/TNV
        run_dtnv_tnv(phantom, algorithm, config_path, dry_run)

    logging.info("=" * 70)
    logging.info(f"Experiment completed: {phantom.upper()} + {algorithm.upper()}")
    logging.info("=" * 70)
    logging.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run phantom reconstruction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python run_phantom_experiments.py --phantom manc --algorithm hkem

  # Run all algorithms for one phantom
  python run_phantom_experiments.py --phantom manc --all-algorithms

  # Run with parameter override
  python run_phantom_experiments.py --phantom anthro --algorithm dtnv --override alpha=0.05

  # Dry run to preview configs
  python run_phantom_experiments.py --phantom manc --algorithm tnv --dry-run

  # Batch mode - run all 9 combinations
  python run_phantom_experiments.py --batch-all
        """
    )

    parser.add_argument("--phantom", choices=PHANTOMS, help="Phantom dataset to use")
    parser.add_argument("--algorithm", choices=ALGORITHMS, help="Algorithm to run")
    parser.add_argument("--all-algorithms", action="store_true", help="Run all algorithms for the specified phantom")
    parser.add_argument("--batch-all", action="store_true", help="Run all phantom/algorithm combinations (9 experiments)")
    parser.add_argument("--override", action="append", help="Override config parameter (format: key=value)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without running")

    args = parser.parse_args()

    setup_logging()

    # Validate arguments
    if args.batch_all:
        # Run all combinations
        for phantom in PHANTOMS:
            for algorithm in ALGORITHMS:
                run_experiment(phantom, algorithm, args.override, args.dry_run)

    elif args.all_algorithms:
        if not args.phantom:
            parser.error("--all-algorithms requires --phantom")

        for algorithm in ALGORITHMS:
            run_experiment(args.phantom, algorithm, args.override, args.dry_run)

    else:
        if not args.phantom or not args.algorithm:
            parser.error("Must specify either (--phantom and --algorithm), --all-algorithms, or --batch-all")

        run_experiment(args.phantom, args.algorithm, args.override, args.dry_run)

    logging.info("All experiments completed successfully!")


if __name__ == "__main__":
    main()
