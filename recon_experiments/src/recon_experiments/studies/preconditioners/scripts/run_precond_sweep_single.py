#!/usr/bin/env python3
"""
Wrapper script to run DTNV reconstructions for the preconditioner sweep.

Calls the appropriate run_dtnv_*bpos.py script with config overrides and writes a
result.csv summary compatible with sweep analysis.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path


def _resolve_base_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src" / "recon_experiments").is_dir():
            return parent
    raise RuntimeError("Could not locate recon_experiments base directory.")


def _infer_bpos(config_path: Path) -> int:
    name = config_path.name.lower()
    if "1bpos" in name:
        return 1
    if "2bpos" in name:
        return 2
    raise ValueError(f"Could not infer bed positions from config name: {config_path.name}")


def _as_bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _split_override(override: str) -> tuple[str, str] | None:
    if "=" not in override:
        return None
    key, value = override.split("=", 1)
    return key.strip(), value.strip()


def _consume_bool_override(overrides: list[str], key: str, default: bool = False) -> tuple[bool, list[str]]:
    value = default
    remaining: list[str] = []
    for override in overrides:
        parsed = _split_override(override)
        if parsed is not None and parsed[0] == key:
            value = _as_bool_text(parsed[1])
        else:
            remaining.append(override)
    return value, remaining


def _has_override(overrides: list[str], key: str) -> bool:
    for override in overrides:
        parsed = _split_override(override)
        if parsed is not None and parsed[0] == key:
            return True
    return False


def _subset_selection_config(base_dir: Path, bpos: int) -> Path:
    config_name = "base_config_anthro.yaml" if bpos == 1 else "base_config_2bpos.yaml"
    return (
        base_dir
        / "src"
        / "recon_experiments"
        / "studies"
        / "subset_selection"
        / "configs"
        / config_name
    )


def _read_final_objective(obj_path: Path) -> float:
    if not obj_path.exists():
        return float("nan")
    last_val = None
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last_val = line
    if last_val is None:
        return float("nan")
    parts = [p for p in last_val.split(",") if p]
    try:
        return float(parts[-1])
    except ValueError:
        return float("nan")


def _build_override_list(args: argparse.Namespace) -> list[str]:
    overrides = list(args.override or [])
    overrides.extend(
        [
            f"precond_type={args.precond_type}",
            f"alpha={args.alpha}",
            f"beta={args.alpha}",
            f"initial_step_size={args.step_size}",
            f"num_epochs={args.epochs}",
            # Sweep runs should use full preconditioning (no damping).
            "precond_safety_scale=1.0",
            f"output_path={args.output}",
        ]
    )
    if args.precond_combine:
        overrides.append(f"precond_combine={args.precond_combine}")
    if args.lehmer_p is not None:
        overrides.append(f"lehmer_p={args.lehmer_p}")
    if args.block_scalar_reduction:
        overrides.append(f"block_scalar_reduction={args.block_scalar_reduction}")
    return overrides


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DTNV sweep job with overrides.")
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--precond-type", required=True, help="Preconditioner method name")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha/beta value")
    parser.add_argument("--step-size", type=float, required=True, help="Initial step size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument(
        "--precond-combine",
        default="majoriser",
        help="Combine mode: none|lehmer|harmonic|majoriser|magez",
    )
    parser.add_argument("--lehmer-p", type=float, default=None, help="Lehmer mean p value")
    parser.add_argument(
        "--block-scalar-reduction",
        choices=["mean", "geometric", "diag"],
        default="diag",
        help="Scalar reduction for block blend (default: diag)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra config overrides (key=value). Can be repeated.",
    )
    parser.add_argument("--bpos", type=int, choices=[1, 2], default=None, help="Bed positions")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid reporting an old status when reusing an existing output directory.
    result_path = output_dir / "result.csv"
    if result_path.exists():
        result_path.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    base_dir = _resolve_base_dir()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    bpos = args.bpos if args.bpos is not None else _infer_bpos(config_path)
    runner_name = f"run_dtnv_{bpos}bpos.py"
    runner_path = base_dir / "src" / "recon_experiments" / "runners" / "scripts" / runner_name
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    overrides = _build_override_list(args)
    emulate_subset_selection, overrides = _consume_bool_override(
        overrides,
        "emulate_subset_selection",
        default=False,
    )
    if emulate_subset_selection:
        runner_path = (
            base_dir
            / "src"
            / "recon_experiments"
            / "studies"
            / "subset_selection"
            / "scripts"
            / "run_subset_selection.py"
        )
        config_path = _subset_selection_config(base_dir, bpos)
        if not _has_override(overrides, "subset_mode"):
            overrides.append("subset_mode=separate")
        if not _has_override(overrides, "prior_mode"):
            overrides.append("prior_mode=always")
        logging.info("Emulating subset-selection path for %dbpos", bpos)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    cmd = [sys.executable, str(runner_path), "--config", str(config_path)]
    for ov in overrides:
        cmd.extend(["--override", ov])

    logging.info("Running: %s", " ".join(cmd))
    if args.dry_run:
        return 0

    start = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - start

    final_objective = _read_final_objective(output_dir / "objective.csv")
    status = "success" if proc.returncode == 0 else "failed"

    result = {
        "precond_type": args.precond_type,
        "precond_combine": args.precond_combine or "",
        "alpha": args.alpha,
        "beta": args.alpha,
        "step_size": args.step_size,
        "num_epochs": args.epochs,
        "final_objective": final_objective,
        "run_time": elapsed,
        "status": status,
        "error": None if status == "success" else f"returncode={proc.returncode}",
    }

    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    if status != "success":
        logging.error("Run failed with return code %s", proc.returncode)
        return proc.returncode

    logging.info("Completed successfully. result.csv written to %s", result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
