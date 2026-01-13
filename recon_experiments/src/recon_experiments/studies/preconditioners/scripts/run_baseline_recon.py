#!/usr/bin/env python3
"""
Run baseline/reference reconstruction for preconditioner comparison.

This script runs a long reconstruction with the most accurate (but slowest) 
preconditioner to establish a near-optimal solution. Other preconditioner tests
can then be compared against this baseline to measure convergence speed.

Usage:
    python run_baseline_recon.py --config config_1bpos_anthro_long.yaml \
                                   --alpha 0.005 \
                                   --epochs 200 \
                                   --output baseline_alpha_0.005
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path to import test_preconditioner_single
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.test_preconditioner_single import (
    setup_logging,
    setup_reconstruction,
    create_prior_for_test,
    create_preconditioner,
    init_run_env,
)
from recon_core.utils.io import load_config
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import SumFunction, SVRGFunction
from cil.optimisation.utilities import Sampler
from recon_core.cil_extensions.functions import BlockIndicatorBox
from recon_core.cil_extensions.utilities import LinearDecayStepSizeRule
from recon_core.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from recon_experiments.runners.common import attach_prior_hessian
from recon_experiments.runners.dtnv_common import get_probabilities
import time


def save_baseline_metrics(output_dir: Path, result: dict, setup_data: dict):
    """
    Save metrics for baseline reconstruction that other runs can compare against.
    
    Saves:
    - Final objective value
    - Final images (PET and SPECT)
    - Convergence criteria info
    - Runtime statistics
    """
    metrics = {
        'final_objective': result['final_objective'],
        'total_runtime': result['run_time'],
        'num_epochs': result.get('num_epochs', 'unknown'),
        'alpha': result['alpha'],
        'precond_type': result['precond_type'],
        'status': result['status'],
    }
    
    # Save metrics as JSON for easy loading
    import json
    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Saved baseline metrics to {output_dir / 'baseline_metrics.json'}")
    
    # Also save as CSV for compatibility
    pd.DataFrame([metrics]).to_csv(output_dir / 'baseline_metrics.csv', index=False)


def run_baseline(
    alpha: float,
    setup_data: dict,
    args: argparse.Namespace,
    output_dir: Path,
    num_epochs: int,
    step_size: float = 1.0,
    precond_type: str = "vtv_svd_principal_alpha",
) -> dict:
    """
    Run baseline reconstruction with most accurate preconditioner.
    
    Args:
        alpha: Penalty strength
        setup_data: Dict from setup_reconstruction()
        args: Config namespace
        output_dir: Where to save results
        num_epochs: Number of epochs to run (should be large, e.g., 200-500)
        step_size: Initial step size
        precond_type: Preconditioner type (default: most accurate SVD-based)
    
    Returns:
        Result dictionary with metrics
    """
    logging.info("=" * 60)
    logging.info("BASELINE RECONSTRUCTION")
    logging.info(f"  Alpha: {alpha}")
    logging.info(f"  Epochs: {num_epochs}")
    logging.info(f"  Step size: {step_size}")
    logging.info(f"  Preconditioner: {precond_type}")
    logging.info("=" * 60)

    initial = setup_data["initial_estimates"].copy()

    # Map precond type to hessian type
    hessian_map = {
        "bsrem": "mm_jensen",
        "vtv_svd_principal_alpha": "svd_principal_alpha",
        "vtv_mm_jensen": "mm_jensen",
        "vtv_frobenius_surrogate_pd": "frobenius_surrogate_pd",
        "vtv_vector_tv_per_modality": "vector_tv_per_modality",
    }
    hessian_type = hessian_map[precond_type]

    # Create priors for objective (always use fast for consistency)
    logging.info("Creating priors for objective...")
    priors_for_objective = create_prior_for_test(alpha, setup_data, args, hessian_type="fast")

    # Create priors for preconditioner with specified hessian type
    if precond_type != "bsrem":
        logging.info(f"Creating priors for preconditioner (hessian={hessian_type})...")
        priors_for_precond = create_prior_for_test(alpha, setup_data, args, hessian_type)
        for prior in priors_for_precond:
            attach_prior_hessian(prior)
    else:
        priors_for_precond = priors_for_objective

    logging.info("Creating preconditioner...")
    start_time = time.time()
    precond = create_preconditioner(precond_type, priors_for_precond, setup_data)
    precond_setup_time = time.time() - start_time
    logging.info(f"Preconditioner setup time: {precond_setup_time:.2f}s")

    # Create objective
    prior = SumFunction(*priors_for_objective)

    probs = get_probabilities(args, setup_data["num_subsets"], len(setup_data["all_funs"]), bpos=1)

    f_obj = -SVRGFunction(
        setup_data["all_funs"],
        sampler=Sampler.random_with_replacement(
            len(setup_data["all_funs"]),
            prob=probs,
        ),
        snapshot_update_interval=len(setup_data["all_funs"]) * 2,
        store_gradients=True,
    )

    objective = SumFunction(f_obj, prior)

    step_size_rule = LinearDecayStepSizeRule(
        initial_step_size=step_size,
        decay=args.relaxation_eta,
    )

    # Save more frequently for baseline to monitor convergence
    save_interval = len(setup_data["all_funs"]) * 5  # Every 5 epochs

    callbacks = [
        PrintObjectiveCallback(len(setup_data["all_funs"])),
        SaveObjectiveCallback(str(output_dir / "objective"), len(setup_data["all_funs"])),
        SaveImageCallback(str(output_dir / "image"), save_interval),
        SavePreconditionerCallback(str(output_dir / "preconditioner"), save_interval),
    ]

    algo = ISTA(
        initial=initial,
        f=objective,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size_rule,
        update_objective_interval=len(setup_data["all_funs"]),
    )

    subiterations = num_epochs * len(setup_data["all_funs"])

    try:
        logging.info(f"Starting baseline reconstruction ({subiterations} subiterations)...")
        start_time = time.time()
        algo.run(subiterations, verbose=1, callbacks=callbacks)
        run_time = time.time() - start_time

        logging.info(f"Baseline reconstruction completed in {run_time:.1f}s")

        # Read final objective
        obj_file = output_dir / "objective.csv"
        if obj_file.exists():
            df = pd.read_csv(obj_file)
            final_obj = float(df.iloc[-1].values[0]) if len(df) > 0 else np.nan
        else:
            final_obj = np.nan

        result = {
            "precond_type": precond_type,
            "hessian_type": hessian_type,
            "alpha": alpha,
            "beta": alpha,
            "step_size": step_size,
            "num_epochs": num_epochs,
            "final_objective": final_obj,
            "run_time": run_time,
            "precond_setup_time": precond_setup_time,
            "status": "success",
            "error": None,
        }

        # Save baseline-specific metrics
        save_baseline_metrics(output_dir, result, setup_data)

        return result

    except Exception as e:
        logging.error(f"Baseline reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "precond_type": precond_type,
            "hessian_type": hessian_type,
            "alpha": alpha,
            "beta": alpha,
            "step_size": step_size,
            "num_epochs": num_epochs,
            "final_objective": np.nan,
            "run_time": np.nan,
            "precond_setup_time": np.nan,
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline/reference reconstruction for preconditioner comparison"
    )
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value")
    parser.add_argument(
        "--step-size", 
        type=float, 
        default=1.0, 
        help="Initial step size (default: 1.0)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=200, 
        help="Number of epochs (default: 200, should be large for baseline)"
    )
    parser.add_argument(
        "--precond-type",
        type=str,
        default="vtv_svd_principal_alpha",
        choices=[
            "bsrem",
            "vtv_svd_principal_alpha",
            "vtv_mm_jensen",
            "vtv_frobenius_surrogate_pd",
            "vtv_vector_tv_per_modality",
        ],
        help="Preconditioner type (default: vtv_svd_principal_alpha, the most accurate)",
    )

    test_args = parser.parse_args()

    config = load_config(test_args.config)
    args = argparse.Namespace(**config)

    # Override epochs
    args.num_epochs = test_args.epochs

    output_dir = Path(test_args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info("=" * 60)
    logging.info("BASELINE RECONSTRUCTION SETUP")
    logging.info("=" * 60)
    logging.info(f"  Config: {test_args.config}")
    logging.info(f"  Preconditioner: {test_args.precond_type}")
    logging.info(f"  Alpha: {test_args.alpha}")
    logging.info(f"  Step size: {test_args.step_size}")
    logging.info(f"  Epochs: {test_args.epochs}")
    logging.info(f"  Output: {output_dir}")
    logging.info("=" * 60)

    msg = init_run_env(args)
    logging.info(msg)

    # Do expensive setup once
    setup_data = setup_reconstruction(args, output_dir)

    result = run_baseline(
        alpha=test_args.alpha,
        setup_data=setup_data,
        args=args,
        output_dir=output_dir,
        num_epochs=test_args.epochs,
        step_size=test_args.step_size,
        precond_type=test_args.precond_type,
    )

    # Save result summary
    df = pd.DataFrame([result])
    df.to_csv(output_dir / "result.csv", index=False)

    logging.info("=" * 60)
    logging.info("BASELINE RECONSTRUCTION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"  Status: {result['status']}")
    logging.info(f"  Final objective: {result['final_objective']:.6f}")
    logging.info(f"  Run time: {result['run_time']:.1f}s ({result['run_time']/3600:.2f} hours)")
    logging.info(f"  Output: {output_dir}")
    logging.info("=" * 60)

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
