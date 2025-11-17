#!/usr/bin/env python3
"""
Single-run preconditioner test script for cluster execution.

This script runs ONE preconditioner test with specific parameters.
Unlike test_preconditioners.py which runs a batch of tests, this sets up
everything fresh for each individual run.

Usage:
    python test_preconditioner_single.py \\
        --config configs/config_2bpos.yaml \\
        --output results/precond_test \\
        --precond-type vtv_mm_jensen \\
        --alpha 500.0 \\
        --step-size 0.1 \\
        --epochs 50
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction, SVRGFunction
from cil.optimisation.operators import (
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.scripts.common import (
    attach_prior_hessian,
    get_resampling_operators,
    get_sensitivity_from_subset_objs,
    get_shift_operators,
    init_run_env,
)
from setr.scripts.dtnv_common import (
    compute_kappa_squared_image_from_partitioned_objective,
    get_block_objective,
    get_prior,
    get_probabilities,
    get_s_inv_from_subset_objs,
    dynamic_range_scale_sirf,
    normalise_kappa_squares,
)
from setr.utils import (
    get_pet_am,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
)
from setr.utils.io import load_config
from setr.utils.sirf import get_array, get_filters

# Monkey-patch ISTA
ISTA.update = ista_update_step


def setup_logging(output_dir):
    """Configure logging."""
    log_file = output_dir / "preconditioner_test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )


def setup_reconstruction(args, output_dir):
    """
    Set up all reconstruction components.

    Returns a dict with everything needed to run the algorithm.
    """
    logging.info("=" * 60)
    logging.info("SETTING UP RECONSTRUCTION")
    logging.info("=" * 60)

    # Prepare data
    logging.info("Loading PET data...")
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    umap = pet_data["attenuation"]
    umap += (-umap).max()
    umap /= umap.max()

    logging.info("Loading SPECT data...")
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    # Set up operators
    logging.info("Setting up shift and resampling operators...")
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)
    spect2pet = get_resampling_operators(pet_data, spect_data)

    # Set up acquisition models
    def get_pet_am_with_res():
        return get_pet_am(not args.no_gpu, gauss_fwhm=getattr(args, "pet_gauss_fwhm", None))

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.keep_all_views_in_cache,
            gauss_fwhm=getattr(args, "spect_gauss_fwhm", None),
            attenuation=True,
        )

    # Partition data
    logging.info("Partitioning PET data...")
    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=args.num_subsets[0],
            mode="staggered",
            create_acq_model=get_pet_am_with_res,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    pet_dfs_raw = [list(df_list) for df_list in pet_dfs]

    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    logging.info("Partitioning SPECT data...")
    spect_dfs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=args.num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am_with_res,
    )[2]
    for obj_fun in spect_dfs:
        obj_fun.set_up(spect_data["initial_image"])

    # Compute kappas
    logging.info("Computing kappa images...")
    pet_kappa_bed_sq = []
    for df_list, suffix in zip(pet_dfs_raw, pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        pet_kappa_bed_sq.append(
            compute_kappa_squared_image_from_partitioned_objective(df_list, tmpl)
        )

    pet_kappa_sq = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(pet_kappa_bed_sq[i]) for i, unshift_op in enumerate(unshift_ops)]
        )
    )

    spect_kappa_sq = compute_kappa_squared_image_from_partitioned_objective(
        spect_dfs, spect_data["initial_image"]
    )

    # Get sensitivities
    logging.info("Computing sensitivities...")
    pet_sens = [get_sensitivity_from_subset_objs(df) for df in pet_dfs]
    spect_s_inv = get_s_inv_from_subset_objs(spect_dfs, spect_data["initial_image"])

    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )
    pet_s_inv = pet_sens_combined.clone()
    pet_sens_array = get_array(pet_sens_combined)
    pet_s_inv.fill(np.reciprocal(pet_sens_array, where=pet_sens_array != 0))
    cyl.apply(pet_s_inv)

    s_inv = EnhancedBlockDataContainer(pet_s_inv, spect_s_inv)

    # Wrap PET objectives
    logging.info("Creating block objectives...")
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(unshift_ops[i], choose_ops[i], uncombine_op),
            )

    pet_combined_dfs = [df for bed in pet_dfs for df in bed]

    pet_dfs_block = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            df,
            order=0,
        )
        for df in pet_combined_dfs
    ]
    spect_dfs_block = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_dfs
    ]

    all_funs = pet_dfs_block + spect_dfs_block

    # Create block operator
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    kappas = EnhancedBlockDataContainer(pet_kappa_sq, spect_kappa_sq)
    kappas = normalise_kappa_squares(bo.direct(kappas))

    # write kappa images for inspection
    if not os.path.exists(os.path.join(output_dir, "kappas")):
        os.makedirs(os.path.join(output_dir, "kappas"))
    kappas.containers[0].write(os.path.join(output_dir, "kappas", "pet_kappa_sq.hv"))
    kappas.containers[1].write(os.path.join(output_dir, "kappas", "spect_kappa_sq.hv"))

    combined = bo.direct(initial_estimates)

    # Compute per-modality scaling (inverse dynamic ranges)
    pet_scale, spect_scale = dynamic_range_scale_sirf(
        combined[0],
        combined[1],
    )

    logging.info(
        "Dynamic range scales -> PET: %.6g, SPECT: %.6g", pet_scale, spect_scale
    )

    return {
        "initial_estimates": initial_estimates,
        "all_funs": all_funs,
        "s_inv": s_inv,
        "umap": umap,
        "bo": bo,
        "pet_scale": pet_scale,
        "spect_scale": spect_scale,
        "kappas": kappas,
        "combined": combined,
        "num_subsets": args.num_subsets,
    }


def create_prior_for_test(
    alpha: float,
    setup_data: dict,
    args: argparse.Namespace,
    hessian_type: str,
):
    """
    Create VTV prior with specified hessian type.

    Args:
        alpha: Penalty strength (beta set equal), BEFORE gradient scaling
        setup_data: Dict from setup_reconstruction() including gradient scale
        args: Base args namespace
        hessian_type: canonical name ('mm_jensen', 'svd_principal_alpha',
            'frobenius_surrogate_pd', 'vector_tv_per_modality') or legacy alias

    Returns:
        List of prior functions
    """
    test_args = argparse.Namespace(**vars(args))

    # Apply gradient energy scaling to alpha/beta
    test_args.alpha = alpha * setup_data["pet_scale"]
    test_args.beta = alpha * setup_data["spect_scale"]

    if test_args.delta is None:
        test_args.delta = max(
            setup_data["initial_estimates"][0].max() / 1e4,
            setup_data["initial_estimates"][1].max() / 1e4,
        ) * max(
            alpha * setup_data["pet_scale"],
            alpha * setup_data["spect_scale"],
        )

    # Set hessian type
    test_args.hessian_type = hessian_type

    return get_prior(
        test_args,
        setup_data["umap"],
        setup_data["combined"],
        setup_data["bo"],
        setup_data["kappas"],
    )


def create_preconditioner(
    precond_type: str,
    priors_list: list,
    setup_data: dict,
):
    """
    Create preconditioner.

    Args:
        precond_type: 'bsrem', 'vtv_mm_jensen', 'vtv_svd_principal_alpha', etc.
        priors_list: List of priors with hessian methods attached
        setup_data: Dict from setup_reconstruction()

    Returns:
        Preconditioner object
    """
    s_inv = setup_data["s_inv"]

    # set epsilon as 1000th of smallest image values
    epsilon = 0.001 * min(
        [
            setup_data["initial_estimates"][0].as_array().min(),
            setup_data["initial_estimates"][1].as_array().min(),
        ]
    )
    bsrem_precond = BSREMPreconditioner(
        s_inv,
        1,
        np.inf,
        epsilon=epsilon,
        smooth=True,
    )

    if precond_type == "bsrem":
        # Just BSREM, no prior preconditioning
        return bsrem_precond

    # For VTV preconditioners, add prior preconditioners
    # CRITICAL: Cap inverse Hessian to prevent huge preconditioner at FOV edges
    # Cap at the scale of the BSREM preconditioner to keep both on same scale
    max_precond_value = 10.0 * max(
        con.max() * s_inv_con.max()
        for con, s_inv_con in zip(
            setup_data["initial_estimates"].containers, s_inv.containers
        )
    )
    prior_precond = [
        ImageFunctionPreconditioner(
            p.inv_hessian_diag,  # Uses the wrapped version from attach_prior_hessian
            1,
            freeze_iter=np.inf,
            epsilon=0,
            max_value=max_precond_value,
        )
        for p in priors_list
    ]

    # Combine with Lehmer mean
    return LehmerMeanPreconditioner(
        [bsrem_precond, *prior_precond],
        update_interval=1,
        freeze_iter=len(setup_data["all_funs"]) * 10,
        epsilon=0,
    )


def run_test(
    precond_type: str,
    hessian_type: str,
    alpha: float,
    step_size: float,
    setup_data: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    """Run a single reconstruction test."""

    logging.info("=" * 60)
    logging.info("PRECONDITIONER TEST")
    logging.info(f"  Type: {precond_type} ({hessian_type})")
    logging.info(f"  Alpha: {alpha}")
    logging.info(f"  Step size: {step_size}")
    logging.info("=" * 60)

    initial = setup_data["initial_estimates"].copy()

    # Create priors for objective (always use fast for consistency)
    logging.info("Creating priors for objective...")
    priors_for_objective = create_prior_for_test(alpha, setup_data, args, hessian_type="fast")

    # Create priors for preconditioner with specified hessian type
    if precond_type != "bsrem":
        logging.info(f"Creating priors for preconditioner (hessian={hessian_type})...")
        priors_for_precond = create_prior_for_test(alpha, setup_data, args, hessian_type)
        # CRITICAL: Attach hessian methods that apply bo.direct/adjoint transformations
        for prior in priors_for_precond:
            attach_prior_hessian(prior)
    else:
        priors_for_precond = priors_for_objective  # Not used, but keep consistent

    logging.info("Creating preconditioner...")
    start_time = time.time()
    precond = create_preconditioner(precond_type, priors_for_precond, setup_data)
    precond_setup_time = time.time() - start_time
    logging.info(f"Preconditioner setup time: {precond_setup_time:.2f}s")

    # Objective is ALWAYS the same: data fidelity + prior
    prior = SumFunction(*priors_for_objective)

    probs = get_probabilities(args, setup_data["num_subsets"], len(setup_data["all_funs"]), bpos=2)

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

    callbacks = [
        PrintObjectiveCallback(len(setup_data["all_funs"])),
        SaveObjectiveCallback(str(output_dir / "objective"), len(setup_data["all_funs"])),
        SaveImageCallback(str(output_dir / "image"), len(setup_data["all_funs"])),
        SavePreconditionerCallback(str(output_dir / "preconditioner"), len(setup_data["all_funs"])),
    ]

    algo = ISTA(
        initial=initial,
        f=objective,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size_rule,
        update_objective_interval=len(setup_data["all_funs"]),
    )

    subiterations = args.num_epochs * len(setup_data["all_funs"])

    try:
        logging.info(f"Starting reconstruction ({subiterations} subiterations)...")
        start_time = time.time()
        algo.run(subiterations, verbose=1, callbacks=callbacks)
        run_time = time.time() - start_time

        logging.info(f"Reconstruction completed in {run_time:.1f}s")

        # Read final objective
        obj_file = output_dir / "objective.csv"
        if obj_file.exists():
            import pandas as pd

            df = pd.read_csv(obj_file)
            final_obj = float(df.iloc[-1].values[0]) if len(df) > 0 else np.nan
        else:
            final_obj = np.nan

        return {
            "precond_type": precond_type,
            "hessian_type": hessian_type,
            "alpha": alpha,
            "beta": alpha,
            "step_size": step_size,
            "final_objective": final_obj,
            "run_time": run_time,
            "precond_setup_time": precond_setup_time,
            "status": "success",
            "error": None,
        }

    except Exception as e:
        logging.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "precond_type": precond_type,
            "hessian_type": hessian_type,
            "alpha": alpha,
            "beta": alpha,
            "step_size": step_size,
            "final_objective": np.nan,
            "run_time": np.nan,
            "precond_setup_time": np.nan,
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Test single VTV preconditioner configuration")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--precond-type",
        type=str,
        required=True,
        choices=[
            "bsrem",
            # canonical names (preferred)
            "vtv_svd_principal_alpha",
            "vtv_mm_jensen",
            "vtv_frobenius_surrogate_pd",
            "vtv_vector_tv_per_modality",
            # legacy aliases (supported for backwards compatibility)
            "vtv_slow",
            "vtv_fast",
            "vtv_fastest_positive",
            "vtv_fastest_exact",
        ],
        help="Preconditioner type",
    )
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value")
    parser.add_argument("--step-size", type=float, required=True, help="Initial step size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")

    test_args = parser.parse_args()

    config = load_config(test_args.config)
    args = argparse.Namespace(**config)

    # Override epochs
    args.num_epochs = test_args.epochs

    output_dir = Path(test_args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info("Starting preconditioner test")
    logging.info(f"  Config: {test_args.config}")
    logging.info(f"  Preconditioner: {test_args.precond_type}")
    logging.info(f"  Alpha: {test_args.alpha}")
    logging.info(f"  Step size: {test_args.step_size}")
    logging.info(f"  Epochs: {test_args.epochs}")

    msg = init_run_env(args)

    # Do expensive setup once
    setup_data = setup_reconstruction(args, output_dir)

    # Map precond_type to hessian_type
    hessian_map = {
        # legacy → canonical
        "bsrem": "mm_jensen",  # placeholder; not used for bsrem
        "vtv_slow": "svd_principal_alpha",
        "vtv_fast": "mm_jensen",
        "vtv_fastest_positive": "frobenius_surrogate_pd",
        "vtv_fastest_exact": "vector_tv_per_modality",
        # canonical
        "vtv_svd_principal_alpha": "svd_principal_alpha",
        "vtv_mm_jensen": "mm_jensen",
        "vtv_frobenius_surrogate_pd": "frobenius_surrogate_pd",
        "vtv_vector_tv_per_modality": "vector_tv_per_modality",
    }
    hessian_type = hessian_map[test_args.precond_type]

    result = run_test(
        precond_type=test_args.precond_type,
        hessian_type=hessian_type,
        alpha=test_args.alpha,
        step_size=test_args.step_size,
        setup_data=setup_data,
        args=args,
        output_dir=output_dir,
    )

    # Save result summary
    import pandas as pd

    df = pd.DataFrame([result])
    df.to_csv(output_dir / "result.csv", index=False)

    logging.info("=" * 60)
    logging.info("TEST COMPLETE")
    logging.info(f"  Status: {result['status']}")
    logging.info(f"  Final objective: {result['final_objective']:.6f}")
    logging.info(f"  Run time: {result['run_time']:.1f}s")
    logging.info("=" * 60)

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
