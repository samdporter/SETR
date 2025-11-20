#!/usr/bin/env python3
"""SETR DTV reconstruction for multiple bed position PET - Directional Total Variation with guided prior.

Note: This handles PET reconstruction across multiple bed positions.
SPECT is always single bed position, so use run_dtv_1bpos_spect.py for SPECT reconstruction.
"""

import argparse
import cProfile
import logging
import os
import pstats

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction, SVRGFunction
from cil.optimisation.operators import CompositionOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.priors import TotalVariation
from setr.scripts.common import (
    apply_combine_sensitivities,
    configure_logging,
    get_sensitivity_from_subset_objs,
    get_shift_operators,
    init_run_env,
    save_results,
)
from setr.scripts.dtnv_common import (
    compute_kappa_squared_image_from_partitioned_objective,
    get_callbacks,
)
from setr.utils import get_pet_am, get_pet_data_multiple_bed_pos
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_array, get_filters


def prepare_data(args):
    """
    Prepare the guidance image, PET data, and initial estimates for multiple bed positions.

    Returns:
        guidance_image: PET guidance image (umap or emission guidance, get_array).
        pet_data: Dictionary containing multi-bed PET data.
    """
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    # PET guidance: use emission guidance if available, otherwise use PET umap
    if getattr(args, "use_emission_guidance", False) and hasattr(args, "emission_guidance_path"):
        logging.info("Using emission guidance for PET reconstruction")
        from sirf.STIR import ImageData

        guidance_image = ImageData(args.emission_guidance_path)
    else:
        logging.info("Using umap guidance for PET reconstruction")
        # Use the combined attenuation map from multi-bed PET data
        guidance_image = pet_data["attenuation"]

    # Normalize guidance
    guidance_image += (-guidance_image).max()
    guidance_image /= guidance_image.max()

    # Apply filters to initial image
    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])
    pet_data["initial_image"].write(os.path.join(args.output_path, "initial_image_pet.hv"))

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        args.delta = pet_data["initial_image"].max() / 1e4 * args.gamma_pet

    # Check for nans
    if np.isnan(get_array(guidance_image)).any():
        logging.warning("PET guidance image contains NaNs")
    if np.isnan(pet_data["initial_image"].as_array()).any():
        logging.warning("PET initial image contains NaNs")

    return guidance_image, pet_data


def get_data_fidelity(
    args, pet_data, get_pet_am, num_subsets, uncombine_op, unshift_ops, choose_ops
):
    """
    Set up data fidelity (objective) functions for multi-bed PET.

    Returns:
        obj_funs: List of PET objective functions (wrapped with bed operators).
        s_inv: Combined sensitivity image ^ -1.
        kappa: Combined kappa-squared weighting image.
    """
    # partition PET by bed
    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=num_subsets,
            mode="staggered",
            create_acq_model=get_pet_am,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # keep raw copies for κ before operator wrapping
    pet_dfs_raw = [list(df_list) for df_list in pet_dfs]

    # set_up subset objs on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # κ² build *before* op wrapping (per-bed PET κ² in bed coords)
    pet_kappa_bed_sq = []
    for df_list, suffix in zip(pet_dfs_raw, pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        pet_kappa_bed_sq.append(
            compute_kappa_squared_image_from_partitioned_objective(df_list, tmpl)
        )

    # add across beds (Fisher additivity)
    kappa = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(pet_kappa_bed_sq[i]) for i, unshift_op in enumerate(unshift_ops)]
        )
    )

    logging.info(f"PET κ² images computed and uncombined with shape {kappa.shape}.")

    # PET sensitivity computation
    pet_sens = [get_sensitivity_from_subset_objs(df) for df in pet_dfs]
    apply_combine_sensitivities(pet_data, pet_sens)
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )
    s_inv = pet_sens_combined.clone()
    pet_sens_array = get_array(pet_sens_combined)
    s_inv.fill(np.reciprocal(pet_sens_array, where=pet_sens_array != 0))

    cyl, gauss = get_filters()
    cyl.apply(s_inv)
    gauss.apply(kappa)

    # save images
    s_inv.write(os.path.join(args.output_path, "s_inv_pet.hv"))
    logging.info(f"Writing s_inv_pet with max {s_inv.max()}")

    # --- now wrap PET objectives with uncombine/choose/unshift ---
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(unshift_ops[i], choose_ops[i], uncombine_op),
            )

    # flatten beds
    obj_funs = [df for bed in pet_dfs for df in bed]

    return obj_funs, s_inv, kappa


def main(args) -> None:
    """Main PET DTV 2bpos reconstruction pipeline."""
    configure_logging()

    # Initialize run environment
    msg = init_run_env(args)

    # Prepare data (may update args such as delta)
    guidance_image, pet_data = prepare_data(args)
    save_args(args, "args.csv")

    # Set up operators for multiple bed positions
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=args.pet_gauss_fwhm,
        )

    # Set up data fidelity
    obj_funs, s_inv, kappa = get_data_fidelity(
        args,
        pet_data,
        get_pet_am_with_res,
        args.num_subsets,
        uncombine_op,
        unshift_ops,
        choose_ops,
    )

    # write κ² image
    if kappa is not None:
        kappa.write(os.path.join(args.output_path, "kappa_sq_pet.hv"))
        logging.info(f"Writing kappa with max {kappa.max()}")

    # Set up the DTV prior
    if args.no_prior:
        prior = None
    else:
        weight = args.gamma_pet
        if kappa is not None:
            logging.info("Using kappa-weighted DTV prior")

        dtv_prior = TotalVariation(
            geometry=pet_data["initial_image"],
            weight=weight,
            delta=args.delta,
            anatomical=guidance_image,
            stencil=getattr(args, "pet_stencil", "6"),
            both_directions=getattr(args, "pet_both_directions", False),
        )

        # Attach Hessian for preconditioner
        dtv_prior.inv_hessian_diag = lambda x, out=None, epsilon=1e-9: dtv_prior.inv_hessian_diag(
            x, out, epsilon
        )
        prior = -dtv_prior

    ui = getattr(args, "update_interval", None)
    update_interval = len(obj_funs) if ui is None else ui

    # Set up preconditioners
    bsrem_precond = BSREMPreconditioner(s_inv, 1, np.inf, epsilon=0, smooth=True)

    if prior is not None:
        # CRITICAL: Cap inverse Hessian to prevent huge preconditioner at FOV edges
        # Cap at the scale of the BSREM preconditioner to keep both on same scale
        max_precond_value = 10.0 * pet_data["initial_image"].max() * s_inv.max()
        prior_precond = ImageFunctionPreconditioner(
            dtv_prior.inv_hessian_diag,
            1.0,
            freeze_iter=np.inf,
            epsilon=0,
            max_value=max_precond_value,
        )
        precond = LehmerMeanPreconditioner(
            [bsrem_precond, prior_precond],
            update_interval=1,
            freeze_iter=len(obj_funs) * 10,
            epsilon=0,
        )
    else:
        precond = bsrem_precond

    # Calculate probabilities for 2 bed positions
    probs = [1.0 / update_interval] * len(obj_funs)
    assert abs(sum(probs) - len(obj_funs) / update_interval) < 1e-10, (
        f"Probabilities incorrect: {sum(probs)}"
    )

    f_obj = SVRGFunction(
        obj_funs,
        sampler=Sampler.random_with_replacement(len(obj_funs), prob=probs),
        snapshot_update_interval=update_interval * 2,
        store_gradients=True,
    )

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks
    callbacks = get_callbacks(args, update_interval)

    # Run algorithm
    subiterations = args.num_epochs * len(obj_funs)
    algo = ISTA(
        initial=pet_data["initial_image"],
        f=-SumFunction(f_obj, prior) if prior else -f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size,
        update_objective_interval=update_interval,
    )
    algo.run(subiterations, verbose=1, callbacks=callbacks)

    # Save results
    save_results(algo, args)
    logging.info("PET DTV 2bpos reconstruction complete")


if __name__ == "__main__":
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = argparse.Namespace(**config)

    if getattr(args, "profile", False):
        logging.info("Profiling is enabled. This may slow down the execution.")
        profiler = cProfile.Profile()
        profiler.enable()
        main(args)
        profiler.disable()
        profiler.dump_stats(f"{args.output_path}/profile_data.prof")

        output_file = os.path.join(args.output_path, "profiling_results.txt")
        with open(output_file, "w") as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats("cumulative").print_stats()
        logging.info(f"Profiling results saved to {output_file}")
    else:
        logging.info("Profiling is disabled.")
        main(args)
    logging.info("Execution completed.")
