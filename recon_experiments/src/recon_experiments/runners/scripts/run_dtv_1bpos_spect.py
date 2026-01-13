#!/usr/bin/env python3
"""SETR DTV reconstruction for single bed position SPECT - Directional Total Variation with guided prior."""

import cProfile
import logging
import os
import pstats
from types import SimpleNamespace

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction, SVRGFunction
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from sirf.STIR import ImageData

from recon_core.cil_extensions.functions import BlockIndicatorBox
from recon_core.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from recon_core.cil_extensions.utilities import LinearDecayStepSizeRule
from recon_core.priors import TotalVariation
from recon_experiments.runners.common import (
    configure_logging,
    init_run_env,
    save_results,
)
from recon_experiments.runners.dtnv_common import (
    compute_kappa_squared_image_from_partitioned_objective,
    get_callbacks,
)
from recon_core.utils import get_spect_am, get_spect_data
from recon_core.utils.io import apply_overrides, load_config, parse_cli, save_args
from recon_core.utils.sirf import get_array, get_filters, get_s_inv_from_subset_objs
from recon_core.cil_extensions.operators.blurring import create_gaussian_blur_operator


def prepare_data(args):
    """
    Prepare the guidance image, SPECT data, and initial estimates.

    Returns:
        guidance_image: SPECT guidance image (always umap, get_array).
        spect_data: Dictionary containing SPECT data.
    """
    spect_data = get_spect_data(args.spect_data_path)

    # SPECT guidance: always use SPECT umap
    logging.info("Using umap guidance for SPECT reconstruction")
    guidance_image = ImageData(os.path.join(args.spect_data_path, "mu_map.hv"))

    # Normalize guidance
    guidance_image += (-guidance_image).max()
    guidance_image /= guidance_image.max()

    # Apply filters to initial image
    _, gauss = get_filters()
    gauss.apply(spect_data["initial_image"])
    spect_data["initial_image"].write("initial_image_spect.hv")

    # Check for nans
    for data in [guidance_image, spect_data["initial_image"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("SPECT image data contains NaNs")
            break
    for data in [spect_data["acquisition_data"], spect_data["additive"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("SPECT projection data contains NaNs")
            break

    return guidance_image, spect_data


def get_data_fidelity(args, spect_data, get_spect_am, num_subsets):
    """
    Set up data fidelity (objective) functions for SPECT.

    Returns:
        obj_funs: List of SPECT objective functions.
        s_inv: Sensitivity image ^ -1.
        kappa: Kappa-squared weighting image.
    """
    # Partition SPECT data
    _, _, obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets,
        mode="staggered",
        create_acq_model=get_spect_am,
    )

    for obj_fun in obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    # Get sensitivity image ^ -1
    s_inv = get_s_inv_from_subset_objs(obj_funs, spect_data["initial_image"])
    s_inv.write(os.path.join(args.output_path, "s_inv_spect.hv"))

    # Compute kappa image if requested
    _, gauss = get_filters()
    if args.use_kappa:
        kappa = compute_kappa_squared_image_from_partitioned_objective(
            obj_funs, spect_data["initial_image"]
        )
        gauss.apply(kappa)
    else:
        kappa = None

    # SPECT uses image_data_processor which works correctly for SPECT projectors
    # No need to wrap objectives with blur operator

    return obj_funs, s_inv, kappa


def main(args) -> None:
    """Main function to execute the SPECT DTV image reconstruction algorithm."""
    configure_logging()

    # Data preparation
    guidance_image, spect_data = prepare_data(args)

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.keep_all_views_in_cache,
            gauss_fwhm=args.spect_gauss_fwhm,
            attenuation=True,
        )

    # Set up data fidelity functions
    obj_funs, s_inv, kappa = get_data_fidelity(
        args, spect_data, get_spect_am_with_res, args.num_subsets
    )

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        args.delta = args.gamma_spect * spect_data["initial_image"].max() / 1e3

    save_args(args, "args.csv")

    # Save kappa image
    if kappa is not None:
        logging.info(f"Writing kappa with max {kappa.max()}")
        kappa.write(os.path.join(args.output_path, "kappa_sq_spect.hv"))

    # Set up the DTV prior
    if args.no_prior:
        prior = None
    else:
        # Apply kappa weighting if available
        weight = args.gamma_spect
        if kappa is not None:
            logging.info("Using kappa-weighted DTV prior")

        dtv_prior = TotalVariation(
            geometry=spect_data["initial_image"],
            weight=weight,
            delta=args.delta,
            anatomical=guidance_image,
            stencil=getattr(args, "spect_stencil", "6"),
            both_directions=getattr(args, "spect_both_directions", False),
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
        max_precond_value = 10.0 * spect_data["initial_image"].max() * s_inv.max()
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

    # Set up probabilities
    probs = [1.0 / len(obj_funs)] * len(obj_funs)

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
        initial=spect_data["initial_image"],
        f=-SumFunction(f_obj, prior) if prior else -f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size,
        update_objective_interval=update_interval,
    )
    algo.run(subiterations, verbose=1, callbacks=callbacks)

    save_results(algo, args)
    logging.info("SPECT DTV reconstruction complete")


if __name__ == "__main__":
    cli = parse_cli()
    cfg_dict = load_config(cli.config)
    cfg_dict = apply_overrides(cfg_dict, cli.override)

    args = SimpleNamespace(**cfg_dict)
    msg = init_run_env(args)

    if args.profile:
        logging.info("Profiling is enabled. This may slow down the execution.")
        profiler = cProfile.Profile()
        profiler.enable()
        main(args)
        profiler.disable()
        profiler.dump_stats(f"{args.output_path}/profile_data.prof")

        with open(f"{args.output_path}/profiling_results.txt", "w") as f:
            logging.info("Writing profiling results to 'profiling_results.txt'")
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats("cumulative").print_stats(None)
    else:
        logging.info("Profiling disabled.")
        main(args)
    logging.info("Execution completed.")
