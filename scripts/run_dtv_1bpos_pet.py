#!/usr/bin/env python3
"""SETR DTV reconstruction for single bed position PET - Directional Total Variation with guided prior."""

import cProfile
import logging
import os
import pstats
from types import SimpleNamespace

import numpy as np
from cil.optimisation.functions import SumFunction, SVRGFunction
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from cil.optimisation.algorithms import ISTA
from sirf.STIR import ImageData, MessageRedirector

from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.priors import TotalVariation
from setr.scripts.common import (
    configure_logging,
    init_run_env,
    save_results,
)
from setr.scripts.dtnv_common import (
    get_algorithm,
    get_callbacks,
    get_s_inv_from_subset_objs,
    compute_kappa_squared_image_from_partitioned_objective,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.preconditioners import BSREMPreconditioner, ImageFunctionPreconditioner, LehmerMeanPreconditioner
from setr.utils import get_pet_am, get_pet_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters, get_array


def prepare_data(args):
    """
    Prepare the guidance image, PET data, and initial estimates.

    Returns:
        guidance_image: PET guidance image (umap or emission guidance, get_array).
        pet_data: Dictionary containing PET data.
    """
    pet_data = get_pet_data(args.pet_data_path)

    # PET guidance: use emission guidance if available, otherwise use PET umap
    if getattr(args, 'use_emission_guidance', False) and hasattr(args, 'emission_guidance_path'):
        logging.info("Using emission guidance for PET reconstruction")
        guidance_image = ImageData(args.emission_guidance_path)
    else:
        logging.info("Using umap guidance for PET reconstruction")
        guidance_image = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    
    # Normalize guidance
    guidance_image += (-guidance_image).max()
    guidance_image /= guidance_image.max()

    # Apply filters to initial image
    cyl, gauss = get_filters()
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])
    pet_data["initial_image"].write("initial_image_pet.hv")

    # Check for nans
    for data in [guidance_image, pet_data["initial_image"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("PET image data contains NaNs")
            break
    for data in [pet_data["acquisition_data"], pet_data["normalisation"], pet_data["additive"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("PET projection data contains NaNs")
            break

    return guidance_image, pet_data


def get_data_fidelity(args, pet_data, get_pet_am, num_subsets):
    """
    Set up data fidelity (objective) functions for PET.

    Returns:
        obj_funs: List of PET objective functions.
        s_inv: Sensitivity image ^ -1.
        kappa: Kappa-squared weighting image.
    """
    # Partition PET data
    _, _, obj_funs = partitioner.data_partition(
        pet_data["acquisition_data"],
        pet_data["additive"],
        pet_data["normalisation"],
        num_batches=num_subsets,
        mode="staggered",
        create_acq_model=get_pet_am,
    )
    
    for obj_fun in obj_funs:
        obj_fun.set_up(pet_data["initial_image"])

    # Get sensitivity image ^ -1
    s_inv = get_s_inv_from_subset_objs(obj_funs, pet_data["initial_image"])
    s_inv.write(os.path.join(args.output_path, "s_inv_pet.hv"))

    # Compute kappa image if requested
    _, gauss = get_filters()
    if args.use_kappa:
        kappa = compute_kappa_squared_image_from_partitioned_objective(obj_funs, pet_data["initial_image"])
        gauss.apply(kappa)
    else:
        kappa = None

    return obj_funs, s_inv, kappa


def main(args) -> None:
    """Main function to execute the PET DTV image reconstruction algorithm."""
    configure_logging()

    # Data preparation
    guidance_image, pet_data = prepare_data(args)

    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=args.pet_gauss_fwhm,
        )

    # Set up data fidelity functions
    obj_funs, s_inv, kappa = get_data_fidelity(
        args, pet_data, get_pet_am_with_res, args.num_subsets
    )

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        args.delta = args.gamma_pet * pet_data["initial_image"].max() / 1e3

    save_args(args, "args.csv")

    # Save kappa image
    if kappa is not None:
        logging.info(f"Writing kappa with max {kappa.max()}")
        kappa.write(os.path.join(args.output_path, "kappa_sq_pet.hv"))

    # Set up the DTV prior
    if args.no_prior:
        prior = None
    else:
        # Apply kappa weighting if available
        weight = args.gamma_pet
        if kappa is not None:
            # For single modality with kappa, we need to handle weighting differently
            # The TotalVariation class expects a scalar weight, so we incorporate kappa into the anatomical guidance
            logging.info("Using kappa-weighted DTV prior")
            
        dtv_prior = TotalVariation(
            geometry=pet_data["initial_image"],
            weight=weight,
            delta=args.delta,
            anatomical=guidance_image,
            stencil=getattr(args, "pet_stencil", '6'),
            both_directions=getattr(args, "pet_both_directions", False),
        )
        
        # Attach Hessian for preconditioner
        dtv_prior.inv_hessian_diag = lambda x, out=None, epsilon=1e-9: dtv_prior.inv_hessian_diag(x, out, epsilon)
        prior = -dtv_prior

    ui = getattr(args, "update_interval", None)
    update_interval = len(obj_funs) if ui is None else ui

    # Set up preconditioners
    bsrem_precond = BSREMPreconditioner(s_inv, 1, np.inf, epsilon=0, smooth=True)
    
    if prior is not None:
        prior_precond = ImageFunctionPreconditioner(
            dtv_prior.inv_hessian_diag, 1.0, freeze_iter=np.inf, epsilon=0,
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
    probs = [1.0/len(obj_funs)] * len(obj_funs)

    f_obj = SVRGFunction(
        obj_funs,
        sampler=Sampler.random_with_replacement(len(obj_funs), prob=probs),
        snapshot_update_interval=update_interval * 2,
        store_gradients=True
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

    save_results(algo, args)
    logging.info("PET DTV reconstruction complete")


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