#!/usr/bin/env python3
"""SETR DTNV reconstruction for single bed position - Simplified version using shared modules."""

import cProfile
import logging
import os
import pstats
from types import SimpleNamespace

import numpy as np
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    ScaledFunction,
    SumFunction,
    SVRGFunction,
)
from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from sirf.STIR import ImageData, MessageRedirector

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.priors import (
    WeightedTotalVariation, 
    WeightedVectorialTotalVariation, 
    WeightedRDP,
)
from setr.scripts.common import (
    attach_prior_hessian,
    configure_logging,
    get_resampling_operators,
    init_run_env,
    save_results,
)
from setr.scripts.dtnv_common import (
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_kappa_squareds,
    get_preconditioners,
    get_probabilities,
    get_s_inv_from_objs,
    normalise_kappa_squares,
    gradient_energy_scale_sirf,
    get_prior,
    apply_gradient_energy_scaling,
)
from setr.utils import get_pet_am, get_pet_data, get_spect_am, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters, get_array


def prepare_data(args):
    """
    Prepare the CT image, PET and SPECT data, and initial estimates.

    Returns:
        ct: Normalized CT image.
        pet_data: Dictionary containing PET data.
        spect_data: Dictionary containing SPECT data.
        initial_estimates: BlockDataContainer combining PET and SPECT initial images.
        cyl, gauss: Filter objects.
    """

    # get guidance image
    ct = ImageData(os.path.join(args.pet_data_path, "umap_zoomed.hv"))
    # Normalize CT image
    ct += (-ct).max()
    ct /= ct.max()

    pet_data = get_pet_data(args.pet_data_path)
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters()

    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    pet_data["initial_image"].write("initial_image_0.hv")
    spect_data["initial_image"].write("initial_image_1.hv")

    # check for nans in all data
    for data in [ct, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(get_array(data)).any():
            logging.warning("An image contains NaNs")
            break
    for data in [
        pet_data["acquisition_data"],
        spect_data["acquisition_data"],
        pet_data["normalisation"],
        pet_data["additive"],
        spect_data["additive"],
    ]:
        if np.isnan(get_array(data)).any():
            logging.warning("A ProjData contains NaNs")
            break

    return ct, pet_data, spect_data


def get_data_fidelity(args, pet_data, spect_data, get_pet_am, get_spect_am, num_subsets):
    """
    Set up data fidelity (objective) functions.

    Returns:
        all_funs: List of all objective functions.
        update_interval: Update interval used by the algorithm.
        s_inv: Sensitivity image ^ -1.
        pet_ams, spect_ams: Acquisition model components.
    """
    # Partition PET data.

    _, _, pet_obj_funs = partitioner.data_partition(
        pet_data["acquisition_data"],
        pet_data["additive"],
        pet_data["normalisation"],
        num_batches=num_subsets[0],
        mode="staggered",
        create_acq_model=get_pet_am,
    )
    _, _, spect_obj_funs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )

    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])
    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    # Get sensitivity image ^ -1 now before we complicate things
    s_inv = get_s_inv_from_objs(
        [pet_obj_funs, spect_obj_funs],
        EnhancedBlockDataContainer(pet_data["initial_image"], spect_data["initial_image"]),
    )

    for i, el in enumerate(s_inv.containers):
        s_inv.containers[i].write(os.path.join(args.output_path, f"s_inv_{i}.hv"))

    pet_obj_funs = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            obj_fun,
            order=0,
        )
        for obj_fun in pet_obj_funs
    ]
    spect_obj_funs = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_obj_funs
    ]

    _, gauss = get_filters()
    
    if args.use_kappa:
        kappa = get_kappa_squareds(
            [pet_obj_funs, spect_obj_funs],
            [pet_data["initial_image"], spect_data["initial_image"]],
        )
        for kappa_image in kappa.containers:
            gauss.apply(kappa_image)
        all_funs = pet_obj_funs + spect_obj_funs
    else:
        kappa = None

    all_funs = pet_obj_funs + spect_obj_funs

    return all_funs, s_inv, kappa


def main(args) -> None:
    """Main function to execute the image reconstruction algorithm."""
    configure_logging()

    # Data preparation.
    umap, pet_data, spect_data = prepare_data(args)
    
    # Set up resampling operators.
    spect2pet = get_resampling_operators(pet_data, spect_data)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )
    
    
    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))
        
    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=args.pet_gauss_fwhm,
        )

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.keep_all_views_in_cache,
            gauss_fwhm=args.spect_gauss_fwhm,
            attenuation=True,
        )

    # Set up data fidelity functions.
    num_subsets = [int(i) for i in args.num_subsets]
    all_funs, s_inv, kappas = get_data_fidelity(
        args,
        pet_data,
        spect_data,
        get_pet_am_with_res,
        get_spect_am_with_res,
        num_subsets,
    )
    
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),  # pet2pet
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),  # zero_spect2pet
        ZeroOperator(pet_data["initial_image"]),  # zero_pet2pet
        spect2pet,  # spect2pet
        shape=(2, 2),
    )
    
    kappas = normalise_kappa_squares(bo.direct(kappas)) if kappas else None
    combined = bo.direct(initial_estimates)
    scale = gradient_energy_scale_sirf(
        combined[0], combined[1], 
        mask=None,
        kappa_pet=kappas.containers[0] if kappas else None,
        kappa_spect=kappas.containers[1] if kappas else None,
    )
    # Apply consistent scaling to all prior weights
    apply_gradient_energy_scaling(args, scale)

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        # set delta as 1000 times smaller than maximum of the minimum dynamic
        # range of initial images
        # multiplied by the weighted alpha/beta
        args.delta = (
            min(
                args.alpha * initial_estimates.containers[0].max(),
                args.beta * initial_estimates.containers[1].max(),
            )
            / 1e3
        )

    save_args(args, "args.csv")

    for i, kappa in enumerate(kappas.containers):
        logging.info(f"Writing kappa {i} with max {kappa.max()}")
        kappa.write(os.path.join(args.output_path, f"kappa_sq_{i}.hv"))

    if args.no_prior:
        prior = None
        priors_list = []
    else:
        # Set up the prior.
        priors_list = get_prior(
            args, umap, combined, bo, kappas
        )        
        for i, p in enumerate(priors_list):
            attach_prior_hessian(priors_list[i])
        prior = -SumFunction(*priors_list)

    ui = getattr(args, "update_interval", None)
    update_interval = len(all_funs) if ui is None else ui

    # Set up preconditioners.
    precond = get_preconditioners(
        args, s_inv, all_funs, 
        update_interval, priors_list, 
        initial_estimates
    )

    probs = get_probabilities(args, num_subsets, len(all_funs))

    f_obj = SVRGFunction(
        all_funs,
        sampler=Sampler.random_with_replacement(
            len(all_funs),
            prob=probs,
        ),
        snapshot_update_interval=len(all_funs) * 2,
        store_gradients=True
    )

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks using shared function
    callbacks = get_callbacks(args, update_interval)

    # Run algorithm using shared function
    subiterations = args.num_epochs * len(all_funs)
    algo = get_algorithm(
        initial_estimates, 
        -SumFunction(f_obj, prior) if prior else -f_obj,
        precond, step_size, 
        update_interval, subiterations, callbacks
    )

    save_results(algo, args)
    logging.info("Done")


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
            ps.strip_dirs()  # remove extraneous path info
            ps.sort_stats("cumulative")  # sort by cumulative time
            ps.print_stats(None)  # print *every* function
    else:
        logging.info("Profiling disabled.")
        main(args)
    logging.info("Execution completed.")
