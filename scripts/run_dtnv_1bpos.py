#!/usr/bin/env python3
"""SETR DTNV reconstruction for single bed position - Simplified version using shared modules."""

import cProfile
import logging
import os
import pstats
from types import SimpleNamespace

import numpy as np
from cil.optimisation.functions import OperatorCompositionFunction, SumFunction
from cil.optimisation.operators import (
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)
from sirf.contrib.partitioner import partitioner
from sirf.STIR import ImageData, SeparableGaussianImageFilter

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.operators import FlipOperator
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.scripts.common import (
    attach_prior_hessian,
    configure_logging,
    get_resampling_operators,
    init_run_env,
    save_results,
)
from setr.scripts.dtnv_common import (
    apply_dynamic_range_scaling,
    build_variance_reduced_function,
    dynamic_range_scale_sirf,
    get_algorithm,
    get_block_objective,
    get_callbacks,
    get_kappa_squareds,
    get_preconditioners,
    get_prior,
    normalise_kappa_squares,
)
from setr.utils import get_pet_am, get_pet_data, get_spect_am, get_spect_data
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_array, get_filters, get_s_inv_from_objs
from setr.cil_extensions.operators.blurring import create_gaussian_blur_operator


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
    ct_smooth = SeparableGaussianImageFilter()
    ct_smooth.set_fwhms((0.5, 0.5, 0.5))
    ct_smooth.apply(ct)

    pet_data = get_pet_data(args.pet_data_path)
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))

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

    # Create Gaussian blurring operator for PET only
    # SPECT uses image_data_processor which works correctly for SPECT projectors
    pet_blur_op = create_gaussian_blur_operator(args.pet_gauss_fwhm, pet_data["initial_image"])

    # Get sensitivity image ^ -1 now before we complicate things
    s_inv = get_s_inv_from_objs(
        [pet_obj_funs, spect_obj_funs],
        EnhancedBlockDataContainer(pet_data["initial_image"], spect_data["initial_image"]),
        clamp_percentile=99.5,
        adjoint_ops=[pet_blur_op, None],
    )

    for i, el in enumerate(s_inv.containers):
        s_inv.containers[i].write(os.path.join(args.output_path, f"s_inv_{i}.hv"))

    # Wrap PET objectives with Gaussian blurring operator (if specified)
    if pet_blur_op is not None:
        pet_obj_funs = [
            OperatorCompositionFunction(obj_fun, pet_blur_op)
            for obj_fun in pet_obj_funs
        ]

    # Convert to block objectives
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
    spect2pet = get_resampling_operators(args, pet_data, spect_data)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=None,
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

    if getattr(args, "flip", False):
        spect2pet = CompositionOperator(
            spect2pet, FlipOperator(axis=(0, 2), image=["initial_image"])
        )

    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),  # pet2pet
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),  # zero_spect2pet
        ZeroOperator(pet_data["initial_image"]),  # zero_pet2pet
        spect2pet,  # spect2pet
        shape=(2, 2),
    )

    kappas = normalise_kappa_squares(bo.direct(kappas)) if kappas else None
    combined = EnhancedBlockDataContainer(*bo.direct(initial_estimates).containers)
    pet_scale, spect_scale = dynamic_range_scale_sirf(
        combined[0],
        combined[1],
    )

    # Apply consistent scaling to all prior weights
    # Skip this for log TNV since the log transform already normalizes dynamic range
    # Skip this for local weighting since voxel-wise weights are applied directly
    use_log_tnv = getattr(args, "use_log_tnv", False)
    use_local_weighting = getattr(args, "use_local_weighting", False)
    if not use_log_tnv and not use_local_weighting:
        apply_dynamic_range_scaling(args, pet_scale, spect_scale)
        logging.info(
            "Applied dynamic range scaling: pet_scale=%.6g, spect_scale=%.6g",
            pet_scale,
            spect_scale,
        )
    else:
        if use_log_tnv:
            logging.info(
                "Skipping dynamic range scaling for log-TNV "
                "(alpha=%.6g, beta=%.6g remain as configured)",
                args.alpha,
                args.beta,
            )
        if use_local_weighting:
            logging.info(
                "Skipping dynamic range scaling for local weighting "
                "(alpha=%.6g, beta=%.6g will be applied voxel-wise)",
                args.alpha,
                args.beta,
            )

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        use_log_tnv = getattr(args, "use_log_tnv", False)

        if use_log_tnv:
            # For log TNV: use fixed relative smoothing parameter
            # The log transform normalizes dynamic range, making delta scale-invariant
            divisor = getattr(args, "delta_divisor", 10.0)
            args.delta = 1.0 / divisor
            logging.info(
                "Auto-set delta to %.6g for log-TNV (1 / %.3g divisor)",
                args.delta,
                divisor,
            )
        else:
            # For standard TNV: estimate from scaled image intensities
            # Delta should be ~divisor fraction of the 95th percentile of scaled intensities
            weighted_pet = args.alpha * get_array(combined[0])
            weighted_spect = args.beta * get_array(combined[1])

            percentile = getattr(args, "delta_percentile", 99.0)
            divisor = getattr(args, "delta_divisor", 100.0)

            # Take minimum of the two weighted image scales
            pet_val = np.percentile(weighted_pet[weighted_pet > 0], percentile)
            spect_val = np.percentile(weighted_spect[weighted_spect > 0], percentile)
            scale_val = min(pet_val, spect_val)

            args.delta = scale_val / divisor
            logging.info(
                "Auto-set delta to %.6g using %sth percentile of scaled intensities / %.3g "
                "(ratio alpha/delta=%.1f, beta/delta=%.1f)",
                args.delta,
                percentile,
                divisor,
                args.alpha / args.delta,
                args.beta / args.delta,
            )

    save_args(args, "args.csv")

    if kappas is not None:
        for i, kappa in enumerate(kappas.containers):
            logging.info(f"Writing kappa {i} with max {kappa.max()}")
            kappa.write(os.path.join(args.output_path, f"kappa_sq_{i}.hv"))

    if args.no_prior:
        prior = None
        priors_list = []
    else:
        # Set up the prior.
        priors_list = get_prior(
            args, umap, combined, bo, kappas,
            pet_scale=pet_scale, spect_scale=spect_scale
        )
        for i, p in enumerate(priors_list):
            attach_prior_hessian(priors_list[i])
        prior = -SumFunction(*priors_list)

    ui = getattr(args, "update_interval", None)
    update_interval = len(all_funs) if ui is None else ui

    # Set up preconditioners.
    precond = get_preconditioners(
        args, s_inv, all_funs, update_interval, priors_list, initial_estimates
    )

    epoch_length = len(all_funs)

    f_obj, probs, prior_prob, prior_in_sampler = build_variance_reduced_function(
        args, all_funs, prior, num_subsets, epoch_length, bpos=1
    )

    variance_reduction = getattr(args, "variance_reduction", "svrg")
    logging.info(
        "Variance reduction: %s | stochastic functions: %d",
        variance_reduction,
        getattr(f_obj, "num_functions", len(all_funs) + int(prior_in_sampler)),
    )

    if prior_in_sampler and prior_prob is not None:
        target_updates = getattr(args, "prior_updates_per_epoch", None)
        target_str = f"{float(target_updates):.3f}" if target_updates not in (None, False) else "n/a"
        expected_updates = (
            prior_prob * epoch_length / (1.0 - prior_prob) if prior_prob < 1 else float("inf")
        )
        logging.info(
            "Prior sampled with probability %.6f (target=%s updates/epoch, expected≈%.3f).",
            prior_prob,
            target_str,
            expected_updates,
        )
    elif prior is not None:
        logging.info("Prior evaluated deterministically each iteration.")

    objective = -f_obj if prior_in_sampler or prior is None else -SumFunction(f_obj, prior)

    # Set up step size
    step_size = LinearDecayStepSizeRule(
        initial_step_size=args.initial_step_size,
        decay=args.relaxation_eta,
    )

    # Set up callbacks using shared function
    callbacks = get_callbacks(args, update_interval)

    # Run algorithm using shared function
    subiterations = args.num_epochs * epoch_length
    algo = get_algorithm(
        initial_estimates,
        objective,
        precond,
        step_size,
        update_interval,
        subiterations,
        callbacks,
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
