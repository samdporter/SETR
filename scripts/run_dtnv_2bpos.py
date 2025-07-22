#!/usr/bin/env python3
# run_dtnv_2bpos.py

import argparse
import cProfile
import logging
import os
import pandas as pd
import pstats
import shutil
from types import SimpleNamespace
from typing import Any, List

import numpy as np

# SIRF imports
from sirf.STIR import AcquisitionData, MessageRedirector
from sirf.Reg import NiftiImageData3DDisplacement
from sirf.contrib.partitioner import partitioner

AcquisitionData.set_storage_scheme("memory")

# CIL imports
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    SAGAFunction, 
    SVRGFunction,
    SumFunction,
)
from cil.optimisation.operators import (
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)
from cil.optimisation.utilities import Sampler

# SETR imports
from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import (
    NiftyResampleOperator,
    CouchShiftOperator,
    ImageCombineOperator,
    AdjointOperator,
)
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.priors import WeightedVectorialTotalVariation
from setr.utils import (
    get_pet_am, get_pet_data_multiple_bed_pos,
    get_spect_am, get_spect_data
)
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import (
    attach_prior_hessian,
    compute_kappa_squared_image_from_partitioned_objective,
    get_block_objective,
    get_filters,
    get_s_inv_from_subset_objs,
    get_sensitivity_from_subset_objs,
)


cli = parse_cli()
cfg_dict = load_config(cli.config)
cfg_dict = apply_overrides(cfg_dict, cli.override)

args = SimpleNamespace(**cfg_dict)

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(args.working_path, exist_ok=True)

# Attach the new update method to ISTA.
ISTA.update = ista_update_step


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

               
def prepare_data(args):
    """
    Prepare theumapimage, PET and SPECT data, and initial estimates.

    Returns:
        ct: Normalizedumapimage.
        pet_data: Dictionary containing PET data.
        spect_data: Dictionary containing SPECT data.
        initial_estimates: BlockDataContainer combining PET and SPECT initial images.
        cyl, gauss: Filter objects.
    """

    
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof = args.use_tof,
        suffixes=["_f1b1", "_f2b1"]
    )
    
    umap = pet_data['attenuation']
    umap+= (-umap).max()
    umap/=umap.max()
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters()

    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])
    
    pet_data["initial_image"].write("initial_image_0.hv")
    spect_data["initial_image"].write("initial_image_1.hv")

    # Set delta (smoothing parameter) if not provided
    if args.delta is None:
        args.delta = max(
            pet_data["initial_image"].max() / 1e4,
            spect_data["initial_image"].max() / 1e4,
        ) * min(args.alpha, args.beta)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )
    
    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    # check for nans in all data
    for data in [umap, pet_data["initial_image"], spect_data["initial_image"]]:
        if np.isnan(data.as_array()).any():
            logging.warning("An image contains NaNs")
            break

    return umap, pet_data, spect_data, initial_estimates

def get_shift_operator(pet_data):
    """
    Set up the shift operator for the image reconstruction.

    Returns:
        shift_operator: The constructed shift operator.
    """
    
    suffixes=["_f1b1", "_f2b1"]
    
    pet_shifts = [
        CouchShiftOperator.get_couch_shift_from_sinogram(
            pet_data['bed_positions'][suffix]["acquisition_data"])
        for suffix in suffixes
    ]
    
    shift_ops = [
        CouchShiftOperator(pet_data['bed_positions'][suffix]["template_image"], pet_shift)
        for suffix, pet_shift in zip(suffixes, pet_shifts)
    ]
    
    shifted_images = [
        op.direct(pet_data['bed_positions'][suffix]["template_image"])
        for suffix, op in zip(suffixes, shift_ops)
    ]
    
    combine_op = ImageCombineOperator(
        EnhancedBlockDataContainer(
            *shifted_images
        )
    )
    
    unshift_ops = [
        AdjointOperator(op)
        for op in shift_ops
    ]
    
    uncombine_op = AdjointOperator(
        combine_op
    )
    
    choose_op_0 = BlockOperator(
        IdentityOperator(
            shifted_images[0]
        ),
        ZeroOperator(
            shifted_images[1],
            shifted_images[0]
        ),
        shape=(1, 2)
    )
    
    choose_op_1 = BlockOperator(
        ZeroOperator(
            shifted_images[0],
            shifted_images[1]
        ), 
        IdentityOperator(
            shifted_images[1]
        ),
        shape=(1, 2)
    )
    
    choose_ops = [
        choose_op_0,
        choose_op_1
    ]

    main_tmpl = pet_data['template_image']
    shift_combine = CouchShiftOperator(main_tmpl, 0)
        
    return uncombine_op, unshift_ops, choose_ops


def get_resampling_operators(
    args, pet_data, spect_data,
):
    """
    Set up resampling operators for SPECT images to PET images.

    Returns:
        spect2ct, zero_spect2ct operators.
    """
    
    spect2pet_nonrigid = NiftyResampleOperator(
        pet_data["initial_image"],
        spect_data["initial_image"],
        NiftiImageData3DDisplacement(
            os.path.join(args.spect_data_path, "spect2pet.nii")
        ),
    )
    
    return spect2pet_nonrigid

def get_prior(
    args, umap, pet_data, spect_data,
    initial_estimates, spect2pet,
    kappas=None,
):
    """
    Set up the prior function for image reconstruction.

    Returns:
        prior: The constructed prior function.
        bo: The block operator used within the prior.
    """
    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),  # pet2pet
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),  # zero_spect2pet
        ZeroOperator(pet_data["initial_image"]),  # zero_pet2pet
        spect2pet,  # spect2pet
        shape=(2, 2),
    )

    if kappas is not None:
        for i, kappa in enumerate(kappas.containers):
            logging.info(f"Writing kappa {i} with max {kappa.max()}")
            kappa.write(os.path.join(args.output_path, f"kappa_{i}.hv"))
        kappas = bo.direct(kappas)
    else:
        kappas = EnhancedBlockDataContainer(
            pet_data["initial_image"].get_uniform_copy(1),
            spect_data["initial_image"].get_uniform_copy(1),
        )
        kappas = bo.direct(kappas)
    for i, (b, el) in enumerate(zip([args.alpha, args.beta], kappas.containers)):
        kappas.containers[i].fill(b * el)

    vtv = WeightedVectorialTotalVariation(
        bo.direct(initial_estimates),
        kappas,
        args.delta,
        anatomical=umap,
        gpu=not args.no_gpu,
        stable=True,
        tail_singular_values=args.tail_singular_values, 
    )
    prior = OperatorCompositionFunction(vtv, bo)
    return prior

def get_data_fidelity(
    args, pet_data, spect_data,
    get_pet_am, get_spect_am,
    num_subsets, uncombine_op,
    unshift_ops, choose_ops
):
    """
    Set up data fidelity (objective) functions.
    
    Returns:
        List of objective functions.
    """
    # Partition data for each bed position.
    pet_dfs = [
        partitioner.data_partition(
            pet_data['bed_positions'][suffix]["acquisition_data"],
            pet_data['bed_positions'][suffix]["additive"],
            pet_data['bed_positions'][suffix]["normalisation"],
            num_batches=num_subsets[0],
            mode="staggered",
            create_acq_model=get_pet_am,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # Set up the objective functions for each bed position.
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(pet_data['bed_positions'][suffix]["template_image"])
            
    _, _, spect_dfs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"],
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am,
    )

    for obj_fun in spect_dfs:
        obj_fun.set_up(spect_data['initial_image'])
        
    # Before we add all the complicated operators, we'll get the kappa images and sensitivity images
    pet_sens = [
        get_sensitivity_from_subset_objs(
            df, pet_data['bed_positions'][suffix]["template_image"]
            ) 
        for df, suffix in zip(pet_dfs, pet_data["bed_positions"])
    ]
    
    spect_s_inv = get_s_inv_from_subset_objs(
        spect_dfs, spect_data['initial_image']
    )
    
    # need to unshift and then combine pet images
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(*[
            unshift_op.adjoint(s)
            for unshift_op, s in zip(unshift_ops, pet_sens)
        ]) 
    )
    pet_s_inv = pet_sens_combined.clone()
    pet_sens_array = pet_sens_combined.as_array()
    pet_s_inv.fill(
        np.reciprocal(
            pet_sens_array,
            where=pet_sens_array != 0,
        )
    )
    cyl, _ = get_filters()
    cyl.apply(pet_s_inv)
    
    s_inv = EnhancedBlockDataContainer(
        pet_s_inv, spect_s_inv
    )
    
    # save the s_inv images
    for i, image in enumerate(s_inv.containers):
        image.write(os.path.join(args.output_path, f"s_inv_{i}.hv"))
        logging.info(f"Writing s_inv_{i} with max {image.max()}")
            
    # For each bed position, update each subset operator.
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            # Replace the operator with the composed operator that applies:
            # uncombine -> choose -> unshift, then the original op.
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                ),
            )

    # At this point, pet_dfs is a list (over beds) of lists (over subsets).
    # Flatten the list so you obtain one function per subset per bed.
    pet_combined_dfs = [df for bed in pet_dfs for df in bed]

    # Convert each operator into a final objective function.
    pet_dfs = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            df,
            order=0,
        )
        for df in pet_combined_dfs  
    ]
 
    
    spect_dfs = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_dfs
    ]

    all_funs = pet_dfs + spect_dfs

    return all_funs, s_inv, None


def get_preconditioners(
    args: argparse.Namespace,
    s_inv: Any,
    all_funs: List[Any],
    update_interval: int,
    prior: Any, initial_estimates: EnhancedBlockDataContainer,
) -> Any:
    """
    Set up the preconditioners.

    Returns:
        The combined preconditioner.
    """
    
    max_vals = [el.max() for el in initial_estimates.containers]
    epsilon = min([el.max() for el in initial_estimates.containers]) * 1e-3
    
    bsrem_precond = BSREMPreconditioner(
        s_inv, 1, np.inf, 
        epsilon=epsilon,
        max_vals=max_vals,
        smooth=True,
    )
    if prior is None:
        return bsrem_precond
    
    prior_precond = ImageFunctionPreconditioner(
        prior.inv_hessian_diag, 1., 
        update_interval, 
        freeze_iter=np.inf,
        epsilon=epsilon,
    )
    
    precond = LehmerMeanPreconditioner(
        [bsrem_precond, prior_precond],
        update_interval=update_interval,
        freeze_iter=len(all_funs) * 10,
    )
    
    return precond

def get_probabilities(args, num_subsets, update_interval):
    pet_probs = [1 / update_interval] * num_subsets[0] * 2
    spect_probs = [1 / update_interval] * num_subsets[1]
    probs = pet_probs + spect_probs
    assert abs(sum(probs) - 1) < 1e-10, f"Probabilities do not sum to 1, got {sum(probs)}"
    return probs

def get_callbacks(args, update_interval: int) -> List[Any]:
    """
    Set up callbacks for the algorithm.

    Returns:
        A list of callback objects.
    """
    return [
        SaveImageCallback(os.path.join(args.output_path, "image"), update_interval),
        SaveGradientUpdateCallback(os.path.join(args.output_path, "gradient"), update_interval),
        SavePreconditionerCallback(os.path.join(args.output_path, "preconditioner"), update_interval),
        PrintObjectiveCallback(update_interval),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), update_interval),
    ]


def get_algorithm(
    init_solution: EnhancedBlockDataContainer,
    f_obj: Any,
    precond: Any,
    step_size: float,
    update_interval: int,
    subiterations: int,
    callbacks: List[Any],
) -> ISTA:
    """
    Set up and run the ISTA algorithm.

    Returns:
        The ISTA instance.
    """
    algo = ISTA(
        initial=init_solution,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=precond,
        step_size=step_size,
        update_objective_interval=update_interval,
    )
    logging.info("Running algorithm")
    algo.run(subiterations, verbose=1, callbacks=callbacks)
    return algo


def save_results(
    bsrem: ISTA, args: argparse.Namespace
) -> None:
    """Save profiling information and results to disk."""

    os.makedirs(args.output_path, exist_ok=True)
    df_objective = pd.DataFrame([l for l in bsrem.loss])
    df_objective.to_csv(
        os.path.join(
            args.output_path, f"bsrem_objective_a_{args.alpha}_b_{args.beta}.csv"
        ),
        index=False,
    )

    for file in os.listdir(args.working_path):
        if file.startswith("tmp_") and (file.endswith(".s") or file.endswith(".hs")):
            os.remove(os.path.join(args.working_path, file))
    for file in os.listdir(args.working_path):
        if file.endswith((".hv", ".v", ".ahv")):
            logging.info(f"Moving file {file} to {args.output_path}")
            shutil.move(
                os.path.join(args.working_path, file),
                os.path.join(args.output_path, file),
            )


def main() -> None:
    """Main function to execute the image reconstruction algorithm."""
    configure_logging()
    os.chdir(args.working_path)

    # Redirect messages if needed.
    msg = MessageRedirector()

    # Data preparation.
    umap, pet_data, spect_data, initial_estimates = prepare_data(args)

    # find alpha weighting using dynamic range of the initial images (95th percentile)
    pet_max = np.percentile(pet_data["initial_image"].as_array(), 95)
    spect_max = np.percentile(spect_data["initial_image"].as_array(), 95)
    args.alpha = args.alpha * spect_max / pet_max
    logging.info(f"Setting alpha to {args.alpha} based on initial images")
        
    save_args(args, "args.csv")

    # Set up resampling operators.
    spect2pet = get_resampling_operators(args, pet_data, spect_data)
    
    get_pet_am_with_res = lambda: get_pet_am(
        not args.no_gpu,
        gauss_fwhm=args.pet_gauss_fwhm,
    )
    
    uncombine_op, unshift_ops, choose_ops = get_shift_operator(pet_data)
    
    get_spect_am_with_res = lambda: get_spect_am(
        spect_data,
        res=args.spect_res,
        keep_all_views_in_cache=args.stop_keep_all_views_in_cache,
        gauss_fwhm=args.spect_gauss_fwhm,
        attenuation=True,
    )

    # Set up data fidelity functions.
    num_subsets = [int(i) for i in args.num_subsets]
    all_funs, s_inv, kappa = get_data_fidelity(
        args, 
        pet_data, spect_data, 
        get_pet_am_with_res,
        get_spect_am_with_res,
        num_subsets,
        uncombine_op,
        unshift_ops,     
        choose_ops
    )
    
    if args.no_prior:
        prior = None
    else:
        # Set up the prior.
        prior = get_prior(
                args, umap, pet_data, 
                spect_data, initial_estimates, 
                spect2pet, kappa
                )
        # Scale and attach Hessian to the prior if needed.
        prior = -1 / len(all_funs) * prior
        attach_prior_hessian(prior, epsilon=1e-3)

        for i, fun in enumerate(all_funs):
            all_funs[i] = SumFunction(fun, prior)
        
    update_interval = len(all_funs)
    
    # Set up preconditioners.
    precond = get_preconditioners(
        args, s_inv, all_funs, 
        update_interval, prior,
        initial_estimates
    )

    probs = get_probabilities(args, num_subsets, update_interval)
    
    f_obj = -SVRGFunction(
            all_funs, sampler=Sampler.random_with_replacement(len(all_funs), prob=probs,),
            snapshot_update_interval=update_interval*2, store_gradients=True
        )
    #f_obj = -SAGAFunction(
    #        all_funs, sampler=Sampler.random_with_replacement(len(all_funs), prob=probs,),
    #    )
    #f_obj.function.warm_start_approximate_gradients(initial_estimates)
    #f_obj = -SGFunction(
    #        all_funs, sampler=Sampler.random_with_replacement(len(all_funs), prob=probs,),
    #    )

    callbacks = get_callbacks(args, update_interval)

    algo = get_algorithm(
        initial_estimates,
        f_obj,
        precond,
        LinearDecayStepSizeRule(
            args.initial_step_size,
            args.relaxation_eta,
        ),
        update_interval,
        args.num_epochs * update_interval,
        callbacks,
    )

    save_results(algo, args)
    logging.info("Done")

#%%
if __name__ == "__main__":

    if args.profile:
        logging.info("Profiling is enabled. This may slow down the execution.")
        profiler = cProfile.Profile()
        profiler.enable()
        
        main()

        profiler.disable()
        profiler.dump_stats(args.output_path + '/profile_data.prof')
        # Output results to a file
        output_file = os.path.join(args.output_path, "profiling_results.txt")
        with open(output_file, "w") as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats("cumulative").print_stats()
        logging.info(f"Profiling results saved to {output_file}")
    else:
        logging.info("Profiling is disabled.")
        main()
    logging.info("Execution completed.")
