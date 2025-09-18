#!/usr/bin/env python3
"""SETR HKEM reconstruction for multiple bed positions - Simplified version using shared modules."""

import logging
import os
from types import SimpleNamespace

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    OperatorCompositionFunction, 
    SumFunction,
    SGFunction,
)
from cil.optimisation.operators import CompositionOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner
from sirf.STIR import MessageRedirector

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.preconditioners import (
    DualModalitySubsetKernelisedEMPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.cil_extensions.operators import AdjointOperator, CouchShiftOperator
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
)
from setr.scripts.common import (
    configure_logging,
    get_sensitivities_from_subset_objs,
    get_shift_operators,
    init_run_env,
)
from setr.scripts.hkem_common import get_kernel_hyperparams, get_kernel_operator
from setr.utils import get_pet_data_multiple_bed_pos
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters, get_pet_am, get_array


def prepare_data(args):
    """Prepare the multi-bed PET data and guidance image."""
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    # Use attenuation map as guidance
    guidance = pet_data["attenuation"]
    guidance += (-guidance).max()
    guidance /= guidance.max()

    # Apply filters to initial images
    pet_data["initial_image"].fill(1)
    
    pet_data["initial_image"].write(
        os.path.join(args.output_path, "initial_image.hv")
    )

    # Create initial estimates - for HKEM we just need PET
    initial_estimates = pet_data["initial_image"]

    if np.isnan(get_array(initial_estimates)).any():
        logging.warning("Initial image contains NaNs")

    return pet_data, guidance, initial_estimates


def get_data_fidelity(
    args, pet_data, get_pet_am, num_subsets, uncombine_op, unshift_ops, choose_ops, unzero_shift_op
):
    """
    Set up data fidelity (objective) functions for multi-bed reconstruction.

    Returns:
        all_funs: List of block objective functions (PET all beds)
        s_inv: Sensitivity inverse images (combined across beds)
        sensitivities: List of sensitivity images for each bed/subset
    """
    # Partition PET data by bed position
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

    # Set up subset objectives on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # Get sensitivities for each bed position
    pet_sens = [
        get_sensitivities_from_subset_objs(df)
        for df in pet_dfs
    ]
    
    # save sensitivities for debugging
    for i, sens in enumerate(pet_sens):
        for j, s in enumerate(sens):
            s.write(os.path.join(args.output_path, f"sens_bed{i}_subset{j}.hv"))

    # Combine corresponding subsets across bed positions
    num_subsets = len(pet_sens[0])  # Get number of subsets from first bed position
    pet_sens_combined = [
        unzero_shift_op.adjoint(
            uncombine_op.adjoint(
                EnhancedBlockDataContainer(
                    *[unshift_op.adjoint(sens[subset_idx]) 
                    for unshift_op, sens in zip(unshift_ops, pet_sens)]
                )
            )
        )
        for subset_idx in range(num_subsets)
    ]
    
    
    # debug print
    print(f"Number of PET sensitivities combined: {len(pet_sens_combined)}")
    print(f"type of first sensitivity: {type(pet_sens_combined[0])}")
    print(f"Shape of first sensitivity: {pet_sens_combined[0].shape}")

    # Wrap PET objectives with uncombine/choose/unshift operators
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                    unzero_shift_op
                ),
            )

    # Combine objectives across bed positions
    all_funs = [SumFunction(*[funs[j] for funs in pet_dfs]) for j in range(num_subsets)]
    print(f"len(all_funs): {len(all_funs)}")


def run_hkem_ista(args, pet_data, guidance, initial_estimates):
    """Run ISTA-based HKEM reconstruction with kernel preconditioner."""

    # Get acquisition model function
    def get_pet_am_with_res():
        return get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.pet_gauss_fwhm)

    # Set up shift operators
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)
    shift = CouchShiftOperator.get_couch_shift_from_sinogram(
        pet_data["bed_positions"]["_f2b1"]["acquisition_data"]
    )
    print( f"Using shift of {shift}mm for bed position 2" )
    zero_shift_op = CouchShiftOperator(
        pet_data["template_image"], 0
    )
    unzero_shift_op = AdjointOperator(zero_shift_op)
    initial_estimates = zero_shift_op.direct(initial_estimates)

    # Set up data fidelity functions
    all_funs, sens = get_data_fidelity(
        args,
        pet_data,
        get_pet_am_with_res,
        args.num_subsets,
        uncombine_op,
        unshift_ops,
        choose_ops,
        unzero_shift_op,
    )

    # Create kernel operators for each bed position
    hyperparams = get_kernel_hyperparams(args)
    if args.guidance == "attenuation":
        guide = zero_shift_op.direct(pet_data["attenuation"])
    elif args.guidance == "emission":
        guide = zero_shift_op.direct(pet_data["spect"])
        assert pet_data["spect"] is not None, "Emission guidance selected but no SPECT data provided"
    else:
        raise ValueError(f"Unknown guidance type: {args.guidance}")
    assert type(guide) is type(initial_estimates), f"Guidance and initial estimates must be same type. Got {type(guide)} and {type(initial_estimates)}"
    
    print("Image shapes:")
    print(f"Guide shape: {guide.shape}")
    print(f"Initial estimates shape: {initial_estimates.shape}")

    kernel = get_kernel_operator(
        args,
        guide,
        initial_estimates,
        pet_data["bed_positions"]["_f2b1"]["acquisition_data"],
        hyperparams,
    )

    # Set up objective function
    update_interval = len(all_funs)

    f_obj = -SGFunction(
        all_funs,
        sampler=Sampler.sequential(len(all_funs)),
    )

    # Set up preconditioners
    max_val = initial_estimates.max()

    # Create dual modality kernel preconditioner
    dual_precond = DualModalitySubsetKernelisedEMPreconditioner(
        sens=sens,
        kernel=kernel,
        num_subsets=len(all_funs),
        update_interval=update_interval,
        freeze_iter=args.freeze_iter,
        epsilon=max_val * 1e-12,
    )

    # Set up callbacks
    class SaveKernelisedImageCallback:
        """Save the kernelised image (x = K(alpha)) to disk."""

        def __init__(self, filename, interval, kernel_op):
            self.filename = filename
            self.interval = interval
            self.kernel_op = kernel_op

        def __call__(self, algo):
            if algo.iteration % self.interval != 0:
                return
            # Save the kernelised image
            image = self.kernel_op.direct(algo.solution)
            image.write(f"{self.filename}_{algo.iteration}.hv")

    callbacks = [
        SaveImageCallback(
            os.path.join(args.output_path, "alpha"),
            interval=args.num_subsets,
        ),
        SaveKernelisedImageCallback(
            os.path.join(args.output_path, "x"),
            interval=args.num_subsets,
            kernel_op=kernel,
        ),
        PrintObjectiveCallback(interval=args.num_subsets),
        SaveObjectiveCallback(
            os.path.join(args.output_path, "objective"),
            interval=args.num_subsets,
        ),
    ]

    # Set up algorithm
    algo = ISTA(
        initial=initial_estimates,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=dual_precond,  # Use kernel preconditioner
        step_size=args.initial_step_size,
        update_objective_interval=update_interval,
    )

    num_subiterations = args.num_epochs * update_interval
    logging.info("Running HKEM-ISTA reconstruction...")
    algo.run(
        num_subiterations, callbacks=callbacks, verbose=True
    )

    # Get final results
    output_alpha = algo.solution

    # Apply first kernel to get kernelised image
    output_x = kernel.direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main():
    """Main function to run HKEM multi-bed reconstruction."""
    configure_logging()

    # Parse arguments and configuration
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = SimpleNamespace(**config)

    # Initialize run environment
    msg = init_run_env(args)

    # Save arguments
    save_args(args, "hkem_2bpos_args.csv")

    logging.info("Starting HKEM multi-bed reconstruction")

    # Prepare data
    pet_data, guidance, initial_estimates = prepare_data(args)

    # Run reconstruction (only ISTA supported for multi-bed)
    output_alpha, output_x = run_hkem_ista(args, pet_data, guidance, initial_estimates)

    logging.info("HKEM multi-bed reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
