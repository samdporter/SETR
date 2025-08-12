#!/usr/bin/env python3
"""
Hybrid Kernelized Expectation Maximization (HKEM) reconstruction for multiple bed positions.

This script implements HKEM reconstruction for multi-bed PET acquisitions using the SETR framework.
It handles couch shifting, image combining/uncombining, and applies kernelized preconditioners
across multiple bed positions.
"""

import logging
import os
from types import SimpleNamespace

import numpy as np
from sirf.contrib.partitioner import partitioner

# SIRF imports
from sirf.STIR import AcquisitionData, MessageRedirector

AcquisitionData.set_storage_scheme("memory")

# CIL imports
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    SVRGFunction,
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
    SaveImageCallback,
    SaveObjectiveCallback,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import (
    AdjointOperator,
    CouchShiftOperator,
)
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    DualModalitySubsetKernelisedEMPreconditioner,
)
from setr.cil_extensions.utilities import LinearDecayStepSizeRule
from setr.kernel.stir import STIRKernelOperator
from setr.scripts.common import (
    get_sensitivity_from_subset_objs,
)
from setr.utils import get_pet_am, get_pet_data_multiple_bed_pos
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import (
    get_filters,
)

# Configure CLI and load config
cli = parse_cli()
cfg_dict = load_config(cli.config)
cfg_dict = apply_overrides(cfg_dict, cli.override)
args = SimpleNamespace(**cfg_dict)

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(args.working_path, exist_ok=True)

# Attach the new update method to ISTA
ISTA.update = ista_update_step


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_data(args):
    """
    Prepare the multi-bed PET data and guidance image.

    Returns:
        pet_data: Dictionary containing multi-bed PET data
        guidance: Guidance image for kernel operator (attenuation map)
        initial_estimates: Initial estimates for reconstruction
    """
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    # Use attenuation map as guidance
    guidance = pet_data["attenuation"]
    guidance += (-guidance).max()
    guidance /= guidance.max()

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    pet_data["initial_image"].write("initial_image.hv")

    # Create initial estimates - for HKEM we just need PET
    initial_estimates = pet_data["initial_image"]

    if np.isnan(initial_estimates.as_array()).any():
        logging.warning("Initial image contains NaNs")

    return pet_data, guidance, initial_estimates


def get_shift_operators(pet_data):
    """
    Set up the couch shift and image combining operators for multi-bed reconstruction.

    Returns:
        uncombine_op: Operator to combine shifted bed images
        unshift_ops: List of operators to unshift each bed position
        choose_ops: List of operators to select specific bed position
    """
    suffixes = ["_f1b1", "_f2b1"]

    # Get couch shifts for each bed position
    pet_shifts = [
        CouchShiftOperator.get_couch_shift_from_sinogram(
            pet_data["bed_positions"][suffix]["acquisition_data"]
        )
        for suffix in suffixes
    ]

    # Create shift operators
    shift_ops = [
        CouchShiftOperator(pet_data["bed_positions"][suffix]["template_image"], pet_shift)
        for suffix, pet_shift in zip(suffixes, pet_shifts)
    ]

    # Create shifted images
    shifted_images = [
        op.direct(pet_data["bed_positions"][suffix]["template_image"])
        for suffix, op in zip(suffixes, shift_ops)
    ]

    # Import ImageCombineOperator (assuming it exists in SETR)
    try:
        from setr.cil_extensions.operators import ImageCombineOperator

        combine_op = ImageCombineOperator(EnhancedBlockDataContainer(*shifted_images))
    except ImportError:
        # Fallback: create a simple combining operator
        logging.warning("ImageCombineOperator not found, using simple sum")
        from setr.cil_extensions.operators import LinearOperator

        class SimpleCombineOperator(LinearOperator):
            def __init__(self, images):
                self.images = images
                super().__init__(
                    domain_geometry=EnhancedBlockDataContainer(*images),
                    range_geometry=images[0],
                )

            def direct(self, x, out=None):
                result = self.range_geometry.get_uniform_copy(0)
                for container in x.containers:
                    result += container
                if out is not None:
                    out.fill(result)
                    return out
                return result

            def adjoint(self, x, out=None):
                containers = [x.clone() for _ in self.images]
                result = self.domain_geometry(*containers)
                if out is not None:
                    out.fill(result)
                    return out
                return result

        combine_op = SimpleCombineOperator(shifted_images)

    # Create unshift operators (adjoints of shift operators)
    unshift_ops = [AdjointOperator(op) for op in shift_ops]

    # Create uncombine operator (adjoint of combine operator)
    uncombine_op = AdjointOperator(combine_op)

    # Create selection operators for each bed position
    choose_op_0 = BlockOperator(
        IdentityOperator(shifted_images[0]),
        ZeroOperator(shifted_images[1], shifted_images[0]),
        shape=(1, 2),
    )

    choose_op_1 = BlockOperator(
        ZeroOperator(shifted_images[0], shifted_images[1]),
        IdentityOperator(shifted_images[1]),
        shape=(1, 2),
    )

    choose_ops = [choose_op_0, choose_op_1]

    return uncombine_op, unshift_ops, choose_ops


def get_kernel_hyperparams(args):
    """Return kernel hyperparameters dictionary."""
    return {
        "num_neighbours": args.num_neighbours,
        "num_non_zero_features": args.num_non_zero_features,
        "sigma_anat": args.sigma_anatomical,
        "sigma_emission": args.sigma_emission,
        "sigma_dist": args.sigma_distance_anatomical,
        "normalize_features": args.normalize_features,
        "normalize_kernel": args.normalize_kernel,
        "use_mask": args.use_mask,
        "mask_k": args.mask_k,
        "recalc_mask": args.recalc_mask,
        "distance_weighting": args.distance_weighting,
        "hybrid": args.hybrid,
        "only_2D": args.only_2D,  # Use 2D kernels only
    }


def get_data_fidelity(
    args, pet_data, get_pet_am, num_subsets, uncombine_op, unshift_ops, choose_ops
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
        get_sensitivity_from_subset_objs(df, pet_data["bed_positions"][suffix]["template_image"])
        for df, suffix in zip(pet_dfs, pet_data["bed_positions"])
    ]

    # Unshift and combine PET sensitivities to common PET grid
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )

    # Create sensitivity inverse
    s_inv = pet_sens_combined.clone()
    sens_array = pet_sens_combined.as_array()
    s_inv.fill(np.reciprocal(sens_array, where=sens_array != 0))

    # Apply cylindrical filter
    cyl, _ = get_filters()
    cyl.apply(s_inv)

    # Save sensitivity inverse
    s_inv.write(os.path.join(args.output_path, "s_inv.hv"))
    logging.info(f"Writing s_inv with max {s_inv.max()}")

    # Wrap PET objectives with uncombine/choose/unshift operators
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                ),
            )

    # Flatten the list to get one function per subset per bed
    all_funs = [df for bed in pet_dfs for df in bed]

    # Create list of sensitivities for preconditioner
    # We need to transform the bed-level sensitivities using the same operators
    sens_bdcs = []
    for bed_idx in range(len(pet_dfs)):  # For each bed
        for _ in range(len(pet_dfs[bed_idx])):
            # Transform the sensitivity from bed coordinates to combined coordinates
            bed_sens = pet_sens[bed_idx]  # Sensitivity for this bed
            combined_sens = uncombine_op.adjoint(
                EnhancedBlockDataContainer(
                    *[
                        unshift_ops[i].adjoint(bed_sens)
                        if i == bed_idx
                        else pet_data["initial_image"].get_uniform_copy(0)
                        for i in range(len(unshift_ops))
                    ]
                )
            )
            sens_bdcs.append(combined_sens)

    return all_funs, s_inv, sens_bdcs


def get_kernel_operator(args, guide_image, template_image, template_sinogram, hyperparams):
    K = STIRKernelOperator(
        template_image,
        template_sinogram,
        guide_image,
        num_neighbours=hyperparams["num_neighbours"],
        num_non_zero_features=hyperparams["num_non_zero_features"],
        sigma_m=hyperparams["sigma_m"],
        sigma_p=hyperparams["sigma_p"],
        sigma_dm=hyperparams["sigma_dm"],
        sigma_dp=hyperparams["sigma_dp"],
        only_2D=hyperparams["only_2D"],
        hybrid=hyperparams["hybrid"],
    )


def run_hkem_ista(args, pet_data, guidance, initial_estimates):
    """Run ISTA-based HKEM reconstruction with kernel preconditioner."""

    # Get acquisition model function
    get_pet_am_with_res = lambda: get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.pet_gauss_fwhm)

    # Set up shift operators
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    # Set up data fidelity functions
    all_funs, s_inv, sens_bdcs = get_data_fidelity(
        args,
        pet_data,
        get_pet_am_with_res,
        args.num_subsets,
        uncombine_op,
        unshift_ops,
        choose_ops,
    )

    # Create kernel operators for each bed position
    hyperparams = get_kernel_hyperparams(args)
    kernels = []
    for suffix in pet_data["bed_positions"]:
        guide = (
            pet_data["bed_positions"][suffix]["attenuation"]
            if args.guidance == "attenuation"
            else pet_data["bed_positions"][suffix]["spect"]
        )
        kernel = get_kernel_operator(
            args,
            guide,
            pet_data["bed_positions"][suffix]["template_image"],
            pet_data["bed_positions"][suffix]["acquisition_data"],
            **hyperparams,
        )
        kernel.set_anatomical_image(guidance)  # Use same guidance for all beds
        kernels.append(kernel)

    # Set up objective function
    update_interval = len(all_funs)
    probs = [1 / update_interval] * len(all_funs)

    f_obj = -SVRGFunction(
        all_funs,
        sampler=Sampler.random_with_replacement(len(all_funs), prob=probs),
        snapshot_update_interval=update_interval * 2,
        store_gradients=True,
    )

    # Set up preconditioners
    max_val = initial_estimates.max()
    bsrem_precond = BSREMPreconditioner(
        s_inv,
        update_interval=1,
        freeze_iter=np.inf,
        epsilon=max_val / 1000,
        max_vals=[max_val],
        smooth=True,
    )

    # Create dual modality kernel preconditioner
    dual_precond = DualModalitySubsetKernelisedEMPreconditioner(
        sens_bdcs=sens_bdcs,
        kernel=kernels,
        uncombine_ops=unshift_ops,
        num_subsets=len(all_funs),
        update_interval=update_interval,
        freeze_iter=args.freeze_iter,
        epsilon=max_val * 1e-12,
    )

    # Set up algorithm
    algo = ISTA(
        initial=initial_estimates,
        f=f_obj,
        g=BlockIndicatorBox(lower=0, upper=np.inf),
        preconditioner=dual_precond,  # Use kernel preconditioner
        step_size=LinearDecayStepSizeRule(args.initial_step_size, args.relaxation_eta),
        update_objective_interval=update_interval,
    )

    # Set up callbacks
    callbacks = [
        SaveImageCallback(os.path.join(args.output_path, "alpha"), update_interval),
        PrintObjectiveCallback(update_interval),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), update_interval),
    ]

    num_subiterations = args.num_epochs * update_interval
    logging.info("Running HKEM-ISTA reconstruction...")
    algo.run(num_subiterations, verbose=True, callbacks=callbacks)

    # Get final results
    output_alpha = algo.solution

    # Apply first kernel to get kernelised image
    output_x = kernels[0].direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main():
    """Main function to run HKEM multi-bed reconstruction."""
    configure_logging()

    # Redirect messages
    msg = MessageRedirector()

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
