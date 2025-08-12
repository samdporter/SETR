"""Universal utilities shared across ALL SETR reconstruction scripts."""

import argparse
import logging
import os

from cil.optimisation.algorithms import ISTA
from cil.optimisation.operators import (
    BlockOperator,
    CompositionOperator,
    IdentityOperator,
    ZeroOperator,
)


def init_run_env(args):
    from sirf.STIR import AcquisitionData, MessageRedirector

    AcquisitionData.set_storage_scheme("memory")
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.working_path, exist_ok=True)
    os.chdir(args.working_path)
    return MessageRedirector()


def configure_logging() -> None:
    """Configure logging for scripts (identical across all scripts)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def save_results(bsrem: ISTA, args: argparse.Namespace) -> None:
    """Save final reconstruction results (identical in DTNV scripts)."""
    logging.info("Saving final image")
    final_image = bsrem.solution
    final_image.write(os.path.join(args.output_path, "final_image.hv"))

    if hasattr(final_image, "containers"):
        for i, container in enumerate(final_image.containers):
            container.write(os.path.join(args.output_path, f"final_image_{i}.hv"))


def get_resampling_operators(pet_data: dict, spect_data: dict):
    """
    Set up resampling operators for SPECT images to PET images.

    Args:
        pet_data: Dictionary containing PET data including initial_image
        spect_data: Dictionary containing SPECT data including initial_image and displacement

    Returns:
        Resampling operator (CompositionOperator with NiftyResampleOperator and NaNToZeroOperator)

    Raises:
        RuntimeError: If displacement field is not available
    """
    from setr.cil_extensions.operators import NaNToZeroOperator, NiftyResampleOperator

    if spect_data["displacement"] is None:
        raise RuntimeError(
            "Displacement field is required for SPECT to PET resampling. "
            "Ensure spect2pet.nii is available in the SPECT data directory."
        )

    logging.info("Setting up resampling operators with displacement field")

    return CompositionOperator(
        NiftyResampleOperator(
            pet_data["initial_image"],
            spect_data["initial_image"],
            spect_data["displacement"],
        ),
        NaNToZeroOperator(pet_data["initial_image"]),
    )


def attach_prior_hessian(prior):
    """Attach Hessian diagonal method to prior if it doesn't exist."""
    if not hasattr(prior, "inv_hessian_diag"):
        logging.warning("Prior doesn't have inv_hessian_diag method - using identity")

        def identity_hessian_diag(x, out=None):
            if out is None:
                return x.get_uniform_copy(1.0)
            out.fill(1.0)
            return out

        prior.inv_hessian_diag = identity_hessian_diag


def get_shift_operators(pet_data):
    """
    Set up the couch shift and image combining operators for multi-bed reconstruction.

    Args:
        pet_data: Dictionary containing multi-bed PET data with bed_positions

    Returns:
        uncombine_op: Operator to combine shifted bed images
        unshift_ops: List of operators to unshift each bed position
        choose_ops: List of operators to select specific bed position
    """
    from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
    from setr.cil_extensions.operators import (
        AdjointOperator,
        CouchShiftOperator,
        ImageCombineOperator,
    )

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

    # Create combine operator
    combine_op = ImageCombineOperator(EnhancedBlockDataContainer(*shifted_images))

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


def get_sensitivity_from_subset_objs(obj_funs):
    # get subset_sensitivity BDC for preconditioner
    for j, obj_fun in enumerate(obj_funs):
        if j == 0:
            sens = obj_fun.get_subset_sensitivity(0)
        else:
            sens += obj_fun.get_subset_sensitivity(0)
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    return sens


def get_sensitivities_from_subset_objs(obj_funs):
    # get subset_sensitivity BDC for preconditioner
    sens_list = []
    for obj_fun in obj_funs:
        sens = obj_fun.get_subset_sensitivity(0)
        sens = sens.maximum(0)
        sens_list.append(sens)
    return sens_list
