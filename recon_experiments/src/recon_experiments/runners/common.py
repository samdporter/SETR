"""Universal utilities shared across ALL SETR reconstruction scripts."""

import argparse
import logging
import os

from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import OperatorCompositionFunction
from cil.optimisation.operators import (
    BlockOperator,
    IdentityOperator,
    ZeroOperator,
)

from recon_core.cil_extensions.operators import AdjointOperator, NiftyResampleOperator
from recon_core.cil_extensions.framework import EnhancedBlockDataContainer


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

    if hasattr(final_image, "containers"):
        for i, container in enumerate(final_image.containers):
            container.write(os.path.join(args.output_path, f"final_image_{i}.hv"))
    else:
        final_image.write(os.path.join(args.output_path, "final_image.hv"))


def get_resampling_operators(*inputs):
    """
    Set up resampling operators for SPECT images to PET images.

    Args:
        pet_data: Dictionary containing PET data including initial_image
        spect_data: Dictionary containing SPECT data including initial_image and displacement

    Returns:
        Resampling operator (direct NiftyResampleOperator using no-zoom displacement)

    Raises:
        RuntimeError: If displacement field is not available
    """

    if len(inputs) == 2:
        pet_data, spect_data = inputs
        args = None
    elif len(inputs) == 3:
        args, pet_data, spect_data = inputs
    else:
        raise TypeError("get_resampling_operators expects (pet_data, spect_data, ...) or (args, pet_data, spect_data, ...)")

    no_zoom_available = spect_data.get("no_zoom_displacement") is not None
    if not no_zoom_available:
        raise RuntimeError(
            "No SPECT→PET no-zoom displacement field found. Expected files named spect2pet_nozoom*.nii"
        )

    logging.info("Setting up resampling operators with direct displacement (no zoom)")
    resampler = NiftyResampleOperator(
        reference=pet_data["initial_image"],
        floating=spect_data["initial_image"],
        transform=spect_data["no_zoom_displacement"],
    )
    return resampler


def get_pet_to_spect_operator(spect2pet):
    """Return the shared-PET-grid to native-SPECT operator."""
    return AdjointOperator(spect2pet)


def build_shared_initial_estimates(pet_initial, spect_initial, spect2pet):
    """Map the SPECT initial image onto PET space and build the shared-grid state."""
    spect_shared = spect2pet.direct(spect_initial)
    return EnhancedBlockDataContainer(pet_initial, spect_shared)


def save_native_spect_image(shared_spect, pet_to_spect, output_path):
    """Export a PET-grid shared SPECT image back to native SPECT space."""
    native_spect = pet_to_spect.direct(shared_spect)
    native_spect.write(output_path)


def get_shift_operators(pet_data, path=""):
    """
    Set up the couch shift and image combining operators for multi-bed reconstruction.

    Args:
        pet_data: Dictionary containing multi-bed PET data with bed_positions

    Returns:
        uncombine_op: Operator to combine shifted bed images
        unshift_ops: List of operators to unshift each bed position
        choose_ops: List of operators to select specific bed position
    """
    from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
    from recon_core.cil_extensions.operators import (
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
        CouchShiftOperator(
            pet_data["bed_positions"][suffix]["template_image"], pet_shift, path=path
        )
        for suffix, pet_shift in zip(suffixes, pet_shifts)
    ]

    # Create shifted images
    shifted_images = [
        op.direct(pet_data["bed_positions"][suffix]["template_image"])
        for suffix, op in zip(suffixes, shift_ops)
    ]

    # Create combine operator
    combine_op = ImageCombineOperator(EnhancedBlockDataContainer(*shifted_images))
    pet_data["combine_operator"] = combine_op
    pet_data["shift_operators"] = shift_ops

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


def apply_combine_sensitivities(pet_data, per_bed_sensitivities):
    """
    Attach per-bed sensitivities to the stored ImageCombineOperator for weighted overlap.
    """
    combine_op = pet_data.get("combine_operator", None)
    shift_ops = pet_data.get("shift_operators", None)

    if combine_op is None or shift_ops is None or per_bed_sensitivities is None:
        return

    shifted_sens = [
        shift_op.direct(sens) for shift_op, sens in zip(shift_ops, per_bed_sensitivities)
    ]
    # Resample sensitivities onto the combine operator geometry
    resampled_sens = combine_op.resample_op.direct(
        EnhancedBlockDataContainer(*shifted_sens)
    )
    combine_op.set_sensitivities(resampled_sens)


def get_sensitivity_from_subset_objs(obj_funs, adjoint_operator=None):
    # get subset_sensitivity BDC for preconditioner
    for j, obj_fun in enumerate(obj_funs):
        # Extract underlying function if wrapped in OperatorCompositionFunction
        obj_fn = obj_fun.function if isinstance(obj_fun, OperatorCompositionFunction) else obj_fun
        subset_sens = obj_fn.get_subset_sensitivity(0)
        # SIRF can emit small negative sensitivity values; clamp on creation.
        subset_sens.maximum(0, out=subset_sens)
        if j == 0:
            sens = subset_sens
        else:
            sens += subset_sens
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    if adjoint_operator is not None:
        sens = adjoint_operator.adjoint(sens)
    sens = sens.maximum(0)
    return sens


def get_sensitivities_from_subset_objs(obj_funs):
    # get subset_sensitivity BDC for preconditioner
    sens_list = []
    for obj_fun in obj_funs:
        # Extract underlying function if wrapped in OperatorCompositionFunction
        obj_fn = obj_fun.function if isinstance(obj_fun, OperatorCompositionFunction) else obj_fun
        sens = obj_fn.get_subset_sensitivity(0)
        sens = sens.maximum(0)
        sens_list.append(sens)
    return sens_list
