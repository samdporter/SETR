#!/usr/bin/env python3
"""Test script for debugging diverging reconstructions in cluster environment.

This script loads data and functions from run_dtnv_2bpos.py setup, then evaluates
objective functions and gradients on various test images including
initial images and scaled ellipsoid phantoms.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cil.optimisation.functions import SumFunction

# Add the src directory to the path so we can import setr modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.scripts.common import (
    configure_logging,
    get_resampling_operators,
    get_shift_operators,
    init_run_env,
)
from setr.scripts.dtnv_common import (
    apply_gradient_energy_scaling,
    get_prior,
    gradient_energy_scale_sirf,
    normalise_kappa_squares,
)
from setr.utils import (
    get_pet_am,
    get_pet_data_multiple_bed_pos,
    get_spect_am,
    get_spect_data,
)
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_filters


def prepare_data(args):
    """Prepare PET and SPECT data - identical to run_dtnv_2bpos.py"""
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    umap = pet_data["attenuation"]
    umap += (-umap).max()
    umap /= umap.max()
    spect_data = get_spect_data(args.spect_data_path)

    # Apply filters to initial images
    cyl, gauss = get_filters(fwhms=(20, 20, 20))

    gauss.apply(spect_data["initial_image"])
    gauss.apply(pet_data["initial_image"])
    cyl.apply(pet_data["initial_image"])

    # Set delta if not provided
    if args.delta is None:
        args.delta = max(
            pet_data["initial_image"].max() / 1e4,
            spect_data["initial_image"].max() / 1e4,
        ) * min(args.alpha, args.beta)

    initial_estimates = EnhancedBlockDataContainer(
        pet_data["initial_image"], spect_data["initial_image"]
    )

    # Save initial images
    for i, image in enumerate(initial_estimates.containers):
        image.write(os.path.join(args.output_path, f"initial_image_{i}.hv"))

    return umap, pet_data, spect_data, initial_estimates


def get_data_fidelity(args, pet_data, spect_data, uncombine_op, unshift_ops, choose_ops):
    """Set up data fidelity functions - adapted from run_dtnv_2bpos.py"""
    from cil.optimisation.functions import OperatorCompositionFunction
    from cil.optimisation.operators import CompositionOperator
    from sirf.contrib.partitioner import partitioner

    from setr.scripts.common import get_sensitivity_from_subset_objs
    from setr.scripts.dtnv_common import (
        compute_kappa_squared_image_from_partitioned_objective,
        get_block_objective,
        get_s_inv_from_subset_objs,
    )

    # PET acquisition model function
    def get_pet_am_with_res():
        return get_pet_am(
            not args.no_gpu,
            gauss_fwhm=args.pet_gauss_fwhm,
        )

    def get_spect_am_with_res():
        return get_spect_am(
            spect_data,
            res=args.spect_res,
            keep_all_views_in_cache=args.stop_keep_all_views_in_cache,
            gauss_fwhm=args.spect_gauss_fwhm,
            attenuation=True,
        )

    print("Partitioning PET data...")
    # Partition PET by bed
    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=args.num_subsets[0],
            mode="staggered",
            create_acq_model=get_pet_am_with_res,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]
    print("Done partitioning PET data.")

    # Keep raw copies for κ before operator wrapping
    pet_dfs_raw = [list(df_list) for df_list in pet_dfs]

    print("Setting up PET subset objectives...")
    # Set up subset objs on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    print("Partitioning SPECT data...")
    # Partition SPECT
    spect_dfs = partitioner.data_partition(
        spect_data["acquisition_data"],
        spect_data["additive"] if args.use_scatter else spect_data["additive"].get_uniform_copy(0),
        spect_data["acquisition_data"].get_uniform_copy(1),
        num_batches=args.num_subsets[1],
        mode="staggered",
        create_acq_model=get_spect_am_with_res,
    )[2]
    print("Done partitioning SPECT data.")

    print("Setting up SPECT subset objectives...")
    for obj_fun in spect_dfs:
        obj_fun.set_up(spect_data["initial_image"])

    print("Done setting up SPECT subset objectives.")
    # Compute κ² images before op wrapping
    pet_kappa_bed_sq = []
    for df_list, suffix in zip(pet_dfs_raw, pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        pet_kappa_bed_sq.append(
            compute_kappa_squared_image_from_partitioned_objective(df_list, tmpl)
        )

    # Add across beds
    pet_kappa_sq = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(pet_kappa_bed_sq[i]) for i, unshift_op in enumerate(unshift_ops)]
        )
    )

    spect_kappa_sq = compute_kappa_squared_image_from_partitioned_objective(
        spect_dfs, spect_data["initial_image"]
    )

    # Compute sensitivities
    pet_sens = [get_sensitivity_from_subset_objs(df) for df in pet_dfs]
    spect_s_inv = get_s_inv_from_subset_objs(spect_dfs, spect_data["initial_image"])

    # Combine PET sensitivities
    pet_sens_combined = uncombine_op.adjoint(
        EnhancedBlockDataContainer(
            *[unshift_op.adjoint(s) for unshift_op, s in zip(unshift_ops, pet_sens)]
        )
    )
    pet_s_inv = pet_sens_combined.clone()
    pet_sens_array = pet_sens_combined.as_array()
    pet_s_inv.fill(np.reciprocal(pet_sens_array, where=pet_sens_array != 0))
    cyl, _ = get_filters()
    cyl.apply(pet_s_inv)

    s_inv = EnhancedBlockDataContainer(pet_s_inv, spect_s_inv)

    # Wrap PET objectives with operators
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(unshift_ops[i], choose_ops[i], uncombine_op),
            )

    # Flatten beds
    pet_combined_dfs = [df for bed in pet_dfs for df in bed]

    # Create block objectives
    pet_dfs_block = [
        get_block_objective(
            pet_data["initial_image"],
            spect_data["initial_image"],
            df,
            order=0,
        )
        for df in pet_combined_dfs
    ]
    spect_dfs_block = [
        get_block_objective(
            spect_data["initial_image"],
            pet_data["initial_image"],
            obj_fun,
            order=1,
        )
        for obj_fun in spect_dfs
    ]

    all_funs = pet_dfs_block + spect_dfs_block
    kappas = EnhancedBlockDataContainer(pet_kappa_sq, spect_kappa_sq)

    return all_funs, s_inv, kappas


def create_ellipsoid_phantom(template_image, center=None, radii=None, intensity=None):
    """Create an ellipsoid phantom with same shape and scaling as template image."""
    shape = template_image.shape

    if center is None:
        center = np.array(shape) // 2
    if radii is None:
        # Default to moderate sized ellipsoid (about 1/4 of image size)
        radii = np.array(shape) // 8
    if intensity is None:
        intensity = template_image.max()

    # Create coordinate grids
    coords = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]

    # Calculate ellipsoid equation
    ellipsoid_eq = 0
    for i in range(3):
        ellipsoid_eq += ((coords[i] - center[i]) / radii[i]) ** 2

    # Create ellipsoid mask
    mask = ellipsoid_eq <= 1.0

    # Create phantom
    phantom = template_image.get_uniform_copy(0)
    phantom_array = phantom.as_array()
    phantom_array[mask] = intensity
    phantom.fill(phantom_array)

    return phantom


def create_test_images(initial_estimates):
    """Create various test images for function evaluation."""
    test_images = {"initial": initial_estimates.copy()}

    # 2. Ellipsoid phantoms scaled to match initial image intensities
    pet_max = initial_estimates.containers[0].max()
    spect_max = initial_estimates.containers[1].max()

    # Central ellipsoid
    pet_ellipsoid = create_ellipsoid_phantom(initial_estimates.containers[0], intensity=pet_max)
    spect_ellipsoid = create_ellipsoid_phantom(initial_estimates.containers[1], intensity=spect_max)
    test_images["ellipsoid_central"] = EnhancedBlockDataContainer(pet_ellipsoid, spect_ellipsoid)

    # Zero background
    test_images["uniform_zero"] = EnhancedBlockDataContainer(
        initial_estimates.containers[0].get_uniform_copy(0),
        initial_estimates.containers[1].get_uniform_copy(0),
    )

    return test_images


def evaluate_functions(test_image, data_fidelity_funs, prior_funs):
    """Evaluate objective functions and their gradients on test image."""
    results = {"data_fidelity_individual": {}}

    for i, fun in enumerate(data_fidelity_funs):
        value = fun(test_image)
        gradient = fun.gradient(test_image)
        results["data_fidelity_individual"][f"subset_{i}"] = {
            "value": value,
            "gradient": gradient.copy(),
            "gradient_norm": gradient.norm(),
        }

    # Evaluate individual prior functions
    results["prior_individual"] = {}
    if prior_funs is not None:
        for i, fun in enumerate(prior_funs):
            value = fun(test_image)
            gradient = fun.gradient(test_image)
            results["prior_individual"][f"prior_{i}"] = {
                "value": value,
                "gradient": gradient.copy(),
                "gradient_norm": gradient.norm(),
            }

    # Evaluate combined functions
    data_sum = SumFunction(*data_fidelity_funs)
    results["data_fidelity_total"] = {
        "value": data_sum(test_image),
        "gradient": data_sum.gradient(test_image).copy(),
        "gradient_norm": data_sum.gradient(test_image).norm(),
    }

    if prior_funs is not None:
        prior_sum = SumFunction(*prior_funs)
        results["prior_total"] = {
            "value": prior_sum(test_image),
            "gradient": prior_sum.gradient(test_image).copy(),
            "gradient_norm": prior_sum.gradient(test_image).norm(),
        }

        # Total objective
        total_obj = SumFunction(data_sum, prior_sum)
        results["total_objective"] = {
            "value": total_obj(test_image),
            "gradient": total_obj.gradient(test_image).copy(),
            "gradient_norm": total_obj.gradient(test_image).norm(),
        }
    else:
        results["total_objective"] = results["data_fidelity_total"].copy()

    return results


def plot_coronal_slice(image_data, title, save_path, central_slice=None):
    """Plot central coronal slice of image data."""
    if hasattr(image_data, "containers"):
        # Block data container
        fig, axes = plt.subplots(1, len(image_data.containers), figsize=(12, 5))
        if len(image_data.containers) == 1:
            axes = [axes]

        for i, container in enumerate(image_data.containers):
            array = container.as_array()
            if central_slice is None:
                y_center = array.shape[1] // 2
            else:
                y_center = central_slice

            slice_data = array[:, y_center, :]
            im = axes[i].imshow(slice_data, aspect="auto", origin="lower")
            axes[i].set_title(f"{title} - Modal {i}")
            axes[i].set_xlabel("Z")
            axes[i].set_ylabel("X")
            plt.colorbar(im, ax=axes[i])
    else:
        # Single image
        fig, ax = plt.subplots(figsize=(8, 6))
        array = image_data.as_array()
        if central_slice is None:
            y_center = array.shape[1] // 2
        else:
            y_center = central_slice

        slice_data = array[:, y_center, :]
        im = ax.imshow(slice_data, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_gradients(results, test_name: str, out_dir: str):
    """Plot gradients for all available components in `results`."""
    # 1) Data fidelity: individual subsets
    if "data_fidelity_individual" in results:
        for k, v in results["data_fidelity_individual"].items():
            plot_coronal_slice(
                v["gradient"],
                f"Gradient ({k}): {test_name}",
                os.path.join(out_dir, f"grad_{test_name}_{k}.png"),
            )

    # 2) Prior: individual terms (if present)
    if "prior_individual" in results:
        for k, v in results["prior_individual"].items():
            plot_coronal_slice(
                v["gradient"],
                f"Gradient ({k}): {test_name}",
                os.path.join(out_dir, f"grad_{test_name}_{k}.png"),
            )

    # 3) Totals
    plot_coronal_slice(
        results["data_fidelity_total"]["gradient"],
        f"Gradient (data_total): {test_name}",
        os.path.join(out_dir, f"grad_{test_name}_data_total.png"),
    )

    if "prior_total" in results:
        plot_coronal_slice(
            results["prior_total"]["gradient"],
            f"Gradient (prior_total): {test_name}",
            os.path.join(out_dir, f"grad_{test_name}_prior_total.png"),
        )

    plot_coronal_slice(
        results["total_objective"]["gradient"],
        f"Gradient (total_objective): {test_name}",
        os.path.join(out_dir, f"grad_{test_name}_total.png"),
    )


def test_shift_operators(initial_estimates, bo, output_path):
    """Test the shift operators by visualizing transformations through image spaces.

    Args:
        initial_estimates: EnhancedBlockDataContainer with PET and SPECT initial images
        bo: BlockOperator for transforming between spaces
        output_path: Directory to save output images
    """
    logging.info("Testing shift operators...")

    # 1. Save original images (in modality-specific spaces)
    logging.info("Saving original images...")
    plot_coronal_slice(
        initial_estimates,
        "Original Images (Modality-Specific Spaces)",
        os.path.join(output_path, "shift_test_01_original.png"),
    )

    # Also save individual modalities
    plot_coronal_slice(
        initial_estimates.containers[0],
        "Original PET Image",
        os.path.join(output_path, "shift_test_01a_original_pet.png"),
    )
    plot_coronal_slice(
        initial_estimates.containers[1],
        "Original SPECT Image",
        os.path.join(output_path, "shift_test_01b_original_spect.png"),
    )

    # 2. Transform to combined (prior) image space
    logging.info("Transforming to combined image space...")
    combined = bo.direct(initial_estimates)

    plot_coronal_slice(
        combined,
        "Images in Combined (Prior) Space - Forward Transform",
        os.path.join(output_path, "shift_test_02_combined_forward.png"),
    )

    # Save individual components of combined space
    plot_coronal_slice(
        combined.containers[0],
        "Combined Space - PET Component",
        os.path.join(output_path, "shift_test_02a_combined_pet.png"),
    )
    plot_coronal_slice(
        combined.containers[1],
        "Combined Space - SPECT Component",
        os.path.join(output_path, "shift_test_02b_combined_spect.png"),
    )

    # 3. Transform back to original spaces (adjoint operation)
    logging.info("Transforming back to original spaces...")
    reconstructed = bo.adjoint(combined)

    plot_coronal_slice(
        reconstructed,
        "Reconstructed Images (Back to Original Spaces)",
        os.path.join(output_path, "shift_test_03_reconstructed.png"),
    )

    # Save individual reconstructed modalities
    plot_coronal_slice(
        reconstructed.containers[0],
        "Reconstructed PET Image",
        os.path.join(output_path, "shift_test_03a_reconstructed_pet.png"),
    )
    plot_coronal_slice(
        reconstructed.containers[1],
        "Reconstructed SPECT Image",
        os.path.join(output_path, "shift_test_03b_reconstructed_spect.png"),
    )

    # 4. Compute and visualize differences (round-trip error)
    logging.info("Computing round-trip transformation errors...")

    # Calculate differences
    pet_diff = initial_estimates.containers[0].copy()
    pet_diff -= reconstructed.containers[0]

    spect_diff = initial_estimates.containers[1].copy()
    spect_diff -= reconstructed.containers[1]

    difference = EnhancedBlockDataContainer(pet_diff, spect_diff)

    plot_coronal_slice(
        difference,
        "Round-trip Error (Original - Reconstructed)",
        os.path.join(output_path, "shift_test_04_roundtrip_error.png"),
    )

    plot_coronal_slice(
        pet_diff, "PET Round-trip Error", os.path.join(output_path, "shift_test_04a_pet_error.png")
    )
    plot_coronal_slice(
        spect_diff,
        "SPECT Round-trip Error",
        os.path.join(output_path, "shift_test_04b_spect_error.png"),
    )

    # 5. Generate numerical summary
    logging.info("Generating shift operator test summary...")

    # Compute norms and relative errors
    original_pet_norm = initial_estimates.containers[0].norm()
    original_spect_norm = initial_estimates.containers[1].norm()

    combined_pet_norm = combined.containers[0].norm()
    combined_spect_norm = combined.containers[1].norm()

    reconstructed_pet_norm = reconstructed.containers[0].norm()
    reconstructed_spect_norm = reconstructed.containers[1].norm()

    pet_error_norm = pet_diff.norm()
    spect_error_norm = spect_diff.norm()

    pet_relative_error = (
        pet_error_norm / original_pet_norm if original_pet_norm > 0 else float("inf")
    )
    spect_relative_error = (
        spect_error_norm / original_spect_norm if original_spect_norm > 0 else float("inf")
    )

    # Check shapes
    original_shapes = [c.shape for c in initial_estimates.containers]
    combined_shapes = [c.shape for c in combined.containers]
    reconstructed_shapes = [c.shape for c in reconstructed.containers]

    with open(os.path.join(output_path, "shift_operator_test_summary.txt"), "w") as f:
        f.write("Shift Operator Test Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Image Shapes:\n")
        f.write(f"Original PET shape:      {original_shapes[0]}\n")
        f.write(f"Original SPECT shape:    {original_shapes[1]}\n")
        f.write(f"Combined PET shape:      {combined_shapes[0]}\n")
        f.write(f"Combined SPECT shape:    {combined_shapes[1]}\n")
        f.write(f"Reconstructed PET shape: {reconstructed_shapes[0]}\n")
        f.write(f"Reconstructed SPECT shape: {reconstructed_shapes[1]}\n\n")

        f.write("Image Norms:\n")
        f.write(f"Original PET norm:      {original_pet_norm:.6e}\n")
        f.write(f"Original SPECT norm:    {original_spect_norm:.6e}\n")
        f.write(f"Combined PET norm:      {combined_pet_norm:.6e}\n")
        f.write(f"Combined SPECT norm:    {combined_spect_norm:.6e}\n")
        f.write(f"Reconstructed PET norm: {reconstructed_pet_norm:.6e}\n")
        f.write(f"Reconstructed SPECT norm: {reconstructed_spect_norm:.6e}\n\n")

        f.write("Round-trip Errors:\n")
        f.write(f"PET absolute error norm:     {pet_error_norm:.6e}\n")
        f.write(f"SPECT absolute error norm:   {spect_error_norm:.6e}\n")
        f.write(f"PET relative error:          {pet_relative_error:.6e}\n")
        f.write(f"SPECT relative error:        {spect_relative_error:.6e}\n\n")

        # Quality checks
        f.write("Quality Assessment:\n")
        if pet_relative_error < 1e-10:
            f.write("✓ PET round-trip error is excellent (< 1e-10)\n")
        elif pet_relative_error < 1e-6:
            f.write("✓ PET round-trip error is good (< 1e-6)\n")
        elif pet_relative_error < 1e-3:
            f.write("⚠ PET round-trip error is acceptable (< 1e-3)\n")
        else:
            f.write("✗ PET round-trip error is large (>= 1e-3)\n")

        if spect_relative_error < 1e-10:
            f.write("✓ SPECT round-trip error is excellent (< 1e-10)\n")
        elif spect_relative_error < 1e-6:
            f.write("✓ SPECT round-trip error is good (< 1e-6)\n")
        elif spect_relative_error < 1e-3:
            f.write("⚠ SPECT round-trip error is acceptable (< 1e-3)\n")
        else:
            f.write("✗ SPECT round-trip error is large (>= 1e-3)\n")

        # Check if shapes are preserved
        if original_shapes == reconstructed_shapes:
            f.write("✓ Image shapes preserved through transformation\n")
        else:
            f.write("✗ Image shapes NOT preserved through transformation\n")

    logging.info("Shift operator test completed successfully!")

    return {
        "original": initial_estimates,
        "combined": combined,
        "reconstructed": reconstructed,
        "pet_relative_error": pet_relative_error,
        "spect_relative_error": spect_relative_error,
    }


def main(args):
    """Main function for testing and debugging."""
    configure_logging()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    save_args(args, "args.csv")

    logging.info("Starting function debugging test...")

    # Prepare data
    logging.info("Loading data...")
    umap, pet_data, spect_data, initial_estimates = prepare_data(args)

    # Set up operators
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)
    spect2pet = get_resampling_operators(pet_data, spect_data)

    # Set up data fidelity
    logging.info("Setting up data fidelity functions...")
    all_funs, s_inv, kappas = get_data_fidelity(
        args, pet_data, spect_data, uncombine_op, unshift_ops, choose_ops
    )

    # Set up block operator and scaling
    from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator

    bo = BlockOperator(
        IdentityOperator(pet_data["initial_image"]),
        ZeroOperator(spect_data["initial_image"], pet_data["initial_image"]),
        ZeroOperator(pet_data["initial_image"]),
        spect2pet,
        shape=(2, 2),
    )

    shift_test_results = test_shift_operators(initial_estimates, bo, args.output_path)

    # Apply scaling
    kappas = normalise_kappa_squares(bo.direct(kappas)) if kappas else None
    combined = bo.direct(initial_estimates)
    scale = gradient_energy_scale_sirf(
        combined[0],
        combined[1],
        mask=None,
        kappa_pet=kappas.containers[0] if kappas else None,
        kappa_spect=kappas.containers[1] if kappas else None,
    )
    apply_gradient_energy_scaling(args, scale)

    # Set up priors
    if not args.no_prior:
        logging.info("Setting up prior functions...")
        priors_list = get_prior(args, umap, combined, bo, kappas)
        # Apply hessian attachment
        from setr.scripts.common import attach_prior_hessian

        for i, p in enumerate(priors_list):
            attach_prior_hessian(priors_list[i])
    else:
        priors_list = None

    # Save kappa and sensitivity images
    for i, image in enumerate(s_inv.containers):
        image.write(os.path.join(args.output_path, f"s_inv_{i}.hv"))

    if kappas:
        for i, image in enumerate(kappas.containers):
            image.write(os.path.join(args.output_path, f"kappa_sq_{i}.hv"))

    # Create test images
    logging.info("Creating test images...")
    test_images = create_test_images(initial_estimates)

    # Save test images
    for name, test_img in test_images.items():
        for i, container in enumerate(test_img.containers):
            container.write(os.path.join(args.output_path, f"test_image_{name}_{i}.hv"))
        plot_coronal_slice(
            test_img,
            f"Test Image: {name}",
            os.path.join(args.output_path, f"test_image_{name}.png"),
        )

    # Evaluate functions on all test images
    logging.info("Evaluating functions on test images...")
    all_results = {}

    for test_name, test_img in test_images.items():
        logging.info(f"Testing on {test_name}...")

        # Evaluate objective functions
        results = evaluate_functions(test_img, all_funs, priors_list)

        all_results[test_name] = results

        # Plot gradients
        plot_all_gradients(results, test_name, args.output_path)

    # Generate summary report
    logging.info("Generating summary report...")
    with open(os.path.join(args.output_path, "function_summary.txt"), "w") as f:
        f.write("Function Evaluation Debug Report\n")
        f.write("=" * 50 + "\n\n")

        for test_name, results in all_results.items():
            f.write(f"Test Image: {test_name}\n")
            f.write("-" * 30 + "\n")

            # Data fidelity summary
            f.write(f"Data Fidelity Total Value: {results['data_fidelity_total']['value']:.6e}\n")
            f.write(
                f"Data Fidelity Gradient Norm: {results['data_fidelity_total']['gradient_norm']:.6e}\n"
            )

            if "prior_total" in results:
                f.write(f"Prior Total Value: {results['prior_total']['value']:.6e}\n")
                f.write(f"Prior Gradient Norm: {results['prior_total']['gradient_norm']:.6e}\n")

            f.write(f"Total Objective Value: {results['total_objective']['value']:.6e}\n")
            f.write(f"Total Gradient Norm: {results['total_objective']['gradient_norm']:.6e}\n")

            # Check for issues
            total_val = results["total_objective"]["value"]
            grad_norm = results["total_objective"]["gradient_norm"]

            if not np.isfinite(total_val):
                f.write("WARNING: Non-finite objective value!\n")
            if not np.isfinite(grad_norm):
                f.write("WARNING: Non-finite gradient norm!\n")
            if grad_norm > 1e10:
                f.write("WARNING: Very large gradient norm!\n")

            f.write("\n")

    logging.info("Function debugging test complete!")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = argparse.Namespace(**config)

    msg = init_run_env(args)

    main(args)
