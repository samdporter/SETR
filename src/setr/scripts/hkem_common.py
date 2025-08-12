"""Shared utilities for HKEM (run_hkem_*) reconstruction scripts.

Contains functions that should be identical across HKEM scripts.
"""

import logging
import os

from sirf.STIR import ImageData

from setr.kernel.stir import STIRKernelOperator


def get_attn_and_normalise(args):
    """Get attention and normalization (identical in run_hkem_1bpos.py and run_hkem_1bpos_my_kem.py)."""
    # Use attenuation map
    result = ImageData(os.path.join(args.data_path, "umap_zoomed.hv"))
    result += (-result).max()
    result /= result.max()
    return result


def get_kernel_hyperparams(args):
    """Return kernel hyperparameters dictionary (standardized version).

    Based on the most complete version from run_hkem_1bpos.py.
    The other scripts had inconsistent/missing parameters.
    """
    return {
        "num_neighbours": args.num_neighbours,
        "num_non_zero_features": args.num_non_zero_features,
        "sigma_anatomical": args.sigma_anatomical,
        "sigma_emission": args.sigma_emission,
        "sigma_distance_anatomical": args.sigma_distance_anatomical,
        "sigma_distance_emission": args.sigma_distance_emission,
        "normalize_features": args.normalize_features,
        "normalize_kernel": args.normalize_kernel,
        "use_mask": args.use_mask,
        "mask_k": args.mask_k,
        "recalc_mask": args.recalc_mask,
        "distance_weighting": args.distance_weighting,
        "hybrid": args.hybrid,
        "only_2D": args.only_2D,  # Use 2D kernels only
    }


def get_kernel_operator(args, guide_image, template_image, template_sinogram, hyperparams):
    """Set up the STIR kernel operator."""
    logging.info("Setting up STIR kernel operator")

    return STIRKernelOperator(
        template_image,
        template_sinogram,
        guide_image,
        num_neighbours=hyperparams["num_neighbours"],
        num_non_zero_features=hyperparams["num_non_zero_features"],
        sigma_m=hyperparams["sigma_anatomical"],
        sigma_p=hyperparams["sigma_emission"],
        sigma_dm=hyperparams["sigma_distance_anatomical"],
        sigma_dp=hyperparams["sigma_distance_emission"],
        only_2D=hyperparams["only_2D"],
        hybrid=hyperparams["hybrid"],
    )


def run_kosmaposl(args, data, guidance, hyperparams):
    """Run KOSMAPOSL reconstruction using SIRF's KOSMAPOSLReconstructor.

    Args:
        args: Configuration arguments
        data: Dictionary containing acquisition data, normalisation, and additive
        guidance: Guidance image for anatomical prior
        hyperparams: Dictionary of kernel hyperparameters

    Returns:
        output_alpha: Final alpha estimates
        output_x: Final kernelised image
    """
    import logging

    from sirf.STIR import (
        AcquisitionSensitivityModel,
        KOSMAPOSLReconstructor,
        make_Poisson_loglikelihood,
    )

    from setr.utils.sirf import get_pet_am, get_spect_am

    # Get acquisition model
    if args.modality.upper() == "PET":
        am = get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.gauss_fwhm)
    else:
        am = get_spect_am(data, args.spect_res, True, args.gauss_fwhm)

    # Set up acquisition model
    if args.modality.upper() == "SPECT":
        data["normalisation"] = data["acquisition_data"].get_uniform_copy(1)

    am.set_acquisition_sensitivity(AcquisitionSensitivityModel(data["normalisation"]))
    am.set_additive_term(data["additive"])

    # Set up objective function
    obj_fun = make_Poisson_loglikelihood(data["acquisition_data"])
    obj_fun.set_acquisition_model(am)

    # Set up reconstructor
    recon = KOSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(args.num_subsets)
    recon.set_num_subiterations(args.num_subsets * args.num_epochs)
    recon.set_anatomical_prior(guidance)

    # Set kernel parameters
    recon.set_num_neighbours(hyperparams["num_neighbours"])
    recon.set_num_non_zero_features(hyperparams["num_non_zero_features"])
    recon.set_sigma_m(hyperparams["sigma_anatomical"])
    recon.set_sigma_p(hyperparams["sigma_emission"])
    recon.set_sigma_dm(hyperparams["sigma_distance_anatomical"])
    recon.set_sigma_dp(hyperparams["sigma_distance_emission"])
    recon.set_only_2D(not args.use_3d)
    recon.set_hybrid(hyperparams["hybrid"])

    recon.enable_output()
    recon.set_save_interval(args.num_subsets)
    recon.set_output_filename_prefix(os.path.join(args.output_path, "kosmaposl"))

    logging.info("Setting up KOSMAPOSL reconstruction...")
    current_alpha = data["initial_image"].get_uniform_copy(1)
    recon.set_up(current_alpha)
    recon.set_current_estimate(current_alpha)

    logging.info("Running KOSMAPOSL reconstruction...")
    recon.reconstruct(current_alpha)

    output_alpha = recon.get_current_estimate()
    output_x = recon.compute_kernelised_image(output_alpha, output_alpha)

    # Save results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x
