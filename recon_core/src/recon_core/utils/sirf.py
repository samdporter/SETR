import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from cil.framework import BlockDataContainer
from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction
from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator
from sirf.contrib import partitioner
from sirf.Reg import NiftiImageData3DDisplacement, AffineTransformation
from sirf.STIR import (
    AcquisitionData,
    AcquisitionModelUsingMatrix,
    AcquisitionModelUsingParallelproj,
    AcquisitionModelUsingRayTracingMatrix,
    ImageData,
    SeparableGaussianImageFilter,
    SPECTUBMatrix,
    TruncateToCylinderProcessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_pet_am(
    gpu=True,
    gauss_fwhm=None,
):
    """Create a PET acquisition model with optional Gaussian filtering.

    Args:
        gpu: If True, use GPU-accelerated parallel projection. If False, use
            CPU-based ray tracing matrix.
        gauss_fwhm: Optional Gaussian FWHM values for point spread function
            modeling. If provided, applies separable Gaussian image filter.

    Returns:
        AcquisitionModel: Configured PET acquisition model.
    """
    if gpu:
        pet_am = AcquisitionModelUsingParallelproj()
    else:
        pet_am = AcquisitionModelUsingRayTracingMatrix()
        pet_am.set_num_tangential_LORs(10)

    if gauss_fwhm:
        pet_psf = SeparableGaussianImageFilter()
        pet_psf.set_fwhms(gauss_fwhm)
        pet_am.set_image_data_processor(pet_psf)

    return pet_am


def get_spect_am(
    spect_data,
    res=None,
    keep_all_views_in_cache=True,
    gauss_fwhm=None,
    attenuation=True,
):
    """Create a SPECT acquisition model with collimator modeling and optional PSF.

    Args:
        spect_data: Dictionary containing SPECT data including attenuation image.
        res: Optional resolution parameters [collimator_slope, collimator_intercept,
            use_psf_correction]. If provided, enables analytical collimator resolution modeling.
        keep_all_views_in_cache: If True, cache all projection views in memory for
            faster computation.
        gauss_fwhm: Optional Gaussian FWHM values for additional point spread function
            modeling.
        attenuation: If True, apply attenuation correction using data from spect_data.

    Returns:
        AcquisitionModel: Configured SPECT acquisition model with UB matrix backend.
    """
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_keep_all_views_in_cache(keep_all_views_in_cache)
    if attenuation:
        try:
            spect_am_mat.set_attenuation_image(spect_data["attenuation"])
        except Exception as e:
            print("No attenuation data:", e)
    if res:
        spect_am_mat.set_resolution_model(*res)
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    if gauss_fwhm:
        spect_psf = SeparableGaussianImageFilter()
        spect_psf.set_fwhms(gauss_fwhm)
        spect_am.set_image_data_processor(spect_psf)
    return spect_am


def get_pet_data(path: str, load_sinos = True, suffix: str = "") -> dict:
    """
    Load PET data from the given path.

    This function always loads a template image and then attempts to load the
    initial image. If the initial image is not found, it creates a uniform copy
    of the template image (filled with ones).

    Args:
        path (str): Path to the data directory.
        suffix (str): Optional suffix appended to filenames.

    Returns:
        dict: A dictionary with keys: "acquisition_data", "additive",
        "normalisation", "attenuation", "template_image", "initial_image", and
        optionally "spect".
    """
    if load_sinos:
        pet_data = {
            "acquisition_data": AcquisitionData(os.path.join(path, f"prompts{suffix}.hs")),
            "additive": AcquisitionData(os.path.join(path, f"additive_term{suffix}.hs")),
            "normalisation": AcquisitionData(os.path.join(path, f"mult_factors{suffix}.hs")),
            "attenuation": ImageData(os.path.join(path, "umap_zoomed.hv")),
        }
    else:
        pet_data = {
            "attenuation": ImageData(os.path.join(path, "umap_zoomed.hv")),
        }

    # Always load the template image.
    template_img_path = os.path.join(path, "template_image.hv")
    try:
        pet_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load PET template image (%s)", str(e_template))
        raise RuntimeError("Unable to load PET template image.") from e_template

    # Try to load the initial image.
    initial_img_path = os.path.join(path, "initial_image.hv")
    try:
        pet_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning(
            "No PET initial image found (%s). Using uniform copy of template image.",
            str(e_initial),
        )
        pet_data["initial_image"] = pet_data["template_image"].get_uniform_copy(1)

    try:
        pet_data["spect"] = ImageData(os.path.join(path, "spect.hv"))
    except Exception as e_spect:
        logging.info("No SPECT guidance image found for PET: %s", str(e_spect))

    return pet_data


def get_pet_data_multiple_bed_pos(
    path: str,
    suffixes: List[str],
    tof: bool = False,
    load_sinos: bool = True,
) -> Dict[str, object]:
    """
    Load PET data for multiple bed positions.

    Returns a dict with:
    - "attenuation", "template_image", "initial_image", "spect" (optional)
    - "bed_positions": mapping suffix → dict with keys
        "acquisition_data", "additive", "normalisation",
        "template_image", "initial_image", "attenuation", "spect" (optional)
    """
    base = Path(path) / ("tof" if tof else "non_tof")

    def load_image(fp: Path, clamp: bool = True, required: bool = False) -> Optional[ImageData]:
        try:
            img = ImageData(str(fp))
            return img.maximum(0) if clamp else img
        except Exception:
            if required:
                logging.error("Failed to load required image %s", fp)
                raise
            return None

    def load_acq(fp: Path) -> AcquisitionData:
        return AcquisitionData(str(fp))

    # shared data
    pet_data: Dict[str, object] = {}
    pet_data["attenuation"] = load_image(base / "umap_zoomed.hv")
    pet_data["template_image"] = load_image(base / "template_image.hv", clamp=False, required=True)
    pet_data["initial_image"] = load_image(base / "initial_image.hv") or pet_data[
        "template_image"
    ].get_uniform_copy(1)
    pet_data["spect"] = load_image(base / "spect.hv", clamp=False)

    # per‐bed data
    beds: Dict[str, Dict[str, object]] = {}
    for suf in suffixes:
        bp = {}
        if load_sinos:
            bp["acquisition_data"] = load_acq(base / f"prompts{suf}.hs")
            bp["additive"] = load_acq(base / f"additive_term{suf}.hs")
            bp["normalisation"] = load_acq(base / f"mult_factors{suf}.hs")
        bp["template_image"] = load_image(
            base / f"template_image{suf}.hv", clamp=False, required=True
        )
        bp["initial_image"] = load_image(base / f"initial_image{suf}.hv") or bp[
            "template_image"
        ].get_uniform_copy(1)
        bp["attenuation"] = load_image(base / f"umap{suf}.hv")
        bp["spect"] = load_image(base / f"spect{suf}.hv", clamp=False)
        beds[suf] = bp

    pet_data["bed_positions"] = beds
    return pet_data


def load_zoom_factors(spect_dir):
    """
    Load previously saved zoom factors from file.

    Args:
        spect_dir: Directory containing the zoom factors file

    Returns:
        tuple: Zoom factors (z, y, x)
    """
    zoom_file_path = os.path.join(spect_dir, "spect_to_pet_zoom_factors.txt")

    if not os.path.exists(zoom_file_path):
        raise FileNotFoundError(f"Zoom factors file not found: {zoom_file_path}")

    with open(zoom_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#") and line:
                zoom_values = line.split()
                return (float(zoom_values[0]), float(zoom_values[1]), float(zoom_values[2]))

    raise ValueError("No zoom factors found in file")


def get_spect_data(path: str, load_sinos: bool = True, displacement_suffix: str = "nii") -> dict:
    """
    Load SPECT data from the given path.

    This function always loads a template image and then attempts to load the
    initial image. If the initial image is not found, it creates a uniform copy
    of the template image (filled with ones). Also, the attenuation image is flipped
    on the x-axis due to a known STIR bug.

    Args:
        path (str): Path to the data directory.
        load_sinos (bool): Whether to load acquisition/additive sinograms.
        displacement_suffix (str): "nii" for NIfTI displacement, otherwise treated as affine.

    Returns:
        dict: Keys may include: "acquisition_data", "additive", "attenuation",
              "template_image", "initial_image", "displacement", "no_zoom_displacement",
              "zoom_factors".
    """
    def _first_existing(candidates):
        for fn in candidates:
            fp = os.path.join(path, fn)
            if os.path.exists(fp):
                return fp
        return None

    def _load_transform(fp: str):
        if displacement_suffix == "nii":
            return NiftiImageData3DDisplacement(fp)
        return AffineTransformation(fp)

    spect_data: dict = {}

    # Sinograms
    if load_sinos:
        spect_data["acquisition_data"] = AcquisitionData(os.path.join(path, "peak.hs"))
        try:
            spect_data["additive"] = AcquisitionData(os.path.join(path, "scatter_dl.hs"))
        except Exception as e_scatter:
            logging.warning("No scatter data found (%s). Using zeros.", str(e_scatter))
            spect_data["additive"] = AcquisitionData(spect_data["acquisition_data"])
            spect_data["additive"].fill(0)

    # Attenuation (flip x-axis due to STIR bug) IS THIS STILL NEEDED?
    spect_data["attenuation"] = ImageData(os.path.join(path, "umap_zoomed.hv"))
    attn_arr = np.flip(spect_data["attenuation"].as_array(), axis=-1)
    spect_data["attenuation"].fill(attn_arr)

    # Template image (required)
    template_img_path = os.path.join(path, "template_image.hv")
    try:
        spect_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load SPECT template image (%s)", str(e_template))
        raise RuntimeError("Unable to load SPECT template image.") from e_template

    # Initial image (optional -> uniform template)
    initial_img_path = os.path.join(path, "initial_image.hv")
    try:
        spect_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning(
            "No SPECT initial image found (%s). Using uniform copy of template image.",
            str(e_initial),
        )
        spect_data["initial_image"] = spect_data["template_image"].get_uniform_copy(1)

    # Displacement fields (zoom)
    displacement_candidates = [
        f"spect2pet_zoom_nonrigid.{displacement_suffix}",
        f"spect2pet_zoom_rigid.{displacement_suffix}",
        f"spect2pet.{displacement_suffix}",
    ]
    displacement_path = _first_existing(displacement_candidates)
    if displacement_path is None:
        logging.warning("No SPECT displacement field found. Registration will not be available.")
        spect_data["displacement"] = None
    else:
        try:
            spect_data["displacement"] = _load_transform(displacement_path)
        except Exception as e_disp:
            logging.warning(
                "Failed to load SPECT displacement field from %s (%s). Registration will not be available.",
                displacement_path,
                str(e_disp),
            )
            spect_data["displacement"] = None

    # Displacement fields (no-zoom)
    no_zoom_candidates = [
        #f"spect2pet_nozoom_nonrigid_liver.{displacement_suffix}",
        f"spect2pet_nozoom_nonrigid.{displacement_suffix}",
        f"spect2pet_nozoom_rigid.{displacement_suffix}",
        f"spect2pet_nozoom.{displacement_suffix}",
    ]
    no_zoom_path = _first_existing(no_zoom_candidates)
    if no_zoom_path is None:
        logging.warning("No SPECT no-zoom displacement field found. Registration will not be available.")
        spect_data["no_zoom_displacement"] = None
    else:
        try:
            spect_data["no_zoom_displacement"] = _load_transform(no_zoom_path)
            print(f"Used no-zoom displacement field from {no_zoom_path}")
        except Exception as e_no_zoom:
            logging.warning(
                "Failed to load SPECT no-zoom displacement field from %s (%s). Registration will not be available.",
                no_zoom_path,
                str(e_no_zoom),
            )
            spect_data["no_zoom_displacement"] = None

    # Zoom factors (optional)
    try:
        spect_data["zoom_factors"] = load_zoom_factors(path)
    except Exception as e_zoom:
        logging.warning(
            "No SPECT zoom factors found (%s). Zooming will not be available.",
            str(e_zoom),
        )
        spect_data["zoom_factors"] = None

    return spect_data



def create_spect_uniform_image(sinogram, origin=None, dims=None):
    """
    Create a uniform image for SPECT data based on the sinogram dimensions.
    Adjusts the z-direction voxel size and image dimensions to create a template
    image.

    Args:
        sinogram (AcquisitionData): The SPECT sinogram.
        origin (tuple, optional): The origin of the image. Defaults to (0, 0, 0)
            if not provided.

    Returns:
        ImageData: A uniform SPECT image initialized with the computed dimensions
            and voxel sizes.
    """
    # Create a uniform image from the sinogram and adjust z-voxel size.
    image = sinogram.create_uniform_image(value=1)
    voxel_size = list(image.voxel_sizes())
    voxel_size[0] *= 2  # Adjust z-direction voxel size.

    if dims is None:
        # Compute new dimensions based on the uniform image.
        dims = list(image.dimensions())
        dims[0] = dims[0] // 2 + dims[0] % 2  # Halve the first dimension (with rounding)
        dims[1] -= dims[1] % 2  # Ensure even number for second dimension
        dims[2] = dims[1]  # Set third dimension equal to second dimension

    if origin is None:
        origin = (0, 0, 0)

    # Initialize a new image with computed dimensions, voxel sizes, and origin.
    new_image = ImageData()
    new_image.initialise(tuple(dims), tuple(voxel_size), tuple(origin))
    return new_image


def get_kappa_squared(am, x, max_value=1e3):
    """Compute kappa squared image from attenuation map using forward-backward projection.

    This computes the sensitivity-based weighting image using:
        kappa² = AM^T [ (AM·1) / AM·x ]

    Where:
    - AM is the attenuation-only acquisition model (no additive term)
    - x is the initial/current image estimate
    - Result is clamped to [0, max_value] after NaN/Inf cleaning

    Args:
        am: SIRF AcquisitionModel (should be attenuation-only, no additive term)
        x: SIRF ImageData - initial image estimate
        max_value: Maximum allowed value for kappa² (default 1e3)

    Returns:
        SIRF ImageData: Kappa squared weighting image

    Notes:
        - This replaces the old Hessian-based kappa calculation
        - The AM should NOT include additive terms (randoms, scatter)
        - max_value=np.inf for SPECT (no clamping), 1e3 for PET (numerical stability)
    """
    import numpy as np

    # Forward project uniform image through AM
    one_image = x.get_uniform_copy(1.0)
    am_one = am.forward(one_image)

    # Create uniform sinogram
    one_data = am_one.get_uniform_copy(1.0)

    # Forward project actual image
    am_x = am.forward(x)

    # Compute ratio: one_data / am_x
    ratio = one_data / am_x

    # Clean NaN/Inf values
    ratio_arr = np.nan_to_num(ratio.asarray(), nan=0.0, posinf=0.0, neginf=0.0)
    ratio.fill(ratio_arr)

    # Clamp to max_value
    ratio = ratio.minimum(max_value)

    # Backproject weighted sinogram
    kappa2 = am.backward(ratio * am_one)

    return kappa2


def smooth_kappa_via_inverse(kappa2, fwhm_mm=(10.0, 10.0, 10.0), save_inverse=False, output_path=None, prefix=""):
    """Smooth kappa² image by smoothing its inverse.

    This provides better noise suppression in low-sensitivity regions:
        1. Compute 1/kappa² (with safe division)
        2. Apply Gaussian smoothing with specified FWHM
        3. Compute 1/(smoothed_inverse) to get smoothed kappa²

    Args:
        kappa2: SIRF ImageData - kappa squared image to smooth
        fwhm_mm: tuple of 3 floats - FWHM in mm for (z,y,x) directions
        save_inverse: bool - if True, save the inverse image to disk
        output_path: str - directory to save inverse image (required if save_inverse=True)
        prefix: str - prefix for saved inverse filename

    Returns:
        tuple: (kappa2_smoothed, kappa2_inv_smoothed) - both as SIRF ImageData
               kappa2_inv_smoothed is the smoothed inverse image
    """
    import numpy as np
    import os

    # Create smoother
    im_smoother = SeparableGaussianImageFilter()
    im_smoother.set_fwhms(fwhm_mm)

    # Compute inverse with safe division
    arr = kappa2.asarray()
    inv_arr = np.reciprocal(arr, where=arr != 0)
    kappa2_inv = kappa2.clone()
    kappa2_inv.fill(inv_arr)

    # Smooth the inverse
    im_smoother.apply(kappa2_inv)

    # Save smoothed inverse if requested
    if save_inverse and output_path is not None:
        kappa2_inv.write(os.path.join(output_path, f"{prefix}_inv_smoothed.hv"))

    # Compute inverse again to get smoothed kappa²
    inv_arr_smoothed = kappa2_inv.asarray()
    kappa2_smoothed = kappa2.clone()
    kappa2_arr = np.reciprocal(inv_arr_smoothed, where=inv_arr_smoothed != 0)
    kappa2_smoothed.fill(kappa2_arr)

    return kappa2_smoothed, kappa2_inv

def set_up_partitioned_objectives(pet_data, spect_data, pet_obj_funs, spect_obj_funs):
    """Returns a CIL SumFunction for the partitioned objective functions"""

    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data["initial_image"])

    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data["initial_image"])

    return pet_obj_funs, spect_obj_funs


def get_block_objective(desired_image, other_image, obj_fun, scale=1, order=0):
    """Returns a block CIL objective function for the given SIRF objective function.

    Args:
        desired_image: The image to apply the objective function to.
        other_image: The other image in the block (receives zero operator).
        obj_fun: The objective function to wrap.
        scale: Scaling factor for the identity operator (default 1).
        order: Position of desired_image in block (0 or 1).

    Returns:
        OperatorCompositionFunction: Block objective function.
    """
    from recon_core.cil_extensions.operators import ScalingOperator

    # Set up zero operators
    o2d_zero = ZeroOperator(other_image, desired_image)
    if scale == 1:
        d2d_id = IdentityOperator(desired_image)
    else:
        d2d_id = ScalingOperator(scale, desired_image)

    if order == 0:
        return OperatorCompositionFunction(obj_fun, BlockOperator(d2d_id, o2d_zero, shape=(1, 2)))
    elif order == 1:
        return OperatorCompositionFunction(obj_fun, BlockOperator(o2d_zero, d2d_id, shape=(1, 2)))
    else:
        raise ValueError("Order must be 0 or 1")


def set_up_kl_objectives(
    pet_data, spect_data, pet_datas, pet_norms, spect_datas, pet_ams, spect_ams
):
    """Returns a CIL SumFunction using KL objective functions for the PET and SPECT data and acq models"""

    for d, am in zip(pet_datas, pet_ams):
        am.set_up(d, pet_data["initial_image"])

    for d, am in zip(spect_datas, spect_ams):
        am.set_up(d, spect_data["initial_image"])

    pet_ads = [am.get_additive_term() * norm for am, norm in zip(pet_ams, pet_norms)]
    spect_ads = [
        am.get_additive_term() for am in spect_ams
    ]  # Do I somehow need to apply the normalisation here?

    pet_ams = [am.get_linear_acquisition_model() for am in pet_ams]
    spect_ams = [am.get_linear_acquisition_model() for am in spect_ams]

    pet_obj_funs = [
        OperatorCompositionFunction(KullbackLeibler(data, eta=add + add.max() / 1e3), am)
        for data, add, am in zip(pet_datas, pet_ads, pet_ams)
    ]
    spect_obj_funs = [
        OperatorCompositionFunction(KullbackLeibler(data, eta=add + add.max() / 1e3), am)
        for data, add, am in zip(spect_datas, spect_ads, spect_ams)
    ]

    return pet_obj_funs, spect_obj_funs


def _clamp_inverse(inv_sens_arr, clamp_percentile):
    if clamp_percentile is None:
        return inv_sens_arr
    finite_vals = inv_sens_arr[np.isfinite(inv_sens_arr)]
    if finite_vals.size == 0:
        return inv_sens_arr
    clamp_value = np.percentile(finite_vals, clamp_percentile)
    return np.minimum(inv_sens_arr, clamp_value)


def get_s_inv_from_objs(
    obj_funs,
    initial_estimates,
    clamp_percentile: float | None = None,
    adjoint_ops=None,
):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates.get_uniform_copy(0)
    if adjoint_ops is None:
        adjoint_ops = [None] * len(obj_funs)
    if len(adjoint_ops) != len(obj_funs):
        raise ValueError("adjoint_ops must match number of objective function blocks")
    for i, el in enumerate(s_inv.containers):
        for j, obj_fun in enumerate(obj_funs[i]):
            # Extract underlying function if wrapped in OperatorCompositionFunction
            obj_fn = obj_fun.function if isinstance(obj_fun, OperatorCompositionFunction) else obj_fun
            if j == 0:
                sens = obj_fn.get_subset_sensitivity(0)
            else:
                sens += obj_fn.get_subset_sensitivity(0)
        # Compute maximum with zero (returning a new container)
        sens.maximum(0, out=sens)
        adjoint_op = adjoint_ops[i]
        if adjoint_op is not None:
            sens = adjoint_op.adjoint(sens)
        sens_arr = get_array(sens).astype(np.float32)
        # We can afford to avoid zeros because
        # a zero sensitivity means we're outside the FOV
        inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
        inv_sens_arr = _clamp_inverse(inv_sens_arr, clamp_percentile)
        # there really shouldn't be any NaNs, but just in case
        s_inv.containers[i].fill(np.nan_to_num(inv_sens_arr))
    return s_inv


def get_s_inv_from_am(
    ams,
    initial_estimates,
    clamp_percentile: float | None = None,
    adjoint_ops=None,
):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates * 0
    if adjoint_ops is None:
        adjoint_ops = [None] * len(ams)
    if len(adjoint_ops) != len(ams):
        raise ValueError("adjoint_ops must match number of acquisition model blocks")
    for i, el in enumerate(s_inv.containers):
        for am in ams[i]:
            one = am.forward(initial_estimates[i]).get_uniform_copy(1)
            tmp = am.backward(one)
            el += tmp
        el = el.maximum(0)
        adjoint_op = adjoint_ops[i]
        if adjoint_op is not None:
            el = adjoint_op.adjoint(el)
        el_arr = get_array(el)
        el_arr = np.reciprocal(el_arr, where=el_arr != 0)
        el_arr = _clamp_inverse(el_arr, clamp_percentile)
        el.fill(np.nan_to_num(el_arr))
    return s_inv


def get_s_inv_from_subset_objs(
    obj_funs,
    initial_estimate,
    clamp_percentile: float | None = None,
    adjoint_operator=None,
):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimate.get_uniform_copy(0)
    for j, obj_fun in enumerate(obj_funs):
        # Extract underlying function if wrapped in OperatorCompositionFunction
        obj_fn = obj_fun.function if isinstance(obj_fun, OperatorCompositionFunction) else obj_fun
        if j == 0:
            sens = obj_fn.get_subset_sensitivity(0)
        else:
            sens += obj_fn.get_subset_sensitivity(0)
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    if adjoint_operator is not None:
        sens = adjoint_operator.adjoint(sens)
    sens_arr = get_array(sens).astype(np.float32)
    # We can afford to avoid zeros because
    # a zero sensitivity means we're outside the FOV
    inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
    inv_sens_arr = _clamp_inverse(inv_sens_arr, clamp_percentile)
    # there really shouldn't be any NaNs, but just in case
    s_inv.fill(np.nan_to_num(inv_sens_arr))
    return s_inv


def get_sensitivity_from_subset_objs(obj_funs, initial_estimate, adjoint_operator=None):
    # get subset_sensitivity BDC for preconditioner
    for j, obj_fun in enumerate(obj_funs):
        # Extract underlying function if wrapped in OperatorCompositionFunction
        obj_fn = obj_fun.function if isinstance(obj_fun, OperatorCompositionFunction) else obj_fun
        if j == 0:
            sens = obj_fn.get_subset_sensitivity(0)
        else:
            sens += obj_fn.get_subset_sensitivity(0)
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    if adjoint_operator is not None:
        sens = adjoint_operator.adjoint(sens)
    return sens


def get_sensitivities_from_subset_objs(obj_funs, initial_estimate):
    # get subset_sensitivity BDC for preconditioner
    sens_list = []
    for obj_fun in obj_funs:
        # Extract underlying function if wrapped in OperatorCompositionFunction
        obj_fn = obj_fun.function if isinstance(obj_fun, OperatorCompositionFunction) else obj_fun
        sens = obj_fn.get_subset_sensitivity(0)
        sens = sens.maximum(0)
        sens_list.append(sens)
    return sens


def compute_inv_hessian_diagonals(bdc, obj_funs_list):
    outputs = []

    for image, obj_funs in zip(bdc.containers, obj_funs_list):
        # Initialize uniform copies
        ones_image = image.get_uniform_copy(1)
        hessian_diag = ones_image.get_uniform_copy(0)

        # Accumulate Hessian contributions
        for obj_fun in obj_funs:
            hessian_diag += obj_fun.function.multiply_with_Hessian(image, ones_image)

        # Take absolute values and write the result
        hessian_diag = hessian_diag.abs()

        hessian_diag_arr = get_array(hessian_diag)
        hessian_diag.fill(np.reciprocal(hessian_diag_arr, where=hessian_diag_arr != 0))

        outputs.append(hessian_diag)

    return BlockDataContainer(*outputs)


def get_subset_data(data, num_subsets, stagger="staggered"):
    views = data.dimensions()[2]
    indices = list(range(views))
    partitions_idxs = partitioner.partition_indices(num_subsets, indices, stagger=stagger)
    return [data.get_subset(partitions_idxs[i]) for i in range(num_subsets)]


def get_array(obj):
    """
    Get array from SIRF object, preferring asarray() over as_array() for performance.

    Falls back to as_array() if asarray() is not available (older SIRF versions).

    Args:
        obj: SIRF object with asarray() or as_array() method

    Returns:
        numpy array or reference to underlying array
    """
    if hasattr(obj, "asarray"):
        return obj.asarray()
    elif hasattr(obj, "as_array"):
        return obj.as_array()
    else:
        raise AttributeError(f"Object {type(obj)} has neither asarray() nor as_array() method")


def get_filters(fwhms=(10, 10, 10)):
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms(fwhms)
    return cyl, gauss
