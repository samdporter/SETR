import logging
import os
from types import MethodType

import numpy as np
from cil.framework import BlockDataContainer
from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction
from cil.optimisation.operators import BlockOperator, IdentityOperator, ZeroOperator
from sirf.STIR import (
    AcquisitionData,
    AcquisitionModelUsingMatrix,
    AcquisitionModelUsingParallelproj,
    AcquisitionModelUsingRayTracingMatrix,
    ImageData,
    SPECTUBMatrix,
    SeparableGaussianImageFilter,
    TruncateToCylinderProcessor,
)
from sirf.contrib import partitioner
from setr.cil_extensions.operators import ScalingOperator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_pet_am(
        gpu=True, gauss_fwhm=None, 
    ):
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
        spect_data, res = None, 
        keep_all_views_in_cache=True, 
        gauss_fwhm=None,
        attenuation=True
    ):
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_keep_all_views_in_cache(
        keep_all_views_in_cache
    )
    if attenuation:
        try:
            spect_am_mat.set_attenuation_image(
                spect_data["attenuation"]
            )
        except:
            print("No attenuation data")
    if res:
        spect_am_mat.set_resolution_model(*res)
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    if gauss_fwhm:
        spect_psf = SeparableGaussianImageFilter()
        spect_psf.set_fwhms(gauss_fwhm) 
        spect_am.set_image_data_processor(spect_psf)
    return spect_am

def get_pet_data(path: str, suffix: str = "") -> dict:
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
    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(
        os.path.join(path, f"prompts{suffix}.hs")
    )
    pet_data["additive"] = AcquisitionData(
        os.path.join(path, f"additive_term{suffix}.hs")
    )
    pet_data["normalisation"] = AcquisitionData(
        os.path.join(path, f"mult_factors{suffix}.hs")
    )
    pet_data["attenuation"] = ImageData(os.path.join(path, f"umap_zoomed.hv"))

    # Always load the template image.
    template_img_path = os.path.join(path, f"template_image.hv")
    try:
        pet_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load PET template image (%s)", str(e_template))
        raise RuntimeError("Unable to load PET template image.") from e_template

    # Try to load the initial image.
    initial_img_path = os.path.join(path, f"initial_image.hv")
    try:
        pet_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning("No PET initial image found (%s). Using uniform copy of template image.",
                        str(e_initial))
        pet_data["initial_image"] = pet_data["template_image"].get_uniform_copy(1)

    try:
        pet_data["spect"] = ImageData(os.path.join(path, "spect.hv"))
    except Exception as e_spect:
        logging.info("No SPECT guidance image found for PET: %s", str(e_spect))

    return pet_data


from pathlib import Path
from typing import Dict, List, Optional

def get_pet_data_multiple_bed_pos(path: str, suffixes: List[str], tof: bool = False) -> Dict[str, object]:
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
    pet_data["attenuation"]    = load_image(base / "umap_zoomed.hv")
    pet_data["template_image"] = load_image(base / "template_image.hv", clamp=False, required=True)
    pet_data["initial_image"]  = load_image(base / "initial_image.hv") \
                                  or pet_data["template_image"].get_uniform_copy(1)
    pet_data["spect"]          = load_image(base / "spect.hv", clamp=False)

    # per‐bed data
    beds: Dict[str, Dict[str, object]] = {}
    for suf in suffixes:
        bp = {}
        bp["acquisition_data"] = load_acq(base / f"prompts{suf}.hs")
        bp["additive"]         = load_acq(base / f"additive_term{suf}.hs")
        bp["normalisation"]    = load_acq(base / f"mult_factors{suf}.hs")
        bp["template_image"]   = load_image(base / f"template_image{suf}.hv", clamp=False, required=True)
        bp["initial_image"]    = load_image(base / f"initial_image{suf}.hv") \
                                  or bp["template_image"].get_uniform_copy(1)
        bp["attenuation"]      = load_image(base / f"umap{suf}.hv")
        bp["spect"]            = load_image(base / f"spect{suf}.hv", clamp=False)
        beds[suf] = bp

    pet_data["bed_positions"] = beds
    return pet_data

        
def get_spect_data(path: str) -> dict:
    """
    Load SPECT data from the given path.
    
    This function always loads a template image and then attempts to load the
    initial image. If the initial image is not found, it creates a uniform copy
    of the template image (filled with ones). Also, the attenuation image is flipped
    on the x-axis due to a known STIR bug.

    Args:
        path (str): Path to the data directory.

    Returns:
        dict: A dictionary with keys: "acquisition_data", "additive", "attenuation",
        "template_image", and "initial_image".
    """
    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(os.path.join(path, "peak.hs"))

    try:
        spect_data["additive"] = AcquisitionData(os.path.join(path, "scatter.hs"))
    except Exception as e_scatter:
        logging.warning("No scatter data found (%s). Using zeros.", str(e_scatter))
        spect_data["additive"] = AcquisitionData(spect_data["acquisition_data"])
        spect_data["additive"].fill(0)

    spect_data["attenuation"] = ImageData(os.path.join(path, "umap_zoomed.hv"))
    # Flip the attenuation image on the x-axis due to bug in STIR.
    attn_arr = spect_data["attenuation"].as_array()
    attn_arr = np.flip(attn_arr, axis=-1)
    spect_data["attenuation"].fill(attn_arr)

    # Always load the template image.
    template_img_path = os.path.join(path, "template_image.hv")
    try:
        spect_data["template_image"] = ImageData(template_img_path)
    except Exception as e_template:
        logging.error("Failed to load SPECT template image (%s)", str(e_template))
        raise RuntimeError("Unable to load SPECT template image.") from e_template

    # Try to load the initial image.
    initial_img_path = os.path.join(path, "initial_image.hv")
    try:
        spect_data["initial_image"] = ImageData(initial_img_path).maximum(0)
    except Exception as e_initial:
        logging.warning("No SPECT initial image found (%s). Using uniform copy of template image.",
                        str(e_initial))
        spect_data["initial_image"] = spect_data["template_image"].get_uniform_copy(1)

    return spect_data

def create_spect_uniform_image(sinogram, xy=None, origin=None):
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
    print(type(xy))
    image = sinogram.create_uniform_image(value=1, xy=int(xy))
    voxel_size = list(image.voxel_sizes())
    voxel_size[0] *= 2  # Adjust z-direction voxel size.

    # Compute new dimensions based on the uniform image.
    dims = list(image.dimensions())
    dims[0] = dims[0] // 2 + dims[0] % 2  # Halve the first dimension (with rounding)
    dims[1] -= dims[1] % 2                # Ensure even number for second dimension
    dims[2] = dims[1]                     # Set third dimension equal to second dimension

    if origin is None:
        origin = (0, 0, 0)

    # Initialize a new image with computed dimensions, voxel sizes, and origin.
    new_image = ImageData()
    new_image.initialise(tuple(dims), tuple(voxel_size), tuple(origin))
    return new_image

def compute_kappa_squared_image_from_partitioned_objective(obj_funs, initial_image, normalise=True):
    """
    Computes a "kappa" image for a prior as sqrt(H.1).
    This will attempt to give uniform "perturbation response".
    See Yu-jung Tsai et al. TMI 2020 https://doi.org/10.1109/TMI.2019.2913889

    WARNING: Assumes the objective function has been set-up already.
    """
    out = initial_image.get_uniform_copy(0)
    for obj_fun in obj_funs:
        # need to get the function from the ScaledFunction OperatorCompositionFunction
        if isinstance(obj_fun.function, OperatorCompositionFunction):
            out += obj_fun.function.function.multiply_with_Hessian(
                initial_image,
                initial_image.allocate(1)
            )
        else:
            out += obj_fun.function.multiply_with_Hessian(initial_image, initial_image.allocate(1))
    out = out.abs()
    # shouldn't really need to do this, but just in case
    out = out.maximum(0)
    # debug printing
    print(f"max: {out.max()}")
    mean = out.sum()/out.size
    print(f"mean: {mean}")
    if normalise:
        # we want to normalise by thye median
        median = out.as_array().flatten()
        median.sort()
        median = median[int(median.size/2)]
        print(f"median: {median}")
        out /= median
    return out

def attach_prior_hessian(prior, epsilon = 0) -> None:
    """Attach an inv_hessian_diag method to the prior function."""

    def inv_hessian_diag(self, x, out=None, epsilon=epsilon):
        ret = self.function.operator.adjoint(
            self.function.function.inv_hessian_diag(
                self.function.operator.direct(x),
            )
        )
        ret = ret.abs()
        if out is not None:
            out.fill(ret)
        return ret
    
    def hessian_diag(self, x, out=None, epsilon=epsilon):
        ret = self.function.operator.adjoint(
            self.function.function.hessian_diag(
                self.function.operator.direct(x),
            )
        )
        ret = ret.abs()
        if out is not None:
            out.fill(ret)
        return ret

    prior.inv_hessian_diag = MethodType(inv_hessian_diag, prior)
    prior.hessian_diag = MethodType(hessian_diag, prior)


def set_up_partitioned_objectives(pet_data, spect_data, pet_obj_funs, spect_obj_funs):

    """ Returns a CIL SumFunction for the partitioned objective functions """
    
    for obj_fun in pet_obj_funs:
        obj_fun.set_up(pet_data['initial_image'])

    for obj_fun in spect_obj_funs:
        obj_fun.set_up(spect_data['initial_image'])
    
    return pet_obj_funs, spect_obj_funs

def get_block_objective(desired_image, other_image, obj_fun, scale=1, order = 0):

    """ Returns a block CIL objective function for the given SIRF objective function """

    # Set up zero operators
    o2d_zero = ZeroOperator(other_image, desired_image)
    if scale == 1:
        d2d_id = IdentityOperator(desired_image)
    else:
        d2d_id = ScalingOperator(scale, desired_image)

    if order == 0:
        return OperatorCompositionFunction(obj_fun, BlockOperator(d2d_id, o2d_zero, shape = (1,2)))
    elif order == 1:
        return OperatorCompositionFunction(obj_fun, BlockOperator(o2d_zero, d2d_id, shape = (1,2)))
    else:
        raise ValueError("Order must be 0 or 1")

def set_up_kl_objectives(pet_data, spect_data, pet_datas, pet_norms, spect_datas, pet_ams, spect_ams):

    """ Returns a CIL SumFunction using KL objective functions for the PET and SPECT data and acq models """
    
    for d, am in zip(pet_datas, pet_ams):
        am.set_up(d, pet_data['initial_image'])

    for d, am in zip(spect_datas, spect_ams):
        am.set_up(d, spect_data['initial_image'])

    pet_ads = [am.get_additive_term()*norm for am, norm in zip(pet_ams, pet_norms)]
    spect_ads = [am.get_additive_term() for am in spect_ams] # Do I somehow need to apply the normalisation here?

    pet_ams = [am.get_linear_acquisition_model() for am in pet_ams]
    spect_ams = [am.get_linear_acquisition_model() for am in spect_ams]

    pet_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add+add.max()/1e3), am) for data, add, am in zip(pet_datas, pet_ads, pet_ams)]
    spect_obj_funs = [OperatorCompositionFunction(KullbackLeibler(data, eta=add+add.max()/1e3), am) for data, add, am in zip(spect_datas, spect_ads, spect_ams)]

    return pet_obj_funs, spect_obj_funs

def get_s_inv_from_objs(obj_funs, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates.get_uniform_copy(0)
    for i, el in enumerate(s_inv.containers):
        for j, obj_fun in enumerate(obj_funs[i]):
            if j == 0: 
                sens = obj_fun.get_subset_sensitivity(0)
            else:
                sens += obj_fun.get_subset_sensitivity(0)
        # Compute maximum with zero (returning a new container)
        sens.maximum(0, out = sens)
        sens_arr = sens.as_array().astype(np.float32)
        # We can afford to avoid zeros because
        # a zero sensitivity means we're outside the FOV
        inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
        # there really shouldn't be any NaNs, but just in case
        s_inv.containers[i].fill(np.nan_to_num(inv_sens_arr) )
    return s_inv


def get_s_inv_from_am(ams, initial_estimates):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimates*0
    for i, el in enumerate(s_inv.containers):
        for am in ams[i]:
            one = am.forward(initial_estimates[i]).get_uniform_copy(1)
            tmp = am.backward(one)
            el += tmp
        el = el.maximum(0)
        el_arr = el.as_array()
        el_arr = np.reciprocal(el_arr, where=el_arr!=0)
        el.fill(np.nan_to_num(el_arr))
    return s_inv

def get_s_inv_from_subset_objs(obj_funs, initial_estimate):
    # get subset_sensitivity BDC for preconditioner
    s_inv = initial_estimate.get_uniform_copy(0)
    for j, obj_fun in enumerate(obj_funs):
        if j == 0: 
            sens = obj_fun.get_subset_sensitivity(0)
        else:
            sens += obj_fun.get_subset_sensitivity(0)
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    sens_arr = sens.as_array().astype(np.float32)
    # We can afford to avoid zeros because
    # a zero sensitivity means we're outside the FOV
    inv_sens_arr = np.reciprocal(sens_arr, where=sens_arr != 0)
    # there really shouldn't be any NaNs, but just in case
    s_inv.fill(np.nan_to_num(inv_sens_arr) )
    return s_inv

def get_sensitivity_from_subset_objs(obj_funs, initial_estimate):
    # get subset_sensitivity BDC for preconditioner
    for j, obj_fun in enumerate(obj_funs):
        if j == 0: 
            sens = obj_fun.get_subset_sensitivity(0)
        else:
            sens += obj_fun.get_subset_sensitivity(0)
    # Compute maximum with zero (returning a new container)
    sens = sens.maximum(0)
    return sens

def get_sensitivities_from_subset_objs(obj_funs, initial_estimate):
    # get subset_sensitivity BDC for preconditioner
    sens_list = [] 
    for j, obj_fun in enumerate(obj_funs):
        sens = obj_fun.get_subset_sensitivity(0)
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
            hessian_diag += obj_fun.function.multiply_with_Hessian(
                image, ones_image
            )

        # Take absolute values and write the result
        hessian_diag = hessian_diag.abs()

        hessian_diag_arr = hessian_diag.as_array()
        hessian_diag.fill(np.reciprocal(hessian_diag_arr, where=hessian_diag_arr!=0))
        
        outputs.append(hessian_diag)

    return BlockDataContainer(*outputs)

def get_subset_data(data, num_subsets, stagger = "staggered"):
        
    views=data.dimensions()[2]
    indices = list(range(views))
    partitions_idxs = partitioner.partition_indices(num_subsets, indices, stagger = stagger)
    datas = [data.get_subset(partitions_idxs[i]) for i in range(num_subsets)]

    return datas

def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((10,10,10))
    return cyl, gauss