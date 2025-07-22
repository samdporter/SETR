import os
import re
import numpy as np
from cil.framework import BlockDataContainer, BlockGeometry
from cil.optimisation.operators import LinearOperator
from sirf.Reg import NiftyResample
from sirf.STIR import ImageData, TruncateToCylinderProcessor

class AdjointOperator(LinearOperator):
    """Very simple adjoint operator that reverses the roles of direct and adjoint methods.

    Args:
        operator (LinearOperator): The operator to be reversed.
    """

    def __init__(self, operator):
        self.operator = operator
        self.domain_geometry=operator.domain_geometry
        self.range_geometry=operator.range_geometry

    def direct(self, x, out=None):
        return self.operator.adjoint(x, out)

    def adjoint(self, x, out=None):
        return self.operator.direct(x, out)
    
class ScalingOperator(LinearOperator):
    def __init__(self, scale, domain_geometry):
        super(ScalingOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        self.scale = scale
    def direct(self, x, out=None):
        """Scale the input image by a constant factor."""
        if out is None:
            return x * self.scale
        else:
            out.fill(x * self.scale)
        return out
        
    def adjoint(self, x, out=None):
        """Scale the input image by a constant factor."""
        return self.direct(x, out=out)
    
class ZeroEndSlicesOperator(LinearOperator):
    """
    Zeros the end slices of the input image.
    Not really linear but we'll pretend it is.

    Args:
        num_slices (int): Number of slices to zero at both ends.
        image (ImageGeometry): The geometry of the image to be processed.
    """
    
    def __init__(self, num_slices, image):
        
        self.num_slices = num_slices
        
        super().__init__(domain_geometry=image, range_geometry=image)
        
    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        out_arr = out.as_array()
        out_arr[-self.num_slices:,:,:] = 0
        out_arr[:self.num_slices,:,:] = 0
        out.fill(out_arr)
        return out
    
    def adjoint(self, x, out=None):
        return self.direct(x, out)
    
class NaNToZeroOperator(LinearOperator):
    """ 
    Puts zeros in NaNs 
    Not really linear but we'll pretend it is.
    
    Args:
        image (ImageGeometry): The geometry of the image to be processed.
    """

    def __init__(self, image):
        super().__init__(domain_geometry=image, range_geometry=image)

    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        out_arr = out.as_array()
        out_arr[np.isnan(out_arr)] = 0
        out.fill(out_arr)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)
    
class TruncationOperator(LinearOperator):

    """CIL Wrapper for SIRF TruncateToCylinderProcessor.
    """

    def __init__(self, domain_geometry, **kwargs):
        super().__init__(
            domain_geometry=domain_geometry, 
            range_geometry=domain_geometry
        )
        
        self.truncate = TruncateToCylinderProcessor()
        self.truncate.set_strictly_less_than_radius(True)

    def __call__(self, x, out=None):
        return self.direct(x, out)

    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        self.truncate.apply(out)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)
    

class DirectionalOperator(LinearOperator):

    def __init__(self, anatomical_gradient, gamma = 1, eta=1e-6):

        self.anatomical_gradient = anatomical_gradient
        geometry = BlockGeometry(*anatomical_gradient.containers) # a little odd. Not sure why I did this...
        self.tmp = self.anatomical_gradient.containers[0].clone()

        self.gamma = gamma

        self.xi = self.anatomical_gradient/(self.anatomical_gradient.pnorm().power(2)+eta**2).sqrt()

        self.calculate_norm = lambda _: 1

        super(DirectionalOperator, self).__init__(domain_geometry=geometry,
                                       range_geometry=geometry,)

        
    def direct(self, x, out=None):

        if out is None:
            return x - self.gamma * self.xi * self.dot(self.xi, x)
        else:
            out.fill(x - self.gamma * self.xi * self.dot(self.xi, x))
    
    def adjoint(self, x, out=None):
        # This is the same as the direct operator
        return self.direct(x, out)
    
    def dot(self, x, y):
        self.tmp.fill(0)
        for el_x, el_y in zip(x.containers, y.containers):
            self.tmp += el_x * el_y
        return self.tmp

class NiftyResampleOperator(LinearOperator):

    def __init__(self, reference, floating, transform):

        self.reference = reference.get_uniform_copy(0)
        self.floating = floating.get_uniform_copy(0)
        self.transform = transform

        self.resampler = NiftyResample()
        self.resampler.set_reference_image(reference)
        self.resampler.set_floating_image(floating)
        self.resampler.set_interpolation_type_to_cubic_spline()
        self.resampler.set_padding_value(0)
        self.resampler.add_transformation(self.transform)

    def direct(self, x, out=None):
        res = self.resampler.forward(x)
        res = res.maximum(0)
        if out is not None:
            out.fill(res)
        return res
    
    def adjoint(self, x, out=None):
        res = self.resampler.backward(x)
        res = res.maximum(0)
        if out is not None:
            out.fill(res)
        return res
    
    def domain_geometry(self):
        return self.floating
    
    def range_geometry(self):
        return self.reference

class CouchShiftOperator(LinearOperator):
    """
    A linear operator that shifts the couch position in an image by modifying the
    'first pixel offset (mm) [3]' value in the associated Interfile header (.hv).

    Parameters:
    -----------
    image : ImageData
        The input image whose couch position is to be shifted.
    shift : float
        The amount by which to shift the couch position along the z-axis (in mm).
    """

    def __init__(self, image, shift):
        """
        Initialize the CouchShiftOperator.

        Parameters:
        -----------
        image : ImageData
            The input image whose couch position is to be shifted.
        shift : float
            The amount by which to shift the couch position along the z-axis (in mm).
        """
        self.shift = shift
        # need to create range geometry by shifting the image
        range_geometry = self.initialise_shift(image, out=None)
        super().__init__(
            domain_geometry=image, 
            range_geometry=range_geometry
        )
        
        self.unshifted_image = image.copy()
        self.shifted_image = range_geometry.copy()

    def initialise_shift(self, x, out=None):
        """
        Apply the couch shift to the input image.

        Parameters:
        -----------
        x : ImageData
            The input image to be shifted.
        out : ImageData, optional
            If provided, the result will be stored in this object. Otherwise, a new
            ImageData object will be created.

        Returns:
        --------
        ImageData
            The shifted image.
        """
        # Write the input image to a temporary file
        x.write("tmp_shifted.hv")

        # Modify the 'first pixel offset (mm) [3]' in the temporary file
        self.modify_pixel_offset("tmp_shifted.hv", self.shift, 3)

        # If `out` is provided, update it
        if out is not None:
            out.read_from_file("tmp_shifted.hv")
        else:
            out = ImageData("tmp_shifted.hv")

        # Delete the temporary file
        os.remove("tmp_shifted.hv")

        return out
    
    def direct(self, x, out=None):
        
        x_arr = x.as_array()
        if out is not None:
            out.fill(x_arr)
            return out
        else:
            self.shifted_image.fill(x_arr)
            return self.shifted_image           
        

    def adjoint(self, x, out=None):

        x_arr = x.as_array()
        if out is not None:
            out.fill(x_arr)
            return out
        else:
            self.unshifted_image.fill(x_arr)
            return self.unshifted_image

    @staticmethod
    def modify_pixel_offset(file_path, new_offset, pixel_index):
        """
        Modify the 'first pixel offset (mm) [pixel_index]' value in an Interfile header (.hv).

        Parameters:
        -----------
        file_path : str
            The path to the Interfile header (.hv) to be modified.
        new_offset : float
            The new value for 'first pixel offset (mm) [pixel_index]'.
        """
        delete_file = False
        if isinstance(file_path, ImageData):
            print("This is supposed to be a file path but got an ImageData object. Writing to a temporary file.")
            delete_file = True
            file_path.write("tmp_shift.hv")
            file_path = "tmp_shift.hv"
        try:
            # Read the file content
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the specific line
            for i, line in enumerate(lines):
                if line.strip().startswith(f"first pixel offset (mm) [{pixel_index}] :="):
                    lines[i] = f"first pixel offset (mm) [{pixel_index}] := {new_offset}\n"
                    break

            # Write the updated content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
        except Exception as e:
            raise RuntimeError(f"Failed to modify the file {file_path}: {e}")
        
        image = ImageData(file_path)

        if delete_file:
            os.remove(file_path)
        
        return image
    
    @staticmethod
    def get_couch_shift_from_header(header_filepath):

        start_horizontal_bed_position = None

        # Read the file and extract the desired value
        with open(header_filepath, 'r') as file:
            for line in file:
                if line.startswith("start horizontal bed position (mm) :="):
                    # Extract the value after ":="
                    start_horizontal_bed_position = float(line.split(":=")[1].strip())

        if start_horizontal_bed_position is None:
            raise ValueError("Could not find 'start horizontal bed position (mm)' in the sinogram file.")

        return start_horizontal_bed_position

    @staticmethod
    def get_couch_shift_from_acqusition_data(sinogram) -> float:

        header = sinogram.get_info()

        pattern = r"start\s+horizontal\s+bed\s+position\s+\(mm\)\s*:=\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern, header)
        if match is None:
            raise ValueError("Horizontal bed position not found.")
        return float(match.group(1))

    @staticmethod
    def get_couch_shift_from_sinogram(sinogram) -> float:

        if isinstance(sinogram, str):
            return CouchShiftOperator.get_couch_shift_from_header(sinogram)
        else:
            return CouchShiftOperator.get_couch_shift_from_acqusition_data(sinogram)

class ImageCombineOperator(LinearOperator):
    def __init__(
        self, images: BlockDataContainer,
    ):
        self.images = images

        self.reference = ImageData()
        dim_xy = images.containers[0].dimensions()[1]
        dim_z = ImageCombineOperator.get_combined_length_voxels(images)
        offset_xy = images.containers[0].get_geometrical_info().get_offset()[0]
        offset_z = -images.containers[-1].get_geometrical_info().get_offset()[2]
        print(f"setting offset_z to {-offset_z}. If something goes wrong try swapping the image order")
        # for some reason, initialising as offset_xy=0 works here
        self.reference.initialise((dim_z, dim_xy, dim_xy), images.containers[0].voxel_sizes(), (offset_z, 0,0))
        self.reference = self.reference
        
        # Ensure all images have the same voxel size as the reference
        assert all(img.voxel_sizes() == self.reference.voxel_sizes() for img in images.containers), "All images must have the same voxel size as the reference"
        
        # Ensure the combined image length matches the reference dimensions
        assert self.get_combined_length_voxels(images) == self.reference.dimensions()[0], f"Combined image length and reference dimensions do not match. Something is wrong \n Combined image length: {self.get_combined_length_voxels(images)} \n Reference dimensions: {self.reference.dimensions()[0]}"
        
        super().__init__(domain_geometry=images, range_geometry=self.reference)
    
    
    @staticmethod
    def get_combined_length(images):
        offsets = [img.get_geometrical_info().get_offset()[2] for img in images.containers]
        lengths = [img.dimensions()[0] * img.voxel_sizes()[0] for img in images.containers]
        
        return max(offset + length for offset, length in zip(offsets, lengths)) - min(offsets)
    
    @staticmethod
    def get_combined_length_voxels(images):
        voxel_size = images.containers[0].voxel_sizes()[0]
        assert all(img.voxel_sizes() == images.containers[0].voxel_sizes() for img in images.containers)
        
        length = ImageCombineOperator.get_combined_length(images)
        
        assert ((length / voxel_size) % 1) - 1 < 1e-3
        return int(round(length / voxel_size))
    

    @staticmethod
    def combine_images(
        reference: ImageData,
        images: BlockDataContainer,
        sens_images: BlockDataContainer = None,
        weight_overlap: bool = False,
    ):
        """
        Combines images onto `reference`. If weight_overlap=True, then:
        - overlap mask M = (coverage_count ≥ 2)
        - num = ∑_i [S_i · f_i],  den = ∑_i [S_i]
        - out = M*(num/den) + (1−M)*∑_i[f_i]
        Else does plain ∑_i[f_i].
        """
        # zoom all images
        zoomed_imgs  = [img.zoom_image_as_template(reference)
                        for img in images.containers]

        if not weight_overlap:
            out = reference.get_uniform_copy(0)
            for z in zoomed_imgs:
                out += z
            return out

        # 1) build coverage masks (1 inside each img's FOV, 0 outside)
        zoomed_masks = [
            img.get_uniform_copy(1).zoom_image_as_template(reference)
            for img in images.containers
        ]
        cov_arrs = [m.as_array() for m in zoomed_masks]
        coverage = sum(cov_arrs)                  # integer count
        overlap = (coverage >= 2)                 # boolean mask

        # 2) zoom sensitivities and pull raw arrays
        zoomed_sens = [
            s.zoom_image_as_template(reference)
            for s in sens_images.containers
        ]
        img_arrs  = [z.as_array() for z in zoomed_imgs]
        sens_arrs = [s.as_array() for s in zoomed_sens]

        # 3) numerator, denominator, simple sum
        num    = sum(f*s for f, s in zip(img_arrs, sens_arrs))  # ∑ S_i·f_i
        den    = sum(sens_arrs)                                 # ∑ S_i
        simple = sum(img_arrs)                                  # ∑ f_i

        # 4) merge
        combined = np.where(overlap, num/den, simple)

        out = reference.get_uniform_copy(0)
        out.fill(combined)
        return out


    @staticmethod
    def retrieve_original_images(combined_image, original_references):
        """
        Retrieves the original images from the combined image.
        
        Parameters:
            combined_image: The image obtained from combine_images.
            original_references: List of original image references.
            weight_overlap: If True, adjust for overlapping regions. Default is False (ignore weighting).
        
        Returns:
            original_images: List of images zoomed to the original references.
        """
        original_images = []

        for ref in original_references:
            original_images.append(combined_image.zoom_image_as_template(ref))

        return original_images

    def direct(self, images: BlockDataContainer, out=None):
        if out is None:
            out = self.range_geometry().allocate(0)
        
        out.fill(ImageCombineOperator.combine_images(self.reference, images))
        return out
    
    def adjoint(self, image, out=None):
        if out is None:
            out = BlockDataContainer(*[img.get_uniform_copy(0) for img in self.images.containers])
        
        original_images = ImageCombineOperator.retrieve_original_images(image, self.images)
        
        for img, container in zip(original_images, out.containers):
            container.fill(img)
        
        return out
    

###############################################################################################
### The following split the ImageCombineOperator into two separate operators:
###############################################################################################

class ImageResampleOperator(LinearOperator):
    """
    An operator to resample a BlockDataContainer of images onto a common, larger grid.

    This operator takes a set of images, each with its own geometry (size, offset),
    and calculates a combined geometry that can contain all of them. The `direct`
    method then resamples each input image into this common space. The output is
    a BlockDataContainer where each image is on the same grid, ready for further
    processing like summation.
    
    The `adjoint` operation performs the reverse: it takes a BlockDataContainer of
    images on the common grid and resamples each one back to its original geometry.
    """
    def __init__(self, images: BlockDataContainer):
        self.images = images

        # 1. Calculate the geometry of the combined reference image
        self.reference = ImageData()
        dim_xy = images.containers[0].dimensions()[1]
        dim_z = ImageResampleOperator.get_combined_length_voxels(images)
        
        # The z-offset of the combined image space is determined by the last image's offset.
        offset_z = -images.containers[-1].get_geometrical_info().get_offset()[2]
        
        # The x,y offset is set to 0 as per the original tested logic.
        offset_xy = 0 
        self.reference.initialise((dim_z, dim_xy, dim_xy), images.containers[0].voxel_sizes(), (offset_z, offset_xy, offset_xy))
        
        # Perform sanity checks
        assert all(img.voxel_sizes() == self.reference.voxel_sizes() for img in images.containers), \
            "All images must have the same voxel size as the reference"
        assert self.get_combined_length_voxels(images) == self.reference.dimensions()[0], \
            f"Combined image length {self.get_combined_length_voxels(images)} and reference Z-dimension " \
            f"{self.reference.dimensions()[0]} do not match."

        # 2. Define domain and range geometries for the operator
        domain_geometry = images
        # The range is a block of images, each with the combined reference geometry
        range_geometry = BlockDataContainer(*[self.reference.get_uniform_copy(0) for _ in images.containers])
        
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)
    
    @staticmethod
    def get_combined_length(images: BlockDataContainer):
        """Calculates the physical length of the combined image space."""
        offsets = [img.get_geometrical_info().get_offset()[2] for img in images.containers]
        lengths = [img.dimensions()[0] * img.voxel_sizes()[0] for img in images.containers]
        return max(offset + length for offset, length in zip(offsets, lengths)) - min(offsets)
    
    @staticmethod
    def get_combined_length_voxels(images: BlockDataContainer):
        """Calculates the length of the combined image space in voxels."""
        voxel_size = images.containers[0].voxel_sizes()[0]
        # Ensure all images have the same voxel size along the combination axis
        assert all(img.voxel_sizes()[0] == voxel_size for img in images.containers)
        
        length = ImageResampleOperator.get_combined_length(images)
        
        # Check if the total length is a near-integer multiple of the voxel size
        assert abs((length / voxel_size) % 1) < 1e-3 or abs(((length / voxel_size) % 1) - 1) < 1e-3
        return int(round(length / voxel_size))

    def direct(self, images: BlockDataContainer, out: BlockDataContainer = None):
        """
        Resamples each image in the input BlockDataContainer to the common reference grid.
        """
        if out is None:
            out = self.range_geometry.allocate(0)
        
        for i, img in enumerate(images.containers):
            # `zoom_image_as_template` handles the resampling/warping
            resampled_img = img.zoom_image_as_template(self.reference)
            out.containers[i].fill(resampled_img)
            
        return out
    
    def adjoint(self, warped_images: BlockDataContainer, out: BlockDataContainer = None):
        """
        Resamples each image from the common grid back to its original geometry.
        """
        if out is None:
            out = self.domain_geometry.allocate(0)
        
        for i, warped_img in enumerate(warped_images.containers):
            # Get the geometry of the original image to use as a template
            original_reference = self.images.containers[i]
            resampled_back = warped_img.zoom_image_as_template(original_reference)
            out.containers[i].fill(resampled_back)
            
        return out


class ImageSummationOperator(LinearOperator):
    """
    An operator to sum a BlockDataContainer of images into a single ImageData.
    
    This operator assumes all images in the input BlockDataContainer are already
    on the same grid (i.e., they are the output of ImageResampleOperator).
    
    If `weight_overlap=True`, it performs a weighted sum in regions where
    multiple images overlap, using provided sensitivity maps. Otherwise, it
    performs a simple addition.
    
    The `adjoint` operation is a broadcast: it takes a single image and creates
    a BlockDataContainer by copying it N times. This behavior matches the
    original combined operator's adjoint.
    """
    def __init__(self, domain_geometry: BlockDataContainer, weight_overlap: bool = False):
        self.weight_overlap = weight_overlap
        
        # All images in the domain are expected to have the same geometry
        self.reference = domain_geometry.containers[0].copy()
        
        range_geometry = self.reference.get_uniform_copy(0)
        
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)

    def direct(self, images: BlockDataContainer, sens_images: BlockDataContainer = None, out: ImageData = None):
        """
        Combines images from a BlockDataContainer into a single ImageData.
        """
        if out is None:
            out = self.range_geometry.allocate(0)
        
        # Case 1: Simple summation (no weighting)
        if not self.weight_overlap:
            summed_image = self.range_geometry.get_uniform_copy(0)
            for img in images.containers:
                summed_image += img
            out.fill(summed_image)
            return out

        # Case 2: Weighted summation for overlapping regions
        if sens_images is None:
            raise ValueError("Sensitivity images (`sens_images`) are required when `weight_overlap` is True.")
        
        # 1) Build coverage masks to find overlapping areas
        zoomed_masks = [
            img.get_uniform_copy(1) # Images are already zoomed, create masks from them
            for img in images.containers
        ]
        cov_arrs = [m.as_array() for m in zoomed_masks]
        coverage = sum(cov_arrs)  # Integer count of how many images cover each pixel
        overlap = (coverage >= 2) # Boolean mask where 2 or more images overlap

        # 2) Pull raw arrays for images and sensitivities
        img_arrs  = [z.as_array() for z in images.containers]
        sens_arrs = [s.as_array() for s in sens_images.containers]

        # 3) Calculate components for the weighted sum formula
        num    = sum(f*s for f, s in zip(img_arrs, sens_arrs))  # Numerator: ∑ (S_i * f_i)
        den    = sum(sens_arrs)                                 # Denominator: ∑ S_i
        # Handle division by zero in denominator, though unlikely with sens maps
        den[den == 0] = 1e-9
        simple_sum = sum(img_arrs)                             # Simple sum for non-overlap regions: ∑ f_i

        # 4) Merge the results based on the overlap mask
        # out = M * (num/den) + (1−M) * simple_sum
        combined_arr = np.where(overlap, num / den, simple_sum)
        
        out.fill(combined_arr)
        return out

    def adjoint(self, image: ImageData, out: BlockDataContainer = None):
        """
        Performs the adjoint operation, which is a broadcast.
        
        It takes a single image and populates a BlockDataContainer by filling
        each container with that image.
        """
        if out is None:
            out = self.domain_geometry.allocate(0)
        
        for container in out.containers:
            container.fill(image)
        
        return out
