import logging
import os
import re
from dataclasses import dataclass

import numpy as np
from cil.framework import BlockDataContainer, BlockGeometry
from cil.optimisation.operators import LinearOperator
from sirf.Reg import NiftyResample
from sirf.STIR import ImageData, TruncateToCylinderProcessor

from recon_core.utils.sirf import create_spect_uniform_image, get_array


class AdjointOperator(LinearOperator):
    """Very simple adjoint operator that reverses the roles of direct and adjoint methods.

    Args:
        operator (LinearOperator): The operator to be reversed.
    """

    def __init__(self, operator):
        self.operator = operator
        self.domain_geometry = operator.range_geometry
        self.range_geometry = operator.domain_geometry
        super().__init__(
            domain_geometry=self.domain_geometry,
            range_geometry=self.range_geometry,
        )

    def direct(self, x, out=None):
        return self.operator.adjoint(x, out)

    def adjoint(self, x, out=None):
        return self.operator.direct(x, out)


class ScalingOperator(LinearOperator):
    def __init__(self, scale, domain_geometry):
        super(ScalingOperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=domain_geometry
        )
        self.scale = scale

    def direct(self, x, out=None):
        """Scale the input image by a constant factor."""
        if out is None:
            return x * self.scale
        else:
            x.multiply(self.scale, out=out)
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
        out_arr = get_array(out)
        out_arr[-self.num_slices :, :, :] = 0
        out_arr[: self.num_slices, :, :] = 0
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
        out_arr = get_array(out)
        out_arr[np.isnan(out_arr)] = 0
        out.fill(out_arr)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)


class TruncationOperator(LinearOperator):
    """CIL Wrapper for SIRF TruncateToCylinderProcessor."""

    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry, range_geometry=domain_geometry)

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
    def __init__(self, anatomical_gradient, gamma=1, eta=1e-6):
        self.anatomical_gradient = anatomical_gradient
        geometry = BlockGeometry(
            *anatomical_gradient.containers
        )  # a little odd. Not sure why I did this...
        self.tmp = self.anatomical_gradient.containers[0].clone()

        self.gamma = gamma

        self.xi = (
            self.anatomical_gradient / (self.anatomical_gradient.pnorm().power(2) + eta**2).sqrt()
        )

        self.calculate_norm = lambda _: 1

        super(DirectionalOperator, self).__init__(
            domain_geometry=geometry,
            range_geometry=geometry,
        )

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


class FlipOperator(LinearOperator):
    def __init__(self, image, axis):
        self.axis = axis
        super().__init__(domain_geometry=image, range_geometry=image)

    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        out_arr = get_array(out)
        out_arr = np.flip(out_arr, axis=self.axis)
        out.fill(out_arr)
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)


def crop_central(volume: np.ndarray, size=(128, 128, 128)) -> np.ndarray:
    """
    Extract the central subvolume of given size from a 3D array,
    padding with zeros if the volume is smaller in any dimension.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D array.
    size : tuple of three ints, optional
        Desired output shape (depth, height, width). Default is (128, 128, 128).

    Returns
    -------
    np.ndarray
        Central subvolume of shape `size`, zero-padded as needed.
    """
    dz, dy, dx = volume.shape
    sz, sy, sx = size
    # compute centre indices
    cz, cy, cx = dz // 2, dy // 2, dx // 2
    # compute start indices (can be negative)
    start_z, start_y, start_x = cz - sz // 2, cy - sy // 2, cx - sx // 2
    end_z, end_y, end_x = start_z + sz, start_y + sy, start_x + sx

    # allocate output
    out = np.zeros((sz, sy, sx), dtype=volume.dtype)

    # clamp input region
    in_z0, in_z1 = max(0, start_z), min(dz, end_z)
    in_y0, in_y1 = max(0, start_y), min(dy, end_y)
    in_x0, in_x1 = max(0, start_x), min(dx, end_x)

    # corresponding output region
    out_z0 = in_z0 - start_z
    out_y0 = in_y0 - start_y
    out_x0 = in_x0 - start_x
    out_z1 = out_z0 + (in_z1 - in_z0)
    out_y1 = out_y0 + (in_y1 - in_y0)
    out_x1 = out_x0 + (in_x1 - in_x0)

    out[out_z0:out_z1, out_y0:out_y1, out_x0:out_x1] = volume[in_z0:in_z1, in_y0:in_y1, in_x0:in_x1]
    return out


class EnlargementOperator(LinearOperator):
    """Operator for enlarging images with zero-padding."""

    def __init__(self, enlarged_shape, enlargement_sino, original_floating):
        self.enlarged_shape = enlarged_shape
        self.enlargement_sino = enlargement_sino
        self.original_floating = original_floating

        # Create the range geometry (enlarged image)
        self.enlarged_image = create_spect_uniform_image(
            self.enlargement_sino,
            dims=self.enlarged_shape,
        )

        # Initialize the LinearOperator with proper geometries
        super().__init__(domain_geometry=self.original_floating, range_geometry=self.enlarged_image)

    def direct(self, x, out=None):
        """Enlarge input by padding with zeros."""
        # logging.info(f"Enlarging image from {self.original_floating.shape} to {self.enlarged_shape}")
        self.enlarged_image.fill(crop_central(get_array(x), size=self.enlarged_shape))
        return self._project_and_fill(self.enlarged_image, out)

    def adjoint(self, x, out=None):
        """Crop back to original size."""
        # logging.info(f"Cropping image from {self.enlarged_shape} to {self.original_floating.shape}")
        x_array = get_array(x)
        cropped_array = crop_central(x_array, size=self.original_floating.shape)
        result = self.original_floating.copy()
        result.fill(cropped_array)
        return self._project_and_fill(result, out)

    def _project_and_fill(self, res, out):
        if out is not None:
            out.fill(res)
            return out
        return res


@dataclass(frozen=True)
class _AxisMapping:
    size: int
    idx_low: np.ndarray
    idx_high: np.ndarray
    weight_high: np.ndarray
    valid_low: np.ndarray
    valid_high: np.ndarray
    
    
class ZoomOperator(LinearOperator):
    """Operator for zooming images."""

    def __init__(self, zoom_factors, input_geometry, target_voxel_sizes=None):
        self.zoom_factors = zoom_factors
        self.inv_zoom_factors = tuple(1.0 / z for z in self.zoom_factors)
        self.input_geometry = input_geometry

        # Create the range geometry (zoomed image)
        self.zoomed_geometry = input_geometry.zoom_image(
            self.zoom_factors, scaling="preserve_projections"
        )

        # Initialize the LinearOperator with proper geometries
        super().__init__(domain_geometry=self.input_geometry, range_geometry=self.zoomed_geometry)

        # Calculate scaling factor for proper adjoint
        if target_voxel_sizes is not None:
            # If user specified target voxel sizes, check if they match
            input_voxel_sizes = input_geometry.voxel_sizes()
            zoomed_voxel_sizes = tuple(input_voxel_sizes[i] / zoom_factors[i] for i in range(3))

            voxel_match = all(
                abs(zoomed_voxel_sizes[i] - target_voxel_sizes[i]) < 1e-6 for i in range(3)
            )
            if voxel_match:
                self.scale = 1.0  # No scaling needed!
                print("Zoom factors chosen to match target voxel sizes exactly - scale = 1.0")
            else:
                # Scale based on voxel volume difference
                input_voxel_volume = (
                    input_voxel_sizes[0] * input_voxel_sizes[1] * input_voxel_sizes[2]
                )
                target_voxel_volume = (
                    target_voxel_sizes[0] * target_voxel_sizes[1] * target_voxel_sizes[2]
                )
                self.scale = target_voxel_volume / input_voxel_volume
        else:
            # Standard zoom scaling based on volume change
            zoom_volume_factor = zoom_factors[0] * zoom_factors[1] * zoom_factors[2]
            self.scale = 1.0 / zoom_volume_factor

    def direct(self, x, out=None):
        """Apply zoom transformation."""
        # logging.info(f"Zoom factors: {self.zoom_factors}, scale: {self.scale}")
        result = x.zoom_image(self.zoom_factors, scaling="preserve_projections")
        return self._project_and_fill(result, out)

    def adjoint(self, x, out=None):
        """Apply inverse zoom transformation."""
        # logging.info(f"Inverse zoom factors: {self.zoom_factors}, scale: {self.scale}")
        result = x.zoom_image(self.inv_zoom_factors, scaling="preserve_projections") * self.scale
        return self._project_and_fill(result, out)

    def _project_and_fill(self, res, out):
        if out is not None:
            out.fill(res)
            return out
        return res


class ZoomOperatorAdjoint(LinearOperator):
    """Pure Python zoom that keeps array dimensions fixed while adjusting voxel sizes."""

    def __init__(self, zoom_factors, input_geometry):
        self.domain_template = input_geometry
        self.zoom_factors = tuple(float(factor) for factor in zoom_factors)
        self.domain_shape = tuple(int(dim) for dim in input_geometry.dimensions())
        self.range_template = self._make_range_template()

        super().__init__(domain_geometry=self.domain_template, range_geometry=self.range_template)

        self.axis_maps = tuple(
            self._build_axis_mapping(self.domain_shape[idx], self.zoom_factors[idx])
            for idx in range(3)
        )
        det = float(np.prod(self.zoom_factors))
        if det == 0:
            raise ValueError("Zoom factors must not include zeros.")
        # Normalise so zoom keeps L2 norm (and therefore scale) roughly constant.
        self.norm_scale = np.sqrt(abs(det))

    def _make_range_template(self):
        zero = self.domain_template.get_uniform_copy(0)
        template = zero.zoom_image(self.zoom_factors, scaling="preserve_projections")
        template.fill(0)
        return template

    @staticmethod
    def _build_axis_mapping(size: int, zoom_factor: float) -> _AxisMapping:
        if zoom_factor == 0:
            raise ValueError("Zoom factor must be non-zero.")
        centre = (size - 1) / 2.0
        positions = (np.arange(size, dtype=np.float32) - centre) / zoom_factor + centre
        idx_low = np.floor(positions).astype(np.int64)
        idx_high = idx_low + 1
        valid_low = (idx_low >= 0) & (idx_low < size)
        valid_high = (idx_high >= 0) & (idx_high < size)
        idx_low = np.clip(idx_low, 0, size - 1)
        idx_high = np.clip(idx_high, 0, size - 1)
        weight_high = (positions - np.floor(positions)).astype(np.float32)
        return _AxisMapping(
            size=size,
            idx_low=idx_low,
            idx_high=idx_high,
            weight_high=weight_high,
            valid_low=valid_low.astype(np.float32),
            valid_high=valid_high.astype(np.float32),
        )

    def direct(self, x, out=None):
        arr = np.asarray(get_array(x), dtype=np.float32)
        zoomed = self._apply_separable_interp(arr)
        zoomed /= self.norm_scale
        return self._project_and_fill(zoomed, out, self.range_template)

    def adjoint(self, x, out=None):
        arr = np.asarray(get_array(x), dtype=np.float32)
        back = self._apply_separable_adjoint(arr)
        back /= self.norm_scale
        return self._project_and_fill(back, out, self.domain_template)

    def _apply_separable_interp(self, arr: np.ndarray) -> np.ndarray:
        result = arr
        for axis, axis_map in enumerate(self.axis_maps):
            result = self._interp_along_axis(result, axis, axis_map)
        return result

    def _apply_separable_adjoint(self, arr: np.ndarray) -> np.ndarray:
        result = arr
        for axis in reversed(range(len(self.axis_maps))):
            result = self._interp_adjoint_along_axis(result, axis, self.axis_maps[axis])
        return result

    @staticmethod
    def _interp_along_axis(data: np.ndarray, axis: int, axis_map: _AxisMapping) -> np.ndarray:
        data = np.moveaxis(data, axis, 0)
        size, *rest = data.shape
        if size != axis_map.size:
            raise ValueError(f"Axis {axis} size mismatch: expected {axis_map.size}, got {size}")
        flat = data.reshape(size, -1)
        low = flat[axis_map.idx_low] * axis_map.valid_low[:, None]
        high = flat[axis_map.idx_high] * axis_map.valid_high[:, None]
        w_high = axis_map.weight_high[:, None]
        w_low = 1.0 - w_high
        out_flat = low * w_low + high * w_high
        out = out_flat.reshape((size,) + tuple(rest))
        return np.moveaxis(out, 0, axis)

    @staticmethod
    def _interp_adjoint_along_axis(data: np.ndarray, axis: int, axis_map: _AxisMapping) -> np.ndarray:
        data = np.moveaxis(data, axis, 0)
        size, *rest = data.shape
        if size != axis_map.size:
            raise ValueError(f"Axis {axis} size mismatch: expected {axis_map.size}, got {size}")
        flat = data.reshape(size, -1)
        out_flat = np.zeros((size, flat.shape[1]), dtype=flat.dtype)
        contrib_low = flat * (1.0 - axis_map.weight_high)[:, None]
        contrib_high = flat * axis_map.weight_high[:, None]
        contrib_low *= axis_map.valid_low[:, None]
        contrib_high *= axis_map.valid_high[:, None]
        np.add.at(out_flat, axis_map.idx_low, contrib_low)
        np.add.at(out_flat, axis_map.idx_high, contrib_high)
        out = out_flat.reshape((size,) + tuple(rest))
        return np.moveaxis(out, 0, axis)

    def _project_and_fill(self, array: np.ndarray, out, template: ImageData):
        array = np.ascontiguousarray(array)
        if out is None:
            out = template.get_uniform_copy(0)
        out.fill(array)
        return out


class NiftyResampleOperator(LinearOperator):
    """Pure registration operator without zoom complications."""

    def __init__(self, reference, floating, transform, assume_matched_voxels=False):
        self.reference = reference.get_uniform_copy(0)
        self.floating = floating
        self.transform = transform
        self.assume_matched_voxels = assume_matched_voxels

        # Initialize the LinearOperator with proper geometries
        super().__init__(domain_geometry=self.floating, range_geometry=self.reference)

        self.resampler = NiftyResample()
        self.resampler.set_reference_image(self.reference)
        self.resampler.set_floating_image(self.floating)
        self.resampler.set_interpolation_type_to_linear()
        self.resampler.set_padding_value(0)
        self.resampler.add_transformation(self.transform)

        if assume_matched_voxels:
            # If voxels are already matched, keep scale unity
            self.scale = 1.0
            print("Assuming voxel sizes are matched - using scale = 1.0")
        else:
            # Calculate scaling factor for different voxel sizes
            vx_ref = self.reference.voxel_sizes()
            vx_flt = self.floating.voxel_sizes()
            self.scale = np.sqrt((vx_ref[0] * vx_ref[1] * vx_ref[2]) / (vx_flt[0] * vx_flt[1] * vx_flt[2]))
            print(f"Voxel size scaling factor: {self.scale}")

    def direct(self, x, out=None):
        """Forward registration transformation."""
        result = self.resampler.forward(x) * self.scale
        return self._project_and_fill(result, out)

    def adjoint(self, x, out=None):
        """Adjoint registration transformation."""
        result = self.resampler.backward(x) * self.scale
        return self._project_and_fill(result, out)

    def _project_and_fill(self, res, out):
        if out is not None:
            out.fill(res)
            return out
        return res


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

    def __init__(self, image, shift, path=""):
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
        self.path = path
        # need to create range geometry by shifting the image
        range_geometry = self.initialise_shift(image)
        super().__init__(domain_geometry=image, range_geometry=range_geometry)

        self.unshifted_image = image.copy()
        self.shifted_image = range_geometry.copy()

    def initialise_shift(self, x):
        """
        Apply the couch shift using an isolated temp directory.
        Returns a new ImageData with updated geometry if out is None.
        If out is provided, copies voxel data into out (geometry unchanged).
        """

        # writer will place the paired .v alongside
        shift_path = os.path.join(self.path, f"shifted{self.shift}.hv")

        x.write(shift_path)
        self.modify_pixel_offset(shift_path, self.shift, 3)

        return ImageData(shift_path)

    def direct(self, x, out=None):
        x_arr = get_array(x)
        if out is not None:
            out.fill(x_arr)
            return out
        else:
            self.shifted_image.fill(x_arr)
            return self.shifted_image.copy()

    def adjoint(self, x, out=None):
        x_arr = get_array(x)
        if out is not None:
            out.fill(x_arr)
            return out
        else:
            self.unshifted_image.fill(x_arr)
            return self.unshifted_image.copy()

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
        try:
            # Read the file content
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Modify the specific line
            for i, line in enumerate(lines):
                if line.strip().startswith(f"first pixel offset (mm) [{pixel_index}] :="):
                    lines[i] = f"first pixel offset (mm) [{pixel_index}] := {new_offset}\n"
                    break

            # Write the updated content back to the file
            with open(file_path, "w") as file:
                file.writelines(lines)
        except Exception as e:
            raise RuntimeError(f"Failed to modify the file {file_path}: {e}")

        return ImageData(file_path)

    @staticmethod
    def get_couch_shift_from_header(header_filepath):
        start_horizontal_bed_position = None

        # Read the file and extract the desired value
        with open(header_filepath, "r") as file:
            for line in file:
                if line.startswith("start horizontal bed position (mm) :="):
                    # Extract the value after ":="
                    start_horizontal_bed_position = float(line.split(":=")[1].strip())

        if start_horizontal_bed_position is None:
            raise ValueError(
                "Could not find 'start horizontal bed position (mm)' in the sinogram file."
            )

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
        self,
        images: BlockDataContainer,
        weight_overlap: bool = False,
        sens_images: BlockDataContainer | None = None,
    ):
        self.images = images
        self.weight_overlap = weight_overlap
        self.sens_images = sens_images

        self.resample_op = ImageResampleOperator(images)
        self.resampled_block = self.resample_op.range_geometry()
        self.summation_op = ImageSummationOperator(
            self.resampled_block, weight_overlap=weight_overlap
        )

        super().__init__(
            domain_geometry=images,
            range_geometry=self.summation_op.range_geometry,
        )

    @staticmethod
    def get_combined_length(images):
        offsets = [img.get_geometrical_info().get_offset()[2] for img in images.containers]
        lengths = [img.dimensions()[0] * img.voxel_sizes()[0] for img in images.containers]

        return max(offset + length for offset, length in zip(offsets, lengths)) - min(offsets)

    @staticmethod
    def get_combined_length_voxels(images):
        voxel_size = images.containers[0].voxel_sizes()[0]
        assert all(
            img.voxel_sizes() == images.containers[0].voxel_sizes() for img in images.containers
        )

        length = ImageCombineOperator.get_combined_length(images)

        assert (length / voxel_size) % 1 < 1.001
        return int(round(length / voxel_size))

    def direct(self, images: BlockDataContainer, out=None):
        resampled = self.resample_op.direct(images)
        if self.weight_overlap and self.sens_images is None:
            raise ValueError("Sensitivity images must be set for weighted combination.")

        combined = self.summation_op.direct(resampled, sens_images=self.sens_images if self.weight_overlap else None)
        if out is not None:
            out.fill(combined)
            return out
        return combined

    def adjoint(self, image, out=None):
        resampled_adj = self.summation_op.adjoint(image)
        original_space = self.resample_op.adjoint(resampled_adj)
        if out is not None:
            out.fill(original_space)
            return out
        return original_space

    def set_sensitivities(self, sens_images: BlockDataContainer | None):
        self.sens_images = sens_images
        self.weight_overlap = sens_images is not None
        self.summation_op.weight_overlap = self.weight_overlap
        self.summation_op.set_sensitivities(sens_images)


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
        self.reference.initialise(
            (dim_z, dim_xy, dim_xy),
            images.containers[0].voxel_sizes(),
            (offset_z, offset_xy, offset_xy),
        )

        # Perform sanity checks
        assert all(
            img.voxel_sizes() == self.reference.voxel_sizes() for img in images.containers
        ), "All images must have the same voxel size as the reference"
        assert self.get_combined_length_voxels(images) == self.reference.dimensions()[0], (
            f"Combined image length {self.get_combined_length_voxels(images)} and "
            f"reference Z-dimension "
            f"{self.reference.dimensions()[0]} do not match."
        )

        # 2. Define domain and range geometries for the operator
        domain_geometry = images
        # The range is a block of images, each with the combined reference geometry
        range_geometry = BlockDataContainer(
            *[self.reference.get_uniform_copy(0) for _ in images.containers]
        )

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

        return int(round(length / voxel_size))

    def direct(self, images: BlockDataContainer, out: BlockDataContainer | None = None):
        """
        Resamples each image in the input BlockDataContainer to the common reference grid.
        """
        if out is None:
            out = BlockDataContainer(
                *[container.get_uniform_copy(0) for container in self.range_geometry().containers]
            )

        for i, img in enumerate(images.containers):
            # `zoom_image_as_template` handles the resampling/warping
            resampled_img = img.zoom_image_as_template(self.reference)
            out.containers[i].fill(resampled_img)

        return out

    def adjoint(self, warped_images: BlockDataContainer, out: BlockDataContainer | None = None):
        """
        Resamples each image from the common grid back to its original geometry.
        """
        if out is None:
            out = BlockDataContainer(
                *[container.get_uniform_copy(0) for container in self.domain_geometry().containers]
            )

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
        self.sens_images = None

        # All images in the domain are expected to have the same geometry
        self.reference = domain_geometry.containers[0].copy()

        range_geometry = self.reference.get_uniform_copy(0)

        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)

    def direct(
        self,
        images: BlockDataContainer,
        sens_images: BlockDataContainer | None = None,
        out: ImageData | None = None,
    ):
        """
        Combines images from a BlockDataContainer into a single ImageData.
        """
        if out is None:
            out = self.range_geometry().get_uniform_copy(0)

        # Case 1: Simple summation (no weighting)
        if not self.weight_overlap:
            summed_image = self.range_geometry().get_uniform_copy(0)
            for img in images.containers:
                summed_image += img
            out.fill(summed_image)
            return out

        # Case 2: Weighted summation for overlapping regions
        if sens_images is None:
            sens_images = self.sens_images
        if sens_images is None:
            raise ValueError(
                "Sensitivity images (`sens_images`) are required when `weight_overlap` is True."
            )

        # 1) Pull raw arrays for images and sensitivities
        img_arrs = [get_array(z) for z in images.containers]
        sens_arrs = [get_array(s) for s in sens_images.containers]

        # 2) Calculate components for the weighted sum formula
        num = sum(f * s for f, s in zip(img_arrs, sens_arrs))  # Numerator: ∑ (S_i * f_i)
        den = sum(sens_arrs)  # Denominator: ∑ S_i
        simple_sum = sum(img_arrs)  # Simple sum for non-overlap regions: ∑ f_i

        mask = den > 0
        weighted = np.zeros_like(den, dtype=np.float32)
        np.divide(num, den, where=mask, out=weighted)
        combined_arr = np.where(mask, weighted, simple_sum)

        out.fill(combined_arr)
        return out

    def adjoint(self, image: ImageData, out: BlockDataContainer | None = None):
        """
        Performs the adjoint operation, which is a broadcast.

        It takes a single image and populates a BlockDataContainer by filling
        each container with that image.
        """
        if out is None:
            out = BlockDataContainer(
                *[container.get_uniform_copy(0) for container in self.domain_geometry().containers]
            )

        if not self.weight_overlap:
            for container in out.containers:
                container.fill(image)
            return out

        sens_images = self.sens_images
        if sens_images is None:
            raise ValueError("Sensitivity images must be set for weighted adjoint.")

        image_arr = get_array(image)
        sens_arrs = [get_array(s) for s in sens_images.containers]
        den = sum(sens_arrs)
        mask = den > 0
        safe_den = np.where(mask, den, 1.0)

        for container, sens in zip(out.containers, sens_arrs):
            contrib = np.where(mask, image_arr * sens / safe_den, image_arr)
            container.fill(contrib)

        return out

    def set_sensitivities(self, sens_images: BlockDataContainer | None):
        self.sens_images = sens_images
