"""
Interactive Jupyter notebook utilities for image viewing and mask creation.

This module provides tools for:
- Viewing 2D slices and 3D volumes in Jupyter notebooks using SIRF.STIR ImageData
- Creating spatial masks interactively using matplotlib widgets
- Saving and loading masks for batch processing
- Batch applying masks across multiple images
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle, Ellipse
    from matplotlib.widgets import Button, Slider
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import sirf.STIR as pet
    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False
    pet = None

from .interfile import load_image, get_image_array, get_image_shape


def _check_dependencies():
    """Check if required dependencies are available."""
    if not HAS_NUMPY:
        raise ImportError(
            "NumPy is required for interactive Jupyter utilities. "
            "Install with: pip install numpy"
        )
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for interactive Jupyter utilities. "
            "Install with: pip install matplotlib"
        )
    if not HAS_SIRF:
        raise ImportError(
            "SIRF.STIR is required for image loading. "
            "Install SIRF following: https://github.com/SyneRBI/SIRF/wiki"
        )


@dataclass
class SpatialMask:
    """
    A spatial mask defined by regions of interest (ROIs).

    Supports multiple ROI types:
    - 'polygon': List of (x, y) coordinates defining a polygon on a slice
    - 'rectangle': (x_min, y_min, z_min, x_max, y_max, z_max) bounding box
    - 'ellipse': 2D ellipse on a specific slice (center_x, center_y, radius_x, radius_y, angle, slice)
    - 'ellipsoid': 3D ellipsoid (center_x, center_y, center_z, radius_x, radius_y, radius_z, angle_x, angle_y, angle_z)
    - 'sphere': (center_x, center_y, center_z, radius)

    For ellipsoids, angle_x, angle_y, angle_z are rotation angles (degrees) around each axis.
    """

    shape: Tuple[int, ...]
    rois: List[Dict[str, Any]]
    name: str = "mask"

    def to_flat_mask(self, total_voxels: Optional[int] = None) -> List[bool]:
        """
        Convert spatial ROIs to a flat boolean mask.

        Returns a list of booleans matching the flattened voxel buffer.
        """
        _check_dependencies()

        if total_voxels is None:
            total_voxels = math.prod(self.shape)

        if len(self.shape) == 2:
            return self._to_flat_mask_2d(total_voxels)
        elif len(self.shape) == 3:
            return self._to_flat_mask_3d(total_voxels)
        else:
            raise ValueError(f"Unsupported shape dimensions: {len(self.shape)}")

    def to_array_mask(self) -> np.ndarray:
        """
        Convert spatial ROIs to a numpy boolean array matching the image shape.

        Returns:
            Boolean numpy array with shape matching the original image
        """
        _check_dependencies()
        flat_mask = self.to_flat_mask()
        return np.array(flat_mask).reshape(self.shape)

    def to_image_mask(self, template_image) -> Any:
        """
        Convert spatial ROIs to a SIRF ImageData mask (1s and 0s).

        Args:
            template_image: SIRF ImageData to use as template

        Returns:
            SIRF ImageData with 1s inside ROIs, 0s outside
        """
        _check_dependencies()

        # Get the mask as a numpy array (float32 for SIRF)
        mask_array = self.to_array_mask().astype(np.float32)

        # Create a new SIRF image from the template
        mask_image = template_image.clone()
        mask_image.fill(mask_array)

        return mask_image

    def _to_flat_mask_2d(self, total_voxels: int) -> List[bool]:
        """Generate flat mask for 2D images."""
        height, width = self.shape
        mask = np.zeros((height, width), dtype=bool)

        for roi in self.rois:
            roi_type = roi.get("type")

            if roi_type == "rectangle":
                x_min, y_min = roi["x_min"], roi["y_min"]
                x_max, y_max = roi["x_max"], roi["y_max"]
                x_min, x_max = int(round(x_min)), int(round(x_max))
                y_min, y_max = int(round(y_min)), int(round(y_max))
                mask[y_min:y_max, x_min:x_max] = True

            elif roi_type == "polygon":
                from matplotlib.path import Path as MplPath
                points = np.array(roi["points"])
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
                path = MplPath(points)
                mask_flat = path.contains_points(coords)
                mask = mask | mask_flat.reshape(height, width)

            elif roi_type == "ellipse":
                cx, cy = roi["center_x"], roi["center_y"]
                rx, ry = roi["radius_x"], roi["radius_y"]
                angle = roi.get("angle", 0)  # angle in degrees

                # Create coordinate grids
                y_coords, x_coords = np.mgrid[0:height, 0:width]

                # Translate to center
                x_shifted = x_coords - cx
                y_shifted = y_coords - cy

                # Rotate coordinates
                angle_rad = np.radians(angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                x_rot = x_shifted * cos_a + y_shifted * sin_a
                y_rot = -x_shifted * sin_a + y_shifted * cos_a

                # Check if inside ellipse
                ellipse_mask = (x_rot / rx)**2 + (y_rot / ry)**2 <= 1
                mask = mask | ellipse_mask

        return mask.ravel().tolist()

    def _to_flat_mask_3d(self, total_voxels: int) -> List[bool]:
        """Generate flat mask for 3D volumes."""
        depth, height, width = self.shape
        mask = np.zeros((depth, height, width), dtype=bool)

        for roi in self.rois:
            roi_type = roi.get("type")

            if roi_type == "rectangle":
                x_min, y_min, z_min = roi["x_min"], roi["y_min"], roi["z_min"]
                x_max, y_max, z_max = roi["x_max"], roi["y_max"], roi["z_max"]
                x_min, x_max = int(round(x_min)), int(round(x_max))
                y_min, y_max = int(round(y_min)), int(round(y_max))
                z_min, z_max = int(round(z_min)), int(round(z_max))
                mask[z_min:z_max, y_min:y_max, x_min:x_max] = True

            elif roi_type == "polygon":
                # Polygon on specific slice
                from matplotlib.path import Path as MplPath
                points = np.array(roi["points"])
                slice_idx = roi.get("slice", depth // 2)
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
                path = MplPath(points)
                mask_flat = path.contains_points(coords)
                mask[slice_idx, :, :] = mask[slice_idx, :, :] | mask_flat.reshape(height, width)

            elif roi_type == "ellipse":
                # 2D ellipse on a specific slice
                cx, cy = roi["center_x"], roi["center_y"]
                rx, ry = roi["radius_x"], roi["radius_y"]
                angle = roi.get("angle", 0)
                slice_idx = roi.get("slice", depth // 2)

                # Create coordinate grids for the slice
                y_coords, x_coords = np.mgrid[0:height, 0:width]

                # Translate to center
                x_shifted = x_coords - cx
                y_shifted = y_coords - cy

                # Rotate coordinates
                angle_rad = np.radians(angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                x_rot = x_shifted * cos_a + y_shifted * sin_a
                y_rot = -x_shifted * sin_a + y_shifted * cos_a

                # Check if inside ellipse
                ellipse_mask = (x_rot / rx)**2 + (y_rot / ry)**2 <= 1
                mask[slice_idx, :, :] = mask[slice_idx, :, :] | ellipse_mask

            elif roi_type == "ellipsoid":
                # 3D ellipsoid with optional rotation
                cx, cy, cz = roi["center_x"], roi["center_y"], roi["center_z"]
                rx, ry, rz = roi["radius_x"], roi["radius_y"], roi["radius_z"]

                # Get rotation angles (default to 0 if not specified)
                angle_x = np.radians(roi.get("angle_x", 0))
                angle_y = np.radians(roi.get("angle_y", 0))
                angle_z = np.radians(roi.get("angle_z", 0))

                # Create coordinate grids
                z_coords, y_coords, x_coords = np.mgrid[0:depth, 0:height, 0:width]

                # Translate to center
                x_shifted = x_coords - cx
                y_shifted = y_coords - cy
                z_shifted = z_coords - cz

                # Apply rotation if any angles are non-zero
                if angle_x != 0 or angle_y != 0 or angle_z != 0:
                    # Rotation matrices
                    # Rotation around X axis
                    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
                    y_rot_x = y_shifted * cos_x - z_shifted * sin_x
                    z_rot_x = y_shifted * sin_x + z_shifted * cos_x
                    x_rot_x = x_shifted

                    # Rotation around Y axis
                    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
                    x_rot_xy = x_rot_x * cos_y + z_rot_x * sin_y
                    z_rot_xy = -x_rot_x * sin_y + z_rot_x * cos_y
                    y_rot_xy = y_rot_x

                    # Rotation around Z axis
                    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
                    x_rot = x_rot_xy * cos_z - y_rot_xy * sin_z
                    y_rot = x_rot_xy * sin_z + y_rot_xy * cos_z
                    z_rot = z_rot_xy
                else:
                    x_rot, y_rot, z_rot = x_shifted, y_shifted, z_shifted

                # Check if inside ellipsoid
                ellipsoid_mask = (x_rot / rx)**2 + (y_rot / ry)**2 + (z_rot / rz)**2 <= 1
                mask = mask | ellipsoid_mask

            elif roi_type == "sphere":
                cx, cy, cz, radius = roi["center_x"], roi["center_y"], roi["center_z"], roi["radius"]
                z_coords, y_coords, x_coords = np.mgrid[0:depth, 0:height, 0:width]
                distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2 + (z_coords - cz)**2)
                mask = mask | (distances <= radius)

        return mask.ravel().tolist()

    def save(self, path: Union[str, Path]) -> Path:
        """Save mask to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2)

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> SpatialMask:
        """Load mask from JSON file."""
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        return cls(
            shape=tuple(data["shape"]),
            rois=data["rois"],
            name=data.get("name", "mask")
        )


class ImageViewer:
    """
    Interactive image viewer for Jupyter notebooks with slice navigation and mask creation.
    Works directly with SIRF.STIR ImageData objects.
    """

    def __init__(self, image, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the interactive viewer.

        Args:
            image: SIRF.STIR ImageData object or path to image file
            figsize: Figure size (width, height)
        """
        _check_dependencies()

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = load_image(image)

        self.image = image
        self.figsize = figsize

        # Convert to numpy array
        self.data = get_image_array(image)
        self.shape = self.data.shape
        self.ndim = len(self.shape)

        if self.ndim == 2:
            self.current_slice = 0
            self.max_slice = 0
        elif self.ndim == 3:
            self.current_slice = self.shape[0] // 2
            self.max_slice = self.shape[0] - 1
        else:
            raise ValueError(f"Unsupported image dimensions: {self.ndim}")

        # ROI tracking
        self.rois: List[Dict[str, Any]] = []
        self.current_roi_points: List[Tuple[float, float]] = []
        self.current_patch: Optional[Any] = None
        self.drawing_mode: Optional[str] = None  # 'polygon', 'rectangle', 'ellipse', None
        self.current_angle: float = 0.0  # Current rotation angle for ellipse

        # UI elements
        self.fig = None
        self.ax = None
        self.im = None
        self.slider = None
        self.cid_click = None
        self.cid_key = None

    def show(self) -> ImageViewer:
        """Display the interactive viewer."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        plt.subplots_adjust(bottom=0.25)

        # Display initial slice
        self._update_display()

        # Add slice slider for 3D volumes
        if self.ndim == 3:
            ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
            self.slider = Slider(
                ax_slider, 'Slice', 0, self.max_slice,
                valinit=self.current_slice, valstep=1
            )
            self.slider.on_changed(self._on_slice_change)

        # Add buttons
        ax_rect = plt.axes([0.1, 0.08, 0.12, 0.04])
        btn_rect = Button(ax_rect, 'Rectangle')
        btn_rect.on_clicked(lambda event: self._start_drawing('rectangle'))

        ax_ellipse = plt.axes([0.23, 0.08, 0.12, 0.04])
        btn_ellipse = Button(ax_ellipse, 'Ellipse')
        btn_ellipse.on_clicked(lambda event: self._start_drawing('ellipse'))

        ax_poly = plt.axes([0.36, 0.08, 0.12, 0.04])
        btn_poly = Button(ax_poly, 'Polygon')
        btn_poly.on_clicked(lambda event: self._start_drawing('polygon'))

        ax_clear = plt.axes([0.55, 0.08, 0.12, 0.04])
        btn_clear = Button(ax_clear, 'Clear')
        btn_clear.on_clicked(self._clear_rois)

        ax_undo = plt.axes([0.68, 0.08, 0.12, 0.04])
        btn_undo = Button(ax_undo, 'Undo')
        btn_undo.on_clicked(self._undo_last_roi)

        ax_stats = plt.axes([0.81, 0.08, 0.12, 0.04])
        btn_stats = Button(ax_stats, 'Stats')
        btn_stats.on_clicked(self._show_stats)

        # Connect mouse and keyboard events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()
        return self

    def _update_display(self):
        """Update the image display."""
        if self.ndim == 2:
            slice_data = self.data
        else:
            slice_data = self.data[self.current_slice, :, :]

        if self.im is None:
            self.im = self.ax.imshow(slice_data, cmap='gray', interpolation='nearest')
            self.ax.set_title(self._get_title())
            plt.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_data(slice_data)
            self.ax.set_title(self._get_title())

        # Redraw ROIs
        self._redraw_rois()

        if self.fig:
            self.fig.canvas.draw_idle()

    def _get_title(self) -> str:
        """Get the title for the current view."""
        if self.ndim == 2:
            return f"Shape: {self.shape} | ROIs: {len(self.rois)}"
        else:
            return f"Slice: {self.current_slice}/{self.max_slice} | Shape: {self.shape} | ROIs: {len(self.rois)}"

    def _on_slice_change(self, val):
        """Handle slice slider change."""
        self.current_slice = int(val)
        self._update_display()

    def _start_drawing(self, mode: str):
        """Start drawing an ROI."""
        self.drawing_mode = mode
        self.current_roi_points = []
        self.current_angle = 0.0
        if mode == 'ellipse':
            print(f"Started {mode} mode. Click center, then click for radii.")
            print("  Use Left/Right arrows to rotate. Press 'Enter' when done, 'Esc' to cancel.")
        else:
            print(f"Started {mode} mode. Click to add points. Press 'Enter' to finish or 'Esc' to cancel.")

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax or self.drawing_mode is None:
            return

        x, y = event.xdata, event.ydata

        if self.drawing_mode == 'rectangle':
            self.current_roi_points.append((x, y))
            if len(self.current_roi_points) == 2:
                self._finish_rectangle()

        elif self.drawing_mode == 'ellipse':
            self.current_roi_points.append((x, y))
            if len(self.current_roi_points) == 1:
                # Just set center, wait for second click or rotation
                self._update_preview()
            elif len(self.current_roi_points) == 2:
                # Second click sets radii, but don't finish yet - allow rotation
                self._update_preview()
                print(f"  Ellipse preview shown. Use Left/Right arrows to rotate, Enter to finish.")

        elif self.drawing_mode == 'polygon':
            self.current_roi_points.append((x, y))
            self._update_preview()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'enter':
            if self.drawing_mode == 'polygon':
                self._finish_polygon()
            elif self.drawing_mode == 'ellipse' and len(self.current_roi_points) == 2:
                self._finish_ellipse()
        elif event.key == 'escape':
            self._cancel_drawing()
        elif event.key == 'left' and self.drawing_mode == 'ellipse' and len(self.current_roi_points) == 2:
            # Rotate ellipse counter-clockwise
            self.current_angle -= 5.0
            self._update_preview()
            print(f"  Angle: {self.current_angle:.1f}°")
        elif event.key == 'right' and self.drawing_mode == 'ellipse' and len(self.current_roi_points) == 2:
            # Rotate ellipse clockwise
            self.current_angle += 5.0
            self._update_preview()
            print(f"  Angle: {self.current_angle:.1f}°")

    def _update_preview(self):
        """Update the preview of the current ROI being drawn."""
        if self.current_patch:
            self.current_patch.remove()

        if len(self.current_roi_points) < 1:
            return

        if self.drawing_mode == 'polygon' and len(self.current_roi_points) >= 2:
            self.current_patch = Polygon(
                self.current_roi_points, fill=False,
                edgecolor='red', linewidth=2, linestyle='--'
            )
            self.ax.add_patch(self.current_patch)

        elif self.drawing_mode == 'ellipse':
            if len(self.current_roi_points) == 1:
                # Show center point
                (cx, cy) = self.current_roi_points[0]
                from matplotlib.patches import Circle
                self.current_patch = Circle(
                    (cx, cy), radius=2, fill=True,
                    edgecolor='red', facecolor='red'
                )
                self.ax.add_patch(self.current_patch)
            elif len(self.current_roi_points) == 2:
                # Show ellipse preview with current angle
                (cx, cy), (x2, y2) = self.current_roi_points
                rx = abs(x2 - cx)
                ry = abs(y2 - cy)
                self.current_patch = Ellipse(
                    (cx, cy), width=2*rx, height=2*ry, angle=self.current_angle,
                    fill=False, edgecolor='red', linewidth=2, linestyle='--'
                )
                self.ax.add_patch(self.current_patch)

        self.fig.canvas.draw_idle()

    def _finish_rectangle(self):
        """Finish drawing a rectangle ROI."""
        if len(self.current_roi_points) != 2:
            return

        (x1, y1), (x2, y2) = self.current_roi_points
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        roi = {
            "type": "rectangle",
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

        if self.ndim == 3:
            roi["z_min"] = 0
            roi["z_max"] = self.shape[0]
            roi["slice"] = self.current_slice

        self.rois.append(roi)
        self._reset_drawing()
        self._update_display()
        print(f"Rectangle ROI added. Total ROIs: {len(self.rois)}")

    def _finish_ellipse(self):
        """Finish drawing an ellipse ROI."""
        if len(self.current_roi_points) != 2:
            return

        (cx, cy), (x2, y2) = self.current_roi_points

        # Calculate radii from center to second point
        rx = abs(x2 - cx)
        ry = abs(y2 - cy)

        roi = {
            "type": "ellipse",
            "center_x": cx,
            "center_y": cy,
            "radius_x": rx,
            "radius_y": ry,
            "angle": self.current_angle  # Use the rotation angle
        }

        if self.ndim == 3:
            roi["slice"] = self.current_slice

        self.rois.append(roi)
        self._reset_drawing()
        self._update_display()
        print(f"Ellipse ROI added (rx={rx:.1f}, ry={ry:.1f}, angle={self.current_angle:.1f}°). Total ROIs: {len(self.rois)}")

    def _finish_polygon(self):
        """Finish drawing a polygon ROI."""
        if len(self.current_roi_points) < 3:
            print("Polygon needs at least 3 points")
            return

        roi = {
            "type": "polygon",
            "points": self.current_roi_points.copy(),
        }

        if self.ndim == 3:
            roi["slice"] = self.current_slice

        self.rois.append(roi)
        self._reset_drawing()
        self._update_display()
        print(f"Polygon ROI added. Total ROIs: {len(self.rois)}")

    def _cancel_drawing(self):
        """Cancel the current drawing operation."""
        self._reset_drawing()
        self._update_display()
        print("Drawing cancelled")

    def _reset_drawing(self):
        """Reset drawing state."""
        self.drawing_mode = None
        self.current_roi_points = []
        self.current_angle = 0.0
        if self.current_patch:
            self.current_patch.remove()
            self.current_patch = None

    def _clear_rois(self, event):
        """Clear all ROIs."""
        self.rois = []
        self._update_display()
        print("All ROIs cleared")

    def _undo_last_roi(self, event):
        """Undo the last ROI."""
        if self.rois:
            self.rois.pop()
            self._update_display()
            print(f"Last ROI removed. Total ROIs: {len(self.rois)}")

    def _show_stats(self, event):
        """Display statistics for current ROIs."""
        if not self.rois:
            print("No ROIs defined. Create an ROI first.")
            return

        from .stats import compute_basic_stats

        flat_mask = self.get_flat_mask()
        values = self.data.ravel().tolist()
        stats = compute_basic_stats(values, flat_mask)

        print("\n" + "="*50)
        print(f"Statistics for {len(self.rois)} ROI(s):")
        print("="*50)
        print(f"  Count:  {int(stats['count'])} voxels")
        print(f"  Mean:   {stats['mean']:.6f}")
        print(f"  Std:    {stats['std']:.6f}")
        print(f"  Min:    {stats['min']:.6f}")
        print(f"  Max:    {stats['max']:.6f}")
        print(f"  CoV:    {stats['coefficient_of_variation']:.6f}")
        print(f"  L2:     {stats['l2_norm']:.6f}")
        print("="*50 + "\n")

    def _redraw_rois(self):
        """Redraw all ROIs on the current slice."""
        # Remove old patches
        for patch in self.ax.patches:
            patch.remove()

        # Draw current ROIs
        for roi in self.rois:
            if self.ndim == 3:
                roi_slice = roi.get("slice", -1)
                if roi_slice != self.current_slice:
                    continue

            if roi["type"] == "rectangle":
                x_min, y_min = roi["x_min"], roi["y_min"]
                x_max, y_max = roi["x_max"], roi["y_max"]
                width, height = x_max - x_min, y_max - y_min
                rect = Rectangle(
                    (x_min, y_min), width, height,
                    fill=False, edgecolor='lime', linewidth=2
                )
                self.ax.add_patch(rect)

            elif roi["type"] == "polygon":
                poly = Polygon(
                    roi["points"], fill=False,
                    edgecolor='lime', linewidth=2
                )
                self.ax.add_patch(poly)

            elif roi["type"] == "ellipse":
                cx, cy = roi["center_x"], roi["center_y"]
                rx, ry = roi["radius_x"], roi["radius_y"]
                angle = roi.get("angle", 0)
                ellipse = Ellipse(
                    (cx, cy), width=2*rx, height=2*ry, angle=angle,
                    fill=False, edgecolor='lime', linewidth=2
                )
                self.ax.add_patch(ellipse)

    def get_spatial_mask(self, name: str = "mask") -> SpatialMask:
        """
        Get the current ROIs as a SpatialMask object.

        Args:
            name: Name for the mask

        Returns:
            SpatialMask object containing all defined ROIs
        """
        return SpatialMask(
            shape=self.shape,
            rois=self.rois.copy(),
            name=name
        )

    def get_flat_mask(self) -> List[bool]:
        """
        Get the current ROIs as a flat boolean mask.

        Returns:
            Flat boolean mask matching the image's flattened voxel buffer
        """
        spatial_mask = self.get_spatial_mask()
        return spatial_mask.to_flat_mask(self.data.size)

    def get_mask_image(self):
        """
        Get the current ROIs as a SIRF ImageData mask (1s and 0s).

        Returns:
            SIRF ImageData with 1s inside ROIs, 0s outside
        """
        spatial_mask = self.get_spatial_mask()
        return spatial_mask.to_image_mask(self.image)

    def get_masked_image(self):
        """
        Apply the current mask to the SIRF ImageData.

        Returns:
            SIRF ImageData with mask applied (values outside ROI set to 0)
        """
        _check_dependencies()

        # Get mask as SIRF ImageData
        mask_image = self.get_mask_image()

        # Multiply original image by mask
        masked_image = self.image * mask_image

        return masked_image

    def get_stats(self) -> Dict[str, float]:
        """
        Compute statistics within the current ROIs.

        Returns:
            Dictionary with statistics (mean, std, cov, etc.)
        """
        from .stats import compute_basic_stats

        flat_mask = self.get_flat_mask()
        values = self.data.ravel().tolist()
        return compute_basic_stats(values, flat_mask)


def view_image(image, **kwargs) -> ImageViewer:
    """
    Display an image interactively in a Jupyter notebook.

    Args:
        image: SIRF.STIR ImageData object or path to image file
        **kwargs: Additional arguments passed to ImageViewer

    Returns:
        ImageViewer instance
    """
    viewer = ImageViewer(image, **kwargs)
    return viewer.show()


def apply_mask_batch(
    image_paths: List[Union[str, Path]],
    mask: SpatialMask,
    callback: Optional[Callable[[Any, List[bool]], Any]] = None
) -> List[Any]:
    """
    Apply a spatial mask to multiple images and optionally process them.

    Args:
        image_paths: List of paths to image files
        mask: SpatialMask to apply
        callback: Optional function to process each (image, flat_mask) pair

    Returns:
        List of callback results, or list of (image, mask) tuples if no callback
    """
    results = []

    for path in image_paths:
        image = load_image(Path(path))
        flat_mask = mask.to_flat_mask(get_image_array(image).size)

        if callback:
            result = callback(image, flat_mask)
        else:
            result = (image, flat_mask)

        results.append(result)

    return results


__all__ = [
    "SpatialMask",
    "ImageViewer",
    "view_image",
    "apply_mask_batch",
]
