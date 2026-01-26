#!/usr/bin/env python3
"""
Lesion detection and mask generation script.

This script detects lesions on a reference image and saves the masks to disk.
The masks can then be loaded by the analysis script for reconstruction evaluation.

Usage:
    python grow_lesions.py --patient sirt3 --method local_background
    python grow_lesions.py --patient sirt3 --method intensity --intensity-tolerance 0.30
    python grow_lesions.py --patient sirt3 --method local_background --k-sigma 1.2 --output-dir /custom/path
"""

import argparse
import logging
import pathlib
import sys
import glob
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sirf.STIR import ImageData

# Add project paths (ensure local code takes precedence over any installed copy)
sys.path.insert(0, "/home/storage")
from cluster_analysis.lesion_detection import (
    compute_lesion_statistics,
    extract_lesion_metadata,
    find_hottest_lesions,
    grow_lesion_mask,
)


def setup_logging(output_dir: pathlib.Path) -> logging.Logger:
    """Set up logging to both console and file."""
    logger = logging.getLogger('grow_lesions')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file = output_dir / 'grow_lesions_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class LesionGrower:
    """Detect and save lesion masks for a patient."""

    def __init__(
        self,
        patient: str,
        method: str,
        output_base_dir: pathlib.Path,
        num_lesions: int = 5,
        min_size: Optional[int] = 10,
        chosen_alpha: float = 0.01,
        chosen_beta: float = 0.01,
        suffix: str = "",
        source_image: str = "vendor",
        # Intensity method parameters
        intensity_tolerance: float = 0.20,
        max_size: Optional[int] = None,
        # Radial gradient method parameters (Mikell et al. 2018)
        gradient_sigma: float = 0.1,
        max_ray_length: float = 25.0,
        min_fraction_of_seed: float = 0.42,
        force: bool = False,
    ):
        """
        Initialize the lesion grower.

        Args:
            patient: Patient ID (e.g., 'sirt3')
            method: Lesion growing method ('intensity' or 'radial_gradient')
            output_base_dir: Base directory for output (e.g., /home/storage/cluster/patient_sweeps/lesion_masks)
            num_lesions: Number of lesions to detect
            min_size: Optional minimum voxel count per lesion (filters small noisy masks)
            chosen_alpha: Alpha value for chosen reconstruction (used for zooming template)
            chosen_beta: Beta value for chosen reconstruction (used for zooming template)
            source_image: Source image for lesion detection ('vendor' or 'combined_recon')
            intensity_tolerance: For 'intensity' method - fractional tolerance (e.g., 0.20 = ±20%)
            max_size: Optional maximum number of voxels per lesion
            gradient_sigma: For 'radial_gradient' method - Gaussian smoothing sigma (default 0.1, minimal since Q.Clear is already smooth)
            max_ray_length: For 'radial_gradient' method - Maximum ray search distance (mm, safety limit)
            min_fraction_of_seed: For 'radial_gradient' method - Stop when intensity drops below this fraction (default 0.42 = 42%)
            force: Overwrite existing masks if True
        """
        self.patient = patient
        self.method = method
        self.source_image = source_image
        self.num_lesions = num_lesions
        self.min_size = min_size
        self.chosen_alpha = chosen_alpha
        self.chosen_beta = chosen_beta
        self.chosen_key = (chosen_alpha, chosen_beta)

        # Set up paths matching the original script
        self.results_root = pathlib.Path(
            f"/home/sam/mnt/comic-sporter/synergistic_Y90/SETR/sweeps/output/2bpos_alpha_beta_{patient}{f'_{suffix}' if suffix else ''}"
        )

        # Method-specific parameters
        self.intensity_tolerance = intensity_tolerance
        self.max_size = max_size
        self.gradient_sigma = gradient_sigma
        self.max_ray_length = max_ray_length
        self.min_fraction_of_seed = min_fraction_of_seed
        self.force = force

        # Set up paths
        self.data_dir = pathlib.Path("/home/storage/prepared_data/oxford_patient_data/")
        self.patient_dir = self.data_dir / patient

        # Output directory: base_dir / patient / method_source
        method_source = f"{method}_{source_image}" if source_image != "vendor" else method
        self.output_dir = output_base_dir / patient / method_source
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = setup_logging(self.output_dir)
        self.logger.info(f"Initialized LesionGrower for patient: {patient}")
        self.logger.info(f"Method: {method}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Discover alpha/beta reconstructions
        self.discover_reconstructions()

    def discover_reconstructions(self):
        """Discover all alpha/beta reconstruction combinations (matches original script)."""
        self.logger.info("Discovering alpha/beta reconstruction combinations...")

        pattern = re.compile(r'alpha_([\d.]+)_beta_([\d.]+)')
        entries = []

        # Use more specific glob pattern for better performance
        for p in self.results_root.glob("**/final_image_*.hv"):
            if p.name in ("final_image_0.hv", "final_image_1.hv"):
                if m := pattern.search(str(p.parent)):
                    a, b = map(float, m.groups())
                    entries.append((a, b, p.name, str(p)))

        # Group by (alpha, beta)
        grouped = defaultdict(dict)
        for a, b, name, path in entries:
            grouped[(a, b)][name] = path

        self.alpha_beta_combinations = sorted(grouped.keys())
        self.alpha_beta_paths = grouped

        self.logger.info(f"Found {len(self.alpha_beta_combinations)} parameter combinations")

        # Verify chosen combination exists
        if self.chosen_key not in self.alpha_beta_combinations:
            self.logger.warning(f"Chosen combination α={self.chosen_alpha}, β={self.chosen_beta} not found in results!")
            self.logger.warning(f"Available combinations: {self.alpha_beta_combinations[:10]}")

    def load_vendor_image(self) -> ImageData:
        """Load and prepare the source reconstruction image (vendor or combined_recon)."""

        if self.source_image == 'combined_recon':
            # Load 5mm smoothed OSEM (combined_recon)
            self.logger.info("Loading combined_recon (5mm smoothed OSEM)...")
            combined_path = self.patient_dir / "PET" / "non_tof" / "combined_recon.hv"

            if not combined_path.exists():
                raise FileNotFoundError(f"Combined recon not found: {combined_path}")

            source_image = ImageData(str(combined_path))
            self.logger.info(f"Loaded combined_recon from {combined_path}")

            # Apply 10mm FWHM Gaussian smoothing for OSEM images
            from sirf.STIR import SeparableGaussianImageFilter
            gaussian_filter = SeparableGaussianImageFilter()
            gaussian_filter.set_fwhms((10, 10, 10))
            source_image = gaussian_filter.process(source_image)
            self.logger.info("Applied 10mm FWHM Gaussian smoothing")

        else:  # vendor
            # Find vendor DICOM file
            self.logger.info("Loading vendor reconstruction...")
            # Patient to ANON ID mapping (from translate_names.md)
            PATIENT_TO_ANON = {
                'sirt1': 'ANON2138',   # SIRT_01
                'sirt2': 'ANON14731',  # SIRT_02
                'sirt3': 'ANON1900',   # SIRT_03
                'sirt4': 'ANON14493',  # SIRT_04
                'sirt5': 'ANON14397',  # SIRT_05
                'sirt6': 'ANON14308',  # SIRT_06
                'sirt7': 'ANON14062',  # SIRT_07
                'sirt8': 'ANON13620',  # SIRT_08
                'sirt9': 'ANON13458',  # SIRT_09
                'sirt10': 'ANON13245', # SIRT_10
            }
            anon = PATIENT_TO_ANON.get(self.patient.lower())
            if anon is None:
                raise ValueError(f"Unknown patient ID: {self.patient}. Valid IDs: {list(PATIENT_TO_ANON.keys())}")
            pattern = f"/home/storage/oxford_patient_data/SIRT_vendor_recons/{anon}/QCFX-B4000-UPDATE/*.dcm"
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No vendor files match: {pattern}")

            vendor_path = max(matches, key=os.path.getmtime)
            source_image = ImageData(vendor_path)
            self.logger.info(f"Loaded vendor from {vendor_path}")

        # Align source image to patient reference space (common for both vendor and combined_recon)
        template_path = self.patient_dir / "PET" / "non_tof" / "combined_recon.hv"
        if not template_path.exists():
            # Fallback to chosen reconstruction if combined_recon isn't available
            chosen_pet_path = None
            if self.chosen_key in self.alpha_beta_paths:
                chosen_pet_path = self.alpha_beta_paths[self.chosen_key].get("final_image_0.hv")
            if chosen_pet_path:
                template_path = pathlib.Path(chosen_pet_path)
            else:
                raise FileNotFoundError(
                    f"Reference template not found for {self.patient}. "
                    f"Expected combined_recon.hv at {self.patient_dir / 'PET' / 'non_tof' / 'combined_recon.hv'} "
                    f"or chosen reconstruction (alpha={self.chosen_alpha}, beta={self.chosen_beta})."
                )

        template_image = ImageData(str(template_path))

        # Step 1: match voxel sizes
        zooms = tuple(v/h for v, h in zip(source_image.voxel_sizes(), template_image.voxel_sizes()))
        source_zoom = source_image.zoom_image(zooms, scaling='preserve_projections')

        # Step 2: match dimensions/origin to HKEM
        source_aligned = source_zoom.zoom_image_as_template(template_image, scaling='preserve_projections')

        # Persist aligned image for downstream tools (e.g., background mask default template)
        # Note: For combined_recon, we still write to vendor_zoomed.hv for compatibility
        vendor_zoomed_path = self.patient_dir / "PET" / "non_tof" / "vendor_zoomed.hv"
        vendor_zoomed_path.parent.mkdir(parents=True, exist_ok=True)
        source_aligned.write(str(vendor_zoomed_path))
        self.logger.info(f"Wrote aligned {self.source_image} to {vendor_zoomed_path}")

        self.logger.info(f"Aligned {self.source_image} to HKEM space")
        self.logger.info(f"  Source aligned shape: {source_aligned.as_array().shape}")
        self.logger.info(f"  Voxel sizes (aligned): {source_aligned.voxel_sizes()}")

        return source_aligned

    def save_hotspot_locations(self, vendor_image: ImageData, hotspot_coords: list):
        """Save hotspot seed locations to CSV."""
        self.logger.info("Saving hotspot locations...")

        hotspot_data = []
        for i, (z, y, x) in enumerate(hotspot_coords, start=1):
            hotspot_data.append({
                'lesion': f'lesion_{i}',
                'z': z,
                'y': y,
                'x': x,
                'intensity': float(vendor_image.as_array()[z, y, x])
            })

        df = pd.DataFrame(hotspot_data)
        hotspot_csv = self.output_dir / "hotspot_locations.csv"
        df.to_csv(hotspot_csv, index=False)
        self.logger.info(f"Saved hotspot locations to {hotspot_csv}")

    def create_lesion_plots(self, vendor_image: ImageData, lesion_masks: dict, hotspot_coords: list):
        """Create coronal slice plots for each lesion centered on seed voxel."""
        self.logger.info("Creating lesion plots...")

        vendor_arr = vendor_image.as_array()
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for i, (lesion_name, mask_obj) in enumerate(lesion_masks.items(), start=1):
            z_seed, y_seed, x_seed = hotspot_coords[i-1]

            # Get coronal slice at y_seed
            # Robustly fetch mask data regardless of ImageData / ndarray interface
            mask_arr = mask_obj.as_array().astype(bool) if hasattr(mask_obj, "as_array") else np.asarray(mask_obj).astype(bool)
            voxel_counts_along_y = mask_arr.sum(axis=(0, 2))
            y_best = int(np.argmax(voxel_counts_along_y)) if voxel_counts_along_y.any() else y_seed

            # Prefer the slice with the most mask voxels to make contours visible
            seed_voxels = int(voxel_counts_along_y[y_seed]) if voxel_counts_along_y.size > y_seed else 0
            best_voxels = int(voxel_counts_along_y[y_best]) if voxel_counts_along_y.size > y_best else 0
            y_plot = y_best if best_voxels >= seed_voxels else y_seed
            if y_plot != y_seed:
                self.logger.warning(
                    f"{lesion_name}: using y={y_plot} (max-mask slice, {best_voxels} voxels) "
                    f"instead of seed slice y={y_seed} ({seed_voxels} voxels) for plotting"
                )

            coronal_slice = vendor_arr[:, y_plot, :]  # (z, x)
            mask_slice = mask_arr[:, y_plot, :]  # (z, x)

            # If the chosen y-slice has no mask voxels (e.g., tolerance too tight or seed off-slice),
            # move to the y-slice with the most voxels so the overlay is always visible in QA plots.
            if not mask_slice.any() and voxel_counts_along_y.any():
                y_fallback = int(np.argmax(voxel_counts_along_y))
                if y_fallback != y_plot:
                    self.logger.warning(
                        f"{lesion_name}: no mask voxels on selected slice y={y_plot}; "
                        f"falling back to y={y_fallback}"
                    )
                y_plot = y_fallback
                coronal_slice = vendor_arr[:, y_plot, :]
                mask_slice = mask_arr[:, y_plot, :]

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))

            # Display vendor image
            vmax = np.percentile(vendor_arr[vendor_arr > 0], 99)
            # Use origin='upper' so the plotted Z axis matches the natural array order (avoids upside-down plots)
            im = ax.imshow(coronal_slice, cmap='gray', vmin=0, vmax=vmax, aspect='auto', origin='upper')

            # Overlay lesion mask
            mask_overlay = np.ma.masked_where(~mask_slice, mask_slice)
            ax.imshow(mask_overlay, cmap='Reds', alpha=0.6, aspect='auto', origin='upper')
            # Add a contour so very small masks remain visible even on bright backgrounds
            ax.contour(mask_slice, levels=[0.5], colors='red', linewidths=1.5, origin='upper')

            # Mark seed voxel with crosshair
            ax.plot(x_seed, z_seed, 'r+', markersize=15, markeredgewidth=2, label='Seed voxel')
            ax.axhline(z_seed, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.axvline(x_seed, color='red', linestyle='--', alpha=0.3, linewidth=1)

            # Labels and title
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Z', fontsize=12)
            ax.set_title(f'{lesion_name} - Coronal slice at Y={y_plot}\nSeed: (z={z_seed}, y={y_seed}, x={x_seed})',
                        fontsize=14)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', fontsize=10)

            # Add legend
            red_patch = mpatches.Patch(color='red', alpha=0.5, label='Lesion mask')
            ax.legend(handles=[red_patch], loc='upper right')

            # Save
            plot_path = plots_dir / f"{lesion_name}_coronal.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"  Saved plot: {plot_path}")

        self.logger.info(f"All plots saved to {plots_dir}")

    def detect_and_save_lesions(self):
        """Detect lesions and save masks to disk."""
        self.logger.info("="*80)
        self.logger.info("Starting lesion detection")
        self.logger.info("="*80)

        existing_masks = sorted(self.output_dir.glob("lesion_*.hv"))
        if existing_masks and not self.force:
            self.logger.info(f"Found existing lesion masks in {self.output_dir}; skipping generation (use --force to regrow)")
            return

        # Load vendor image
        vendor_image = self.load_vendor_image()

        # Load background mask and compute its mean to guide hotspot thresholds
        background_mean = None
        bg_mask_path = pathlib.Path(f"/home/storage/cluster/patient_sweeps/lesion_masks/{self.patient}/background_mask.hv")
        if bg_mask_path.exists():
            try:
                bg_mask_img = ImageData(str(bg_mask_path))
                bg_mask_arr = bg_mask_img.as_array() > 0
                vendor_arr = vendor_image.as_array()
                if np.any(bg_mask_arr):
                    background_mean = float(vendor_arr[bg_mask_arr].mean())
                    self.logger.info(f"Using background mean {background_mean:.2e} from {bg_mask_path}")
                else:
                    self.logger.warning(f"Background mask {bg_mask_path} has no voxels > 0; falling back to percentile threshold")
            except Exception as e:
                self.logger.warning(f"Failed to load background mask {bg_mask_path}: {e}")

        # Find hotspot locations first
        requested_seed_count = self.num_lesions
        if self.min_size is not None:
            requested_seed_count = self.num_lesions + 10
        self.logger.info(f"Finding {requested_seed_count} hotspot seed locations...")
        hotspot_coords = find_hottest_lesions(
            vendor_image,
            num_lesions=requested_seed_count,
            min_separation=30.0,  # enforce 30 mm between hotspot seeds (matches mask-growing)
            background_percentile=50.0,
            background_value=background_mean,
        )
        self.logger.info(f"Found {len(hotspot_coords)} hotspots")
        for i, (z, y, x) in enumerate(hotspot_coords, start=1):
            intensity = vendor_image.as_array()[z, y, x]
            self.logger.info(f"  Hotspot {i}: (z={z}, y={y}, x={x}), intensity={intensity:.2e}")

        # Detect lesions
        self.logger.info(f"Detecting {self.num_lesions} lesions using method='{self.method}'...")
        self.logger.info("Parameters:")
        if self.method == 'intensity':
            self.logger.info(f"  intensity_tolerance: {self.intensity_tolerance}")
            if self.max_size:
                self.logger.info(f"  max_size: {self.max_size}")
            if self.min_size:
                self.logger.info(f"  min_size: {self.min_size}")
        elif self.method == 'radial_gradient':
            self.logger.info(f"  gradient_sigma: {self.gradient_sigma}")
            self.logger.info(f"  max_ray_length: {self.max_ray_length} mm")
            self.logger.info(f"  min_fraction_of_seed: {self.min_fraction_of_seed}")
            if self.max_size:
                self.logger.info(f"  max_size: {self.max_size}")
            if self.min_size:
                self.logger.info(f"  min_size: {self.min_size}")

        # Grow lesions from the preselected hotspot seeds
        lesion_masks = {}
        accepted_hotspots = []
        for seed in hotspot_coords:
            mask_arr = grow_lesion_mask(
                image=vendor_image,
                seed_point=seed,
                method=self.method,
                intensity_tolerance=self.intensity_tolerance,
                background_mean=background_mean,
                max_size=self.max_size,
                connectivity=1,  # face-adjacent only (6-neighbour)
                gradient_sigma=self.gradient_sigma,
                max_ray_length=self.max_ray_length,
                min_fraction_of_seed=self.min_fraction_of_seed,
            )

            mask_voxels = int(mask_arr.sum())
            if self.min_size is not None and mask_voxels < self.min_size:
                self.logger.warning(
                    f"Skipping seed {seed}: lesion size {mask_voxels} voxels < min_size {self.min_size}"
                )
                continue

            mask_img = vendor_image.clone()
            mask_img.fill(mask_arr.astype(np.float32))
            accepted_hotspots.append(seed)
            lesion_idx = len(accepted_hotspots)
            lesion_masks[f"lesion_{lesion_idx}"] = mask_img
            if lesion_idx >= self.num_lesions:
                break

        if len(lesion_masks) < self.num_lesions:
            self.logger.warning(
                f"Detected {len(lesion_masks)} lesions (requested {self.num_lesions}); "
                f"consider adjusting min_size or hotspot settings"
            )
        else:
            self.logger.info(f"Detected {len(lesion_masks)} lesions")

        # Compute and log statistics
        stats = compute_lesion_statistics(vendor_image, lesion_masks)
        for lesion_name, lesion_stats in stats.items():
            self.logger.info(f"  {lesion_name}:")
            self.logger.info(f"    max: {lesion_stats['max']:.2e}")
            self.logger.info(f"    mean: {lesion_stats['mean']:.2e}")
            self.logger.info(f"    volume: {lesion_stats['volume']} voxels")
            self.logger.info(f"    center: {lesion_stats['center']}")

        # Save lesion masks
        self.logger.info(f"Saving lesion masks to {self.output_dir}...")
        for lesion_name, mask_obj in lesion_masks.items():
            mask_path = self.output_dir / f"{lesion_name}.hv"
            mask_obj.write(str(mask_path))
            self.logger.info(f"  Saved {mask_path}")

        # Extract and save metadata
        metadata = extract_lesion_metadata(vendor_image, lesion_masks)
        metadata_path = self.output_dir / "lesion_metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Patient: {self.patient}\n")
            f.write(f"Method: {self.method}\n")
            f.write(f"Number of lesions: {len(lesion_masks)}\n")
            f.write(f"Detection date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            for lesion_name, info in metadata.items():
                f.write(f"{lesion_name}:\n")
                f.write(f"  seed_point: {info['seed_point']}\n")
                f.write(f"  voxel_count: {info['voxel_count']}\n")
                f.write(f"  peak_value: {info['peak_value']:.2e}\n")
                f.write(f"  center_of_mass: {info['center_of_mass']}\n")
                f.write("\n")
        self.logger.info(f"Saved metadata to {metadata_path}")

        # Save vendor image for reference
        vendor_path = self.output_dir / "vendor_reference.hv"
        vendor_image.write(str(vendor_path))
        self.logger.info(f"Saved vendor reference image to {vendor_path}")

        # Save hotspot locations (only accepted lesions)
        self.save_hotspot_locations(vendor_image, accepted_hotspots)

        # Create lesion plots
        self.create_lesion_plots(vendor_image, lesion_masks, accepted_hotspots)

        self.logger.info("="*80)
        self.logger.info("Lesion detection complete!")
        self.logger.info(f"Output saved to: {self.output_dir}")
        self.logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect and save lesion masks for a patient",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--patient', type=str, required=True,
                       help='Patient ID (e.g., sirt3)')
    parser.add_argument('--suffix', type=str, default='',
                       help='Optional suffix for results directory (e.g., "nodtv")')
    parser.add_argument('--method', type=str, default='intensity',
                       choices=['intensity', 'radial_gradient'],
                       help='Lesion growing method: intensity (fixed threshold) or radial_gradient (ray-casting)')
    parser.add_argument('--source-image', type=str, default='vendor',
                       choices=['vendor', 'combined_recon'],
                       help='Source image for lesion detection: vendor (original) or combined_recon (5mm smoothed OSEM)')
    parser.add_argument('--output-dir', type=pathlib.Path,
                       default=pathlib.Path("/home/storage/cluster/patient_sweeps/lesion_masks"),
                       help='Base output directory (masks will be saved to output-dir/patient/method_source/)')
    parser.add_argument('--num-lesions', type=int, default=20,
                       help='Number of lesions to detect')
    parser.add_argument('--min-size', type=int, default=10,
                       help='Minimum number of voxels per lesion (filters small noisy masks)')

    # Chosen reconstruction parameters (for zooming template)
    parser.add_argument('--chosen-alpha', type=float, default=0.1,
                       help='Alpha value for chosen reconstruction (used as zoom template)')
    parser.add_argument('--chosen-beta', type=float, default=0.1,
                       help='Beta value for chosen reconstruction (used as zoom template)')

    # Intensity method parameters
    parser.add_argument('--intensity-tolerance', type=float, default=0.40,
                       help='For intensity method: fractional tolerance (e.g., 0.30 = ±40%%, 60%% threshold)')
    parser.add_argument('--max-size', type=int, default=None,
                       help='Optional maximum number of voxels per lesion (applies to both methods)')

    # Radial gradient method parameters (Mikell et al. 2018)
    parser.add_argument('--gradient-sigma', type=float, default=0.1,
                       help='For radial_gradient: Gaussian smoothing sigma (default 0.1, minimal smoothing since Q.Clear is already smooth)')
    parser.add_argument('--max-ray-length', type=float, default=25.0,
                       help='For radial_gradient: maximum ray search distance in mm (safety limit, default 25.0)')
    parser.add_argument('--min-fraction-of-seed', type=float, default=0.42,
                       help='For radial_gradient: minimum intensity fraction - stop when < 42%% of seed value (PET literature standard)')

    parser.add_argument('--force', action='store_true',
                       help='Overwrite/regrow lesions even if masks already exist')

    args = parser.parse_args()

    # Create grower
    grower = LesionGrower(
        patient=args.patient,
        method=args.method,
        output_base_dir=args.output_dir,
        num_lesions=args.num_lesions,
        min_size=args.min_size,
        chosen_alpha=args.chosen_alpha,
        chosen_beta=args.chosen_beta,
        suffix=args.suffix,
        source_image=args.source_image,
        intensity_tolerance=args.intensity_tolerance,
        max_size=args.max_size,
        gradient_sigma=args.gradient_sigma,
        max_ray_length=args.max_ray_length,
        min_fraction_of_seed=args.min_fraction_of_seed,
        force=args.force,
    )

    # Run lesion detection
    try:
        grower.detect_and_save_lesions()
    except Exception as e:
        grower.logger.error(f"Lesion detection failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
