#!/usr/bin/env python3
"""
Automated reconstruction analysis script.

This script automates the analysis workflow from view_results_streamlined.ipynb
for all alpha/beta reconstruction combinations and HKEM iterations from 9-90.

Usage:
    python analyse_reconstruction.py --patient sirt3
    python analyse_reconstruction.py --patient sirt3 --chosen-alpha 1.55 --chosen-beta 0.2
    python analyse_reconstruction.py --patient sirt3 --hkem-start 9 --hkem-end 90 --hkem-step 9
"""

import argparse
import logging
import pathlib
import re
import sys
import glob
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import cmasher

from sirf.STIR import ImageData

# Add project paths
sys.path.append("/home/storage")
from cluster_analysis.lesion_detection import (
    compute_lesion_statistics,
    extract_lesion_metadata,
)
from cluster_analysis.jupyter_utils import SpatialMask

from recon_experiments.runners.common import get_resampling_operators
from recon_core.utils.sirf import get_pet_data_multiple_bed_pos, get_spect_data


# Configure logging
def setup_logging(output_dir: pathlib.Path) -> logging.Logger:
    """Set up logging to both console and file."""
    logger = logging.getLogger('analyse_reconstruction')
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file = output_dir / 'analysis_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class ReconstructionAnalyzer:
    """Main class for automated reconstruction analysis."""

    def __init__(self, patient: str, suffix: str,
                 chosen_alpha: float, chosen_beta: float,
                 hkem_start: int, hkem_end: int, hkem_step: int,
                 iteration: Optional[int] = None,
                 output_dir: Optional[pathlib.Path] = None,
                 save_lesions: bool = False,
                 lesion_method: str = 'local_background',
                 lesion_masks_dir: Optional[pathlib.Path] = None):
        """
        Initialize the analyzer.

        Args:
            patient: Patient ID (e.g., 'sirt3')
            chosen_alpha: Alpha value for chosen reconstruction
            chosen_beta: Beta value for chosen reconstruction
            hkem_start: Starting HKEM iteration
            hkem_end: Ending HKEM iteration
            hkem_step: Step size for HKEM iterations
            iteration: Iteration number for image files (default: None uses final_image_*.hv)
            output_dir: Output directory (default: auto-generated)
            save_lesions: If True, save lesion masks to disk
            lesion_method: Method used for lesion growing (e.g., 'local_background', 'intensity')
            lesion_masks_dir: Directory containing pre-computed lesion masks (default: /home/storage/cluster/patient_sweeps/lesion_masks/patient/method/)
        """
        self.patient = patient
        self.chosen_key = (chosen_alpha, chosen_beta)
        self.hkem_iterations = list(range(hkem_start, hkem_end + 1, hkem_step))
        self.iteration = iteration
        self.save_lesions = save_lesions
        self.lesion_method = lesion_method

        # Set image filenames based on iteration parameter
        if self.iteration is None:
            self.image_0_name = "final_image_0.hv"
            self.image_1_name = "final_image_1.hv"
        else:
            self.image_0_name = f"image_0_{self.iteration}.hv"
            self.image_1_name = f"image_1_{self.iteration}.hv"

        # Set lesion masks directory
        if lesion_masks_dir is None:
            self.lesion_masks_dir = pathlib.Path("/home/storage/cluster/patient_sweeps/lesion_masks") / patient / lesion_method
        else:
            self.lesion_masks_dir = lesion_masks_dir

        # Set up paths
        self.data_dir = pathlib.Path("/home/storage/prepared_data/oxford_patient_data/")
        self.patient_dir = self.data_dir / patient
        self.results_root = pathlib.Path(
            f"/home/sam/mnt/comic-sporter/synergistic_Y90/SETR/sweeps/output/2bpos_alpha_beta_{patient}{f'_{suffix}' if suffix else ''}"
        )
        self.hkem_dir = pathlib.Path(f"/home/sam/working/synergistic_recon/results/{patient}/non_tof/hkem_pet_2bpos")

        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.results_root / f"automated_analysis_{timestamp}"
        else:
            self.output_dir = output_dir

        self.csv_dir = self.output_dir / "csv_results"
        self.plots_dir = self.output_dir / "plots"
        self.lesions_dir = self.output_dir / "lesion_masks" if save_lesions else None

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        if self.lesions_dir is not None:
            self.lesions_dir.mkdir(exist_ok=True)

        # Set up logging
        self.logger = setup_logging(self.output_dir)
        self.logger.info(f"Initialized ReconstructionAnalyzer for patient: {patient}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Chosen reconstruction: α={chosen_alpha}, β={chosen_beta}")
        self.logger.info(f"HKEM iterations: {self.hkem_iterations}")
        self.logger.info(f"Lesion method: {lesion_method}")
        self.logger.info(f"Lesion masks directory: {self.lesion_masks_dir}")

        # Data containers
        self.pet_data = None
        self.spect_data = None
        self.spect2pet = None
        self.alpha_beta_combinations = []
        self.alpha_beta_paths = {}

        # Reference images for lesion detection
        self.reference_images = {}
        self.lesion_metadata_by_reference = {}
        self.auto_lesions_by_reference = {}

        # Cached arrays for performance
        self.cached_arrays = {}
        self.cached_mask_arrays = {}

    def load_patient_data(self):
        """Load patient PET and SPECT data."""
        self.logger.info("Loading patient PET and SPECT data...")

        try:
            self.pet_data = get_pet_data_multiple_bed_pos(
                f"{self.patient_dir}/PET",
                ["_f1b1", "_f2b1"],
                load_sinos=True
            )
            self.spect_data = get_spect_data(
                f"{self.patient_dir}/SPECT",
                load_sinos=True
            )
            self.spect2pet = get_resampling_operators(self.pet_data, self.spect_data)

            self.logger.info("Patient data loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load patient data: {e}")
            raise

    def discover_reconstructions(self):
        """Discover all alpha/beta reconstruction combinations."""
        self.logger.info("Discovering alpha/beta reconstruction combinations...")

        # Determine glob pattern based on iteration parameter
        if self.iteration is None:
            glob_pattern = "**/final_image_*.hv"
            self.logger.info("Using final images (no iteration specified)")
        else:
            glob_pattern = f"**/image_*_{self.iteration}.hv"
            self.logger.info(f"Using iteration {self.iteration} images")

        pattern = re.compile(r'alpha_([\d.]+)_beta_([\d.]+)')
        entries = []

        # Use more specific glob pattern for better performance
        for p in self.results_root.glob(glob_pattern):
            if p.name in (self.image_0_name, self.image_1_name):
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
            self.logger.error(f"Chosen combination α={self.chosen_key[0]}, β={self.chosen_key[1]} not found in results!")
            self.logger.error(f"Available alpha values: {sorted(set(a for a, b in self.alpha_beta_combinations))}")
            self.logger.error(f"Available beta values: {sorted(set(b for a, b in self.alpha_beta_combinations))}")
            raise ValueError(f"Chosen combination {self.chosen_key} not found. Please use an existing combination.")

    def load_reference_images(self):
        """Load the three reference images for lesion detection."""
        self.logger.info("Loading reference images...")

        try:

            # 2. Vendor reconstruction
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
            vendor = ImageData(vendor_path)

            # Zoom vendor to match reconstruction space
            # First zoom to match voxel sizes, then use zoom_image_as_template to fix any dimension mismatches
            hkem_27_path = self.hkem_dir / "x_27.hv"
            hkem_ref = ImageData(str(hkem_27_path))
            zooms = tuple(v/h for v, h in zip(vendor.voxel_sizes(), hkem_ref.voxel_sizes()))
            vendor_zoom = vendor.zoom_image(zooms, scaling='preserve_projections')
            
            # write venodor zoome to data path
            vendor_zoom.write(f"{self.patient_dir}/PET/non_tof/vendor_zoomed.hv")

            # Use zoom_image_as_template with the chosen reconstruction to fix dimension mismatches (e.g., 256->255)
            if self.chosen_key in self.alpha_beta_paths:
                chosen_pet_path = self.alpha_beta_paths[self.chosen_key].get(self.image_0_name)
                if chosen_pet_path:
                    template_pet = ImageData(chosen_pet_path)
                    vendor_zoom = vendor_zoom.zoom_image_as_template(template_pet, scaling='preserve_projections')
                else:
                    self.logger.warning("Chosen PET reconstruction not found for vendor zoom template")
            else:
                self.logger.warning("Chosen key not in alpha_beta_paths, vendor may have dimension mismatch")

            self.reference_images['vendor'] = vendor_zoom
            self.logger.info(f"Loaded and zoomed vendor from {vendor_path}")

            # 3. Chosen reconstruction (alpha, beta)
            if self.chosen_key in self.alpha_beta_paths:
                chosen_pet_path = self.alpha_beta_paths[self.chosen_key].get(self.image_0_name)
                if chosen_pet_path:
                    #self.reference_images['chosen'] = ImageData(chosen_pet_path)
                    self.logger.info(f"Loaded chosen reconstruction (α={self.chosen_key[0]}, β={self.chosen_key[1]})")
                else:
                    self.logger.warning("Chosen reconstruction PET image not found")
            else:
                self.logger.warning("Chosen reconstruction combination not found")

        except Exception as e:
            self.logger.error(f"Failed to load reference images: {e}")
            raise

    def load_lesion_masks(self):
        """Load pre-computed lesion masks from disk."""
        self.logger.info(f"Loading pre-computed lesion masks from {self.lesion_masks_dir}...")

        if not self.lesion_masks_dir.exists():
            raise FileNotFoundError(
                f"Lesion masks directory not found: {self.lesion_masks_dir}\n"
                f"Please run grow_lesions.py first:\n"
                f"  python grow_lesions.py --patient {self.patient} --method {self.lesion_method}"
            )

        # Load vendor reference image
        vendor_ref_path = self.lesion_masks_dir / "vendor_reference.hv"
        if not vendor_ref_path.exists():
            raise FileNotFoundError(
                f"Vendor reference image not found: {vendor_ref_path}\n"
                f"Please run grow_lesions.py first to generate lesion masks."
            )

        vendor_ref = ImageData(str(vendor_ref_path))
        self.logger.info(f"Loaded vendor reference image from {vendor_ref_path}")

        # Find all lesion mask files
        lesion_mask_files = sorted(self.lesion_masks_dir.glob("lesion_*.hv"))
        if not lesion_mask_files:
            raise FileNotFoundError(
                f"No lesion mask files found in {self.lesion_masks_dir}\n"
                f"Please run grow_lesions.py first to generate lesion masks."
            )

        # Load lesion masks
        ref_name = 'vendor'
        auto_lesions = {}

        for mask_file in lesion_mask_files:
            lesion_name = mask_file.stem  # e.g., 'lesion_1'
            mask_img = ImageData(str(mask_file))
            auto_lesions[lesion_name] = mask_img
            self.logger.info(f"  Loaded {lesion_name} from {mask_file}")

        self.auto_lesions_by_reference[ref_name] = auto_lesions

        # Cache mask arrays for performance
        self.cached_mask_arrays[ref_name] = {}
        for lesion_name, mask_obj in auto_lesions.items():
            self.cached_mask_arrays[ref_name][lesion_name] = mask_obj.as_array().astype(bool)

        # Extract metadata
        lesion_metadata = extract_lesion_metadata(vendor_ref, auto_lesions)
        self.lesion_metadata_by_reference[ref_name] = lesion_metadata

        # Compute and log statistics
        stats = compute_lesion_statistics(vendor_ref, auto_lesions)
        self.logger.info(f"Loaded {len(auto_lesions)} lesions")
        for lesion_name, lesion_stats in stats.items():
            self.logger.debug(f"  {lesion_name}: max={lesion_stats['max']:.2e}, "
                            f"volume={lesion_stats['volume']} voxels")

        self.logger.info(f"Successfully loaded {len(auto_lesions)} lesion masks")

    def create_background_mask(self, template_image):
        """Create background ROI mask."""
        bg_rois = [{
            "type": "ellipsoid",
            "center_x": 85, "center_y": 125, "center_z": 40,
            "radius_x": 8, "radius_y": 8, "radius_z": 6,
            "angle_x": 0, "angle_y": 0, "angle_z": 0
        }]

        bg_mask = SpatialMask(
            shape=template_image.shape,
            rois=bg_rois
        )

        return bg_mask.to_image_mask(template_image)

    def get_cached_array(self, key, image_data):
        """Get cached numpy array for an ImageData object."""
        if key not in self.cached_arrays:
            self.cached_arrays[key] = image_data.as_array()
        return self.cached_arrays[key]

    def analyze_hkem_iteration(self, hkem_iter: int, bg_mask_img) -> Optional[pd.DataFrame]:
        """
        Analyze a single HKEM iteration using original vendor masks only.

        Returns:
            DataFrame with results or None if failed
        """
        try:
            self.logger.info(f"[HKEM] Starting analysis for iteration {hkem_iter}")
            # Load HKEM for this iteration
            hkem_path = self.hkem_dir / f"x_{hkem_iter}.hv"
            if not hkem_path.exists():
                self.logger.warning(f"HKEM iteration {hkem_iter} not found: {hkem_path}")
                return None

            hkem = ImageData(str(hkem_path))

            # Build list of reconstruction images to analyze
            recon_images = [
                ("HKEM", hkem),
                ("Vendor", self.reference_images['vendor']),
            ]

            # Cache background mask array
            bg_mask_arr = self.get_cached_array('bg_mask', bg_mask_img).astype(bool)

            # Compute background statistics using cached arrays
            bg_stats = {}
            for recon_name, recon_img in recon_images:
                recon_arr = self.get_cached_array(f"{recon_name}_hkem_{hkem_iter}", recon_img)
                bg_values = recon_arr[bg_mask_arr]
                bg_mean = float(np.mean(bg_values))
                bg_std = float(np.std(bg_values))
                bg_cov = (bg_std / bg_mean) * 100 if bg_mean > 0 else 0.0
                bg_stats[recon_name] = {
                    "mean": bg_mean,
                    "std": bg_std,
                    "cov": bg_cov
                }

            # Analyze lesions using ONLY vendor masks (no regrown masks)
            all_rows = []
            ref_name = 'vendor'  # Only use vendor masks

            if ref_name in self.auto_lesions_by_reference:
                auto_lesions = self.auto_lesions_by_reference[ref_name]

                for lesion_name in auto_lesions:
                    # Use cached mask array
                    mask_arr = self.cached_mask_arrays[ref_name][lesion_name]

                    for recon_name, recon_img in recon_images:
                        # Use cached reconstruction array
                        recon_arr = self.get_cached_array(f"{recon_name}_hkem_{hkem_iter}", recon_img)
                        lesion_values = recon_arr[mask_arr]

                        if lesion_values.size == 0:
                            continue

                        lesion_mean = float(np.mean(lesion_values))
                        lesion_max = float(np.max(lesion_values))
                        bg = bg_stats[recon_name]

                        all_rows.append({
                            "hkem_iter": hkem_iter,
                            "reference_image": ref_name,
                            "mask_type": "original",
                            "lesion": lesion_name,
                            "reconstruction": recon_name,
                            "lesion_mean": lesion_mean,
                            "lesion_max": lesion_max,
                            "tbr_mean": lesion_mean / bg["mean"] if bg["mean"] > 0 else np.nan,
                            "tbr_max": lesion_max / bg["mean"] if bg["mean"] > 0 else np.nan,
                            "background_mean": bg["mean"],
                            "background_std": bg["std"],
                            "background_cov_pct": bg["cov"]
                        })

            return {
                "hkem": hkem,
                "dataframe": pd.DataFrame(all_rows)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing HKEM iteration {hkem_iter}: {e}")
            return None
        finally:
            self.logger.info(f"[HKEM] Finished analysis for iteration {hkem_iter}")

    def analyze_alpha_beta_combination(self, alpha: float, beta: float,
                                      bg_mask_img) -> Optional[pd.DataFrame]:
        """
        Analyze a single alpha/beta reconstruction combination using original vendor masks only.

        Returns:
            DataFrame with results or None if failed
        """
        try:
            self.logger.info(f"[Alpha/Beta] Starting analysis for α={alpha}, β={beta}")
            # Load reconstruction images for this alpha/beta
            combo_key = (alpha, beta)
            if combo_key not in self.alpha_beta_paths:
                return None

            paths = self.alpha_beta_paths[combo_key]
            pet_path = paths.get(self.image_0_name)
            spect_path = paths.get(self.image_1_name)

            if not pet_path or not spect_path:
                self.logger.warning(f"Missing reconstruction files for α={alpha}, β={beta}")
                return None

            recon_pet = ImageData(pet_path)
            recon_spect = ImageData(spect_path)
            recon_spect_res = self.spect2pet.direct(recon_spect)

            # Build list of reconstruction images to analyze
            recon_images = [
                (f"Recon_PET_a{alpha}_b{beta}", recon_pet),
                (f"Recon_SPECT_a{alpha}_b{beta}", recon_spect_res),
            ]

            # Cache background mask array
            bg_mask_arr = self.get_cached_array('bg_mask', bg_mask_img).astype(bool)

            # Compute background statistics using cached arrays
            bg_stats = {}
            for recon_name, recon_img in recon_images:
                recon_arr = self.get_cached_array(f"{recon_name}_a{alpha}_b{beta}", recon_img)
                bg_values = recon_arr[bg_mask_arr]
                bg_mean = float(np.mean(bg_values))
                bg_std = float(np.std(bg_values))
                bg_cov = (bg_std / bg_mean) * 100 if bg_mean > 0 else 0.0
                bg_stats[recon_name] = {
                    "mean": bg_mean,
                    "std": bg_std,
                    "cov": bg_cov
                }

            # Analyze lesions using ONLY vendor masks (no regrown masks)
            all_rows = []
            ref_name = 'vendor'  # Only use vendor masks

            if ref_name in self.auto_lesions_by_reference:
                auto_lesions = self.auto_lesions_by_reference[ref_name]

                for lesion_name in auto_lesions:
                    # Use cached mask array
                    mask_arr = self.cached_mask_arrays[ref_name][lesion_name]

                    for recon_name, recon_img in recon_images:
                        # Use cached reconstruction array
                        recon_arr = self.get_cached_array(f"{recon_name}_a{alpha}_b{beta}", recon_img)
                        lesion_values = recon_arr[mask_arr]

                        if lesion_values.size == 0:
                            continue

                        lesion_mean = float(np.mean(lesion_values))
                        lesion_max = float(np.max(lesion_values))
                        bg = bg_stats[recon_name]

                        all_rows.append({
                            "alpha": alpha,
                            "beta": beta,
                            "reference_image": ref_name,
                            "mask_type": "original",
                            "lesion": lesion_name,
                            "reconstruction": recon_name,
                            "lesion_mean": lesion_mean,
                            "lesion_max": lesion_max,
                            "tbr_mean": lesion_mean / bg["mean"] if bg["mean"] > 0 else np.nan,
                            "tbr_max": lesion_max / bg["mean"] if bg["mean"] > 0 else np.nan,
                            "background_mean": bg["mean"],
                            "background_std": bg["std"],
                            "background_cov_pct": bg["cov"]
                        })

            return {
                "recon_pet": recon_pet,
                "recon_spect_res": recon_spect_res,
                "dataframe": pd.DataFrame(all_rows)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing α={alpha}, β={beta}: {e}")
            return None
        finally:
            self.logger.info(f"[Alpha/Beta] Finished analysis for α={alpha}, β={beta}")

    def save_hkem_results(self, hkem_iter: int, dataframe: pd.DataFrame):
        """Save HKEM analysis results to CSV file."""
        output_path = self.csv_dir / f"lesion_stats_hkem_iter{hkem_iter}.csv"
        dataframe.to_csv(output_path, index=False)
        self.logger.debug(f"Saved HKEM iter {hkem_iter} results to {output_path}")

    def save_alpha_beta_results(self, alpha: float, beta: float, dataframe: pd.DataFrame):
        """Save alpha/beta reconstruction results to CSV file."""
        def _sanitize(value):
            return str(value).replace('.', 'p')

        suffix = f"alpha_{_sanitize(alpha)}_beta_{_sanitize(beta)}"
        output_path = self.csv_dir / f"lesion_stats_{suffix}.csv"
        dataframe.to_csv(output_path, index=False)
        self.logger.debug(f"Saved α={alpha}, β={beta} results to {output_path}")

    def analyze_vendor(self, bg_mask_img) -> Optional[pd.DataFrame]:
        """
        Analyze the vendor reconstruction using original vendor masks.

        Returns:
            DataFrame with results or None if failed
        """
        try:
            self.logger.info("[Vendor] Starting analysis for vendor reconstruction")

            vendor = self.reference_images['vendor']

            # Cache background mask array
            bg_mask_arr = self.get_cached_array('bg_mask', bg_mask_img).astype(bool)

            # Compute background statistics
            vendor_arr = self.get_cached_array('vendor', vendor)
            bg_values = vendor_arr[bg_mask_arr]
            bg_mean = float(np.mean(bg_values))
            bg_std = float(np.std(bg_values))
            bg_cov = (bg_std / bg_mean) * 100 if bg_mean > 0 else 0.0

            # Analyze lesions using vendor masks
            all_rows = []
            ref_name = 'vendor'

            if ref_name in self.auto_lesions_by_reference:
                auto_lesions = self.auto_lesions_by_reference[ref_name]

                for lesion_name in auto_lesions:
                    # Use cached mask array
                    mask_arr = self.cached_mask_arrays[ref_name][lesion_name]
                    lesion_values = vendor_arr[mask_arr]

                    if lesion_values.size == 0:
                        continue

                    lesion_mean = float(np.mean(lesion_values))
                    lesion_max = float(np.max(lesion_values))

                    all_rows.append({
                        "reference_image": ref_name,
                        "mask_type": "original",
                        "lesion": lesion_name,
                        "reconstruction": "Vendor",
                        "lesion_mean": lesion_mean,
                        "lesion_max": lesion_max,
                        "tbr_mean": lesion_mean / bg_mean if bg_mean > 0 else np.nan,
                        "tbr_max": lesion_max / bg_mean if bg_mean > 0 else np.nan,
                        "background_mean": bg_mean,
                        "background_std": bg_std,
                        "background_cov_pct": bg_cov
                    })

            return pd.DataFrame(all_rows)

        except Exception as e:
            self.logger.error(f"Error analyzing vendor reconstruction: {e}")
            return None
        finally:
            self.logger.info("[Vendor] Finished analysis for vendor reconstruction")

    def save_vendor_results(self, dataframe: pd.DataFrame):
        """Save vendor reconstruction results to CSV file."""
        output_path = self.csv_dir / "lesion_stats_vendor.csv"
        dataframe.to_csv(output_path, index=False)
        self.logger.debug(f"Saved vendor results to {output_path}")

    def generate_plots_for_chosen(self, hkem_27_results: Dict, chosen_results: Dict):
        """Generate visualization plots for HKEM 27 and chosen combination."""
        if self.chosen_key not in self.alpha_beta_combinations:
            self.logger.warning("Cannot generate plots: chosen combination not found")
            return

        self.logger.info(f"Generating plots for HKEM 27 and chosen combination α={self.chosen_key[0]}, β={self.chosen_key[1]}")

        try:
            # Define slices for viewing
            SL1 = (slice(None), 130, slice(0, 180))  # Axial
            SL2 = (58, slice(80, 180), slice(30, 130))  # Sagittal
            SL3 = (slice(10, 70), slice(80, 180), 90)  # Coronal
            slices = (SL1, SL2, SL3)

            hkem_27 = hkem_27_results['hkem']
            recon_pet = chosen_results['recon_pet']
            recon_spect_res = chosen_results['recon_spect_res']
            vendor = self.reference_images['vendor']
            attenuation = self.pet_data["attenuation"]

            # Plot PET overlays for main reconstructions
            self._plot_pet_overlay(recon_pet, attenuation, slices,
                                  title=f"Recon PET (α={self.chosen_key[0]}, β={self.chosen_key[1]})",
                                  filename="recon_pet.png", vmax=0.000015)

            self._plot_pet_overlay(recon_spect_res, attenuation, slices,
                                  title=f"SPECT Resampled (α={self.chosen_key[0]}, β={self.chosen_key[1]})",
                                  filename="recon_spect.png")

            self._plot_pet_overlay(hkem_27, attenuation, slices,
                                  title="HKEM iter 27",
                                  filename="hkem_27.png", vmax=0.00001)

            self._plot_pet_overlay(vendor, attenuation, slices,
                                  title="Vendor",
                                  filename="vendor.png")

            # Plot with lesion masks from each reference
            for ref_name, auto_lesions in self.auto_lesions_by_reference.items():
                self._plot_with_masks(hkem_27, auto_lesions, "lesion_1", slices,
                                    title=f"HKEM 27 with Lesions (ref: {ref_name})",
                                    filename=f"hkem_27_lesions_ref_{ref_name}.png")

                self._plot_with_masks(recon_pet, auto_lesions, "lesion_1", slices,
                                    title=f"Recon PET with Lesions (ref: {ref_name})",
                                    filename=f"recon_pet_lesions_ref_{ref_name}.png")

            self.logger.info("Plots generated successfully")

        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")

    def _plot_pet_overlay(self, pet_img, attenuation_img, slices, title="",
                         filename="", vmax=None):
        """Plot PET image overlaid on attenuation map."""
        SL1, SL2, SL3 = slices
        pet_arr = pet_img.as_array()
        att_arr = attenuation_img.as_array()

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        # Slice 1
        ax[0].imshow(att_arr[SL1], cmap='gray', aspect='auto')
        im0 = ax[0].imshow(pet_arr[SL1], cmap=cmasher.fall, aspect='auto',
                          alpha=0.5, vmin=0, vmax=vmax)
        ax[0].set_title(f"{title} - Axial")
        ax[0].axis('off')
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        # Slice 2
        ax[1].imshow(att_arr[SL2], cmap='gray', aspect='auto')
        im1 = ax[1].imshow(pet_arr[SL2], cmap=cmasher.fall, aspect='auto',
                          alpha=0.5, vmin=0, vmax=vmax)
        ax[1].set_title(f"{title} - Sagittal")
        ax[1].axis('off')
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        # Slice 3
        ax[2].imshow(att_arr[SL3], cmap='gray', aspect='auto')
        im2 = ax[2].imshow(pet_arr[SL3], cmap=cmasher.fall, aspect='auto',
                          alpha=0.5, vmin=0, vmax=vmax)
        ax[2].set_title(f"{title} - Coronal")
        ax[2].axis('off')
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_with_masks(self, img, masks, focus, slices, title="", filename=""):
        """Plot image with ROI mask contours."""
        import itertools

        SL1, SL2, SL3 = slices
        img_arr = img.as_array()

        palette = ["#ff69b4", "#9370db", "#4169e1", "#40e0d0"]
        cycle = itertools.cycle(palette)
        colors = {k: ('yellow' if k == focus else next(cycle)) for k in masks}

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        # Slice 1
        ax[0].imshow(img_arr[SL1], cmap=cmasher.fall, aspect='auto')
        for k, mask_img in masks.items():
            m = mask_img.as_array()[SL1].astype(bool)
            if m.any():
                ax[0].contour(m, levels=[0.5], colors=colors[k], linewidths=1.5)
        ax[0].set_title(f"Axial (focus: {focus})")
        ax[0].axis('off')

        # Slice 2
        ax[1].imshow(img_arr[SL2], cmap=cmasher.fall, aspect='auto')
        for k, mask_img in masks.items():
            m = mask_img.as_array()[SL2].astype(bool)
            if m.any():
                ax[1].contour(m, levels=[0.5], colors=colors[k], linewidths=1.5)
        ax[1].set_title(f"Sagittal (focus: {focus})")
        ax[1].axis('off')

        # Slice 3
        ax[2].imshow(img_arr[SL3], cmap=cmasher.fall, aspect='auto')
        for k, mask_img in masks.items():
            m = mask_img.as_array()[SL3].astype(bool)
            if m.any():
                ax[2].contour(m, levels=[0.5], colors=colors[k], linewidths=1.5)
        ax[2].set_title(f"Coronal (focus: {focus})")
        ax[2].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def run_analysis(self, num_workers=4):
        """Run the complete analysis workflow with optional parallelization."""
        self.logger.info("=" * 80)
        self.logger.info("Starting automated reconstruction analysis")
        num_workers = max(1, int(num_workers))
        self.logger.info(f"Using {num_workers} parallel workers")
        self.logger.info("=" * 80)

        # Step 1: Load patient data
        self.load_patient_data()

        # Step 2: Discover reconstructions
        self.discover_reconstructions()

        # Step 3: Load reference images
        self.load_reference_images()

        # Step 4: Load pre-computed lesion masks
        self.load_lesion_masks()

        # Step 5: Create background mask
        template_img = self.reference_images['vendor']
        bg_mask_img = self.create_background_mask(template_img)

        # Save background mask if requested
        if self.save_lesions:
            bg_mask_path = self.lesions_dir / "background_mask.hv"
            bg_mask_img.write(str(bg_mask_path))
            self.logger.info(f"Saved background mask to {bg_mask_path}")

        # Step 6: Analyze vendor reconstruction
        vendor_csv = self.csv_dir / "lesion_stats_vendor.csv"
        if not vendor_csv.exists():
            self.logger.info("=" * 80)
            self.logger.info("Analyzing vendor reconstruction")
            self.logger.info("=" * 80)
            vendor_df = self.analyze_vendor(bg_mask_img)
            if vendor_df is not None and not vendor_df.empty:
                self.save_vendor_results(vendor_df)
                self.logger.info("Vendor analysis complete")
            else:
                self.logger.warning("Vendor analysis failed or returned no results")
        else:
            self.logger.info(f"Vendor analysis already exists, skipping: {vendor_csv}")

        # Step 7: Analyze HKEM iterations
        total_hkem = len(self.hkem_iterations)
        total_alpha_beta = len(self.alpha_beta_combinations)

        self.logger.info("=" * 80)
        self.logger.info(f"Processing {total_hkem} HKEM iterations + "
                        f"{total_alpha_beta} alpha/beta combinations")
        self.logger.info("=" * 80)

        hkem_27_results = None
        chosen_results = None

        # Process HKEM iterations
        self.logger.info("\n" + "="*80)
        self.logger.info("Processing HKEM iterations in parallel")
        self.logger.info("="*80)

        # Identify which iterations need processing
        hkem_to_process = []
        for hkem_iter in self.hkem_iterations:
            check_file = self.csv_dir / f"lesion_stats_hkem_iter{hkem_iter}.csv"
            if not check_file.exists():
                hkem_to_process.append(hkem_iter)
            elif hkem_iter == 27:
                # Load HKEM 27 for plotting even if already processed
                result = self.analyze_hkem_iteration(hkem_iter, bg_mask_img)
                if result is not None:
                    hkem_27_results = result

        self.logger.info(f"Found {len(hkem_to_process)} HKEM iterations to process (skipping {total_hkem - len(hkem_to_process)} existing)")

        if hkem_to_process:
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_iter = {
                    executor.submit(self.analyze_hkem_iteration, hkem_iter, bg_mask_img): hkem_iter
                    for hkem_iter in hkem_to_process
                }

                for future in as_completed(future_to_iter):
                    hkem_iter = future_to_iter[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self.logger.error(f"HKEM worker failed for iteration {hkem_iter}: {exc}")
                        continue
                    results.append((hkem_iter, result))

            # Save results
            for hkem_iter, result in sorted(results, key=lambda item: item[0]):
                if result is None:
                    self.logger.warning(f"Analysis failed for HKEM iteration {hkem_iter}")
                    continue

                # Save HKEM 27 for plotting
                if hkem_iter == 27:
                    hkem_27_results = result

                # Save results to CSV
                self.save_hkem_results(hkem_iter, result['dataframe'])
                self.logger.info(f"Completed HKEM iteration {hkem_iter}")

        # Process alpha/beta combinations
        self.logger.info("\n" + "="*80)
        self.logger.info("Processing alpha/beta reconstructions in parallel")
        self.logger.info("="*80)

        def _sanitize(value):
            return str(value).replace('.', 'p')

        # Identify which combinations need processing
        alpha_beta_to_process = []
        for alpha, beta in self.alpha_beta_combinations:
            suffix = f"alpha_{_sanitize(alpha)}_beta_{_sanitize(beta)}"
            check_file = self.csv_dir / f"lesion_stats_{suffix}.csv"
            if not check_file.exists():
                alpha_beta_to_process.append((alpha, beta))
            elif (alpha, beta) == self.chosen_key:
                # Load chosen combination for plotting even if already processed
                result = self.analyze_alpha_beta_combination(alpha, beta, bg_mask_img)
                if result is not None:
                    chosen_results = result

        self.logger.info(f"Found {len(alpha_beta_to_process)} alpha/beta combinations to process (skipping {total_alpha_beta - len(alpha_beta_to_process)} existing)")

        if alpha_beta_to_process:
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_combo = {
                    executor.submit(self.analyze_alpha_beta_combination, alpha, beta, bg_mask_img): (alpha, beta)
                    for (alpha, beta) in alpha_beta_to_process
                }

                for future in as_completed(future_to_combo):
                    combo = future_to_combo[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self.logger.error(f"Worker failed for α={combo[0]}, β={combo[1]}: {exc}")
                        continue
                    results.append((combo, result))

            # Save results
            for (alpha, beta), result in sorted(results, key=lambda item: item[0]):
                if result is None:
                    self.logger.warning(f"Analysis failed for α={alpha}, β={beta}")
                    continue

                # Save chosen combination for plotting
                if (alpha, beta) == self.chosen_key:
                    chosen_results = result

                # Save results to CSV
                self.save_alpha_beta_results(alpha, beta, result['dataframe'])
                self.logger.info(f"Completed α={alpha}, β={beta}")

        # Generate plots if we have both HKEM 27 and chosen combination
        if hkem_27_results is not None and chosen_results is not None:
            self.logger.info(f"\n{'='*80}")
            self.logger.info("Generating visualization plots")
            self.logger.info(f"{'='*80}")
            self.generate_plots_for_chosen(hkem_27_results, chosen_results)
        else:
            self.logger.warning("Cannot generate plots: missing HKEM 27 or chosen combination results")

        self.logger.info("=" * 80)
        self.logger.info("Analysis complete!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"CSV files: {self.csv_dir}")
        self.logger.info(f"Plots: {self.plots_dir}")
        self.logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated reconstruction analysis for all alpha/beta combinations and HKEM iterations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--patient', type=str, required=True,
                       help='Patient ID (e.g., sirt3)')
    parser.add_argument('--suffix', type=str, default="",
                       help='Suffix for results directory naming')
    parser.add_argument('--chosen-alpha', type=float, default=0.01,
                       help='Alpha value for chosen reference reconstruction')
    parser.add_argument('--chosen-beta', type=float, default=0.01,
                       help='Beta value for chosen reference reconstruction')
    parser.add_argument('--hkem-start', type=int, default=9,
                       help='Starting HKEM iteration')
    parser.add_argument('--hkem-end', type=int, default=450,
                       help='Ending HKEM iteration')
    parser.add_argument('--hkem-step', type=int, default=9,
                       help='Step size for HKEM iterations')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Iteration number for image files (default: None uses final_image_*.hv)')
    parser.add_argument('--output-dir', type=pathlib.Path, default=None,
                       help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for processing (default: 4)')
    parser.add_argument('--save-lesions', action='store_true',
                       help='Save lesion masks to disk (default: False)')
    parser.add_argument('--lesion-method', type=str, default='local_background',
                       help='Lesion growing method used to generate masks (default: local_background)')
    parser.add_argument('--lesion-masks-dir', type=pathlib.Path, default=None,
                       help='Directory containing pre-computed lesion masks (default: /home/storage/cluster/patient_sweeps/lesion_masks/{patient}/{method}/)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = ReconstructionAnalyzer(
        patient=args.patient,
        suffix=args.suffix,
        chosen_alpha=args.chosen_alpha,
        chosen_beta=args.chosen_beta,
        hkem_start=args.hkem_start,
        hkem_end=args.hkem_end,
        hkem_step=args.hkem_step,
        iteration=args.iteration,
        output_dir=args.output_dir,
        save_lesions=args.save_lesions,
        lesion_method=args.lesion_method,
        lesion_masks_dir=args.lesion_masks_dir
    )

    # Run analysis
    try:
        analyzer.run_analysis(num_workers=args.workers)
    except Exception as e:
        analyzer.logger.error(f"Analysis failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
