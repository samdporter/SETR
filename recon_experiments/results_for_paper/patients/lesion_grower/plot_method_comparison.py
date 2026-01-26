#!/usr/bin/env python3
"""
Create side-by-side comparison plots of intensity vs radial_gradient methods.
"""

import sys
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sirf.STIR as pet


def create_comparison_plot(patient: str, lesion_id: str, output_dir: pathlib.Path, source_image: str = "vendor"):
    """Create side-by-side comparison plot for a single lesion."""

    # Construct method directory names (same logic as grow_lesions.py)
    intensity_method = f"intensity_{source_image}" if source_image != "vendor" else "intensity"
    radial_method = f"radial_gradient_{source_image}" if source_image != "vendor" else "radial_gradient"

    # Load masks (from source-specific directories)
    intensity_path = pathlib.Path(f"/home/storage/cluster/patient_sweeps/lesion_masks/{patient}/{intensity_method}/{lesion_id}.hv")
    radial_path = pathlib.Path(f"/home/storage/cluster/patient_sweeps/lesion_masks/{patient}/{radial_method}/{lesion_id}.hv")

    # Always load vendor image for display (even if masks were grown on combined_recon)
    vendor_path = pathlib.Path(f"/home/storage/prepared_data/oxford_patient_data/{patient}/PET/non_tof/vendor_zoomed.hv")

    if not all([intensity_path.exists(), radial_path.exists(), vendor_path.exists()]):
        print(f"Warning: Missing files for {patient} {lesion_id}")
        return

    try:
        intensity_mask = pet.ImageData(str(intensity_path)).as_array()
        radial_mask = pet.ImageData(str(radial_path)).as_array()
        vendor = pet.ImageData(str(vendor_path)).as_array()
    except Exception as e:
        print(f"Error loading data for {patient} {lesion_id}: {e}")
        return

    # Find slice with most lesion voxels in intensity mask
    intensity_counts = intensity_mask.sum(axis=(1, 2))
    best_z = np.argmax(intensity_counts)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Get slice data
    vendor_slice = vendor[best_z, :, :]
    intensity_slice = intensity_mask[best_z, :, :]
    radial_slice = radial_mask[best_z, :, :]

    # Calculate intensity range using percentiles for better visibility
    vmin = np.percentile(vendor_slice[vendor_slice > 0], 1)
    vmax = np.percentile(vendor_slice[vendor_slice > 0], 99.5)

    # Plot vendor image
    im0 = axes[0].imshow(vendor_slice, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    axes[0].set_title('Vendor Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Plot intensity method overlay
    axes[1].imshow(vendor_slice, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    axes[1].contour(intensity_slice, colors='red', linewidths=2, levels=[0.5])
    axes[1].set_title(f'Intensity Method\n({int(intensity_mask.sum())} voxels)')
    axes[1].axis('off')

    # Plot radial gradient method overlay
    axes[2].imshow(vendor_slice, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    axes[2].contour(radial_slice, colors='blue', linewidths=2, levels=[0.5])
    axes[2].set_title(f'Radial Gradient Method\n({int(radial_mask.sum())} voxels)')
    axes[2].axis('off')

    # Plot overlap using contours (better PET visibility)
    # Draw both method contours directly overlaid
    axes[3].imshow(vendor_slice, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    axes[3].contour(intensity_slice, colors='red', linewidths=2, levels=[0.5], alpha=0.8)
    axes[3].contour(radial_slice, colors='blue', linewidths=2, levels=[0.5], alpha=0.8)
    axes[3].set_title('Overlap\n(Red=intensity, Blue=radial)')
    axes[3].axis('off')

    source_label = f"({source_image})" if source_image != "vendor" else ""
    plt.suptitle(f'{patient} - {lesion_id} {source_label} (z={best_z})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save 4-panel comparison plot
    suffix = f"_{source_image}" if source_image != "vendor" else ""
    output_path = output_dir / f"{patient}_{lesion_id}{suffix}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # Create standalone overlap-only plot (using contours for better visibility)
    fig_overlap, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(vendor_slice, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    # Draw both method contours directly overlaid
    ax.contour(intensity_slice, colors='red', linewidths=2, levels=[0.5], alpha=0.8)
    ax.contour(radial_slice, colors='blue', linewidths=2, levels=[0.5], alpha=0.8)
    ax.set_title(f'{patient} - {lesion_id} {source_label}\nOverlap (Red=intensity, Blue=radial)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    # Save overlap-only plot
    overlap_path = output_dir / f"{patient}_{lesion_id}{suffix}_overlap.png"
    plt.savefig(overlap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {overlap_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create comparison plots")
    parser.add_argument('--patients', nargs='+', default=['sirt3', 'sirt6'],
                       help='Patient IDs to plot')
    parser.add_argument('--num-lesions', type=int, default=5,
                       help='Number of lesions per patient')
    parser.add_argument('--source-image', type=str, default='vendor',
                       choices=['vendor', 'combined_recon'],
                       help='Source image for comparison: vendor or combined_recon (default: vendor)')
    parser.add_argument('--output-dir', type=pathlib.Path,
                       default=pathlib.Path("/home/storage/cluster/patient_sweeps/lesion_grower/comparison_plots"),
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating comparison plots...")
    print(f"Source image: {args.source_image}")
    print(f"Output directory: {args.output_dir}\n")

    for patient in args.patients:
        print(f"\nPatient: {patient}")
        for i in range(1, args.num_lesions + 1):
            lesion_id = f"lesion_{i}"
            create_comparison_plot(patient, lesion_id, args.output_dir, args.source_image)

    print(f"\n{'='*80}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
