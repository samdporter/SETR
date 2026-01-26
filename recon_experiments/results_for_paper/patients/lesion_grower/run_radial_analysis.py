#!/usr/bin/env python3
"""
Radial gradient-based lesion analysis workflow.

This script generates lesion masks using the radial gradient method (Mikell et al. 2018)
at different intensity thresholds and runs reconstruction analysis on each set of masks.

Usage:
    python run_radial_analysis.py --patient sirt3
    python run_radial_analysis.py --patient sirt3 --only-grow
    python run_radial_analysis.py --patient sirt3 --only-analyse
"""

import argparse
import pathlib
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate and analyse lesions using radial gradient method at different thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--patient', type=str, required=True,
                       help='Patient ID (e.g., sirt3)')
    parser.add_argument('--source-image', type=str, default='vendor',
                       choices=['vendor', 'combined_recon'],
                       help='Source image for lesion detection')
    parser.add_argument('--suffix', type=str, default="",
                       help='Suffix for results directory naming (matches analyse_reconstruction)')
    parser.add_argument('--force', action='store_true',
                       help='Regrow lesions even if masks already exist')
    parser.add_argument('--only-grow', action='store_true',
                       help='Only generate lesion masks, skip analysis')
    parser.add_argument('--only-analyse', action='store_true',
                       help='Only run analysis, skip lesion generation (assumes masks exist)')
    parser.add_argument('--chosen-alpha', type=float, default=0.01,
                       help='Alpha value for chosen reconstruction')
    parser.add_argument('--chosen-beta', type=float, default=0.01,
                       help='Beta value for chosen reconstruction')
    parser.add_argument('--hkem-start', type=int, default=9,
                       help='Starting HKEM iteration')
    parser.add_argument('--hkem-end', type=int, default=450,
                       help='Ending HKEM iteration')
    parser.add_argument('--hkem-step', type=int, default=9,
                       help='Step size for HKEM iterations')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Iteration number for image files (default: None uses final_image_*.hv)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers for analysis')
    parser.add_argument('--output-base', type=pathlib.Path, default=None,
                       help='Base output directory for analysis results')
    parser.add_argument('--background-template', type=pathlib.Path, default=None,
                       help='Template image for background mask (defaults to patient vendor_zoomed.hv)')
    parser.add_argument('--background-roi-json', type=str, default=None,
                       help='Optional JSON file overriding default ellipsoid params for background mask')
    parser.add_argument('--background-force', action='store_true',
                       help='Overwrite existing background mask')

    args = parser.parse_args()

    # Define radial gradient configuration (PET standard: 42%)
    radial_configs = [
        {
            'name': f'radial_gradient{"_combined_recon" if args.source_image == "combined_recon" else ""}',
            'min_fraction': 0.42,
            'description': '42% intensity threshold (PET standard)'
        }
    ]

    print("="*80)
    print("RADIAL GRADIENT LESION ANALYSIS WORKFLOW")
    print("="*80)
    print(f"Patient: {args.patient}")
    print(f"Source image: {args.source_image}")
    print(f"Suffix: {args.suffix or '(none)'}")
    print(f"Force regrow: {args.force}")
    print("Threshold: 42% (PET standard)")
    print(f"Mode: {'Generate only' if args.only_grow else 'analyse only' if args.only_analyse else 'Generate and analyse'}")
    print("="*80)

    # Step 1: Generate lesion masks (unless --only-analyse)
    if not args.only_analyse:
        print("\n\n" + "="*80)
        print("STEP 1: GENERATING LESION MASKS")
        print("="*80)

        # Ensure background mask exists
        print("\nCreating background mask (if needed)")
        bg_cmd = [
            'python', 'create_background_mask.py',
            '--patient', args.patient
        ]
        if args.background_template:
            bg_cmd += ['--template', str(args.background_template)]
        if args.background_roi_json:
            bg_cmd += ['--roi-json', args.background_roi_json]
        if args.background_force:
            bg_cmd += ['--force']
        run_command(bg_cmd, "Create background mask")

        for config in radial_configs:
            cmd = [
                'python', 'grow_lesions.py',
                '--patient', args.patient,
                '--method', 'radial_gradient',
                '--source-image', args.source_image,
                '--min-fraction-of-seed', str(config['min_fraction']),
                '--chosen-alpha', str(args.chosen_alpha),
                '--chosen-beta', str(args.chosen_beta),
            ]
            if args.force:
                cmd.append('--force')

            description = f"Generating lesions with {config['description']}"

            print(f"\n\nConfiguration: {config['name']}")
            print(f"  Method: radial_gradient")
            print(f"  Source: {args.source_image}")
            print(f"  Min fraction of seed: {config['min_fraction']}")
            print(f"  Description: {config['description']}")

            result = run_command(cmd, description)

            # The output directory already has the correct name (no rename needed)
            output_dir = pathlib.Path('/home/storage/cluster/patient_sweeps/lesion_masks') / args.patient / config['name']
            if output_dir.exists():
                print(f"  → Saved to: {output_dir}")

    # Stop here if --only-grow
    if args.only_grow:
        print("\n\n" + "="*80)
        print("Lesion generation complete! (--only-grow specified)")
        print("="*80)
        return

    # Step 2: Run analysis for each set of lesions
    print("\n\n" + "="*80)
    print("STEP 2: RUNNING RECONSTRUCTION ANALYSIS")
    print("="*80)

    for config in radial_configs:
        lesion_dir = pathlib.Path('/home/storage/cluster/patient_sweeps/lesion_masks') / args.patient / config['name']

        # Check if lesion masks exist
        if not lesion_dir.exists():
            print(f"\nWARNING: Lesion masks not found for {config['name']}: {lesion_dir}")
            print("  Run without --only-analyse first to generate masks")
            continue

        # Set up output directory
        if args.output_base:
            output_dir = args.output_base / args.patient / config['name']
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = pathlib.Path(
                f"/home/sam/mnt/comic-sporter/synergistic_Y90/SETR/sweeps/output/2bpos_alpha_beta_{args.patient}{f'_{args.suffix}' if args.suffix else ''}"
            )
            output_dir = base_dir / f"analysis_{config['name']}_{timestamp}"

        cmd = [
            'python', 'analyse_reconstruction_v2.py',
            '--patient', args.patient,
            '--chosen-alpha', str(args.chosen_alpha),
            '--chosen-beta', str(args.chosen_beta),
            '--hkem-start', str(args.hkem_start),
            '--hkem-end', str(args.hkem_end),
            '--hkem-step', str(args.hkem_step),
            '--suffix', args.suffix,
            '--lesion-method', config['name'],
            '--lesion-masks-dir', str(lesion_dir),
            '--output-dir', str(output_dir),
            '--workers', str(args.workers)
        ]

        # Add iteration parameter if specified
        if args.iteration is not None:
            cmd.extend(['--iteration', str(args.iteration)])

        description = f"Analyzing reconstructions with {config['description']} lesions"

        print(f"\n\nConfiguration: {config['name']}")
        print(f"  Lesion masks: {lesion_dir}")
        print(f"  Output: {output_dir}")

        run_command(cmd, description)

    # Summary
    print("\n\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated lesion masks:")
    for config in radial_configs:
        lesion_dir = pathlib.Path('/home/storage/cluster/patient_sweeps/lesion_masks') / args.patient / config['name']
        status = "✓" if lesion_dir.exists() else "✗"
        print(f"  {status} {config['name']}: {lesion_dir}")

    print("\nAnalysis results saved to respective output directories")
    print("="*80)


if __name__ == "__main__":
    main()
