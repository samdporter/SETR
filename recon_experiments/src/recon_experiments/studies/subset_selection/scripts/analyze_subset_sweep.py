#!/usr/bin/env python3
"""
Analyze subset selection sweep results against convergence references.

This script compares different subset/prior configurations to convergence references,
measuring both convergence speed and accuracy.

Usage:
    python analyze_subset_sweep.py --sweep subset_selection_main --baseline subset_selection_convergence
    
    # For online monitoring:
    python analyze_subset_sweep.py --sweep subset_selection_main --baseline subset_selection_convergence --watch
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sirf.STIR import ImageData


def parse_config_from_dirname(dirname: str) -> Dict[str, str]:
    """Extract configuration from directory name."""
    # Expected format: subset_<mode>_prior_<mode>_precond_<type>_gamma_<value>
    parts = {}
    match = re.search(
        r"subset_(?P<subset_mode>.+?)_prior_(?P<prior_mode>.+?)_precond_(?P<precond_type>.+?)_gamma_(?P<gamma_tnv>.+)$",
        dirname,
    )
    if not match:
        return parts

    parts.update({k: v for k, v in match.groupdict().items() if k != "gamma_tnv"})
    try:
        parts["gamma_tnv"] = float(match.group("gamma_tnv"))
    except ValueError:
        parts["gamma_tnv"] = None
    return parts


def load_convergence_reference(ref_dir: Path, gamma: float) -> Optional[Dict]:
    """Load convergence reference for a specific gamma value."""
    # Find matching gamma directory
    ref_subdirs = [d for d in ref_dir.iterdir() if d.is_dir() and f'gamma_{gamma}' in d.name]
    
    if not ref_subdirs:
        print(f"Warning: No convergence reference found for gamma={gamma}")
        return None
    
    ref_path = ref_subdirs[0]
    
    # Load objective history
    obj_file = ref_path / "objective.csv"
    if not obj_file.exists():
        print(f"Warning: No objective file at {obj_file}")
        return None
    
    obj_df = pd.read_csv(obj_file, header=None, names=['objective'])
    
    # Load args to get final runtime
    args_file = ref_path / "args.csv"
    runtime = None
    if args_file.exists():
        args_df = pd.read_csv(args_file)
        if 'runtime' in args_df.columns:
            runtime = float(args_df['runtime'].iloc[0])
    
    # Find final images
    image_pet = sorted(ref_path.glob("image_0_*.hv"))
    image_spect = sorted(ref_path.glob("image_1_*.hv"))
    
    return {
        'gamma': gamma,
        'objective_history': obj_df['objective'].values,
        'final_objective': obj_df['objective'].iloc[-1],
        'total_runtime': runtime,
        'final_image_pet': str(image_pet[-1]) if image_pet else None,
        'final_image_spect': str(image_spect[-1]) if image_spect else None,
        'path': str(ref_path),
    }


def load_sweep_result(result_dir: Path) -> Optional[Dict]:
    """Load a single sweep result."""
    if not result_dir.exists():
        return None
    
    # Parse configuration from directory name
    config = parse_config_from_dirname(result_dir.name)
    
    # Load objective history
    obj_file = result_dir / "objective.csv"
    if not obj_file.exists():
        return None
    
    obj_df = pd.read_csv(obj_file, header=None, names=['objective'])
    config['objective_history'] = obj_df['objective'].values
    config['final_objective'] = obj_df['objective'].iloc[-1]
    
    # Load runtime from args
    args_file = result_dir / "args.csv"
    if args_file.exists():
        args_df = pd.read_csv(args_file)
        if 'runtime' in args_df.columns:
            config['runtime'] = float(args_df['runtime'].iloc[0])
        else:
            config['runtime'] = None
    else:
        config['runtime'] = None
    
    # Find final images
    image_pet = sorted(result_dir.glob("image_0_*.hv"))
    image_spect = sorted(result_dir.glob("image_1_*.hv"))
    
    config['final_image_pet'] = str(image_pet[-1]) if image_pet else None
    config['final_image_spect'] = str(image_spect[-1]) if image_spect else None
    config['path'] = str(result_dir)
    
    return config


def compute_image_error(test_image_path: str, ref_image_path: str) -> Dict[str, float]:
    """Compute error metrics between test and reference images."""
    try:
        test_img = ImageData(test_image_path)
        ref_img = ImageData(ref_image_path)
        
        test_arr = test_img.as_array()
        ref_arr = ref_img.as_array()
        
        diff = test_arr - ref_arr
        
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_error = np.max(np.abs(diff))
        
        ref_norm = np.linalg.norm(ref_arr)
        relative_error = np.linalg.norm(diff) / ref_norm if ref_norm > 0 else np.inf
        
        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'relative_error': relative_error,
        }
    except Exception as e:
        print(f"Error computing image metrics: {e}")
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'max_error': np.nan,
            'relative_error': np.nan,
        }


def compute_convergence_metrics(
    test_obj_history: np.ndarray,
    ref_final_obj: float,
    threshold: float = 0.01,
) -> Dict:
    """Compute convergence speed metrics."""
    if test_obj_history is None or len(test_obj_history) == 0:
        return {
            'iterations_to_convergence': np.nan,
            'converged': False,
            'final_obj_gap': np.nan,
            'relative_gaps': None,
        }
    
    # Find when test objective gets within threshold of reference
    relative_gaps = np.abs(test_obj_history - ref_final_obj) / np.abs(ref_final_obj)
    converged_mask = relative_gaps < threshold
    
    if np.any(converged_mask):
        iterations_to_convergence = np.argmax(converged_mask)
        converged = True
    else:
        iterations_to_convergence = len(test_obj_history)
        converged = False
    
    final_obj_gap = relative_gaps[-1] if len(relative_gaps) > 0 else np.nan
    
    return {
        'iterations_to_convergence': iterations_to_convergence,
        'converged': converged,
        'final_obj_gap': final_obj_gap,
        'relative_gaps': relative_gaps,
    }


def analyze_sweep(
    sweep_dir: Path,
    ref_dir: Path,
    convergence_threshold: float = 0.01
) -> pd.DataFrame:
    """Analyze all sweep results against convergence references."""
    results = []
    
    # Find all sweep result directories
    sweep_subdirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('subset_')]
    
    print(f"Found {len(sweep_subdirs)} sweep results")
    
    for result_dir in sweep_subdirs:
        print(f"Processing {result_dir.name}...")
        
        # Load sweep result
        sweep_result = load_sweep_result(result_dir)
        if sweep_result is None:
            print(f"  Skipping {result_dir.name} - no objective file")
            continue
        
        gamma = sweep_result.get('gamma_tnv')
        if gamma is None:
            print(f"  Skipping {result_dir.name} - no gamma value")
            continue
        
        # Load corresponding convergence reference
        ref = load_convergence_reference(ref_dir, gamma)
        if ref is None:
            print(f"  Warning: No convergence reference for gamma={gamma}, skipping comparison")
            # Still record the result without comparison metrics
            results.append({
                'subset_mode': sweep_result.get('subset_mode'),
                'prior_mode': sweep_result.get('prior_mode'),
                'precond_type': sweep_result.get('precond_type'),
                'gamma_tnv': gamma,
                'final_objective': sweep_result['final_objective'],
                'runtime': sweep_result['runtime'],
                'ref_available': False,
            })
            continue
        
        # Compute convergence metrics
        conv_metrics = compute_convergence_metrics(
            sweep_result['objective_history'],
            ref['final_objective'],
            convergence_threshold,
        )
        
        # Compute image error metrics for both PET and SPECT
        pet_metrics = {}
        spect_metrics = {}
        
        if sweep_result['final_image_pet'] and ref['final_image_pet']:
            pet_metrics = compute_image_error(
                sweep_result['final_image_pet'],
                ref['final_image_pet'],
            )
            pet_metrics = {f'pet_{k}': v for k, v in pet_metrics.items()}
        
        if sweep_result['final_image_spect'] and ref['final_image_spect']:
            spect_metrics = compute_image_error(
                sweep_result['final_image_spect'],
                ref['final_image_spect'],
            )
            spect_metrics = {f'spect_{k}': v for k, v in spect_metrics.items()}
        
        # Combine all metrics
        result_summary = {
            'subset_mode': sweep_result.get('subset_mode'),
            'prior_mode': sweep_result.get('prior_mode'),
            'precond_type': sweep_result.get('precond_type'),
            'gamma_tnv': gamma,
            'final_objective': sweep_result['final_objective'],
            'ref_final_objective': ref['final_objective'],
            'runtime': sweep_result['runtime'],
            'ref_runtime': ref['total_runtime'],
            'converged': conv_metrics['converged'],
            'iterations_to_convergence': conv_metrics['iterations_to_convergence'],
            'final_obj_gap': conv_metrics['final_obj_gap'],
            'ref_available': True,
            **pet_metrics,
            **spect_metrics,
        }
        
        if sweep_result['runtime'] and ref['total_runtime']:
            result_summary['speedup_vs_ref'] = ref['total_runtime'] / sweep_result['runtime']
        else:
            result_summary['speedup_vs_ref'] = np.nan
        
        results.append(result_summary)
    
    return pd.DataFrame(results)


def plot_convergence_curves(
    sweep_dir: Path,
    ref_dir: Path,
    output_dir: Path,
    gamma_values: Optional[List[float]] = None,
):
    """Plot convergence curves grouped by gamma and factor."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    sweep_subdirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('subset_')]
    
    # Organize by gamma
    gamma_results = {}
    for result_dir in sweep_subdirs:
        result = load_sweep_result(result_dir)
        if result is None or result.get('objective_history') is None:
            continue
        
        gamma = result.get('gamma_tnv')
        if gamma_values and gamma not in gamma_values:
            continue
        
        if gamma not in gamma_results:
            gamma_results[gamma] = []
        
        gamma_results[gamma].append((result_dir.name, result))
    
    # Plot for each gamma value
    for gamma, results in gamma_results.items():
        ref = load_convergence_reference(ref_dir, gamma)
        if ref is None or ref['objective_history'] is None:
            print(f"No convergence reference for gamma={gamma}, skipping plot")
            continue
        
        # Create subplot for each factor combination
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Convergence Comparison (γ={gamma})', fontsize=16)
        
        # Plot reference on all subplots
        for ax in axes.flat:
            ax.plot(
                ref['objective_history'],
                label='Reference (1000 epochs)',
                linewidth=2,
                linestyle='--',
                color='black',
                alpha=0.7,
            )
        
        # Group by factors
        for name, result in results:
            subset_mode = result.get('subset_mode', 'unknown')
            prior_mode = result.get('prior_mode', 'unknown')
            precond_type = result.get('precond_type', 'unknown')
            
            # Determine which subplot
            row = 0 if subset_mode == 'separate' else 1
            col = 0 if prior_mode == 'always' else 1
            
            ax = axes[row, col]
            ax.plot(
                result['objective_history'],
                label=precond_type,
                alpha=0.8,
            )
            ax.set_title(f"{subset_mode} + {prior_mode}")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Objective Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'convergence_gamma_{gamma}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved convergence plot for gamma={gamma}")


def generate_summary_report(df: pd.DataFrame, output_file: Path):
    """Generate a summary report of experimental results."""
    with open(output_file, 'w') as f:
        f.write("# Subset Selection Analysis Report\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total tests: {len(df)}\n")
        f.write(f"- Tests with reference: {df['ref_available'].sum()}\n")
        f.write(f"- Converged tests: {df['converged'].sum()}\n\n")
        
        # Performance by factor
        f.write("## Performance by Factor\n\n")
        
        factors = ['subset_mode', 'prior_mode', 'precond_type', 'gamma_tnv']
        
        for factor in factors:
            if factor not in df.columns:
                continue
            
            f.write(f"### {factor.replace('_', ' ').title()}\n\n")
            
            summary = df.groupby(factor).agg({
                'converged': 'mean',
                'iterations_to_convergence': 'median',
                'runtime': 'median',
                'final_obj_gap': 'median',
            }).round(3)
            
            f.write(summary.to_markdown())
            f.write("\n\n")
        
        # Best configurations per gamma
        f.write("## Best Configurations by Gamma\n\n")
        
        for gamma in sorted(df['gamma_tnv'].unique()):
            gamma_df = df[df['gamma_tnv'] == gamma]
            
            f.write(f"### Gamma = {gamma}\n\n")
            
            # Fastest to converge
            converged = gamma_df[gamma_df['converged']]
            if len(converged) > 0:
                fastest = converged.nsmallest(1, 'iterations_to_convergence').iloc[0]
                f.write(f"**Fastest convergence:**\n")
                f.write(f"- Config: {fastest['subset_mode']} + {fastest['prior_mode']} + {fastest['precond_type']}\n")
                f.write(f"- Iterations: {fastest['iterations_to_convergence']:.0f}\n")
                f.write(f"- Runtime: {fastest['runtime']:.1f}s\n\n")
            
            # Most accurate
            with_ref = gamma_df[gamma_df['ref_available']]
            if len(with_ref) > 0:
                most_accurate = with_ref.nsmallest(1, 'final_obj_gap').iloc[0]
                f.write(f"**Most accurate:**\n")
                f.write(f"- Config: {most_accurate['subset_mode']} + {most_accurate['prior_mode']} + {most_accurate['precond_type']}\n")
                f.write(f"- Objective gap: {most_accurate['final_obj_gap']:.6f}\n")
                f.write(f"- Runtime: {most_accurate['runtime']:.1f}s\n\n")
        
        # Interaction effects
        f.write("## Interaction Effects\n\n")
        f.write("### Subset Mode × Prior Mode\n\n")
        
        interaction = df.groupby(['subset_mode', 'prior_mode']).agg({
            'iterations_to_convergence': 'median',
            'converged': 'mean',
        }).round(2)
        
        f.write(interaction.to_markdown())
        f.write("\n\n")
    
    print(f"Saved summary report to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze subset selection sweep results")
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Sweep results directory (e.g., 'subset_selection_main')",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Convergence reference directory (e.g., 'subset_selection_convergence')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis (default: <sweep>_analysis)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.01,
        help="Relative objective gap threshold for convergence (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Monitor results in real-time (rerun analysis periodically)",
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=300,
        help="Watch interval in seconds (default: 300)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent / "output"
    sweep_dir = base_dir / args.sweep
    ref_dir = base_dir / args.baseline
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_dir / f"{args.sweep}_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sweep directory: {sweep_dir}")
    print(f"Reference directory: {ref_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return
    
    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}")
        return
    
    def run_analysis():
        print("Running analysis...")
        
        # Analyze all results
        df = analyze_sweep(sweep_dir, ref_dir, args.convergence_threshold)
        
        # Save results
        results_file = output_dir / "analysis_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Saved analysis results to {results_file}")
        
        # Generate plots
        plot_convergence_curves(sweep_dir, ref_dir, output_dir)
        
        # Generate summary report
        report_file = output_dir / "summary_report.md"
        generate_summary_report(df, report_file)
        
        print()
        print("=" * 60)
        print("Analysis complete!")
        print(f"Results: {results_file}")
        print(f"Plots: {output_dir}")
        print(f"Report: {report_file}")
        print("=" * 60)
    
    if args.watch:
        import time
        print(f"Watching mode enabled (checking every {args.watch_interval}s)")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                run_analysis()
                print(f"\nWaiting {args.watch_interval}s before next check...")
                time.sleep(args.watch_interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        run_analysis()


if __name__ == "__main__":
    main()
