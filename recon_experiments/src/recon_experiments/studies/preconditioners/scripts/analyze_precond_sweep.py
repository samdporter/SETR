#!/usr/bin/env python3
"""
Analyze preconditioner sweep results against baseline reconstructions.

This script compares different preconditioners' convergence to the baseline solution,
measuring both speed and accuracy.

Usage:
    python analyze_precond_sweep.py --sweep precond_1bpos --baseline baselines_1bpos
    
    # For online monitoring during sweep:
    python analyze_precond_sweep.py --sweep precond_1bpos --baseline baselines_1bpos --watch
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sirf.STIR import ImageData


def load_baseline_data(baseline_dir: Path, alpha: float) -> Optional[Dict]:
    """Load baseline reconstruction data for a specific alpha value."""
    baseline_path = baseline_dir / f"baseline_alpha_{alpha}"
    
    if not baseline_path.exists():
        print(f"Warning: Baseline not found for alpha={alpha} at {baseline_path}")
        return None
    
    # Load metrics
    metrics_file = baseline_path / "baseline_metrics.json"
    if not metrics_file.exists():
        print(f"Warning: No metrics file at {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load objective history
    obj_file = baseline_path / "objective.csv"
    if obj_file.exists():
        obj_df = pd.read_csv(obj_file, header=None, names=['objective'])
        metrics['objective_history'] = obj_df['objective'].values
    else:
        metrics['objective_history'] = None
    
    # Find final image files
    image_files = sorted(baseline_path.glob("image_*.hv"))
    if image_files:
        metrics['final_image_path'] = str(image_files[-1])
    else:
        metrics['final_image_path'] = None
    
    return metrics


def load_sweep_result(result_dir: Path) -> Optional[Dict]:
    """Load a single sweep result."""
    if not result_dir.exists():
        return None
    
    # Load result summary
    result_file = result_dir / "result.csv"
    if not result_file.exists():
        return None
    
    result_df = pd.read_csv(result_file)
    result = result_df.iloc[0].to_dict()
    precond_type = result.get('precond_type', 'unknown')
    precond_combine = result.get('precond_combine', '') or result.get('combine', '')
    result['precond_combine'] = precond_combine
    result['precond_label'] = (
        f\"{precond_type}:{precond_combine}\" if precond_combine else precond_type
    )
    
    # Load objective history
    obj_file = result_dir / "objective.csv"
    if obj_file.exists():
        obj_df = pd.read_csv(obj_file, header=None, names=['objective'])
        result['objective_history'] = obj_df['objective'].values
    else:
        result['objective_history'] = None
    
    # Find final image
    image_files = sorted(result_dir.glob("image_*.hv"))
    if image_files:
        result['final_image_path'] = str(image_files[-1])
    else:
        result['final_image_path'] = None
    
    return result


def compute_image_error(test_image_path: str, baseline_image_path: str) -> Dict[str, float]:
    """Compute error metrics between test and baseline images."""
    try:
        test_img = ImageData(test_image_path)
        baseline_img = ImageData(baseline_image_path)
        
        test_arr = test_img.as_array()
        baseline_arr = baseline_img.as_array()
        
        # Compute various error metrics
        diff = test_arr - baseline_arr
        
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_error = np.max(np.abs(diff))
        
        # Normalized metrics
        baseline_norm = np.linalg.norm(baseline_arr)
        relative_error = np.linalg.norm(diff) / baseline_norm if baseline_norm > 0 else np.inf
        
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
    baseline_final_obj: float,
    threshold: float = 0.01,
) -> Dict[str, float]:
    """
    Compute convergence speed metrics.
    
    Args:
        test_obj_history: Objective values over iterations
        baseline_final_obj: Final objective of baseline
        threshold: Relative difference threshold for "converged"
    
    Returns:
        Dict with convergence metrics
    """
    if test_obj_history is None or len(test_obj_history) == 0:
        return {
            'iterations_to_convergence': np.nan,
            'converged': False,
            'final_obj_gap': np.nan,
        }
    
    # Find when test objective gets within threshold of baseline
    relative_gaps = np.abs(test_obj_history - baseline_final_obj) / np.abs(baseline_final_obj)
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


def analyze_sweep(sweep_dir: Path, baseline_dir: Path, convergence_threshold: float = 0.01) -> pd.DataFrame:
    """
    Analyze all sweep results against baselines.
    
    Returns:
        DataFrame with comparative metrics for each test
    """
    results = []
    
    # Find all sweep result directories
    sweep_subdirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('precond_')]
    
    print(f"Found {len(sweep_subdirs)} sweep results")
    
    for result_dir in sweep_subdirs:
        print(f"Processing {result_dir.name}...")
        
        # Load sweep result
        sweep_result = load_sweep_result(result_dir)
        if sweep_result is None:
            print(f"  Skipping {result_dir.name} - no result file")
            continue
        
        alpha = sweep_result.get('alpha', None)
        if alpha is None:
            print(f"  Skipping {result_dir.name} - no alpha value")
            continue

        precond_type = sweep_result.get('precond_type', 'unknown')
        precond_combine = sweep_result.get('precond_combine', '') or sweep_result.get('combine', '')
        precond_label = (
            f\"{precond_type}:{precond_combine}\" if precond_combine else precond_type
        )
        
        # Load corresponding baseline
        baseline = load_baseline_data(baseline_dir, alpha)
        if baseline is None:
            print(f"  Warning: No baseline for alpha={alpha}, skipping comparison")
            # Still record the result without comparison metrics
            results.append({
                'precond_type': precond_type,
                'precond_combine': precond_combine,
                'precond_label': precond_label,
                'alpha': alpha,
                'step_size': sweep_result['step_size'],
                'final_objective': sweep_result['final_objective'],
                'run_time': sweep_result['run_time'],
                'status': sweep_result['status'],
                'baseline_available': False,
            })
            continue
        
        # Compute convergence metrics
        conv_metrics = compute_convergence_metrics(
            sweep_result.get('objective_history'),
            baseline['final_objective'],
            convergence_threshold,
        )
        
        # Compute image error metrics if images available
        if sweep_result['final_image_path'] and baseline['final_image_path']:
            img_metrics = compute_image_error(
                sweep_result['final_image_path'],
                baseline['final_image_path'],
            )
        else:
            img_metrics = {
                'rmse': np.nan,
                'mae': np.nan,
                'max_error': np.nan,
                'relative_error': np.nan,
            }
        
        # Combine all metrics
        result_summary = {
            'precond_type': precond_type,
            'precond_combine': precond_combine,
            'precond_label': precond_label,
            'alpha': alpha,
            'step_size': sweep_result['step_size'],
            'final_objective': sweep_result['final_objective'],
            'baseline_final_objective': baseline['final_objective'],
            'run_time': sweep_result['run_time'],
            'baseline_run_time': baseline['total_runtime'],
            'status': sweep_result['status'],
            'baseline_available': True,
            'converged': conv_metrics['converged'],
            'iterations_to_convergence': conv_metrics['iterations_to_convergence'],
            'final_obj_gap': conv_metrics['final_obj_gap'],
            'image_rmse': img_metrics['rmse'],
            'image_relative_error': img_metrics['relative_error'],
            'speedup_vs_baseline': baseline['total_runtime'] / sweep_result['run_time'] if sweep_result['run_time'] > 0 else np.nan,
        }
        
        results.append(result_summary)
    
    return pd.DataFrame(results)


def plot_convergence_curves(
    sweep_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    alpha_values: Optional[List[float]] = None,
):
    """Plot convergence curves comparing different preconditioners to baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by alpha
    sweep_subdirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('precond_')]
    
    # Organize by alpha
    alpha_results = {}
    for result_dir in sweep_subdirs:
        sweep_result = load_sweep_result(result_dir)
        if sweep_result is None or sweep_result.get('objective_history') is None:
            continue
        
        alpha = sweep_result.get('alpha')
        if alpha_values and alpha not in alpha_values:
            continue
        
        if alpha not in alpha_results:
            alpha_results[alpha] = []
        
        alpha_results[alpha].append((result_dir.name, sweep_result))
    
    # Plot for each alpha
    for alpha, results in alpha_results.items():
        baseline = load_baseline_data(baseline_dir, alpha)
        if baseline is None or baseline['objective_history'] is None:
            print(f"No baseline for alpha={alpha}, skipping plot")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot baseline
        ax.plot(
            baseline['objective_history'],
            label=f"Baseline ({baseline.get('precond_type', 'unknown')})",
            linewidth=2,
            linestyle='--',
            color='black',
        )
        
        # Plot each test result
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (name, result), color in zip(results, colors):
            precond_label = result.get('precond_label', result.get('precond_type', 'unknown'))
            label = f\"{precond_label}, step={result['step_size']}\"
            ax.plot(result['objective_history'], label=label, alpha=0.7, color=color)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title(f'Convergence Comparison (α={alpha})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'convergence_alpha_{alpha}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved convergence plot for alpha={alpha}")


def generate_summary_report(df: pd.DataFrame, output_file: Path):
    """Generate a summary report of preconditioner performance."""
    with open(output_file, 'w') as f:
        f.write("# Preconditioner Sweep Analysis Report\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total tests: {len(df)}\n")
        f.write(f"- Successful tests: {df['status'].eq('success').sum()}\n")
        f.write(f"- Tests with baseline: {df['baseline_available'].sum()}\n")
        f.write(f"- Converged tests: {df['converged'].sum()}\n\n")
        
        # Best performers per alpha
        f.write("## Best Performers by Alpha\n\n")
        for alpha in sorted(df['alpha'].unique()):
            alpha_df = df[df['alpha'] == alpha]
            
            f.write(f"### Alpha = {alpha}\n\n")
            
            # Fastest to converge
            converged = alpha_df[alpha_df['converged']]
            if len(converged) > 0:
                fastest = converged.nsmallest(1, 'iterations_to_convergence').iloc[0]
                f.write(f"**Fastest convergence:**\n")
                f.write(f"- Preconditioner: {fastest.get('precond_label', fastest['precond_type'])}\n")
                f.write(f"- Step size: {fastest['step_size']}\n")
                f.write(f"- Iterations: {fastest['iterations_to_convergence']}\n")
                f.write(f"- Runtime: {fastest['run_time']:.1f}s\n\n")
            
            # Most accurate
            with_baseline = alpha_df[alpha_df['baseline_available']]
            if len(with_baseline) > 0:
                most_accurate = with_baseline.nsmallest(1, 'final_obj_gap').iloc[0]
                f.write(f"**Most accurate:**\n")
                f.write(f"- Preconditioner: {most_accurate.get('precond_label', most_accurate['precond_type'])}\n")
                f.write(f"- Step size: {most_accurate['step_size']}\n")
                f.write(f"- Objective gap: {most_accurate['final_obj_gap']:.6f}\n")
                f.write(f"- Runtime: {most_accurate['run_time']:.1f}s\n\n")
        
        # Preconditioner comparison
        f.write("## Preconditioner Comparison\n\n")
        precond_summary = df.groupby('precond_label').agg({
            'converged': 'mean',
            'iterations_to_convergence': 'median',
            'run_time': 'median',
            'final_obj_gap': 'median',
        }).round(3)
        f.write(precond_summary.to_markdown())
        f.write("\n\n")
    
    print(f"Saved summary report to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze preconditioner sweep results")
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Sweep results directory (e.g., 'precond_1bpos')",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline results directory (e.g., 'baselines_1bpos')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis results (default: <sweep>_analysis)",
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
    baseline_dir = base_dir / args.baseline
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_dir / f"{args.sweep}_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sweep directory: {sweep_dir}")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return
    
    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        return
    
    def run_analysis():
        print("Running analysis...")
        
        # Analyze all results
        df = analyze_sweep(sweep_dir, baseline_dir, args.convergence_threshold)
        
        # Save results
        results_file = output_dir / "analysis_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Saved analysis results to {results_file}")
        
        # Generate plots
        plot_convergence_curves(sweep_dir, baseline_dir, output_dir)
        
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
