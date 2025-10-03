#!/usr/bin/env python3
"""
Analyze and visualize results from preconditioner testing.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(results_file):
    """Load results from CSV."""
    df = pd.read_csv(results_file)
    # Filter to successful runs
    df = df[df['status'] == 'success'].copy()
    return df


def plot_objective_vs_params(df, output_dir):
    """Plot final objective vs alpha and step size for each preconditioner."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    precond_types = df['precond_type'].unique()
    colors = sns.color_palette("husl", len(precond_types))

    # Plot vs alpha
    ax = axes[0]
    for precond, color in zip(precond_types, colors):
        precond_df = df[df['precond_type'] == precond]
        # Group by alpha, take best step size
        best_per_alpha = precond_df.groupby('alpha')['final_objective'].min()

        ax.plot(best_per_alpha.index, best_per_alpha.values,
               marker='o', label=precond, color=color, linewidth=2)

    ax.set_xlabel('Alpha (=Beta)', fontsize=12)
    ax.set_ylabel('Best Final Objective', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Best Objective vs Penalty Strength', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot vs step size
    ax = axes[1]
    for precond, color in zip(precond_types, colors):
        precond_df = df[df['precond_type'] == precond]
        # Group by step size, take best alpha
        best_per_step = precond_df.groupby('step_size')['final_objective'].min()

        ax.plot(best_per_step.index, best_per_step.values,
               marker='o', label=precond, color=color, linewidth=2)

    ax.set_xlabel('Initial Step Size', fontsize=12)
    ax.set_ylabel('Best Final Objective', fontsize=12)
    ax.set_title('Best Objective vs Step Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'objective_vs_params.png', dpi=150)
    print(f"Saved: {output_dir / 'objective_vs_params.png'}")
    plt.close()


def plot_heatmaps(df, output_dir):
    """Create heatmaps of objective values for each preconditioner type."""

    precond_types = df['precond_type'].unique()
    n_precond = len(precond_types)

    fig, axes = plt.subplots(1, n_precond, figsize=(6*n_precond, 5))
    if n_precond == 1:
        axes = [axes]

    for ax, precond in zip(axes, precond_types):
        precond_df = df[df['precond_type'] == precond]

        # Pivot to create heatmap data
        pivot = precond_df.pivot_table(
            values='final_objective',
            index='alpha',
            columns='step_size',
            aggfunc='min'
        )

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis_r',
                   ax=ax, cbar_kws={'label': 'Final Objective'})
        ax.set_title(f'{precond}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step Size', fontsize=12)
        ax.set_ylabel('Alpha (=Beta)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'objective_heatmaps.png', dpi=150)
    print(f"Saved: {output_dir / 'objective_heatmaps.png'}")
    plt.close()


def plot_convergence(df, output_dir):
    """Plot convergence metrics for each preconditioner."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    precond_types = df['precond_type'].unique()
    colors = sns.color_palette("husl", len(precond_types))

    # Plot convergence vs alpha
    ax = axes[0]
    for precond, color in zip(precond_types, colors):
        precond_df = df[df['precond_type'] == precond]
        # Group by alpha, take median convergence
        conv_per_alpha = precond_df.groupby('alpha')['convergence_metric'].median()

        ax.plot(conv_per_alpha.index, conv_per_alpha.values,
               marker='o', label=precond, color=color, linewidth=2)

    ax.set_xlabel('Alpha (=Beta)', fontsize=12)
    ax.set_ylabel('Convergence Metric (median)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Convergence vs Penalty Strength', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot runtime vs preconditioner
    ax = axes[1]
    runtime_data = []
    labels = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]
        runtime_data.append(precond_df['run_time'].values)
        labels.append(precond)

    bp = ax.boxplot(runtime_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Run Time (seconds)', fontsize=12)
    ax.set_title('Runtime Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_runtime.png', dpi=150)
    print(f"Saved: {output_dir / 'convergence_runtime.png'}")
    plt.close()


def plot_precond_setup_time(df, output_dir):
    """Plot preconditioner setup time comparison."""

    precond_types = df['precond_type'].unique()
    colors = sns.color_palette("husl", len(precond_types))

    fig, ax = plt.subplots(figsize=(8, 5))

    setup_data = []
    labels = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]
        setup_data.append(precond_df['precond_setup_time'].values)
        labels.append(precond)

    bp = ax.boxplot(setup_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Preconditioner Setup Time (seconds)', fontsize=12)
    ax.set_title('Preconditioner Computation Cost', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'precond_setup_time.png', dpi=150)
    print(f"Saved: {output_dir / 'precond_setup_time.png'}")
    plt.close()


def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""

    precond_types = df['precond_type'].unique()

    summary = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]

        best_run = precond_df.loc[precond_df['final_objective'].idxmin()]

        summary.append({
            'Preconditioner': precond,
            'Best Objective': best_run['final_objective'],
            'Best Alpha': best_run['alpha'],
            'Best Step Size': best_run['step_size'],
            'Best Convergence': best_run['convergence_metric'],
            'Median Runtime (s)': precond_df['run_time'].median(),
            'Median Setup Time (s)': precond_df['precond_setup_time'].median(),
        })

    summary_df = pd.DataFrame(summary)

    # Save to file
    summary_file = output_dir / 'summary_table.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PRECONDITIONER COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

    print(f"Saved: {summary_file}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    return summary_df


def plot_detailed_comparison(df, output_dir):
    """Create detailed comparison for best configurations."""

    # Find best configuration for each preconditioner
    precond_types = df['precond_type'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(precond_types))

    # 1. Best objective values
    ax = axes[0]
    best_objs = []
    labels = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]
        best_objs.append(precond_df['final_objective'].min())
        labels.append(precond)

    bars = ax.bar(range(len(precond_types)), best_objs, color=colors)
    ax.set_xticks(range(len(precond_types)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Final Objective', fontsize=12)
    ax.set_title('Best Final Objective', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, best_objs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 2. Optimal alpha values
    ax = axes[1]
    best_alphas = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]
        best_run = precond_df.loc[precond_df['final_objective'].idxmin()]
        best_alphas.append(best_run['alpha'])

    bars = ax.bar(range(len(precond_types)), best_alphas, color=colors)
    ax.set_xticks(range(len(precond_types)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Optimal Alpha Value', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Optimal step size
    ax = axes[2]
    best_steps = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]
        best_run = precond_df.loc[precond_df['final_objective'].idxmin()]
        best_steps.append(best_run['step_size'])

    bars = ax.bar(range(len(precond_types)), best_steps, color=colors)
    ax.set_xticks(range(len(precond_types)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Step Size', fontsize=12)
    ax.set_title('Optimal Step Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Speedup relative to slowest
    ax = axes[3]
    runtimes = []
    for precond in precond_types:
        precond_df = df[df['precond_type'] == precond]
        runtimes.append(precond_df['run_time'].median())

    max_time = max(runtimes)
    speedups = [max_time / t for t in runtimes]

    bars = ax.bar(range(len(precond_types)), speedups, color=colors)
    ax.set_xticks(range(len(precond_types)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Speedup (relative to slowest)', fontsize=14, fontweight='bold')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}x', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_comparison.png', dpi=150)
    print(f"Saved: {output_dir / 'detailed_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze preconditioner test results')
    parser.add_argument('--results', type=str, default='results/preconditioner_tests/results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: same as results)')

    args = parser.parse_args()

    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    df = load_results(results_file)
    print(f"Loaded {len(df)} successful runs from {results_file}")

    if len(df) == 0:
        print("No successful runs found!")
        return

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating analysis plots in {output_dir}...")

    # Generate all plots
    generate_summary_table(df, output_dir)
    plot_objective_vs_params(df, output_dir)
    plot_heatmaps(df, output_dir)
    plot_convergence(df, output_dir)
    plot_precond_setup_time(df, output_dir)
    plot_detailed_comparison(df, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()