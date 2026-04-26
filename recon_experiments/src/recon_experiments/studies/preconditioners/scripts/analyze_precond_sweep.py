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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sirf.STIR import ImageData


def _read_objective_history(obj_file: Path) -> Optional[np.ndarray]:
    """Read objective history from CSV, supporting 1- or 2-column formats."""
    if not obj_file.exists():
        return None
    try:
        obj_df = pd.read_csv(obj_file, header=None)
    except Exception as exc:
        print(f"Warning: Could not read objective file {obj_file}: {exc}")
        return None
    if obj_df.empty:
        return None
    series = pd.to_numeric(obj_df.iloc[:, -1], errors="coerce")
    series = series.dropna()
    if series.empty:
        return None
    return series.to_numpy(dtype=float)


def _read_single_row_csv(path: Path) -> Optional[Dict]:
    """Read a single-row CSV as a dict."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: Could not read CSV {path}: {exc}")
        return None
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _coerce_float(value) -> Optional[float]:
    """Convert value to float, returning None on failure/NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _normalise_combine(value) -> str:
    """Normalize combine-mode values, treating NaN/None as empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _parse_repeat_metadata(result_dir_name: str) -> Tuple[str, Optional[int]]:
    """
    Parse repeat metadata from directory names.

    Returns:
        setting_id: Directory name with optional _rep_<n> suffix removed.
        repeat_id: Parsed repeat index (1-based) if available, else None.
    """
    match = re.match(r"^(.*)_rep_(\d+)$", result_dir_name)
    if match:
        return match.group(1), int(match.group(2))
    return result_dir_name, None


def _safe_percentile(series: pd.Series, q: float) -> float:
    """NaN-safe percentile helper for pandas series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.percentile(clean, q))


def _parse_percentiles(spec: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """Parse comma-separated percentile specification."""
    if spec is None:
        return default
    values: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            p = int(token)
        except ValueError:
            raise ValueError(f"Invalid percentile value '{token}' in '{spec}'.")
        if not (0 <= p <= 100):
            raise ValueError(f"Percentile {p} out of range [0, 100].")
        values.append(p)
    if not values:
        return default
    return tuple(sorted(set(values)))


def _load_allowed_alphas_from_csv(csv_path: Path) -> Optional[Tuple[float, ...]]:
    """Load allowed alpha values from a CSV with an 'alpha' column (or first column fallback)."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Warning: Could not read alpha CSV {csv_path}: {exc}")
        return None
    if df.empty:
        return None

    if "alpha" in df.columns:
        series = df["alpha"]
    else:
        series = df.iloc[:, 0]

    vals = []
    for raw in series:
        a = _coerce_float(raw)
        if a is not None:
            vals.append(float(a))
    if not vals:
        return None
    return tuple(sorted(set(vals)))


def _parse_alpha_values(spec: Optional[str]) -> Optional[Tuple[float, ...]]:
    """Parse comma-separated alpha values string."""
    if spec is None:
        return None
    vals = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(float(token))
        except ValueError:
            raise ValueError(f"Invalid alpha value '{token}' in --alpha-values")
    if not vals:
        return None
    return tuple(sorted(set(vals)))


def _build_curve_statistics(histories: List[np.ndarray], inner_band: Tuple[int, int] = (10, 90)) -> Optional[Dict[str, np.ndarray]]:
    """Build mean/min/max and inner-percentile bands from repeated objective histories."""
    if not histories:
        return None
    lengths = [len(h) for h in histories if h is not None and len(h) > 0]
    if not lengths:
        return None

    max_len = max(lengths)
    stack = np.full((len(histories), max_len), np.nan, dtype=float)
    for idx, h in enumerate(histories):
        if h is None or len(h) == 0:
            continue
        n = len(h)
        stack[idx, :n] = h

    valid = np.any(np.isfinite(stack), axis=0)
    if not np.any(valid):
        return None

    low_q, high_q = inner_band
    x = np.arange(max_len)
    stats = {
        "x": x[valid],
        "mean": np.nanmean(stack, axis=0)[valid],
        "min": np.nanmin(stack, axis=0)[valid],
        "max": np.nanmax(stack, axis=0)[valid],
        "inner_low": np.nanpercentile(stack, low_q, axis=0)[valid],
        "inner_high": np.nanpercentile(stack, high_q, axis=0)[valid],
        "num_runs": len(histories),
    }
    return stats


def load_baseline_data(baseline_dir: Path, alpha: float) -> Optional[Dict]:
    """Load baseline reconstruction data for a specific alpha value."""
    baseline_path = baseline_dir / f"baseline_alpha_{alpha}"
    
    if not baseline_path.exists():
        print(f"Warning: Baseline not found for alpha={alpha} at {baseline_path}")
        return None
    
    metrics = {}

    # Load metrics (preferred)
    metrics_file = baseline_path / "baseline_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        print(f"Warning: No metrics file at {metrics_file}. Falling back to args/result/objective.")
    
    # Load objective history
    obj_file = baseline_path / "objective.csv"
    objective_history = _read_objective_history(obj_file)
    metrics['objective_history'] = objective_history

    if _coerce_float(metrics.get('final_objective')) is None and objective_history is not None:
        metrics['final_objective'] = float(objective_history[-1])

    if _coerce_float(metrics.get('total_runtime')) is None:
        result_row = _read_single_row_csv(baseline_path / "result.csv")
        if result_row is not None:
            metrics['total_runtime'] = (
                _coerce_float(result_row.get('run_time'))
                or _coerce_float(result_row.get('runtime'))
                or np.nan
            )

    if not metrics.get('precond_type'):
        args_row = _read_single_row_csv(baseline_path / "args.csv")
        if args_row is not None:
            metrics['precond_type'] = args_row.get('precond_type', 'unknown')

    if _coerce_float(metrics.get('final_objective')) is None:
        print(f"Warning: Could not determine baseline final objective at {baseline_path}")
        return None
    metrics['final_objective'] = float(metrics['final_objective'])
    if _coerce_float(metrics.get('total_runtime')) is None:
        metrics['total_runtime'] = np.nan
    
    # Find final image files
    image_files = sorted(baseline_path.glob("image_*.hv"))
    if not image_files:
        image_files = sorted(baseline_path.glob("final_image_*.hv"))
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
    precond_combine = _normalise_combine(result.get('precond_combine', ''))
    if not precond_combine:
        precond_combine = _normalise_combine(result.get('combine', ''))
    result['precond_combine'] = precond_combine
    result['precond_label'] = (
        f"{precond_type}:{precond_combine}" if precond_combine else precond_type
    )
    setting_id, repeat_id = _parse_repeat_metadata(result_dir.name)
    result['run_dir'] = result_dir.name
    result['setting_id'] = setting_id
    result['repeat_id'] = repeat_id
    
    # Load objective history
    obj_file = result_dir / "objective.csv"
    result['objective_history'] = _read_objective_history(obj_file)
    
    # Find final image
    image_files = sorted(result_dir.glob("image_*.hv"))
    if not image_files:
        image_files = sorted(result_dir.glob("final_image_*.hv"))
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


def analyze_sweep(
    sweep_dir: Path,
    baseline_dir: Path,
    convergence_threshold: float = 0.01,
    allowed_alphas: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """
    Analyze all sweep results against baselines.
    
    Returns:
        DataFrame with comparative metrics for each test
    """
    results = []
    
    # Find all sweep result directories
    sweep_subdirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('precond_')]
    
    print(f"Found {len(sweep_subdirs)} sweep results")
    
    baseline_cache: Dict[float, Optional[Dict]] = {}

    for result_dir in sweep_subdirs:
        print(f"Processing {result_dir.name}...")
        
        # Load sweep result
        sweep_result = load_sweep_result(result_dir)
        if sweep_result is None:
            print(f"  Skipping {result_dir.name} - no result file")
            continue
        
        alpha = _coerce_float(sweep_result.get('alpha', None))
        if alpha is None:
            print(f"  Skipping {result_dir.name} - no alpha value")
            continue
        if allowed_alphas is not None and alpha not in set(allowed_alphas):
            continue

        precond_type = sweep_result.get('precond_type', 'unknown')
        precond_combine = _normalise_combine(sweep_result.get('precond_combine', ''))
        if not precond_combine:
            precond_combine = _normalise_combine(sweep_result.get('combine', ''))
        precond_label = (
            f"{precond_type}:{precond_combine}" if precond_combine else precond_type
        )
        setting_id = sweep_result.get('setting_id', result_dir.name)
        repeat_id = sweep_result.get('repeat_id', None)
        
        # Load corresponding baseline
        if alpha not in baseline_cache:
            baseline_cache[alpha] = load_baseline_data(baseline_dir, alpha)
        baseline = baseline_cache[alpha]
        if baseline is None:
            print(f"  Warning: No baseline for alpha={alpha}, skipping comparison")
            # Still record the result without comparison metrics
            results.append({
                'run_dir': result_dir.name,
                'setting_id': setting_id,
                'repeat_id': repeat_id,
                'precond_type': precond_type,
                'precond_combine': precond_combine,
                'precond_label': precond_label,
                'alpha': alpha,
                'step_size': sweep_result['step_size'],
                'final_objective': sweep_result['final_objective'],
                'run_time': sweep_result['run_time'],
                'status': sweep_result['status'],
                'baseline_available': False,
                'objective_length': (
                    len(sweep_result['objective_history'])
                    if sweep_result.get('objective_history') is not None else np.nan
                ),
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
            'run_dir': result_dir.name,
            'setting_id': setting_id,
            'repeat_id': repeat_id,
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
            'objective_length': (
                len(sweep_result['objective_history'])
                if sweep_result.get('objective_history') is not None else np.nan
            ),
        }
        
        results.append(result_summary)
    
    return pd.DataFrame(results)


def aggregate_repeat_statistics(
    df: pd.DataFrame,
    percentiles: Tuple[int, ...] = (10, 50, 90),
) -> pd.DataFrame:
    """
    Aggregate repeated runs per parameter setting using mean + percentile summaries.

    Grouping keys:
        setting_id, precond_type, precond_combine, precond_label, alpha, step_size
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "setting_id",
        "precond_type",
        "precond_combine",
        "precond_label",
        "alpha",
        "step_size",
    ]
    numeric_metrics = [
        "run_time",
        "speedup_vs_baseline",
        "iterations_to_convergence",
        "final_obj_gap",
        "image_rmse",
        "image_relative_error",
        "final_objective",
        "objective_length",
    ]

    rows: List[Dict] = []
    for key, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["num_runs"] = int(len(group))
        row["num_success"] = int(group["status"].eq("success").sum())
        row["success_rate"] = float(group["status"].eq("success").mean())
        row["num_converged"] = int(group["converged"].fillna(False).sum())
        row["converged_rate"] = float(group["converged"].fillna(False).mean())
        row["baseline_available"] = bool(group["baseline_available"].all())
        row["baseline_final_objective"] = _coerce_float(group["baseline_final_objective"].dropna().iloc[0]) if "baseline_final_objective" in group and not group["baseline_final_objective"].dropna().empty else np.nan
        row["baseline_run_time"] = _coerce_float(group["baseline_run_time"].dropna().iloc[0]) if "baseline_run_time" in group and not group["baseline_run_time"].dropna().empty else np.nan

        for metric in numeric_metrics:
            if metric not in group:
                continue
            series = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(series.mean()) if not series.dropna().empty else np.nan
            for p in percentiles:
                row[f"{metric}_p{p}"] = _safe_percentile(series, p)

        # Converged-only convergence speed summary
        converged_iters = pd.to_numeric(
            group.loc[group["converged"].fillna(False), "iterations_to_convergence"],
            errors="coerce",
        )
        row["iterations_to_convergence_converged_mean"] = (
            float(converged_iters.mean()) if not converged_iters.dropna().empty else np.nan
        )
        for p in percentiles:
            row[f"iterations_to_convergence_converged_p{p}"] = _safe_percentile(
                converged_iters, p
            )

        rows.append(row)

    agg_df = pd.DataFrame(rows)
    if not agg_df.empty:
        sort_cols = [c for c in ("alpha", "precond_label", "step_size", "setting_id") if c in agg_df]
        agg_df = agg_df.sort_values(sort_cols).reset_index(drop=True)
    return agg_df


def plot_convergence_curves(
    sweep_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    alpha_values: Optional[List[float]] = None,
    inner_band_percentiles: Tuple[int, int] = (10, 90),
):
    """
    Plot convergence curves comparing different preconditioners to baseline.

    For repeated runs of the same parameter setting, this plots:
    - mean objective trajectory (line),
    - min/max envelope across runs (shaded region),
    - inner percentile band (default P10-P90, shaded region).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by alpha
    sweep_subdirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('precond_')]
    
    # Organize by alpha
    alpha_results: Dict[float, Dict[str, List[Dict]]] = {}
    for result_dir in sweep_subdirs:
        sweep_result = load_sweep_result(result_dir)
        if sweep_result is None or sweep_result.get('objective_history') is None:
            continue
        
        alpha = _coerce_float(sweep_result.get('alpha'))
        if alpha is None:
            continue
        if alpha_values and alpha not in alpha_values:
            continue
        
        if alpha not in alpha_results:
            alpha_results[alpha] = {}
        
        setting_id = sweep_result.get("setting_id", result_dir.name)
        alpha_results[alpha].setdefault(setting_id, []).append(sweep_result)
    
    # Plot for each alpha
    baseline_cache: Dict[float, Optional[Dict]] = {}

    for alpha, settings in alpha_results.items():
        if alpha not in baseline_cache:
            baseline_cache[alpha] = load_baseline_data(baseline_dir, alpha)
        baseline = baseline_cache[alpha]
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
        
        # Plot each setting summary across repeats
        settings_items = sorted(
            settings.items(),
            key=lambda item: (
                str(item[1][0].get("precond_label", "")),
                float(item[1][0].get("step_size", np.nan)),
                item[0],
            ),
        )
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(settings_items))))
        for (setting_id, runs), color in zip(settings_items, colors):
            histories = [run.get("objective_history") for run in runs if run.get("objective_history") is not None]
            curve_stats = _build_curve_statistics(histories, inner_band=inner_band_percentiles)
            if curve_stats is None:
                continue

            precond_label = runs[0].get('precond_label', runs[0].get('precond_type', 'unknown'))
            step_size = runs[0].get("step_size", np.nan)
            n_runs = curve_stats["num_runs"]
            label = f"{precond_label}, step={step_size}, n={n_runs}"

            x = curve_stats["x"]
            # Envelope across all repeats
            ax.fill_between(
                x,
                curve_stats["min"],
                curve_stats["max"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            # Inner percentile band
            ax.fill_between(
                x,
                curve_stats["inner_low"],
                curve_stats["inner_high"],
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            # Mean trajectory
            ax.plot(x, curve_stats["mean"], label=label, alpha=0.95, color=color, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title(
            f"Convergence Comparison (α={alpha}) "
            f"[envelope=min-max, inner=P{inner_band_percentiles[0]}-P{inner_band_percentiles[1]}]"
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'convergence_alpha_{alpha}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved convergence plot for alpha={alpha}")


def generate_summary_report(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_file: Path,
    percentiles: Tuple[int, ...] = (10, 50, 90),
):
    """Generate a summary report of preconditioner performance."""
    p_low = percentiles[0]
    p_mid = percentiles[len(percentiles) // 2]
    p_high = percentiles[-1]

    with open(output_file, 'w') as f:
        f.write("# Preconditioner Sweep Analysis Report\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total tests: {len(df)}\n")
        f.write(f"- Successful tests: {df['status'].eq('success').sum()}\n")
        f.write(f"- Tests with baseline: {df['baseline_available'].sum()}\n")
        f.write(f"- Converged tests: {df['converged'].sum()}\n\n")
        f.write(f"- Unique parameter settings: {agg_df['setting_id'].nunique() if not agg_df.empty else 0}\n")
        f.write(f"- Repeat summary percentiles: P{p_low}, P{p_mid}, P{p_high}\n\n")
        
        # Best performers per alpha from repeat-aggregated settings
        f.write("## Best Performers by Alpha (Aggregated Over Repeats)\n\n")
        if agg_df.empty:
            f.write("No aggregated settings available.\n\n")
        else:
            for alpha in sorted(agg_df['alpha'].dropna().unique()):
                alpha_df = agg_df[agg_df['alpha'] == alpha]
                f.write(f"### Alpha = {alpha}\n\n")
                
                # Fastest (use converged-only percentile median where available)
                fastest_candidates = alpha_df.copy()
                fastest_candidates = fastest_candidates[
                    fastest_candidates['iterations_to_convergence_converged_p50'].notna()
                ]
                if len(fastest_candidates) > 0:
                    fastest = fastest_candidates.nsmallest(1, 'iterations_to_convergence_converged_p50').iloc[0]
                    f.write("**Fastest convergence (aggregated):**\n")
                    f.write(f"- Setting: {fastest.get('setting_id', 'unknown')}\n")
                    f.write(f"- Preconditioner: {fastest.get('precond_label', fastest.get('precond_type', 'unknown'))}\n")
                    f.write(f"- Step size: {fastest['step_size']}\n")
                    f.write(f"- Runs: {int(fastest['num_runs'])}\n")
                    f.write(
                        f"- Iterations (converged-only, P{p_low}/P{p_mid}/P{p_high}): "
                        f"{fastest.get(f'iterations_to_convergence_converged_p{p_low}', np.nan):.2f} / "
                        f"{fastest.get(f'iterations_to_convergence_converged_p{p_mid}', np.nan):.2f} / "
                        f"{fastest.get(f'iterations_to_convergence_converged_p{p_high}', np.nan):.2f}\n"
                    )
                    f.write(
                        f"- Runtime mean (s): {fastest.get('run_time_mean', np.nan):.2f}\n\n"
                    )
                
                # Most accurate by objective gap
                accurate_candidates = alpha_df[alpha_df['final_obj_gap_p50'].notna()]
                if len(accurate_candidates) > 0:
                    most_accurate = accurate_candidates.nsmallest(1, 'final_obj_gap_p50').iloc[0]
                    f.write("**Most accurate (aggregated):**\n")
                    f.write(f"- Setting: {most_accurate.get('setting_id', 'unknown')}\n")
                    f.write(f"- Preconditioner: {most_accurate.get('precond_label', most_accurate.get('precond_type', 'unknown'))}\n")
                    f.write(f"- Step size: {most_accurate['step_size']}\n")
                    f.write(f"- Runs: {int(most_accurate['num_runs'])}\n")
                    f.write(
                        f"- Final objective gap (P{p_low}/P{p_mid}/P{p_high}): "
                        f"{most_accurate.get(f'final_obj_gap_p{p_low}', np.nan):.6f} / "
                        f"{most_accurate.get(f'final_obj_gap_p{p_mid}', np.nan):.6f} / "
                        f"{most_accurate.get(f'final_obj_gap_p{p_high}', np.nan):.6f}\n"
                    )
                    f.write(
                        f"- RMSE mean: {most_accurate.get('image_rmse_mean', np.nan):.6g}\n\n"
                    )

        # Aggregated setting table
        f.write("## Aggregated Setting Statistics\n\n")
        if agg_df.empty:
            f.write("No aggregated setting statistics available.\n\n")
        else:
            summary_cols = [
                "alpha",
                "precond_label",
                "step_size",
                "num_runs",
                "success_rate",
                "converged_rate",
                "iterations_to_convergence_mean",
                f"iterations_to_convergence_p{p_low}",
                f"iterations_to_convergence_p{p_mid}",
                f"iterations_to_convergence_p{p_high}",
                "final_obj_gap_mean",
                f"final_obj_gap_p{p_low}",
                f"final_obj_gap_p{p_mid}",
                f"final_obj_gap_p{p_high}",
                "run_time_mean",
                f"run_time_p{p_low}",
                f"run_time_p{p_mid}",
                f"run_time_p{p_high}",
            ]
            summary_cols = [c for c in summary_cols if c in agg_df.columns]
            table = agg_df[summary_cols].sort_values(
                ["alpha", "precond_label", "step_size"]
            ).round(6)
            f.write(table.to_markdown(index=False))
            f.write("\n\n")
            
        # Per-run preconditioner comparison with percentiles
        f.write("## Preconditioner Comparison (Per-Run Distribution)\n\n")
        if df.empty:
            f.write("No per-run data available.\n\n")
        else:
            grouped_rows: List[Dict] = []
            for label, group in df.groupby("precond_label", dropna=False):
                row = {
                    "precond_label": label,
                    "runs": int(len(group)),
                    "converged_rate": float(group["converged"].fillna(False).mean()),
                }
                for metric in ("iterations_to_convergence", "run_time", "final_obj_gap"):
                    vals = pd.to_numeric(group[metric], errors="coerce")
                    row[f"{metric}_mean"] = float(vals.mean()) if not vals.dropna().empty else np.nan
                    row[f"{metric}_p{p_low}"] = _safe_percentile(vals, p_low)
                    row[f"{metric}_p{p_mid}"] = _safe_percentile(vals, p_mid)
                    row[f"{metric}_p{p_high}"] = _safe_percentile(vals, p_high)
                grouped_rows.append(row)
            precond_summary = pd.DataFrame(grouped_rows).sort_values("precond_label").round(6)
            f.write(precond_summary.to_markdown(index=False))
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
    parser.add_argument(
        "--summary-percentiles",
        type=str,
        default="10,50,90",
        help="Comma-separated percentiles for quantitative summaries (default: 10,50,90)",
    )
    parser.add_argument(
        "--curve-inner-band",
        type=str,
        default="10,90",
        help="Comma-separated low,high percentiles for convergence-curve inner band (default: 10,90)",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default=None,
        help="Comma-separated alpha values to include (e.g. '0.01'). "
             "If omitted, uses parameters/alphas.csv when available.",
    )
    
    args = parser.parse_args()
    summary_percentiles = _parse_percentiles(args.summary_percentiles, default=(10, 50, 90))
    curve_band = _parse_percentiles(args.curve_inner_band, default=(10, 90))
    if len(curve_band) != 2:
        raise ValueError("--curve-inner-band must contain exactly two percentiles, e.g. 10,90.")
    curve_band_tuple = (curve_band[0], curve_band[1])
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent / "output"
    sweep_dir = base_dir / args.sweep
    baseline_dir = base_dir / args.baseline
    study_dir = Path(__file__).parent.parent
    default_alpha_csv = study_dir / "parameters" / "alphas.csv"

    allowed_alphas = _parse_alpha_values(args.alpha_values)
    if allowed_alphas is None:
        allowed_alphas = _load_allowed_alphas_from_csv(default_alpha_csv)
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_dir / f"{args.sweep}_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sweep directory: {sweep_dir}")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Summary percentiles: {summary_percentiles}")
    print(f"Curve inner band: P{curve_band_tuple[0]}-P{curve_band_tuple[1]}")
    if allowed_alphas is not None:
        print(f"Alpha filter: {allowed_alphas}")
    else:
        print("Alpha filter: <none>")
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
        df = analyze_sweep(
            sweep_dir,
            baseline_dir,
            args.convergence_threshold,
            allowed_alphas=allowed_alphas,
        )
        agg_df = aggregate_repeat_statistics(df, percentiles=summary_percentiles)
        
        # Save results
        results_file = output_dir / "analysis_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Saved analysis results to {results_file}")
        agg_results_file = output_dir / "analysis_results_aggregated.csv"
        agg_df.to_csv(agg_results_file, index=False)
        print(f"Saved aggregated analysis results to {agg_results_file}")
        
        # Generate plots
        plot_convergence_curves(
            sweep_dir,
            baseline_dir,
            output_dir,
            alpha_values=list(allowed_alphas) if allowed_alphas is not None else None,
            inner_band_percentiles=curve_band_tuple,
        )
        
        # Generate summary report
        report_file = output_dir / "summary_report.md"
        generate_summary_report(
            df,
            agg_df,
            report_file,
            percentiles=summary_percentiles,
        )
        
        print()
        print("=" * 60)
        print("Analysis complete!")
        print(f"Results: {results_file}")
        print(f"Aggregated results: {agg_results_file}")
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
