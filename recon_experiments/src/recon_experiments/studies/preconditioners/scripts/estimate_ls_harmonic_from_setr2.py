#!/usr/bin/env python3
"""
Estimate LS harmonic/majoriser and Lehmer preconditioner performance from SETR2 runs.

This is intentionally lightweight. It reads:
  - SETR2 preconditioner sweep result/objective CSVs,
  - the local preconditioner-study baseline convergence summary, and
  - optional PET final images when grids match.

The main metric is progress against the baseline objective decrease:

    progress = (J0 - J_run) / (J0 - J_star)

where J0 and J_star are taken from baseline_convergence_metrics.csv. This is more
informative than |J - J_star| / |J_star| here, because all runs begin inside a
1 percent relative-to-objective gap.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - image metrics are optional
    np = None


STUDY_OUTPUT = Path(
    "recon_experiments/src/recon_experiments/studies/preconditioners/output"
)
DEFAULT_SETR2_OUTPUT = Path(
    "/home/sam/mnt/comic-sporter/synergistic_Y90/SETR2/"
    "recon_experiments/src/recon_experiments/studies/preconditioners/output"
)
DEFAULT_BASELINE_SUMMARY = STUDY_OUTPUT / "baseline_convergence_analysis" / "baseline_convergence_metrics.csv"
DEFAULT_BASELINE_ROOT = STUDY_OUTPUT
DEFAULT_OUTPUT_DIR = STUDY_OUTPUT / "ls_lehmer_vs_harmonic_estimate"

DEFAULT_GROUPS = ("precond_1bpos", "precond_2bpos")
DEFAULT_METHODS = ("ls_block_diag", "ls_block_gershgorin")
DEFAULT_COMBINES = ("majoriser", "lehmer")
DEFAULT_TARGETS = (0.90, 0.99, 0.999, 0.9999)


@dataclass(frozen=True)
class BaselineMetrics:
    bpos: int
    label: str
    j0: float
    j_star: float
    total_decrease: float
    updates_per_epoch: float


def _coerce_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def _read_first_csv_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            return {k: v for k, v in row.items() if k is not None}
    return {}


def _read_objective_history(path: Path) -> List[float]:
    if not path.exists():
        return []
    values: List[float] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = _coerce_float(row[-1])
            if value is not None:
                values.append(value)

    # Objective files in this study commonly start with a sentinel "0" line.
    if len(values) > 1 and values[0] <= 0 < values[1]:
        values = values[1:]
    return values


def _normalise_combine(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan"}:
        return ""
    if text in {"harmonic", "majorise", "majorize"}:
        return "majoriser"
    return text


def _parse_bpos(text: str) -> Optional[int]:
    match = re.search(r"(\d+)bpos", text)
    if match:
        return int(match.group(1))
    return None


def _parse_baseline_label(label: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*bed", label)
    if match:
        return int(match.group(1))
    return None


def _parse_repeat(name: str) -> Optional[int]:
    match = re.search(r"_rep_(\d+)$", name)
    if match:
        return int(match.group(1))
    return None


def _updates_per_epoch(row: Dict[str, str], fallback: float) -> float:
    raw = row.get("num_subsets", "")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and parsed:
            vals = [_coerce_float(v) for v in parsed]
            if all(v is not None for v in vals):
                return float(sum(v for v in vals if v is not None))
        value = _coerce_float(raw)
        if value is not None:
            return value
    return fallback


def _load_baseline_metrics(path: Path) -> Dict[int, BaselineMetrics]:
    baselines: Dict[int, BaselineMetrics] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("baseline", "")
            bpos = _parse_baseline_label(label)
            j0 = _coerce_float(row.get("J0"))
            j_star = _coerce_float(row.get("J_star"))
            total = _coerce_float(row.get("total_decrease"))
            updates = _coerce_float(row.get("updates_per_epoch_L"))
            if bpos is None or j0 is None or j_star is None:
                continue
            if total is None:
                total = j0 - j_star
            if updates is None:
                updates = 1.0
            baselines[bpos] = BaselineMetrics(
                bpos=bpos,
                label=label,
                j0=j0,
                j_star=j_star,
                total_decrease=total,
                updates_per_epoch=updates,
            )
    return baselines


def _read_interfile_float_image(header_path: Path) -> Optional["np.ndarray"]:
    if np is None or not header_path.exists():
        return None
    text = header_path.read_text(errors="replace")

    def find(pattern: str) -> Optional[str]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    data_name = find(r"name of data file\s*:=\s*(.+)")
    number_format = (find(r"!number format\s*:=\s*(.+)") or "").lower()
    bytes_per_pixel = find(r"!number of bytes per pixel\s*:=\s*(\d+)")
    byte_order = (find(r"imagedata byte order\s*:=\s*(.+)") or "littleendian").lower()
    sizes = re.findall(r"!matrix size \[(\d+)\]\s*:=\s*(\d+)", text, flags=re.IGNORECASE)

    if not data_name or "float" not in number_format or bytes_per_pixel != "4":
        return None
    if not sizes:
        return None

    sizes_sorted = [int(size) for _, size in sorted((int(axis), size) for axis, size in sizes)]
    data_path = header_path.parent / data_name
    if not data_path.exists():
        return None

    dtype = ">f4" if "big" in byte_order else "<f4"
    arr = np.fromfile(data_path, dtype=dtype)
    expected = 1
    for size in sizes_sorted:
        expected *= size
    if arr.size != expected:
        return None
    return arr.astype(np.float64, copy=False)


def _pet_image_metrics(run_dir: Path, baseline_root: Path, bpos: int, baseline_alpha_dir: str) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "pet_rmse": None,
        "pet_nrmse": None,
        "pet_mae": None,
    }
    if np is None:
        return metrics

    baseline_header = (
        baseline_root
        / f"baselines_{bpos}bpos"
        / f"baseline_alpha_{baseline_alpha_dir}"
        / "final_image_0.hv"
    )
    run_header = run_dir / "final_image_0.hv"
    base = _read_interfile_float_image(baseline_header)
    test = _read_interfile_float_image(run_header)
    if base is None or test is None or base.shape != test.shape:
        return metrics

    diff = test - base
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sqrt(np.mean(base * base)))
    metrics["pet_rmse"] = rmse
    metrics["pet_mae"] = mae
    metrics["pet_nrmse"] = rmse / denom if denom > 0 else None
    return metrics


def _run_paths(sweep_root: Path, groups: Sequence[str]) -> Iterable[Path]:
    for group in groups:
        group_dir = sweep_root / group
        if not group_dir.exists():
            continue
        for path in sorted(group_dir.iterdir()):
            if path.is_dir() and path.name.startswith("precond_"):
                yield path


def _first_threshold_epoch(
    objectives: Sequence[float],
    baseline: BaselineMetrics,
    updates_per_epoch: float,
    target: float,
) -> Optional[float]:
    if baseline.total_decrease <= 0 or updates_per_epoch <= 0:
        return None
    for idx, obj in enumerate(objectives):
        progress = (baseline.j0 - obj) / baseline.total_decrease
        if progress >= target:
            return idx / updates_per_epoch
    return None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _std(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((v - mean) ** 2 for v in clean) / (len(clean) - 1))


def analyse_runs(args: argparse.Namespace) -> List[Dict[str, object]]:
    baselines = _load_baseline_metrics(args.baseline_summary)
    rows: List[Dict[str, object]] = []
    allowed_methods = set(args.methods)
    allowed_combines = {_normalise_combine(c) for c in args.combines}

    for run_dir in _run_paths(args.sweep_root, args.groups):
        result = _read_first_csv_row(run_dir / "result.csv")
        arg_row = _read_first_csv_row(run_dir / "args.csv")
        info = dict(arg_row)
        info.update(result)

        method = str(info.get("precond_type", "")).strip()
        combine = _normalise_combine(info.get("precond_combine", "")) or _normalise_combine(
            info.get("combine", "")
        )
        if method not in allowed_methods:
            continue
        if combine and combine not in allowed_combines:
            continue

        bpos = _parse_bpos(str(info.get("bpos", ""))) or _parse_bpos(run_dir.parent.name)
        if bpos is None or bpos not in baselines:
            continue
        baseline = baselines[bpos]
        updates = _updates_per_epoch(info, baseline.updates_per_epoch)

        objectives = _read_objective_history(run_dir / "objective.csv")
        final_objective = _coerce_float(info.get("final_objective"))
        if final_objective is None and objectives:
            final_objective = objectives[-1]

        base_row: Dict[str, object] = {
            "bpos": bpos,
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "precond_type": method,
            "precond_combine": combine or "majoriser",
            "lehmer_p": _coerce_float(info.get("lehmer_p")),
            "lehmer_scale": _coerce_float(info.get("lehmer_scale")),
            "block_scalar_reduction": info.get("block_scalar_reduction", ""),
            "repeat": _parse_repeat(run_dir.name),
            "step_size": _coerce_float(info.get("step_size") or info.get("initial_step_size")),
            "alpha_reported": _coerce_float(info.get("alpha")),
            "alpha_initial": _coerce_float(info.get("alpha_initial")),
            "alpha_scaled": _coerce_float(info.get("alpha_scaled")),
            "num_epochs": _coerce_float(info.get("num_epochs")),
            "updates_per_epoch": updates,
            "num_objective_values": len(objectives),
            "status": info.get("status", ""),
            "error": info.get("error", ""),
            "run_time_s": _coerce_float(info.get("run_time")),
            "baseline_j0": baseline.j0,
            "baseline_j_star": baseline.j_star,
            "baseline_total_decrease": baseline.total_decrease,
        }
        if final_objective is None:
            base_row.update(
                {
                    "final_objective": None,
                    "final_abs_gap": None,
                    "final_rel_obj_gap": None,
                    "final_remaining_frac_of_baseline_decrease": None,
                    "final_progress_frac": None,
                }
            )
            for target in args.targets:
                key = f"epoch_to_{target * 100:.2f}pct_baseline_decrease"
                base_row[key] = None
            base_row.update({"pet_rmse": None, "pet_nrmse": None, "pet_mae": None})
            rows.append(base_row)
            continue

        progress = (baseline.j0 - final_objective) / baseline.total_decrease
        remaining = (final_objective - baseline.j_star) / baseline.total_decrease
        abs_gap = final_objective - baseline.j_star
        rel_obj_gap = abs(abs_gap) / abs(baseline.j_star) if baseline.j_star else None

        row: Dict[str, object] = {
            **base_row,
            "final_objective": final_objective,
            "final_abs_gap": abs_gap,
            "final_rel_obj_gap": rel_obj_gap,
            "final_remaining_frac_of_baseline_decrease": remaining,
            "final_progress_frac": progress,
        }
        for target in args.targets:
            key = f"epoch_to_{target * 100:.2f}pct_baseline_decrease"
            row[key] = _first_threshold_epoch(objectives, baseline, updates, target)

        if args.with_image_metrics:
            row.update(_pet_image_metrics(run_dir, args.baseline_root, bpos, args.baseline_alpha_dir))
        else:
            row.update({"pet_rmse": None, "pet_nrmse": None, "pet_mae": None})
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarise(rows: Sequence[Dict[str, object]], targets: Sequence[float]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["bpos"], row["precond_type"], row["precond_combine"])].append(row)

    summaries: List[Dict[str, object]] = []
    for (bpos, method, combine), group_rows in sorted(groups.items()):
        metric_rows = [r for r in group_rows if r.get("final_objective") is not None]
        summary: Dict[str, object] = {
            "bpos": bpos,
            "precond_type": method,
            "precond_combine": combine,
            "lehmer_p_mean": _mean([r.get("lehmer_p") for r in metric_rows]),
            "lehmer_scale_mean": _mean([r.get("lehmer_scale") for r in metric_rows]),
            "n_runs": len(group_rows),
            "n_success": sum(str(r.get("status", "")).lower() == "success" for r in group_rows),
            "n_with_objective": len(metric_rows),
            "final_objective_mean": _mean([r.get("final_objective") for r in metric_rows]),
            "final_objective_std": _std([r.get("final_objective") for r in metric_rows]),
            "final_abs_gap_mean": _mean([r.get("final_abs_gap") for r in metric_rows]),
            "final_remaining_frac_mean": _mean(
                [r.get("final_remaining_frac_of_baseline_decrease") for r in metric_rows]
            ),
            "final_progress_frac_mean": _mean([r.get("final_progress_frac") for r in metric_rows]),
            "run_time_min_mean": _mean(
                [
                    (r.get("run_time_s") / 60.0 if isinstance(r.get("run_time_s"), (int, float)) else None)
                    for r in metric_rows
                ]
            ),
            "pet_nrmse_mean": _mean([r.get("pet_nrmse") for r in metric_rows]),
            "pet_rmse_mean": _mean([r.get("pet_rmse") for r in metric_rows]),
        }
        for target in targets:
            key = f"epoch_to_{target * 100:.2f}pct_baseline_decrease"
            vals = [r.get(key) for r in metric_rows]
            summary[f"{key}_mean"] = _mean(vals)
            summary[f"{key}_not_reached"] = sum(v is None for v in vals)
        summaries.append(summary)
    return summaries


def _ratio(num: object, den: object) -> Optional[float]:
    if not isinstance(num, (int, float)) or not isinstance(den, (int, float)):
        return None
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return None
    return float(num) / float(den)


def _diff(a: object, b: object) -> Optional[float]:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return float(a) - float(b)


def _compare_lehmer_to_majoriser(summaries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_setting = {
        (row.get("bpos"), row.get("precond_type"), row.get("precond_combine")): row
        for row in summaries
    }
    comparisons: List[Dict[str, object]] = []
    settings = sorted({(row.get("bpos"), row.get("precond_type")) for row in summaries})
    for bpos, method in settings:
        majoriser = by_setting.get((bpos, method, "majoriser"))
        lehmer = by_setting.get((bpos, method, "lehmer"))
        if majoriser is None or lehmer is None:
            continue
        comparisons.append(
            {
                "bpos": bpos,
                "precond_type": method,
                "lehmer_p_mean": lehmer.get("lehmer_p_mean"),
                "lehmer_scale_mean": lehmer.get("lehmer_scale_mean"),
                "majoriser_n_success": majoriser.get("n_success"),
                "lehmer_n_success": lehmer.get("n_success"),
                "final_progress_frac_delta": _diff(
                    lehmer.get("final_progress_frac_mean"),
                    majoriser.get("final_progress_frac_mean"),
                ),
                "final_remaining_frac_ratio": _ratio(
                    lehmer.get("final_remaining_frac_mean"),
                    majoriser.get("final_remaining_frac_mean"),
                ),
                "epoch_to_99pct_ratio": _ratio(
                    lehmer.get("epoch_to_99.00pct_baseline_decrease_mean"),
                    majoriser.get("epoch_to_99.00pct_baseline_decrease_mean"),
                ),
                "run_time_min_ratio": _ratio(
                    lehmer.get("run_time_min_mean"),
                    majoriser.get("run_time_min_mean"),
                ),
            }
        )
    return comparisons


def _markdown_table(rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        rendered = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                rendered.append(_format_float(value))
            elif value is None:
                rendered.append("")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    rows: Sequence[Dict[str, object]],
    summaries: Sequence[Dict[str, object]],
    comparisons: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    baseline_rows = []
    for baseline in _load_baseline_metrics(args.baseline_summary).values():
        baseline_rows.append(
            {
                "bpos": baseline.bpos,
                "J0": baseline.j0,
                "J_star": baseline.j_star,
                "total_decrease": baseline.total_decrease,
                "updates_per_epoch": baseline.updates_per_epoch,
            }
        )

    lines = [
        "# LS Lehmer vs Harmonic/Majoriser Estimate",
        "",
        f"SETR2 sweep root: `{args.sweep_root}`",
        f"Baseline summary: `{args.baseline_summary}`",
        "",
        "The study scripts canonicalise `harmonic` to `majoriser`; those rows are inverse-sum/majoriser LS runs.",
        "Rows with `precond_combine=lehmer` use the saved Lehmer order reported in `result.csv`.",
        "The convergence metric is progress through the baseline objective decrease, not raw relative objective gap.",
        "",
        "## Baselines",
        "",
        _markdown_table(baseline_rows, ["bpos", "J0", "J_star", "total_decrease", "updates_per_epoch"]),
        "",
        "## Summary",
        "",
        _markdown_table(
            summaries,
            [
                "bpos",
                "precond_type",
                "precond_combine",
                "lehmer_p_mean",
                "lehmer_scale_mean",
                "n_runs",
                "n_success",
                "n_with_objective",
                "final_abs_gap_mean",
                "final_remaining_frac_mean",
                "final_progress_frac_mean",
                "epoch_to_99.00pct_baseline_decrease_mean",
                "epoch_to_99.90pct_baseline_decrease_mean",
                "epoch_to_99.90pct_baseline_decrease_not_reached",
                "run_time_min_mean",
                "pet_nrmse_mean",
            ],
        ),
        "",
        "## Lehmer - Majoriser Comparison",
        "",
        (
            _markdown_table(
                comparisons,
                [
                    "bpos",
                    "precond_type",
                    "lehmer_p_mean",
                    "lehmer_scale_mean",
                    "majoriser_n_success",
                    "lehmer_n_success",
                    "final_progress_frac_delta",
                    "final_remaining_frac_ratio",
                    "epoch_to_99pct_ratio",
                    "run_time_min_ratio",
                ],
            )
            if comparisons
            else "No pairwise comparison yet: matching `precond_combine=lehmer` rows were not found."
        ),
        "",
        "## Caveats",
        "",
        "- This is only a direct Lehmer-vs-harmonic A/B test once matching `precond_combine=lehmer` runs exist beside the harmonic/majoriser rows.",
        "- For block LS runs, Lehmer with `p>0` requires scalarising the data preconditioner (`block_scalar_reduction=mean` or `geometric`); `diag` is only implemented for `p=0`.",
        "- The 1 percent relative-to-objective criterion is not used because the initial iterate already satisfies it.",
        "- `n_runs` counts attempted run directories; objective statistics use only runs with finite objective traces.",
        "- PET final-image RMSE/NRMSE is reported only when `--with-image-metrics` is used and Interfile grids match; SPECT final images are not compared here because saved baseline/run grids differ in the available outputs.",
        "",
        f"Per-run CSV: `{args.output_dir / 'per_run.csv'}`",
        f"Summary CSV: `{args.output_dir / 'summary.csv'}`",
        f"Comparison CSV: `{args.output_dir / 'comparison.csv'}`",
        "",
        f"Analysed runs: {len(rows)}",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SETR2_OUTPUT)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--baseline-alpha-dir", default="0.01")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--combines", nargs="+", default=list(DEFAULT_COMBINES))
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument(
        "--with-image-metrics",
        action="store_true",
        help="Also compute PET final-image RMSE/NRMSE where grids match. This is slower on mounted outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyse_runs(args)
    summaries = _summarise(rows, args.targets)
    comparisons = _compare_lehmer_to_majoriser(summaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_run.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summaries)
    _write_csv(args.output_dir / "comparison.csv", comparisons)
    _write_report(args.output_dir / "report.md", rows, summaries, comparisons, args)

    print(f"Analysed {len(rows)} runs")
    print(f"Wrote {args.output_dir / 'per_run.csv'}")
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    if summaries:
        print()
        print(_markdown_table(
            summaries,
            [
                "bpos",
                "precond_type",
                "precond_combine",
                "lehmer_p_mean",
                "lehmer_scale_mean",
                "n_runs",
                "n_success",
                "n_with_objective",
                "final_abs_gap_mean",
                "final_progress_frac_mean",
                "epoch_to_99.00pct_baseline_decrease_mean",
                "epoch_to_99.90pct_baseline_decrease_mean",
                "epoch_to_99.90pct_baseline_decrease_not_reached",
                "run_time_min_mean",
                "pet_nrmse_mean",
            ],
        ))


if __name__ == "__main__":
    main()
