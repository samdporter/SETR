#!/usr/bin/env python3
"""
Check that sweep runs and their baseline share the SAME objective-defining
hyperparameters -- i.e. the parameters that define the objective function (and
therefore its unique minimiser), as opposed to optimisation knobs that only
affect the path taken to that minimiser.

For a (strictly convex, uniquely-minimised) TV/TNV objective, the solution is
determined by:
  * the data-fidelity weighting (effective alpha/beta on each modality),
  * the regularisation (prior type, gamma, delta/Huber, kappa, stencil, bnd
    conditions, directional flags, ...),
  * the forward/acquisition model (resolution modelling, TOF, scatter,
    zoom-registration) and the data itself (data paths),
  * the feasible set (support-mask construction).

It is NOT determined by: number of subsets, number of epochs, step size,
relaxation eta, preconditioner type/combine, seed, snapshot cadence, initial
image (convex => init-independent), or any output/bookkeeping field.

Usage:
  check_objective_consistency.py --sweep <sweep_dir> --baseline <baseline_dir>
      [--alpha 1.0] [--output <report.csv>]

Exit status is 0 if every processed run matches its baseline on the
objective-defining set, 1 otherwise. A per-run CSV report and a list of failing
runs (to be hard-gated out of the convergence analysis) are written next to the
sweep directory's analysis output.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Objective-defining parameters (hard gate) --------------------------------
# Fields whose value uniquely determines the objective/minimiser.
OBJECTIVE_DEFINING_FIELDS: Tuple[str, ...] = (
    # Regularisation (VTV/TNV)
    "no_prior",
    "use_tnv_prior",
    "gamma_tnv",
    "use_log_tnv",
    "tnv_stencil",
    "tnv_bnd_cond",
    "directional_tnv",
    "tnv_both_directions",
    "tail_singular_values",
    "stable",
    "smoothing",
    "delta",
    "use_kappa",
    # Acquisition / forward model
    "pet_gauss_fwhm",
    "spect_gauss_fwhm",
    "spect_res",
    "use_scatter",
    "use_tof",
    "use_zoom_registration",
    # Feasible set / support (the sensitivity-based mask is always active here)
    "support_mask_from_sensitivity",
    "support_mask_rel_threshold",
    "support_mask_abs_threshold",
    "support_mask_from_spect_attenuation",
    # Data
    "pet_data_path",
    "spect_data_path",
)

# Only relevant when modality-specific priors are enabled; gated conditionally.
MODALITY_SPECIFIC_PRIOR_FIELDS: Tuple[str, ...] = (
    "prior",
    "tv_stencil",
    "tv_bnd_cond",
    "directional_tv",
    "tv_both_directions",
)

# Only relevant when the SPECT-attenuation support mask is enabled; gated
# conditionally (the thresholds are inert while the toggle is off).
SPECT_ATTN_SUPPORT_FIELDS: Tuple[str, ...] = (
    "support_mask_spect_attn_rel_threshold",
    "support_mask_spect_attn_abs_threshold",
)

# Known feature-OFF defaults. If a run's args predate a column (key absent) and
# the baseline holds the OFF default, the effective objective is unchanged, so
# the absence is treated as a match (older-schema logging artifact).
DEFAULT_WHEN_ABSENT: Dict[str, str] = {
    "support_mask_from_spect_attenuation": "False",
}

# Fields reported for information but NOT gated (optimisation / bookkeeping).
INFORMATIONAL_FIELDS: Tuple[str, ...] = (
    "num_subsets",
    "num_epochs",
    "initial_step_size",
    "relaxation_eta",
    "precond_type",
    "precond_combine",
    "variance_reduction",
    "seed",
    "flip",
    "pet_initial_image_path",
    "spect_initial_image_path",
)

FLOAT_TOL = 1e-6


def _read_first_csv_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return None


def _norm_str(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    # Normalise whitespace inside list-like values e.g. "[5.61,  4.83]".
    s = re.sub(r"\s+", " ", s)
    return s


def _try_float(v: object) -> Optional[float]:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _values_match(a: object, b: object) -> bool:
    fa, fb = _try_float(a), _try_float(b)
    if fa is not None and fb is not None:
        denom = max(1.0, abs(fa), abs(fb))
        return abs(fa - fb) <= FLOAT_TOL * denom
    return _norm_str(a).lower() == _norm_str(b).lower()


def _effective_weight(row: Dict[str, str], base_key: str) -> Optional[float]:
    """Return the effective (scaled) objective weight for alpha/beta."""
    scaled = _try_float(row.get(f"{base_key}_scaled"))
    if scaled is not None:
        return scaled
    return _try_float(row.get(base_key))


def _truthy(v: object) -> bool:
    return _norm_str(v).lower() in ("true", "1", "yes")


def _gate_fields(baseline: Dict[str, str], run: Dict[str, str]) -> Tuple[str, ...]:
    fields = list(OBJECTIVE_DEFINING_FIELDS)
    # Always gate the flag itself; add the dependent fields only when relevant.
    fields.append("use_modality_specific_priors")
    if _truthy(baseline.get("use_modality_specific_priors")) or _truthy(
        run.get("use_modality_specific_priors")
    ):
        fields.extend(MODALITY_SPECIFIC_PRIOR_FIELDS)
    if _truthy(baseline.get("support_mask_from_spect_attenuation")) or _truthy(
        run.get("support_mask_from_spect_attenuation")
    ):
        fields.extend(SPECT_ATTN_SUPPORT_FIELDS)
    return tuple(fields)


def compare_run(
    baseline: Dict[str, str], run: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """Return (gated_mismatches, informational_diffs)."""
    gated: List[str] = []
    info: List[str] = []

    # Effective data-fidelity weights (handle *_scaled vs raw).
    for w in ("alpha", "beta"):
        b = _effective_weight(baseline, w)
        r = _effective_weight(run, w)
        if not _values_match(b, r):
            gated.append(f"{w}(effective): {b} -> {r}")

    for field in _gate_fields(baseline, run):
        b_present, r_present = field in baseline, field in run
        if not b_present and not r_present:
            continue
        b, r = baseline.get(field), run.get(field)
        # Older-schema runs may omit a column entirely. If the run lacks it and
        # the baseline holds the known feature-OFF default, the objective is
        # unchanged -> treat as a (noted) match rather than a mismatch.
        if (not r_present or _norm_str(r) == "") and field in DEFAULT_WHEN_ABSENT:
            if _values_match(b, DEFAULT_WHEN_ABSENT[field]):
                info.append(f"{field}: {_norm_str(b)} -> <absent; default {DEFAULT_WHEN_ABSENT[field]}>")
                continue
        if not _values_match(b, r):
            gated.append(f"{field}: {_norm_str(b)} -> {_norm_str(r)}")

    for field in INFORMATIONAL_FIELDS:
        b, r = baseline.get(field), run.get(field)
        if (b is not None or r is not None) and not _values_match(b, r):
            info.append(f"{field}: {_norm_str(b)} -> {_norm_str(r)}")

    return gated, info


def _gather_run_dirs(sweep_dir: Path) -> List[Path]:
    return sorted(
        d
        for d in sweep_dir.iterdir()
        if d.is_dir() and (d.name.startswith("precond_") or d.name.startswith("subset_"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path, help="Sweep directory (absolute or under output/).")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline directory (absolute or under output/).")
    parser.add_argument("--alpha", default="1.0", help="Baseline alpha subdir tag (default: 1.0).")
    parser.add_argument("--output", type=Path, default=None, help="Report CSV path (default: <sweep>_analysis/objective_consistency_report.csv).")
    args = parser.parse_args()

    base_output = Path(__file__).resolve().parent.parent / "output"
    sweep_dir = args.sweep if args.sweep.is_absolute() else base_output / args.sweep
    baseline_dir = args.baseline if args.baseline.is_absolute() else base_output / args.baseline
    baseline_path = baseline_dir / f"baseline_alpha_{args.alpha}"

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")
    baseline_row = _read_first_csv_row(baseline_path / "args.csv")
    if baseline_row is None:
        raise FileNotFoundError(f"Baseline args.csv not found: {baseline_path / 'args.csv'}")

    out_csv = args.output or (base_output / f"{sweep_dir.name}_analysis" / "objective_consistency_report.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    gate_txt = out_csv.parent / "objective_consistency_failing_runs.txt"

    run_dirs = _gather_run_dirs(sweep_dir)
    report_rows: List[Dict[str, object]] = []
    failing: List[str] = []

    print("=" * 78)
    print(f"Objective-consistency check")
    print(f"  sweep    : {sweep_dir}")
    print(f"  baseline : {baseline_path}")
    print(f"  runs     : {len(run_dirs)}")
    print("=" * 78)

    for run_dir in run_dirs:
        run_row = _read_first_csv_row(run_dir / "args.csv")
        if run_row is None:
            print(f"[SKIP] {run_dir.name}: no args.csv")
            report_rows.append({"run_dir": run_dir.name, "status": "no_args", "n_gated_mismatch": "", "gated_mismatches": "", "informational_diffs": ""})
            continue

        gated, info = compare_run(baseline_row, run_row)
        passed = len(gated) == 0
        if not passed:
            failing.append(run_dir.name)
        report_rows.append(
            {
                "run_dir": run_dir.name,
                "status": "PASS" if passed else "FAIL",
                "n_gated_mismatch": len(gated),
                "gated_mismatches": " | ".join(gated),
                "informational_diffs": " | ".join(info),
            }
        )
        tag = "PASS" if passed else "FAIL"
        extra = "" if passed else f"  <-- {'; '.join(gated)}"
        print(f"[{tag}] {run_dir.name}{extra}")

    # Write report CSV.
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_dir", "status", "n_gated_mismatch", "gated_mismatches", "informational_diffs"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    with gate_txt.open("w", encoding="utf-8") as f:
        for name in failing:
            f.write(name + "\n")

    n_pass = sum(1 for r in report_rows if r["status"] == "PASS")
    n_fail = len(failing)
    print("-" * 78)
    # Summarise the distinct informational-diff signatures (expected: step size,
    # relaxation eta, num_epochs, precond_type -- all optimisation-only).
    info_signatures = sorted({str(r["informational_diffs"]) for r in report_rows if r.get("informational_diffs")})
    print("Informational (non-gated) diff signatures across runs:")
    for sig in info_signatures:
        print(f"  * {sig}")
    print("-" * 78)
    print(f"PASS: {n_pass}   FAIL: {n_fail}   (total {len(report_rows)})")
    print(f"Report : {out_csv}")
    print(f"Failing runs list ({n_fail}): {gate_txt}")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
