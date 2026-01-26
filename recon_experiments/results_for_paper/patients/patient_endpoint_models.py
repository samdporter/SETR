"""Mixed-effects models for patient study endpoints (TBR and background CoV).

When use_log=True, ratio_est and its CI are on the ratio scale; interpret
evidence by whether the CI excludes 1.0 rather than only by p-values.
Holm adjustment is applied across the planned contrasts within each endpoint.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


def prepare_background_cov_df(
    long_df: pd.DataFrame, metric: str = "background_cov_pct"
) -> pd.DataFrame:
    """
    Deduplicate to one row per patient x reconstruction_type for background CoV.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-format dataframe with columns: patient, reconstruction_type, metric.
    metric : str
        Column name for background CoV percentage.

    Returns
    -------
    pandas.DataFrame
        Patient-level dataframe with mean background CoV per patient x method.
    """
    _validate_columns(long_df, ["patient", "reconstruction_type", metric])
    return (
        long_df.groupby(["patient", "reconstruction_type"], as_index=False)
        .agg(**{metric: (metric, "mean")})
        .reset_index(drop=True)
    )


def fit_tbr_lmm(
    long_df: pd.DataFrame,
    metric: str = "tbr_mean",
    baseline_recon: str = "dtnv",
    use_log: bool = True,
    use_bootstrap: bool = True,
    n_boot: int = 1000,
    seed: int | None = 123,
    expected_methods: Iterable[str] | None = None,
    on_incomplete: str = "filter",
    contrast_direction: str = "alt_vs_baseline",
    allow_empty_bootstrap: bool = False,
) -> Tuple[object, pd.DataFrame]:
    """
    Fit lesion-level mixed-effects model for TBR with patient and lesion nesting.

    Model: log(TBR) ~ reconstruction_type + (1|patient) + (1|patient:lesion)

    Inference uses patient-cluster bootstrap by default. If disabled, Wald z-tests
    with Holm adjustment are used. For final analyses, use n_boot >= 5000.

    Output columns (use_log=True):
    - log_ratio_est: log difference vs baseline (dTNV)
    - ratio_est, ci_low, ci_high: ratio scale (exp of log estimates)
    - p_raw, p_holm: bootstrap or Wald p-values with Holm adjustment

    Output columns (use_log=False):
    - diff_est, ci_low, ci_high: difference vs baseline (raw scale)
    - p_raw, p_holm: bootstrap or Wald p-values with Holm adjustment

    expected_methods optionally restricts analysis to a provided list of methods.
    on_incomplete controls handling of non-complete method crossing per
    patient_lesion: "filter" drops incomplete units with a warning; "error"
    raises with diagnostics.
    contrast_direction can be "alt_vs_baseline" (comparator/dTNV) or
    "baseline_vs_alt" (dTNV/comparator).
    """
    required = ["patient", "lesion", "reconstruction_type", metric]
    _validate_columns(long_df, required)

    data = long_df.dropna(subset=required).copy()
    data["patient"] = data["patient"].astype(str)
    data["lesion"] = data["lesion"].astype(str)
    data["patient_lesion"] = data["patient"] + "__" + data["lesion"]
    data, expected_methods = _apply_expected_methods(
        data, expected_methods, baseline_recon
    )
    data, drop_summary = _enforce_complete_crossing_tbr(
        data, expected_methods, on_incomplete
    )
    if data.empty:
        raise ValueError(
            "No TBR observations remain after enforcing complete method crossing."
        )
    data = _prepare_response(data, metric, use_log)

    if baseline_recon not in set(data["reconstruction_type"]):
        raise ValueError(f"Baseline recon '{baseline_recon}' not found in data.")

    result = _fit_tbr_model(data, baseline_recon)
    contrast_estimates = _extract_contrast_estimates(result, baseline_recon)
    contrast_estimates = _apply_contrast_direction(
        contrast_estimates, contrast_direction
    )
    contrast_names = list(contrast_estimates.keys())

    if use_bootstrap:
        boot_estimates, n_failed = _bootstrap_tbr_contrasts(
            data,
            baseline_recon,
            contrast_names=contrast_names,
            n_boot=n_boot,
            seed=seed,
        )
        boot_estimates = _apply_contrast_direction_samples(
            boot_estimates, contrast_direction
        )
        empty = _find_empty_bootstrap(boot_estimates)
        if empty and not allow_empty_bootstrap:
            raise RuntimeError(
                f"TBR bootstrap produced no samples for contrasts: {empty}."
            )
        _warn_if_empty_bootstrap(boot_estimates, label="TBR")
        bootstrap_meta = _bootstrap_meta(
            n_boot=n_boot,
            n_failed=n_failed,
            empty_contrasts=empty,
        )
        results_df = _build_bootstrap_results(
            contrast_estimates,
            boot_estimates,
            use_log=use_log,
            baseline_recon=baseline_recon,
            contrast_direction=contrast_direction,
        )
        if n_failed:
            warnings.warn(
                f"Bootstrap: {n_failed} fits failed out of {n_boot} resamples."
            )
    else:
        bootstrap_meta = _bootstrap_meta(n_boot=0, n_failed=0, empty_contrasts=[])
        results_df = _build_wald_results(
            result,
            baseline_recon,
            use_log=use_log,
            contrast_direction=contrast_direction,
        )

    results_df = _attach_metadata(
        results_df,
        drop_summary=drop_summary,
        bootstrap_meta=bootstrap_meta,
    )
    return result, results_df


def fit_background_cov_lmm(
    df: pd.DataFrame,
    metric: str = "background_cov_pct",
    baseline_recon: str = "dtnv",
    use_log: bool = True,
    use_bootstrap: bool = True,
    n_boot: int = 1000,
    seed: int | None = 123,
    deduplicate: bool = True,
    expected_methods: Iterable[str] | None = None,
    on_incomplete: str = "filter",
    contrast_direction: str = "alt_vs_baseline",
    allow_empty_bootstrap: bool = False,
) -> Tuple[object, pd.DataFrame]:
    """
    Fit patient-level mixed-effects model for background CoV.

    Model: log(CoV%) ~ reconstruction_type + (1|patient)

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataframe or patient-level dataframe.
    deduplicate : bool
        If True, collapse to one row per patient x method.

    Output columns (use_log=True):
    - log_ratio_est: log difference vs baseline (dTNV)
    - ratio_est, ci_low, ci_high: ratio scale (exp of log estimates)
    - p_raw, p_holm: bootstrap or Wald p-values with Holm adjustment

    Output columns (use_log=False):
    - diff_est, ci_low, ci_high: difference vs baseline (raw scale)
    - p_raw, p_holm: bootstrap or Wald p-values with Holm adjustment

    expected_methods optionally restricts analysis to a provided list of methods.
    on_incomplete controls handling of non-complete method crossing per
    patient: "filter" drops incomplete patients with a warning; "error"
    raises with diagnostics.
    contrast_direction can be "alt_vs_baseline" (comparator/dTNV) or
    "baseline_vs_alt" (dTNV/comparator).
    """
    if deduplicate:
        data = prepare_background_cov_df(df, metric=metric)
        data = data.dropna(subset=["patient", "reconstruction_type", metric]).copy()
    else:
        _validate_columns(df, ["patient", "reconstruction_type", metric])
        data = df.dropna(subset=["patient", "reconstruction_type", metric]).copy()

    data["patient"] = data["patient"].astype(str)
    data, expected_methods = _apply_expected_methods(
        data, expected_methods, baseline_recon
    )
    data, drop_summary = _enforce_complete_crossing_background(
        data, expected_methods, on_incomplete
    )
    if data.empty:
        raise ValueError(
            "No background CoV observations remain after enforcing complete "
            "method crossing."
        )
    data = _prepare_response(data, metric, use_log)

    if baseline_recon not in set(data["reconstruction_type"]):
        raise ValueError(f"Baseline recon '{baseline_recon}' not found in data.")

    result = _fit_background_model(data, baseline_recon)
    contrast_estimates = _extract_contrast_estimates(result, baseline_recon)
    contrast_estimates = _apply_contrast_direction(
        contrast_estimates, contrast_direction
    )
    contrast_names = list(contrast_estimates.keys())

    if use_bootstrap:
        boot_estimates, n_failed = _bootstrap_background_contrasts(
            data,
            baseline_recon,
            contrast_names=contrast_names,
            n_boot=n_boot,
            seed=seed,
        )
        boot_estimates = _apply_contrast_direction_samples(
            boot_estimates, contrast_direction
        )
        empty = _find_empty_bootstrap(boot_estimates)
        if empty and not allow_empty_bootstrap:
            raise RuntimeError(
                "Background CoV bootstrap produced no samples for contrasts: "
                f"{empty}."
            )
        _warn_if_empty_bootstrap(boot_estimates, label="Background CoV")
        bootstrap_meta = _bootstrap_meta(
            n_boot=n_boot,
            n_failed=n_failed,
            empty_contrasts=empty,
        )
        results_df = _build_bootstrap_results(
            contrast_estimates,
            boot_estimates,
            use_log=use_log,
            baseline_recon=baseline_recon,
            contrast_direction=contrast_direction,
        )
        if n_failed:
            warnings.warn(
                f"Bootstrap: {n_failed} fits failed out of {n_boot} resamples."
            )
    else:
        bootstrap_meta = _bootstrap_meta(n_boot=0, n_failed=0, empty_contrasts=[])
        results_df = _build_wald_results(
            result,
            baseline_recon,
            use_log=use_log,
            contrast_direction=contrast_direction,
        )

    results_df = _attach_metadata(
        results_df,
        drop_summary=drop_summary,
        bootstrap_meta=bootstrap_meta,
    )
    return result, results_df


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_response(
    data: pd.DataFrame, metric: str, use_log: bool
) -> pd.DataFrame:
    data = data.copy()
    if use_log:
        if (data[metric] <= 0).any():
            raise ValueError(
                f"Cannot apply log transform: {metric} contains non-positive values."
            )
        data["response"] = np.log(data[metric].astype(float))
    else:
        data["response"] = data[metric].astype(float)
    return data


def _expected_methods(data: pd.DataFrame) -> List[str]:
    return sorted(pd.unique(data["reconstruction_type"].dropna()))


def _apply_expected_methods(
    data: pd.DataFrame,
    expected_methods: Iterable[str] | None,
    baseline_recon: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if expected_methods is None:
        expected = _expected_methods(data)
        if baseline_recon not in expected:
            raise ValueError(
                f"Baseline recon '{baseline_recon}' not found in data."
            )
        return data, expected
    expected = [str(m) for m in expected_methods]
    if not expected:
        raise ValueError("expected_methods must include at least one method.")
    data_methods = set(data["reconstruction_type"].astype(str))
    missing_overall = [m for m in expected if m not in data_methods]
    if missing_overall:
        raise ValueError(
            "expected_methods not found in data: "
            f"{sorted(missing_overall)}"
        )
    extra = sorted(data_methods - set(expected))
    if extra:
        warnings.warn(
            "Dropping methods not in expected_methods: "
            f"{extra}"
        )
        data = data[data["reconstruction_type"].isin(expected)].copy()
    if baseline_recon not in expected:
        raise ValueError(
            "baseline_recon must be included in expected_methods."
        )
    return data, expected


def _enforce_complete_crossing_background(
    data: pd.DataFrame, expected_methods: List[str], on_incomplete: str
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if on_incomplete not in {"filter", "error"}:
        raise ValueError("on_incomplete must be 'filter' or 'error'.")
    missing_by_patient: Dict[str, List[str]] = {}
    for patient, methods in data.groupby("patient")["reconstruction_type"]:
        missing = sorted(set(expected_methods) - set(methods))
        if missing:
            missing_by_patient[str(patient)] = missing
    drop_summary = _summarize_dropped_units(
        missing_by_patient,
        unit_label="patient",
        expected_methods=expected_methods,
    )
    if missing_by_patient:
        msg = _format_missing_summary(
            missing_by_patient,
            unit_label="patient",
            expected_methods=expected_methods,
        )
        if on_incomplete == "error":
            raise ValueError(msg)
        dropped = sorted(missing_by_patient.keys())
        warnings.warn(f"{msg}. Dropping {len(dropped)} patients.")
        data = data[~data["patient"].isin(dropped)].copy()
    return data, drop_summary


def _enforce_complete_crossing_tbr(
    data: pd.DataFrame, expected_methods: List[str], on_incomplete: str
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if on_incomplete not in {"filter", "error"}:
        raise ValueError("on_incomplete must be 'filter' or 'error'.")
    missing_by_lesion: Dict[Tuple[str, str], List[str]] = {}
    grouped = data.groupby(["patient", "lesion"])["reconstruction_type"]
    for (patient, lesion), methods in grouped:
        missing = sorted(set(expected_methods) - set(methods))
        if missing:
            key = (str(patient), str(lesion))
            missing_by_lesion[key] = missing
    per_patient_counts: Dict[str, int] = {}
    for (patient, _lesion) in missing_by_lesion:
        per_patient_counts[patient] = per_patient_counts.get(patient, 0) + 1
    drop_summary = _summarize_dropped_units(
        missing_by_lesion,
        unit_label="patient_lesion",
        expected_methods=expected_methods,
        per_patient_counts=per_patient_counts,
    )
    if missing_by_lesion:
        msg = _format_missing_summary(
            missing_by_lesion,
            unit_label="patient_lesion",
            expected_methods=expected_methods,
            per_patient_counts=per_patient_counts,
        )
        if on_incomplete == "error":
            raise ValueError(msg)
        dropped_keys = {
            f"{patient}__{lesion}" for patient, lesion in missing_by_lesion.keys()
        }
        warnings.warn(
            f"{msg}. Dropping {len(dropped_keys)} patient_lesion units."
        )
        data = data[
            ~(
                data["patient"].astype(str)
                + "__"
                + data["lesion"].astype(str)
            ).isin(dropped_keys)
        ].copy()
    return data, drop_summary


def _missing_pattern_counts(
    missing_by_unit: Dict[object, List[str]]
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for missing in missing_by_unit.values():
        key = ", ".join(missing)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_missing_summary(
    missing_by_unit: Dict[object, List[str]],
    unit_label: str,
    expected_methods: List[str],
    per_patient_counts: Dict[str, int] | None = None,
    max_examples: int = 6,
) -> str:
    pattern_counts = _missing_pattern_counts(missing_by_unit)
    pattern_msg = ", ".join(
        f"[{pattern}]={count}" for pattern, count in pattern_counts.items()
    )
    examples = list(missing_by_unit.keys())[:max_examples]
    if unit_label == "patient_lesion":
        example_str = ", ".join(
            f"{p}__{l}" for (p, l) in examples
        )
    else:
        example_str = ", ".join(str(e) for e in examples)
    msg = (
        f"Incomplete method crossing for {unit_label}. "
        f"Expected methods: {expected_methods}. "
        f"Missing patterns: {pattern_msg}. "
        f"Example {unit_label} units: {example_str}"
    )
    if per_patient_counts:
        per_patient_msg = ", ".join(
            f"{patient}:{count}"
            for patient, count in sorted(per_patient_counts.items())
        )
        msg += f". Per-patient dropped counts: {per_patient_msg}"
    return msg


def _summarize_dropped_units(
    missing_by_unit: Dict[object, List[str]],
    unit_label: str,
    expected_methods: List[str],
    per_patient_counts: Dict[str, int] | None = None,
    max_examples: int = 6,
) -> Dict[str, object]:
    n_dropped = len(missing_by_unit)
    if n_dropped == 0:
        return {
            "unit_label": unit_label,
            "n_dropped_units": 0,
            "dropped_units_example": "",
            "missing_patterns": "",
            "per_patient_dropped": "",
            "expected_methods": ", ".join(expected_methods),
        }
    pattern_counts = _missing_pattern_counts(missing_by_unit)
    patterns = ", ".join(
        f"[{pattern}]={count}" for pattern, count in pattern_counts.items()
    )
    examples = list(missing_by_unit.keys())[:max_examples]
    if unit_label == "patient_lesion":
        example_str = ", ".join(f"{p}__{l}" for (p, l) in examples)
    else:
        example_str = ", ".join(str(e) for e in examples)
    per_patient_str = ""
    if per_patient_counts:
        per_patient_str = ", ".join(
            f"{patient}:{count}"
            for patient, count in sorted(per_patient_counts.items())
        )
    return {
        "unit_label": unit_label,
        "n_dropped_units": n_dropped,
        "dropped_units_example": example_str,
        "missing_patterns": patterns,
        "per_patient_dropped": per_patient_str,
        "expected_methods": ", ".join(expected_methods),
    }


def _apply_contrast_direction(
    contrast_estimates: Dict[str, float],
    contrast_direction: str,
) -> Dict[str, float]:
    if contrast_direction not in {"alt_vs_baseline", "baseline_vs_alt"}:
        raise ValueError(
            "contrast_direction must be 'alt_vs_baseline' or 'baseline_vs_alt'."
        )
    if contrast_direction == "alt_vs_baseline":
        return contrast_estimates
    return {name: -value for name, value in contrast_estimates.items()}


def _apply_contrast_direction_samples(
    boot_estimates: Dict[str, np.ndarray],
    contrast_direction: str,
) -> Dict[str, np.ndarray]:
    if contrast_direction not in {"alt_vs_baseline", "baseline_vs_alt"}:
        raise ValueError(
            "contrast_direction must be 'alt_vs_baseline' or 'baseline_vs_alt'."
        )
    if contrast_direction == "alt_vs_baseline":
        return boot_estimates
    return {name: -values for name, values in boot_estimates.items()}


def _apply_direction_ci(
    est: float, ci_low: float, ci_high: float, contrast_direction: str
) -> Tuple[float, float, float]:
    if contrast_direction == "alt_vs_baseline":
        return est, ci_low, ci_high
    return -est, -ci_high, -ci_low


def _ratio_definition(baseline_recon: str, contrast_direction: str) -> str:
    if contrast_direction == "alt_vs_baseline":
        return f"comparator / {baseline_recon}"
    return f"{baseline_recon} / comparator"


def _fit_tbr_model(data: pd.DataFrame, baseline_recon: str) -> object:
    formula = (
        "response ~ C(reconstruction_type, "
        f"Treatment(reference='{baseline_recon}'))"
    )
    model = smf.mixedlm(
        formula,
        data,
        groups=data["patient"],
        vc_formula={"patient_lesion": "0 + C(patient_lesion)"},
        re_formula="1",
    )
    return _fit_mixedlm(model)


def _fit_background_model(data: pd.DataFrame, baseline_recon: str) -> object:
    formula = (
        "response ~ C(reconstruction_type, "
        f"Treatment(reference='{baseline_recon}'))"
    )
    model = smf.mixedlm(
        formula,
        data,
        groups=data["patient"],
        re_formula="1",
    )
    return _fit_mixedlm(model)


def _fit_mixedlm(model: object) -> object:
    methods = ("lbfgs", "powell")
    last_error = None
    for method in methods:
        try:
            return model.fit(
                reml=True,
                method=method,
                maxiter=500,
                disp=False,
            )
        except Exception as exc:
            last_error = exc
            warnings.warn(f"{method} failed: {exc}.")
    raise RuntimeError("MixedLM failed to converge.") from last_error


def _extract_contrast_estimates(
    result: object, baseline_recon: str
) -> Dict[str, float]:
    prefix = (
        "C(reconstruction_type, "
        f"Treatment(reference='{baseline_recon}'))[T."
    )
    estimates: Dict[str, float] = {}
    for name in result.params.index:
        if not name.startswith(prefix):
            continue
        recon = name[len(prefix) : -1]
        estimates[recon] = float(result.params[name])
    if not estimates:
        raise ValueError("No reconstruction contrasts found in the model.")
    return estimates


def _build_wald_results(
    result: object,
    baseline_recon: str,
    use_log: bool,
    contrast_direction: str,
) -> pd.DataFrame:
    rows = []
    prefix = (
        "C(reconstruction_type, "
        f"Treatment(reference='{baseline_recon}'))[T."
    )
    z_crit = norm.ppf(0.975)
    for name in result.params.index:
        if not name.startswith(prefix):
            continue
        recon = name[len(prefix) : -1]
        est = float(result.params[name])
        se = float(result.bse[name])
        z_val = est / se if se else np.nan
        p_val = 2 * norm.sf(abs(z_val)) if np.isfinite(z_val) else np.nan
        ci_low_log = est - z_crit * se
        ci_high_log = est + z_crit * se
        est_adj, ci_low_adj, ci_high_adj = _apply_direction_ci(
            est, ci_low_log, ci_high_log, contrast_direction
        )
        rows.append(
            _format_result_row(
                recon,
                est_adj,
                ci_low_adj,
                ci_high_adj,
                p_val,
                use_log,
            )
        )
    results_df = pd.DataFrame(rows)
    results_df["p_holm"] = _holm_adjust(results_df["p_raw"].values)
    results_df["baseline_recon"] = baseline_recon
    results_df["contrast_direction"] = contrast_direction
    if use_log:
        results_df["ratio_definition"] = _ratio_definition(
            baseline_recon, contrast_direction
        )
    return results_df.sort_values("recon_alternative").reset_index(drop=True)


def _build_bootstrap_results(
    contrast_estimates: Dict[str, float],
    boot_estimates: Dict[str, np.ndarray],
    use_log: bool,
    baseline_recon: str,
    contrast_direction: str,
) -> pd.DataFrame:
    rows = []
    for recon, est in contrast_estimates.items():
        samples = np.asarray(boot_estimates.get(recon, []), dtype=float)
        rows.append(
            _format_bootstrap_row(recon, est, samples, use_log)
        )
    results_df = pd.DataFrame(rows)
    results_df["p_holm"] = _holm_adjust(results_df["p_raw"].values)
    results_df["baseline_recon"] = baseline_recon
    results_df["contrast_direction"] = contrast_direction
    if use_log:
        results_df["ratio_definition"] = _ratio_definition(
            baseline_recon, contrast_direction
        )
    return results_df.sort_values("recon_alternative").reset_index(drop=True)


def _format_result_row(
    recon: str,
    est: float,
    ci_low: float,
    ci_high: float,
    p_raw: float,
    use_log: bool,
) -> Dict[str, float]:
    if use_log:
        ratio_est = float(np.exp(est))
        ci_low_ratio = float(np.exp(ci_low)) if np.isfinite(ci_low) else np.nan
        ci_high_ratio = float(np.exp(ci_high)) if np.isfinite(ci_high) else np.nan
        return {
            "recon_alternative": recon,
            "log_ratio_est": est,
            "ratio_est": ratio_est,
            "ci_low": ci_low_ratio,
            "ci_high": ci_high_ratio,
            "p_raw": p_raw,
        }
    return {
        "recon_alternative": recon,
        "diff_est": est,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_raw": p_raw,
    }


def _format_bootstrap_row(
    recon: str, est: float, samples: np.ndarray, use_log: bool
) -> Dict[str, float]:
    if samples.size:
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        p_raw = _bootstrap_pvalue(samples)
    else:
        ci_low, ci_high, p_raw = np.nan, np.nan, np.nan
    return _format_result_row(recon, est, ci_low, ci_high, p_raw, use_log)


def _bootstrap_pvalue(samples: np.ndarray) -> float:
    if samples.size == 0:
        return np.nan
    prop_le = float(np.mean(samples <= 0))
    prop_ge = float(np.mean(samples >= 0))
    return min(1.0, 2.0 * min(prop_le, prop_ge))


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    adjusted = np.full_like(pvals, np.nan)
    if pvals.size == 0:
        return adjusted
    finite_mask = np.isfinite(pvals)
    if not finite_mask.any():
        return adjusted
    finite_pvals = pvals[finite_mask]
    try:
        from statsmodels.stats.multitest import multipletests

        adjusted_vals = multipletests(finite_pvals, method="holm")[1]
    except Exception:
        order = np.argsort(finite_pvals)
        adjusted_vals = np.empty_like(finite_pvals)
        m = finite_pvals.size
        for i, idx in enumerate(order):
            adjusted_vals[idx] = min(1.0, finite_pvals[idx] * (m - i))
            if i > 0:
                prev = order[i - 1]
                adjusted_vals[idx] = max(adjusted_vals[idx], adjusted_vals[prev])
    adjusted[finite_mask] = adjusted_vals
    return adjusted


def _bootstrap_tbr_contrasts(
    data: pd.DataFrame,
    baseline_recon: str,
    contrast_names: List[str],
    n_boot: int,
    seed: int | None,
) -> Tuple[Dict[str, np.ndarray], int]:
    rng = np.random.default_rng(seed)
    patients = data["patient"].unique()
    estimates = {name: [] for name in contrast_names}
    n_failed = 0
    for b in range(n_boot):
        boot_df = _resample_patients(data, patients, rng, boot_id=b, with_lesions=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _fit_tbr_model(boot_df, baseline_recon)
                boot_contrasts = _extract_contrast_estimates(result, baseline_recon)
        except Exception:
            n_failed += 1
            continue
        if set(boot_contrasts) != set(contrast_names):
            n_failed += 1
            continue
        for name in contrast_names:
            estimates[name].append(boot_contrasts[name])
    estimates = {k: np.asarray(v) for k, v in estimates.items()}
    return estimates, n_failed


def _bootstrap_background_contrasts(
    data: pd.DataFrame,
    baseline_recon: str,
    contrast_names: List[str],
    n_boot: int,
    seed: int | None,
) -> Tuple[Dict[str, np.ndarray], int]:
    rng = np.random.default_rng(seed)
    patients = data["patient"].unique()
    estimates = {name: [] for name in contrast_names}
    n_failed = 0
    for b in range(n_boot):
        boot_df = _resample_patients(data, patients, rng, boot_id=b, with_lesions=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _fit_background_model(boot_df, baseline_recon)
                boot_contrasts = _extract_contrast_estimates(result, baseline_recon)
        except Exception:
            n_failed += 1
            continue
        if set(boot_contrasts) != set(contrast_names):
            n_failed += 1
            continue
        for name in contrast_names:
            estimates[name].append(boot_contrasts[name])
    estimates = {k: np.asarray(v) for k, v in estimates.items()}
    return estimates, n_failed


def _resample_patients(
    data: pd.DataFrame,
    patients: np.ndarray,
    rng: np.random.Generator,
    boot_id: int,
    with_lesions: bool,
) -> pd.DataFrame:
    sampled = rng.choice(patients, size=len(patients), replace=True)
    parts = []
    for i, patient in enumerate(sampled):
        subset = data[data["patient"] == patient].copy()
        new_patient = f"{patient}__boot{boot_id}_{i}"
        subset["patient"] = new_patient
        if with_lesions:
            subset["patient_lesion"] = new_patient + "__" + subset["lesion"].astype(str)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def _find_empty_bootstrap(
    boot_estimates: Dict[str, np.ndarray]
) -> List[str]:
    return [name for name, vals in boot_estimates.items() if len(vals) == 0]


def _warn_if_empty_bootstrap(
    boot_estimates: Dict[str, np.ndarray], label: str
) -> None:
    empty = _find_empty_bootstrap(boot_estimates)
    if empty:
        warnings.warn(
            f"{label} bootstrap produced no samples for contrasts: {empty}."
        )


def _bootstrap_meta(
    n_boot: int, n_failed: int, empty_contrasts: List[str]
) -> Dict[str, object]:
    n_success = n_boot - n_failed
    return {
        "bootstrap_used": bool(n_boot),
        "bootstrap_n_attempted": n_boot,
        "bootstrap_n_failed": n_failed,
        "bootstrap_n_success": n_success,
        "bootstrap_empty_contrasts": ", ".join(empty_contrasts),
    }


def _attach_metadata(
    results_df: pd.DataFrame,
    drop_summary: Dict[str, object],
    bootstrap_meta: Dict[str, object],
) -> pd.DataFrame:
    results_df["dropped_unit_label"] = drop_summary.get("unit_label", "")
    results_df["n_dropped_units"] = drop_summary.get("n_dropped_units", "")
    results_df["dropped_units_example"] = drop_summary.get(
        "dropped_units_example", ""
    )
    results_df["missing_patterns"] = drop_summary.get("missing_patterns", "")
    results_df["per_patient_dropped"] = drop_summary.get(
        "per_patient_dropped", ""
    )
    results_df["expected_methods"] = drop_summary.get("expected_methods", "")
    results_df["bootstrap_used"] = bootstrap_meta.get("bootstrap_used", "")
    results_df["bootstrap_n_attempted"] = bootstrap_meta.get(
        "bootstrap_n_attempted", ""
    )
    results_df["bootstrap_n_failed"] = bootstrap_meta.get(
        "bootstrap_n_failed", ""
    )
    results_df["bootstrap_n_success"] = bootstrap_meta.get(
        "bootstrap_n_success", ""
    )
    results_df["bootstrap_empty_contrasts"] = bootstrap_meta.get(
        "bootstrap_empty_contrasts", ""
    )
    return results_df


__all__ = [
    "prepare_background_cov_df",
    "fit_tbr_lmm",
    "fit_background_cov_lmm",
]

# Usage example (commented)
# ---------------------------------------------------------------------------
# from patient_endpoint_models import fit_tbr_lmm, fit_background_cov_lmm
#
# # TBR model (lesion-level)
# tbr_model, tbr_results = fit_tbr_lmm(
#     long_df,
#     metric="tbr_mean",
#     baseline_recon="dtnv",
#     use_log=True,
#     use_bootstrap=True,
#     n_boot=5000,
#     seed=123,
#     expected_methods=RECON_TYPES,
#     contrast_direction="alt_vs_baseline",
#     on_incomplete="filter",
# )
# # Interpret ratios and CIs on the ratio scale (ci_low/ci_high). CI excludes 1.0
# # indicates evidence against the null ratio of 1.
#
# # Background CoV model (patient-level)
# cov_model, cov_results = fit_background_cov_lmm(
#     long_df,
#     metric="background_cov_pct",
#     baseline_recon="dtnv",
#     use_log=True,
#     use_bootstrap=True,
#     n_boot=5000,
#     seed=123,
#     expected_methods=RECON_TYPES,
#     contrast_direction="alt_vs_baseline",
#     on_incomplete="filter",
# )
# # Holm-adjusted p-values are in cov_results["p_holm"].
