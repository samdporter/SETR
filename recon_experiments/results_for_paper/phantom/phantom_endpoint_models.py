from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd


def enforce_complete_crossing(
    df: pd.DataFrame,
    subject_col: str,
    factor_cols: Sequence[str],
    levels_dict: Mapping[str, Sequence] | None = None,
    on_incomplete: str = "filter",
) -> pd.DataFrame:
    """Ensure each subject has all factor levels; filter or raise on gaps."""
    if on_incomplete not in {"filter", "raise"}:
        raise ValueError("on_incomplete must be 'filter' or 'raise'")

    levels_dict = levels_dict or {}
    expected_subjects = list(levels_dict.get(subject_col, df[subject_col].dropna().unique()))
    expected_levels = {
        col: list(levels_dict.get(col, df[col].dropna().unique())) for col in factor_cols
    }

    df = df[df[subject_col].isin(expected_subjects)].copy()
    missing_subjects: list = []

    for subj, sub_df in df.groupby(subject_col):
        complete = True
        for col, levels in expected_levels.items():
            have = set(sub_df[col].dropna().unique())
            if not set(levels).issubset(have):
                complete = False
                break
        if not complete:
            missing_subjects.append(subj)

    if missing_subjects:
        msg = (
            f"Incomplete crossing for {len(missing_subjects)} subject(s): "
            f"{sorted(missing_subjects)[:10]}"
        )
        if on_incomplete == "raise":
            raise ValueError(msg)
        warnings.warn(msg)
        df = df[~df[subject_col].isin(missing_subjects)].copy()

    return df


def compute_cov_by_bootstrap(
    arrays_by_method: Mapping[str, Mapping[int, np.ndarray]],
    masks: Mapping[str, np.ndarray],
    bootstraps: Sequence[int],
    eps: float = 1e-10,
) -> pd.DataFrame:
    """Compute per-bootstrap voxelwise mean/std/CoV within each mask."""
    records: list[dict] = []

    for method, arrays in arrays_by_method.items():
        for bootstrap_id in bootstraps:
            if bootstrap_id not in arrays:
                continue
            arr = arrays[bootstrap_id]
            for region, mask in masks.items():
                vox = arr[mask]
                mean_val = float(vox.mean())
                std_val = float(vox.std())
                cov_val = std_val / (mean_val + eps)
                records.append(
                    {
                        "bootstrap": bootstrap_id,
                        "Method": method,
                        "Region": region,
                        "mean": mean_val,
                        "std": std_val,
                        "cov": cov_val,
                    }
                )

    return pd.DataFrame.from_records(records)


def summarise_cov(
    df_long: pd.DataFrame,
    cov_col: str = "cov",
    group_cols: Sequence[str] = ("Method", "Region"),
    ci: float | None = 0.95,
) -> pd.DataFrame:
    """Summarise CoV across bootstraps by method/region."""
    grouped = df_long.groupby(list(group_cols))[cov_col]
    summary = grouped.agg(["count", "mean", "std"]).reset_index()
    summary = summary.rename(
        columns={"count": "n_boot", "mean": "mean_cov", "std": "sd_cov"}
    )

    if ci is not None:
        from scipy.stats import t

        alpha = 1.0 - ci
        summary["sem_cov"] = summary["sd_cov"] / np.sqrt(summary["n_boot"])
        summary["ci_low"] = summary["mean_cov"] + t.ppf(alpha / 2, summary["n_boot"] - 1) * summary["sem_cov"]
        summary["ci_high"] = summary["mean_cov"] + t.ppf(1 - alpha / 2, summary["n_boot"] - 1) * summary["sem_cov"]

    return summary



def compute_rc_by_bootstrap(
    arrays_by_method: Mapping[str, Mapping[int, np.ndarray]],
    sphere_masks: Mapping[str, np.ndarray],
    B_true: float | None = None,
    bootstraps: Sequence[int] = (),
    background_mask: np.ndarray | None = None,
    true_ratio: float = 1.0,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """Compute recovery coefficient per bootstrap and sphere.

    If background_mask is provided, the denominator is background_mean * true_ratio
    per bootstrap and method. Otherwise B_true is treated as the background mean
    and the denominator is B_true * true_ratio.
    """
    if background_mask is None and B_true is None:
        raise ValueError("B_true must be provided when background_mask is None")
    if true_ratio <= 0:
        raise ValueError("true_ratio must be positive")

    records: list[dict] = []

    for method, arrays in arrays_by_method.items():
        for bootstrap_id in bootstraps:
            if bootstrap_id not in arrays:
                continue
            arr = arrays[bootstrap_id]
            if background_mask is not None:
                background_mean = float(arr[background_mask].mean())
                denom = background_mean * true_ratio
            else:
                background_mean = float(B_true)
                denom = background_mean * true_ratio
            for sphere, mask in sphere_masks.items():
                sphere_mean = float(arr[mask].mean())
                rc_val = sphere_mean / (denom + eps)
                records.append(
                    {
                        "bootstrap": bootstrap_id,
                        "Method": method,
                        "Sphere": sphere,
                        "sphere_mean": sphere_mean,
                        "background_mean": background_mean,
                        "RC": rc_val,
                    }
                )

    return pd.DataFrame.from_records(records)


def summarise_rc(
    df_long: pd.DataFrame,
    rc_col: str = "RC",
    group_cols: Sequence[str] = ("Method", "Sphere"),
    ci: float | None = 0.95,
) -> pd.DataFrame:
    """Summarise RC across bootstraps by method/sphere."""
    grouped = df_long.groupby(list(group_cols))[rc_col]
    summary = grouped.agg(["count", "mean", "std"]).reset_index()
    summary = summary.rename(
        columns={"count": "n_boot", "mean": "mean_rc", "std": "sd_rc"}
    )

    if ci is not None:
        from scipy.stats import t

        alpha = 1.0 - ci
        summary["sem_rc"] = summary["sd_rc"] / np.sqrt(summary["n_boot"])
        summary["ci_low"] = summary["mean_rc"] + t.ppf(alpha / 2, summary["n_boot"] - 1) * summary["sem_rc"]
        summary["ci_high"] = summary["mean_rc"] + t.ppf(1 - alpha / 2, summary["n_boot"] - 1) * summary["sem_rc"]

    return summary


def fit_rc_mixed_model(
    df: pd.DataFrame,
    baseline_method: str = "DTNV",
    method_col: str = "Method",
    sphere_col: str = "Sphere",
    bootstrap_col: str = "Bootstrap",
    rc_col: str = "RC",
    eps: float = 1e-10,
    fit_kwargs: Mapping[str, object] | None = None,
):
    """Fit a MixedLM on log(RC) with Method*Sphere fixed effects and Bootstrap random intercept."""
    if df.empty:
        raise ValueError("df must be non-empty")
    if baseline_method not in df[method_col].unique():
        raise ValueError(f"baseline_method '{baseline_method}' not found in {method_col}")

    work = df.copy()
    work = work[np.isfinite(work[rc_col])].copy()
    if (work[rc_col] <= 0).any():
        warnings.warn("Non-positive RC values found; adding eps before log.")
    work["logRC"] = np.log(work[rc_col].to_numpy(dtype=float) + eps)

    method_levels = list(pd.unique(work[method_col]))
    if baseline_method in method_levels:
        method_levels = [baseline_method] + [m for m in method_levels if m != baseline_method]
    work[method_col] = pd.Categorical(work[method_col], categories=method_levels, ordered=True)

    sphere_levels = list(pd.unique(work[sphere_col]))
    work[sphere_col] = pd.Categorical(work[sphere_col], categories=sphere_levels, ordered=True)

    import statsmodels.formula.api as smf

    formula = (
        f"logRC ~ C({method_col}, Treatment(reference='{baseline_method}'))"
        f" * C({sphere_col})"
    )
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    model = smf.mixedlm(formula, data=work, groups=work[bootstrap_col])
    return model.fit(reml=False, **fit_kwargs)


def predict_condition_mean(
    fit,
    sphere: str,
    method: str,
    method_col: str = "Method",
    sphere_col: str = "Sphere",
) -> float:
    """Return the fixed-effect mean log(RC) for a sphere/method condition."""
    import patsy

    design_info = fit.model.data.design_info
    new_df = pd.DataFrame({method_col: [method], sphere_col: [sphere]})
    exog = patsy.build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    return float(np.dot(exog, fit.fe_params))


def extract_per_sphere_contrasts(
    fit,
    spheres: Sequence[str],
    comparators: Sequence[str],
    baseline: str = "DTNV",
    method_col: str = "Method",
    sphere_col: str = "Sphere",
) -> pd.DataFrame:
    """Return per-sphere contrasts vs baseline on log scale (with exp ratio)."""
    rows: list[dict] = []
    for sphere in spheres:
        base_mean = predict_condition_mean(
            fit, sphere, baseline, method_col=method_col, sphere_col=sphere_col
        )
        for method in comparators:
            comp_mean = predict_condition_mean(
                fit, sphere, method, method_col=method_col, sphere_col=sphere_col
            )
            delta = comp_mean - base_mean
            rows.append(
                {
                    "Sphere": sphere,
                    "Comparator": method,
                    "Delta_hat": float(delta),
                    "Delta_ratio": float(np.exp(delta)),
                }
            )
    return pd.DataFrame(rows)


def _resample_clusters(
    df: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_col: str,
    cluster_ids: np.ndarray,
) -> pd.DataFrame:
    """Cluster bootstrap: resample Bootstrap IDs with replacement and relabel groups."""
    n_boot = len(cluster_ids)
    sample_ids = rng.choice(cluster_ids, size=n_boot, replace=True)
    frames = []
    for j, bid in enumerate(sample_ids):
        sub = df[df[bootstrap_col] == bid].copy()
        sub[bootstrap_col] = f"{bid}_{j}"
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def bootstrap_mixed_contrasts(
    df: pd.DataFrame,
    K: int = 2000,
    seed: int = 0,
    baseline: str = "DTNV",
    comparators: Sequence[str] | None = None,
    spheres: Sequence[str] | None = None,
    method_col: str = "Method",
    sphere_col: str = "Sphere",
    bootstrap_col: str = "Bootstrap",
    rc_col: str = "RC",
    eps: float = 1e-10,
    max_fail_frac: float = 0.2,
    fit_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Cluster bootstrap MixedLM contrasts; Holm-adjust within each sphere."""
    if df.empty:
        raise ValueError("df must be non-empty")
    if baseline not in df[method_col].unique():
        raise ValueError(f"baseline '{baseline}' not found in {method_col}")

    data = df.copy()
    if comparators is None:
        comparators = [m for m in pd.unique(data[method_col]) if m != baseline]
    if spheres is None:
        spheres = list(pd.unique(data[sphere_col]))

    fit = fit_rc_mixed_model(
        data,
        baseline_method=baseline,
        method_col=method_col,
        sphere_col=sphere_col,
        bootstrap_col=bootstrap_col,
        rc_col=rc_col,
        eps=eps,
        fit_kwargs=fit_kwargs,
    )
    point_df = extract_per_sphere_contrasts(fit, spheres, comparators, baseline=baseline)

    rng = np.random.default_rng(seed)
    cluster_ids = np.array(pd.unique(data[bootstrap_col]))

    samples: dict[tuple[str, str], list[float]] = {
        (sphere, comp): [] for sphere in spheres for comp in comparators
    }
    failures = 0

    for _ in range(K):
        boot_df = _resample_clusters(data, rng, bootstrap_col, cluster_ids)
        try:
            boot_fit = fit_rc_mixed_model(
                boot_df,
                baseline_method=baseline,
                method_col=method_col,
                sphere_col=sphere_col,
                bootstrap_col=bootstrap_col,
                rc_col=rc_col,
                eps=eps,
                fit_kwargs=fit_kwargs,
            )
            boot_contrasts = extract_per_sphere_contrasts(
                boot_fit, spheres, comparators, baseline=baseline
            )
            for _, row in boot_contrasts.iterrows():
                samples[(row["Sphere"], row["Comparator"])].append(float(row["Delta_hat"]))
        except Exception:
            failures += 1
            for key in samples:
                samples[key].append(np.nan)

    if K > 0 and failures / K > max_fail_frac:
        warnings.warn(f"High MixedLM failure rate in bootstrap: {failures}/{K}")

    rows: list[dict] = []
    for _, row in point_df.iterrows():
        key = (row["Sphere"], row["Comparator"])
        vals = np.array(samples.get(key, []), dtype=float)
        valid = vals[np.isfinite(vals)]
        if valid.size:
            ci_low, ci_high = np.nanpercentile(vals, [2.5, 97.5])
            p_lower = float(np.mean(valid <= 0))
            p_upper = float(np.mean(valid >= 0))
            p_raw = 2.0 * min(p_lower, p_upper)
        else:
            ci_low = np.nan
            ci_high = np.nan
            p_raw = np.nan

        rows.append(
            {
                "Sphere": row["Sphere"],
                "Comparator": row["Comparator"],
                "Delta_hat": float(row["Delta_hat"]),
                "Delta_ratio": float(row["Delta_ratio"]),
                "CI_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                "CI_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "n_success": int(valid.size),
                "n_fail": int(np.isnan(vals).sum()),
            }
        )

    result = pd.DataFrame(rows)

    from statsmodels.stats.multitest import multipletests

    result["p_holm"] = np.nan
    result["Reject_0.05"] = False
    for sphere in spheres:
        mask = result["Sphere"] == sphere
        pvals = result.loc[mask, "p_raw"].to_numpy()
        valid = np.isfinite(pvals)
        if not valid.any():
            continue
        reject, p_adj, _, _ = multipletests(pvals[valid], alpha=0.05, method="holm")
        adj = np.full_like(pvals, np.nan, dtype=float)
        rej = np.full_like(pvals, False, dtype=bool)
        adj[valid] = p_adj
        rej[valid] = reject
        result.loc[mask, "p_holm"] = adj
        result.loc[mask, "Reject_0.05"] = rej

    return result
