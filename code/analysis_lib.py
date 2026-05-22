"""Shared regression helpers used by 05_analysis.py and 06_figures.py."""

import numpy as np
import pandas as pd
import pyhdfe
from scipy import stats

from utils import get_logger

log = get_logger("analysis_lib")


def prep_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to counties with ≥2 years and add integer FE codes."""
    county_years = df.groupby("fips")["year"].nunique()
    keep = county_years[county_years >= 2].index
    df = df[df["fips"].isin(keep)].copy()
    df["county_code"]    = pd.Categorical(df["fips"]).codes.astype(np.int32)
    df["dow_month_code"] = pd.Categorical(
        df["dow"].astype(str) + "_" + df["month"].astype(str)
    ).codes.astype(np.int32)
    return df


def fe_ols_from_panel(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "night_alert",
    controls: list = [],
    county: bool = True,
    dm: bool = True,
    label: str = "",
) -> dict:
    """
    Two-way FE OLS (county + dow×month) via pyhdfe absorption.
    Clustered SEs at the county level via vectorised bincount sandwich.

    Parameters
    ----------
    df : prepared panel (must have county_code, dow_month_code columns)
    outcome : dependent variable column name
    treatment : treatment indicator column name
    controls : additional regressor columns
    county : absorb county FE
    dm : absorb dow×month FE
    label : string label for results dict
    """
    cols = ["fips", "county_code", "dow_month_code", outcome, treatment] + controls
    sub  = df[[c for c in cols if c in df.columns]].dropna()
    n    = len(sub)

    if n < 500:
        return {"model": label, "error": f"only {n} obs"}

    y = sub[outcome].to_numpy(dtype=float)
    X = sub[[treatment] + controls].to_numpy(dtype=float)

    # Build FE id arrays
    fe_parts = []
    if county:
        fe_parts.append(sub["county_code"].to_numpy())
    if dm:
        fe_parts.append(sub["dow_month_code"].to_numpy())

    if fe_parts:
        ids_arr = (np.column_stack(fe_parts) if len(fe_parts) > 1
                   else fe_parts[0].reshape(-1, 1))
        try:
            algo = pyhdfe.create(ids_arr, drop_singletons=False,
                                 compute_degrees=False)
            resid = algo.residualize(np.column_stack([y, X]))
            y_r, X_r = resid[:, 0], resid[:, 1:]
        except Exception as exc:
            return {"model": label, "error": str(exc)}
    else:
        X_r = np.column_stack([np.ones(n), X])
        y_r = y

    # OLS
    coef, _, _, _ = np.linalg.lstsq(X_r, y_r, rcond=None)
    e = y_r - X_r @ coef
    k = X_r.shape[1]

    # Degrees of freedom
    n_fe = (sub["county_code"].nunique() if county else 0) + \
           (sub["dow_month_code"].nunique() if dm else 0)
    dof_resid = max(n - k - n_fe, 1)

    # Vectorised cluster-robust sandwich
    c_codes = sub["county_code"].to_numpy()
    G       = int(c_codes.max()) + 1
    XtX_inv = np.linalg.pinv(X_r.T @ X_r)
    scores  = X_r * e[:, None]                        # n × k
    meat    = np.zeros((k, k))
    c_scores = np.zeros((G, k))
    for j in range(k):
        c_scores[:, j] = np.bincount(c_codes, weights=scores[:, j], minlength=G)
    # Keep only clusters that appear in this sub-sample
    active = np.unique(c_codes)
    c_scores = c_scores[active]
    G_active = len(active)
    meat  = c_scores.T @ c_scores
    scale = (G_active / (G_active - 1)) * (n / dof_resid)
    V     = scale * XtX_inv @ meat @ XtX_inv

    # Treatment is index 0 of X_r (post-absorption, no const)
    # If pooled OLS (no FEs), const is at index 0, treatment at 1
    treat_idx = 1 if not (county or dm) else 0
    b   = float(coef[treat_idx])
    se  = float(np.sqrt(max(V[treat_idx, treat_idx], 0)))
    t   = b / se if se > 0 else np.nan
    pv  = float(2 * stats.t.sf(abs(t), df=G_active - 1)) if not np.isnan(t) else np.nan
    tc  = float(stats.t.ppf(0.975, df=G_active - 1))

    ss_res = float(e @ e)
    ss_tot = float(np.sum((y_r - y_r.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "model":      label,
        "coef":       b,
        "se":         se,
        "pval":       pv,
        "ci_lo":      b - tc * se,
        "ci_hi":      b + tc * se,
        "n_obs":      int(n),
        "n_counties": G_active,
        "r2":         r2,
        "mean_y":     float(np.mean(y)),
    }


# convenience alias used in figures
fe_ols = fe_ols_from_panel
