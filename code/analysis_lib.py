"""Shared regression helpers used by 05_analysis.py and 06_figures.py."""

import numpy as np
import pandas as pd
import pyhdfe
from scipy import stats

from config import MIN_FATALS_PER_YEAR
from utils import get_logger

log = get_logger("analysis_lib")


def prep_panel(df: pd.DataFrame, min_fatals: int = MIN_FATALS_PER_YEAR) -> pd.DataFrame:
    """Filter sample and add integer FE codes.

    Keeps counties with ≥2 years of data AND ≥min_fatals mean annual fatalities.
    Pass min_fatals=0 to skip the fatality threshold (e.g. for sensitivity checks).
    """
    county_years = df.groupby("fips")["year"].nunique()
    keep = county_years[county_years >= 2].index
    df = df[df["fips"].isin(keep)].copy()

    # Drop near-zero counties
    if min_fatals > 0:
        mean_annual = (
            df.groupby(["fips", "year"])["fatals_t0"].sum()
            .groupby("fips").mean()
        )
        keep_fat = mean_annual[mean_annual >= min_fatals].index
        before = df["fips"].nunique()
        df = df[df["fips"].isin(keep_fat)].copy()
        log.info("County restriction (≥%d fatals/yr): %d → %d counties, %d rows",
                 min_fatals, before, df["fips"].nunique(), len(df))

    df["county_code"]    = pd.Categorical(df["fips"]).codes.astype(np.int32)
    df["state_code"]     = pd.Categorical(df["fips"].str[:2]).codes.astype(np.int32)
    df["dow_month_code"] = pd.Categorical(
        df["dow"].astype(str) + "_" + df["month"].astype(str)
    ).codes.astype(np.int32)
    df["year_code"] = pd.Categorical(df["year"]).codes.astype(np.int32)
    df["dow_year_code"] = pd.Categorical(
        df["dow"].astype(str) + "_" + df["year"].astype(str)
    ).codes.astype(np.int32)
    df["year_month_code"] = pd.Categorical(
        df["year"].astype(str) + "_" + df["month"].astype(str)
    ).codes.astype(np.int32)
    return df


def _weighted_demean(arr: np.ndarray, group_codes: np.ndarray,
                     weights: np.ndarray, n_groups: int) -> np.ndarray:
    """Subtract weighted group means from arr (single FE dimension)."""
    w_sum  = np.bincount(group_codes, weights=weights,          minlength=n_groups)
    wa_sum = np.bincount(group_codes, weights=weights * arr,    minlength=n_groups)
    g_mean = np.where(w_sum > 0, wa_sum / w_sum, 0.0)
    return arr - g_mean[group_codes]


def _weighted_two_way_demean(arr: np.ndarray, g1: np.ndarray, g2: np.ndarray,
                              weights: np.ndarray, n1: int, n2: int,
                              tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """Alternating weighted projections for two-way FE absorption."""
    r = arr.copy()
    for _ in range(max_iter):
        r_new = _weighted_demean(r, g1, weights, n1)
        r_new = _weighted_demean(r_new, g2, weights, n2)
        if np.max(np.abs(r_new - r)) < tol:
            return r_new
        r = r_new
    return r


def fe_ols_from_panel(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "night_alert",
    controls: list = [],
    county: bool = True,
    dm: bool = True,
    extra_fes: list = [],
    weights_col: str = "",
    cluster_col: str = "county_code",
    label: str = "",
) -> dict:
    """
    Multi-way FE OLS via pyhdfe absorption.
    Clustered SEs at the county level via vectorised bincount sandwich.

    Parameters
    ----------
    df : prepared panel (must have county_code, dow_month_code columns)
    outcome : dependent variable column name
    treatment : treatment indicator column name
    controls : additional regressor columns
    county : absorb county FE
    dm : absorb dow×month FE
    extra_fes : additional FE column names to absorb (e.g. ["year_code"])
    label : string label for results dict
    """
    cols = ["fips", "county_code", "state_code", "dow_month_code"] + extra_fes + \
           [outcome, treatment] + controls + \
           ([weights_col] if weights_col else []) + \
           ([cluster_col] if cluster_col not in ("county_code", "state_code") else [])
    sub  = df[[c for c in dict.fromkeys(cols) if c in df.columns]].dropna()
    n    = len(sub)

    if n < 500:
        return {"model": label, "error": f"only {n} obs"}

    y = sub[outcome].to_numpy(dtype=float)
    X = sub[[treatment] + controls].to_numpy(dtype=float)
    use_wls = bool(weights_col and weights_col in sub.columns)
    w = sub[weights_col].to_numpy(dtype=float) if use_wls else None

    if use_wls:
        # Weighted FE absorption via alternating weighted projections.
        # county + dow_month are absorbed with full weights.
        # Extra FEs (e.g. outcome-date DoW×Month) are absorbed via unweighted
        # demean on the residuals — acceptable approximation for robustness specs.
        c_codes = sub["county_code"].to_numpy()
        n_c = int(c_codes.max()) + 1
        yw, Xw = y.copy(), X.copy()
        if county and dm:
            dm_codes = sub["dow_month_code"].to_numpy()
            n_dm = int(dm_codes.max()) + 1
            yw = _weighted_two_way_demean(yw, c_codes, dm_codes, w, n_c, n_dm)
            for j in range(Xw.shape[1]):
                Xw[:, j] = _weighted_two_way_demean(Xw[:, j], c_codes,
                                                     dm_codes, w, n_c, n_dm)
        elif county:
            yw = _weighted_demean(yw, c_codes, w, n_c)
            for j in range(Xw.shape[1]):
                Xw[:, j] = _weighted_demean(Xw[:, j], c_codes, w, n_c)
        # Absorb any extra FEs via unweighted demean (approximation)
        ones = np.ones(len(yw))
        for col in extra_fes:
            if col in sub.columns and col not in ("county_code", "dow_month_code"):
                fe_codes = sub[col].to_numpy(dtype=np.int32)
                n_fe = int(fe_codes.max()) + 1
                yw = _weighted_demean(yw, fe_codes, ones, n_fe)
                for j in range(Xw.shape[1]):
                    Xw[:, j] = _weighted_demean(Xw[:, j], fe_codes, ones, n_fe)
        # WLS: multiply by sqrt(w)
        sw = np.sqrt(w)
        y_r = yw * sw
        X_r = Xw * sw[:, None]
    else:
        # Unweighted path: pyhdfe FE absorption
        fe_parts = []
        if county:
            fe_parts.append(sub["county_code"].to_numpy())
        if dm:
            fe_parts.append(sub["dow_month_code"].to_numpy())
        for col in extra_fes:
            if col in sub.columns and col not in ("county_code", "dow_month_code"):
                fe_parts.append(sub[col].to_numpy())

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

    # OLS / WLS coefficient
    coef, _, _, _ = np.linalg.lstsq(X_r, y_r, rcond=None)
    e = y_r - X_r @ coef
    k = X_r.shape[1]

    # Degrees of freedom
    n_fe = (sub["county_code"].nunique() if county else 0) + \
           (sub["dow_month_code"].nunique() if dm else 0) + \
           sum(sub[c].nunique() for c in extra_fes
               if c in sub.columns and c not in ("county_code", "dow_month_code"))
    dof_resid = max(n - k - n_fe, 1)

    # Cluster-robust sandwich — cluster at county or state level
    cl_col  = cluster_col if cluster_col in sub.columns else "county_code"
    c_codes = pd.Categorical(sub[cl_col]).codes.astype(np.int32)
    G       = int(c_codes.max()) + 1
    XtX_inv = np.linalg.pinv(X_r.T @ X_r)
    # For WLS the score is already scaled by sqrt(w) via X_r and e
    scores   = X_r * e[:, None]
    c_scores = np.zeros((G, k))
    for j in range(k):
        c_scores[:, j] = np.bincount(c_codes, weights=scores[:, j], minlength=G)
    active   = np.unique(c_codes)
    c_scores = c_scores[active]
    G_active = len(active)
    meat     = c_scores.T @ c_scores
    scale    = (G_active / (G_active - 1)) * (n / dof_resid)
    V        = scale * XtX_inv @ meat @ XtX_inv

    treat_idx = 1 if not (county or dm or use_wls) else 0
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
