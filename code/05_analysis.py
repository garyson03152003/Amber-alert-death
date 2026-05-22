"""
05_analysis.py — Main regression analysis.

Baseline specification:
    fatals_{c,t+1} = α + β·NightAlert_{c,t} + γ_c + δ_{dow×month} + X_{c,t}·θ + ε_{c,t}

where γ_c = county FE, δ_{dow×month} = day-of-week × month FE.
Standard errors clustered at the county level.

Models estimated:
    (1) No controls                    (sanity check)
    (2) + county FE
    (3) + county FE + dow×month FE     (baseline)
    (4) + county FE + dow×month FE + weather controls
    (5) Heterogeneity by time-of-night band
    (6) Heterogeneity by next-day type (weekday vs. weekend)
    (7) Placebo: outcome = fatals at t-1 and t+2

Output:
    output/tables/reg_baseline.tex     — Table 1 (LaTeX)
    output/tables/reg_baseline.csv     — machine-readable
    output/tables/reg_hetero.csv       — heterogeneity results
    output/tables/reg_placebo.csv      — placebo results

Run: python code/05_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS, CLUSTER_VAR
from utils import get_logger

log = get_logger("05_analysis")
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    path = DATA_PROC / "panel_county_day.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Panel not found at {path}. Run 04_build_panel.py first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded panel: %d rows, %d counties", len(df), df["fips"].nunique())
    return df


def set_panel_index(df: pd.DataFrame) -> pd.DataFrame:
    """linearmodels requires a MultiIndex of (entity, time)."""
    df = df.set_index(["fips", "date"])
    return df


def extract_results(result, model_label: str) -> dict:
    """Pull key stats from a linearmodels result object."""
    coef_table = result.summary.tables[1]    # parameter estimates
    # Convert to DataFrame regardless of output type
    try:
        ct = result.params
        se = result.std_errors
        pval = result.pvalues
        conf = result.conf_int()
    except Exception:
        return {"model": model_label, "error": str(result)}

    row = {
        "model":       model_label,
        "coef":        ct.get("night_alert", np.nan),
        "se":          se.get("night_alert", np.nan),
        "pval":        pval.get("night_alert", np.nan),
        "ci_lo":       conf["lower"].get("night_alert", np.nan),
        "ci_hi":       conf["upper"].get("night_alert", np.nan),
        "n_obs":       result.nobs,
        "r2_within":   getattr(result, "rsquared_within", np.nan),
        "r2_overall":  getattr(result, "rsquared", np.nan),
    }
    return row


def run_ols(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: list[str],
    entity_effects: bool,
    time_effects: bool,
    label: str,
) -> dict:
    """
    Estimate county-FE OLS using linearmodels.PanelOLS.

    For dow×month effects: we absorb them via entity_effects=False and
    include dow_x_month dummies as explicit regressors (time_effects=False),
    or alternatively use linearmodels AbsorbingLS for two-way FE.

    We use the simpler but correct approach: include dow_x_month as
    categorical dummies within PanelOLS with entity_effects=True.
    """
    sub = df[[outcome, treatment] + controls + ["dow_x_month"]].dropna()
    sub_idx = set_panel_index(sub.reset_index())

    regressors = [treatment] + controls
    # Add dow×month dummies manually (PanelOLS absorbs only the entity FE)
    if "dow_x_month" in sub.columns:
        dow_dummies = pd.get_dummies(sub["dow_x_month"], drop_first=True, prefix="dm")
        sub_idx = sub_idx.join(dow_dummies)
        regressors += list(dow_dummies.columns)

    X = sub_idx[regressors].astype(float)
    y = sub_idx[outcome].astype(float)

    model = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    )
    try:
        result = model.fit(
            cov_type="clustered",
            cluster_entity=True,    # cluster by county
        )
        return extract_results(result, label)
    except Exception as exc:
        log.error("Model %s failed: %s", label, exc)
        return {"model": label, "error": str(exc)}


# ---------------------------------------------------------------------------
# Model suite
# ---------------------------------------------------------------------------

WEATHER_CONTROLS = ["prcp_mm", "tmax_c"]


def run_baseline_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run models (1)–(4) for Table 1.
    """
    results = []

    # (1) No FEs, no controls
    res = run_ols(df, "fatals_t1", "night_alert", [], False, False, "(1) Pooled OLS")
    results.append(res)

    # (2) County FE only
    res = run_ols(df, "fatals_t1", "night_alert", [], True, False, "(2) County FE")
    results.append(res)

    # (3) County FE + dow×month FE  [baseline]
    res = run_ols(df, "fatals_t1", "night_alert", [], True, False, "(3) Baseline")
    results.append(res)

    # (4) Baseline + weather controls
    weather_available = all(c in df.columns for c in WEATHER_CONTROLS)
    if weather_available:
        controls = [c for c in WEATHER_CONTROLS if df[c].notna().any()]
        res = run_ols(df, "fatals_t1", "night_alert", controls, True, False,
                      "(4) + Weather")
        results.append(res)
    else:
        log.warning("Weather controls not available — skipping model (4).")

    return pd.DataFrame(results)


def run_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model (5): separate β by night_band (early / deep / late night).
    Model (6): separate β by next-day type (weekday vs. weekend).
    """
    results = []

    # (5) By night band: replace binary night_alert with band dummies
    band_dummies = pd.get_dummies(df["night_band"], prefix="band", drop_first=False)
    df_hetero = df.join(band_dummies)

    for band in ["band_early_night", "band_deep_night", "band_late_night"]:
        if band not in df_hetero.columns:
            continue
        sub = df_hetero.copy()
        sub["treatment"] = sub[band].fillna(0).astype(int)
        res = run_ols(sub.rename(columns={"treatment": "night_alert"}),
                      "fatals_t1", "night_alert", [], True, False,
                      f"Hetero: {band.replace('band_', '')}")
        results.append(res)

    # (6) By next-day type
    for next_day_type, mask in [("weekday_next", df["dow"] < 4),
                                  ("weekend_next", df["dow"] >= 4)]:
        sub = df[mask].copy()
        res = run_ols(sub, "fatals_t1", "night_alert", [], True, False,
                      f"Hetero: {next_day_type}")
        results.append(res)

    return pd.DataFrame(results)


def run_placebo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placebo tests:
      - Outcome = fatals at t-1 (before the alert — must be null)
      - Outcome = fatals at t+2 (two days later — should fade)
      - Outcome = fatals at t   (same day — partially pre-treatment)
    """
    results = []
    for outcome, label in [
        ("fatals_tm1",  "Placebo: t-1"),
        ("fatals_t0",   "Same-day: t"),
        ("fatals_t1",   "Main: t+1"),
        ("fatals_t2",   "Placebo: t+2"),
    ]:
        if outcome not in df.columns:
            log.warning("Column %s missing — skipping placebo.", outcome)
            continue
        sub = df.dropna(subset=[outcome]).copy()
        res = run_ols(sub, outcome, "night_alert", [], True, False, label)
        results.append(res)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(results: pd.DataFrame, title: str) -> str:
    """Return a simple LaTeX table string."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + title + r"}",
        r"\begin{tabular}{lcccccc}",
        r"\hline\hline",
        r"Model & Coef. & SE & p-value & CI lo & CI hi & N \\",
        r"\hline",
    ]
    for _, row in results.iterrows():
        if "error" in row and pd.notna(row.get("error", np.nan)):
            lines.append(
                f"{row['model']} & \\multicolumn{{6}}{{c}}{{Error: {row.get('error','')}}} \\\\"
            )
            continue
        coef  = f"{row.get('coef',np.nan):.4f}"  if pd.notna(row.get('coef'))  else "—"
        se    = f"{row.get('se',np.nan):.4f}"    if pd.notna(row.get('se'))    else "—"
        pval  = f"{row.get('pval',np.nan):.3f}"  if pd.notna(row.get('pval'))  else "—"
        ci_lo = f"{row.get('ci_lo',np.nan):.4f}" if pd.notna(row.get('ci_lo')) else "—"
        ci_hi = f"{row.get('ci_hi',np.nan):.4f}" if pd.notna(row.get('ci_hi')) else "—"
        n     = f"{int(row.get('n_obs',0)):,}"   if pd.notna(row.get('n_obs')) else "—"
        lines.append(
            f"{row['model']} & {coef} & {se} & {pval} & {ci_lo} & {ci_hi} & {n} \\\\"
        )
    lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

    df = load_panel()

    # Restrict to counties observed in at least 2 years (drop noise-only cells)
    county_years = df.groupby("fips")["year"].nunique()
    keep_fips = county_years[county_years >= 2].index
    df = df[df["fips"].isin(keep_fips)].copy()
    log.info("After county filter: %d rows, %d counties", len(df), df["fips"].nunique())

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    log.info("Running baseline models...")
    baseline_results = run_baseline_models(df)
    log.info("\n%s", baseline_results.to_string())
    baseline_results.to_csv(OUTPUT_TABS / "reg_baseline.csv", index=False)

    tex = format_table(
        baseline_results,
        "Effect of Nighttime AMBER Alert on Next-Day Traffic Fatalities",
    )
    (OUTPUT_TABS / "reg_baseline.tex").write_text(tex)
    log.info("Saved baseline results → output/tables/")

    # -----------------------------------------------------------------------
    # Heterogeneity
    # -----------------------------------------------------------------------
    log.info("Running heterogeneity analysis...")
    hetero_results = run_heterogeneity(df)
    log.info("\n%s", hetero_results.to_string())
    hetero_results.to_csv(OUTPUT_TABS / "reg_hetero.csv", index=False)

    tex_h = format_table(hetero_results, "Heterogeneity by Alert Timing and Next-Day Type")
    (OUTPUT_TABS / "reg_hetero.tex").write_text(tex_h)

    # -----------------------------------------------------------------------
    # Placebo
    # -----------------------------------------------------------------------
    log.info("Running placebo tests...")
    placebo_results = run_placebo(df)
    log.info("\n%s", placebo_results.to_string())
    placebo_results.to_csv(OUTPUT_TABS / "reg_placebo.csv", index=False)

    tex_p = format_table(placebo_results,
                         "Placebo Tests: Effect of AMBER Alert on Non-Next-Day Fatalities")
    (OUTPUT_TABS / "reg_placebo.tex").write_text(tex_p)

    log.info("All regression results saved to output/tables/")


if __name__ == "__main__":
    main()
