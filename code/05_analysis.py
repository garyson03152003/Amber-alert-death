"""
05_analysis.py — Main regression analysis.

Baseline:  fatals_{c,t+1} = α + β·NightAlert_{c,t} + γ_c + δ_{dow×month} + X_{c,t}·θ + ε
Estimator: two-way FE via pyhdfe; SE clustered at county level.

Models  (1) Pooled OLS   (2) County FE   (3) Baseline   (4) + Weather
Hetero  night band · weekday/weekend · alert hour
Placebo outcomes at t−1, same-day, t+2

Output: output/tables/reg_baseline.csv/.tex  reg_hetero.csv  reg_placebo.csv

Run: python code/05_analysis.py
"""

import sys, warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel, fe_ols_from_panel
import numpy as np

log = get_logger("05_analysis")
warnings.filterwarnings("ignore")

WEATHER  = ["prcp_mm", "tmax_c"]
HOLIDAY  = ["is_holiday"]        # federal + state public holiday indicator
SERIOUS_INJ_PATH  = DATA_PROC / "fars_serious_injuries.parquet"
AMBER_CLEAN_PATH  = DATA_PROC / "amber_alerts_clean.parquet"


def load_panel() -> pd.DataFrame:
    path = DATA_PROC / "panel_county_day.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Panel not found: {path}")
    df = pd.read_parquet(path)
    log.info("Panel: {:,} rows, {:,} counties".format(len(df), df["fips"].nunique()))

    # Merge serious injuries if available (from 01b_extract_serious_injuries.py)
    if SERIOUS_INJ_PATH.exists():
        inj = pd.read_parquet(SERIOUS_INJ_PATH)
        df = df.merge(inj, on=["fips", "date"], how="left")
        df["serious_injuries"] = df["serious_injuries"].fillna(0).astype(int)
        log.info("Serious injuries merged (mean %.3f/county-day)",
                 df["serious_injuries"].mean())
    else:
        log.warning("Serious injury file not found — run 01b_extract_serious_injuries.py")
        df["serious_injuries"] = 0

    # Merge alert breadth (counties per alert) so we can restrict to narrow alerts
    if AMBER_CLEAN_PATH.exists():
        am = pd.read_parquet(AMBER_CLEAN_PATH)
        breadth = am.groupby("alert_id")["county_fips"].nunique().rename("n_counties")
        am = am.merge(breadth, on="alert_id")
        am["date"] = pd.to_datetime(am["issued_local"].dt.date)
        # For each (county, night), minimum breadth among night alerts that fired
        min_breadth = (
            am[am["is_night"]]
            .groupby(["county_fips", "date"])["n_counties"]
            .min()
            .reset_index()
            .rename(columns={"county_fips": "fips", "n_counties": "alert_breadth"})
        )
        df = df.merge(min_breadth, on=["fips", "date"], how="left")
        df["alert_breadth"] = df["alert_breadth"].fillna(0).astype(int)
        log.info("Alert breadth merged. Night-alert rows: %d; mean breadth: %.1f counties",
                 (df["alert_breadth"] > 0).sum(),
                 df.loc[df["alert_breadth"] > 0, "alert_breadth"].mean())
    else:
        log.warning("Amber clean file not found — alert breadth unavailable")
        df["alert_breadth"] = 0

    return df


# ---------------------------------------------------------------------------
# Model suites  (thin wrappers around fe_ols_from_panel)
# ---------------------------------------------------------------------------

def run_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Timing-aligned baseline models using fatals_next_commute as the outcome.

    fatals_next_commute = fatals_t0 for midnight-6am alerts (commute disrupted
    same morning) and fatals_t1 for early-night alerts and all control rows
    (commute disrupted the following morning).
    """
    df_al = add_aligned_outcome(df)
    avail_w   = [c for c in WEATHER  if df_al[c].notna().mean() > 0.01]
    avail_hol = [c for c in HOLIDAY  if c in df_al.columns]
    base_ctrl = avail_hol   # holiday indicator included in all specs where available
    results = []

    log.info("(1) Pooled OLS [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     controls=base_ctrl,
                                     county=False, dm=False,
                                     label="(1) Pooled OLS"))
    log.info("(2) County FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     controls=base_ctrl,
                                     county=True, dm=False,
                                     label="(2) County FE"))
    log.info("(3) County FE + DoW×Month FE  [baseline, aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     controls=base_ctrl,
                                     county=True, dm=True,
                                     label="(3) Baseline"))
    if avail_w:
        log.info("(4) Baseline + weather [aligned]")
        results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                         controls=base_ctrl + avail_w,
                                         county=True, dm=True,
                                         label="(4) + Weather"))
    else:
        log.warning("Weather controls sparse — skipping model (4).")

    log.info("(5) Baseline + Year FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     controls=base_ctrl,
                                     county=True, dm=True,
                                     extra_fes=["year_code"],
                                     label="(5) + Year FE"))
    log.info("(6) County FE + DoW×Year FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     controls=base_ctrl,
                                     county=True, dm=False,
                                     extra_fes=["dow_year_code"],
                                     label="(6) DoW×Year FE"))
    log.info("(7) County FE + Year×Month FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     controls=base_ctrl,
                                     county=True, dm=False,
                                     extra_fes=["year_month_code"],
                                     label="(7) Year×Month FE"))

    return pd.DataFrame(results)


def add_aligned_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct fatals_next_commute and fatals_rate_next_commute.

    Timing rule: midnight–6am alerts fire on calendar day t, disrupting the
    same morning's commute → outcome is fatals_t0.  Early-night (10pm–midnight)
    alerts fire on day t-1, disrupting the following morning → outcome is fatals_t1.
    Control rows default to fatals_t1.

    Rate outcomes (per 100k population) are built from the same raw counts.
    Missing county-year population is imputed with the county's cross-year mean.
    """
    df = df.copy()
    # --- aligned raw count ---
    df["fatals_next_commute"] = df["fatals_t1"]
    midnight_mask = df["night_band"].isin(["deep_night", "late_night"])
    df.loc[midnight_mask, "fatals_next_commute"] = df.loc[midnight_mask, "fatals_t0"]

    # --- combined: fatalities + serious injuries (in fatal crashes) ---
    # serious_injuries column is same-day (t0); build t1 via within-county shift
    if "serious_injuries" in df.columns:
        df = df.sort_values(["fips", "date"])
        df["serious_inj_t1"] = (
            df.groupby("fips")["serious_injuries"].shift(-1).fillna(0)
        )
        # Align serious injuries using same timing rule as fatalities
        df["serious_inj_next_commute"] = df["serious_inj_t1"]
        df.loc[midnight_mask, "serious_inj_next_commute"] = \
            df.loc[midnight_mask, "serious_injuries"]

        df["combined_next_commute"] = (
            df["fatals_next_commute"] + df["serious_inj_next_commute"]
        )

    # --- population: fill missing with county cross-year mean ---
    if "population" in df.columns:
        county_mean_pop = df.groupby("fips")["population"].transform("mean")
        pop = df["population"].fillna(county_mean_pop)
        pop_100k = pop / 100_000

        for raw, rate in [
            ("fatals_t0",           "fatals_rate_t0"),
            ("fatals_t1",           "fatals_rate_t1"),
            ("fatals_tm1",          "fatals_rate_tm1"),
            ("fatals_t2",           "fatals_rate_t2"),
            ("fatals_next_commute", "fatals_rate_next_commute"),
        ]:
            if raw in df.columns:
                df[rate] = df[raw] / pop_100k

        import numpy as np
        df["log_pop"] = np.log(pop.clip(lower=1))

    return df


def run_rate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same seven specifications as run_baseline but using fatals_rate_next_commute
    (traffic fatalities per 100,000 county population) as the outcome.
    This normalises for county size and reduces heteroskedasticity.
    """
    df_al = add_aligned_outcome(df)
    if "fatals_rate_next_commute" not in df_al.columns:
        log.warning("Population data missing — cannot compute rate outcome.")
        return pd.DataFrame()

    df_al = df_al.dropna(subset=["fatals_rate_next_commute"])
    pop_coverage = df_al["fatals_rate_next_commute"].notna().mean()
    log.info("Rate outcome coverage: %.1f%%", pop_coverage * 100)

    avail_w   = [c for c in WEATHER if df_al[c].notna().mean() > 0.01]
    avail_hol = [c for c in HOLIDAY if c in df_al.columns]
    base_ctrl = avail_hol
    results = []

    for label, kwargs in [
        ("(1) Pooled OLS",        dict(county=False, dm=False)),
        ("(2) County FE",         dict(county=True,  dm=False)),
        ("(3) Baseline",          dict(county=True,  dm=True)),
        ("(5) + Year FE",         dict(county=True,  dm=True,  extra_fes=["year_code"])),
        ("(6) DoW×Year FE",       dict(county=True,  dm=False, extra_fes=["dow_year_code"])),
        ("(7) Year×Month FE",     dict(county=True,  dm=False, extra_fes=["year_month_code"])),
    ]:
        log.info("%s [rate]", label)
        results.append(fe_ols_from_panel(df_al, "fatals_rate_next_commute",
                                         controls=base_ctrl, label=label, **kwargs))
    if avail_w:
        log.info("(4) Baseline + weather [rate]")
        results.append(fe_ols_from_panel(df_al, "fatals_rate_next_commute",
                                         controls=base_ctrl + avail_w,
                                         county=True, dm=True,
                                         label="(4) + Weather"))

    return pd.DataFrame(results)


def _rate_specs(df_al: pd.DataFrame, outcome: str, tag: str,
                weights_col: str = "",
                extra_ctrl: list = None) -> list:
    """Run the three core FE specs on a rate outcome; return list of result dicts."""
    ctrl = (extra_ctrl or []) + [c for c in HOLIDAY if c in df_al.columns]
    results = []
    for label, kwargs in [
        (f"(2) County FE [{tag}]",   dict(county=True, dm=False)),
        (f"(3) Baseline [{tag}]",    dict(county=True, dm=True)),
        (f"(5) + Year FE [{tag}]",   dict(county=True, dm=True,
                                          extra_fes=["year_code"])),
    ]:
        results.append(fe_ols_from_panel(df_al, outcome, label=label,
                                         controls=ctrl,
                                         weights_col=weights_col, **kwargs))
    return results


def run_se_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    WLS robustness checks on the (already restricted) sample, using
    fatals_rate_next_commute. County restriction is applied in prep_panel.

    Unweighted (OLS) vs log-population weighted (WLS).
    """
    df_al = add_aligned_outcome(df)
    if "fatals_rate_next_commute" not in df_al.columns:
        log.warning("Population data missing — skipping SE reduction specs.")
        return pd.DataFrame()

    df_al = df_al.dropna(subset=["fatals_rate_next_commute", "population"])
    results = []

    log.info("SE reduction: OLS (rate)")
    for r in _rate_specs(df_al, "fatals_rate_next_commute", "OLS"):
        results.append(r)

    log.info("SE reduction: log-pop WLS (rate)")
    for r in _rate_specs(df_al, "fatals_rate_next_commute", "logWLS",
                         weights_col="log_pop"):
        results.append(r)

    return pd.DataFrame(results)


def run_aligned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three aligned specifications that use the correct fatality window per band.
    Memory: modifies night_alert in-place, restores original value after each spec.
    """
    import gc
    orig_night_alert = df["night_alert"].copy()
    results = []

    # (A) Early-night alerts → fatals_t1
    log.info("(A) Aligned: early_night → fatals_t1")
    df["night_alert"] = (df["night_band"] == "early_night").astype(int)
    results.append(fe_ols_from_panel(df, "fatals_t1", county=True, dm=True,
                                     label="(A) Early-night → t+1"))

    # (B) Midnight–6am alerts → fatals_t0
    log.info("(B) Aligned: midnight-6am → fatals_t0")
    df["night_alert"] = df["night_band"].isin(["deep_night", "late_night"]).astype(int)
    results.append(fe_ols_from_panel(df, "fatals_t0", county=True, dm=True,
                                     label="(B) Midnight-6am → t+0"))

    # Restore original night_alert before building aligned outcome
    df["night_alert"] = orig_night_alert
    del orig_night_alert; gc.collect()

    # (C) Pooled aligned: fatals_next_commute
    log.info("(C) Aligned: pooled (fatals_next_commute)")
    df_al = add_aligned_outcome(df)
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=True,
                                     label="(C) Pooled aligned"))

    # (D) + Year FE
    log.info("(D) Aligned pooled + Year FE")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=True,
                                     extra_fes=["year_code"],
                                     label="(D) Pooled aligned + Year FE"))

    # (M) Misaligned memo
    log.info("(M) Memo: misaligned baseline (fatals_t1, all night)")
    results.append(fe_ols_from_panel(df_al, "fatals_t1", county=True, dm=True,
                                     label="(M) Misaligned baseline [memo]"))
    del df_al; gc.collect()

    return pd.DataFrame(results)


def run_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heterogeneity analysis using timing-aligned outcomes per band.
    Memory: modifies night_alert in-place to avoid copies; restores after each spec.
    """
    import gc
    df_al = add_aligned_outcome(df)
    orig_night = df_al["night_alert"].copy()
    results = []

    # Band splits: modify night_alert in-place, restore after each
    for band, outcome in [
        ("early_night", "fatals_t1"),
        ("deep_night",  "fatals_t0"),
        ("late_night",  "fatals_t0"),
    ]:
        df_al["night_alert"] = (df_al["night_band"] == band).astype(int)
        results.append(fe_ols_from_panel(df_al, outcome,
                                         county=True, dm=True,
                                         label=f"Band: {band}"))
    df_al["night_alert"] = orig_night

    # Weekday / weekend — pass view (no copy needed)
    for lbl, mask in [
        ("Weekday", df_al["dow"].isin([0, 1, 2, 3])),
        ("Weekend", df_al["dow"].isin([4, 5, 6])),
    ]:
        results.append(fe_ols_from_panel(df_al[mask], "fatals_next_commute",
                                         county=True, dm=True, label=lbl))

    # Alert hour splits: modify in-place, restore after each
    for lbl, hrs, outcome in [
        ("Alert 10pm-midnight", list(range(22, 24)), "fatals_t1"),
        ("Alert midnight-3am",  list(range(0,  3)),  "fatals_t0"),
        ("Alert 3am-6am",       list(range(3,  6)),  "fatals_t0"),
    ]:
        df_al["night_alert"] = (
            orig_night.astype(bool) & df_al["alert_hour"].isin(hrs)
        ).astype(int)
        results.append(fe_ols_from_panel(df_al, outcome,
                                         county=True, dm=True, label=lbl))

    del df_al, orig_night; gc.collect()
    return pd.DataFrame(results)


def run_placebo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placebo tests using aligned outcomes.
    t-1 and t+2 should show no effect; aligned main spec is included for reference.
    """
    import gc
    df_al = add_aligned_outcome(df)
    results = []
    for outcome, lbl in [
        ("fatals_tm1",          "Placebo: t-1"),
        ("fatals_next_commute", "Main: aligned"),
        ("fatals_t2",           "Placebo: t+2"),
    ]:
        if outcome not in df_al.columns:
            continue
        results.append(fe_ols_from_panel(df_al.dropna(subset=[outcome]),
                                         outcome, county=True, dm=True, label=lbl))
    del df_al; gc.collect()
    return pd.DataFrame(results)


def run_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined outcome: fatalities + serious injuries (in fatal crashes), per 100k.
    County restriction is already applied via prep_panel.
    Runs OLS and log-pop WLS variants.
    """
    df_al = add_aligned_outcome(df)
    if "combined_next_commute" not in df_al.columns:
        log.warning("Serious injury data missing — skipping combined outcome.")
        return pd.DataFrame()

    if "population" not in df_al.columns:
        log.warning("Population missing — cannot compute combined rate.")
        return pd.DataFrame()

    county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
    pop = df_al["population"].fillna(county_mean_pop)
    df_al["combined_rate"] = df_al["combined_next_commute"] / (pop / 100_000)
    df_al = df_al.dropna(subset=["combined_rate", "population"])
    log.info("Combined rate mean: %.4f per 100k", df_al["combined_rate"].mean())

    results = []
    log.info("Combined OLS")
    for r in _rate_specs(df_al, "combined_rate", "OLS"):
        results.append(r)
    log.info("Combined log-pop WLS")
    for r in _rate_specs(df_al, "combined_rate", "logWLS",
                         weights_col="log_pop"):
        results.append(r)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def to_latex(results: pd.DataFrame, title: str, note: str = "") -> str:
    STARS = [(0.01, "***"), (0.05, "**"), (0.10, "*")]
    def star(p):
        for t, s in STARS:
            if p < t: return s
        return ""

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{" + title + r"}",
        r"\begin{tabular}{lccccc}",
        r"\hline\hline",
        r"Model & $\hat\beta$ & SE & $p$-value & $N$ & Mean $y$ \\",
        r"\midrule",
    ]
    for _, r in results.iterrows():
        if "error" in r and pd.notna(r.get("error")):
            lines.append(f"{r['model']} & \\multicolumn{{5}}{{c}}{{—}} \\\\")
            continue
        s    = star(r.get("pval", 1.0))
        coef = f"{r['coef']:.4f}{s}"   if pd.notna(r.get("coef"))  else "—"
        se   = f"({r['se']:.4f})"      if pd.notna(r.get("se"))    else "—"
        pv   = f"{r['pval']:.3f}"      if pd.notna(r.get("pval"))  else "—"
        n    = f"{int(r['n_obs']):,}"  if pd.notna(r.get("n_obs")) else "—"
        my   = f"{r['mean_y']:.4f}"    if pd.notna(r.get("mean_y")) else "—"
        lines.append(f"{r['model']} & {coef} & {se} & {pv} & {n} & {my} \\\\")

    lines += [r"\hline\hline"]
    if note:
        lines.append(r"\multicolumn{6}{p{0.9\textwidth}}{\footnotesize " + note + r"} \\")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Weekday / weekend interaction
# ---------------------------------------------------------------------------

def _make_workday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add night_alert_workday and night_alert_weekend to df (in-place copy).

    Uses band-specific timing to determine which calendar night precedes a
    workday commute (see run_event_study_workday for the full explanation):

    • early_night (10pm–midnight on day D, disrupts D+1 commute):
        workday if D ∈ {Sun(6), Mon(0), Tue(1), Wed(2), Thu(3)}
    • deep_night / late_night (midnight–6am on day D, disrupts D commute):
        workday if D ∈ {Mon(0), Tue(1), Wed(2), Thu(3), Fri(4)}

    Returns a copy of df with new columns; does NOT modify the original.
    """
    df = df.copy()
    early_night   = df["night_band"] == "early_night"
    midnight_band = df["night_band"].isin(["deep_night", "late_night"])

    early_workday    = early_night   & df["dow"].isin([0, 1, 2, 3, 6])
    midnight_workday = midnight_band & df["dow"].isin([0, 1, 2, 3, 4])
    workday_mask = early_workday | midnight_workday

    df["night_alert_workday"] = (df["night_alert"].astype(bool) & workday_mask).astype(int)
    df["night_alert_weekend"] = (df["night_alert"].astype(bool) & ~workday_mask).astype(int)

    n_wd = int(df["night_alert_workday"].sum())
    n_we = int(df["night_alert_weekend"].sum())
    log.info("Workday nights: %d | Weekend nights: %d | Total: %d",
             n_wd, n_we, n_wd + n_we)
    return df


def run_weekday_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether the AMBER alert effect is driven by workday nights.

    Instead of restricting the sample (as in run_event_study_workday),
    this specification includes BOTH night_alert_workday and night_alert_weekend
    as separate regressors in a single model:

        Y = α + β_wd·WorkdayNight + β_we·WeekendNight + γ·X + FE + ε

    β_wd: effect on commute fatalities of alerts that precede a workday
    β_we: effect on commute fatalities of alerts that precede a weekend

    Identification: both coefficients come from the same FE-cleaned residuals;
    the difference β_wd − β_we directly tests the commute-channel hypothesis.

    Implementation: fe_ols_from_panel is called twice with the two indicators
    swapped between treatment and controls positions — this is equivalent to
    a single multivariate OLS with both as regressors.

    Three FE specs (state-clustered SEs throughout):
      (I)   county + DoW×Month FE  [baseline, aligned count]
      (II)  (I) + county×year FE + lag(fatals_tm1)  [TWFE2 robustness]
      (III) (I), combined/100k WLS
    """
    import gc

    df_al = add_aligned_outcome(df)
    df_al = _make_workday_indicators(df_al)

    avail_hol = [c for c in HOLIDAY if c in df_al.columns]
    has_lag   = "fatals_tm1" in df_al.columns

    # Build county×year FE for TWFE2
    df_al["county_year_code"] = pd.Categorical(
        df_al["county_code"].astype(str) + "_" + df_al["year"].astype(str)
    ).codes.astype(np.int32)

    # Rate outcome
    if "population" in df_al.columns:
        county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
        pop = df_al["population"].fillna(county_mean_pop)
        df_al["_log_pop"] = np.log(pop.clip(lower=1))
        combined_col = "combined_next_commute" if "combined_next_commute" in df_al.columns \
                       else "fatals_next_commute"
        df_al["combined_rate"] = df_al[combined_col] / (pop / 100_000)
    else:
        df_al["_log_pop"] = None
        df_al["combined_rate"] = None

    results = []

    def _both_coefs(outcome, ctrl_base, weights_col="",
                    county=True, dm=True, extra_fes=None, tag=""):
        """Run model twice, collect β_workday and β_weekend."""
        extra = extra_fes or []
        sub = df_al.dropna(subset=[outcome]).copy()
        if weights_col and weights_col not in sub.columns:
            return

        # β_workday: treatment=workday, control=weekend (+ base controls)
        r_wd = fe_ols_from_panel(
            sub, outcome,
            treatment="night_alert_workday",
            controls=["night_alert_weekend"] + ctrl_base,
            county=county, dm=dm, extra_fes=extra,
            weights_col=weights_col, cluster_col="state_code",
            label=f"Workday night [{tag}]",
        )
        r_wd["split"] = "workday"
        results.append(r_wd)

        # β_weekend: treatment=weekend, control=workday (same model, different column order)
        r_we = fe_ols_from_panel(
            sub, outcome,
            treatment="night_alert_weekend",
            controls=["night_alert_workday"] + ctrl_base,
            county=county, dm=dm, extra_fes=extra,
            weights_col=weights_col, cluster_col="state_code",
            label=f"Weekend night [{tag}]",
        )
        r_we["split"] = "weekend"
        results.append(r_we)

    # Spec I: baseline aligned count
    log.info("Weekday interaction — (I) baseline count")
    _both_coefs("fatals_next_commute", avail_hol, tag="count, baseline")

    # Spec II: TWFE2 (county×year FE + lag fatals)
    log.info("Weekday interaction — (II) TWFE2 count")
    twfe_ctrl = avail_hol + (["fatals_tm1"] if has_lag else [])
    _both_coefs("fatals_next_commute", twfe_ctrl,
                county=False, dm=True, extra_fes=["county_year_code"],
                tag="count, TWFE2")

    # Spec III: combined/100k WLS
    if "combined_rate" in df_al.columns and df_al["combined_rate"].notna().any():
        log.info("Weekday interaction — (III) combined/100k WLS")
        _both_coefs("combined_rate", avail_hol, weights_col="_log_pop",
                    tag="comb/100k WLS")

    del df_al; gc.collect()

    out = pd.DataFrame(results)
    # Print a clean comparison table
    if not out.empty:
        wd = out[out["split"] == "workday"][["model", "coef", "se", "pval"]].copy()
        we = out[out["split"] == "weekend"][["model", "coef", "se", "pval"]].copy()
        log.info("=== Weekday vs Weekend night alert effects ===")
        log.info("Workday nights:\n%s", wd.to_string(index=False))
        log.info("Weekend nights:\n%s", we.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Daytime alert placebo
# ---------------------------------------------------------------------------

def run_daytime_placebo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Falsification: AMBER alerts issued 9am–4pm (local time) should not
    disrupt sleep and thus should have zero effect on same-day or next-day
    fatalities. Uses day_alert as treatment; county + DoW×Month FE.
    """
    if "day_alert" not in df.columns:
        log.warning("day_alert column missing — skipping daytime placebo")
        return pd.DataFrame()

    n_day = int(df["day_alert"].sum())
    log.info("Daytime alert county-days: %d", n_day)

    results = []
    for outcome, lbl in [
        ("fatals_t0", "Same-day (daytime alert)"),
        ("fatals_t1", "Next-day (daytime alert)"),
    ]:
        sub = df.dropna(subset=[outcome]).copy()
        r = fe_ols_from_panel(sub, outcome, treatment="day_alert",
                              county=True, dm=True,
                              cluster_col="state_code",
                              label=lbl)
        results.append(r)

    # Compare side-by-side with nighttime for reference
    df_al = add_aligned_outcome(df)
    r = fe_ols_from_panel(df_al, "fatals_next_commute", treatment="night_alert",
                          county=True, dm=True, cluster_col="state_code",
                          label="Next-commute (night alert) [ref]")
    results.append(r)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Event study (dynamic effects t-3 … t+3)
# ---------------------------------------------------------------------------

EVENT_STUDY_WINDOW = 5   # k = -EVENT_STUDY_WINDOW ... +EVENT_STUDY_WINDOW


def run_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamic effects k ∈ {-W,...,+W} where W = EVENT_STUDY_WINDOW (default 5).

    Four specs per k:
      (1) count          — raw fatality count, county+DoW×Month FE, holiday control
      (2) count_outDM    — (1) + outcome-date DoW×Month FE (robustness for Friday clustering)
      (3) comb_rate_logWLS      — combined (fatal+serious) per 100k, log-pop WLS
      (4) comb_rate_logWLS_outDM — (3) + outcome-date DoW×Month FE

    Memory-optimized: computes aligned outcomes in-place on df (avoiding the 2.5 GB
    copy that add_aligned_outcome() makes), uses integer _out_dm_code (avoiding ~1 GB
    of Python string allocations), and uses county_code (int) for groupby shifts
    instead of fips (object string).
    """
    import gc

    # --- Step 1: compute aligned outcome columns in-place on df ---
    # (add_aligned_outcome makes a full df.copy() — here we avoid that)
    df.sort_values(["fips", "date"], inplace=True)
    midnight_mask = df["night_band"].isin(["deep_night", "late_night"])

    df["fatals_next_commute"] = df["fatals_t1"]
    df.loc[midnight_mask, "fatals_next_commute"] = df.loc[midnight_mask, "fatals_t0"]

    has_combined = False
    if "serious_injuries" in df.columns:
        df["_sinj_t1"] = df.groupby("fips")["serious_injuries"].shift(-1).fillna(0)
        df["_sinj_nc"] = df["_sinj_t1"].copy()
        df.loc[midnight_mask, "_sinj_nc"] = df.loc[midnight_mask, "serious_injuries"]
        df["combined_next_commute"] = df["fatals_next_commute"] + df["_sinj_nc"]
        df.drop(columns=["_sinj_t1", "_sinj_nc"], inplace=True)
        has_combined = True

    has_pop = False
    if "population" in df.columns:
        county_mean_pop = df.groupby("fips")["population"].transform("mean")
        _pop = df["population"].fillna(county_mean_pop)
        df["_log_pop"] = np.log(_pop.clip(lower=1))
        df["_pop_100k"] = _pop / 100_000
        del _pop
        has_pop = True

    # --- Step 2: build lean DataFrame (only needed columns, no fips object string) ---
    # Using county_code (int) instead of fips (object) saves ~400 MB and speeds groupby.
    BASE_COLS = ["date", "county_code", "state_code",
                 "dow_month_code", "night_alert", "fatals_next_commute"]
    extra_cols = (
        (["combined_next_commute"] if has_combined else [])
        + (["_log_pop", "_pop_100k"] if has_pop else [])
        + [c for c in HOLIDAY if c in df.columns]
    )
    lean = df[BASE_COLS + extra_cols].sort_values(["county_code", "date"]).copy()
    gc.collect()

    # Rename internal columns to public names
    if has_pop:
        lean.rename(columns={"_log_pop": "log_pop", "_pop_100k": "pop_100k"}, inplace=True)
        df.drop(columns=["_log_pop", "_pop_100k"], inplace=True)

    hol_ctrl = [c for c in HOLIDAY if c in lean.columns]

    log.info("Event study: %d k-values, 4 specs each, %d obs",
             2 * EVENT_STUDY_WINDOW + 1, len(lean))

    import time as _time
    results = []
    for k in range(-EVENT_STUDY_WINDOW, EVENT_STUDY_WINDOW + 1):
        _t_k = _time.time()

        # --- Integer _out_dm_code: dayofweek ∈ {0..6} × month ∈ {1..12} ---
        # Integer arithmetic saves ~1 GB vs string concatenation on 7.2M rows.
        # Codes: 0..83  (7 days × 12 months = 84 cells)
        _t0 = _time.time()
        outcome_dates = lean["date"] + pd.Timedelta(days=k)
        lean["_out_dm_code"] = (
            (outcome_dates.dt.dayofweek * 12 + outcome_dates.dt.month - 1)
            .astype(np.int32)
        )
        del outcome_dates
        log.debug("  k=%+d setup _out_dm_code: %.2fs", k, _time.time()-_t0)

        # --- spec 1 & 2: raw count ---
        col_c = "_yk_count"
        _t0 = _time.time()
        lean[col_c] = lean.groupby("county_code")["fatals_next_commute"].shift(-k)
        sub = lean.dropna(subset=[col_c])
        log.debug("  k=%+d shift+dropna: %.2fs (%d rows)", k, _time.time()-_t0, len(sub))

        _t0 = _time.time()
        r = fe_ols_from_panel(sub, col_c, county=True, dm=True,
                              controls=hol_ctrl, cluster_col="state_code",
                              label=f"k={k:+d} [count]")
        r["k"] = k; r["spec"] = "count"; results.append(r)
        log.debug("  k=%+d spec1 OLS: %.2fs", k, _time.time()-_t0)

        _t0 = _time.time()
        r_odm = fe_ols_from_panel(sub, col_c, county=True, dm=True,
                                  controls=hol_ctrl, extra_fes=["_out_dm_code"],
                                  cluster_col="state_code",
                                  label=f"k={k:+d} [count+outDM]")
        r_odm["k"] = k; r_odm["spec"] = "count_outDM"; results.append(r_odm)
        log.debug("  k=%+d spec2 OLS+outDM: %.2fs", k, _time.time()-_t0)

        del sub
        lean.drop(columns=[col_c], inplace=True)

        # --- spec 3 & 4: combined rate, log-pop WLS ---
        if has_combined and has_pop:
            col_r = "_yk_rate"
            _t0 = _time.time()
            lean[col_r] = (lean.groupby("county_code")["combined_next_commute"].shift(-k)
                           / lean["pop_100k"])
            sub_r = lean.dropna(subset=[col_r, "log_pop"])
            log.debug("  k=%+d rate shift+dropna: %.2fs (%d rows)", k, _time.time()-_t0, len(sub_r))

            _t0 = _time.time()
            r2 = fe_ols_from_panel(sub_r, col_r, county=True, dm=True,
                                   controls=hol_ctrl, weights_col="log_pop",
                                   cluster_col="state_code",
                                   label=f"k={k:+d} [comb/100k logWLS]")
            r2["k"] = k; r2["spec"] = "comb_rate_logWLS"; results.append(r2)
            log.debug("  k=%+d spec3 WLS: %.2fs", k, _time.time()-_t0)

            _t0 = _time.time()
            r2_odm = fe_ols_from_panel(sub_r, col_r, county=True, dm=True,
                                       controls=hol_ctrl, extra_fes=["_out_dm_code"],
                                       weights_col="log_pop", cluster_col="state_code",
                                       label=f"k={k:+d} [comb/100k logWLS+outDM]")
            r2_odm["k"] = k; r2_odm["spec"] = "comb_rate_logWLS_outDM"
            results.append(r2_odm)
            log.debug("  k=%+d spec4 WLS+outDM: %.2fs", k, _time.time()-_t0)

            del sub_r
            lean.drop(columns=[col_r], inplace=True)

        lean.drop(columns=["_out_dm_code"], inplace=True)
        gc.collect()
        log.info("  Event study k=%+d done in %.1fs (%d specs)",
                 k, _time.time()-_t_k,
                 len([r for r in results if r.get("k") == k]))

    del lean
    gc.collect()
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Workday-night event study (Sun–Thu nights only)
# ---------------------------------------------------------------------------

def run_event_study_workday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Event study restricted to nights that precede a workday (Mon–Fri commute).

    The sleep-disruption → commute-fatality mechanism only applies when disrupted
    sleep precedes a workday.  The filter is BAND-SPECIFIC because early-night and
    midnight-band alerts fall on different calendar days:

    • early_night  (10pm–midnight on calendar date D, disrupts morning of D+1):
        keep if D+1 is Mon–Fri → D ∈ {Sun(6), Mon(0), Tue(1), Wed(2), Thu(3)}
        e.g. Thu 10pm–midnight → disrupts Friday ✓ ; Fri 10pm–midnight → Saturday ✗

    • deep_night / late_night  (midnight–6am on date D, disrupts morning of D):
        keep if D is Mon–Fri → D ∈ {Mon(0), Tue(1), Wed(2), Thu(3), Fri(4)}
        e.g. Fri midnight–6am → disrupts Friday ✓ ; Sat midnight–6am → Saturday ✗

    Combined, the "workday night" covers Thu 10pm → Fri 6am, Fri 10pm → Sat 6am
    is excluded, and so on.  (dow encoding: 0=Mon, …, 4=Fri, 5=Sat, 6=Sun)

    Four specs per k:
      count      — raw count, county+DoW×Month FE, holiday, state-clustered SEs
      count_twfe — (above) + county×year FE + lag(fatals_tm1)  [TWFE2 robustness]
      comb_rate  — combined (fatal+serious)/100k, log-pop WLS, state-clustered
      comb_twfe  — (above) + county×year FE + lag(fatals_tm1)

    Memory-optimised: same lean-DataFrame + in-place shift approach as
    run_event_study.
    """
    import gc
    import time as _time

    # ------------------------------------------------------------------
    # Step 0: redefine treatment — keep alert only on workday nights
    # ------------------------------------------------------------------
    # dow: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    early_night   = df["night_band"] == "early_night"
    midnight_band = df["night_band"].isin(["deep_night", "late_night"])

    # early_night (10pm–midnight on D) disrupts D+1 → include if D∈{Sun,Mon,Tue,Wed,Thu}
    early_workday    = early_night   & df["dow"].isin([0, 1, 2, 3, 6])
    # midnight band (midnight–6am on D) disrupts D → include if D∈{Mon,Tue,Wed,Thu,Fri}
    midnight_workday = midnight_band & df["dow"].isin([0, 1, 2, 3, 4])

    workday_nights = early_workday | midnight_workday

    orig_night_alert = df["night_alert"].copy()
    df["night_alert"] = (df["night_alert"].astype(bool) & workday_nights).astype(int)

    n_treated = int(df["night_alert"].sum())
    n_dropped  = int(orig_night_alert.sum()) - n_treated
    log.info("Workday-night filter: %d treated county-days kept, %d non-workday dropped",
             n_treated, n_dropped)

    # ------------------------------------------------------------------
    # Step 1: aligned outcome columns in-place
    # ------------------------------------------------------------------
    df.sort_values(["fips", "date"], inplace=True)
    midnight_mask = df["night_band"].isin(["deep_night", "late_night"])

    df["fatals_next_commute"] = df["fatals_t1"]
    df.loc[midnight_mask, "fatals_next_commute"] = df.loc[midnight_mask, "fatals_t0"]

    has_combined = False
    if "serious_injuries" in df.columns:
        df["_sinj_t1"] = df.groupby("fips")["serious_injuries"].shift(-1).fillna(0)
        df["_sinj_nc"] = df["_sinj_t1"].copy()
        df.loc[midnight_mask, "_sinj_nc"] = df.loc[midnight_mask, "serious_injuries"]
        df["combined_next_commute"] = df["fatals_next_commute"] + df["_sinj_nc"]
        df.drop(columns=["_sinj_t1", "_sinj_nc"], inplace=True)
        has_combined = True

    has_pop = False
    if "population" in df.columns:
        county_mean_pop = df.groupby("fips")["population"].transform("mean")
        _pop = df["population"].fillna(county_mean_pop)
        df["_log_pop"] = np.log(_pop.clip(lower=1))
        df["_pop_100k"] = _pop / 100_000
        del _pop
        has_pop = True

    # ------------------------------------------------------------------
    # Step 2: county×year FE for TWFE2 specs
    # ------------------------------------------------------------------
    df["county_year_code"] = pd.Categorical(
        df["county_code"].astype(str) + "_" + df["year"].astype(str)
    ).codes.astype(np.int32)

    # ------------------------------------------------------------------
    # Step 3: lean DataFrame
    # ------------------------------------------------------------------
    BASE_COLS = ["date", "county_code", "state_code",
                 "dow_month_code", "night_alert", "fatals_next_commute",
                 "county_year_code"]
    hol_cols  = [c for c in HOLIDAY if c in df.columns]
    extra_cols = (
        (["combined_next_commute"] if has_combined else [])
        + (["_log_pop", "_pop_100k"] if has_pop else [])
        + hol_cols
        + (["fatals_tm1"] if "fatals_tm1" in df.columns else [])
    )
    lean = df[BASE_COLS + extra_cols].sort_values(["county_code", "date"]).copy()
    gc.collect()

    if has_pop:
        lean.rename(columns={"_log_pop": "log_pop", "_pop_100k": "pop_100k"}, inplace=True)
        df.drop(columns=["_log_pop", "_pop_100k"], inplace=True)

    # Restore original treatment on df (so caller's df is unchanged)
    df["night_alert"] = orig_night_alert
    del orig_night_alert, midnight_mask, workday_nights
    del early_night, midnight_band, early_workday, midnight_workday

    hol_ctrl  = [c for c in HOLIDAY if c in lean.columns]
    has_lag   = "fatals_tm1" in lean.columns
    twfe_ctrl = hol_ctrl + (["fatals_tm1"] if has_lag else [])

    log.info("Workday event study: %d k-values × 4 specs, %d obs",
             2 * EVENT_STUDY_WINDOW + 1, len(lean))

    results = []
    for k in range(-EVENT_STUDY_WINDOW, EVENT_STUDY_WINDOW + 1):
        _t_k = _time.time()

        # Outcome-date DoW×Month integer code (same trick as run_event_study)
        outcome_dates = lean["date"] + pd.Timedelta(days=k)
        lean["_out_dm_code"] = (
            (outcome_dates.dt.dayofweek * 12 + outcome_dates.dt.month - 1)
            .astype(np.int32)
        )
        del outcome_dates

        # ---- Spec A: count, baseline FE ----
        col_c = "_yk_count"
        lean[col_c] = lean.groupby("county_code")["fatals_next_commute"].shift(-k)
        sub = lean.dropna(subset=[col_c])

        r = fe_ols_from_panel(sub, col_c, county=True, dm=True,
                              controls=hol_ctrl, cluster_col="state_code",
                              label=f"k={k:+d} [count]")
        r["k"] = k; r["spec"] = "count"; results.append(r)

        # ---- Spec B: count, TWFE2 (county×year FE + lag fatals) ----
        r_tw = fe_ols_from_panel(sub, col_c, county=False, dm=True,
                                 extra_fes=["county_year_code"],
                                 controls=twfe_ctrl, cluster_col="state_code",
                                 label=f"k={k:+d} [count+TWFE2]")
        r_tw["k"] = k; r_tw["spec"] = "count_twfe2"; results.append(r_tw)

        del sub
        lean.drop(columns=[col_c], inplace=True)

        # ---- Spec C & D: combined rate, log-pop WLS ----
        if has_combined and has_pop:
            col_r = "_yk_rate"
            lean[col_r] = (lean.groupby("county_code")["combined_next_commute"].shift(-k)
                           / lean["pop_100k"])
            sub_r = lean.dropna(subset=[col_r, "log_pop"])

            r2 = fe_ols_from_panel(sub_r, col_r, county=True, dm=True,
                                   controls=hol_ctrl, weights_col="log_pop",
                                   cluster_col="state_code",
                                   label=f"k={k:+d} [comb/100k logWLS]")
            r2["k"] = k; r2["spec"] = "comb_rate_logWLS"; results.append(r2)

            r2_tw = fe_ols_from_panel(sub_r, col_r, county=False, dm=True,
                                      extra_fes=["county_year_code"],
                                      controls=twfe_ctrl, weights_col="log_pop",
                                      cluster_col="state_code",
                                      label=f"k={k:+d} [comb/100k logWLS+TWFE2]")
            r2_tw["k"] = k; r2_tw["spec"] = "comb_rate_logWLS_twfe2"; results.append(r2_tw)

            del sub_r
            lean.drop(columns=[col_r], inplace=True)

        lean.drop(columns=["_out_dm_code"], inplace=True)
        gc.collect()
        log.info("  Workday event study k=%+d done in %.1fs",
                 k, _time.time() - _t_k)

    del lean
    gc.collect()
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Population-reached as continuous treatment dosage
# ---------------------------------------------------------------------------

def run_population_dosage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace binary night_alert with log(total population reached by alert).
    Each alert reaches the sum of populations of all counties it covers.
    For county c on night t, dosage = log(pop_reached) if treated, 0 otherwise.

    This leverages the finding that broad alerts drive the effect: the dose
    scales with people woken up, consistent with a mass sleep-disruption channel.
    """
    if "alert_breadth" not in df.columns or not AMBER_CLEAN_PATH.exists():
        log.warning("Alert breadth / amber data missing — skipping dosage spec")
        return pd.DataFrame()

    if "population" not in df.columns:
        log.warning("Population missing — skipping dosage spec")
        return pd.DataFrame()

    # Build alert-level total population reached
    am = pd.read_parquet(AMBER_CLEAN_PATH)
    am["date"] = pd.to_datetime(am["issued_local"].dt.date)

    # We need county populations — use the panel's population column
    county_pop = (
        df.groupby("fips")["population"]
        .mean()
        .reset_index()
        .rename(columns={"population": "county_pop"})
    )
    am = am.merge(county_pop, left_on="county_fips", right_on="fips", how="left")

    # Total population per alert
    alert_pop = (
        am[am["is_night"]]
        .groupby("alert_id")["county_pop"]
        .sum()
        .rename("pop_reached")
    )
    am_night = am[am["is_night"]].merge(alert_pop, on="alert_id")

    # For each (county, night), sum pop_reached across all alerts
    dosage = (
        am_night.groupby(["county_fips", "date"])["pop_reached"]
        .sum()
        .reset_index()
        .rename(columns={"county_fips": "fips", "pop_reached": "pop_reached_total"})
    )

    df2 = df.merge(dosage, on=["fips", "date"], how="left")
    df2["pop_reached_total"] = df2["pop_reached_total"].fillna(0)
    df2["log_pop_reached"] = np.log(df2["pop_reached_total"].clip(lower=1))
    # Zero out log for untreated rows
    df2.loc[df2["night_alert"] == 0, "log_pop_reached"] = 0

    n_nonzero = int((df2["log_pop_reached"] > 0).sum())
    log.info("Population dosage: %d treated county-days, mean log_pop_reached=%.2f",
             n_nonzero,
             df2.loc[df2["log_pop_reached"] > 0, "log_pop_reached"].mean())

    df_al = add_aligned_outcome(df2)
    county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
    pop = df_al["population"].fillna(county_mean_pop)
    df_al["combined_rate"] = (
        df_al.get("combined_next_commute", df_al["fatals_next_commute"])
        / (pop / 100_000)
    )

    results = []
    for outcome, tag in [
        ("fatals_next_commute", "count"),
        ("combined_rate",       "comb/100k"),
    ]:
        sub = df_al.dropna(subset=[outcome])
        r = fe_ols_from_panel(sub, outcome, treatment="log_pop_reached",
                              county=True, dm=True,
                              cluster_col="state_code",
                              label=f"log(pop reached) [{tag}]")
        results.append(r)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Narrow-alert sensitivity  (drop broad / statewide alerts)
# ---------------------------------------------------------------------------

def run_narrow_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict treatment to alerts covering ≤K counties.

    Broad (statewide) alerts are very diffuse treatments that add noise.
    This sweeps K ∈ {5, 10, 20, unrestricted} and runs the baseline spec
    on both raw count and combined rate outcomes.
    """
    df_al = add_aligned_outcome(df)

    county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
    pop = df_al["population"].fillna(county_mean_pop)
    df_al["combined_rate"] = (
        df_al.get("combined_next_commute", df_al["fatals_next_commute"])
        / (pop / 100_000)
    )

    results = []
    thresholds = [5, 10, 20, 9999]   # 9999 = unrestricted

    for k in thresholds:
        label_k = f"≤{k}" if k < 9999 else "all"
        df_k = df_al.copy()
        # Re-define night_alert: only flag county-days where breadth ≤ k
        if k < 9999:
            df_k["night_alert"] = (
                (df_k["night_alert"] == 1) & (df_k["alert_breadth"] <= k)
            ).astype(int)
        n_treated = int(df_k["night_alert"].sum())
        log.info("Breadth ≤%s: %d treated county-days", label_k, n_treated)

        for outcome, tag in [
            ("fatals_next_commute", "count"),
            ("combined_rate",       "comb/100k"),
        ]:
            sub = df_k.dropna(subset=[outcome])
            r = fe_ols_from_panel(sub, outcome, county=True, dm=True,
                                  cluster_col="state_code",
                                  label=f"Breadth {label_k} [{tag}]")
            r["breadth_threshold"] = k if k < 9999 else None
            r["n_treated"] = n_treated
            results.append(r)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# State-level clustering robustness
# ---------------------------------------------------------------------------

def run_state_clustered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run the three core baseline specs clustering at state (not county) level.
    Statewide AMBER alerts mean treated counties within a state are correlated,
    so county-level clustering may understate SEs.  State clustering is more
    conservative and accounts for this.
    """
    df_al = add_aligned_outcome(df)
    results = []
    for label, kwargs in [
        ("(2) County FE [state-cl]",   dict(county=True, dm=False)),
        ("(3) Baseline [state-cl]",    dict(county=True, dm=True)),
        ("(5) + Year FE [state-cl]",   dict(county=True, dm=True,
                                            extra_fes=["year_code"])),
    ]:
        log.info("%s", label)
        results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                         cluster_col="state_code",
                                         label=label, **kwargs))

    # Also combined rate with state clustering
    county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
    pop = df_al["population"].fillna(county_mean_pop)
    df_al["combined_rate"] = (
        df_al.get("combined_next_commute", df_al["fatals_next_commute"])
        / (pop / 100_000)
    )
    df_al = df_al.dropna(subset=["combined_rate"])
    for label, kwargs in [
        ("(2) County FE comb [state-cl]",  dict(county=True, dm=False)),
        ("(3) Baseline comb [state-cl]",   dict(county=True, dm=True)),
    ]:
        log.info("%s", label)
        results.append(fe_ols_from_panel(df_al, "combined_rate",
                                         cluster_col="state_code",
                                         label=label, **kwargs))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def run_threshold_sensitivity(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run the baseline spec at different county fatality thresholds (0, 1, 3, 5, 10, 20).
    Shows how coefficients and SEs vary with sample restriction.
    Outcome: fatals_next_commute (raw count).
    """
    import gc
    thresholds = [0, 1, 3, 5, 10, 20]
    results = []
    for thr in thresholds:
        df_t = prep_panel(df_raw.copy(), min_fatals=thr)
        df_al = add_aligned_outcome(df_t)
        del df_t; gc.collect()   # free before computing rate columns

        # Rate outcome
        county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
        pop = df_al["population"].fillna(county_mean_pop)
        df_al["fatals_rate_next_commute"] = df_al["fatals_next_commute"] / (pop / 100_000)
        df_al["log_pop"] = np.log(pop.clip(lower=1))
        df_al = df_al.dropna(subset=["fatals_rate_next_commute"])

        n_counties = df_al["fips"].nunique()
        n_treated  = int(df_al["night_alert"].sum())
        log.info("Threshold ≥%d: %d counties, %d treated county-days",
                 thr, n_counties, n_treated)

        for outcome, tag in [
            ("fatals_next_commute",      "count"),
            ("fatals_rate_next_commute", "rate/100k"),
        ]:
            r = fe_ols_from_panel(df_al, outcome, county=True, dm=True,
                                  label=f"≥{thr}/yr [{tag}]")
            r["threshold"]  = thr
            r["n_counties"] = n_counties
            r["n_treated"]  = n_treated
            results.append(r)

        del df_al; gc.collect()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import gc
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

    # Load full panel (unfiltered) and apply restriction.
    # We immediately drop df_raw to free ~3 GB; it is reloaded only when
    # run_threshold_sensitivity needs the unrestricted panel at the very end.
    panel_path = DATA_PROC / "panel_county_day.parquet"
    df_raw = load_panel()
    df = prep_panel(df_raw.copy())
    del df_raw; gc.collect()   # free ~3 GB before the analysis loop

    note = ("SE clustered by county. County FE and DoW$\\times$Month FE absorbed "
            "via pyhdfe. *** $p{<}$0.01, ** $p{<}$0.05, * $p{<}$0.10.")

    log.info("=== BASELINE ===")
    base = run_baseline(df)
    log.info("\n%s", base[["model","coef","se","pval","n_obs"]].to_string(index=False))
    base.to_csv(OUTPUT_TABS / "reg_baseline.csv", index=False)
    (OUTPUT_TABS / "reg_baseline.tex").write_text(
        to_latex(base,
                 "Effect of Nighttime AMBER Alert on Traffic Fatalities "
                 "(Timing-Aligned Outcome)",
                 note + " Outcome is \\textit{fatals\\_next\\_commute}: "
                        "fatals$_{t+0}$ for midnight--6am alerts, fatals$_{t+1}$ otherwise."))

    log.info("=== RATE OUTCOME (per 100k population) ===")
    rate = run_rate_baseline(df)
    if not rate.empty:
        log.info("\n%s", rate[["model","coef","se","pval","n_obs"]].to_string(index=False))
        rate.to_csv(OUTPUT_TABS / "reg_rate.csv", index=False)
        (OUTPUT_TABS / "reg_rate.tex").write_text(
            to_latex(rate,
                     "Effect of Nighttime AMBER Alert on Traffic Fatality Rate "
                     "(per 100,000 Population)",
                     note + " Outcome is \\textit{fatals\\_rate\\_next\\_commute}: "
                            "fatalities per 100,000 county population, timing-aligned."))
    del rate; gc.collect()

    log.info("=== SE REDUCTION (A: restrict, B: WLS, C: A+B) ===")
    se_red = run_se_reduction(df)
    if not se_red.empty:
        log.info("\n%s", se_red[["model","coef","se","pval","n_obs"]].to_string(index=False))
        se_red.to_csv(OUTPUT_TABS / "reg_se_reduction.csv", index=False)
        (OUTPUT_TABS / "reg_se_reduction.tex").write_text(
            to_latex(se_red,
                     "SE Reduction: Sample Restriction and WLS "
                     "(Rate Outcome, per 100k)",
                     note + " (A) counties with $\\geq$5 mean annual fatalities; "
                            "(B) WLS weighted by county population; "
                            "(C) both combined."))
    del se_red; gc.collect()

    log.info("=== COMBINED OUTCOME (fatalities + serious injuries) ===")
    comb = run_combined(df)
    if not comb.empty:
        log.info("\n%s", comb[["model","coef","se","pval","n_obs"]].to_string(index=False))
        comb.to_csv(OUTPUT_TABS / "reg_combined.csv", index=False)
        (OUTPUT_TABS / "reg_combined.tex").write_text(
            to_latex(comb,
                     "Combined Outcome: Fatalities + Serious Injuries per 100k",
                     note + " Serious injuries defined as INJ\\_SEV=3 in FARS person "
                            "file (incapacitating injuries in fatal crashes)."))
    del comb; gc.collect()

    log.info("=== WEEKDAY / WEEKEND INTERACTION ===")
    wd_int = run_weekday_interaction(df)
    if not wd_int.empty:
        log.info("\n%s", wd_int[["model","split","coef","se","pval","n_obs"]].to_string(index=False))
        wd_int.to_csv(OUTPUT_TABS / "reg_weekday_interaction.csv", index=False)
        (OUTPUT_TABS / "reg_weekday_interaction.tex").write_text(
            to_latex(wd_int,
                     "Weekday vs Weekend Night Alert Effects "
                     "(Interaction with Commute-Day)',",
                     note + " \\textit{Workday night}: alert precedes Mon--Fri commute. "
                            "\\textit{Weekend night}: alert precedes Sat--Sun. "
                            "Both regressors included simultaneously. "
                            "SEs clustered at state level."))
    del wd_int; gc.collect()

    log.info("=== ALIGNED (timing-corrected) ===")
    aligned = run_aligned(df)
    log.info("\n%s", aligned[["model","coef","se","pval","n_obs"]].to_string(index=False))
    aligned.to_csv(OUTPUT_TABS / "reg_aligned.csv", index=False)
    (OUTPUT_TABS / "reg_aligned.tex").write_text(
        to_latex(aligned,
                 "Timing-Corrected Estimates: Outcome Aligned to Disrupted Commute",
                 note + " Specification (C)/(D) uses \\textit{fatals\\_next\\_commute}: "
                        "fatals$_{t+0}$ for midnight--6am alerts, fatals$_{t+1}$ otherwise."))

    log.info("=== HETEROGENEITY ===")
    het = run_heterogeneity(df)
    log.info("\n%s", het[["model","coef","se","pval","n_obs"]].to_string(index=False))
    het.to_csv(OUTPUT_TABS / "reg_hetero.csv", index=False)
    (OUTPUT_TABS / "reg_hetero.tex").write_text(to_latex(het, "Heterogeneity Analysis", note))
    del het; gc.collect()

    log.info("=== PLACEBO ===")
    plac = run_placebo(df)
    log.info("\n%s", plac[["model","coef","se","pval","n_obs"]].to_string(index=False))
    plac.to_csv(OUTPUT_TABS / "reg_placebo.csv", index=False)
    (OUTPUT_TABS / "reg_placebo.tex").write_text(to_latex(plac, "Placebo Tests", note))
    del plac; gc.collect()

    log.info("=== DAYTIME ALERT PLACEBO ===")
    day_plac = run_daytime_placebo(df)
    if not day_plac.empty:
        log.info("\n%s", day_plac[["model","coef","se","pval","n_obs"]].to_string(index=False))
        day_plac.to_csv(OUTPUT_TABS / "reg_daytime_placebo.csv", index=False)
        (OUTPUT_TABS / "reg_daytime_placebo.tex").write_text(
            to_latex(day_plac, "Falsification: Daytime AMBER Alert Has No Effect",
                     note + " Day alert = issued 9am--4pm local time. "
                            "Night alert shown for comparison."))

    log.info("=== EVENT STUDY (t-3 to t+3) ===")
    evs = run_event_study(df)
    if not evs.empty:
        log.info("\n%s", evs[["k","model","coef","se","pval"]].to_string(index=False))
        evs.to_csv(OUTPUT_TABS / "reg_event_study.csv", index=False)

    log.info("=== WORKDAY-NIGHT EVENT STUDY (Sun–Thu only) ===")
    evs_wd = run_event_study_workday(df)
    if not evs_wd.empty:
        log.info("\n%s", evs_wd[["k","model","coef","se","pval"]].to_string(index=False))
        evs_wd.to_csv(OUTPUT_TABS / "reg_event_study_workday.csv", index=False)
    del evs_wd; gc.collect()

    log.info("=== POPULATION DOSAGE ===")
    dosage = run_population_dosage(df)
    if not dosage.empty:
        log.info("\n%s", dosage[["model","coef","se","pval","n_obs"]].to_string(index=False))
        dosage.to_csv(OUTPUT_TABS / "reg_dosage.csv", index=False)
        (OUTPUT_TABS / "reg_dosage.tex").write_text(
            to_latex(dosage,
                     "Population Dosage: log(Total Population Reached by Alert)",
                     note + " Treatment is log(sum of county populations covered by "
                            "nighttime alerts). SEs clustered at state level."))

    log.info("=== NARROW ALERT SENSITIVITY ===")
    narrow = run_narrow_alerts(df)
    log.info("\n%s", narrow[["model","n_treated","coef","se","pval"]].to_string(index=False))
    narrow.to_csv(OUTPUT_TABS / "reg_narrow_alerts.csv", index=False)
    (OUTPUT_TABS / "reg_narrow_alerts.tex").write_text(
        to_latex(narrow,
                 "Robustness: Restricting to Narrowly-Targeted Alerts",
                 note + " Treatment restricted to alerts covering $\\leq K$ counties. "
                        "SEs clustered at state level."))

    log.info("=== STATE-LEVEL CLUSTERING ===")
    state_cl = run_state_clustered(df)
    log.info("\n%s", state_cl[["model","coef","se","pval","n_obs"]].to_string(index=False))
    state_cl.to_csv(OUTPUT_TABS / "reg_state_clustered.csv", index=False)
    (OUTPUT_TABS / "reg_state_clustered.tex").write_text(
        to_latex(state_cl, "Robustness: State-Level Clustering",
                 note.replace("county", "state") +
                 " SEs clustered at state level to account for correlated treatment "
                 "within states (statewide AMBER alerts)."))

    log.info("=== THRESHOLD SENSITIVITY ===")
    # Free the restricted panel before reloading the full panel (saves ~1.5 GB)
    del df; gc.collect()
    df_raw = load_panel()
    thresh = run_threshold_sensitivity(df_raw)
    del df_raw; gc.collect()
    log.info("\n%s",
             thresh[["model","threshold","n_counties","n_treated","coef","se","pval"]]
             .to_string(index=False))
    thresh.to_csv(OUTPUT_TABS / "reg_thresholds.csv", index=False)

    log.info("Done. Results saved to output/tables/")


if __name__ == "__main__":
    main()
