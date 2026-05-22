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

WEATHER = ["prcp_mm", "tmax_c"]
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
    avail_w = [c for c in WEATHER if df_al[c].notna().mean() > 0.01]
    results = []

    log.info("(1) Pooled OLS [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=False, dm=False,
                                     label="(1) Pooled OLS"))
    log.info("(2) County FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=False,
                                     label="(2) County FE"))
    log.info("(3) County FE + DoW×Month FE  [baseline, aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=True,
                                     label="(3) Baseline"))
    if avail_w:
        log.info("(4) Baseline + weather [aligned]")
        results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                         controls=avail_w, county=True, dm=True,
                                         label="(4) + Weather"))
    else:
        log.warning("Weather controls sparse — skipping model (4).")

    log.info("(5) Baseline + Year FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=True,
                                     extra_fes=["year_code"],
                                     label="(5) + Year FE"))
    log.info("(6) County FE + DoW×Year FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=False,
                                     extra_fes=["dow_year_code"],
                                     label="(6) DoW×Year FE"))
    log.info("(7) County FE + Year×Month FE [aligned]")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
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

    avail_w = [c for c in WEATHER if df_al[c].notna().mean() > 0.01]
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
                                         label=label, **kwargs))
    if avail_w:
        log.info("(4) Baseline + weather [rate]")
        results.append(fe_ols_from_panel(df_al, "fatals_rate_next_commute",
                                         controls=avail_w, county=True, dm=True,
                                         label="(4) + Weather"))

    return pd.DataFrame(results)


def _rate_specs(df_al: pd.DataFrame, outcome: str, tag: str,
                weights_col: str = "") -> list:
    """Run the three core FE specs on a rate outcome; return list of result dicts."""
    results = []
    for label, kwargs in [
        (f"(2) County FE [{tag}]",   dict(county=True, dm=False)),
        (f"(3) Baseline [{tag}]",    dict(county=True, dm=True)),
        (f"(5) + Year FE [{tag}]",   dict(county=True, dm=True,
                                          extra_fes=["year_code"])),
    ]:
        results.append(fe_ols_from_panel(df_al, outcome, label=label,
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

    (A) Early-night only (10pm–midnight) → fatals_t1
    (B) Midnight–6am only               → fatals_t0
    (C) Pooled aligned (fatals_next_commute): fatals_t0 for midnight–6am treated
        rows, fatals_t1 for all others.  This is the preferred combined estimate.
    """
    results = []

    # (A) Early-night alerts: next-day outcome is correct
    log.info("(A) Aligned: early_night → fatals_t1")
    sub_a = df.copy()
    sub_a["night_alert"] = (sub_a["night_band"] == "early_night").astype(int)
    results.append(fe_ols_from_panel(sub_a, "fatals_t1", county=True, dm=True,
                                     label="(A) Early-night → t+1"))

    # (B) Midnight–6am alerts: same-day outcome is correct
    log.info("(B) Aligned: midnight-6am → fatals_t0")
    sub_b = df.copy()
    sub_b["night_alert"] = (sub_b["night_band"].isin(
        ["deep_night", "late_night"])).astype(int)
    results.append(fe_ols_from_panel(sub_b, "fatals_t0", county=True, dm=True,
                                     label="(B) Midnight-6am → t+0"))

    # (C) Pooled aligned: fatals_next_commute
    log.info("(C) Aligned: pooled (fatals_next_commute)")
    df_al = add_aligned_outcome(df)
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=True,
                                     label="(C) Pooled aligned"))

    # (D) Same as (C) but with year FE added for robustness
    log.info("(D) Aligned pooled + Year FE")
    results.append(fe_ols_from_panel(df_al, "fatals_next_commute",
                                     county=True, dm=True,
                                     extra_fes=["year_code"],
                                     label="(D) Pooled aligned + Year FE"))

    # Memo: misaligned baseline for direct comparison
    log.info("(M) Memo: misaligned baseline (fatals_t1, all night)")
    results.append(fe_ols_from_panel(df, "fatals_t1", county=True, dm=True,
                                     label="(M) Misaligned baseline [memo]"))

    return pd.DataFrame(results)


def run_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heterogeneity analysis using timing-aligned outcomes per band.

    Night bands: early_night → fatals_t1 (next-day commute)
                 deep_night / late_night → fatals_t0 (same-day commute)
    Weekday/weekend and alert-hour splits use fatals_next_commute.
    """
    df_al = add_aligned_outcome(df)
    results = []

    for band, outcome in [
        ("early_night", "fatals_t1"),
        ("deep_night",  "fatals_t0"),
        ("late_night",  "fatals_t0"),
    ]:
        sub = df_al.copy()
        sub["night_alert"] = (sub["night_band"] == band).astype(int)
        results.append(fe_ols_from_panel(sub, outcome,
                                         county=True, dm=True,
                                         label=f"Band: {band}"))

    for lbl, mask in [
        ("Weekday", df_al["dow"].isin([0, 1, 2, 3])),
        ("Weekend", df_al["dow"].isin([4, 5, 6])),
    ]:
        results.append(fe_ols_from_panel(df_al[mask], "fatals_next_commute",
                                         county=True, dm=True, label=lbl))

    for lbl, hrs, outcome in [
        ("Alert 10pm-midnight", list(range(22, 24)), "fatals_t1"),
        ("Alert midnight-3am",  list(range(0,  3)),  "fatals_t0"),
        ("Alert 3am-6am",       list(range(3,  6)),  "fatals_t0"),
    ]:
        sub = df_al.copy()
        sub["night_alert"] = (
            df_al["night_alert"].astype(bool) & df_al["alert_hour"].isin(hrs)
        ).astype(int)
        results.append(fe_ols_from_panel(sub, outcome,
                                         county=True, dm=True, label=lbl))

    return pd.DataFrame(results)


def run_placebo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placebo tests using aligned outcomes.
    t-1 and t+2 should show no effect; aligned main spec is included for reference.
    """
    df_al = add_aligned_outcome(df)
    results = []
    for outcome, lbl in [
        ("fatals_tm1",          "Placebo: t-1"),
        ("fatals_next_commute", "Main: aligned"),
        ("fatals_t2",           "Placebo: t+2"),
    ]:
        if outcome not in df_al.columns:
            continue
        sub = df_al.dropna(subset=[outcome]).copy()
        results.append(fe_ols_from_panel(sub, outcome,
                                         county=True, dm=True, label=lbl))
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

def run_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamic effects using timing-aligned outcome (fatals_next_commute).

    For each k ∈ {-3,...,+3} we shift fatals_next_commute by k days within
    each county, so k=0 is the aligned disrupted-commute morning, k=-1 is
    the morning before, k=+1 is the morning after, etc.

    Pre-trends (k < 0) should be flat; effect should peak at k=0.

    Memory note: we work on a lean DataFrame (only FE-identifier columns +
    treatment + shifted outcome) to avoid OOM when copying within the loop.
    """
    # Build aligned outcome, then slim down to only what fe_ols needs
    df_al = add_aligned_outcome(df)
    FE_COLS = ["fips", "date", "county_code", "state_code",
               "dow_month_code", "night_alert", "fatals_next_commute"]
    lean = df_al[FE_COLS].sort_values(["fips", "date"]).copy()
    del df_al  # free ~3 GB

    results = []
    for k in range(-3, 4):
        col = f"aligned_k{k:+d}"
        # shift(-k): positive k → look forward (future commute), negative → pre-trend
        lean[col] = lean.groupby("fips")["fatals_next_commute"].shift(-k)
        sub = lean.dropna(subset=[col])
        r = fe_ols_from_panel(sub, col, county=True, dm=True,
                              cluster_col="state_code",
                              label=f"k={k:+d}")
        r["k"] = k
        results.append(r)
        lean.drop(columns=[col], inplace=True)   # keep lean footprint
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
    thresholds = [0, 1, 3, 5, 10, 20]
    results = []
    for thr in thresholds:
        df_t = prep_panel(df_raw.copy(), min_fatals=thr)
        df_al = add_aligned_outcome(df_t)

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

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

    df_raw = load_panel()   # unfiltered — used for threshold sensitivity sweep
    df = prep_panel(df_raw.copy())

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

    log.info("=== PLACEBO ===")
    plac = run_placebo(df)
    log.info("\n%s", plac[["model","coef","se","pval","n_obs"]].to_string(index=False))
    plac.to_csv(OUTPUT_TABS / "reg_placebo.csv", index=False)
    (OUTPUT_TABS / "reg_placebo.tex").write_text(to_latex(plac, "Placebo Tests", note))

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
    thresh = run_threshold_sensitivity(df_raw)
    log.info("\n%s",
             thresh[["model","threshold","n_counties","n_treated","coef","se","pval"]]
             .to_string(index=False))
    thresh.to_csv(OUTPUT_TABS / "reg_thresholds.csv", index=False)

    log.info("Done. Results saved to output/tables/")


if __name__ == "__main__":
    main()
