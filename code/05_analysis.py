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

log = get_logger("05_analysis")
warnings.filterwarnings("ignore")

WEATHER = ["prcp_mm", "tmax_c"]
SERIOUS_INJ_PATH = DATA_PROC / "fars_serious_injuries.parquet"


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
    Two SE-reduction strategies, both using fatals_rate_next_commute:

    (A) Restrict to counties with ≥5 mean annual fatalities — removes zero-heavy
        small counties that contribute noise but no signal.
    (B) WLS weighted by county population — downweights small counties while
        keeping them in sample. County + DoW×Month FE absorbed via weighted
        alternating projections.
    (C) A + B combined: restricted sample with population weights.
    """
    df_al = add_aligned_outcome(df)
    if "fatals_rate_next_commute" not in df_al.columns:
        log.warning("Population data missing — skipping SE reduction specs.")
        return pd.DataFrame()

    df_al = df_al.dropna(subset=["fatals_rate_next_commute", "population"])
    results = []

    # --- Baseline rate (unrestricted, unweighted) for reference ---
    log.info("SE reduction: baseline rate (unrestricted)")
    for r in _rate_specs(df_al, "fatals_rate_next_commute", "rate"):
        results.append(r)

    # --- (A) Restrict: counties with ≥5 mean annual fatalities ---
    log.info("SE reduction: (A) county restriction ≥5 fatals/year")
    min_fatals = 5
    mean_annual = (
        df_al.groupby(["fips", "year"])["fatals_t0"].sum()
        .groupby("fips").mean()
    )
    keep_fips = mean_annual[mean_annual >= min_fatals].index
    df_A = df_al[df_al["fips"].isin(keep_fips)].copy()
    log.info("  Kept %d / %d counties (%.0f%% of rows)",
             len(keep_fips), df_al["fips"].nunique(),
             100 * len(df_A) / len(df_al))
    for r in _rate_specs(df_A, "fatals_rate_next_commute", "A:≥5/yr"):
        results.append(r)

    # --- (B) WLS weighted by population ---
    log.info("SE reduction: (B) WLS population weights")
    for r in _rate_specs(df_al, "fatals_rate_next_commute", "B:WLS",
                         weights_col="population"):
        results.append(r)

    # --- (C) A + B ---
    log.info("SE reduction: (C) restricted + WLS")
    for r in _rate_specs(df_A, "fatals_rate_next_commute", "C:A+B",
                         weights_col="population"):
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
    Uses timing-aligned combined_next_commute / (population / 100k).
    Runs the three core FE specs with and without sample restriction.
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
    for r in _rate_specs(df_al, "combined_rate", "comb"):
        results.append(r)

    # Also run with sample restriction ≥5 fatals/year
    mean_annual = (
        df_al.groupby(["fips", "year"])["fatals_t0"].sum()
        .groupby("fips").mean()
    )
    keep = mean_annual[mean_annual >= 5].index
    df_R = df_al[df_al["fips"].isin(keep)].copy()
    log.info("Combined restricted: %d counties", len(keep))
    for r in _rate_specs(df_R, "combined_rate", "comb:≥5/yr"):
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    df = prep_panel(df)

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

    log.info("Done. Results saved to output/tables/")


if __name__ == "__main__":
    main()
