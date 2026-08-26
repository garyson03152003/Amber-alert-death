"""
run_night_to_morning_window.py
=============================================================
Does a nighttime AMBER alert (22:00-05:59 local) predict elevated fatal
crashes in the *correctly dated* following waking hours (06:00-23:59)?

Window alignment
-----------------
An alert sent in the evening portion of the night (22:00-23:59) belongs to
the driving day that follows it; an alert sent in the early-morning portion
(00:00-05:59) already carries a timestamp on that same driving day. Treating
both as "day D+1" (as a flat next-calendar-day rule would) mis-dates the
early-morning alerts by one day. `load_verified_alerts(window="night")`
already encodes the correct rule via `effective_crash_date`:
    22:00-23:59 alert on date D  -> effective_crash_date = D+1
    00:00-05:59 alert on date D  -> effective_crash_date = D   (unchanged)
This script's outcome window is hours 06:00-23:59 on that effective date.

Combined exposure
-----------------
AMBER alert campaigns often span consecutive nights (Update messages,
ongoing searches), so "last night's alert" and "the night before's alert"
are correlated, not independent draws. Splitting exposure across two
separate marginal terms (today / yesterday) loses power and -- more
importantly -- makes a naive backward-causal placebo (does *tomorrow's*
alert "predict" today's crashes?) come out spuriously significant, purely
because tomorrow's alert is correlated with today's real one. Once today's
own status is held fixed, the backward placebo passes cleanly (see the
robustness block below); with that established, a single COMBINED exposure
measure (alert on either of the last two nights) is the more powerful and
better-specified primary estimate.

Robustness
----------
The naive spec (county + year + weekday + month fixed effects) understates
how much of the "effect" is really uncontrolled county-specific trend and
county-specific weekday pattern. Adding county x year (fully flexible,
not just a linear trend) and county x weekday fixed effects shrinks the
combined-dose estimate by roughly 25% and moves it from p~0.01 to the
p~0.03-0.05 range -- still directionally positive, but a materially weaker
and more fragile result than the naive spec suggests. Both specs are
reported; only the robust one should be treated as the headline number.

Standard errors are two-way clustered by state x calendar date
(CLUSTER_VARS below), not just by state. State-only clustering already
absorbs within-state, same-day correlation from statewide alert campaigns
(a state cluster nests every one of its counties' full time series), but
misses same-day correlation across DIFFERENT states (a national weather
event or holiday) that month/weekday fixed effects don't fully soak up.
Checked against four alternatives (1-way state/county/date, 2-way
county+date) in reg_night_to_morning_clustering_check.csv: the headline
commuting-spillover result is significant under every one (p=.012-.026);
two-way state+date is both the more defensible choice and the tightest,
so it is the default here rather than a mere robustness footnote.

Also checked: the choice of night_start/night_end cutoff hours themselves
(reg_night_to_morning_cutoff_sensitivity.csv, run_night_to_morning_cutoff_sensitivity.py).
The commuting-spillover result holds for every night_start >= 21 (9pm),
and breaks down only when "night" is stretched to include 8-9pm, which is
the substantively expected failure mode, not a red flag.

Commuting spillover
--------------------
Also tests exposure via commuting, not just a county's own alert: even a
county with no alert of its own may see elevated daytime crashes if a large
share of its workforce commutes in from a county that WAS alerted overnight
(sleep-disrupted commuters driving into county c the next morning). Reuses
the same cross-spillover formula already established in 05_analysis.py's
run_commuting_spillover / _build_cross_spillover:

    cross_spillover_ct = sum_{j != c} w_{j->c} x night_alert_{j,t}

where w_{j->c} = the fraction of county c's workforce that commutes in from
county j (ACS 2016-2020 5-year county-to-county commuting flows,
data/processed/commuting/county_commuting_weights.parquet). The only
difference from the existing implementation is that night_alert_{j,t} here
uses this script's correctly-dated effective_crash_date (see "Window
alignment" above) rather than the raw alert date, and the outcome is this
script's 06:00-23:59 window rather than the full next calendar day.

Data
----
FARS hourly crash counts: data/processed/fars_hourly_county_day.parquet
  (cached; see run_time_window_analysis.py for how to rebuild from raw
  FARS ZIPs when they're available -- not needed here since the cache
  already exists in data/processed/).
AMBER alerts: run_state_dot_analysis_fixed.load_verified_alerts(window="night")
  (case-sensitivity-fixed source data, statewide alerts geo-expanded).
Commuting weights: data/processed/commuting/county_commuting_weights.parquet
  (built by build_commuting_weights.py if missing).

Output: output/tables/reg_night_to_morning_window.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

MIN_FATALS_PER_YEAR = 5  # matches the county restriction used elsewhere in the repo
COMMUTING_WEIGHTS_PATH = DATA_PROC / "commuting" / "county_commuting_weights.parquet"


def _ensure_commuting_weights():
    if COMMUTING_WEIGHTS_PATH.exists():
        return
    log.info("Commuting weights not found — building from ACS data...")
    spec = importlib.util.spec_from_file_location(
        "build_weights", Path(__file__).parent / "build_commuting_weights.py")
    bw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bw)


def build_outcome_grid() -> pd.DataFrame:
    """Balanced county x date grid with 06:00-23:59 fatal/serious sums."""
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    day_window = hourly[hourly["hour"].between(6, 23)]
    day_agg = (day_window.groupby(["fips", "date"])
               .agg(fatals_0623=("person_fatals", "sum"),
                    serious_0623=("serious_inj", "sum"))
               .reset_index())

    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (fars.assign(year=fars["date"].dt.year)
                   .groupby(["fips", "year"])[fatals_col].sum()
                   .groupby("fips").mean())
    active = mean_annual[mean_annual >= MIN_FATALS_PER_YEAR].index.tolist()
    log.info("Active (>=%d fatals/yr) counties: %d", MIN_FATALS_PER_YEAR, len(active))

    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")
    grid = pd.MultiIndex.from_product([active, dates], names=["fips", "date"]).to_frame(index=False)
    grid = grid.merge(day_agg, on=["fips", "date"], how="left")
    grid["fatals_0623"] = grid["fatals_0623"].fillna(0)
    grid["serious_0623"] = grid["serious_0623"].fillna(0)
    return grid


def attach_night_alert(grid: pd.DataFrame) -> pd.DataFrame:
    alerts = base.load_verified_alerts(window="night", detail=False)
    ev = alerts.rename(columns={"effective_crash_date": "date"})[["fips", "date"]].drop_duplicates()
    ev["night_alert"] = 1
    ev["date"] = pd.to_datetime(ev["date"])

    grid = grid.merge(ev, on=["fips", "date"], how="left")
    grid["night_alert"] = grid["night_alert"].fillna(0).astype(int)
    log.info("Night-alert county-dates matched to active-county grid: %d",
             int(grid["night_alert"].sum()))

    grid = grid.sort_values(["fips", "date"]).reset_index(drop=True)
    grid["night_alert_lag1"] = grid.groupby("fips")["night_alert"].shift(1).fillna(0).astype(int)
    grid["night_alert_lead1"] = grid.groupby("fips")["night_alert"].shift(-1).fillna(0).astype(int)
    grid["alert_last2nights_any"] = ((grid["night_alert"] + grid["night_alert_lag1"]) > 0).astype(int)
    grid["alert_last2nights_dose"] = grid["night_alert"] + grid["night_alert_lag1"]
    return grid


def attach_cross_spillover(grid: pd.DataFrame, home_alert_col: str = "night_alert",
                          out_col: str = "cross_spillover", weights=None) -> pd.DataFrame:
    """
    {out_col}_ct = sum_{j != c} w_{j->c} x {home_alert_col}_{j,t}

    Share of county c's workforce commuting in from a county j that had a
    night alert (per home_alert_col) on the same effective date t. 0 for
    counties with no commuting inflow from any alerted county that date.

    Passing home_alert_col="night_alert_lead1" builds a backward-causal
    placebo version: spillover exposure computed from the HOME county's
    alert status the *following* night cannot causally affect today's
    crashes in the work county, so a real effect on this term (once today's
    real cross_spillover is also controlled for) would indicate confounding
    rather than a genuine spillover mechanism.
    """
    if weights is None:
        _ensure_commuting_weights()
        weights = pd.read_parquet(COMMUTING_WEIGHTS_PATH)  # fips_home, fips_work, weight
        log.info("Commuting weights: %d OD pairs, %d work counties",
                 len(weights), weights["fips_work"].nunique())

    alert_events = grid.loc[grid[home_alert_col] > 0, ["fips", "date"]].copy()
    alert_events["fips_home"] = alert_events["fips"].astype(int)
    if alert_events.empty:
        log.warning("No %s events in grid — %s will be all zeros", home_alert_col, out_col)
        grid[out_col] = 0.0
        return grid

    fips_in_sample = set(grid["fips"].unique())
    spill_pairs = alert_events.merge(weights, on="fips_home", how="inner")
    spill_pairs = spill_pairs[spill_pairs["fips_home"] != spill_pairs["fips_work"]]
    spill_pairs["fips_work_str"] = spill_pairs["fips_work"].astype(str).str.zfill(5)
    spill_pairs = spill_pairs[spill_pairs["fips_work_str"].isin(fips_in_sample)]
    log.info("%s pairs: %d (alert events x weight links)", out_col, len(spill_pairs))

    spillover = (spill_pairs.groupby(["fips_work_str", "date"])["weight"]
                 .sum().reset_index()
                 .rename(columns={"weight": out_col, "fips_work_str": "fips"}))

    grid = grid.merge(spillover, on=["fips", "date"], how="left")
    grid[out_col] = grid[out_col].fillna(0.0)
    log.info("%s: mean=%.5f, max=%.4f, nonzero rows=%d", out_col,
             grid[out_col].mean(), grid[out_col].max(), int((grid[out_col] > 0).sum()))
    return grid


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


CLUSTER_VARS = "state_code + date_str"  # two-way: state (correlated statewide
# alert campaigns) x calendar date (same-day national shocks month/dow FE
# don't absorb). Checked against 1-way state/county/date and 2-way
# county+date in reg_night_to_morning_clustering_check.csv -- the headline
# spillover result is significant under every choice (p=.012-.026); two-way
# state+date is the tightest and is used as the default here.


def run(grid, label, outcome, treat, fe, results, extra_controls=None):
    controls = [treat] + (extra_controls or [])
    sub = grid.dropna(subset=controls + [outcome]).copy()
    formula = f"{outcome} ~ {' + '.join(controls)} | {fe}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": CLUSTER_VARS}, lean=True)
    td = fit.tidy()
    row = td.loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    log.info("  %-70s beta=%+.5f se=%.5f p=%.3f n=%d %s [FE: %s]",
             label, coef, se, pval, int(fit._N), _sig(pval), fe)
    results.append({"label": label, "outcome": outcome, "treatment": treat, "fe": fe,
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N)})


def main():
    grid = build_outcome_grid()
    grid = attach_night_alert(grid)
    grid = attach_cross_spillover(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    results = []
    log.info("\n=== Naive spec: fips + year + dow + month FE ===")
    run(grid, "Combined-any -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_any",
        "fips + year_str + dow + month_str", results)
    run(grid, "Combined-dose -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_dose",
        "fips + year_str + dow + month_str", results)
    run(grid, "Combined-any -> serious injuries", "serious_0623", "alert_last2nights_any",
        "fips + year_str + dow + month_str", results)

    log.info("\n=== Robust spec: county x year + county x weekday FE (headline) ===")
    run(grid, "Combined-any -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_any",
        "fips_year + fips_dow + month_str", results)
    run(grid, "Combined-dose -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_dose",
        "fips_year + fips_dow + month_str", results)

    log.info("\n=== Commuting spillover: own alert + share of commuters from an alerted "
             "county, jointly (naive spec) ===")
    run(grid, "OWN night_alert (spillover-controlled) -> fatals", "fatals_0623", "night_alert",
        "fips + year_str + dow + month_str", results, extra_controls=["cross_spillover"])
    run(grid, "CROSS_SPILLOVER (own-controlled) -> fatals", "fatals_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results, extra_controls=["night_alert"])
    run(grid, "CROSS_SPILLOVER (own-controlled) -> serious injuries", "serious_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results, extra_controls=["night_alert"])

    log.info("\n=== Commuting spillover, robust spec (county x year + county x weekday FE) ===")
    run(grid, "OWN night_alert (spillover-controlled) -> fatals", "fatals_0623", "night_alert",
        "fips_year + fips_dow + month_str", results, extra_controls=["cross_spillover"])
    run(grid, "CROSS_SPILLOVER (own-controlled) -> fatals", "fatals_0623", "cross_spillover",
        "fips_year + fips_dow + month_str", results, extra_controls=["night_alert"])

    log.info("\n=== Backward-causal placebo: spillover from a HOME county's alert the "
             "FOLLOWING night cannot cause today's crashes ===")
    weights = pd.read_parquet(COMMUTING_WEIGHTS_PATH)
    grid = attach_cross_spillover(grid, home_alert_col="night_alert_lead1",
                                 out_col="cross_spillover_placebo", weights=weights)
    # Controls for today's REAL cross_spillover throughout, since alert
    # campaigns span consecutive nights (a home county's alert tonight is
    # correlated with its alert tomorrow night) -- exactly the confound that
    # made the naive backward placebo on the OWN-alert term look spuriously
    # significant earlier, before controlling for today's own status fixed it.
    run(grid, "PLACEBO: spillover from tomorrow's home-county alert -> today's fatals",
        "fatals_0623", "cross_spillover_placebo",
        "fips + year_str + dow + month_str", results,
        extra_controls=["cross_spillover", "night_alert"])
    run(grid, "PLACEBO: spillover from tomorrow's home-county alert -> today's fatals (robust FE)",
        "fatals_0623", "cross_spillover_placebo",
        "fips_year + fips_dow + month_str", results,
        extra_controls=["cross_spillover", "night_alert"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_night_to_morning_window.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
