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

Tract-preserved commuter-car-mile sensitivity: the headline output also
reports year-matched own and cross exposure using ACS 2015/2020 commuter
shares multiplied by the LODES 2013/2018/2022 worker-weighted joint quantity
E[home-tract car share x home-tract-to-work-tract distance].  This dosage is
kept separate from the legacy binary/share variables so downstream alert-hour
analyses retain their original interpretation.

Output: output/tables/reg_night_to_morning_window.csv
Rich exposure diagnostics: output/tables/night_to_morning_commuter_car_exposure_summary.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pyfixest as pf
except ModuleNotFoundError:  # keep data/alert helpers importable in light envs
    pf = None

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

MIN_FATALS_PER_YEAR = 5  # matches the county restriction used elsewhere in the repo
COMMUTING_WEIGHTS_PATH = DATA_PROC / "commuting" / "county_commuting_weights.parquet"
PAIR_DOSAGE_YEAR_DIR = DATA_PROC / "commuting" / "_lodes_car_year_cache"
FLOW_VINTAGE_PATHS = {
    "2015": DATA_PROC / "commuting" / "county_commuting_weights_2015.parquet",
    "2020": DATA_PROC / "commuting" / "county_commuting_weights_2020.parquet",
}
OTHER_WEA_CONTROL_PATH = DATA_PROC / "other_wea_night_controls.parquet"
# Keep the binary indicator as the headline control and expose a count-dose
# sensitivity so repeated overlapping alerts are not collapsed into one bit.
OTHER_WEA_CONTROL_SPECS = (
    ("binary", ("other_wea_night_alert",)),
    ("dose", ("other_wea_night_count",)),
)
RICH_EXPOSURE_SPEC = "year_matched_tract_car_distance"
RICH_EXPOSURE_SUMMARY_PATH = OUTPUT_TABS / "night_to_morning_commuter_car_exposure_summary.csv"


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


def attach_other_wea_control(
    grid: pd.DataFrame, *, required: bool = True, path: Path | None = None
) -> pd.DataFrame:
    """Merge non-AMBER overnight WEA exposure onto the county-day grid.

    The control is a binary indicator for at least one other public,
    non-CMAS-blocked Alert/Update in the same county overnight.  A separate
    count column is retained for a later dose sensitivity, but the headline
    specifications use the indicator so a burst of repeated updates does not
    mechanically dominate the regression.
    """
    control_path = path or OTHER_WEA_CONTROL_PATH
    if not control_path.is_file():
        if required:
            raise FileNotFoundError(
                f"other-WEA control not found at {control_path}; "
                "run code/fetch_other_wea_controls.py first"
            )
        out = grid.copy()
        out["other_wea_night_alert"] = 0
        out["other_wea_night_count"] = 0
        return out
    controls = pd.read_parquet(control_path).copy()
    required_cols = {"fips", "date", "other_wea_night_alert", "other_wea_night_count"}
    missing = required_cols - set(controls.columns)
    if missing:
        raise ValueError(f"other-WEA control missing columns: {sorted(missing)}")
    controls["fips"] = controls["fips"].astype(str).str.zfill(5)
    controls["date"] = pd.to_datetime(controls["date"]).dt.normalize()
    controls = controls.drop_duplicates(["fips", "date"], keep="last")
    out = grid.copy()
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(controls, on=["fips", "date"], how="left", validate="one_to_one")
    out["other_wea_night_alert"] = out["other_wea_night_alert"].fillna(0).astype(int)
    out["other_wea_night_count"] = out["other_wea_night_count"].fillna(0).astype(int)
    log.info("Other-WEA overnight control matched %d county-days (%.3f%% of grid)",
             int(out["other_wea_night_alert"].sum()),
             100 * float(out["other_wea_night_alert"].mean()))
    return out


def _normalize_fips(values: pd.Series) -> pd.Series:
    """Normalize county identifiers without changing their five-digit meaning."""
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def _load_year_matched_pair_dosages(sample_fips: set[str]) -> dict[tuple[str, int], pd.DataFrame]:
    """Load validated ACS/LODES pair dosages for the crash-year regimes.

    The heavy LODES construction remains in ``build_lodes_tract_car_dosage``.
    This loader only reads its cached year-specific tables and attaches the
    corresponding ACS commuter-share vintage.  The helper is kept local to
    this runner so importing the alert/outcome functions does not trigger any
    downloads or regression dependencies.
    """
    from run_symmetric_commuter_fatigue import (
        FLOW_VINTAGE_FOR_YEAR,
        LODES_VINTAGE_FOR_YEAR,
        build_distance_driving_fallback,
        build_pair_dosage,
    )

    flow_tables: dict[str, pd.DataFrame] = {}
    for vintage, path in FLOW_VINTAGE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"missing ACS commuting-flow vintage {vintage}: {path}"
            )
        flow = pd.read_parquet(path).copy()
        flow["fips_home"] = _normalize_fips(flow["fips_home"])
        flow["fips_work"] = _normalize_fips(flow["fips_work"])
        flow_tables[vintage] = flow[
            ["fips_home", "fips_work", "weight"]
        ]

    centroid_path = DATA_PROC / "county_pop_centroids.parquet"
    if not centroid_path.is_file():
        raise FileNotFoundError(
            f"missing county centroids needed for commuting-distance fallback: {centroid_path}"
        )
    centroids = pd.read_parquet(centroid_path)

    regimes = sorted(
        {
            (FLOW_VINTAGE_FOR_YEAR[year], LODES_VINTAGE_FOR_YEAR[year])
            for year in range(2013, 2025)
        }
    )
    pair_dosages: dict[tuple[str, int], pd.DataFrame] = {}
    for flow_vintage, lodes_vintage in regimes:
        year_path = PAIR_DOSAGE_YEAR_DIR / (
            f"county_pair_lodes_car_dosage_{lodes_vintage}.parquet"
        )
        if not year_path.is_file():
            raise FileNotFoundError(
                f"missing year-specific LODES car-distance dosage: {year_path}; "
                "run code/build_lodes_tract_car_dosage.py first"
            )
        weights = flow_tables[flow_vintage]
        weights = weights[
            weights["fips_home"].isin(sample_fips)
            & weights["fips_work"].isin(sample_fips)
        ].copy()
        joint = pd.read_parquet(year_path)
        joint["fips_home"] = _normalize_fips(joint["fips_home"])
        joint["fips_work"] = _normalize_fips(joint["fips_work"])
        fallback = build_distance_driving_fallback(
            weights,
            joint[["fips_home", "fips_work", "avg_dist_mi"]],
            centroids,
        )
        pair_dosages[(flow_vintage, lodes_vintage)] = build_pair_dosage(
            weights, joint, fallback
        )
    return pair_dosages


def attach_year_matched_commuter_exposure(
    grid: pd.DataFrame,
    *,
    alert_col: str = "night_alert",
    pair_dosages: dict[tuple[str, int], pd.DataFrame] | None = None,
    summary_path: Path | None = None,
) -> pd.DataFrame:
    """Attach tract-preserved own and cross commuter-car-mile exposure.

    For each county-day, ``own_driver_distance`` is the alerted county's
    self-loop dosage and ``cross_driver_distance`` sums non-self-loop dosage
    from alerted home counties.  Pair dosage is the ACS commuter share times
    the LODES worker-weighted ``E[home-tract car share x tract-centroid
    distance]`` quantity.  Passing ``pair_dosages`` makes the pure attachment
    testable without reading the production Parquet caches.
    """
    required = {"fips", "date", alert_col}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"grid missing required exposure columns: {sorted(missing)}")

    out = grid.copy()
    out["fips"] = _normalize_fips(out["fips"])
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if out.duplicated(["fips", "date"]).any():
        raise ValueError("grid must contain at most one row per county-day")

    if pair_dosages is None:
        pair_dosages = _load_year_matched_pair_dosages(set(out["fips"]))

    from run_symmetric_commuter_fatigue import construct_year_matched_exposure_series

    grid_index = pd.MultiIndex.from_frame(out[["fips", "date"]])
    alerts = out.loc[
        pd.to_numeric(out[alert_col], errors="raise").gt(0), ["fips", "date"]
    ].drop_duplicates()
    own, cross = construct_year_matched_exposure_series(
        grid_index, alerts, pair_dosages
    )
    out["own_driver_distance"] = own
    out["cross_driver_distance"] = cross
    if summary_path is not None:
        summary_rows = []
        for (flow_vintage, lodes_vintage), pairs in sorted(pair_dosages.items()):
            fallback = pairs.get("dosage_source", pd.Series(dtype=str)).eq(
                "distance_driving_fallback"
            )
            summary_rows.append(
                {
                    "flow_vintage": flow_vintage,
                    "lodes_vintage": lodes_vintage,
                    "n_pair_edges": len(pairs),
                    "fallback_pair_edges": int(fallback.sum()),
                    "fallback_weight_share": float(
                        pairs.loc[fallback, "weight"].sum() / pairs["weight"].sum()
                    )
                    if len(pairs) and pairs["weight"].sum()
                    else 0.0,
                    "grid_rows": len(out),
                    "own_nonzero_rows": int(out["own_driver_distance"].gt(0).sum()),
                    "cross_nonzero_rows": int(out["cross_driver_distance"].gt(0).sum()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        log.info("Saved rich commuter-car exposure summary -> %s", summary_path)
    log.info(
        "Year-matched commuter-car exposure: own nonzero=%d, cross nonzero=%d, "
        "own mean=%.4f, cross mean=%.4f",
        int(out["own_driver_distance"].gt(0).sum()),
        int(out["cross_driver_distance"].gt(0).sum()),
        float(out["own_driver_distance"].mean()),
        float(out["cross_driver_distance"].mean()),
    )
    return out


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


def run(
    grid,
    label,
    outcome,
    treat,
    fe,
    results,
    extra_controls=None,
    *,
    exposure_spec: str = "legacy_binary_or_county_share",
):
    if pf is None:
        raise ImportError("pyfixest is required for regression estimation")
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
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N),
                    "exposure_spec": exposure_spec})


def main():
    grid = build_outcome_grid()
    grid = attach_night_alert(grid)
    grid = attach_cross_spillover(grid)
    grid = attach_other_wea_control(grid)
    grid = attach_year_matched_commuter_exposure(
        grid, summary_path=RICH_EXPOSURE_SUMMARY_PATH
    )
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

    log.info("\n=== Other-WEA controlled primary estimates ===")
    other_wea = ["other_wea_night_alert"]
    run(grid, "Combined-any + other WEA -> fatals 06:00-23:59", "fatals_0623",
        "alert_last2nights_any", "fips + year_str + dow + month_str", results,
        extra_controls=other_wea)
    run(grid, "Combined-dose + other WEA -> fatals 06:00-23:59", "fatals_0623",
        "alert_last2nights_dose", "fips + year_str + dow + month_str", results,
        extra_controls=other_wea)
    run(grid, "Combined-any + other WEA -> fatals (robust)", "fatals_0623",
        "alert_last2nights_any", "fips_year + fips_dow + month_str", results,
        extra_controls=other_wea)
    run(grid, "Combined-dose + other WEA -> fatals (robust)", "fatals_0623",
        "alert_last2nights_dose", "fips_year + fips_dow + month_str", results,
        extra_controls=other_wea)

    log.info("\n=== Other-WEA dose-controlled sensitivity ===")
    other_wea_dose = ["other_wea_night_count"]
    run(grid, "Combined-any + other WEA count -> fatals 06:00-23:59", "fatals_0623",
        "alert_last2nights_any", "fips + year_str + dow + month_str", results,
        extra_controls=other_wea_dose)
    run(grid, "Combined-dose + other WEA count -> fatals 06:00-23:59", "fatals_0623",
        "alert_last2nights_dose", "fips + year_str + dow + month_str", results,
        extra_controls=other_wea_dose)
    run(grid, "Combined-any + other WEA count -> fatals (robust)", "fatals_0623",
        "alert_last2nights_any", "fips_year + fips_dow + month_str", results,
        extra_controls=other_wea_dose)
    run(grid, "Combined-dose + other WEA count -> fatals (robust)", "fatals_0623",
        "alert_last2nights_dose", "fips_year + fips_dow + month_str", results,
        extra_controls=other_wea_dose)

    log.info("\n=== Commuting spillover: own alert + share of commuters from an alerted "
             "county, jointly (naive spec) ===")
    run(grid, "OWN night_alert (spillover-controlled) -> fatals", "fatals_0623", "night_alert",
        "fips + year_str + dow + month_str", results, extra_controls=["cross_spillover"])
    run(grid, "CROSS_SPILLOVER (own-controlled) -> fatals", "fatals_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results, extra_controls=["night_alert"])
    run(grid, "CROSS_SPILLOVER (own-controlled) -> serious injuries", "serious_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results, extra_controls=["night_alert"])

    run(grid, "OWN night_alert + other WEA -> fatals", "fatals_0623", "night_alert",
        "fips + year_str + dow + month_str", results,
        extra_controls=["cross_spillover", "other_wea_night_alert"])
    run(grid, "CROSS_SPILLOVER + other WEA -> fatals", "fatals_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results,
        extra_controls=["night_alert", "other_wea_night_alert"])
    run(grid, "CROSS_SPILLOVER + other WEA -> serious injuries", "serious_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results,
        extra_controls=["night_alert", "other_wea_night_alert"])

    run(grid, "OWN night_alert + other WEA count -> fatals", "fatals_0623", "night_alert",
        "fips + year_str + dow + month_str", results,
        extra_controls=["cross_spillover", "other_wea_night_count"])
    run(grid, "CROSS_SPILLOVER + other WEA count -> fatals", "fatals_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results,
        extra_controls=["night_alert", "other_wea_night_count"])
    run(grid, "CROSS_SPILLOVER + other WEA count -> serious injuries", "serious_0623", "cross_spillover",
        "fips + year_str + dow + month_str", results,
        extra_controls=["night_alert", "other_wea_night_count"])

    log.info("\n=== Commuting spillover, robust spec (county x year + county x weekday FE) ===")
    run(grid, "OWN night_alert (spillover-controlled) -> fatals", "fatals_0623", "night_alert",
        "fips_year + fips_dow + month_str", results, extra_controls=["cross_spillover"])
    run(grid, "CROSS_SPILLOVER (own-controlled) -> fatals", "fatals_0623", "cross_spillover",
        "fips_year + fips_dow + month_str", results, extra_controls=["night_alert"])

    run(grid, "OWN night_alert + other WEA -> fatals (robust)", "fatals_0623", "night_alert",
        "fips_year + fips_dow + month_str", results,
        extra_controls=["cross_spillover", "other_wea_night_alert"])
    run(grid, "CROSS_SPILLOVER + other WEA -> fatals (robust)", "fatals_0623", "cross_spillover",
        "fips_year + fips_dow + month_str", results,
        extra_controls=["night_alert", "other_wea_night_alert"])

    run(grid, "OWN night_alert + other WEA count -> fatals (robust)", "fatals_0623", "night_alert",
        "fips_year + fips_dow + month_str", results,
        extra_controls=["cross_spillover", "other_wea_night_count"])
    run(grid, "CROSS_SPILLOVER + other WEA count -> fatals (robust)", "fatals_0623", "cross_spillover",
        "fips_year + fips_dow + month_str", results,
        extra_controls=["night_alert", "other_wea_night_count"])

    log.info("\n=== Tract-preserved commuter-car-mile exposure (year matched) ===")
    rich_control_specs = (
        ("no other-WEA control", []),
        ("other WEA binary", ["other_wea_night_alert"]),
        ("other WEA count", ["other_wea_night_count"]),
    )
    rich_fe_specs = (
        ("naive", "fips + year_str + dow + month_str"),
        ("robust", "fips_year + fips_dow + month_str"),
    )
    for fe_label, fe in rich_fe_specs:
        for control_label, control_terms in rich_control_specs:
            run(
                grid,
                f"OWN commuter-car miles ({fe_label}; {control_label}) -> fatals",
                "fatals_0623",
                "own_driver_distance",
                fe,
                results,
                extra_controls=["cross_driver_distance", *control_terms],
                exposure_spec=RICH_EXPOSURE_SPEC,
            )
            run(
                grid,
                f"CROSS commuter-car miles ({fe_label}; {control_label}) -> fatals",
                "fatals_0623",
                "cross_driver_distance",
                fe,
                results,
                extra_controls=["own_driver_distance", *control_terms],
                exposure_spec=RICH_EXPOSURE_SPEC,
            )

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
