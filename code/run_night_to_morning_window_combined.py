"""
run_night_to_morning_window_combined.py
=============================================================
Does the H2 sleep/commuting-spillover test (run_night_to_morning_window.py's
"own night_alert (spillover-controlled)" and "CROSS_SPILLOVER
(own-controlled)" specs) change once the missing-person data
(02f_geocode_missing_person_alerts.py) is combined with AMBER as the
source of both direct night_alert exposure and the home-county alerts
that generate commuting-spillover exposure?

Combined treatment definition
-------------------------------
Same population combination as run_same_hour_road_type_split_combined.py
(per instruction): 'missing_person' + 'child_amber_adjacent' unioned with
AMBER at the (fips, effective_crash_date) grain -- a county-night is
"night_alert=1" if EITHER source has a night-window alert (22:00-05:59
local, same NIGHT_START_HOUR/NIGHT_END_HOUR as
run_state_dot_analysis_fixed.load_verified_alerts, with the same
evening-alert-belongs-to-next-day date shift). cross_spillover is then
built from this COMBINED night_alert flag via
run_night_to_morning_window.attach_cross_spillover(), so a work county's
exposure now reflects commuters from ANY alerted home county (AMBER or
missing-person), not just AMBER ones.

Given the underlying H2 result was already established as robustly null
across five independent checks in this repo (geographic leave-one-out
fragility, an exposure-composition confound with statewide alert
campaigns, a requirement for implausibly short commuting distances, an
inconclusive road-type dose-response, and a clean null on the most
targeted tract-level car x distance x hours-since-wake x non-alert-
affected-counties combination), this reruns only the core "own +
spillover" specs (not the full 12-regression original suite) to check
whether combining data sources changes that picture, rather than
re-litigating checks already run against AMBER alone.

Output: output/tables/reg_night_to_morning_window_combined.csv
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from load_missing_person_alerts import load_missing_person_night_alert_dates
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning_combined")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)


def attach_combined_night_alert(grid: pd.DataFrame) -> pd.DataFrame:
    """Same shape/derived-columns as ntm.attach_night_alert, but night_alert
    is 1 if EITHER AMBER or the missing-person data has a night-window
    alert for that (fips, effective_crash_date)."""
    amber_events = ntm.base.load_verified_alerts(window="night", detail=True)
    amber_events = amber_events.rename(columns={"effective_crash_date": "date"})[["fips", "date"]].drop_duplicates()
    mp_events = load_missing_person_night_alert_dates().rename(columns={"effective_crash_date": "date"})

    combined_events = pd.concat([amber_events, mp_events], ignore_index=True).drop_duplicates()
    combined_events["date"] = pd.to_datetime(combined_events["date"])
    combined_events["night_alert"] = 1
    log.info("Combined night-alert county-dates: %d AMBER-only + %d missing-person -> %d unique "
             "(%d overlapping exactly)",
             len(amber_events), len(mp_events), len(combined_events),
             len(amber_events) + len(mp_events) - len(combined_events))

    grid = grid.merge(combined_events, on=["fips", "date"], how="left")
    grid["night_alert"] = grid["night_alert"].fillna(0).astype(int)
    log.info("Combined night-alert county-dates matched to active-county grid: %d",
             int(grid["night_alert"].sum()))

    grid = grid.sort_values(["fips", "date"]).reset_index(drop=True)
    grid["night_alert_lag1"] = grid.groupby("fips")["night_alert"].shift(1).fillna(0).astype(int)
    grid["night_alert_lead1"] = grid.groupby("fips")["night_alert"].shift(-1).fillna(0).astype(int)
    grid["alert_last2nights_any"] = ((grid["night_alert"] + grid["night_alert_lag1"]) > 0).astype(int)
    grid["alert_last2nights_dose"] = grid["night_alert"] + grid["night_alert_lag1"]
    return grid


def main():
    grid = ntm.build_outcome_grid()
    grid = attach_combined_night_alert(grid)
    grid = ntm.attach_cross_spillover(grid)  # uses the (now combined) night_alert column as the source
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    results = []
    log.info("\n=== Commuting spillover, COMBINED treatment (naive spec) ===")
    ntm.run(grid, "OWN night_alert (spillover-controlled) -> fatals, combined", "fatals_0623", "night_alert",
            "fips + year_str + dow + month_str", results, extra_controls=["cross_spillover"])
    ntm.run(grid, "CROSS_SPILLOVER (own-controlled) -> fatals, combined", "fatals_0623", "cross_spillover",
            "fips + year_str + dow + month_str", results, extra_controls=["night_alert"])
    ntm.run(grid, "CROSS_SPILLOVER (own-controlled) -> serious injuries, combined", "serious_0623", "cross_spillover",
            "fips + year_str + dow + month_str", results, extra_controls=["night_alert"])

    log.info("\n=== Commuting spillover, COMBINED treatment (robust spec) ===")
    ntm.run(grid, "OWN night_alert (spillover-controlled) -> fatals, combined", "fatals_0623", "night_alert",
            "fips_year + fips_dow + month_str", results, extra_controls=["cross_spillover"])
    ntm.run(grid, "CROSS_SPILLOVER (own-controlled) -> fatals, combined", "fatals_0623", "cross_spillover",
            "fips_year + fips_dow + month_str", results, extra_controls=["night_alert"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_night_to_morning_window_combined.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
