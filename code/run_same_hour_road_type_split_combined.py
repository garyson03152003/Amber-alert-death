"""
run_same_hour_road_type_split_combined.py
=============================================================
Does the H1 same-hour highway-fatals effect
(reg_same_hour_road_type_split.csv: beta=-0.000175, se=0.000035,
p=6.7e-06, n=948,423, AMBER-only) hold up, and does it strengthen, once
the treatment definition also includes the Silver-Alert-type
missing-person WEA alerts found by 02f_geocode_missing_person_alerts.py
(1,418 unique alerts, 2014-2024 -- no dedicated IPAWS event code existed
for this population before September 2025, so these were only findable
via free-text screening; see that script's docstring for the full
provenance)?

Combined treatment definition
-------------------------------
Per instruction, the missing-person data's two population labels are
combined into a single "any WEA missing-person alert" exposure rather
than analyzed separately: 'missing_person' (elderly/adult, the actual
Silver-Alert-equivalent population) and 'child_amber_adjacent' (missing/
endangered minors caught by generic event codes, treated as the same
population AMBER/CAE covers rather than as its own category). Both are
unioned with the existing AMBER alert-hours from
run_same_hour_event_study.load_any_time_alert_hours() at the
(fips, date, hour) grain -- an hour is "treated" if EITHER source has an
alert in it, matching how AMBER's own Alert/Update/Cancel messages are
already unioned into one is_alert_hour flag.

Everything else -- the matched-referent case-crossover grid
construction, fips_hour_dow + fips_year + year_month FE, two-way
state+date clustering, the road-type outcome split
(fars_road_type_county_day.parquet) -- is identical to
run_same_hour_road_type_split.py, so the AMBER-only and combined runs
are an apples-to-apples comparison except for the treatment definition.

Output: output/tables/reg_same_hour_road_type_split_combined.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_same_hour_event_study as base
import run_same_hour_road_type_split as rt
from load_missing_person_alerts import load_missing_person_alert_hours
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("same_hour_road_type_combined")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

CLUSTER_VARS = base.CLUSTER_VARS


def load_combined_alert_hours(active: set[str]) -> pd.DataFrame:
    ev_amber = base.load_any_time_alert_hours(active)
    n_amber = len(ev_amber)

    ev_mp = load_missing_person_alert_hours(populations=("missing_person", "child_amber_adjacent"))
    ev_mp = ev_mp[ev_mp["fips"].isin(active)].copy()
    ev_mp["date"] = pd.to_datetime(ev_mp["sent_local"]).dt.normalize()
    ev_mp = ev_mp.rename(columns={"hour_local": "hour"})
    ev_mp = ev_mp[["fips", "date", "hour", "msg_type"]].drop_duplicates()
    n_mp_active = len(ev_mp)

    combined = pd.concat([ev_amber, ev_mp], ignore_index=True)
    combined = combined.drop_duplicates(subset=["fips", "date", "hour"], keep="first")
    n_overlap = n_amber + n_mp_active - len(combined)
    log.info("Combined alert-hours: %d AMBER-only rows + %d missing-person rows in active "
             "counties -> %d unique (fips,date,hour) treated hours (%d overlapping exactly)",
             n_amber, n_mp_active, len(combined), n_overlap)
    return combined


def main():
    active = base.active_counties()
    log.info("Active (>=%d fatals/yr) counties: %d", base.MIN_FATALS_PER_YEAR, len(active))
    ev = load_combined_alert_hours(active)

    grid = rt.build_road_type_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    fe = "fips_hour_dow + fips_year + year_month"

    results = []
    log.info("\n=== Same-hour case-crossover, COMBINED (AMBER + missing-person) treatment ===")
    rt.run(grid, "highway fatals, combined treatment", "highway_fatals", "is_alert_hour", fe, results)
    rt.run(grid, "non-highway fatals, combined treatment", "nonhighway_fatals", "is_alert_hour", fe, results)

    log.info("\n=== Backward-causal placebo (tomorrow's alert, controlling for today's real one) ===")
    rt.run(grid, "highway fatals, placebo, combined treatment", "highway_fatals", "is_alert_hour_tomorrow", fe,
           results, extra_controls=["is_alert_hour"])
    rt.run(grid, "non-highway fatals, placebo, combined treatment", "nonhighway_fatals", "is_alert_hour_tomorrow",
           fe, results, extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_road_type_split_combined.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
