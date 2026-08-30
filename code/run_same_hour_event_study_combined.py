"""
run_same_hour_event_study_combined.py
=============================================================
Does the H1 "immediate distraction" same-hour test
(run_same_hour_event_study.py's pooled fatals/serious-injury specs --
the headline numbers before the road-type split) change once the
missing-person data (02f_geocode_missing_person_alerts.py) is combined
with AMBER as the treatment source, the same way
run_same_hour_road_type_split_combined.py already did for the
road-type-split version?

Combined treatment definition: identical to
run_same_hour_road_type_split_combined.py -- 'missing_person' +
'child_amber_adjacent' (per instruction, the latter treated as the same
population AMBER/CAE covers) unioned with AMBER's alert-hours at the
(fips, date, hour) grain. Reuses that script's load_combined_alert_hours()
directly rather than reimplementing it.

Only the pooled (all-road-types) fatals/serious-injury specs and the
backward-causal placebo are rerun here -- not the full original 18-run
suite (first-alert-only variants, PEAK/MID/LOW tier breakdown, the
secondary weekend-matched FE spec), since those are refinements of the
same underlying question and the road-type-split combined run already
established the headline result replicates; this fills in the pooled
(non-road-type) number and the serious-injury outcome, which the
road-type file can't provide (fars_road_type_county_day.parquet has no
serious_inj column).

Output: output/tables/reg_same_hour_event_study_combined.csv
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_same_hour_event_study as base
from run_same_hour_road_type_split_combined import load_combined_alert_hours
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("same_hour_event_study_combined")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)


def main():
    active = base.active_counties()
    log.info("Active (>=%d fatals/yr) counties: %d", base.MIN_FATALS_PER_YEAR, len(active))
    ev = load_combined_alert_hours(active)

    grid = base.build_matched_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    results = []
    log.info("\n=== Same-hour case-crossover, COMBINED treatment, robust FE ===")
    base.run(grid, "fatals: robust FE, combined treatment", "person_fatals", "is_alert_hour",
              "fips_hour_dow + fips_year + year_month", "ols", results)
    base.run(grid, "serious: robust FE, combined treatment", "serious_inj", "is_alert_hour",
              "fips_hour_dow + fips_year + year_month", "ols", results)

    log.info("\n=== Backward-causal placebo, combined treatment ===")
    base.run(grid, "fatals: placebo (tomorrow's alert), combined treatment", "person_fatals",
              "is_alert_hour_tomorrow", "fips_hour_dow + fips_year + year_month", "ols", results,
              extra_controls=["is_alert_hour"])
    base.run(grid, "serious: placebo (tomorrow's alert), combined treatment", "serious_inj",
              "is_alert_hour_tomorrow", "fips_hour_dow + fips_year + year_month", "ols", results,
              extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_event_study_combined.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
