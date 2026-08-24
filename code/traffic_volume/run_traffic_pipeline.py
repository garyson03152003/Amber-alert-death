"""Run Tasks 1-3 year-by-year, memory-bounded: for each downloaded year,
build the station-hour panel, attach the alert treatment, aggregate to
station-day outcomes and event-study hours, then discard the raw station-hour
data before moving to the next year. Concatenating only the (much smaller)
day-level and event-window outputs avoids holding the full multi-year
station-hour panel in memory at once.

Usage: python3 run_traffic_pipeline.py [--years 2013-2022]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_station_hour_panel import process_year, RAW_DIR
from merge_alert_treatment import build_station_day_treatment
from build_traffic_outcomes import build_station_day_outcomes, build_event_study_hours
from utils import get_logger

log = get_logger("run_traffic_pipeline")

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DAY_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_day_outcomes.parquet"
OUT_EVENT_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_event_study_hours.parquet"


def _parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=None, help="e.g. 2013-2022; default = all downloaded years")
    args = ap.parse_args()

    if args.years:
        years = _parse_years(args.years)
    else:
        years = sorted(int(p.name) for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())

    day_parts, event_parts = [], []
    for year in years:
        log.info("=== processing %d ===", year)
        sh = process_year(year)
        if sh is None:
            log.warning("[%d] no station-hour data, skipping", year)
            continue
        treatment = build_station_day_treatment(sh)
        day = build_station_day_outcomes(sh)
        day = day.merge(
            treatment.drop(columns=["state_fips"], errors="ignore"),
            on=["station_id", "county_fips", "date"], how="left",
        )
        event = build_event_study_hours(sh, treatment)
        day_parts.append(day)
        event_parts.append(event)
        log.info(
            "[%d] station-days=%s event-hours=%s exposed_direct=%s",
            year, f"{len(day):,}", f"{len(event):,}",
            int((treatment["night_alert_ct"] == 1).sum()),
        )
        del sh, treatment
        gc.collect()

    if not day_parts:
        log.error("no years produced data")
        sys.exit(1)

    day_outcomes = pd.concat(day_parts, ignore_index=True)
    event_hours = pd.concat(event_parts, ignore_index=True)
    OUT_DAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    day_outcomes.to_parquet(OUT_DAY_PATH, index=False)
    event_hours.to_parquet(OUT_EVENT_PATH, index=False)
    log.info("Wrote %s station-day rows -> %s", f"{len(day_outcomes):,}", OUT_DAY_PATH)
    log.info("Wrote %s event-study rows -> %s", f"{len(event_hours):,}", OUT_EVENT_PATH)
    log.info("Exposure classes: %s", day_outcomes["exposure_class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
