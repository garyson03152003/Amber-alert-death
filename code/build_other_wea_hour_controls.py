"""Build local county-hour controls from non-treatment WEA messages.

The overnight control is not aligned with the exact-hour case-crossover
analysis.  This module reuses the cached FEMA CAP records and the project's
vetted filters, then aggregates each eligible non-AMBER/non-person WEA to the
county's local date and clock hour.  Both an all-WEA series and a
``category=Met``-excluded series are retained in one parquet file so the
hourly sensitivity can switch control definitions without refetching data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fetch_other_wea_controls as archive
import run_state_dot_analysis_fixed as amber_base
from config import DATA_PROC, OUTPUT_TABS, STUDY_YEARS
from utils import get_logger

log = get_logger("other_wea_hour_controls")

CONTROL_PATH = DATA_PROC / "other_wea_hour_controls.parquet"
SUMMARY_PATH = OUTPUT_TABS / "other_wea_hour_control_summary.csv"

# Re-export the vetted SAME parser as a small seam for unit tests and callers.
parse_same_rows = archive.parse_same_rows


def iter_cached_records(cache_dir: Path = archive.MONTH_CACHE_DIR) -> Iterator[dict]:
    """Yield records from the completed month-batched FEMA cache."""
    paths = sorted(cache_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no monthly WEA cache files found under {cache_dir}; "
            "run code/fetch_other_wea_controls.py first"
        )
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _county_targets(fips: str, state_map: dict[str, list[str]]) -> list[str]:
    """Expand a numeric county/state SAME code to outcome counties."""
    return archive._county_targets(fips, state_map)


def aggregate_hour_controls(
    records: Iterable[dict],
    timezone_map: dict[str, str],
    state_map: dict[str, list[str]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_records: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate eligible records to county/local-date/hour controls.

    ``all_wea_hour_*`` includes every retained non-treatment WEA message;
    ``non_weather_wea_hour_*`` additionally excludes CAP records carrying the
    structured ``category=Met`` value.  Counts are distinct source alert IDs,
    while the corresponding ``*_alert`` columns are binary indicators.
    """
    all_ids: dict[tuple[str, pd.Timestamp, int], set[str]] = defaultdict(set)
    non_weather_ids: dict[tuple[str, pd.Timestamp, int], set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    seen = 0
    for rec in records:
        if max_records is not None and seen >= max_records:
            break
        seen += 1
        classified = archive.classify_record(rec, start=start, end=end)
        reason = str(classified["reason"])
        counts[reason] += 1
        if reason != "keep":
            continue

        rows = parse_same_rows(rec)
        if not rows:
            counts["no_numeric_same"] += 1
            continue
        alert_id = str(rec.get("id") or rec.get("identifier") or "")
        if not alert_id:
            counts["missing_alert_id"] += 1
            continue
        targets: set[str] = set()
        for row in rows:
            targets.update(_county_targets(str(row.get("fips", "")), state_map))
        if not targets:
            counts["no_outcome_county"] += 1
            continue

        sent = classified["sent"]
        assert isinstance(sent, pd.Timestamp)
        is_weather = archive.is_weather_wea(rec)
        if is_weather:
            counts["weather_excluded"] += 1
        for fips in targets:
            tz_name = timezone_map.get(fips)
            if not tz_name:
                counts["missing_timezone"] += 1
                continue
            local = sent.tz_convert(tz_name)
            key = (fips, local.tz_localize(None).normalize(), int(local.hour))
            all_ids[key].add(alert_id)
            if not is_weather:
                non_weather_ids[key].add(alert_id)
            counts["kept_county_hours"] += 1

    keys = sorted(all_ids)
    rows = [
        {
            "fips": fips,
            "date": date,
            "hour": hour,
            "all_wea_hour_alert": 1,
            "all_wea_hour_count": len(all_ids[(fips, date, hour)]),
            "non_weather_wea_hour_alert": int(bool(non_weather_ids.get((fips, date, hour)))),
            "non_weather_wea_hour_count": len(non_weather_ids.get((fips, date, hour), set())),
        }
        for fips, date, hour in keys
    ]
    controls = pd.DataFrame(
        rows,
        columns=[
            "fips", "date", "hour", "all_wea_hour_alert", "all_wea_hour_count",
            "non_weather_wea_hour_alert", "non_weather_wea_hour_count",
        ],
    )
    if not controls.empty:
        controls["fips"] = controls["fips"].astype(str).str.zfill(5)
        controls["date"] = pd.to_datetime(controls["date"]).dt.normalize()
        controls["hour"] = controls["hour"].astype(int)
    summary = pd.DataFrame(
        [{"reason": reason, "count": int(count)} for reason, count in sorted(counts.items())]
    )
    return controls, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=min(STUDY_YEARS))
    parser.add_argument("--end-year", type=int, default=max(STUDY_YEARS))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.end_year < args.start_year:
        parser.error("--end-year must be at least --start-year")
    if CONTROL_PATH.exists() and not args.force:
        log.info("Control file already exists at %s; use --force to rebuild", CONTROL_PATH)
        return

    start = pd.Timestamp(args.start_year, 1, 1, tz="UTC")
    end = pd.Timestamp(args.end_year + 1, 1, 1, tz="UTC")
    timezone_map = amber_base.county_timezone_map(DATA_PROC / "county_pop_centroids.parquet")
    state_map = amber_base._state_county_map()
    controls, summary = aggregate_hour_controls(
        iter_cached_records(), timezone_map, state_map,
        start=start, end=end, max_records=args.max_records,
    )
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    controls.to_parquet(CONTROL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    log.info("Saved %s (%d county-hours) and %s", CONTROL_PATH, len(controls), SUMMARY_PATH)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
