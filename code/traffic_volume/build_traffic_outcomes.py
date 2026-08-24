"""Construct traffic-volume outcomes from the station-hour panel (Task 3 of
TRAFFIC_VOLUME_INSTRUCTIONS.md).

Builds, per station-day:
  - total_volume        -- total station traffic volume that calendar day
  - vol_05_10           -- 05:00-10:00 (next-morning window)
  - vol_07_10           -- 07:00-10:00 (morning-commute window)
  - vol_10_16           -- 10:00-16:00 (midday window)
  - vol_16_19           -- 16:00-19:00 (evening window)

Each window sum uses ``min_count=1`` so an all-missing window stays NaN
rather than becoming a spurious zero (a station with zero *reported* hours
that day is not the same as a station that recorded zero vehicles).

The hourly event-study panel keeps one row per (station, date, hour) with
an `event_hour` column: hours-since-alert-issuance for the station's county,
relative to the alert's local hour on ``alert_time_local`` from
merge_alert_treatment.build_station_day_treatment. Only stations/dates with
a same-night verified alert get a non-null event_hour; this window is built
separately from the station-day aggregates per the instructions' requirement
to analyze the hourly event-study window explicitly rather than folding it
into the daily windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger

log = get_logger("build_traffic_outcomes")

ROOT = Path(__file__).resolve().parent.parent.parent
STATION_HOUR_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_hour.parquet"
TREATMENT_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_day_treatment.parquet"
OUT_DAY_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_day_outcomes.parquet"
OUT_EVENT_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_event_study_hours.parquet"

_WINDOWS = {
    "vol_05_10": (5, 10),
    "vol_07_10": (7, 10),
    "vol_10_16": (10, 16),
    "vol_16_19": (16, 19),
}
EVENT_STUDY_HALF_WIDTH = 12  # hours on either side of the alert hour


def _window_sum(df: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    """Sum traffic_volume for hour in [lo, hi), preserving all-missing as NaN."""
    sub = df[(df["hour"] >= lo) & (df["hour"] < hi)]
    return sub.groupby(["state_fips", "station_id", "date"])["traffic_volume"].agg(
        lambda s: s.sum(min_count=1)
    )


def build_station_day_outcomes(station_hour: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["state_fips", "station_id", "county_fips", "date"]
    base = station_hour[key_cols + ["latitude", "longitude", "source"]].drop_duplicates(
        subset=["state_fips", "station_id", "date"]
    )

    total = station_hour.groupby(["state_fips", "station_id", "date"])["traffic_volume"].agg(
        lambda s: s.sum(min_count=1)
    ).rename("total_volume")

    out = base.set_index(["state_fips", "station_id", "date"]).join(total)
    for name, (lo, hi) in _WINDOWS.items():
        out = out.join(_window_sum(station_hour, lo, hi).rename(name))

    n_hours = station_hour.groupby(["state_fips", "station_id", "date"])["hour"].nunique().rename("n_hours_reported")
    out = out.join(n_hours)
    return out.reset_index()


def build_event_study_hours(
    station_hour: pd.DataFrame, treatment: pd.DataFrame,
) -> pd.DataFrame:
    """Hourly volume in a window around alert issuance, for exposed county-days only."""
    exposed = treatment[treatment["night_alert_ct"] == 1][
        ["state_fips", "station_id", "county_fips", "date", "alert_hour_local", "exposure_class"]
    ].dropna(subset=["alert_hour_local"])
    if exposed.empty:
        return pd.DataFrame(columns=[
            "state_fips", "station_id", "county_fips", "date", "hour",
            "traffic_volume", "event_hour", "exposure_class",
        ])

    merged = station_hour.merge(
        exposed, on=["state_fips", "station_id", "county_fips", "date"], how="inner"
    )
    merged["event_hour"] = merged["hour"] - merged["alert_hour_local"]
    merged = merged[merged["event_hour"].abs() <= EVENT_STUDY_HALF_WIDTH]
    return merged[[
        "state_fips", "station_id", "county_fips", "date", "hour",
        "traffic_volume", "event_hour", "exposure_class",
    ]].reset_index(drop=True)


def main() -> None:
    if not STATION_HOUR_PATH.is_file():
        raise FileNotFoundError(f"station-hour panel not found at {STATION_HOUR_PATH}")
    if not TREATMENT_PATH.is_file():
        raise FileNotFoundError(f"station-day treatment not found at {TREATMENT_PATH}")

    station_hour = pd.read_parquet(STATION_HOUR_PATH)
    treatment = pd.read_parquet(TREATMENT_PATH)

    day_outcomes = build_station_day_outcomes(station_hour)
    day_outcomes = day_outcomes.merge(
        treatment.drop(columns=["state_fips"], errors="ignore"),
        on=["station_id", "county_fips", "date"], how="left",
    )
    OUT_DAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    day_outcomes.to_parquet(OUT_DAY_PATH, index=False)
    log.info("Wrote %s station-day outcome rows -> %s", f"{len(day_outcomes):,}", OUT_DAY_PATH)

    event_hours = build_event_study_hours(station_hour, treatment)
    event_hours.to_parquet(OUT_EVENT_PATH, index=False)
    log.info("Wrote %s event-study station-hour rows -> %s", f"{len(event_hours):,}", OUT_EVENT_PATH)


if __name__ == "__main__":
    main()
