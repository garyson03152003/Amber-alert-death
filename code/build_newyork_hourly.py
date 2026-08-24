"""Build NY county x date x HOUR crash panel.

NY DMV's Socrata feed carries `date` and `time` as separate, already-local,
naive fields (e.g. date="2021-01-01T00:00:00.000", time="12:00") -- no epoch,
no UTC, no timezone inference needed at all. This is the cleanest source of
any state built this session.

fetch_year in build_newyork_dot.py has signature (year, session), reversed
from every other builder's (session, year), so it is called directly rather
than through the shared build_state_hourly_panels.py SourceSpec framework.

Output: data/processed/ny_county_hour.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from build_newyork_dot import fetch_year, NY_COUNTY_FIPS
YEARS = [2021, 2022, 2023, 2024]
from build_state_hourly_panels import validate_diurnal_profile, reconcile_with_day_panel, SourceSpec

log = get_logger("ny_hourly")
OUT_PATH = DATA_PROC / "ny_county_hour.parquet"


def build() -> pd.DataFrame | None:
    session = requests.Session()
    frames = []
    for year in YEARS:
        try:
            raw = fetch_year(year, session)
        except Exception as exc:                                # noqa: BLE001
            log.warning("[NY] %d fetch failed: %s", year, exc)
            continue
        if raw is None or len(raw) == 0:
            log.warning("[NY] %d returned no rows", year)
            continue
        df = raw.copy()
        if "time" not in df.columns:
            log.warning("[NY] %d has no time column, skipping", year)
            continue
        local = pd.to_datetime(
            df["date"].astype(str).str.slice(0, 10) + " " + df["time"].astype(str),
            errors="coerce",
        )
        df = df.assign(_local=local).dropna(subset=["_local"])
        df["fips"] = df["county_name"].astype(str).str.strip().str.upper().map(NY_COUNTY_FIPS)
        df = df.dropna(subset=["fips"])
        df["date"] = df["_local"].dt.normalize()
        df["hour"] = df["_local"].dt.hour
        agg = (df.groupby(["fips", "date", "hour"], as_index=False)
                 .size().rename(columns={"size": "ny_crashes"}))
        frames.append(agg)
        log.info("[NY] %d -> %s county-hours", year, f"{len(agg):,}")

    if not frames:
        log.error("[NY] no data built")
        return None
    return (pd.concat(frames, ignore_index=True)
              .groupby(["fips", "date", "hour"], as_index=False)["ny_crashes"].sum())


def main() -> None:
    hourly = build()
    if hourly is None:
        return
    profile = validate_diurnal_profile(
        hourly["hour"].repeat(hourly["ny_crashes"].astype(int)), label="NY"
    )
    spec = SourceSpec(
        key="NY", module="build_newyork_dot", datetime_col="date", tz="America/New_York",
        datetime_kind="naive_local", county_mapper=("_", "county_name"),
        years=YEARS, crash_col="ny_crashes",
        day_panel="newyork_dot_county_day.parquet", day_crash_col="ny_crashes",
    )
    recon = reconcile_with_day_panel(hourly, spec)
    if not profile["plausible"]:
        log.error("[NY] REFUSING to write panel: diurnal check failed")
        return
    hourly.to_parquet(OUT_PATH, index=False)
    log.info("[NY] wrote %s rows -> %s (%s)", f"{len(hourly):,}", OUT_PATH, recon)


if __name__ == "__main__":
    main()
