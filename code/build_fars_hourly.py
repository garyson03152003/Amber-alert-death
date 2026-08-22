"""Build an hour-level validated FARS fatal-crash panel for time-window analysis.

This exists to test the "distraction" mechanism (does an AMBER alert coincide
with a same-night spike in nearby crashes, rather than only a next-morning
one) which requires the hour of each crash, not just the date. It reuses the
exact geography/date/fatality validity rules from ``build_fars_county_day.py``
(the same crosswalk-corrected ``permitted_fips_for_year``, the same
unresolvable-vs-invalid split) so a row is included here if and only if it
would be included in the already-validated county-day panel -- this module
only changes the aggregation grain, never the inclusion criteria.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from build_fars_county_day import (
    US_STATE_FIPS,
    YEARS,
    _numeric,
    _required_columns,
    clean_cols,
    fetch_zip,
    permitted_fips_for_year,
    read_file,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT = DATA_PROC / "fars_hourly_county_day.parquet"


def build_fars_hourly_year(year: int, session: requests.Session) -> pd.DataFrame:
    """Return validated fatal-crash rows for one year, retaining HOUR and ST_CASE."""
    archive, _ = fetch_zip(int(year), session)
    accidents = clean_cols(read_file(archive, "accident"))
    _required_columns(
        accidents, {"ST_CASE", "STATE", "COUNTY", "YEAR", "MONTH", "DAY", "HOUR", "FATALS"},
        int(year), "accident",
    )
    accidents = accidents.drop_duplicates(subset=["ST_CASE"], keep="first").copy()

    state = _numeric(accidents, "STATE")
    county = _numeric(accidents, "COUNTY")
    state_integral = state.notna() & state.eq(state.round())
    county_integral = county.notna() & county.eq(county.round())
    state_code = state.where(state_integral).astype("Int64").astype(str).str.zfill(2)
    county_code = county.where(county_integral).astype("Int64").astype(str).str.zfill(3)
    fips = state_code + county_code
    geography_valid = (
        state_integral & county_integral
        & state_code.isin(US_STATE_FIPS)
        & county.between(1, 998)
        & fips.isin(permitted_fips_for_year(int(year)))
    )
    archive_year = _numeric(accidents, "YEAR")
    date = pd.to_datetime(
        {"year": archive_year, "month": _numeric(accidents, "MONTH"), "day": _numeric(accidents, "DAY")},
        errors="coerce",
    )
    date_valid = date.notna() & archive_year.eq(int(year)) & date.dt.year.eq(int(year))
    fatalities = _numeric(accidents, "FATALS")
    fatality_valid = fatalities.notna() & fatalities.ge(1)
    hour = _numeric(accidents, "HOUR")
    hour_valid = hour.notna() & hour.between(0, 23)

    retained = accidents.loc[geography_valid & date_valid & fatality_valid & hour_valid].copy()
    retained["date"] = date.loc[retained.index].dt.normalize()
    retained["hour"] = hour.loc[retained.index].astype(int)
    retained["fips"] = fips.loc[retained.index]
    retained["person_fatals"] = fatalities.loc[retained.index].astype(int)

    persons = clean_cols(read_file(archive, "person"))
    _required_columns(persons, {"ST_CASE", "INJ_SEV"}, int(year), "person")
    serious = _numeric(persons, "INJ_SEV").eq(3)
    serious_counts = (
        persons.loc[serious, "ST_CASE"].value_counts().rename("serious_inj")
    )
    retained = retained.join(serious_counts, on="ST_CASE")
    retained["serious_inj"] = retained["serious_inj"].fillna(0).astype(int)

    events = (
        retained.groupby(["fips", "date", "hour"], as_index=False)
        .agg(fatal_crashes=("ST_CASE", "nunique"), person_fatals=("person_fatals", "sum"),
             serious_inj=("serious_inj", "sum"))
    )
    return events


def build_fars_hourly(years: list[int] | None = None) -> pd.DataFrame:
    session = requests.Session()
    frames = [build_fars_hourly_year(year, session) for year in (years or YEARS)]
    return pd.concat(frames, ignore_index=True).sort_values(["fips", "date", "hour"]).reset_index(drop=True)


def main() -> None:
    events = build_fars_hourly()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUT, index=False)
    print(f"Saved {OUT} ({len(events):,} fatal-event county-date-hours, "
          f"{events['fips'].nunique():,} counties)")


if __name__ == "__main__":
    main()
