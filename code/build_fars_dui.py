"""
build_fars_dui.py
=============================================================
Splits FARS fatal crashes by alcohol involvement (drunk vs sober), at
HOURLY resolution, to test whether the H1 same-hour highway-fatals
effect (reg_same_hour_road_type_split.csv: beta=-0.000175, p=6.7e-06)
concentrates on DUI-adjacent crashes -- consistent with heightened
public vigilance / police presence / traffic stops during an active
AMBER alert search -- versus holding for sober driving too, which would
argue against a DUI-deterrence-specific channel and toward a broader
attention/distraction mechanism.

Reuses the exact validated geography/date/fatality/hour inclusion rules
from build_fars_hourly.py / build_fars_road_type.py (same
permitted_fips_for_year crosswalk, same integrality checks), so a crash
is included here iff it would be included in the existing validated
hourly panel -- only the aggregation adds a drunk/sober split. Alcohol
involvement uses the same vehicle-level DR_DRINK==1 definition as
build_fars_county_day.py's _drunk_cases() (the daily county-level
drunk_fatals/sober_fatals split already used elsewhere in this repo,
e.g. run_fars_national_analysis.py's drunk_fatals_per_100k /
sober_fatals_per_100k outcomes) -- a crash is "drunk" if ANY involved
vehicle had a drinking driver, matching FARS's own case-level standard.

Output: data/processed/fars_dui_county_day.parquet
  Columns: fips, date, hour, is_drunk, person_fatals
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
    _drunk_cases,
    _numeric,
    _required_columns,
    clean_cols,
    fetch_zip,
    permitted_fips_for_year,
    read_file,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT = DATA_PROC / "fars_dui_county_day.parquet"


def build_dui_year(year: int, session: requests.Session) -> pd.DataFrame:
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

    drunk_cases = _drunk_cases(read_file(archive, "vehicle"), int(year))
    retained["is_drunk"] = retained["ST_CASE"].isin(drunk_cases)

    events = (
        retained.groupby(["fips", "date", "hour", "is_drunk"], as_index=False)
        .agg(person_fatals=("person_fatals", "sum"))
    )
    return events


def build_dui(years: list[int] | None = None) -> pd.DataFrame:
    session = requests.Session()
    frames = [build_dui_year(year, session) for year in (years or YEARS)]
    return pd.concat(frames, ignore_index=True).sort_values(["fips", "date", "hour"]).reset_index(drop=True)


def main() -> None:
    events = build_dui()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUT, index=False)
    n_drunk = events.loc[events["is_drunk"], "person_fatals"].sum()
    n_sober = events.loc[~events["is_drunk"], "person_fatals"].sum()
    print(f"Saved {OUT} ({len(events):,} rows)")
    print(f"Drunk fatalities: {n_drunk:,} ({100*n_drunk/(n_drunk+n_sober):.1f}%), sober: {n_sober:,}")


if __name__ == "__main__":
    main()
