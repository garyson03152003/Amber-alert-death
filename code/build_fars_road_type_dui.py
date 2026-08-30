"""
build_fars_road_type_dui.py
=============================================================
Combines build_fars_road_type.py's highway/non-highway split and
build_fars_dui.py's drunk/sober split into ONE hourly extract, so the H1
same-hour highway-fatals effect (reg_same_hour_road_type_split.csv:
beta=-0.000175, p=6.7e-06) can be tested for DUI concentration WITHIN
highway crashes specifically -- not pooled across road types, where the
headline effect (highway-only, negative) would be diluted/offset by the
non-highway coefficient (positive, non-significant) exactly the way a
first pass at the plain drunk/sober split (no road-type control) came
back with both drunk_fatals and sober_fatals showing the wrong (positive)
sign relative to the highway-only headline number.

Reuses the exact validated geography/date/fatality/hour inclusion rules
from build_fars_county_day.py / build_fars_hourly.py / build_fars_road_
type.py / build_fars_dui.py (same permitted_fips_for_year crosswalk,
same integrality checks, same FUNC_SYS highway definition, same
vehicle-level DR_DRINK==1 alcohol-involvement definition) -- only the
aggregation now crosses BOTH splits at once, one download pass per year
instead of two.

Output: data/processed/fars_road_type_dui_county_day.parquet
  Columns: fips, date, hour, is_highway, is_drunk, person_fatals
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
from build_fars_road_type import HIGHWAY_FUNC_SYS_NAMES, HIGHWAY_ROAD_FNC_CODES

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT = DATA_PROC / "fars_road_type_dui_county_day.parquet"


def build_year(year: int, session: requests.Session) -> pd.DataFrame:
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

    if "FUNC_SYSNAME" in retained.columns:
        retained["is_highway"] = retained["FUNC_SYSNAME"].isin(HIGHWAY_FUNC_SYS_NAMES)
    elif "ROAD_FNC" in retained.columns:
        road_fnc = _numeric(retained, "ROAD_FNC")
        retained["is_highway"] = road_fnc.isin(HIGHWAY_ROAD_FNC_CODES)
    else:
        raise ValueError(f"Neither FUNC_SYSNAME nor ROAD_FNC present in {year} accident.csv")

    drunk_cases = _drunk_cases(read_file(archive, "vehicle"), int(year))
    retained["is_drunk"] = retained["ST_CASE"].isin(drunk_cases)

    events = (
        retained.groupby(["fips", "date", "hour", "is_highway", "is_drunk"], as_index=False)
        .agg(person_fatals=("person_fatals", "sum"))
    )
    return events


def build(years: list[int] | None = None) -> pd.DataFrame:
    session = requests.Session()
    frames = [build_year(year, session) for year in (years or YEARS)]
    return pd.concat(frames, ignore_index=True).sort_values(["fips", "date", "hour"]).reset_index(drop=True)


def main() -> None:
    events = build()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUT, index=False)
    tot = events["person_fatals"].sum()
    for hw in (True, False):
        for dr in (True, False):
            n = events.loc[(events.is_highway == hw) & (events.is_drunk == dr), "person_fatals"].sum()
            print(f"  highway={hw!s:5} drunk={dr!s:5}: {n:,} ({100*n/tot:.1f}%)")
    print(f"Saved {OUT} ({len(events):,} rows, {tot:,} total fatalities)")


if __name__ == "__main__":
    main()
