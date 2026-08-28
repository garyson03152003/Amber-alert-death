"""
build_fars_road_type.py
=============================================================
Splits FARS fatal crashes by road type (highway vs non-highway), to test
whether the commuting-spillover effect is concentrated on the sustained,
monotonous highway/arterial driving a real commute involves, versus the
short, stop-and-go local-road driving that dominates own-county's
intra-county trips (mean ~4.6 miles) -- classic drowsy-driving research
points to sustained highway driving as the primary fatigue-crash
scenario, not short errands. If so, own-county's null result could
partly reflect that its "at-risk" population (residents who stay and
drive locally -- the only ones its own crash count can capture, since a
commuter who crashes OUT of county shows up in the destination county's
count instead, i.e. as cross_spillover) is mechanically doing the wrong
kind of driving for this mechanism to show up in fatal crashes, even if
they are equally fatigued.

Uses FARS's own FUNC_SYS field (verified present in the raw
accident.csv: Interstate, Principal Arterial - Other Freeways and
Expressways, Principal Arterial - Other, Minor Arterial, Major/Minor
Collector, Local). Reuses the exact validated geography/date/fatality
inclusion rules from build_fars_county_day.py / build_fars_hourly.py
(same permitted_fips_for_year crosswalk, same integrality checks) so a
crash is included here iff it would be included in the existing
validated panels -- only the aggregation adds a road-type split.

highway = Interstate + Principal Arterial - Other Freeways and Expressways
non_highway = everything else (arterials, collectors, local roads)

Output: data/processed/fars_road_type_county_day.parquet
  Columns: fips, date, hour, is_highway, person_fatals
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
OUT = DATA_PROC / "fars_road_type_county_day.parquet"

HIGHWAY_FUNC_SYS_NAMES = {
    "Interstate",
    "Principal Arterial - Other Freeways and Expressways",
}
# 2013-2014 use the older FHWA 16-category rural/urban-split scheme
# (ROAD_FNC) instead of the simplified 1-7 FUNC_SYS scheme adopted from
# 2015 onward (verified directly against both years' raw accident.csv):
# 1=Rural Interstate, 11=Urban Interstate, 12=Urban Freeway/Expressway
# are the highway-equivalent codes; everything else (rural/urban
# arterials, collectors, locals, unknown codes 9/19/99) is non-highway.
HIGHWAY_ROAD_FNC_CODES = {1, 11, 12}


def build_road_type_year(year: int, session: requests.Session) -> pd.DataFrame:
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

    events = (
        retained.groupby(["fips", "date", "hour", "is_highway"], as_index=False)
        .agg(person_fatals=("person_fatals", "sum"))
    )
    return events


def build_road_type(years: list[int] | None = None) -> pd.DataFrame:
    session = requests.Session()
    frames = [build_road_type_year(year, session) for year in (years or YEARS)]
    return pd.concat(frames, ignore_index=True).sort_values(["fips", "date", "hour"]).reset_index(drop=True)


def main() -> None:
    events = build_road_type()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUT, index=False)
    n_hw = events.loc[events["is_highway"], "person_fatals"].sum()
    n_non = events.loc[~events["is_highway"], "person_fatals"].sum()
    print(f"Saved {OUT} ({len(events):,} rows)")
    print(f"Highway fatalities: {n_hw:,} ({100*n_hw/(n_hw+n_non):.1f}%), non-highway: {n_non:,}")


if __name__ == "__main__":
    main()
