"""Build a year-accurate county FIPS crosswalk from Census Gazetteer files.

The Census Population Estimates Program backcasts historical population rows
onto *current* county boundaries, so it cannot tell whether an older FIPS code
(for example South Dakota's Shannon County, `46113`, renamed Oglala Lakota
County `46102` in 2015) was genuinely in effect in a given year. The annual
Gazetteer county file records the geography as it actually stood that year and
is the correct reference for validating historical FARS records. This module
is deliberately import-safe: downloads happen only through :func:`main` or an
explicit build function, never at import time.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
CROSSWALK_OUT = DATA_PROC / "county_fips_crosswalk.parquet"
YEARS = tuple(range(2013, 2025))
GAZETTEER_URL_TEMPLATE = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "{year}_Gazetteer/{year}_Gaz_counties_national.zip"
)


def fetch_gazetteer_fips(year: int, session: requests.Session) -> set[str]:
    """Return the county-equivalent FIPS set Census recognized for one year."""
    url = GAZETTEER_URL_TEMPLATE.format(year=year)
    response = session.get(url, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    name = archive.namelist()[0]
    df = pd.read_csv(io.BytesIO(archive.read(name)), sep="\t", encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    if "GEOID" not in df.columns:
        raise ValueError(f"Gazetteer {year} file missing GEOID column")
    fips = df["GEOID"].astype(str).str.zfill(5)
    return set(fips)


def build_crosswalk(years: Iterable[int] = YEARS) -> pd.DataFrame:
    """Download every requested year and assemble a long fips/year table."""
    session = requests.Session()
    rows = []
    for year in years:
        fips_set = fetch_gazetteer_fips(int(year), session)
        if not fips_set:
            raise ValueError(f"Gazetteer {year} returned no counties")
        rows.append(pd.DataFrame({"fips": sorted(fips_set), "year": int(year)}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    crosswalk = build_crosswalk()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(CROSSWALK_OUT, index=False)
    print(f"Saved {CROSSWALK_OUT} ({len(crosswalk):,} year/county rows, "
          f"{crosswalk['year'].nunique()} years)")


if __name__ == "__main__":
    main()
