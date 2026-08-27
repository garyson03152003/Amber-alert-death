"""
build_county_pop_centroids.py
=============================================================
Fetches the Census Bureau's official 2020 population-weighted mean
center of population for every US county, as a more accurate distance
anchor than a geometric centroid for commuting-distance measures.

Why this matters: the repo's existing data/processed/county_centroids.parquet
is a GEOMETRIC centroid (the geographic middle of the county polygon) and
only covers 1,646/3,144 commuting-network counties. For large, unevenly
populated counties -- especially in the rural West -- the geometric
center can sit 60-100+ miles from where people actually live (e.g. Nye
County NV: 108.6 mi off; Washoe County NV: 82.9 mi off; San Bernardino
County CA: 79.0 mi off -- verified interactively before writing this
script). Since commuting distance is exactly the measure
run_commuting_distance_robustness.py uses to test whether the
spillover effect is a proximity artifact, an inaccurate/incomplete
centroid source directly weakens that test. The population-weighted
center of population is the standard Census Bureau alternative: the
point where the county's population "balances" if every resident weighs
the same -- i.e. close to where people actually live, not the polygon's
geometric middle.

Source:
  2020 Census Mean Center of Population by county
  https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt
  Columns: STATEFP, COUNTYFP, COUNAME, STNAME, POPULATION, LATITUDE, LONGITUDE
  Full national coverage: 3,221 counties/county-equivalents.

Output: data/processed/county_pop_centroids.parquet
  Columns: fips (str5), lat, lon, population
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC
from utils import get_logger

log = get_logger("county_pop_centroids")

URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
RAW_PATH = CROSSWALK_RAW / "CenPop2020_Mean_CO.txt"
OUT_PATH = DATA_PROC / "county_pop_centroids.parquet"


def download_if_missing():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        log.info("Already downloaded: %s", RAW_PATH)
        return
    log.info("Downloading 2020 Census population-weighted county centroids...")
    urllib.request.urlretrieve(URL, RAW_PATH)
    log.info("Saved to %s", RAW_PATH)


def build():
    download_if_missing()
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig",
                      dtype={"STATEFP": str, "COUNTYFP": str})
    raw["fips"] = raw["STATEFP"].str.zfill(2) + raw["COUNTYFP"].str.zfill(3)
    out = raw.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon",
                              "POPULATION": "population"})[["fips", "lat", "lon", "population"]]
    out = out.drop_duplicates(subset="fips").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d counties -> %s", len(out), OUT_PATH)
    return out


if __name__ == "__main__":
    build()
