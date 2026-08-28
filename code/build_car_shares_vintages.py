"""Build county car-commute shares (ACS B08301) for several vintages, so each
crash year can be matched to the nearest measurement.

Two Census formats, one per era
-------------------------------
    <= 2020  legacy summary file: 51 state zips, table located by a
             vintage-specific sequence number (see 01e_fetch_car_commuters.py,
             which resolves the sequence from Census's own lookup because it
             is NOT stable -- B08301 is sequence 0027 in 2020 but 0028 in 2015)
    >= 2021  table-based summary file: ONE pipe-delimited .dat per table,
             columns B08301_E001.. (estimates) and B08301_M001.. (margins)

This module handles the modern format and stitches all available vintages
into a single lookup.

Why several vintages
--------------------
ACS 5-year car shares are not static. Between the 2011-2015 and 2016-2020
windows the worker-weighted national car share fell 0.859 -> 0.838, and the
decline is concentrated in large white-collar metros (King County WA
0.743 -> 0.676; Fulton County GA 0.808 -> 0.744) as remote work grew. Small
counties mostly show sampling noise -- correlation between the two vintages
rises from 0.895 across all counties to 0.988 for counties with >=100k
workers -- so the movement that matters is a real trend in big counties, not
churn in small ones.

Note the windows OVERLAP by four years, so adjacent vintages are far from
independent: the 2016 and 2017 files share four-fifths of their sample.
Fetching every year buys little over fetching roughly every fourth, which is
why the default set is spaced rather than exhaustive.

Output: data/processed/county_car_commuters_{vintage}.parquet
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

log = get_logger("car_shares_vintages")

TABLE_BASED_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/{yr}/"
    "table-based-SF/data/5YRData/acsdt5y{yr}-b08301.dat"
)
COUNTY_PREFIX = "0500000US"

# crash year -> car-share vintage (the 5-year window containing or nearest it)
VINTAGE_FOR_YEAR = {
    **{y: "2015" for y in range(2013, 2016)},   # window 2011-2015
    **{y: "2017" for y in range(2016, 2018)},   # window 2013-2017
    **{y: "2020" for y in range(2018, 2021)},   # window 2016-2020
    **{y: "2023" for y in range(2021, 2026)},   # window 2019-2023
}


def fetch_table_based(vintage: int) -> pd.DataFrame:
    """Parse one modern (2021+) B08301 table into county car shares."""
    url = TABLE_BASED_URL.format(yr=vintage)
    log.info("[%d] downloading table-based B08301", vintage)
    with urllib.request.urlopen(url, timeout=600) as r:
        raw = r.read()
    log.info("[%d] %.1f MB", vintage, len(raw) / 1e6)

    # Only estimate columns are needed; margins are ignored.
    usecols = ["GEO_ID", "B08301_E001", "B08301_E002", "B08301_E003", "B08301_E010"]
    df = pd.read_csv(io.StringIO(raw.decode("latin-1")), sep="|",
                     usecols=lambda c: c in usecols, dtype=str, low_memory=False)

    df = df[df["GEO_ID"].astype(str).str.startswith(COUNTY_PREFIX)].copy()
    df["fips"] = df["GEO_ID"].str[len(COUNTY_PREFIX):].str.zfill(5)
    for c in ("B08301_E001", "B08301_E002", "B08301_E003", "B08301_E010"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    out = df.rename(columns={
        "B08301_E001": "total_workers", "B08301_E002": "car_total",
        "B08301_E003": "drove_alone", "B08301_E010": "carpooled",
    })[["fips", "total_workers", "car_total", "drove_alone", "carpooled"]]
    out = out[out["total_workers"] > 0].copy()
    out["car_share"] = out["car_total"] / out["total_workers"]
    log.info("[%d] %s counties | worker-weighted car share %.4f", vintage,
             f"{len(out):,}",
             (out["car_share"] * out["total_workers"]).sum() / out["total_workers"].sum())
    return out


def load_car_share_for_year(year: int) -> pd.DataFrame:
    """Car shares from the vintage nearest ``year``."""
    vintage = VINTAGE_FOR_YEAR.get(year, "2020")
    path = DATA_PROC / (
        "county_car_commuters.parquet" if vintage == "2020"
        else f"county_car_commuters_{vintage}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"car-share vintage {vintage} not built: {path}")
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vintages", nargs="*", type=int, default=[2023],
                    help="modern (2021+) vintages to fetch")
    args = ap.parse_args()
    for v in args.vintages:
        out = fetch_table_based(v)
        path = DATA_PROC / f"county_car_commuters_{v}.parquet"
        out.to_parquet(path, index=False)
        log.info("[%d] wrote -> %s", v, path)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
