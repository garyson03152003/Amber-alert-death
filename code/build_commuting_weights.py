"""
build_commuting_weights.py
Builds county-to-county commuting weight matrix from ACS 2016-2020 flows.

Source:
  ACS 5-Year County-to-County Commuting Flows, 2016-2020
  https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2020/
         commuting-flows-2020/table1.xlsx

Output:
  data/processed/commuting/county_commuting_weights.parquet
  Columns: fips_home (int), fips_work (int), workers (float), weight (float)
  weight = fraction of fips_work's workforce that commutes from fips_home.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

log = get_logger("build_commuting_weights")

ACS_URL = (
    "https://www2.census.gov/programs-surveys/demo/tables/metro-micro/"
    "2020/commuting-flows-2020/table1.xlsx"
)
DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "commuting"
XLSX_PATH = DATA_DIR / "acs_commuting_flows_2020.xlsx"
OUT_PATH  = DATA_DIR / "county_commuting_weights.parquet"


def download_if_missing():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if XLSX_PATH.exists():
        log.info("ACS Excel already present: %s", XLSX_PATH)
        return
    import urllib.request
    log.info("Downloading ACS commuting flows (~6 MB)…")
    urllib.request.urlretrieve(ACS_URL, XLSX_PATH)
    log.info("Saved to %s", XLSX_PATH)


def build():
    download_if_missing()

    log.info("Parsing ACS commuting flows Excel…")
    raw = pd.read_excel(XLSX_PATH, skiprows=7, header=0)
    raw = raw.dropna(subset=["Workers in Commuting Flow"])
    raw = raw.dropna(subset=["State FIPS Code.1", "County FIPS Code.1"])

    # Build integer 5-digit FIPS
    raw["fips_home"] = (
        raw["State FIPS Code"].astype(float).astype(int) * 1000 +
        raw["County FIPS Code"].astype(float).astype(int)
    )
    raw["fips_work"] = (
        raw["State FIPS Code.1"].astype(float).astype(int) * 1000 +
        raw["County FIPS Code.1"].astype(float).astype(int)
    )
    raw["workers"] = raw["Workers in Commuting Flow"].astype(float)

    # 50 states + DC only (exclude territories: FIPS state > 56)
    state_h = raw["fips_home"] // 1000
    state_w = raw["fips_work"] // 1000
    raw = raw[(state_h <= 56) & (state_w <= 56)].copy()

    # Weights: fraction of work county's inflow from each home county
    work_totals = raw.groupby("fips_work")["workers"].sum().rename("total_to_work")
    raw = raw.merge(work_totals, on="fips_work")
    raw["weight"] = raw["workers"] / raw["total_to_work"]

    out = raw[["fips_home", "fips_work", "workers", "weight"]].copy()
    out.to_parquet(OUT_PATH, index=False)

    log.info("Saved %d OD pairs → %s", len(out), OUT_PATH)
    log.info("Work counties: %d | Home counties: %d",
             out["fips_work"].nunique(), out["fips_home"].nunique())

    # Diagnostics
    check = out.groupby("fips_work")["weight"].sum()
    log.info("Weight sums: min=%.4f max=%.4f (should be 1.0)",
             check.min(), check.max())
    own = out[out["fips_home"] == out["fips_work"]]
    log.info("Own-county weight: median=%.3f mean=%.3f",
             own["weight"].median(), own["weight"].mean())


if __name__ == "__main__":
    build()
