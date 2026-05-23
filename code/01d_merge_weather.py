"""
01d_merge_weather.py
Merges daily weather into the county-day panel.

Adds columns:
  prcp_mm  — total daily precipitation (mm)
  tmax_c   — daily maximum temperature (°C)

Source: ACIS NWS 5-km gridded reanalysis at county centroid
        (output of 01c_fetch_weather.py)

Coverage check: prints counties without weather data so the analyst can
decide whether to impute or drop them.

Output: data/processed/panel_county_day_weather.parquet
        (panel with weather columns appended; rows are unchanged)
"""
import sys, warnings
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("merge_weather")

WEATHER_PATH = DATA_PROC / "weather_county_day.parquet"
PANEL_PATH   = DATA_PROC / "panel_county_day.parquet"
OUT_PATH     = DATA_PROC / "panel_county_day_weather.parquet"


def main():
    log.info("Loading panel…")
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Panel: %d rows, %d counties, %d–%d",
             len(panel), panel["fips"].nunique(),
             panel["year"].min(), panel["year"].max())

    log.info("Loading weather…")
    wx = pd.read_parquet(WEATHER_PATH)
    wx["date"] = pd.to_datetime(wx["date"])
    log.info("Weather: %d county-days, %d counties",
             len(wx), wx["fips"].nunique())

    # Coverage check
    panel_fips = set(panel["fips"].unique())
    wx_fips    = set(wx["fips"].unique())
    missing_wx = panel_fips - wx_fips
    if missing_wx:
        log.warning("%d panel counties lack weather data: %s%s",
                    len(missing_wx),
                    ", ".join(sorted(missing_wx)[:10]),
                    "…" if len(missing_wx) > 10 else "")

    # Merge
    log.info("Merging…")
    merged = panel.merge(wx[["fips","date","prcp_mm","tmax_c"]],
                         on=["fips","date"], how="left")

    cov = merged["prcp_mm"].notna().mean()
    log.info("Weather coverage after merge: %.1f%%", cov * 100)
    log.info("PRCP: mean=%.1f mm (non-missing)", merged["prcp_mm"].mean())
    log.info("TMAX: mean=%.1f °C (non-missing)", merged["tmax_c"].mean())

    merged.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s  (%d rows)", OUT_PATH, len(merged))


if __name__ == "__main__":
    main()
