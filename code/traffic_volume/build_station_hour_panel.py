"""Build a station x calendar-date x hour traffic volume panel from
downloaded FHWA TMAS files (see download_tmas.py).

Per-month volume records are aggregated across travel direction/lane to one
row per (station_id, date, hour) -- the required schema does not separate
by direction/lane. Station metadata (county FIPS, lat/lon) is joined from
the year's station-description file. Processing is done year-by-year to
bound memory, writing one intermediate parquet per year before
concatenating.

Preferred output: data/processed/traffic/tmas_station_hour.parquet
Coverage report: output/tables/traffic_counter_coverage.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from parse_tmas import read_vol_zip, read_sta_zip
from utils import get_logger

log = get_logger("build_station_hour_panel")

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw" / "tmas"
OUT_DIR = ROOT / "data" / "processed" / "traffic"
OUT_PATH = OUT_DIR / "tmas_station_hour.parquet"
COVERAGE_PATH = ROOT / "output" / "tables" / "traffic_counter_coverage.csv"

MONTHS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
# VOL format is auto-detected per zip in parse_tmas.read_vol_zip -- there is
# no clean year cutoff (2020 is still legacy fixed-width, 2021 is a distinct
# headerless-pipe format, only 2022+ has the documented header+pipe layout).


def _melt_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the 24-element `hours` list column into long (hour, volume) rows."""
    if df.empty:
        return pd.DataFrame(columns=["state_fips", "station_id", "year", "month", "day", "hour", "traffic_volume"])
    base = df[["state_fips", "station_id", "year", "month", "day"]].copy()
    hours_matrix = pd.DataFrame(df["hours"].tolist(), index=df.index)
    hours_matrix.columns = range(24)
    long = hours_matrix.stack()  # pandas 3.0's stack() never drops NaN rows
    long.index.set_names(["_row", "hour"], inplace=True)
    long = long.rename("traffic_volume").reset_index()
    out = base.loc[long["_row"]].reset_index(drop=True)
    out["hour"] = long["hour"].to_numpy()
    out["traffic_volume"] = long["traffic_volume"].to_numpy()
    return out


def process_year(year: int) -> pd.DataFrame | None:
    year_dir = RAW_DIR / str(year)
    station_zip = year_dir / f"{year}_station_data.zip"
    if not station_zip.is_file():
        log.warning("[%d] no station data downloaded, skipping year", year)
        return None
    stations = read_sta_zip(station_zip)
    if stations.empty:
        log.warning("[%d] station file parsed empty, skipping year", year)
        return None
    stations = stations.drop_duplicates(subset=["state_fips", "station_id"])
    stations["county_fips_full"] = (
        stations["state_fips"].str.zfill(2) + stations["county_fips"].str.zfill(3)
    )
    station_lookup = stations.set_index(["state_fips", "station_id"])[
        ["county_fips_full", "latitude", "longitude"]
    ]

    monthly_parts = []
    for month in MONTHS:
        vol_zip = year_dir / f"{month}_{year}_ccs_data.zip"
        if not vol_zip.is_file():
            log.warning("[%d-%s] no volume data downloaded, skipping month", year, month)
            continue
        raw = read_vol_zip(vol_zip)
        if raw.empty:
            log.warning("[%d-%s] volume file parsed empty", year, month)
            continue
        raw["state_fips"] = raw["state_fips"].astype(str).str.zfill(2)
        raw["station_id"] = raw["station_id"].astype(str).str.strip()
        long = _melt_hours(raw)
        long["source"] = "TMAS"
        agg = (
            long.groupby(["state_fips", "station_id", "year", "month", "day", "hour"], as_index=False)
                .agg(traffic_volume=("traffic_volume", lambda s: s.sum(min_count=1)))
        )
        agg["source"] = "TMAS"
        monthly_parts.append(agg)
        log.info("  [%d-%s] %s station-hour rows", year, month, f"{len(agg):,}")

    if not monthly_parts:
        return None
    year_panel = pd.concat(monthly_parts, ignore_index=True)
    year_panel = year_panel.join(
        station_lookup, on=["state_fips", "station_id"]
    )
    year_panel["date"] = pd.to_datetime(
        dict(year=year_panel["year"], month=year_panel["month"], day=year_panel["day"]),
        errors="coerce",
    )
    n_bad_date = year_panel["date"].isna().sum()
    if n_bad_date:
        log.warning("[%d] %d rows with invalid calendar date dropped", year, n_bad_date)
    year_panel = year_panel.dropna(subset=["date"])
    year_panel = year_panel.rename(columns={"county_fips_full": "county_fips"})
    return year_panel[[
        "state_fips", "station_id", "county_fips", "date", "hour",
        "traffic_volume", "latitude", "longitude", "source",
    ]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    years = sorted(int(p.name) for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    coverage_rows = []
    parts = []
    for year in years:
        log.info("=== %d ===", year)
        panel = process_year(year)
        if panel is None:
            coverage_rows.append({"year": year, "n_stations": 0, "n_counties": 0, "n_rows": 0})
            continue
        parts.append(panel)
        coverage_rows.append({
            "year": year,
            "n_stations": panel["station_id"].nunique(),
            "n_counties": panel["county_fips"].nunique(),
            "n_rows": len(panel),
            "share_missing_volume": float(panel["traffic_volume"].isna().mean()),
        })

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(COVERAGE_PATH, index=False)
    log.info("Wrote coverage summary -> %s", COVERAGE_PATH)

    if not parts:
        log.error("No years produced data -- aborting without writing panel.")
        sys.exit(1)

    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s rows -> %s", f"{len(full):,}", OUT_PATH)


if __name__ == "__main__":
    main()
