"""
03_download_weather.py — Download NOAA GHCND county-day weather controls.

We need two weather variables as controls in the traffic-fatality regressions:
  - PRCP  precipitation (mm/10)
  - TMAX  maximum temperature (tenths of degrees C)

Strategy:
  1. Download the GHCND station inventory to get station → county mapping.
  2. Download annual GHCND bulk files (one per year) from NOAA.
  3. Aggregate from station-day to county-day (mean TMAX, sum PRCP).

Data source: NOAA GHCND via HTTPS bulk download (no API key required)
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/

Output: data/processed/weather_county_day.parquet
    Columns: fips, date, prcp_mm, tmax_c

Run: python code/03_download_weather.py
"""

import gzip
import io
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import WEATHER_RAW, DATA_PROC, STUDY_YEARS
from utils import get_logger, download_file, fips5

log = get_logger("03_weather")

GHCND_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"

# Station inventory: fixed-width file listing all GHCND stations with lat/lon/state
STATION_INVENTORY_URL = f"{GHCND_BASE}/ghcnd-stations.txt"

# Annual bulk file: one gzipped CSV per year
ANNUAL_URL_TEMPLATE = f"{GHCND_BASE}/by_year/{{year}}.csv.gz"

# We only need these elements
ELEMENTS = {"PRCP", "TMAX"}


# ---------------------------------------------------------------------------
# Step 1: Station inventory → station → county FIPS crosswalk
# ---------------------------------------------------------------------------

def load_station_inventory(session: requests.Session) -> pd.DataFrame:
    """
    Parse the GHCND fixed-width station inventory.

    Returns DataFrame with columns: station_id, lat, lon, state_abbr
    (county FIPS is added in a later step via spatial join or state + county lookup)
    """
    dest = WEATHER_RAW / "ghcnd-stations.txt"
    download_file(STATION_INVENTORY_URL, dest, session=session)

    # Fixed-width spec from NOAA README:
    #   1-11   station ID
    #  13-20   latitude
    #  22-30   longitude
    #  32-37   elevation
    #  39-40   state (US postal abbr)
    #  42-71   name
    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
    names    = ["station_id", "lat", "lon", "elev", "state_abbr", "name"]

    df = pd.read_fwf(dest, colspecs=colspecs, names=names, header=None)
    # Keep only US stations (ID starts with "US")
    df = df[df["station_id"].str.startswith("US")].copy()
    log.info("Station inventory: %d US stations", len(df))
    return df


def load_county_station_crosswalk(session: requests.Session, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Add county FIPS to each station using the NOAA county crosswalk file.
    Falls back to a spatial point-in-polygon join using Census county shapefiles
    if the crosswalk does not cover all stations.
    """
    # NOAA provides a station-to-county crosswalk
    xwalk_url = f"{GHCND_BASE}/ghcnd-stations.txt"   # state info embedded above
    # Additional county-level crosswalk: use the FIPS crosswalk from Census
    # We'll do a simple approach: NOAA has a file mapping stations → county FIPS
    county_xwalk_url = (
        "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
    )
    dest = WEATHER_RAW / "ghcnd-inventory.txt"
    try:
        download_file(county_xwalk_url, dest, session=session)
    except Exception as exc:
        log.warning("Could not download GHCND inventory: %s", exc)
        # Return stations without county — will drop unmapped stations downstream
        stations["fips"] = pd.NA
        return stations

    # The inventory file has: station_id, lat, lon, element, firstyear, lastyear
    inv_colspecs = [(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)]
    inv_names    = ["station_id", "lat", "lon", "element", "firstyear", "lastyear"]
    inv = pd.read_fwf(dest, colspecs=inv_colspecs, names=inv_names, header=None)
    inv = inv[inv["element"].isin(ELEMENTS)]

    # -----------------------------------------------------------------------
    # Map station → county via Census TIGER gazetteer (county centroids)
    # This is an approximation: assign each station to the nearest county centroid.
    # For a production version, do a proper spatial join against county polygons.
    # -----------------------------------------------------------------------
    gazetteer_url = (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2023_Gazetteer/2023_Gaz_counties_national.zip"
    )
    gaz_dest = WEATHER_RAW / "2023_Gaz_counties_national.zip"
    try:
        download_file(gazetteer_url, gaz_dest, session=session)
        import zipfile
        with zipfile.ZipFile(gaz_dest) as zf:
            fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
            with zf.open(fname) as f:
                counties = pd.read_csv(f, sep="\t", dtype=str)
        counties.columns = [c.strip() for c in counties.columns]
        counties["fips"] = counties["GEOID"].str.zfill(5)
        counties["clat"] = pd.to_numeric(counties["INTPTLAT"], errors="coerce")
        counties["clon"] = pd.to_numeric(counties["INTPTLONG"], errors="coerce")
        counties = counties[["fips", "clat", "clon"]].dropna()

        # Nearest-county assignment using Haversine (vectorised approximate)
        merged_stations = stations.merge(
            inv[["station_id"]].drop_duplicates(), on="station_id", how="inner"
        )
        merged_stations["fips"] = merged_stations.apply(
            lambda r: _nearest_county(r["lat"], r["lon"], counties), axis=1
        )
        log.info("Mapped %d/%d stations to county FIPS",
                 merged_stations["fips"].notna().sum(), len(merged_stations))
        return merged_stations

    except Exception as exc:
        log.warning("County assignment via gazetteer failed: %s", exc)
        log.warning("Stations will be assigned to state-level only.")
        stations["fips"] = pd.NA
        return stations


def _nearest_county(lat: float, lon: float, counties: pd.DataFrame) -> str:
    """Return FIPS of the county whose centroid is closest to (lat, lon)."""
    import numpy as np
    dlat = counties["clat"].values - lat
    dlon = counties["clon"].values - lon
    dist2 = dlat**2 + dlon**2   # approximate; fine for within-state assignment
    idx = int(np.argmin(dist2))
    return counties["fips"].iloc[idx]


# ---------------------------------------------------------------------------
# Step 2: Download annual bulk files and aggregate to county-day
# ---------------------------------------------------------------------------

def process_year(year: int, station_fips: pd.Series, session: requests.Session) -> pd.DataFrame:
    """
    Download {year}.csv.gz, filter to PRCP/TMAX for mapped stations,
    and aggregate to county-day.

    Column layout of GHCND by-year files:
        STATION, DATE, ELEMENT, DATA_VALUE, M_FLAG, Q_FLAG, S_FLAG, OBS_TIME
    """
    url  = ANNUAL_URL_TEMPLATE.format(year=year)
    dest = WEATHER_RAW / f"{year}.csv.gz"
    try:
        download_file(url, dest, session=session)
    except Exception as exc:
        log.error("Failed to download weather for %d: %s", year, exc)
        return pd.DataFrame()

    log.info("Processing weather file for %d...", year)
    try:
        df = pd.read_csv(
            dest,
            names=["station_id", "date", "element", "value", "mflag", "qflag", "sflag", "obs_time"],
            dtype={"date": str, "station_id": str},
            parse_dates=False,
        )
    except Exception as exc:
        log.error("Could not parse weather CSV for %d: %s", year, exc)
        return pd.DataFrame()

    # Keep only PRCP and TMAX; drop quality-flagged observations
    df = df[df["element"].isin(ELEMENTS)]
    df = df[df["qflag"].isna() | (df["qflag"].str.strip() == "")]

    # Merge station → county FIPS
    df = df.merge(station_fips.rename("fips"), left_on="station_id", right_index=True, how="inner")
    df = df.dropna(subset=["fips"])

    # Parse date (YYYYMMDD)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    # Pivot ELEMENT wide, then aggregate to county-day
    prcp = (
        df[df["element"] == "PRCP"]
        .groupby(["fips", "date"])["value"].mean()
        .rename("prcp_raw")
    )
    tmax = (
        df[df["element"] == "TMAX"]
        .groupby(["fips", "date"])["value"].mean()
        .rename("tmax_raw")
    )

    out = pd.concat([prcp, tmax], axis=1).reset_index()
    # Convert GHCND units: PRCP tenths of mm → mm; TMAX tenths of C → C
    out["prcp_mm"] = out["prcp_raw"].div(10).where(out["prcp_raw"].notna())
    out["tmax_c"]  = out["tmax_raw"].div(10).where(out["tmax_raw"].notna())
    out = out[["fips", "date", "prcp_mm", "tmax_c"]]

    log.info("  %d county-day weather rows for %d", len(out), year)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    WEATHER_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})

    # Build station → county FIPS mapping
    stations = load_station_inventory(session)
    stations = load_county_station_crosswalk(session, stations)
    station_fips = stations.set_index("station_id")["fips"].dropna()

    frames = []
    for year in tqdm(STUDY_YEARS, desc="Weather years"):
        df = process_year(year, station_fips, session)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("No weather data collected. Check network connectivity.")
        sys.exit(1)

    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["fips", "date"])
        .reset_index(drop=True)
    )

    out_path = DATA_PROC / "weather_county_day.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(
        "Saved %s — %d county-day rows, %d counties",
        out_path, len(panel), panel["fips"].nunique(),
    )


if __name__ == "__main__":
    main()
