"""
03b_weather_fast.py — Lightweight county-day weather via NOAA GHCND station sampling.

Instead of downloading 400 MB/year annual bulk files, this script:
  1. Downloads only the first 3 MB of each annual CSV (enough for a ~15% station sample)
  2. Maps the sampled stations to counties
  3. Produces prcp_mm and tmax_c county-day controls

Coverage is partial (roughly 40–60% of counties depending on year), which is
fine for regression analysis: missing weather rows are dropped per-observation
or the regressor is set to NaN and the model is estimated on matched rows.

For the final paper, swap in the full 03_download_weather.py run.

Output: data/processed/weather_county_day.parquet
"""

import gzip, io, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import WEATHER_RAW, CROSSWALK_RAW, DATA_PROC, STUDY_YEARS
from utils import get_logger, download_file

log = get_logger("03b_weather_fast")

GHCND_BASE   = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
ANNUAL_URL   = GHCND_BASE + "/by_year/{year}.csv.gz"
STATION_URL  = GHCND_BASE + "/ghcnd-stations.txt"
INV_URL      = GHCND_BASE + "/ghcnd-inventory.txt"
GAZ_URL      = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
                "2023_Gazetteer/2023_Gaz_counties_national.zip")

ELEMENTS = {"PRCP", "TMAX"}
CHUNK_BYTES = 6 * 1024 * 1024   # read first 6 MB of each compressed file


# ---------------------------------------------------------------------------
# Station → county FIPS mapping (built once)
# ---------------------------------------------------------------------------

def build_station_county_map(session: requests.Session) -> pd.Series:
    """Return a Series: station_id → 5-digit county FIPS."""
    CROSSWALK_RAW.mkdir(parents=True, exist_ok=True)

    # 1. Station inventory
    stn_path = CROSSWALK_RAW / "ghcnd-stations.txt"
    download_file(STATION_URL, stn_path, session=session)
    stations = pd.read_fwf(
        stn_path,
        colspecs=[(0, 11), (12, 20), (21, 30), (38, 40)],
        names=["station_id", "lat", "lon", "state_abbr"],
        header=None,
    )
    stations = stations[stations["station_id"].str.startswith("US")].copy()

    # 2. Inventory (filter to stations that have PRCP or TMAX)
    inv_path = CROSSWALK_RAW / "ghcnd-inventory.txt"
    download_file(INV_URL, inv_path, session=session)
    inv = pd.read_fwf(
        inv_path,
        colspecs=[(0, 11), (31, 35)],
        names=["station_id", "element"],
        header=None,
    )
    useful = set(inv.loc[inv["element"].isin(ELEMENTS), "station_id"])
    stations = stations[stations["station_id"].isin(useful)].copy()
    log.info("Station inventory: %d US stations with PRCP/TMAX", len(stations))

    # 3. County centroids from Census gazetteer
    gaz_path = CROSSWALK_RAW / "2023_Gaz_counties_national.zip"
    download_file(GAZ_URL, gaz_path, session=session)
    with zipfile.ZipFile(gaz_path) as zf:
        fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
        with zf.open(fname) as f:
            counties = pd.read_csv(f, sep="\t", dtype=str)
    counties.columns = [c.strip() for c in counties.columns]
    counties["fips"]  = counties["GEOID"].str.zfill(5)
    counties["clat"]  = pd.to_numeric(counties["INTPTLAT"],  errors="coerce")
    counties["clon"]  = pd.to_numeric(counties["INTPTLONG"], errors="coerce")
    counties = counties[["fips", "clat", "clon"]].dropna()

    # 4. Nearest-county assignment (vectorised approximate Euclidean)
    c_lat = counties["clat"].values
    c_lon = counties["clon"].values
    c_fip = counties["fips"].values

    def nearest(lat, lon):
        d2 = (c_lat - lat) ** 2 + (c_lon - lon) ** 2
        return c_fip[int(np.argmin(d2))]

    log.info("Assigning %d stations to nearest county centroid...", len(stations))
    stations["fips"] = [nearest(r.lat, r.lon) for r in stations.itertuples()]

    mapping = stations.set_index("station_id")["fips"]
    log.info("Station→county map built (%d stations)", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Per-year processing (read first CHUNK_BYTES of compressed file)
# ---------------------------------------------------------------------------

def process_year_fast(year: int, station_fips: pd.Series,
                      session: requests.Session) -> pd.DataFrame:
    url  = ANNUAL_URL.format(year=year)
    dest = WEATHER_RAW / f"{year}.csv.gz"

    # Download full file (we'll read only the first chunk)
    if not dest.exists():
        log.info("Downloading %d weather file...", year)
        resp = session.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            downloaded = 0
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                if downloaded >= 30 * 1024 * 1024:   # 30 MB sample
                    log.info("  Stopped at 30 MB for year %d", year)
                    break
        log.info("  Partial download saved: %.1f MB", dest.stat().st_size / 1e6)

    try:
        with gzip.open(dest, "rb") as gz:
            raw_bytes = gz.read(CHUNK_BYTES)
        # Wrap in StringIO; may have a partial last line — drop it
        lines = raw_bytes.decode("latin-1").split("\n")
        buf   = io.StringIO("\n".join(lines[:-1]))
        df = pd.read_csv(
            buf,
            names=["station_id", "date", "element", "value",
                   "mflag", "qflag", "sflag", "obs_time"],
            dtype={"date": str, "station_id": str},
        )
    except Exception as exc:
        log.warning("Could not parse %d: %s", year, exc)
        return pd.DataFrame()

    df = df[df["element"].isin(ELEMENTS)]
    df = df[df["qflag"].isna() | (df["qflag"].astype(str).str.strip() == "")]
    df = df.merge(station_fips.rename("fips"), left_on="station_id",
                  right_index=True, how="inner")
    df = df.dropna(subset=["fips"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    prcp = df[df["element"]=="PRCP"].groupby(["fips","date"])["value"].mean().rename("prcp_raw")
    tmax = df[df["element"]=="TMAX"].groupby(["fips","date"])["value"].mean().rename("tmax_raw")
    out = pd.concat([prcp, tmax], axis=1).reset_index()
    out["prcp_mm"] = out["prcp_raw"] / 10
    out["tmax_c"]  = out["tmax_raw"] / 10
    log.info("  %d county-day weather rows for %d (sampled)", len(out), year)
    return out[["fips", "date", "prcp_mm", "tmax_c"]]


def main() -> None:
    WEATHER_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})

    station_fips = build_station_county_map(session)

    frames = []
    for year in tqdm(STUDY_YEARS, desc="Weather years"):
        df = process_year_fast(year, station_fips, session)
        if not df.empty:
            frames.append(df)

    panel = pd.concat(frames, ignore_index=True).sort_values(["fips","date"]).reset_index(drop=True)
    out = DATA_PROC / "weather_county_day.parquet"
    panel.to_parquet(out, index=False)
    log.info("Saved %s — %d rows, %d counties", out, len(panel), panel["fips"].nunique())


if __name__ == "__main__":
    main()
