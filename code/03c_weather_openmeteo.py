"""
03c_weather_openmeteo.py — Full county-day weather via Open-Meteo historical API.

Uses ERA5 reanalysis (0.25° / ~28 km grid) queried at each county centroid.
Free, no API key, covers 2013-present.

Batches up to 100 county centroids per API call to minimise round-trips.
Output: data/processed/weather_county_day.parquet

Run: python code/03c_weather_openmeteo.py
"""

import sys, time, zipfile, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC, STUDY_YEARS
from utils import get_logger

log = get_logger("03c_weather_openmeteo")

GAZ_URL      = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
                "2023_Gazetteer/2023_Gaz_counties_national.zip")
API_BASE     = "https://archive-api.open-meteo.com/v1/archive"

SDATE        = f"{min(STUDY_YEARS)}-01-01"
EDATE        = f"{max(STUDY_YEARS)}-12-31"
BATCH_SIZE   = 20     # locations per Open-Meteo call (100×12yr response too large)
BATCH_PAUSE  = 6.0    # seconds between batches (archive API: ~10 req/min free tier)
MAX_RETRIES  = 4


# ---------------------------------------------------------------------------
# County centroids
# ---------------------------------------------------------------------------

def load_county_centroids() -> pd.DataFrame:
    gaz_path = CROSSWALK_RAW / "2023_Gaz_counties_national.zip"
    if not gaz_path.exists():
        log.info("Downloading Census gazetteer...")
        r = requests.get(GAZ_URL, timeout=120)
        r.raise_for_status()
        gaz_path.write_bytes(r.content)

    with zipfile.ZipFile(gaz_path) as zf:
        fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
        with zf.open(fname) as f:
            gaz = pd.read_csv(f, sep="\t", dtype=str)

    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["fips"] = gaz["GEOID"].str.zfill(5)
    gaz["lat"]  = pd.to_numeric(gaz["INTPTLAT"],  errors="coerce")
    gaz["lon"]  = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")
    gaz = gaz[["fips", "lat", "lon"]].dropna()
    log.info("Loaded %d county centroids", len(gaz))
    return gaz


# ---------------------------------------------------------------------------
# Open-Meteo batch call (up to 100 locations)
# ---------------------------------------------------------------------------

def fetch_batch(batch: pd.DataFrame,
                session: requests.Session) -> list[pd.DataFrame]:
    """
    Fetch daily precip + tmax for a batch of counties.
    Returns a list of DataFrames (one per county), same order as batch.
    """
    # Build query string manually — requests.get url-encodes commas in values,
    # but Open-Meteo requires literal commas for multi-location batches.
    lats  = ",".join(batch["lat"].round(5).astype(str))
    lons  = ",".join(batch["lon"].round(5).astype(str))
    url   = (f"{API_BASE}?latitude={lats}&longitude={lons}"
             f"&start_date={SDATE}&end_date={EDATE}"
             f"&daily=precipitation_sum,temperature_2m_max&timezone=UTC")

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=300)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 3)
                log.warning("Rate-limited — waiting %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:
            wait = 2 ** (attempt + 1)
            log.debug("Attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        log.warning("All retries exhausted for batch starting at fips=%s",
                    batch["fips"].iloc[0])
        return [pd.DataFrame()] * len(batch)

    # Normalise: single result → list
    if isinstance(payload, dict):
        payload = [payload]

    results = []
    for i, item in enumerate(payload):
        if "daily" not in item or "error" in item:
            results.append(pd.DataFrame())
            continue
        df = pd.DataFrame({
            "date":    pd.to_datetime(item["daily"]["time"]),
            "prcp_mm": item["daily"]["precipitation_sum"],
            "tmax_c":  item["daily"]["temperature_2m_max"],
        })
        df["fips"] = batch.iloc[i]["fips"]
        results.append(df[["fips", "date", "prcp_mm", "tmax_c"]])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROC / "weather_county_day.parquet"

    centroids = load_county_centroids()

    # Resume support: skip counties already in output
    done_fips: set = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["fips"])
        # Only consider a county "done" if it has enough rows (≥ 365 × years)
        min_rows = 365 * len(STUDY_YEARS) * 0.9
        counts   = existing["fips"].value_counts()
        done_fips = set(counts[counts >= min_rows].index)
        log.info("Resuming: %d counties complete, %d remaining",
                 len(done_fips), len(centroids) - len(done_fips))

    todo = centroids[~centroids["fips"].isin(done_fips)].reset_index(drop=True)
    if todo.empty:
        log.info("All %d counties already complete.", len(centroids))
        return

    log.info("Fetching weather for %d counties in batches of %d (years %s–%s)...",
             len(todo), BATCH_SIZE, SDATE[:4], EDATE[:4])

    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})

    all_frames = []
    batches = [todo.iloc[i:i+BATCH_SIZE]
               for i in range(0, len(todo), BATCH_SIZE)]

    for i, batch in enumerate(tqdm(batches, desc="Batches")):
        dfs = fetch_batch(batch, session)
        for df in dfs:
            if not df.empty:
                all_frames.append(df)
        time.sleep(BATCH_PAUSE)

        # Save and clear every 25 batches to keep memory low
        if (i + 1) % 25 == 0 and all_frames:
            _save(all_frames, done_fips, out_path, checkpoint=True)
            # Update done_fips so next _save doesn't double-count
            if out_path.exists():
                done_fips = set(pd.read_parquet(out_path, columns=["fips"])["fips"].unique())
            all_frames = []

    _save(all_frames, done_fips, out_path, checkpoint=False)


def _save(frames: list, done_fips: set, out_path: Path,
          checkpoint: bool = False) -> None:
    if not frames:
        return
    new_data = pd.concat(frames, ignore_index=True)

    if out_path.exists() and done_fips:
        old_data = pd.read_parquet(out_path)
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = (combined
                .drop_duplicates(subset=["fips", "date"])
                .sort_values(["fips", "date"])
                .reset_index(drop=True))
    combined.to_parquet(out_path, index=False)
    tag = "[checkpoint]" if checkpoint else "[final]"
    log.info("%s Saved %d rows, %d counties → %s",
             tag, len(combined), combined["fips"].nunique(), out_path)


if __name__ == "__main__":
    main()
