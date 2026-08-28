"""
01c_fetch_weather.py
Downloads daily precipitation (mm) and max temperature (°C) for each
analysis county from PRISM via the ACIS GridData API.

Source:  http://data.rcc-acis.org/GridData  (grid=1, PRISM 4-km)
         PRISM (Parameter-elevation Regressions on Independent Slopes Model)
         is the standard gridded climate dataset used in economics research.
         Queries at county interior-point centroid from Census gazetteer.
Period:  2013-01-01 – 2024-12-31
Output:  data/processed/weather_county_day.parquet
         Columns: fips (str), date (date), prcp_mm (float), tmax_c (float)
"""
import sys, time, json, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.parse, urllib.error

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("fetch_weather")

CENTROIDS  = Path(__file__).parent.parent / "data" / "processed" / "county_centroids.parquet"
OUT_PATH   = Path(__file__).parent.parent / "data" / "processed" / "weather_county_day.parquet"
CACHE_DIR  = Path(__file__).parent.parent / "data" / "processed" / "weather_cache"

SDATE      = "2013-01-01"
EDATE      = "2024-12-31"
GRID       = "1"            # PRISM 4-km (Parameter-elevation Regressions on Independent Slopes Model)
MAX_WORKERS = 8
RETRY       = 3
PAUSE       = 0.05          # seconds between requests per worker


def fetch_one(fips: str, lat: float, lon: float,
              sdate: str, edate: str) -> pd.DataFrame | None:
    """Fetch daily PRCP + TMAX for one county centroid. Returns None on failure."""
    cache_file = CACHE_DIR / f"{fips}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    params = {
        "loc":   f"{lon:.4f},{lat:.4f}",
        "sdate": sdate,
        "edate": edate,
        "grid":  GRID,
        "elems": "pcpn,maxt",
        "output": "json",
    }
    url = "http://data.rcc-acis.org/GridData?" + urllib.parse.urlencode(params)

    for attempt in range(RETRY):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = json.loads(resp.read())
            break
        except Exception as exc:
            if attempt == RETRY - 1:
                log.warning("FAILED %s after %d attempts: %s", fips, RETRY, exc)
                return None
            time.sleep(2 ** attempt)

    rows = []
    for rec in raw.get("data", []):
        date_str, prcp_raw, tmax_raw = rec[0], rec[1], rec[2]
        try:
            prcp = float(prcp_raw) * 25.4   # inches → mm
        except (ValueError, TypeError):
            prcp = np.nan
        try:
            tmax = (float(tmax_raw) - 32) * 5 / 9   # °F → °C
        except (ValueError, TypeError):
            tmax = np.nan
        rows.append({"fips": fips, "date": pd.to_datetime(date_str).date(),
                     "prcp_mm": prcp, "tmax_c": tmax})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file, index=False)
    time.sleep(PAUSE)
    return df


def main():
    centroids = pd.read_parquet(CENTROIDS)
    n = len(centroids)
    log.info("Fetching weather for %d counties, %s – %s", n, SDATE, EDATE)
    log.info("Using %d parallel workers (cache dir: %s)", MAX_WORKERS, CACHE_DIR)

    # Skip already-completed if final output exists (resume support)
    done_fips: set = set()
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH, columns=["fips"])
        done_fips = set(existing["fips"].unique())
        log.info("Resuming: %d counties already in output, %d remaining",
                 len(done_fips), n - len(done_fips))

    todo = centroids[~centroids["fips"].isin(done_fips)]
    if todo.empty:
        log.info("All counties already downloaded.")
        return

    records = todo.to_dict("records")
    results = []
    errors  = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_one, r["fips"], r["lat"], r["lon"], SDATE, EDATE): r["fips"]
            for r in records
        }
        for i, fut in enumerate(as_completed(futures), 1):
            fips = futures[fut]
            try:
                df = fut.result()
            except Exception as exc:
                log.warning("Exception for %s: %s", fips, exc)
                df = None

            if df is not None:
                results.append(df)
            else:
                errors += 1

            if i % 100 == 0 or i == len(records):
                log.info("  Progress: %d/%d done (%d errors)", i, len(records), errors)

    if not results:
        log.error("No data fetched!")
        return

    new_data = pd.concat(results, ignore_index=True)

    # Merge with any existing data
    if done_fips:
        old_data = pd.read_parquet(OUT_PATH)
        final = pd.concat([old_data, new_data], ignore_index=True)
    else:
        final = new_data

    final = final.sort_values(["fips", "date"]).reset_index(drop=True)
    final["date"] = pd.to_datetime(final["date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)

    n_counties = final["fips"].nunique()
    n_days     = final["date"].nunique()
    log.info("Saved %d rows (%d counties × ~%d days) → %s",
             len(final), n_counties, n_days, OUT_PATH)
    log.info("PRCP: mean=%.1f mm, missing=%.1f%%",
             final["prcp_mm"].mean(),
             final["prcp_mm"].isna().mean() * 100)
    log.info("TMAX: mean=%.1f °C, missing=%.1f%%",
             final["tmax_c"].mean(),
             final["tmax_c"].isna().mean() * 100)


if __name__ == "__main__":
    main()
