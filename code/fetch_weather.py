"""
fetch_weather.py — Download daily county weather from Open-Meteo archive API.
Fetches precipitation, max temperature, and snowfall for each panel county
centroid for 2013-01-01 through 2024-12-31.

Output: data/processed/county_weather_daily.parquet
  Columns: fips, date, precip_mm, tmax_c, snow_mm
"""
import time, sys
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import glob

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "processed" / "county_weather_daily.parquet"

# --- gather panel counties ---------------------------------------------------
panel_fips = set()
for f in glob.glob(str(ROOT / "data" / "processed" / "*county_day*.parquet")):
    df = pd.read_parquet(f, columns=["fips"])
    panel_fips.update(df["fips"].astype(str).str.zfill(5).unique())
panel_fips = sorted(panel_fips)
print(f"Panel counties: {len(panel_fips)}")

centroids = pd.read_parquet(ROOT / "data" / "processed" / "county_centroids.parquet")
centroids["fips"] = centroids["fips"].astype(str).str.zfill(5)
centroids = centroids[centroids["fips"].isin(panel_fips)].set_index("fips")
print(f"Centroids matched: {len(centroids)} of {len(panel_fips)}")

# --- resume from existing file -----------------------------------------------
done_fips: set = set()
if OUT.exists():
    existing = pd.read_parquet(OUT, columns=["fips"])
    done_fips = set(existing["fips"].astype(str).str.zfill(5).unique())
    print(f"Resuming — {len(done_fips)} counties already fetched")

todo = [f for f in panel_fips if f in centroids.index and f not in done_fips]
print(f"Fetching {len(todo)} counties …")

API = "https://archive-api.open-meteo.com/v1/archive"
PARAMS_BASE = {
    "start_date": "2013-01-01",
    "end_date":   "2024-12-31",
    "daily":      "precipitation_sum,temperature_2m_max,snowfall_sum",
    "timezone":   "UTC",
}
SLEEP_S = 0.5
RETRY   = [2, 4, 8, 16]

frames = []

for fips in tqdm(todo, desc="weather"):
    lat = float(centroids.loc[fips, "lat"])
    lon = float(centroids.loc[fips, "lon"])
    params = {**PARAMS_BASE, "latitude": lat, "longitude": lon}

    data = None
    for delay in [0] + RETRY:
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                break
            elif r.status_code == 429:
                continue          # rate-limited, retry with backoff
            else:
                print(f"\n  {fips}: HTTP {r.status_code}")
                break
        except Exception as e:
            print(f"\n  {fips}: {e}")

    if data and "daily" in data:
        d = data["daily"]
        df = pd.DataFrame({
            "fips":     fips,
            "date":     pd.to_datetime(d["time"]),
            "precip_mm": d["precipitation_sum"],
            "tmax_c":    d["temperature_2m_max"],
            "snow_mm":   d["snowfall_sum"],
        })
        frames.append(df)

    time.sleep(SLEEP_S)

if not frames and not done_fips:
    print("No data fetched.")
    sys.exit(1)

new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# Merge with existing
if OUT.exists() and done_fips:
    old_df = pd.read_parquet(OUT)
    combined = pd.concat([old_df, new_df], ignore_index=True)
else:
    combined = new_df

combined["fips"] = combined["fips"].astype(str).str.zfill(5)
combined = combined.drop_duplicates(subset=["fips","date"])
combined = combined.sort_values(["fips","date"]).reset_index(drop=True)
combined.to_parquet(OUT, index=False)

print(f"\nSaved → {OUT}")
print(f"  {len(combined):,} rows, {combined['fips'].nunique()} counties, "
      f"{combined['date'].dt.year.min()}–{combined['date'].dt.year.max()}")
print(f"  Missing precip: {combined['precip_mm'].isna().mean()*100:.1f}%")
print(f"  Missing tmax:   {combined['tmax_c'].isna().mean()*100:.1f}%")
