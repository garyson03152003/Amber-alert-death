"""
build_fars_county_day.py
Download FARS accident + vehicle files for 2013–2022, build a
county-day panel with:
  - total_fatals   : all traffic fatalities
  - drunk_fatals   : fatalities in crashes with ≥1 drinking driver (DR_DRINK=1)
  - sober_fatals   : total_fatals − drunk_fatals
  - weather_adverse: 1 if WEATHER code indicates rain/snow/fog/ice

Output: data/processed/fars_county_day.parquet
        columns: fips, date, total_fatals, drunk_fatals, sober_fatals, weather_adverse
"""
import io, sys, time, zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "data" / "processed" / "fars_county_day.parquet"
YEARS = range(2013, 2023)   # 2013–2022

ADVERSE_WEATHER = {2, 3, 4, 5, 6, 11, 12}   # rain, sleet, snow, fog, crosswinds, blowing snow, freezing rain

def fetch_zip(year: int, session: requests.Session) -> zipfile.ZipFile:
    url = (f"https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}"
           f"/National/FARS{year}NationalCSV.zip")
    for delay in [0, 4, 8, 16]:
        if delay:
            time.sleep(delay)
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"  {year}: attempt failed ({e}), retrying…")
    raise RuntimeError(f"Could not download FARS {year}")

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.encode("ascii", "ignore").decode().strip() for c in df.columns]
    return df

def read_file(zf: zipfile.ZipFile, keyword: str) -> pd.DataFrame:
    """Find and read the CSV whose name contains `keyword` (case-insensitive)."""
    # Match .csv or .CSV, exclude *sf* (supplemental files)
    candidates = [f for f in zf.namelist()
                  if keyword.lower() in f.lower()
                  and f.lower().endswith(".csv")
                  and "sf" not in f.lower().split("/")[-1].lower()]
    if not candidates:
        raise FileNotFoundError(f"No file matching '{keyword}' in zip")
    with zf.open(candidates[0]) as fh:
        return clean_cols(pd.read_csv(fh, encoding="latin1", low_memory=False))

def build_year(year: int, session: requests.Session) -> pd.DataFrame:
    print(f"  {year}: downloading…", flush=True)
    zf = fetch_zip(year, session)

    # ── accident.csv ──────────────────────────────────────────────────────────
    acc = read_file(zf, "accident")
    acc = acc[["ST_CASE", "STATE", "COUNTY", "MONTH", "DAY", "YEAR",
               "HOUR", "FATALS", "WEATHER"]].copy()
    acc["fips"] = (acc["STATE"].astype(str).str.zfill(2) +
                   acc["COUNTY"].astype(str).str.zfill(3))
    # COUNTY==0 means unknown/not coded — drop
    acc = acc[acc["COUNTY"] > 0].copy()
    acc["date"] = pd.to_datetime(
        acc[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR":"year","MONTH":"month","DAY":"day"})
    )
    # Weather: map to binary adverse flag
    # WEATHER codes: 1=clear, 2=rain, 3=sleet, 4=snow, 5=fog, 6=crosswind,
    #                10=cloudy, 11=blowing snow, 12=freezing rain
    # Some years use WEATHER1 as first of up to 3 weather codes; handle both
    if "WEATHER" in acc.columns:
        acc["weather_adverse"] = acc["WEATHER"].isin(ADVERSE_WEATHER).astype(int)
    else:
        acc["weather_adverse"] = 0

    # ── vehicle.csv: DR_DRINK per crash (any drinking driver) ─────────────────
    veh = read_file(zf, "vehicle")
    veh = clean_cols(veh)
    if "DR_DRINK" not in veh.columns:
        # Older FARS years may use different field name
        drink_col = next((c for c in veh.columns if "DRINK" in c.upper()), None)
        if drink_col:
            veh = veh.rename(columns={drink_col: "DR_DRINK"})
        else:
            veh["DR_DRINK"] = 0
    veh = veh[["ST_CASE", "DR_DRINK"]].copy()
    veh["DR_DRINK"] = pd.to_numeric(veh["DR_DRINK"], errors="coerce").fillna(0)
    # Flag crash as drunk if ANY driver was drinking (DR_DRINK == 1)
    drunk_cases = veh[veh["DR_DRINK"] == 1]["ST_CASE"].unique()
    acc["is_drunk"] = acc["ST_CASE"].isin(drunk_cases).astype(int)

    # ── Aggregate to county-day ───────────────────────────────────────────────
    grp = acc.groupby(["fips", "date"]).agg(
        total_fatals    = ("FATALS",       "sum"),
        drunk_fatals    = ("is_drunk",     lambda x: acc.loc[x.index, "FATALS"].where(acc.loc[x.index, "is_drunk"]==1, 0).sum()),
        weather_adverse = ("weather_adverse", "max"),
    ).reset_index()
    grp["sober_fatals"] = grp["total_fatals"] - grp["drunk_fatals"]
    print(f"  {year}: {len(acc):,} crashes → {len(grp):,} county-days  "
          f"({grp.drunk_fatals.sum():.0f} drunk / {grp.total_fatals.sum():.0f} total fatals = "
          f"{100*grp.drunk_fatals.sum()/max(grp.total_fatals.sum(),1):.1f}%)")
    return grp

# ── Main ──────────────────────────────────────────────────────────────────────
session = requests.Session()
session.headers["User-Agent"] = "amber-alert-research/1.0 (academic)"

frames = []
for yr in tqdm(YEARS, desc="FARS years"):
    try:
        frames.append(build_year(yr, session))
    except Exception as e:
        print(f"  SKIP {yr}: {e}")

if not frames:
    print("No data downloaded.")
    sys.exit(1)

df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=["fips", "date"])
df = df.sort_values(["fips", "date"]).reset_index(drop=True)
df.to_parquet(OUT, index=False)

print(f"\nSaved → {OUT}")
print(f"  {len(df):,} county-days with ≥1 fatality")
print(f"  {df.fips.nunique():,} unique counties")
print(f"  Total fatals: {df.total_fatals.sum():,.0f}")
print(f"  Drunk fatals: {df.drunk_fatals.sum():,.0f}  ({100*df.drunk_fatals.sum()/df.total_fatals.sum():.1f}%)")
print(f"  Year range: {df.date.dt.year.min()}–{df.date.dt.year.max()}")
