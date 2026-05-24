"""
build_ca_ccrs_hourly.py
============================================================
Download California CCRS crash data and build a county-hour panel,
retaining the crash hour so we can test whether AMBER alert effects
concentrate in the morning commute (sleep-disruption mechanism) vs.
the same night (immediate driving-suppression mechanism).

Output: data/processed/california_ccrs_county_hour.parquet
  Columns: fips, date, hour, ca_crashes, ca_fatals, ca_serious_inj
  Rows: one per (county, calendar-date, hour-of-day) that had ≥1 crash
  Zero-crash county-hour slots are NOT included (sparse format).

Use the same direct-download URLs as build_california_ccrs.py.
Only downloads years that overlap with the AMBER alert data (2016–2022).
"""
import sys, warnings, io, gc, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("ca_ccrs_hourly")

OUT_PATH  = DATA_PROC / "california_ccrs_county_hour.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# Same resource IDs as build_california_ccrs.py (verified direct download URLs)
CCRS_CRASH_URLS = {
    2016: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/3d5f2586-cf68-4213-aa1c-60df37399d10/download/crashes_2016.csv",
    2017: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/4784664d-b7cf-4427-af25-7c7307bad56c/download/crashes_2017.csv",
    2018: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/a4b57216-5110-43d3-884c-d95366b19158/download/crashes_2018.csv",
    2019: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/2b4c7d03-e684-435e-80da-17935de9499f/download/crashes_2019.csv",
    2020: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/a2e0605d-0695-4bce-806d-4d0dda7ace68/download/crashes_2020.csv",
    2021: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/d08692e2-6d36-487e-bca0-28cd127a626f/download/crashes_2021.csv",
    2022: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/7828780b-117b-455e-9275-986ad3ffde50/download/crashes_2022.csv",
}

# CA county number (1-58) → 5-digit FIPS
CA_COUNTY_FIPS = {
    1:"06001",3:"06003",5:"06005",7:"06007",9:"06009",11:"06011",
    13:"06013",15:"06015",17:"06017",19:"06019",21:"06021",23:"06023",
    25:"06025",27:"06027",29:"06029",31:"06031",33:"06033",35:"06035",
    37:"06037",39:"06039",41:"06041",43:"06043",45:"06045",47:"06047",
    49:"06049",51:"06051",53:"06053",55:"06055",57:"06057",59:"06059",
    61:"06061",63:"06063",65:"06065",67:"06067",69:"06069",71:"06071",
    73:"06073",75:"06075",77:"06077",79:"06079",81:"06081",83:"06083",
    85:"06085",87:"06087",89:"06089",91:"06091",93:"06093",95:"06095",
    97:"06097",99:"06099",101:"06101",103:"06103",105:"06105",107:"06107",
    109:"06109",111:"06111",113:"06113",115:"06115",
}
VALID_FIPS = set(CA_COUNTY_FIPS.values())


def download_year_hourly(year: int) -> pd.DataFrame | None:
    """Download one year of CA CCRS, return county-hour aggregation."""
    import requests as _requests
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Referer": "https://data.ca.gov/",
        "Accept": "text/html,application/xhtml+xml,*/*",
    }

    url = CCRS_CRASH_URLS.get(year)
    if not url:
        log.warning("  No URL for year %d", year)
        return None

    try:
        log.info("  Downloading %d …", year)
        resp = _requests.get(url, headers=HEADERS, timeout=300,
                             stream=True, allow_redirects=True)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode("latin1")),
                         low_memory=False)
        log.info("  Downloaded %d rows", len(df))
    except Exception as e:
        log.warning("  Download failed for %d: %s", year, e)
        return None

    df.columns = [c.upper().strip() for c in df.columns]
    log.info("  Columns sample: %s", list(df.columns)[:12])

    # ── Date-time column ──────────────────────────────────────────────────────
    date_col = next((c for c in df.columns if c == "CRASH DATE TIME"), None)
    if date_col is None:
        date_col = next((c for c in df.columns
                         if "DATE" in c
                         and not any(x in c for x in
                                     ("CREATED","MODIFIED","REVIEWED","PREPARED","NOTIF"))),
                        None)
    if date_col is None:
        log.warning("  No date column in year %d; skipping", year)
        return None

    df["crash_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["crash_dt"])
    df["crash_date"] = df["crash_dt"].dt.normalize()   # calendar date
    df["crash_hour"] = df["crash_dt"].dt.hour           # 0-23

    # ── County → FIPS ────────────────────────────────────────────────────────
    county_col = next((c for c in df.columns if c == "COUNTY CODE"), None)
    if county_col is None:
        county_col = next((c for c in df.columns if "COUNTY" in c), None)
    if county_col is None:
        log.warning("  No county column in year %d; skipping", year)
        return None

    df["_cn"] = pd.to_numeric(df[county_col], errors="coerce")
    df = df.dropna(subset=["_cn"])
    df["_cn"] = df["_cn"].astype(int)
    df["fips"] = "06" + ((df["_cn"] * 2 - 1).astype(str).str.zfill(3))
    df = df[df["fips"].isin(VALID_FIPS)]

    # ── Severity ──────────────────────────────────────────────────────────────
    fatal_col = next((c for c in df.columns
                      if c in ("NUMBERKILLED", "NUMBER KILLED")), None)
    if fatal_col is None:
        fatal_col = next((c for c in df.columns
                          if "KILLED" in c or "FATAL" in c), None)
    inj_col = next((c for c in df.columns
                    if c in ("NUMBERINJURED", "NUMBER INJURED")), None)
    if inj_col is None:
        inj_col = next((c for c in df.columns
                        if "INJUR" in c and "NUMBER" in c), None)

    df["fatals"]     = pd.to_numeric(df[fatal_col], errors="coerce").fillna(0) if fatal_col else 0
    df["serious_inj"]= pd.to_numeric(df[inj_col],   errors="coerce").fillna(0) if inj_col  else 0

    # ── Aggregate to county × date × hour ────────────────────────────────────
    agg = (df.groupby(["fips", "crash_date", "crash_hour"])
             .agg(ca_crashes    =("fips",        "count"),
                  ca_fatals     =("fatals",       "sum"),
                  ca_serious_inj=("serious_inj",  "sum"))
             .reset_index()
             .rename(columns={"crash_date": "date", "crash_hour": "hour"}))

    log.info("  → %d county-hour rows  fatals=%.0f",
             len(agg), agg["ca_fatals"].sum())
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────
log.info("Building California CCRS county-hour panel (2016–2022) …")
parts = []
for yr in sorted(CCRS_CRASH_URLS):
    part = download_year_hourly(yr)
    if part is not None:
        parts.append(part)
    time.sleep(1.5)
    gc.collect()

if not parts:
    log.error("No data downloaded.")
    sys.exit(1)

hourly = pd.concat(parts, ignore_index=True)
hourly["date"] = pd.to_datetime(hourly["date"])
hourly["hour"] = hourly["hour"].astype(int)

# Collapse duplicate (fips, date, hour) rows from concat
hourly = (hourly.groupby(["fips", "date", "hour"])
                .agg(ca_crashes    =("ca_crashes",     "sum"),
                     ca_fatals     =("ca_fatals",      "sum"),
                     ca_serious_inj=("ca_serious_inj", "sum"))
                .reset_index())

log.info("\nFinal CA CCRS county-hour panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(hourly), hourly["fips"].nunique(),
         hourly["date"].min().date(), hourly["date"].max().date())
log.info("  Total crashes: %.0f  fatals: %.0f",
         hourly["ca_crashes"].sum(), hourly["ca_fatals"].sum())

hourly.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
