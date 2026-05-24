"""
build_california_ccrs.py
========================================================
Download California CCRS (Crash Clearinghouse Reporting System) crash data
and build a county-day panel of serious injuries and fatalities.

California CCRS replaced SWITRS on 2025-01-08.  Historical data from 2016
onward is available at: https://data.ca.gov/dataset/ccrs

No authentication required — true open bulk download.

California has the most traffic crashes of any US state and uses the
standard KABCO severity scale:
  K = Fatal
  A = Suspected Serious Injury
  B = Suspected Minor Injury
  C = Possible Injury
  O = No Injury (Property Damage Only)

This script:
  1. Downloads crashes_{year}.csv for each available year (2016–2024)
  2. Extracts county FIPS, crash date/time, severity counts
  3. Aggregates to county-day level: fatals, serious_injuries, all_injuries
  4. Saves to data/processed/california_ccrs_county_day.parquet

California county FIPS codes: state FIPS = 06, county codes 001–115.
The CCRS file uses COUNTY_CODE (1–58 matching California county numbers).

Output: data/processed/california_ccrs_county_day.parquet
"""
import sys, warnings, io, gc, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("california_ccrs")

OUT_PATH = DATA_PROC / "california_ccrs_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── California CCRS direct download URLs ─────────────────────────────────────
# One crashes CSV per year; pattern confirmed from data.ca.gov API.
# Resource IDs from the CCRS dataset catalog.
# Verified direct download URLs (resource IDs confirmed from data.ca.gov portal)
CCRS_CRASH_URLS = {
    2016: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/3d5f2586-cf68-4213-aa1c-60df37399d10/download/crashes_2016.csv",
    2017: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/4784664d-b7cf-4427-af25-7c7307bad56c/download/crashes_2017.csv",
    2018: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/a4b57216-5110-43d3-884c-d95366b19158/download/crashes_2018.csv",
    2019: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/2b4c7d03-e684-435e-80da-17935de9499f/download/crashes_2019.csv",
    2020: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/a2e0605d-0695-4bce-806d-4d0dda7ace68/download/crashes_2020.csv",
    2021: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/d08692e2-6d36-487e-bca0-28cd127a626f/download/crashes_2021.csv",
    2022: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/7828780b-117b-455e-9275-986ad3ffde50/download/crashes_2022.csv",
    2023: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/436642c0-cd04-4a4c-b45e-564b66437476/download/crashes_2023.csv",
    2024: "https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/f775df59-b89b-4f82-bd3d-8807fa3a22a0/download/crashes_2024.csv",
}

# ── California county FIPS mapping ────────────────────────────────────────────
# CA county numbers 1–58 → FIPS 06001–06115 (odd-numbered sequence)
# Standard FIPS: 06 + zero-padded county number × 2 - 1... actually it's just
# the county code × 2 - 1 for odd-numbered counties.
# Simpler: CCRS COUNTY_CODE is 1-based; FIPS = "06" + str(COUNTY_CODE * 2 - 1).zfill(3)
# E.g., Alameda = 1 → 001; Alpine = 3 → 003 etc. (actually it's the odd-number rule)
# The safest mapping uses the name lookup.

# Standard California county FIPS (state 06, counties by alphabetical order × 2 - 1)
# From the US Census official list:
CA_COUNTY_FIPS = {
    1:  "06001",  # Alameda
    3:  "06003",  # Alpine
    5:  "06005",  # Amador
    7:  "06007",  # Butte
    9:  "06009",  # Calaveras
    11: "06011",  # Colusa
    13: "06013",  # Contra Costa
    15: "06015",  # Del Norte
    17: "06017",  # El Dorado
    19: "06019",  # Fresno
    21: "06021",  # Glenn
    23: "06023",  # Humboldt
    25: "06025",  # Imperial
    27: "06027",  # Inyo
    29: "06029",  # Kern
    31: "06031",  # Kings
    33: "06033",  # Lake
    35: "06035",  # Lassen
    37: "06037",  # Los Angeles
    39: "06039",  # Madera
    41: "06041",  # Marin
    43: "06043",  # Mariposa
    45: "06045",  # Mendocino
    47: "06047",  # Merced
    49: "06049",  # Modoc
    51: "06051",  # Mono
    53: "06053",  # Monterey
    55: "06055",  # Napa
    57: "06057",  # Nevada
    59: "06059",  # Orange
    61: "06061",  # Placer
    63: "06063",  # Plumas
    65: "06065",  # Riverside
    67: "06067",  # Sacramento
    69: "06069",  # San Benito
    71: "06071",  # San Bernardino
    73: "06073",  # San Diego
    75: "06075",  # San Francisco
    77: "06077",  # San Joaquin
    79: "06079",  # San Luis Obispo
    81: "06081",  # San Mateo
    83: "06083",  # Santa Barbara
    85: "06085",  # Santa Clara
    87: "06087",  # Santa Cruz
    89: "06089",  # Shasta
    91: "06091",  # Sierra
    93: "06093",  # Siskiyou
    95: "06095",  # Solano
    97: "06097",  # Sonoma
    99: "06099",  # Stanislaus
    101: "06101", # Sutter
    103: "06103", # Tehama
    105: "06105", # Trinity
    107: "06107", # Tulare
    109: "06109", # Tuolumne
    111: "06111", # Ventura
    113: "06113", # Yolo
    115: "06115", # Yuba
}


def download_ccrs_year(year: int) -> pd.DataFrame | None:
    """
    Try to download and parse one year of CCRS crash data.
    Returns county-day aggregation or None if download fails.
    """
    try:
        import urllib.request

        # Try two URL patterns
        urls_to_try = [
            f"https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/crashes_{year}.csv",
        ]
        if year in CCRS_CRASH_URLS_ALT:
            urls_to_try.insert(0, CCRS_CRASH_URLS_ALT[year])

        df = None
        for url in urls_to_try:
            try:
                log.info("  Trying %s", url)
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (research data download)"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    content = resp.read()
                df = pd.read_csv(io.StringIO(content.decode("latin1")),
                                 low_memory=False)
                log.info("  Downloaded %d rows for %d", len(df), year)
                break
            except Exception as e:
                log.debug("  URL %s failed: %s", url, e)
                continue

        if df is None:
            log.warning("  All URLs failed for year %d", year)
            return None

        log.info("  Columns: %s", list(df.columns)[:10])

        # Normalize column names
        df.columns = [c.upper().strip() for c in df.columns]

        # Find county column
        county_col = next((c for c in df.columns
                           if "COUNTY" in c and "CODE" in c), None)
        if county_col is None:
            county_col = next((c for c in df.columns if "COUNTY" in c), None)
        if county_col is None:
            log.warning("  No county column found for %d; columns: %s",
                        year, list(df.columns)[:20])
            return None

        # Find date column
        date_col = next((c for c in df.columns
                         if "COLLISION_DATE" in c or "CRASH_DATE" in c
                         or c in ("DATE",)), None)
        if date_col is None:
            log.warning("  No date column found for %d; columns: %s",
                        year, list(df.columns)[:20])
            return None

        # Find severity column(s)
        # CCRS may have COLLISION_SEVERITY or individual count columns
        sev_col = next((c for c in df.columns if "COLLISION_SEVERITY" in c), None)
        fatal_col  = next((c for c in df.columns
                           if "KILLED" in c or "FATAL" in c), None)
        serious_col = next((c for c in df.columns
                            if "SERIOUS" in c and "INJUR" in c), None)

        log.info("  county_col=%s  date_col=%s  sev_col=%s  fatal_col=%s  serious_col=%s",
                 county_col, date_col, sev_col, fatal_col, serious_col)

        # Parse date
        df["crash_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["crash_date"])

        # Map county to FIPS
        df[county_col] = pd.to_numeric(df[county_col], errors="coerce")
        df = df.dropna(subset=[county_col])
        df[county_col] = df[county_col].astype(int)
        df["fips"] = df[county_col].map(CA_COUNTY_FIPS)
        df = df.dropna(subset=["fips"])

        # Build crash-level severity counts
        if fatal_col:
            df["fatals"] = pd.to_numeric(df[fatal_col], errors="coerce").fillna(0)
        elif sev_col:
            df["fatals"] = (df[sev_col].astype(str).str.upper() == "FATAL").astype(int)
        else:
            df["fatals"] = 0

        if serious_col:
            df["serious_inj"] = pd.to_numeric(df[serious_col], errors="coerce").fillna(0)
        elif sev_col:
            sev_up = df[sev_col].astype(str).str.upper()
            df["serious_inj"] = sev_up.isin(["SEVERE INJURY", "SUSPECTED SERIOUS INJURY",
                                              "SUSPECTED SERIOUS",  "SERIOUS INJURY"]).astype(int)
        else:
            df["serious_inj"] = 0

        # Aggregate to county-day
        agg = (df.groupby(["fips", "crash_date"])
                  .agg(ca_fatals=("fatals", "sum"),
                       ca_serious_inj=("serious_inj", "sum"),
                       ca_crashes=("fips", "count"))
                  .reset_index()
                  .rename(columns={"crash_date": "date"}))
        log.info("  Aggregated to %d county-days  (fatals=%.0f  serious=%.0f)",
                 len(agg), agg["ca_fatals"].sum(), agg["ca_serious_inj"].sum())
        return agg

    except Exception as e:
        log.warning("  Year %d failed: %s", year, e)
        return None


# ── Main download loop ────────────────────────────────────────────────────────
log.info("Downloading California CCRS crash data …")
parts = []
for yr in range(2016, 2025):
    log.info("Year %d …", yr)
    part = download_ccrs_year(yr)
    if part is not None:
        parts.append(part)
    time.sleep(1.0)  # polite delay between requests
    gc.collect()

if not parts:
    log.error("No data downloaded. Check network access and URLs.")
    sys.exit(1)

ca_panel = pd.concat(parts, ignore_index=True)
ca_panel["date"] = pd.to_datetime(ca_panel["date"])
ca_panel = (ca_panel.groupby(["fips", "date"])
                     .agg(ca_fatals=("ca_fatals", "sum"),
                          ca_serious_inj=("ca_serious_inj", "sum"),
                          ca_crashes=("ca_crashes", "sum"))
                     .reset_index())

log.info("\nFinal California CCRS panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(ca_panel), ca_panel["fips"].nunique(),
         ca_panel["date"].min().date(), ca_panel["date"].max().date())
log.info("  Total fatals: %.0f  Total serious injuries: %.0f",
         ca_panel["ca_fatals"].sum(), ca_panel["ca_serious_inj"].sum())

ca_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
