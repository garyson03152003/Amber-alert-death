"""
build_illinois_idot.py
========================================================
Download Illinois IDOT crash data from the ArcGIS Open Data Hub
and build a county-day panel of serious injuries and fatalities.

Source: https://gis-idot.opendata.arcgis.com/
No authentication required; one dataset per year (2014–2024).

Illinois is the 9th most-alerted state (1,608 alerts in our sample).
The IDOT data uses KABCO severity with individual injury count columns:
  INJURIES_FATAL
  INJURIES_INCAPACITATING    ← Suspected Serious Injury (A)
  INJURIES_NON_INCAPACITATING
  INJURIES_REPORTED_NOT_EVIDENT
  INJURIES_NO_INDICATION

County is coded as COUNTY or COUNTY_CITY_CD (county FIPS available via name lookup).

The ArcGIS FeatureServer query endpoint allows bulk CSV export.
Base URL:  https://gis.idot.illinois.gov/arcgis/rest/services/StatewideData/Crashes_{year}/FeatureServer/0

Output: data/processed/illinois_idot_county_day.parquet
"""
import sys, warnings, io, gc, time, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("illinois_idot")

OUT_PATH = DATA_PROC / "illinois_idot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# Illinois county FIPS (state 17, counties 001–203, every other)
# Map: county name → FIPS code (from US Census)
IL_COUNTY_FIPS_BY_NAME = {
    "ADAMS": "17001", "ALEXANDER": "17003", "BOND": "17005", "BOONE": "17007",
    "BROWN": "17009", "BUREAU": "17011", "CALHOUN": "17013", "CARROLL": "17015",
    "CASS": "17017", "CHAMPAIGN": "17019", "CHRISTIAN": "17021", "CLARK": "17023",
    "CLAY": "17025", "CLINTON": "17027", "COLES": "17029", "COOK": "17031",
    "CRAWFORD": "17033", "CUMBERLAND": "17035", "DEKALB": "17037", "DE WITT": "17039",
    "DEWITT": "17039", "DOUGLAS": "17041", "DUPAGE": "17043", "EDGAR": "17045",
    "EDWARDS": "17047", "EFFINGHAM": "17049", "FAYETTE": "17051", "FORD": "17053",
    "FRANKLIN": "17055", "FULTON": "17057", "GALLATIN": "17059", "GREENE": "17061",
    "GRUNDY": "17063", "HAMILTON": "17065", "HANCOCK": "17067", "HARDIN": "17069",
    "HENDERSON": "17071", "HENRY": "17073", "IROQUOIS": "17075", "JACKSON": "17077",
    "JASPER": "17079", "JEFFERSON": "17081", "JERSEY": "17083", "JO DAVIESS": "17085",
    "JOHNSON": "17087", "KANE": "17089", "KANKAKEE": "17091", "KENDALL": "17093",
    "KNOX": "17095", "LAKE": "17097", "LASALLE": "17099", "LAWRENCE": "17101",
    "LEE": "17103", "LIVINGSTON": "17105", "LOGAN": "17107", "MCDONOUGH": "17109",
    "MCHENRY": "17111", "MCLEAN": "17113", "MACON": "17115", "MACOUPIN": "17117",
    "MADISON": "17119", "MARION": "17121", "MARSHALL": "17123", "MASON": "17125",
    "MASSAC": "17127", "MENARD": "17129", "MERCER": "17131", "MONROE": "17133",
    "MONTGOMERY": "17135", "MORGAN": "17137", "MOULTRIE": "17139", "OGLE": "17141",
    "PEORIA": "17143", "PERRY": "17145", "PIATT": "17147", "PIKE": "17149",
    "POPE": "17151", "PULASKI": "17153", "PUTNAM": "17155", "RANDOLPH": "17157",
    "RICHLAND": "17159", "ROCK ISLAND": "17161", "ST. CLAIR": "17163",
    "SALINE": "17165", "SANGAMON": "17167", "SCHUYLER": "17169", "SCOTT": "17171",
    "SHELBY": "17173", "STARK": "17175", "STEPHENSON": "17177", "TAZEWELL": "17179",
    "UNION": "17181", "VERMILION": "17183", "WABASH": "17185", "WARREN": "17187",
    "WASHINGTON": "17189", "WAYNE": "17191", "WHITE": "17193", "WHITESIDE": "17195",
    "WILL": "17197", "WILLIAMSON": "17199", "WINNEBAGO": "17201", "WOODFORD": "17203",
}

# ArcGIS FeatureServer endpoints per year
IDOT_ENDPOINTS = {
    yr: (f"https://gis.idot.illinois.gov/arcgis/rest/services/StatewideData/"
         f"Crashes_{yr}/FeatureServer/0/query")
    for yr in range(2016, 2025)
}

def fetch_idot_year_arcgis(year: int) -> pd.DataFrame | None:
    """
    Fetch all crash records for one year via ArcGIS FeatureServer query.
    Paginates in chunks of 1,000 records (ArcGIS max per request).
    """
    import urllib.request, urllib.parse

    base_url = IDOT_ENDPOINTS.get(year)
    if not base_url:
        return None

    # First: get total record count
    count_params = urllib.parse.urlencode({
        "where": "1=1",
        "returnCountOnly": "true",
        "f": "json",
    })
    try:
        req = urllib.request.Request(
            f"{base_url}?{count_params}",
            headers={"User-Agent": "Mozilla/5.0 (research)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read().decode())
        total = info.get("count", 0)
        log.info("  Year %d: %d total records", year, total)
    except Exception as e:
        log.warning("  Count query failed for %d: %s", year, e)
        return None

    if total == 0:
        return None

    # Try CSV export with all records (if server supports it)
    # ArcGIS 10.8+ allows resultRecordCount=-1 for bulk export
    bulk_params = urllib.parse.urlencode({
        "where": "1=1",
        "outFields": "CRASH_DATE,COUNTY,INJURIES_FATAL,INJURIES_INCAPACITATING,"
                     "INJURIES_NON_INCAPACITATING,COUNTY_NAME",
        "resultRecordCount": min(total, 32000),  # reasonable chunk
        "f": "csv",
    })
    try:
        req = urllib.request.Request(
            f"{base_url}?{bulk_params}",
            headers={"User-Agent": "Mozilla/5.0 (research)"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
        df = pd.read_csv(io.StringIO(content.decode("latin1")), low_memory=False)
        log.info("  Got %d rows in bulk request", len(df))
        if len(df) >= total * 0.9:
            return df
    except Exception as e:
        log.debug("  Bulk CSV failed: %s — falling back to pagination", e)

    # Paginate
    parts = []
    offset = 0
    page_size = 1000
    while offset < total:
        params = urllib.parse.urlencode({
            "where": "1=1",
            "outFields": "CRASH_DATE,COUNTY,COUNTY_NAME,INJURIES_FATAL,"
                         "INJURIES_INCAPACITATING,INJURIES_NON_INCAPACITATING",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "csv",
        })
        try:
            req = urllib.request.Request(
                f"{base_url}?{params}",
                headers={"User-Agent": "Mozilla/5.0 (research)"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                chunk = resp.read()
            chunk_df = pd.read_csv(io.StringIO(chunk.decode("latin1")), low_memory=False)
            parts.append(chunk_df)
            offset += len(chunk_df)
            if len(chunk_df) < page_size:
                break
        except Exception as e:
            log.warning("  Page offset=%d failed: %s", offset, e)
            break
        time.sleep(0.2)

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def process_idot_df(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Convert raw IDOT CSV to county-day aggregation."""
    if df is None or df.empty:
        return None

    df.columns = [c.upper().strip() for c in df.columns]
    log.info("  Columns available: %s", list(df.columns)[:15])

    # Date column
    date_col = next((c for c in df.columns if "CRASH_DATE" in c or "DATE" in c), None)
    if date_col is None:
        log.warning("  No date column for %d", year)
        return None
    df["crash_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["crash_date"])

    # County → FIPS
    county_name_col = next((c for c in df.columns if "COUNTY" in c and "NAME" in c), None)
    county_code_col = next((c for c in df.columns if c == "COUNTY"), None)

    if county_name_col:
        df["county_upper"] = df[county_name_col].astype(str).str.upper().str.strip()
        df["fips"] = df["county_upper"].map(IL_COUNTY_FIPS_BY_NAME)
    elif county_code_col:
        # numeric code → need offset; try assuming it's 1–102 (IL has 102 counties)
        df["county_num"] = pd.to_numeric(df[county_code_col], errors="coerce")
        df["fips"] = ("17" + (df["county_num"] * 2 - 1).astype("Int64").astype(str).str.zfill(3))
        # validate
        valid = df["fips"].isin(IL_COUNTY_FIPS_BY_NAME.values())
        if valid.mean() < 0.8:
            # try direct mapping as FIPS suffix
            df["fips"] = "17" + df["county_num"].astype("Int64").astype(str).str.zfill(3)
    else:
        log.warning("  No county identifier for %d", year)
        return None

    df = df.dropna(subset=["fips"])

    # Injury counts
    fatal_col   = next((c for c in df.columns if "FATAL" in c and "INJUR" in c), None)
    serious_col = next((c for c in df.columns if "INCAPAC" in c), None)

    if fatal_col:
        df["fatals"] = pd.to_numeric(df[fatal_col], errors="coerce").fillna(0)
    else:
        df["fatals"] = 0
    if serious_col:
        df["serious_inj"] = pd.to_numeric(df[serious_col], errors="coerce").fillna(0)
    else:
        df["serious_inj"] = 0

    # Aggregate
    agg = (df.groupby(["fips", "crash_date"])
              .agg(il_fatals=("fatals", "sum"),
                   il_serious_inj=("serious_inj", "sum"),
                   il_crashes=("fips", "count"))
              .reset_index()
              .rename(columns={"crash_date": "date"}))
    log.info("  → %d county-days  fatals=%.0f  serious=%.0f",
             len(agg), agg["il_fatals"].sum(), agg["il_serious_inj"].sum())
    return agg


# ── Main download loop ────────────────────────────────────────────────────────
log.info("Downloading Illinois IDOT crash data via ArcGIS …")
parts = []
for yr in range(2016, 2025):
    log.info("Year %d …", yr)
    raw = fetch_idot_year_arcgis(yr)
    agg = process_idot_df(raw, yr)
    if agg is not None:
        parts.append(agg)
    time.sleep(2.0)
    gc.collect()

if not parts:
    log.error("No Illinois data downloaded. Check network access.")
    log.info("Manual alternative: visit https://gis-idot.opendata.arcgis.com/ "
             "and download per-year CSV files, then re-run with local files.")
    sys.exit(1)

il_panel = pd.concat(parts, ignore_index=True)
il_panel["date"] = pd.to_datetime(il_panel["date"])
il_panel = (il_panel.groupby(["fips", "date"])
                     .agg(il_fatals=("il_fatals", "sum"),
                          il_serious_inj=("il_serious_inj", "sum"),
                          il_crashes=("il_crashes", "sum"))
                     .reset_index())

log.info("\nFinal Illinois IDOT panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(il_panel), il_panel["fips"].nunique(),
         il_panel["date"].min().date(), il_panel["date"].max().date())
log.info("  Total fatals: %.0f  Total serious injuries: %.0f",
         il_panel["il_fatals"].sum(), il_panel["il_serious_inj"].sum())

il_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
