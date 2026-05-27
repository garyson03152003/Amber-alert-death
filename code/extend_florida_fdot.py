"""
extend_florida_fdot.py
=========================================================
Extends the existing Florida FDOT parquet (2013–2016) to include
2017 and 2018, which are now available in the FDOT ArcGIS service.

Reads the existing parquet, downloads only the new years, and
merges them into a combined 2013–2018 file.

Source: https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0
"""
import sys, warnings, gc, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("fl_extend")

OUT_PATH      = DATA_PROC / "florida_fdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FEATURE_SERVER = (
    "https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0/query"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

NEW_YEARS = [2017, 2018]  # 2019 is very incomplete (38K vs 440K for 2018)

OUT_FIELDS = (
    "CRASH_DATE,COUNTY_TXT,DOT_CNTY_CD,"
    "NUMBER_OF_KILLED,NUMBER_OF_SERIOUS_INJURIES,CALENDAR_YEAR"
)

PAGE_SIZE  = 1000
SLEEP_PAGE = 0.3
SLEEP_YEAR = 2.0

FL_COUNTY_FIPS = {
    "ALACHUA": "12001", "BAKER": "12003", "BAY": "12005", "BRADFORD": "12007",
    "BREVARD": "12009", "BROWARD": "12011", "CALHOUN": "12013", "CHARLOTTE": "12015",
    "CITRUS": "12017", "CLAY": "12019", "COLLIER": "12021", "COLUMBIA": "12023",
    "DESOTO": "12027", "DIXIE": "12029", "DUVAL": "12031", "ESCAMBIA": "12033",
    "FLAGLER": "12035", "FRANKLIN": "12037", "GADSDEN": "12039", "GILCHRIST": "12041",
    "GLADES": "12043", "GULF": "12045", "HAMILTON": "12047", "HARDEE": "12049",
    "HENDRY": "12051", "HERNANDO": "12053", "HIGHLANDS": "12055", "HILLSBOROUGH": "12057",
    "HOLMES": "12059", "INDIAN RIVER": "12061", "JACKSON": "12063", "JEFFERSON": "12065",
    "LAFAYETTE": "12067", "LAKE": "12069", "LEE": "12071", "LEON": "12073",
    "LEVY": "12075", "LIBERTY": "12077", "MADISON": "12079", "MANATEE": "12081",
    "MARION": "12083", "MARTIN": "12085", "MIAMI-DADE": "12086", "MONROE": "12087",
    "NASSAU": "12089", "OKALOOSA": "12091", "OKEECHOBEE": "12093", "ORANGE": "12095",
    "OSCEOLA": "12097", "PALM BEACH": "12099", "PASCO": "12101", "PINELLAS": "12103",
    "POLK": "12105", "PUTNAM": "12107", "SAINT JOHNS": "12109", "SAINT LUCIE": "12111",
    "SANTA ROSA": "12113", "SARASOTA": "12115", "SEMINOLE": "12117", "SUMTER": "12119",
    "SUWANNEE": "12121", "TAYLOR": "12123", "UNION": "12125", "VOLUSIA": "12127",
    "WAKULLA": "12129", "WALTON": "12131", "WASHINGTON": "12133",
    "ST. JOHNS": "12109", "ST. LUCIE": "12111", "ST JOHNS": "12109", "ST LUCIE": "12111",
    "DADE": "12086",
}


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    where = f"CALENDAR_YEAR={year}"
    try:
        r = session.get(FEATURE_SERVER, params={"where": where, "returnCountOnly": "true", "f": "json"}, timeout=30)
        r.raise_for_status()
        total = r.json().get("count", 0)
        log.info("  [%d] total records: %d", year, total)
    except Exception as e:
        log.warning("  [%d] count failed: %s", year, e)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skipping", year)
        return None

    parts, offset = [], 0
    while offset < total:
        params = {"where": where, "outFields": OUT_FIELDS,
                  "resultOffset": offset, "resultRecordCount": PAGE_SIZE, "f": "json"}
        try:
            r = session.get(FEATURE_SERVER, params=params, timeout=120)
            r.raise_for_status()
            feats = r.json().get("features", [])
        except Exception as e:
            log.warning("  [%d] page offset=%d failed: %s", year, offset, e)
            break
        if not feats:
            break
        rows = [f["attributes"] for f in feats]
        parts.append(pd.DataFrame(rows))
        offset += len(rows)
        if offset % 50000 == 0:
            log.info("  [%d] … %d / %d", year, offset, total)
        time.sleep(SLEEP_PAGE)

    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] fetched %d rows", year, len(df))
    return df


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["crash_date"] = pd.to_datetime(df["CRASH_DATE"], unit="ms", errors="coerce")
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    df["county_upper"] = df["COUNTY_TXT"].astype(str).str.strip().str.upper()
    df["fips"] = df["county_upper"].map(FL_COUNTY_FIPS)
    df = df.dropna(subset=["fips"])

    df["fatals"]     = pd.to_numeric(df.get("NUMBER_OF_KILLED",           0), errors="coerce").fillna(0)
    df["serious_inj"] = pd.to_numeric(df.get("NUMBER_OF_SERIOUS_INJURIES", 0), errors="coerce").fillna(0)

    agg = (
        df.groupby(["fips", "crash_date"])
        .agg(fl_fatals=("fatals", "sum"), fl_serious_inj=("serious_inj", "sum"),
             fl_crashes=("fatals", "count"))
        .reset_index()
        .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  fatals=%.0f  serious_inj=%.0f",
             year, len(agg), agg["fl_fatals"].sum(), agg["fl_serious_inj"].sum())
    return agg


# ── Load existing parquet ─────────────────────────────────────────────────────
log.info("Loading existing Florida parquet …")
existing = pd.read_parquet(OUT_PATH)
existing["date"] = pd.to_datetime(existing["date"])
log.info("  Existing: %d rows  date range %s – %s  counties %d",
         len(existing), existing["date"].min().date(), existing["date"].max().date(),
         existing["fips"].nunique())

# ── Download only new years ───────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)
new_parts = []

for yr in NEW_YEARS:
    log.info("Year %d …", yr)
    raw = fetch_year(session, yr)
    agg = process_year(raw, yr)
    if agg is not None:
        new_parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(SLEEP_YEAR)

session.close()

if not new_parts:
    log.error("No new data downloaded.")
    sys.exit(1)

new_df = pd.concat(new_parts, ignore_index=True)
new_df["date"] = pd.to_datetime(new_df["date"])
log.info("New years total: %d county-days", len(new_df))

# ── Combine and deduplicate ───────────────────────────────────────────────────
combined = pd.concat([existing, new_df], ignore_index=True)
combined = (
    combined.groupby(["fips", "date"])
    .agg(fl_fatals=("fl_fatals", "sum"), fl_serious_inj=("fl_serious_inj", "sum"),
         fl_crashes=("fl_crashes", "sum"))
    .reset_index()
)
combined = combined.sort_values(["fips", "date"]).reset_index(drop=True)

log.info("\nCombined Florida panel:")
log.info("  Rows: %d  Counties: %d  %s – %s",
         len(combined), combined["fips"].nunique(),
         combined["date"].min().date(), combined["date"].max().date())
log.info("  Total fl_fatals: %.0f  fl_serious_inj: %.0f  fl_crashes: %.0f",
         combined["fl_fatals"].sum(), combined["fl_serious_inj"].sum(),
         combined["fl_crashes"].sum())

combined.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
