"""
build_texas_txdot.py
=================================================================
Download TxDOT CRIS crash data (all crashes statewide, 2020–2024)
from the TxDOT ArcGIS FeatureServer and build a county-day panel.

Source:
  https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/
  TXDOT_Statewide_Bicyclist_Involved_Crashes/FeatureServer/0
  (name is misleading — contains ALL Texas crashes, not bicycle-only)

Key fields used:
  crash_id         — unique crash identifier (one row per crash)
  cnty_id          — CRIS county code (1–254, alphabetical)
  crash_date       — crash date "YYYY-MM-DD"
  death_cnt        — number of fatalities
  sus_serious_injry_cnt — suspected serious injuries
  crash_fatal_fl   — 1 if any fatality

County FIPS formula: CRIS code N → "48" + f"{2*N-1:03d}"
  e.g. CRIS 1 (Anderson) → 48001, CRIS 15 (Bexar) → 48029,
       CRIS 101 (Harris/Houston) → 48201, CRIS 254 (Zavala) → 48507

Output: data/processed/texas_txdot_county_day.parquet
  columns: fips, date, tx_crashes, tx_fatals, tx_serious_inj
"""
import gc, sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("texas_txdot")

OUT_PATH = DATA_PROC / "texas_txdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FS_URL   = ("https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/"
             "TXDOT_Statewide_Bicyclist_Involved_Crashes/FeatureServer/0/query")
HEADERS  = {"User-Agent": "amber-research/1.0 (academic)"}
YEARS    = list(range(2020, 2025))   # 2020–2024 available
PAGE_SIZE = 2000                      # maxRecordCount
OUT_FIELDS = "crash_id,cnty_id,crash_date,death_cnt,sus_serious_injry_cnt,crash_fatal_fl"
SLEEP_PAGE = 0.25   # seconds between pages
SLEEP_YEAR = 3.0    # seconds between years


def cris_to_fips(cris_id: int) -> str:
    """Convert TxDOT CRIS county code (1–254) to 5-digit FIPS string."""
    return f"48{2 * cris_id - 1:03d}"


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    where = f"crash_date >= '{year}-01-01' AND crash_date <= '{year}-12-31'"

    # Get total count
    try:
        r = session.get(FS_URL, params={"where": where, "returnCountOnly": "true", "f": "json"},
                        timeout=30)
        r.raise_for_status()
        total = r.json().get("count", 0)
        log.info("  [%d] total records: %d", year, total)
    except Exception as e:
        log.warning("  [%d] count query failed: %s", year, e)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skip", year)
        return None

    # Paginate
    parts, offset = [], 0
    while offset < total:
        params = {
            "where": where,
            "outFields": OUT_FIELDS,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        try:
            r = session.get(FS_URL, params=params, timeout=120)
            r.raise_for_status()
            feats = r.json().get("features", [])
        except Exception as e:
            log.warning("  [%d] page offset=%d failed: %s — retry", year, offset, e)
            time.sleep(4)
            try:
                r = session.get(FS_URL, params=params, timeout=120)
                r.raise_for_status()
                feats = r.json().get("features", [])
            except Exception as e2:
                log.warning("  [%d] retry failed: %s — stopping", year, e2)
                break

        if not feats:
            break

        parts.append(pd.DataFrame([f["attributes"] for f in feats]))
        offset += len(feats)

        if offset % 100_000 == 0 or offset >= total:
            log.info("  [%d] … %d / %d", year, offset, total)
        time.sleep(SLEEP_PAGE)

    if not parts:
        log.warning("  [%d] no pages collected", year)
        return None

    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] fetched %d rows", year, len(df))
    return df


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()

    # Parse date
    df["date"] = pd.to_datetime(df["crash_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # Keep valid county codes
    df["cnty_id"] = pd.to_numeric(df["cnty_id"], errors="coerce")
    df = df[(df["cnty_id"] >= 1) & (df["cnty_id"] <= 254)].copy()

    # Map to FIPS
    df["fips"] = df["cnty_id"].apply(lambda x: cris_to_fips(int(x)))

    # Outcomes
    df["death_cnt"]           = pd.to_numeric(df["death_cnt"],           errors="coerce").fillna(0)
    df["sus_serious_injry_cnt"] = pd.to_numeric(df["sus_serious_injry_cnt"], errors="coerce").fillna(0)
    df["crash_fatal_fl"]      = pd.to_numeric(df["crash_fatal_fl"],      errors="coerce").fillna(0)

    # Deduplicate by crash_id (each crash_id should be unique, but just in case)
    df = df.drop_duplicates(subset=["crash_id"]).copy()

    # Aggregate to county-day
    agg = (
        df.groupby(["fips", "date"])
        .agg(
            tx_crashes     =("crash_id",           "count"),
            tx_fatals      =("death_cnt",           "sum"),
            tx_serious_inj =("sus_serious_injry_cnt","sum"),
        )
        .reset_index()
    )
    log.info(
        "  [%d] → %d county-days  crashes=%.0f  fatals=%.0f  serious_inj=%.0f",
        year, len(agg), agg["tx_crashes"].sum(),
        agg["tx_fatals"].sum(), agg["tx_serious_inj"].sum()
    )
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────
log.info("Downloading TxDOT statewide crash data (2020–2024) …")
session = requests.Session()
session.headers.update(HEADERS)
parts = []

for yr in YEARS:
    log.info("Year %d …", yr)
    raw = fetch_year(session, yr)
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(SLEEP_YEAR)

session.close()

if not parts:
    log.error("No Texas data downloaded.")
    sys.exit(1)

panel = pd.concat(parts, ignore_index=True)
panel = (
    panel.groupby(["fips", "date"])
    .agg(tx_crashes=("tx_crashes","sum"), tx_fatals=("tx_fatals","sum"),
         tx_serious_inj=("tx_serious_inj","sum"))
    .reset_index()
)
panel = panel.sort_values(["fips", "date"]).reset_index(drop=True)
panel["date"] = pd.to_datetime(panel["date"])

log.info("\nFinal Texas TxDOT panel:")
log.info("  Rows: %d  Counties: %d  %s – %s",
         len(panel), panel["fips"].nunique(),
         panel["date"].min().date(), panel["date"].max().date())
log.info("  tx_crashes=%.0f  tx_fatals=%.0f  tx_serious_inj=%.0f",
         panel["tx_crashes"].sum(), panel["tx_fatals"].sum(),
         panel["tx_serious_inj"].sum())

panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
