"""
build_pennsylvania_penndot.py
========================================================
Download Pennsylvania PennDOT crash data from the Socrata open-data portal
and build a county-month panel of serious injuries and fatalities.

Source: https://data.pa.gov/resource/dc5b-gebx.json
Coverage: 2013–2020 (only years available in this dataset within our target range).
No authentication required; Socrata API is public.

NOTE on date granularity:
  The dataset contains crash_year and crash_month but NO exact day-of-month
  field (day_of_week 1–7 is present but is the weekday, not the calendar day).
  We therefore aggregate to county-MONTH and represent each period as the 1st
  of that month (e.g., 2013-01-01 for January 2013).  The output column is
  still called `date` for consistency with the rest of the pipeline.

Key fields used:
  crash_year     – 4-digit year string ("2013" … "2020")
  crash_month    – numeric month string ("1" … "12")
  county_name    – title-case county name ("Philadelphia", "Allegheny", …)
  fatal_count    – number of fatalities in the crash record
  maj_inj_count  – suspected serious (major) injuries
  mod_inj_count  – moderate injuries
  tot_inj_count  – total injured persons

Output: data/processed/pennsylvania_penndot_county_month.parquet
"""

import sys
import warnings
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("pennsylvania_penndot")

# ── Output path ────────────────────────────────────────────────────────────────
OUT_PATH = DATA_PROC / "pennsylvania_penndot_county_month.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Socrata API settings ───────────────────────────────────────────────────────
BASE_URL   = "https://data.pa.gov/resource/dc5b-gebx.json"
PAGE_LIMIT = 50_000          # rows per request (Socrata allows up to 50k+)
TARGET_YEARS = [str(y) for y in range(2013, 2021)]   # 2013–2020

# ── Pennsylvania county → FIPS mapping ────────────────────────────────────────
# county_name in the data is title case, e.g. "Philadelphia", "Allegheny"
PA_COUNTY_FIPS = {
    "Adams":          "42001",
    "Allegheny":      "42003",
    "Armstrong":      "42005",
    "Beaver":         "42007",
    "Bedford":        "42009",
    "Berks":          "42011",
    "Blair":          "42013",
    "Bradford":       "42015",
    "Bucks":          "42017",
    "Butler":         "42019",
    "Cambria":        "42021",
    "Cameron":        "42023",
    "Carbon":         "42025",
    "Centre":         "42027",
    "Chester":        "42029",
    "Clarion":        "42031",
    "Clearfield":     "42033",
    "Clinton":        "42035",
    "Columbia":       "42037",
    "Crawford":       "42039",
    "Cumberland":     "42041",
    "Dauphin":        "42043",
    "Delaware":       "42045",
    "Elk":            "42047",
    "Erie":           "42049",
    "Fayette":        "42051",
    "Forest":         "42053",
    "Franklin":       "42055",
    "Fulton":         "42057",
    "Greene":         "42059",
    "Huntingdon":     "42061",
    "Indiana":        "42063",
    "Jefferson":      "42065",
    "Juniata":        "42067",
    "Lackawanna":     "42069",
    "Lancaster":      "42071",
    "Lawrence":       "42073",
    "Lebanon":        "42075",
    "Lehigh":         "42077",
    "Luzerne":        "42079",
    "Lycoming":       "42081",
    "McKean":         "42083",
    "Mckean":         "42083",   # str.title() lowercases the K
    "Mercer":         "42085",
    "Mifflin":        "42087",
    "Monroe":         "42089",
    "Montgomery":     "42091",
    "Montour":        "42093",
    "Northampton":    "42095",
    "Northumberland": "42097",
    "Perry":          "42099",
    "Philadelphia":   "42101",
    "Pike":           "42103",
    "Potter":         "42105",
    "Schuylkill":     "42107",
    "Snyder":         "42109",
    "Somerset":       "42111",
    "Sullivan":       "42113",
    "Susquehanna":    "42115",
    "Tioga":          "42117",
    "Union":          "42119",
    "Venango":        "42121",
    "Warren":         "42123",
    "Washington":     "42125",
    "Wayne":          "42127",
    "Westmoreland":   "42129",
    "Wyoming":        "42131",
    "York":           "42133",
}


# ── Step 1: probe all available columns ───────────────────────────────────────
def probe_columns() -> list[str]:
    """Fetch 2 rows with $select=* to discover all column names."""
    params = {"$select": "*", "$limit": 2}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            cols = list(rows[0].keys())
            log.info("Available columns (%d): %s", len(cols), cols)
            return cols
    except Exception as exc:
        log.warning("Column probe failed: %s", exc)
    return []


# ── Step 2: paginate the full dataset (year by year) ──────────────────────────
def fetch_year(year: str, retries: int = 3) -> pd.DataFrame:
    """
    Download all crash records for a single year via Socrata $limit/$offset
    pagination.  Returns a DataFrame for that year.
    """
    where_clause = f"crash_year = '{year}'"
    parts = []
    offset = 0
    page_num = 0

    while True:
        page_num += 1
        params = {
            "$limit":  PAGE_LIMIT,
            "$offset": offset,
            "$where":  where_clause,
        }
        log.info("  [%s] Page %d  (offset=%d) …", year, page_num, offset)
        for attempt in range(retries):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=180)
                resp.raise_for_status()
                rows = resp.json()
                break
            except Exception as exc:
                wait = 5 * (attempt + 1)
                log.warning("  [%s] attempt %d failed: %s; retrying in %ds",
                            year, attempt + 1, exc, wait)
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    log.error("  [%s] page offset=%d: all retries failed", year, offset)
                    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if not rows:
            log.info("  [%s] No more rows at offset=%d — done.", year, offset)
            break

        chunk = pd.DataFrame(rows)
        parts.append(chunk)
        log.info("  [%s] → %d rows (cumulative: %d)", year, len(chunk),
                 sum(len(p) for p in parts))

        if len(rows) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT
        time.sleep(0.5)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_all_pages() -> pd.DataFrame:
    """Download all crash records for TARGET_YEARS, one year at a time."""
    all_parts = []
    for yr in TARGET_YEARS:
        log.info("Fetching year %s …", yr)
        df_yr = fetch_year(yr)
        if not df_yr.empty:
            all_parts.append(df_yr)
            log.info("  Year %s: %d rows", yr, len(df_yr))
        time.sleep(1.0)
    if not all_parts:
        return pd.DataFrame()
    df = pd.concat(all_parts, ignore_index=True)
    log.info("Total raw rows fetched: %d", len(df))
    return df


# ── Step 3: clean and map to county-month panel ───────────────────────────────
def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw API rows to an aggregated county-month panel with columns:
      fips, date, pa_fatals, pa_serious_inj, pa_crashes
    """
    if df.empty:
        log.error("Input DataFrame is empty — cannot build panel.")
        return pd.DataFrame()

    log.info("Raw columns: %s", list(df.columns))

    # ── Date construction ──────────────────────────────────────────────────────
    # The API has crash_year and crash_month but no exact calendar day.
    # Check whether a date or day-of-month column exists.
    # time_of_day is NOT a date column — it contains military-time integers (e.g. 1423)
    date_col = next(
        (c for c in df.columns
         if c.lower() in ("crash_date", "date", "crash_datetime")),
        None
    )
    # day_of_month style columns (not day_of_week, time_of_day, or hour_of_day)
    # We specifically want a column like "crash_day" or "day_of_month"
    # hour_of_day is the hour (0–23), not the calendar day
    dom_col = next(
        (c for c in df.columns
         if "day" in c.lower()
            and "week" not in c.lower()
            and "time" not in c.lower()
            and "hour" not in c.lower()
            and c.lower() not in ("crash_year",)),
        None
    )

    if date_col:
        log.info("Found date column '%s' — parsing directly.", date_col)
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        # Floor to month-start for monthly aggregation
        df["_date"] = df["_date"].dt.to_period("M").dt.to_timestamp()
    elif dom_col and dom_col.lower() not in ("day_of_week",):
        log.info("Found day-of-month column '%s' — building exact date.", dom_col)
        yr = df["crash_year"].astype(str).str.strip()
        mo = df["crash_month"].astype(str).str.strip().str.zfill(2)
        dy = pd.to_numeric(df[dom_col], errors="coerce").fillna(1).astype(int).astype(str).str.zfill(2)
        df["_date"] = pd.to_datetime(yr + "-" + mo + "-" + dy, errors="coerce")
        # Floor to month-start
        df["_date"] = df["_date"].dt.to_period("M").dt.to_timestamp()
    else:
        log.info("No exact day available — aggregating to year-month (date = 1st of month).")
        yr = df["crash_year"].astype(str).str.strip()
        mo = df["crash_month"].astype(str).str.strip().str.zfill(2)
        df["_date"] = pd.to_datetime(yr + "-" + mo + "-01", errors="coerce")

    n_bad_date = df["_date"].isna().sum()
    if n_bad_date:
        log.warning("Dropping %d rows with unparseable date.", n_bad_date)
    df = df.dropna(subset=["_date"])

    # ── County → FIPS ─────────────────────────────────────────────────────────
    # county_name is title case ("Philadelphia", "Allegheny", etc.)
    # Normalise to title case in case of any variation
    if "county_name" not in df.columns:
        log.error("'county_name' column not found; columns are: %s", list(df.columns))
        return pd.DataFrame()

    df["_county_title"] = df["county_name"].astype(str).str.strip().str.title()
    df["fips"] = df["_county_title"].map(PA_COUNTY_FIPS)

    n_missing_fips = df["fips"].isna().sum()
    if n_missing_fips:
        unmatched = sorted(df.loc[df["fips"].isna(), "_county_title"].unique())
        log.warning("Could not map %d rows to FIPS (%.1f%%). Unmatched: %s",
                    n_missing_fips, 100 * n_missing_fips / len(df), unmatched[:20])
    df = df.dropna(subset=["fips"])

    # ── Severity fields ────────────────────────────────────────────────────────
    # fatal_count     – fatalities
    # maj_inj_count   – major/serious injuries (KABCO: A-injury)
    # mod_inj_count   – moderate injuries
    # tot_inj_count   – total injured

    def _num(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
        log.warning("Column '%s' not found — treating as 0.", col)
        return pd.Series(0, index=df.index)

    df["_fatals"]      = _num("fatal_count")
    df["_serious_inj"] = _num("maj_inj_count")
    df["_crashes"]     = 1   # one row = one crash record

    log.info("Severity columns mapped:  fatal_count → pa_fatals,  "
             "maj_inj_count → pa_serious_inj")

    # ── Aggregate to county-month ─────────────────────────────────────────────
    panel = (
        df.groupby(["fips", "_date"])
          .agg(
              pa_fatals     =("_fatals",      "sum"),
              pa_serious_inj=("_serious_inj", "sum"),
              pa_crashes    =("_crashes",     "sum"),
          )
          .reset_index()
          .rename(columns={"_date": "date"})
    )

    panel["date"]          = pd.to_datetime(panel["date"])
    panel["pa_fatals"]     = panel["pa_fatals"].astype(int)
    panel["pa_serious_inj"]= panel["pa_serious_inj"].astype(int)
    panel["pa_crashes"]    = panel["pa_crashes"].astype(int)

    log.info("Panel shape: %d county-months   Counties: %d   Date range: %s – %s",
             len(panel), panel["fips"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())
    log.info("Totals:  pa_fatals=%.0f   pa_serious_inj=%.0f   pa_crashes=%.0f",
             panel["pa_fatals"].sum(), panel["pa_serious_inj"].sum(),
             panel["pa_crashes"].sum())

    return panel


# ── Main ──────────────────────────────────────────────────────────────────────
log.info("=== Pennsylvania PennDOT crash data download ===")
log.info("Source: %s", BASE_URL)
log.info("Years:  %s", TARGET_YEARS)

# Probe available columns first
log.info("Probing available columns …")
available_cols = probe_columns()

# Download all pages
log.info("Fetching crash records …")
raw_df = fetch_all_pages()
gc.collect()

if raw_df.empty:
    log.error("No data downloaded. Check network access and API availability.")
    sys.exit(1)

# Build the county-month panel
log.info("Building county-month panel …")
panel = build_panel(raw_df)
del raw_df
gc.collect()

if panel.empty:
    log.error("Panel is empty after processing. Exiting.")
    sys.exit(1)

# Save
panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
log.info("Done.")
