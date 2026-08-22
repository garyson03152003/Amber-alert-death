"""
build_nevada_ndot.py
========================================================
Download Nevada DOT crash data from the NDOT open-data FeatureServer and
build a county-day panel of fatalities and serious injuries.

Source: Nevada DOT CrashData OpenData FeatureServer
URL: https://gis.dot.nv.gov/arcgis/rest/services/ArcGISOnline/CrashData_OpenData/FeatureServer/0
Coverage: 2016–2024 (489k+ records; 2013-2015 return 0 records)
No authentication required.

Key fields:
  Crash_Date    — epoch milliseconds
  County        — county name (UPPERCASE, e.g. "CLARK")
  Crash_Year    — 4-digit integer year
  Fatalities    — count of fatalities per crash
  Injured       — count of injured persons per crash (all severity levels)
  Crash_Severity — "FATAL CRASH", "INJURY CRASH", "PROPERTY DAMAGE ONLY"
  Injury_Type   — KABCO code for most serious injury in crash (K/A/B/C/null)

Serious injuries proxy:
  sum(Injured) for crashes where Injury_Type = 'A'
  (A = Suspected Serious Injury, the most severe injury in the crash)

Nevada has 17 counties. Max record count = 300,000 per page.
Largest year (2019) = 98,592 records — fits in one request.

Output: data/processed/nevada_ndot_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from state_dot_sources import strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise

warnings.filterwarnings("ignore")
log = get_logger("nevada_ndot")

OUT_PATH = DATA_PROC / "nevada_ndot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FEATURE_SERVER = (
    "https://gis.dot.nv.gov/arcgis/rest/services/ArcGISOnline/CrashData_OpenData/FeatureServer/0/query"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS      = list(range(2016, 2025))   # 2016–2024 (2013-2015 have 0 records)
OUT_FIELDS = "Crash_Date,County,Crash_Year,Fatalities,Injured,Crash_Severity,Injury_Type"
PAGE_SIZE  = 100_000   # well under server's 300k max
FETCH_FAILURES: dict[int, BaseException] = {}

# ── Nevada county FIPS mapping ────────────────────────────────────────────────
NV_COUNTY_FIPS = {
    "CHURCHILL":   "32001",
    "CLARK":       "32003",
    "DOUGLAS":     "32005",
    "ELKO":        "32007",
    "ESMERALDA":   "32009",
    "EUREKA":      "32011",
    "HUMBOLDT":    "32013",
    "LANDER":      "32015",
    "LINCOLN":     "32017",
    "LYON":        "32019",
    "MINERAL":     "32021",
    "NYE":         "32023",
    "PERSHING":    "32027",
    "STOREY":      "32029",
    "WASHOE":      "32031",
    "WHITE PINE":  "32033",
    "CARSON CITY": "32510",  # independent city / consolidated county
    "CARSON":      "32510",  # alternate short form
}
VALID_NV_FIPS = set(NV_COUNTY_FIPS.values())


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    """Download all crash records for one year with pagination."""
    where_clause = f"Crash_Year = {year}"

    # Count
    try:
        r = session.get(FEATURE_SERVER, params={
            "where": where_clause, "returnCountOnly": "true", "f": "json"
        }, timeout=45)
        r.raise_for_status()
        total = r.json().get("count", 0)
        log.info("  [%d] %d records", year, total)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.warning("  [%d] count query failed: %s", year, exc)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skipping", year)
        return None

    try:
        return strict_arcgis_dataframe(session, url=FEATURE_SERVER, where=where_clause,
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return None

    # Paginate
    parts = []
    offset = 0
    while offset < total:
        try:
            r = session.get(FEATURE_SERVER, params={
                "where": where_clause,
                "outFields": OUT_FIELDS,
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "json",
            }, timeout=120)
            r.raise_for_status()
            page = r.json()
        except Exception as exc:
            log.warning("  [%d] page offset=%d failed: %s", year, offset, exc)
            break

        features = page.get("features", [])
        if not features:
            break
        rows = [f["attributes"] for f in features]
        parts.append(pd.DataFrame(rows))
        offset += len(rows)
        log.info("  [%d] fetched %d/%d", year, offset, total)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.5)

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Aggregate raw crash records to county-day panel."""
    if df is None or df.empty:
        return None
    df = df.copy()

    # ── Date ─────────────────────────────────────────────────────────────────
    df["crash_date"] = pd.to_datetime(df["Crash_Date"], unit="ms", errors="coerce")
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    # ── County → FIPS ────────────────────────────────────────────────────────
    df["county_upper"] = df["County"].astype(str).str.strip().str.upper()
    df["fips"] = df["county_upper"].map(NV_COUNTY_FIPS)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        log.warning("  [%d] %d rows unmapped county: %s",
                    year, n_miss,
                    df.loc[df["fips"].isna(), "county_upper"].value_counts().head(5).to_dict())
    df = df.dropna(subset=["fips"])

    # ── Severity ─────────────────────────────────────────────────────────────
    df["fatals"]      = pd.to_numeric(df["Fatalities"], errors="coerce").fillna(0)
    df["all_injured"] = pd.to_numeric(df["Injured"],    errors="coerce").fillna(0)

    # Serious injuries = injured persons in crashes whose most-severe injury is A
    df["injury_type_up"] = df["Injury_Type"].astype(str).str.upper().str.strip()
    df["serious_inj"]    = df["all_injured"].where(df["injury_type_up"] == "A", 0)

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(
              nv_fatals     =("fatals",      "sum"),
              nv_injury_proxy=("serious_inj", "sum"),
              nv_all_injured=("all_injured", "sum"),
              nv_crashes    =("fatals",      "count"),
          )
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  nv_fatals=%.0f  nv_serious_inj=%.0f",
             year, len(agg), agg["nv_fatals"].sum(), agg["nv_injury_proxy"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Nevada NDOT crash data (2016–2024) …")

session = requests.Session()
session.headers.update(HEADERS)
parts = []
coverage_rows = []

for yr in YEARS:
    log.info("Year %d …", yr)
    raw = fetch_year(session, yr)
    coverage_rows.append(validate_source_frame("NV", yr, raw,
        required_columns={"Crash_Date", "County", "Fatalities", "Injured", "Injury_Type"},
        date_column="Crash_Date", outcome_columns={"Fatalities", "Injured"}, date_unit="ms",
        geography_column="County", geography_mapper=NV_COUNTY_FIPS,
        terminal_error=FETCH_FAILURES.get(yr)))
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    time.sleep(2.0)
    gc.collect()

session.close()
write_state_manifest_or_raise("NV", coverage_rows, output_dir=DATA_PROC / "coverage")

if not parts:
    log.error("No Nevada data downloaded.")
    sys.exit(1)

nv_panel = pd.concat(parts, ignore_index=True)
nv_panel["date"] = pd.to_datetime(nv_panel["date"])

nv_panel = (
    nv_panel.groupby(["fips", "date"])
      .agg(
          nv_fatals     =("nv_fatals",      "sum"),
          nv_injury_proxy=("nv_injury_proxy", "sum"),
          nv_all_injured=("nv_all_injured", "sum"),
          nv_crashes    =("nv_crashes",     "sum"),
      )
      .reset_index()
)

log.info("\nFinal Nevada NDOT panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(nv_panel), nv_panel["fips"].nunique(),
         nv_panel["date"].min().date(), nv_panel["date"].max().date())
nv_panel["nv_serious_inj"] = np.nan
log.info("  Total nv_fatals: %.0f  Total nv_injury_proxy: %.0f",
         nv_panel["nv_fatals"].sum(), nv_panel["nv_injury_proxy"].sum())

nv_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
