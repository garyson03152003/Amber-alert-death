"""
build_florida_fdot.py
========================================================
Download Florida FDOT crash data from the ArcGIS FeatureServer and
build a county-day panel of fatalities and serious injuries.

Source: https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0
Coverage: 2013–2019 (service returns 0 records for 2020+)

Key fields:
  CRASH_DATE               — epoch milliseconds
  COUNTY_TXT               — county name (uppercase)
  DOT_CNTY_CD              — FDOT county code
  NUMBER_OF_KILLED         — fatalities
  NUMBER_OF_SERIOUS_INJURIES — serious (incapacitating) injuries
  CALENDAR_YEAR            — used for year-by-year WHERE clause

Output: data/processed/florida_fdot_county_day.parquet
"""
import sys, warnings, gc, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from state_dot_sources import strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise

warnings.filterwarnings("ignore")
log = get_logger("florida_fdot")

OUT_PATH = DATA_PROC / "florida_fdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FEATURE_SERVER = (
    "https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0/query"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2013, 2019))   # 2019 is documented incomplete and excluded

OUT_FIELDS = (
    "CRASH_DATE,COUNTY_TXT,DOT_CNTY_CD,"
    "NUMBER_OF_KILLED,NUMBER_OF_SERIOUS_INJURIES,CALENDAR_YEAR"
)

PAGE_SIZE = 1000   # gis.fdot.gov maxRecordCount is 1000
SLEEP_PAGE = 0.3   # seconds between pages within a year
SLEEP_YEAR = 2.0   # seconds between years
FETCH_FAILURES: dict[int, BaseException] = {}

# ── Florida county FIPS mapping ───────────────────────────────────────────────
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
    # Common FDOT text variants
    "ST. JOHNS": "12109", "ST. LUCIE": "12111", "ST JOHNS": "12109", "ST LUCIE": "12111",
    "DADE": "12086",
}
VALID_FL_FIPS = set(FL_COUNTY_FIPS.values())


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    """
    Query the FDOT FeatureServer for one calendar year, paginating with
    resultOffset/resultRecordCount.  Returns a raw DataFrame of all records
    for that year, or None if the service returns nothing.
    """
    where_clause = f"CALENDAR_YEAR={year}"

    # ── 1. Get total record count ──────────────────────────────────────────
    try:
        resp = session.get(
            FEATURE_SERVER,
            params={
                "where": where_clause,
                "returnCountOnly": "true",
                "f": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        info = resp.json()
        total = info.get("count", 0)
        log.info("  [%d] total records: %d", year, total)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.warning("  [%d] count query failed: %s", year, exc)
        return None

    if total == 0:
        log.warning("  [%d] service returned 0 records — skipping", year)
        return None

    try:
        return strict_arcgis_dataframe(session, url=FEATURE_SERVER, where=where_clause,
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return None

    # ── 2. Paginate ────────────────────────────────────────────────────────
    parts = []
    offset = 0

    while offset < total:
        params = {
            "where": where_clause,
            "outFields": OUT_FIELDS,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        try:
            resp = session.get(FEATURE_SERVER, params=params, timeout=120)
            resp.raise_for_status()
            page_data = resp.json()
        except Exception as exc:
            log.warning("  [%d] page offset=%d failed: %s", year, offset, exc)
            break

        features = page_data.get("features", [])
        if not features:
            log.info("  [%d] no features at offset=%d — done paginating", year, offset)
            break

        rows = [f["attributes"] for f in features]
        chunk_df = pd.DataFrame(rows)
        parts.append(chunk_df)
        offset += len(chunk_df)

        log.info("  [%d] fetched offset=%d … +%d rows (total so far: %d/%d)",
                 year, offset - len(chunk_df), len(chunk_df), offset, total)

        # Only stop if server returned no rows (true empty response)
        if len(chunk_df) == 0:
            break

        time.sleep(SLEEP_PAGE)

    if not parts:
        log.warning("  [%d] no pages collected", year)
        return None

    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] raw total fetched: %d rows", year, len(df))
    return df


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """
    Convert raw FDOT records for one year into a county-day aggregation.

    Steps:
      1. Parse CRASH_DATE (epoch ms) → date
      2. Map COUNTY_TXT → FIPS
      3. Sum NUMBER_OF_KILLED, NUMBER_OF_SERIOUS_INJURIES, crash count
         by (fips, date)
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # ── Date ─────────────────────────────────────────────────────────────────
    if "CRASH_DATE" not in df.columns:
        log.warning("  [%d] CRASH_DATE column missing; cols=%s", year, list(df.columns))
        return None

    df["crash_date"] = pd.to_datetime(df["CRASH_DATE"], unit="ms", errors="coerce")
    n_bad_date = df["crash_date"].isna().sum()
    if n_bad_date:
        log.warning("  [%d] %d rows with unparseable CRASH_DATE dropped", year, n_bad_date)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()  # keep date part only

    # ── County → FIPS ─────────────────────────────────────────────────────────
    if "COUNTY_TXT" not in df.columns:
        log.warning("  [%d] COUNTY_TXT column missing; cols=%s", year, list(df.columns))
        return None

    df["county_upper"] = df["COUNTY_TXT"].astype(str).str.strip().str.upper()
    df["fips"] = df["county_upper"].map(FL_COUNTY_FIPS)

    unmatched = df["fips"].isna()
    if unmatched.any():
        bad_names = df.loc[unmatched, "county_upper"].value_counts().head(10)
        log.warning(
            "  [%d] %d rows (%.1f%%) with unrecognised COUNTY_TXT:\n%s",
            year, unmatched.sum(), 100 * unmatched.mean(), bad_names.to_string()
        )
    df = df.dropna(subset=["fips"])

    # ── Severity columns ───────────────────────────────────────────────────────
    df["fatals"] = pd.to_numeric(
        df.get("NUMBER_OF_KILLED", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    df["serious_inj"] = pd.to_numeric(
        df.get("NUMBER_OF_SERIOUS_INJURIES", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)

    # ── Aggregate to county-day ────────────────────────────────────────────────
    agg = (
        df.groupby(["fips", "crash_date"])
        .agg(
            fl_fatals     =("fatals",      "sum"),
            fl_serious_inj=("serious_inj", "sum"),
            fl_crashes    =("fatals",       "count"),
        )
        .reset_index()
        .rename(columns={"crash_date": "date"})
    )

    log.info(
        "  [%d] → %d county-days  fl_fatals=%.0f  fl_serious_inj=%.0f",
        year, len(agg), agg["fl_fatals"].sum(), agg["fl_serious_inj"].sum()
    )
    return agg


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main download loop ────────────────────────────────────────────────────────
    log.info("Downloading Florida FDOT crash data (2013–2019) …")

    session = requests.Session()
    session.headers.update(HEADERS)

    parts = []
    coverage_rows = []

    for yr in YEARS:
        log.info("Year %d …", yr)

        raw = fetch_year(session, yr)
        coverage_rows.append(validate_source_frame("FL", yr, raw,
            required_columns={"CRASH_DATE", "COUNTY_TXT", "NUMBER_OF_KILLED", "NUMBER_OF_SERIOUS_INJURIES"},
            date_column="CRASH_DATE", outcome_columns={"NUMBER_OF_KILLED", "NUMBER_OF_SERIOUS_INJURIES"}, date_unit="ms",
            geography_column="COUNTY_TXT", geography_mapper=FL_COUNTY_FIPS,
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(raw, yr)

        if agg is not None:
            parts.append(agg)
        else:
            log.warning("  Year %d: no usable data", yr)

        del raw, agg
        time.sleep(SLEEP_YEAR)
        gc.collect()

    session.close()
    write_state_manifest_or_raise("FL", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Florida data downloaded. Check network access or FDOT service availability.")
        log.info(
            "Manual alternative: visit "
            "https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0 "
            "and query per year."
        )
        sys.exit(1)

    # ── Combine and de-duplicate across years ─────────────────────────────────────
    fl_panel = pd.concat(parts, ignore_index=True)
    fl_panel["date"] = pd.to_datetime(fl_panel["date"])

    # In case any (fips, date) appears in multiple year chunks, sum them
    fl_panel = (
        fl_panel.groupby(["fips", "date"])
        .agg(
            fl_fatals     =("fl_fatals",      "sum"),
            fl_serious_inj=("fl_serious_inj", "sum"),
            fl_crashes    =("fl_crashes",     "sum"),
        )
        .reset_index()
    )

    log.info("\nFinal Florida FDOT panel:")
    log.info(
        "  Rows: %d  Counties: %d  Date range: %s – %s",
        len(fl_panel),
        fl_panel["fips"].nunique(),
        fl_panel["date"].min().date(),
        fl_panel["date"].max().date(),
    )
    log.info(
        "  Total fl_fatals: %.0f  Total fl_serious_inj: %.0f",
        fl_panel["fl_fatals"].sum(),
        fl_panel["fl_serious_inj"].sum(),
    )

    fl_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
