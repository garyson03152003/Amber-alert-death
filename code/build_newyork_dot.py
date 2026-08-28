"""
build_newyork_dot.py
========================================================
Download New York State police-reported crash data from the NY Open Data
portal and build a county-day panel of crash counts and fatalities.

Source: NY Open Data — Motor Vehicle Crashes (Statewide)
URL: https://data.ny.gov/resource/e8ky-4vqe.json  (Socrata)
Coverage: 2021–2024 (earlier years not available on this portal)
No authentication required.

NOTE on severity:
  The dataset uses `accident_descriptor` for severity classification:
    "Fatal Accident"                  → at least one fatality
    "Injury Accident"                 → injuries, no fatality
    "Property Damage & Injury Accident" → injuries + property damage
    "Property Damage Accident"        → property damage only (PDO)

  There are NO explicit killed/injured person counts. We therefore:
    ny_fatals      = crash count where accident_descriptor = 'Fatal Accident'
    ny_injury_crashes = crash count where descriptor contains 'Injury'
  Note: ny_fatals counts fatal CRASHES, not individual fatalities.

NOTE on granularity:
  No exact day-of-month precision issues; the date field is date-only (no time).

Output: data/processed/newyork_dot_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from state_dot_sources import strict_socrata_dataframe, validate_source_frame, write_state_manifest_or_raise

warnings.filterwarnings("ignore")
log = get_logger("newyork_dot")

OUT_PATH = DATA_PROC / "newyork_dot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

BASE_URL   = "https://data.ny.gov/resource/e8ky-4vqe.json"
PAGE_LIMIT = 50_000
HEADERS    = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}
FETCH_FAILURES: dict[int, BaseException] = {}

# ── New York county FIPS mapping ──────────────────────────────────────────────
# county_name in the data is UPPERCASE
NY_COUNTY_FIPS = {
    "ALBANY":        "36001",
    "ALLEGANY":      "36003",
    "BRONX":         "36005",
    "BROOME":        "36007",
    "CATTARAUGUS":   "36009",
    "CAYUGA":        "36011",
    "CHAUTAUQUA":    "36013",
    "CHEMUNG":       "36015",
    "CHENANGO":      "36017",
    "CLINTON":       "36019",
    "COLUMBIA":      "36021",
    "CORTLAND":      "36023",
    "DELAWARE":      "36025",
    "DUTCHESS":      "36027",
    "ERIE":          "36029",
    "ESSEX":         "36031",
    "FRANKLIN":      "36033",
    "FULTON":        "36035",
    "GENESEE":       "36037",
    "GREENE":        "36039",
    "HAMILTON":      "36041",
    "HERKIMER":      "36043",
    "JEFFERSON":     "36045",
    "KINGS":         "36047",   # Brooklyn
    "LEWIS":         "36049",
    "LIVINGSTON":    "36051",
    "MADISON":       "36053",
    "MONROE":        "36055",
    "MONTGOMERY":    "36057",
    "NASSAU":        "36059",
    "NEW YORK":      "36061",   # Manhattan
    "NIAGARA":       "36063",
    "ONEIDA":        "36065",
    "ONONDAGA":      "36067",
    "ONTARIO":       "36069",
    "ORANGE":        "36071",
    "ORLEANS":       "36073",
    "OSWEGO":        "36075",
    "OTSEGO":        "36077",
    "PUTNAM":        "36079",
    "QUEENS":        "36081",
    "RENSSELAER":    "36083",
    "RICHMOND":      "36085",   # Staten Island
    "ROCKLAND":      "36087",
    "ST. LAWRENCE":  "36089",
    "SARATOGA":      "36091",
    "SCHENECTADY":   "36093",
    "SCHOHARIE":     "36095",
    "SCHUYLER":      "36097",
    "SENECA":        "36099",
    "STEUBEN":       "36101",
    "SUFFOLK":       "36103",
    "SULLIVAN":      "36105",
    "TIOGA":         "36107",
    "TOMPKINS":      "36109",
    "ULSTER":        "36111",
    "WARREN":        "36113",
    "WASHINGTON":    "36115",
    "WAYNE":         "36117",
    "WESTCHESTER":   "36119",
    "WYOMING":       "36121",
    "YATES":         "36123",
    # Common alternate spellings
    "ST LAWRENCE":   "36089",
}


def fetch_year(year: int, session: requests.Session, retries: int = 3) -> pd.DataFrame:
    """Download one year with count-first stable-ID Socrata pagination."""
    where = f"year = '{year}'"
    try:
        return strict_socrata_dataframe(
            session, url=BASE_URL, where=where, id_field=":id", page_size=PAGE_LIMIT
        )
    except Exception as exc:
        # A failed page is intentionally not converted to a partial DataFrame.
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict Socrata pagination failed: %s", year, exc)
        return pd.DataFrame()

    # Historical permissive pager retained below only for provenance; it is
    # unreachable and must not be used for validated builds.
    parts = []
    offset = 0
    page = 0
    while True:
        page += 1
        params = {
            "$limit":  PAGE_LIMIT,
            "$offset": offset,
            "$where":  where,
            "$select": "date,county_name,accident_descriptor",
        }
        for attempt in range(retries):
            try:
                r = session.get(BASE_URL, params=params, timeout=120)
                r.raise_for_status()
                rows = r.json()
                break
            except Exception as exc:
                wait = 5 * (attempt + 1)
                log.warning("  [%d] page %d attempt %d failed: %s; retry in %ds",
                            year, page, attempt + 1, exc, wait)
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    log.error("  [%d] page %d: all retries failed", year, page)
                    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if not rows:
            break
        chunk = pd.DataFrame(rows)
        parts.append(chunk)
        log.info("  [%d] page %d: %d rows (cumulative: %d)", year, page, len(chunk),
                 sum(len(p) for p in parts))
        if len(rows) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(0.5)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main ─────────────────────────────────────────────────────────────────────
    log.info("Downloading New York State crash data (2021–2024) …")

    session = requests.Session()
    session.headers.update(HEADERS)

    # A moving API maximum can be a partial current year.  Validated builds use
    # only the closed source-contract years.
    YEARS = [2021, 2022, 2023, 2024]
    log.info("Fetching years: %s", YEARS)

    all_parts = []
    coverage_rows = []
    for yr in YEARS:
        log.info("Year %d …", yr)
        df_yr = fetch_year(yr, session)
        coverage_rows.append(validate_source_frame("NY", yr,
            None if FETCH_FAILURES.get(yr) is not None else df_yr,
            required_columns={"date", "county_name", "accident_descriptor"}, date_column="date", outcome_columns=set(),
            geography_column="county_name", geography_mapper=lambda value: NY_COUNTY_FIPS.get(str(value).strip().upper()),
            unresolvable_geography_values=frozenset({"UNKNOWN"}),
            terminal_error=FETCH_FAILURES.get(yr)))
        if not df_yr.empty:
            all_parts.append(df_yr)
        time.sleep(1.0)
        gc.collect()

    session.close()
    write_state_manifest_or_raise("NY", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not all_parts:
        log.error("No data downloaded.")
        sys.exit(1)

    raw = pd.concat(all_parts, ignore_index=True)
    log.info("Raw rows: %d", len(raw))

    # ── Parse date ────────────────────────────────────────────────────────────────
    raw["crash_date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw = raw.dropna(subset=["crash_date"])

    # ── County → FIPS ────────────────────────────────────────────────────────────
    raw["county_upper"] = raw["county_name"].astype(str).str.strip().str.upper()
    raw["fips"] = raw["county_upper"].map(NY_COUNTY_FIPS)
    n_miss = raw["fips"].isna().sum()
    if n_miss:
        log.warning("%d rows unmapped county: %s", n_miss,
                    raw.loc[raw["fips"].isna(), "county_upper"].value_counts().head(10).to_dict())
    raw = raw.dropna(subset=["fips"])

    # ── Severity ─────────────────────────────────────────────────────────────────
    desc = raw["accident_descriptor"].astype(str).str.strip()
    raw["is_fatal"]  = desc == "Fatal Accident"
    raw["is_injury"] = desc.str.contains("Injury", case=False, na=False)
    raw["crashes"]   = 1

    # ── Aggregate to county-day ───────────────────────────────────────────────────
    panel = (
        raw.groupby(["fips", "crash_date"])
           .agg(
               ny_crashes       =("crashes",   "sum"),
               ny_fatal_crashes =("is_fatal",  "sum"),   # fatal crashes (not fatalities)
               ny_injury_crashes=("is_injury", "sum"),
           )
           .reset_index()
           .rename(columns={"crash_date": "date"})
    )

    # Deduplicate
    panel = (
        panel.groupby(["fips", "date"])
             .agg(ny_crashes       =("ny_crashes",       "sum"),
                  ny_fatal_crashes =("ny_fatal_crashes",  "sum"),
                  ny_injury_crashes=("ny_injury_crashes", "sum"))
             .reset_index()
    )

    log.info("\nFinal New York panel:")
    log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
             len(panel), panel["fips"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())
    log.info("  Total ny_crashes: %.0f  ny_fatal_crashes: %.0f  ny_injury_crashes: %.0f",
             panel["ny_crashes"].sum(), panel["ny_fatal_crashes"].sum(),
             panel["ny_injury_crashes"].sum())

    panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
