"""
build_utah_udot.py
========================================================
Download Utah statewide crash data from UDOT's Crash_Locations MapServer
(one layer per year) and build a county-day panel of fatalities and serious
injuries.

Source: UDOT Crash_Locations MapServer
URL: https://maps.udot.utah.gov/central/rest/services/TrafficAndSafety/Crash_Locations/MapServer
Coverage: 2018-2024 requested (source has 2018-2025; 2025 is partial and
outside the FOIA AMBER-alert window). One layer ID per year.
No authentication required.

Key fields (confirmed by probe):
  CRASH_DATETIME       — epoch milliseconds
  COUNTY_NAME          — county name, uppercase (e.g. "SALT LAKE", "WEBER")
  NUMBER_FATALITIES    — person-level fatality count per crash
  NUMBER_FOUR_INJURIES — person-level "Level 4" (suspected serious) injury count

Output columns: fips, date, ut_fatals, ut_serious_inj, ut_crashes
Output: data/processed/utah_udot_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from state_dot_sources import _odd_fips, strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("utah_udot")

OUT_PATH = DATA_PROC / "utah_udot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

BASE = "https://maps.udot.utah.gov/central/rest/services/TrafficAndSafety/Crash_Locations/MapServer"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

# One separate MapServer layer per year.
YEAR_LAYER = {2018: 10, 2019: 9, 2020: 8, 2021: 6, 2022: 7, 2023: 12, 2024: 13}
PAGE_SIZE = 2_000  # server maxRecordCount
FETCH_FAILURES: dict[int, BaseException] = {}
OUT_FIELDS = "CRASH_DATETIME,COUNTY_NAME,NUMBER_FATALITIES,NUMBER_FOUR_INJURIES"

# Utah's 29 counties are FIPS-coded sequentially in alphabetical order
# (49001, 49003, ... 49057), the same convention already used for several
# other states in this codebase.
_UT_COUNTIES_ALPHA = [
    "BEAVER", "BOX ELDER", "CACHE", "CARBON", "DAGGETT", "DAVIS", "DUCHESNE",
    "EMERY", "GARFIELD", "GRAND", "IRON", "JUAB", "KANE", "MILLARD",
    "MORGAN", "PIUTE", "RICH", "SALT LAKE", "SAN JUAN", "SANPETE", "SEVIER",
    "SUMMIT", "TOOELE", "UINTAH", "UTAH", "WASATCH", "WASHINGTON", "WAYNE",
    "WEBER",
]
UT_COUNTY_FIPS = dict(zip(_UT_COUNTIES_ALPHA, sorted(_odd_fips("49", 29))))


def county_to_fips(name: str) -> str | None:
    if name is None:
        return None
    return UT_COUNTY_FIPS.get(str(name).strip().upper())


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    layer = YEAR_LAYER[year]
    query_url = f"{BASE}/{layer}/query"
    try:
        r = session.get(query_url, params={
            "where": "1=1", "returnCountOnly": "true", "f": "json",
        }, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if "error" in resp:
            log.warning("  [%d] count query error: %s", year, resp["error"])
            return None
        total = resp.get("count", 0)
        log.info("  [%d] %d records to fetch (layer %d)", year, total, layer)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.warning("  [%d] count query failed: %s", year, exc)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skipping", year)
        return None

    try:
        return strict_arcgis_dataframe(session, url=query_url, where="1=1",
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return None


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    # CRASH_DATETIME is ArcGIS epoch milliseconds, which are UTC, and it
    # carries a real time-of-day component. It MUST be converted to Utah
    # local time before the calendar date is taken.
    #
    # Parsing the epoch without a timezone yields naive UTC: Denver is
    # UTC-7/-6, so every crash at local hour >= 17 falls on the *following*
    # UTC date. That silently shifted 32.8% of all Utah crashes forward by a
    # day. Verified directly against the county-hour panel -- reassigning
    # local timestamps to their UTC date reproduces the old daily series
    # exactly (match share 1.0000, mean abs diff 0.0).
    #
    # Local dates are the correct grain here: FARS and the AMBER-alert
    # treatment are both aligned on local calendar date.
    df["crash_date"] = (
        pd.to_datetime(df["CRASH_DATETIME"], unit="ms", errors="coerce", utc=True)
        .dt.tz_convert("America/Denver")
        .dt.tz_localize(None)
    )
    n_bad_dt = df["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable CRASH_DATETIME dropped", year, n_bad_dt)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    df["fips"] = df["COUNTY_NAME"].map(county_to_fips)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        unmapped = df.loc[df["fips"].isna(), "COUNTY_NAME"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped county: %s", year, n_miss, unmapped)
    df = df.dropna(subset=["fips"])

    df["fatals"] = pd.to_numeric(df["NUMBER_FATALITIES"], errors="coerce").fillna(0)
    df["serious_inj"] = pd.to_numeric(df["NUMBER_FOUR_INJURIES"], errors="coerce").fillna(0)

    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(ut_fatals=("fatals", "sum"), ut_serious_inj=("serious_inj", "sum"),
               ut_crashes=("fatals", "count"))
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  ut_fatals=%.0f  ut_serious_inj=%.0f  ut_crashes=%d",
             year, len(agg), agg["ut_fatals"].sum(), agg["ut_serious_inj"].sum(), agg["ut_crashes"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
# Executed only as a script. Without this guard the whole download-and-
# write pipeline ran on *import*, so merely importing this module (from a
# test, a notebook, or another builder) silently re-downloaded the source
# and overwrote the Utah panel on disk.
if __name__ == "__main__":
    log.info("Downloading Utah UDOT crash data (2018–2024) …")
    log.info("Source: %s", BASE)

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    coverage_rows = []

    for yr in sorted(YEAR_LAYER):
        log.info("=== Year %d ===", yr)
        raw = fetch_year(session, yr)
        coverage_rows.append(validate_source_frame("UT", yr, raw,
            required_columns={"CRASH_DATETIME", "COUNTY_NAME", "NUMBER_FATALITIES", "NUMBER_FOUR_INJURIES"},
            date_column="CRASH_DATETIME", outcome_columns={"NUMBER_FATALITIES", "NUMBER_FOUR_INJURIES"}, date_unit="ms",
            geography_column="COUNTY_NAME", geography_mapper=county_to_fips,
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(raw, yr)
        if agg is not None:
            parts.append(agg)
        del raw, agg
        gc.collect()
        time.sleep(1.0)

    session.close()
    write_state_manifest_or_raise("UT", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Utah data downloaded — aborting.")
        sys.exit(1)

    ut_panel = pd.concat(parts, ignore_index=True)
    ut_panel["date"] = pd.to_datetime(ut_panel["date"])
    ut_panel = (
        ut_panel.groupby(["fips", "date"])
          .agg(ut_fatals=("ut_fatals", "sum"), ut_serious_inj=("ut_serious_inj", "sum"),
               ut_crashes=("ut_crashes", "sum"))
          .reset_index()
    )

    log.info("")
    log.info("Final Utah UDOT panel:")
    log.info("  Rows          : %d", len(ut_panel))
    log.info("  Counties      : %d", ut_panel["fips"].nunique())
    log.info("  Date range    : %s – %s", ut_panel["date"].min().date(), ut_panel["date"].max().date())
    log.info("  ut_fatals     : %.0f", ut_panel["ut_fatals"].sum())
    log.info("  ut_serious_inj: %.0f", ut_panel["ut_serious_inj"].sum())
    log.info("  ut_crashes    : %d", int(ut_panel["ut_crashes"].sum()))

    ut_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
