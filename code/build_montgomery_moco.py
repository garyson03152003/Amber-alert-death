"""
build_montgomery_moco.py
========================================================
Download Montgomery County, Maryland crash data from the county's own
Socrata open-data portal and build a county-day panel of crashes,
fatalities, and serious injuries.

Sub-state (single-county) addition: no Maryland *statewide* crash-level
feed exists (MDOT SHA's public ArcGIS layers are fatal-crash-only; an older
data.maryland.gov statewide Socrata dataset is retired). Montgomery County
publishes its own live, actively-updated, crash-level Socrata source with
genuine person-level severity fields -- richer than several full-state
sources already in this project.

Source: Montgomery County MD Open Data (data.montgomerycountymd.gov)
  Crash Reporting - Incidents Data     (bhju-22kf) -- one row per crash
  Crash Reporting - Drivers Data       (mmzv-x632) -- one row per
                                          driver/occupant, person-level
                                          injury_severity, joined by
                                          report_number
  Crash Reporting - Non-Motorists Data (n7fk-dce5) -- one row per
                                          pedestrian/cyclist, same join key
Coverage: 2015-2025 requested (11 full calendar years; the source's current
year, 2026, is still in progress and excluded).
No authentication required.

The incidents table's own `acrs_report_type` is a crash-level severity
*flag* ("Fatal Crash"/"Injury Crash"/"Property Damage Crash"), not a
person-level count -- the true fatality/serious-injury counts come from
summing `injury_severity` across the Drivers and Non-Motorists tables for
each report_number. Observed value casing is inconsistent across years
("Fatal Injury" vs "FATAL INJURY"), so matching is case-insensitive.

Key fields (confirmed by probe):
  crash_date_time  — ISO 8601 timestamp (real calendar date, not just
                      year/month/day-of-week)
  report_number    — unique per crash, joins all three tables
  injury_severity  — person-level, on Drivers/Non-Motorists only

Every crash in this dataset is in Montgomery County (FIPS 24031) by
construction -- there is no county field to validate.

Output columns: fips, date, moco_crashes, moco_fatals, moco_serious_inj
Output: data/processed/montgomery_moco_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from state_dot_sources import strict_socrata_dataframe, validate_source_frame, write_state_manifest_or_raise
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("montgomery_moco")

OUT_PATH = DATA_PROC / "montgomery_moco_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

INCIDENTS_URL = "https://data.montgomerycountymd.gov/resource/bhju-22kf.json"
DRIVERS_URL = "https://data.montgomerycountymd.gov/resource/mmzv-x632.json"
NONMOTORIST_URL = "https://data.montgomerycountymd.gov/resource/n7fk-dce5.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2015, 2026))
PAGE_LIMIT = 50_000
FETCH_FAILURES: dict[int, BaseException] = {}
MOCO_FIPS = "24031"

FATAL_VALUES = {"FATAL INJURY"}
SERIOUS_VALUES = {"SUSPECTED SERIOUS INJURY"}


def fetch_year(session: requests.Session, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    where = f"date_extract_y(crash_date_time) = {year}"
    incidents = strict_socrata_dataframe(session, url=INCIDENTS_URL, where=where,
                                          id_field="report_number", page_size=PAGE_LIMIT)
    # Person-level tables: use the Socrata system row id for pagination --
    # person_id is not guaranteed unique in the Non-Motorists table (a small
    # handful of repeats observed), and uniqueness of the join key is not
    # something this builder depends on; only the crash-level report_number
    # (validated via the incidents table) needs to be unique.
    drivers = strict_socrata_dataframe(session, url=DRIVERS_URL, where=where,
                                        id_field=":id", page_size=PAGE_LIMIT)
    nonmotorists = strict_socrata_dataframe(session, url=NONMOTORIST_URL, where=where,
                                             id_field=":id", page_size=PAGE_LIMIT)
    return incidents, drivers, nonmotorists


def _severity_counts(persons: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if persons is None or persons.empty or "injury_severity" not in persons.columns:
        empty = pd.Series(dtype=int)
        return empty, empty
    severity = persons["injury_severity"].astype(str).str.strip().str.upper()
    fatal = persons.loc[severity.isin(FATAL_VALUES)].groupby("report_number").size()
    serious = persons.loc[severity.isin(SERIOUS_VALUES)].groupby("report_number").size()
    return fatal, serious


def process_year(incidents: pd.DataFrame, drivers: pd.DataFrame, nonmotorists: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if incidents is None or incidents.empty:
        return None
    incidents = incidents.copy()
    incidents["crash_date"] = pd.to_datetime(incidents["crash_date_time"], errors="coerce")
    n_bad_dt = incidents["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable crash_date_time dropped", year, n_bad_dt)
    incidents = incidents.dropna(subset=["crash_date"])
    incidents["crash_date"] = incidents["crash_date"].dt.normalize()
    incidents["fips"] = MOCO_FIPS

    driver_fatal, driver_serious = _severity_counts(drivers)
    nonmotorist_fatal, nonmotorist_serious = _severity_counts(nonmotorists)
    fatal_per_crash = driver_fatal.add(nonmotorist_fatal, fill_value=0)
    serious_per_crash = driver_serious.add(nonmotorist_serious, fill_value=0)

    incidents["moco_fatals"] = incidents["report_number"].map(fatal_per_crash).fillna(0)
    incidents["moco_serious_inj"] = incidents["report_number"].map(serious_per_crash).fillna(0)

    agg = (
        incidents.groupby(["fips", "crash_date"])
          .agg(moco_fatals=("moco_fatals", "sum"), moco_serious_inj=("moco_serious_inj", "sum"),
               moco_crashes=("report_number", "nunique"))
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  moco_crashes=%d  moco_fatals=%.0f  moco_serious_inj=%.0f",
             year, len(agg), agg["moco_crashes"].sum(), agg["moco_fatals"].sum(), agg["moco_serious_inj"].sum())
    return agg


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main ─────────────────────────────────────────────────────────────────────
    log.info("Downloading Montgomery County MD crash data (2015–2025) …")
    log.info("Source: %s", INCIDENTS_URL)

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    coverage_rows = []

    for yr in YEARS:
        log.info("=== Year %d ===", yr)
        try:
            incidents, drivers, nonmotorists = fetch_year(session, yr)
        except Exception as exc:
            FETCH_FAILURES[yr] = exc
            log.error("  [%d] strict Socrata pagination failed: %s", yr, exc)
            incidents, drivers, nonmotorists = None, None, None
        coverage_rows.append(validate_source_frame("MOCO", yr, incidents,
            required_columns={"report_number", "crash_date_time"},
            date_column="crash_date_time", outcome_columns=set(),
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(incidents, drivers, nonmotorists, yr) if incidents is not None else None
        if agg is not None:
            parts.append(agg)
        del incidents, drivers, nonmotorists, agg
        gc.collect()
        time.sleep(0.5)

    session.close()
    write_state_manifest_or_raise("MOCO", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Montgomery County data downloaded — aborting.")
        sys.exit(1)

    moco_panel = pd.concat(parts, ignore_index=True)
    moco_panel["date"] = pd.to_datetime(moco_panel["date"])
    moco_panel = (
        moco_panel.groupby(["fips", "date"])
          .agg(moco_crashes=("moco_crashes", "sum"), moco_fatals=("moco_fatals", "sum"),
               moco_serious_inj=("moco_serious_inj", "sum"))
          .reset_index()
    )

    log.info("")
    log.info("Final Montgomery County MD panel:")
    log.info("  Rows            : %d", len(moco_panel))
    log.info("  Counties        : %d", moco_panel["fips"].nunique())
    log.info("  Date range      : %s – %s", moco_panel["date"].min().date(), moco_panel["date"].max().date())
    log.info("  moco_crashes    : %d", int(moco_panel["moco_crashes"].sum()))
    log.info("  moco_fatals     : %.0f", moco_panel["moco_fatals"].sum())
    log.info("  moco_serious_inj: %.0f", moco_panel["moco_serious_inj"].sum())

    moco_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
