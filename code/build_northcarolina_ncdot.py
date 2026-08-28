"""
build_northcarolina_ncdot.py
========================================================
Download North Carolina statewide crash data from NCDOT's public
StatewideCrashTable FeatureServer (the table backing the NCDOT Statewide
Crash Dashboard) and build a county-day panel of fatalities and serious
injuries.

Source: NCDOT StatewideCrashTable FeatureServer (table layer 3)
URL: https://services.arcgis.com/NuWFvHYDMVmmxMeM/arcgis/rest/services/StatewideCrashTable/FeatureServer/3
Coverage: 2021-2025 available; only 2021-2024 requested (2025 is partial and
outside the FOIA AMBER-alert window).
No authentication required.

Key fields (confirmed by probe):
  Date            — epoch milliseconds
  County          — full county name, title case (e.g. "Durham", "Mecklenburg")
  NumFatalities   — person-level fatality count per crash
  NumAInjuries    — person-level KABCO-A (suspected serious injury) count per crash

Output columns: fips, date, nc_fatals, nc_serious_inj, nc_crashes
Output: data/processed/northcarolina_ncdot_county_day.parquet
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
log = get_logger("northcarolina_ncdot")

OUT_PATH = DATA_PROC / "northcarolina_ncdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FEATURE_SERVER = (
    "https://services.arcgis.com/NuWFvHYDMVmmxMeM/arcgis/rest/services/"
    "StatewideCrashTable/FeatureServer/3/query"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2021, 2025))
PAGE_SIZE = 2_000  # server maxRecordCount
FETCH_FAILURES: dict[int, BaseException] = {}
OUT_FIELDS = "Date,County,NumFatalities,NumAInjuries"

# NC's 100 counties are FIPS-coded sequentially in alphabetical order
# (37001, 37003, ... 37199), the same convention already used for several
# other states in this codebase.
_NC_COUNTIES_ALPHA = [
    "Alamance", "Alexander", "Alleghany", "Anson", "Ashe", "Avery", "Beaufort",
    "Bertie", "Bladen", "Brunswick", "Buncombe", "Burke", "Cabarrus", "Caldwell",
    "Camden", "Carteret", "Caswell", "Catawba", "Chatham", "Cherokee", "Chowan",
    "Clay", "Cleveland", "Columbus", "Craven", "Cumberland", "Currituck", "Dare",
    "Davidson", "Davie", "Duplin", "Durham", "Edgecombe", "Forsyth", "Franklin",
    "Gaston", "Gates", "Graham", "Granville", "Greene", "Guilford", "Halifax",
    "Harnett", "Haywood", "Henderson", "Hertford", "Hoke", "Hyde", "Iredell",
    "Jackson", "Johnston", "Jones", "Lee", "Lenoir", "Lincoln", "Macon",
    "Madison", "Martin", "Mcdowell", "Mecklenburg", "Mitchell", "Montgomery",
    "Moore", "Nash", "New Hanover", "Northampton", "Onslow", "Orange", "Pamlico",
    "Pasquotank", "Pender", "Perquimans", "Person", "Pitt", "Polk", "Randolph",
    "Richmond", "Robeson", "Rockingham", "Rowan", "Rutherford", "Sampson",
    "Scotland", "Stanly", "Stokes", "Surry", "Swain", "Transylvania", "Tyrrell",
    "Union", "Vance", "Wake", "Warren", "Washington", "Watauga", "Wayne",
    "Wilkes", "Wilson", "Yadkin", "Yancey",
]
NC_COUNTY_FIPS = dict(zip(_NC_COUNTIES_ALPHA, sorted(_odd_fips("37", 100))))


def county_to_fips(name: str) -> str | None:
    if name is None:
        return None
    return NC_COUNTY_FIPS.get(str(name).strip())


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    where_clause = f"Year = {year}"
    try:
        r = session.get(FEATURE_SERVER, params={
            "where": where_clause, "returnCountOnly": "true", "f": "json",
        }, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if "error" in resp:
            log.warning("  [%d] count query error: %s", year, resp["error"])
            return None
        total = resp.get("count", 0)
        log.info("  [%d] %d records to fetch", year, total)
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


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    df["crash_date"] = pd.to_datetime(df["Date"], unit="ms", errors="coerce")
    n_bad_dt = df["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable Date dropped", year, n_bad_dt)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    df["fips"] = df["County"].map(county_to_fips)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        unmapped = df.loc[df["fips"].isna(), "County"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped county: %s", year, n_miss, unmapped)
    df = df.dropna(subset=["fips"])

    df["fatals"] = pd.to_numeric(df["NumFatalities"], errors="coerce").fillna(0)
    df["serious_inj"] = pd.to_numeric(df["NumAInjuries"], errors="coerce").fillna(0)

    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(nc_fatals=("fatals", "sum"), nc_serious_inj=("serious_inj", "sum"),
               nc_crashes=("fatals", "count"))
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  nc_fatals=%.0f  nc_serious_inj=%.0f  nc_crashes=%d",
             year, len(agg), agg["nc_fatals"].sum(), agg["nc_serious_inj"].sum(), agg["nc_crashes"].sum())
    return agg


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main ─────────────────────────────────────────────────────────────────────
    log.info("Downloading North Carolina NCDOT crash data (2021–2024) …")
    log.info("Source: %s", FEATURE_SERVER)

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    coverage_rows = []

    for yr in YEARS:
        log.info("=== Year %d ===", yr)
        raw = fetch_year(session, yr)
        coverage_rows.append(validate_source_frame("NC", yr, raw,
            required_columns={"Date", "County", "NumFatalities", "NumAInjuries"},
            date_column="Date", outcome_columns={"NumFatalities", "NumAInjuries"}, date_unit="ms",
            geography_column="County", geography_mapper=county_to_fips,
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(raw, yr)
        if agg is not None:
            parts.append(agg)
        del raw, agg
        gc.collect()
        time.sleep(1.0)

    session.close()
    write_state_manifest_or_raise("NC", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No North Carolina data downloaded — aborting.")
        sys.exit(1)

    nc_panel = pd.concat(parts, ignore_index=True)
    nc_panel["date"] = pd.to_datetime(nc_panel["date"])
    nc_panel = (
        nc_panel.groupby(["fips", "date"])
          .agg(nc_fatals=("nc_fatals", "sum"), nc_serious_inj=("nc_serious_inj", "sum"),
               nc_crashes=("nc_crashes", "sum"))
          .reset_index()
    )

    log.info("")
    log.info("Final North Carolina NCDOT panel:")
    log.info("  Rows          : %d", len(nc_panel))
    log.info("  Counties      : %d", nc_panel["fips"].nunique())
    log.info("  Date range    : %s – %s", nc_panel["date"].min().date(), nc_panel["date"].max().date())
    log.info("  nc_fatals     : %.0f", nc_panel["nc_fatals"].sum())
    log.info("  nc_serious_inj: %.0f", nc_panel["nc_serious_inj"].sum())
    log.info("  nc_crashes    : %d", int(nc_panel["nc_crashes"].sum()))

    nc_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
