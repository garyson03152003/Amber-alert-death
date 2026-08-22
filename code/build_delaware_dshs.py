"""
build_delaware_dshs.py
========================================================
Download Delaware statewide crash data from DSHS's "Public Crash Data"
Socrata dataset (data.delaware.gov) and build a county-day panel of crash
counts.

Source: Delaware DSHS Public Crash Data
URL: https://data.delaware.gov/resource/827n-m6xc.json
Coverage: 2013-2024 requested (source has 2009-2026; this is the current,
actively-maintained statewide crash release, launched publicly in 2023 after
DSHS became the sole owner of Delaware crash data in 2019 -- see
code/build_delaware_deldot.py for the older, abandoned DelDOT ArcGIS layer
this supersedes).
No authentication required.

Delaware has only 3 counties. crash_class_desc includes a "Fatality Crash"
category but there is no person-level fatality/injury COUNT field -- only
categorical crash-severity flags -- so person_fatals and
serious_injury_persons are not comparable outcomes (same situation as New
York's crash-only contract); only crashes is reported.

Key fields (confirmed by probe):
  crash_datetime — ISO 8601 timestamp
  county         — single-letter county code: K=Kent, N=New Castle, S=Sussex
  year           — string year

Output columns: fips, date, de_crashes
Output: data/processed/delaware_deldot_county_day.parquet (same output path
as the retired DelDOT-source builder; this is the successor, not an
additional source).
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
log = get_logger("delaware_dshs")

OUT_PATH = DATA_PROC / "delaware_deldot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://data.delaware.gov/resource/827n-m6xc.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2013, 2025))
PAGE_LIMIT = 50_000
FETCH_FAILURES: dict[int, BaseException] = {}

DE_COUNTY_FIPS = {"K": "10001", "N": "10003", "S": "10005"}


def county_to_fips(code: str) -> str | None:
    if code is None:
        return None
    return DE_COUNTY_FIPS.get(str(code).strip().upper())


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame:
    where = f"year = '{year}'"
    try:
        return strict_socrata_dataframe(
            session, url=BASE_URL, where=where, id_field=":id", page_size=PAGE_LIMIT
        )
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict Socrata pagination failed: %s", year, exc)
        return pd.DataFrame()


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    # crash_datetime is ISO 8601 with a "Z" (UTC) suffix; parse then drop the
    # timezone so this matches every other state's naive datetime64 columns
    # (the panel's calendar-date grain makes the UTC/local distinction
    # immaterial here -- this dataset has no separate local-time field).
    df["crash_date"] = pd.to_datetime(df["crash_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    n_bad_dt = df["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable crash_datetime dropped", year, n_bad_dt)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    df["fips"] = df["county"].map(county_to_fips)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        unmapped = df.loc[df["fips"].isna(), "county"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped county: %s", year, n_miss, unmapped)
    df = df.dropna(subset=["fips"])

    agg = (
        df.groupby(["fips", "crash_date"])
          .size()
          .reset_index(name="de_crashes")
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  de_crashes=%d", year, len(agg), agg["de_crashes"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Delaware DSHS crash data (2013–2024) …")
log.info("Source: %s", BASE_URL)

session = requests.Session()
session.headers.update(HEADERS)
parts = []
coverage_rows = []

for yr in YEARS:
    log.info("=== Year %d ===", yr)
    raw = fetch_year(session, yr)
    coverage_rows.append(validate_source_frame("DE", yr, None if raw.empty else raw,
        required_columns={"crash_datetime", "county", "year"},
        date_column="crash_datetime", outcome_columns=set(),
        geography_column="county", geography_mapper=county_to_fips,
        terminal_error=FETCH_FAILURES.get(yr)))
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(0.5)

session.close()
write_state_manifest_or_raise("DE", coverage_rows, output_dir=DATA_PROC / "coverage")

if not parts:
    log.error("No Delaware data downloaded — aborting.")
    sys.exit(1)

de_panel = pd.concat(parts, ignore_index=True)
de_panel["date"] = pd.to_datetime(de_panel["date"])
de_panel = de_panel.groupby(["fips", "date"], as_index=False)["de_crashes"].sum()

log.info("")
log.info("Final Delaware DSHS panel:")
log.info("  Rows       : %d", len(de_panel))
log.info("  Counties   : %d", de_panel["fips"].nunique())
log.info("  Date range : %s – %s", de_panel["date"].min().date(), de_panel["date"].max().date())
log.info("  de_crashes : %d", int(de_panel["de_crashes"].sum()))

de_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
