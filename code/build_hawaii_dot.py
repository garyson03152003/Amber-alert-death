"""
build_hawaii_dot.py
========================================================
Download Hawaii statewide fatal-crash data from a live ArcGIS FeatureServer
and build a county-day panel of fatalities.

Source: Hawaii FatalCrash FeatureServer (statewide)
URL: https://services.arcgis.com/HQ0xoN0EzDPBOEci/arcgis/rest/services/FatalCrash/FeatureServer/0
Coverage: 2019, 2021-2024 only (5 years). Full 2012-2024 was requested and
is structurally present (~80-115 fatal crashes/year, no volume cliff), but
`Crash_Date` is not a genuine per-crash date for 2012-2018 -- confirmed
directly against the live source, nearly every record in each of those
years collapses onto just 1-3 distinct dates (a bulk-load artifact, not a
real crash-date distribution); 2020 is separately excluded for a smaller
year-tag/date disagreement (a 2020-tagged crash with Crash_Date in early
2021). Only 2019 and 2021-2024 have real per-crash date variance (record
count roughly equals unique-date count) and are kept.
No authentication required.

This source is **fatal-crash-only**: every row is a fatal crash
(`Total_Fatalities` >= 1), with no all-crash denominator and no
serious-injury field -- the mirror image of New York/Delaware's
crashes-only contract. `crashes` and `serious_injury_persons` are therefore
not comparable outcomes; only `person_fatals` is reported.

Key fields (confirmed by probe):
  Crash_Date        — epoch milliseconds (real calendar date)
  Crash_Year         — integer year (agrees with parsed Crash_Date)
  County             — county name string: "Hawaii", "Honolulu", "Kauai",
                        "Maui" (Kalawao, the state's 5th and smallest
                        county, never appears -- population ~90, a
                        plausible structural zero, not a mapping gap)
  Total_Fatalities   — person-level fatality count for that crash
  ObjectId           — system-maintained unique id (CrashId is null for
                        most rows, so ObjectId is used for pagination
                        instead)

Output columns: fips, date, hi_fatals
Output: data/processed/hawaii_dot_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from state_dot_sources import strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("hawaii_dot")

OUT_PATH = DATA_PROC / "hawaii_dot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

BASE = "https://services.arcgis.com/HQ0xoN0EzDPBOEci/arcgis/rest/services/FatalCrash/FeatureServer"
QUERY_URL = f"{BASE}/0/query"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

# 2012-2015, 2017, 2018, 2020 excluded -- see state_dot_sources.py for the
# full investigation: for 2012-2015/2017/2018, Crash_Date is not a real
# per-crash date at all (almost every record in each of those years shares
# the exact same timestamp, a bulk-load artifact); 2020 has a small
# above-threshold share of year-tag/date disagreement. Only 2019 and
# 2021-2024 have genuine per-crash date variance.
YEARS = [2019, 2021, 2022, 2023, 2024]
PAGE_SIZE = 1_000  # server maxRecordCount observed via exceededTransferLimit
FETCH_FAILURES: dict[int, BaseException] = {}
OUT_FIELDS = "Crash_Date,Crash_Year,County,Total_Fatalities"

HI_COUNTY_FIPS = {
    "HAWAII": "15001", "HONOLULU": "15003", "KALAWAO": "15005",
    "KAUAI": "15007", "MAUI": "15009",
}


def county_to_fips(name: str) -> str | None:
    if name is None:
        return None
    return HI_COUNTY_FIPS.get(str(name).strip().upper())


def _count(session: requests.Session, where: str) -> int:
    r = session.get(QUERY_URL, params={"where": where, "returnCountOnly": "true", "f": "json"}, timeout=45)
    r.raise_for_status()
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(f"count query error: {resp['error']}")
    return resp.get("count", 0)


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame:
    where = f"Crash_Year = {year}"
    try:
        return strict_arcgis_dataframe(session, url=QUERY_URL, where=where,
                                        expected_count=_count(session, where),
                                        id_field="ObjectId", out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return pd.DataFrame()


# For 2019/2021/2022, Crash_Date's calendar date is systematically one day
# AFTER the true crash date -- confirmed by matching this source's
# person_fatals against the validated FARS county-day panel: shifting the
# parsed date back by 1 day raised the exact-match rate from 2-9% to
# 84-100% for each of those three years (a clean, near-total flip, not
# noise). 2023-2024 need no shift (0-day match rate is already 91-100%).
# The most likely explanation is a mid-stream change in how the source
# system serializes its date field (e.g. a local-midnight-as-UTC encoding
# used through 2022, replaced by a correctly-localized one from 2023), but
# the exact cause is not confirmable from this API alone -- the per-year
# correction below is applied on direct evidence against FARS, not a guess.
_DATE_SHIFT_DAYS = {2019: -1, 2021: -1, 2022: -1, 2023: 0, 2024: 0}


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    df["crash_date"] = pd.to_datetime(df["Crash_Date"], unit="ms", errors="coerce")
    n_bad_dt = df["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable Crash_Date dropped", year, n_bad_dt)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize() + pd.Timedelta(days=_DATE_SHIFT_DAYS[year])

    df["fips"] = df["County"].map(county_to_fips)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        unmapped = df.loc[df["fips"].isna(), "County"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped county: %s", year, n_miss, unmapped)
    df = df.dropna(subset=["fips"])
    df["Total_Fatalities"] = pd.to_numeric(df["Total_Fatalities"], errors="coerce").fillna(0)

    agg = (
        df.groupby(["fips", "crash_date"])["Total_Fatalities"].sum()
          .reset_index(name="hi_fatals")
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  hi_fatals=%.0f", year, len(agg), agg["hi_fatals"].sum())
    return agg


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main ─────────────────────────────────────────────────────────────────────
    log.info("Downloading Hawaii statewide fatal-crash data (2012–2024) …")
    log.info("Source: %s", QUERY_URL)

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    coverage_rows = []

    for yr in YEARS:
        log.info("=== Year %d ===", yr)
        raw = fetch_year(session, yr)
        coverage_rows.append(validate_source_frame("HI", yr, None if raw.empty else raw,
            required_columns={"Crash_Date", "Crash_Year", "County", "Total_Fatalities"},
            date_column="Crash_Date", outcome_columns={"Total_Fatalities"}, date_unit="ms",
            geography_column="County", geography_mapper=county_to_fips,
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(raw, yr)
        if agg is not None:
            parts.append(agg)
        del raw, agg
        gc.collect()
        time.sleep(0.5)

    session.close()
    write_state_manifest_or_raise("HI", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Hawaii data downloaded — aborting.")
        sys.exit(1)

    hi_panel = pd.concat(parts, ignore_index=True)
    hi_panel["date"] = pd.to_datetime(hi_panel["date"])
    hi_panel = hi_panel.groupby(["fips", "date"], as_index=False)["hi_fatals"].sum()

    log.info("")
    log.info("Final Hawaii panel:")
    log.info("  Rows       : %d", len(hi_panel))
    log.info("  Counties   : %d", hi_panel["fips"].nunique())
    log.info("  Date range : %s – %s", hi_panel["date"].min().date(), hi_panel["date"].max().date())
    log.info("  hi_fatals  : %.0f", hi_panel["hi_fatals"].sum())

    hi_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
