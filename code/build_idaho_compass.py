"""
build_idaho_compass.py
========================================================
Download crash data for the 2-county (Ada, Canyon) Boise-metro region from
COMPASS's (the region's MPO) ArcGIS FeatureServer, republishing ITD source
crash data, and build a county-day panel.

Multi-county (2-county) addition: no Idaho statewide crash-level feed was
found. Verified genuinely county-wide (not a single-city-PD jurisdiction)
by checking that both counties' records span all their member cities and
multiple reporting agencies -- Ada: Boise, Meridian, Eagle, Garden City,
Kuna, Star, via Boise PD, Ada Co Sheriff, Meridian PD, Idaho State Police,
and Garden City PD; Canyon: Nampa, Caldwell, Middleton, Parma, and 7 more.

Source: COMPASS (Community Planning Association of Southwest Idaho) /
ITD CrashData
URL: https://swidrdc.org/arcgis/rest/services/COMPASSData/CrashData/FeatureServer/0
Coverage: 2013-2024 requested (source has 2008-2024; restricted to this
project's usual window).
No authentication required.

Unlike several other sub-state additions this session, this source has a
genuine per-crash fatality COUNT (not a severity flag) and a KABCO
severity classification enabling a serious-injury proxy. `person_fatals`
matched FARS almost exactly for 5 sample years checked directly against
the live source before building (ratio 0.95-1.00).

Key fields (confirmed by probe):
  accident_date  — epoch milliseconds (real calendar date)
  countyname     — "Ada" or "Canyon"
  fatalities     — person-level fatality count per crash
  injuries       — person-level injury count per crash
  severity       — "Fatal Accident" / "A/B/C Injury Accident" / "Property
                    Dmg Report" (KABCO-style crash-level classification)

Output columns: fips, date, idc_crashes, idc_fatals, idc_serious_inj
Output: data/processed/idaho_compass_county_day.parquet
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
log = get_logger("idaho_compass")

OUT_PATH = DATA_PROC / "idaho_compass_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

QUERY_URL = "https://swidrdc.org/arcgis/rest/services/COMPASSData/CrashData/FeatureServer/0/query"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2013, 2025))
PAGE_SIZE = 2_000
FETCH_FAILURES: dict[int, BaseException] = {}
OUT_FIELDS = "objectid,accident_date,countyname,fatalities,injuries,severity"

ID_COUNTY_FIPS = {"ADA": "16001", "CANYON": "16027"}


def county_to_fips(name: str) -> str | None:
    if name is None:
        return None
    return ID_COUNTY_FIPS.get(str(name).strip().upper())


def _count(session: requests.Session, where: str) -> int:
    r = session.get(QUERY_URL, params={"where": where, "returnCountOnly": "true", "f": "json"}, timeout=45)
    r.raise_for_status()
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(f"count query error: {resp['error']}")
    return resp.get("count", 0)


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame:
    where = f"year = {year}"
    try:
        return strict_arcgis_dataframe(session, url=QUERY_URL, where=where,
                                        expected_count=_count(session, where),
                                        id_field="objectid", out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return pd.DataFrame()


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    df["crash_date"] = pd.to_datetime(df["accident_date"], unit="ms", errors="coerce")
    n_bad_dt = df["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable accident_date dropped", year, n_bad_dt)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    df["fips"] = df["countyname"].map(county_to_fips)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        unmapped = df.loc[df["fips"].isna(), "countyname"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped county: %s", year, n_miss, unmapped)
    df = df.dropna(subset=["fips"])

    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    df["injuries"] = pd.to_numeric(df["injuries"], errors="coerce").fillna(0)
    is_a_severity = df["severity"].astype(str).str.strip().eq("A Injury Accident")
    df["a_injuries"] = df["injuries"].where(is_a_severity, 0)

    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(idc_fatals=("fatalities", "sum"), idc_serious_inj=("a_injuries", "sum"),
               idc_crashes=("objectid", "nunique"))
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  idc_crashes=%d  idc_fatals=%.0f  idc_serious_inj=%.0f",
             year, len(agg), agg["idc_crashes"].sum(), agg["idc_fatals"].sum(), agg["idc_serious_inj"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Idaho COMPASS crash data (2013–2024) …")
log.info("Source: %s", QUERY_URL)

session = requests.Session()
session.headers.update(HEADERS)
parts = []
coverage_rows = []

for yr in YEARS:
    log.info("=== Year %d ===", yr)
    raw = fetch_year(session, yr)
    coverage_rows.append(validate_source_frame("IDCOMPASS", yr, None if raw.empty else raw,
        required_columns={"accident_date", "countyname", "fatalities", "injuries", "severity"},
        date_column="accident_date", outcome_columns={"fatalities", "injuries"}, date_unit="ms",
        geography_column="countyname", geography_mapper=county_to_fips,
        terminal_error=FETCH_FAILURES.get(yr)))
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(0.5)

session.close()
write_state_manifest_or_raise("IDCOMPASS", coverage_rows, output_dir=DATA_PROC / "coverage")

if not parts:
    log.error("No Idaho COMPASS data downloaded — aborting.")
    sys.exit(1)

panel = pd.concat(parts, ignore_index=True)
panel["date"] = pd.to_datetime(panel["date"])
panel = (
    panel.groupby(["fips", "date"])
      .agg(idc_crashes=("idc_crashes", "sum"), idc_fatals=("idc_fatals", "sum"),
           idc_serious_inj=("idc_serious_inj", "sum"))
      .reset_index()
)

log.info("")
log.info("Final Idaho COMPASS panel:")
log.info("  Rows            : %d", len(panel))
log.info("  Counties        : %d", panel["fips"].nunique())
log.info("  Date range      : %s – %s", panel["date"].min().date(), panel["date"].max().date())
log.info("  idc_crashes     : %d", int(panel["idc_crashes"].sum()))
log.info("  idc_fatals      : %.0f", panel["idc_fatals"].sum())
log.info("  idc_serious_inj : %.0f", panel["idc_serious_inj"].sum())

panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
