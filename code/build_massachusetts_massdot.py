"""
build_massachusetts_massdot.py
========================================================
Download Massachusetts MassDOT IMPACT crash data and build a county-day
panel of fatalities and serious injuries.

Source: MassDOT GIS ArcGIS server (gis.massdot.state.ma.us)
  2013–2019 (FeatureServer): CrashClosedYear/CrashClosedYear{yr}/FeatureServer/0
  2020 (MapServer):          Dashboard/CrashClosedYear2020_Views/MapServer/4
No authentication required.

Coverage: 2013–2020 (~100k–141k crashes/year; 14 counties)
maxRecordCount: 2000 → ~50–70 pages per year

Key fields:
  CRASH_DATE      — epoch milliseconds (2018–2020 layers only)
  CRASH_DATETIME  — epoch milliseconds (2013–2017 layers only; same semantic)
  CNTY_NAME           — county name (e.g. "MIDDLESEX", "SUFFOLK")
  NUMB_FATAL_INJR     — number of fatalities per crash
  NUMB_NONFATAL_INJR  — number of non-fatal injuries per crash (all levels)
  MAX_INJR_SVRTY_CL   — most severe injury in crash:
                          "Fatal injury (K)"
                          "Non-fatal injury - Incapacitating"   ← KABCO-A
                          "Non-fatal injury - Non-incapacitating"
                          "Non-fatal injury - Possible"
                          "No injury" / "No Apparent Injury (O)"

NOTE on date field schema versioning:
  2013–2017 FeatureServer layers use CRASH_DATETIME (not CRASH_DATE).
  Requesting CRASH_DATE on those layers causes ArcGIS to return an embedded
  JSON error {"code": 400, "message": "Unable to complete operation"} with
  HTTP 200, yielding 0 features.  Per-year outFields are used to avoid this.

Serious injury proxy: sum(NUMB_NONFATAL_INJR) for crashes where
  MAX_INJR_SVRTY_CL contains "Incapacitating" (= KABCO-A)

Output: data/processed/massachusetts_massdot_county_day.parquet
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
log = get_logger("massachusetts_massdot")

OUT_PATH = DATA_PROC / "massachusetts_massdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://gis.massdot.state.ma.us/arcgis/rest/services"
HEADERS  = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

# Per-year service URLs and layer IDs
YEAR_SERVICES = {
    **{yr: (f"CrashClosedYear/CrashClosedYear{yr}/FeatureServer/0", "FeatureServer")
       for yr in range(2013, 2020)},
    2020: ("Dashboard/CrashClosedYear2020_Views/MapServer/4", "MapServer"),
}
YEARS = sorted(YEAR_SERVICES.keys())   # 2013–2020

COMMON_FIELDS = "CNTY_NAME,NUMB_FATAL_INJR,NUMB_NONFATAL_INJR,MAX_INJR_SVRTY_CL"
# 2013–2017 layers use CRASH_DATETIME; 2018–2020 layers use CRASH_DATE
YEAR_DATE_FIELD = {yr: "CRASH_DATETIME" for yr in range(2013, 2018)}
YEAR_DATE_FIELD.update({yr: "CRASH_DATE" for yr in range(2018, 2021)})

def out_fields(year: int) -> str:
    return f"{YEAR_DATE_FIELD[year]},{COMMON_FIELDS}"

PAGE_SIZE = 2000   # server maxRecordCount
FETCH_FAILURES: dict[int, BaseException] = {}

# ── Massachusetts county FIPS mapping ────────────────────────────────────────
MA_COUNTY_FIPS = {
    "BARNSTABLE":  "25001",
    "BERKSHIRE":   "25003",
    "BRISTOL":     "25005",
    "DUKES":       "25007",   # Martha's Vineyard
    "ESSEX":       "25009",
    "FRANKLIN":    "25011",
    "HAMPDEN":     "25013",
    "HAMPSHIRE":   "25015",
    "MIDDLESEX":   "25017",
    "NANTUCKET":   "25019",
    "NORFOLK":     "25021",
    "PLYMOUTH":    "25023",
    "SUFFOLK":     "25025",
    "WORCESTER":   "25027",
}


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    """Paginate all crash records for one year from the appropriate service."""
    svc_path, svc_type = YEAR_SERVICES[year]
    query_url = f"{BASE_URL}/{svc_path}/query"
    fields = out_fields(year)
    date_field = YEAR_DATE_FIELD[year]

    # Count
    try:
        r = session.get(query_url, params={
            "where": "1=1", "returnCountOnly": "true", "f": "json"
        }, timeout=45)
        r.raise_for_status()
        total = r.json().get("count", 0)
        log.info("  [%d] %d records  (%s, date_field=%s)", year, total, svc_type, date_field)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.warning("  [%d] count failed: %s", year, exc)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skipping", year)
        return None

    try:
        return strict_arcgis_dataframe(session, url=query_url, where="1=1",
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields=fields, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return None

    # Paginate
    parts = []
    offset = 0
    page = {}  # initialise so the features fallback below is safe
    while offset < total:
        for attempt in range(3):
            try:
                r = session.get(query_url, params={
                    "where":             "1=1",
                    "outFields":         fields,
                    "resultOffset":      offset,
                    "resultRecordCount": PAGE_SIZE,
                    "f":                 "json",
                }, timeout=60)
                r.raise_for_status()
                page = r.json()
                # ArcGIS can return HTTP 200 with embedded error
                if "error" in page:
                    raise ValueError(f"ArcGIS error: {page['error']}")
                break
            except Exception as exc:
                wait = 3 * (attempt + 1)
                log.warning("  [%d] offset=%d attempt %d failed: %s; retry in %ds",
                            year, offset, attempt + 1, exc, wait)
                if attempt < 2:
                    time.sleep(wait)
                else:
                    log.error("  [%d] gave up at offset=%d", year, offset)
                    offset = total  # exit outer loop
                    page = {}
                    break

        features = page.get("features", [])
        if not features:
            break
        rows = [f["attributes"] for f in features]
        parts.append(pd.DataFrame(rows))
        offset += len(rows)

        if offset % 20000 == 0 or len(rows) < PAGE_SIZE:
            log.info("  [%d] fetched %d/%d", year, offset, total)

        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.2)

    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] raw total: %d rows", year, len(df))
    return df


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Convert raw crash records to county-day panel."""
    if df is None or df.empty:
        return None
    df = df.copy()

    # ── Date ─────────────────────────────────────────────────────────────────
    # 2013-2017 layers → CRASH_DATETIME; 2018-2020 layers → CRASH_DATE
    _date_col = "CRASH_DATE" if "CRASH_DATE" in df.columns else "CRASH_DATETIME"
    df["crash_date"] = pd.to_datetime(df[_date_col], unit="ms", errors="coerce")
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    # ── County → FIPS ────────────────────────────────────────────────────────
    df["county_upper"] = df["CNTY_NAME"].astype(str).str.strip().str.upper()
    df["fips"] = df["county_upper"].map(MA_COUNTY_FIPS)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        log.warning("  [%d] %d rows unmapped county: %s", year, n_miss,
                    df.loc[df["fips"].isna(), "county_upper"].value_counts().head(5).to_dict())
    df = df.dropna(subset=["fips"])

    # ── Severity ─────────────────────────────────────────────────────────────
    df["fatals"]      = pd.to_numeric(df["NUMB_FATAL_INJR"],   errors="coerce").fillna(0)
    df["all_injured"] = pd.to_numeric(df["NUMB_NONFATAL_INJR"], errors="coerce").fillna(0)

    # Serious injuries = non-fatal injured in crashes where max severity is "Incapacitating"
    sev = df["MAX_INJR_SVRTY_CL"].astype(str).str.strip()
    is_serious = sev.str.contains("Incapacitating", case=False, na=False)
    df["serious_inj"] = df["all_injured"].where(is_serious, 0)

    # ── Aggregate to county-day ───────────────────────────────────────────────
    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(
              ma_fatals     =("fatals",      "sum"),
              ma_injury_proxy=("serious_inj", "sum"),
              ma_crashes    =("fatals",      "count"),
          )
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  ma_fatals=%.0f  ma_serious_inj=%.0f",
             year, len(agg), agg["ma_fatals"].sum(), agg["ma_injury_proxy"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Massachusetts MassDOT crash data (2013–2020) …")

session = requests.Session()
session.headers.update(HEADERS)
parts = []
coverage_rows = []

for yr in YEARS:
    log.info("Year %d …", yr)
    raw = fetch_year(session, yr)
    coverage_rows.append(validate_source_frame("MA", yr, raw,
        required_columns={YEAR_DATE_FIELD[yr], "CNTY_NAME", "NUMB_FATAL_INJR", "NUMB_NONFATAL_INJR", "MAX_INJR_SVRTY_CL"},
        date_column=YEAR_DATE_FIELD[yr], outcome_columns={"NUMB_FATAL_INJR", "NUMB_NONFATAL_INJR"}, date_unit="ms",
        geography_column="CNTY_NAME", geography_mapper=lambda value: MA_COUNTY_FIPS.get(str(value).strip().upper()),
        terminal_error=FETCH_FAILURES.get(yr)))
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    time.sleep(2.0)
    gc.collect()

session.close()
write_state_manifest_or_raise("MA", coverage_rows, output_dir=DATA_PROC / "coverage")

if not parts:
    log.error("No Massachusetts data downloaded.")
    sys.exit(1)

ma_panel = pd.concat(parts, ignore_index=True)
ma_panel["date"] = pd.to_datetime(ma_panel["date"])

ma_panel = (
    ma_panel.groupby(["fips", "date"])
      .agg(
          ma_fatals     =("ma_fatals",      "sum"),
          ma_injury_proxy=("ma_injury_proxy", "sum"),
          ma_crashes    =("ma_crashes",     "sum"),
      )
      .reset_index()
)

log.info("\nFinal Massachusetts panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(ma_panel), ma_panel["fips"].nunique(),
         ma_panel["date"].min().date(), ma_panel["date"].max().date())
ma_panel["ma_serious_inj"] = np.nan
log.info("  Total ma_fatals: %.0f  Total ma_injury_proxy: %.0f",
         ma_panel["ma_fatals"].sum(), ma_panel["ma_injury_proxy"].sum())

ma_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
