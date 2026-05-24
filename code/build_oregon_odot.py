"""
build_oregon_odot.py
========================================================
Download Oregon DOT crash data from the ODOT ArcGIS MapServer and
build a county-day panel of fatalities and serious injuries.

Source: Oregon DOT OTSDE_Crash MapServer
URL: https://gis.odot.state.or.us/arcgis1006/rest/services/agol/OTSDE_Crash/MapServer/0
Coverage: 2019–2024 (273,810 records total)
No authentication required.

Key fields (confirmed by probe):
  CRASH_DT         — epoch milliseconds (e.g. 1561032000000)
  CNTY_NM          — county name, title-case (e.g. "Baker", "Multnomah")
  CRASH_YR_NO      — year as string (e.g. "2019")
  KABCO            — severity code: K/A/B/C/O
  TOT_FATAL_CNT    — fatality count per crash
  TOT_INJ_LVL_A_CNT — count of serious-injury (level A) persons per crash

Output columns:
  fips, date, or_fatals, or_serious_inj, or_crashes

Output: data/processed/oregon_odot_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("oregon_odot")

OUT_PATH = DATA_PROC / "oregon_odot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

MAP_SERVER = (
    "https://gis.odot.state.or.us/arcgis1006/rest/services/agol/OTSDE_Crash/MapServer/0/query"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS      = list(range(2019, 2025))   # 2019–2024 (data starts 2019)
PAGE_SIZE  = 10_000                    # server maxRecordCount = 10,000
OUT_FIELDS = "CRASH_DT,CNTY_NM,CRASH_YR_NO,KABCO,TOT_FATAL_CNT,TOT_INJ_LVL_A_CNT"

# ── Oregon county FIPS mapping (title-case as returned by server) ─────────────
# 36 counties, all FIPS odd-numbered (41001–41071)
OR_COUNTY_FIPS = {
    "Baker":      "41001",
    "Benton":     "41003",
    "Clackamas":  "41005",
    "Clatsop":    "41007",
    "Columbia":   "41009",
    "Coos":       "41011",
    "Crook":      "41013",
    "Curry":      "41015",
    "Deschutes":  "41017",
    "Douglas":    "41019",
    "Gilliam":    "41021",
    "Grant":      "41023",
    "Harney":     "41025",
    "Hood River": "41027",
    "Jackson":    "41029",
    "Jefferson":  "41031",
    "Josephine":  "41033",
    "Klamath":    "41035",
    "Lake":       "41037",
    "Lane":       "41039",
    "Lincoln":    "41041",
    "Linn":       "41043",
    "Malheur":    "41045",
    "Marion":     "41047",
    "Morrow":     "41049",
    "Multnomah":  "41051",
    "Polk":       "41053",
    "Sherman":    "41055",
    "Tillamook":  "41057",
    "Umatilla":   "41059",
    "Union":      "41061",
    "Wallowa":    "41063",
    "Wasco":      "41065",
    "Washington": "41067",
    "Wheeler":    "41069",
    "Yamhill":    "41071",
}
# Also handle common alternate capitalizations / spacing from the server
OR_COUNTY_FIPS_UPPER = {k.upper(): v for k, v in OR_COUNTY_FIPS.items()}


def county_to_fips(name: str) -> str | None:
    """Try title-case lookup first, then upper-case fallback."""
    if name is None:
        return None
    stripped = str(name).strip()
    fips = OR_COUNTY_FIPS.get(stripped)
    if fips is None:
        fips = OR_COUNTY_FIPS_UPPER.get(stripped.upper())
    return fips


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    """Download all crash records for one year with offset-based pagination."""
    where_clause = f"CRASH_YR_NO = '{year}'"

    # ── Count ─────────────────────────────────────────────────────────────────
    try:
        r = session.get(MAP_SERVER, params={
            "where": where_clause,
            "returnCountOnly": "true",
            "f": "json",
        }, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if "error" in resp:
            log.warning("  [%d] count query error: %s", year, resp["error"])
            return None
        total = resp.get("count", 0)
        log.info("  [%d] %d records to fetch", year, total)
    except Exception as exc:
        log.warning("  [%d] count query failed: %s", year, exc)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skipping", year)
        return None

    # ── Paginate ──────────────────────────────────────────────────────────────
    parts = []
    offset = 0
    page_num = 0
    while offset < total:
        page_num += 1
        try:
            r = session.get(MAP_SERVER, params={
                "where": where_clause,
                "outFields": OUT_FIELDS,
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "json",
            }, timeout=120)
            r.raise_for_status()
            page = r.json()
        except Exception as exc:
            log.warning("  [%d] page %d (offset=%d) failed: %s", year, page_num, offset, exc)
            break

        # Check for embedded ArcGIS JSON errors
        if "error" in page:
            log.warning("  [%d] page %d ArcGIS error: %s", year, page_num, page["error"])
            break

        features = page.get("features", [])
        if not features:
            log.info("  [%d] page %d returned 0 features — done", year, page_num)
            break

        rows = [f["attributes"] for f in features]
        parts.append(pd.DataFrame(rows))
        offset += len(rows)
        log.info("  [%d] page %d → %d/%d fetched", year, page_num, offset, total)

        if len(rows) < PAGE_SIZE:
            break   # last page
        time.sleep(0.2)

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Aggregate raw crash records into a county-day panel."""
    if df is None or df.empty:
        return None
    df = df.copy()

    # ── Date ──────────────────────────────────────────────────────────────────
    # CRASH_DT is epoch milliseconds (confirmed by probe: 1561032000000 → 2019-06-20)
    df["crash_date"] = pd.to_datetime(df["CRASH_DT"], unit="ms", errors="coerce")
    n_bad_dt = df["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable CRASH_DT dropped", year, n_bad_dt)
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()   # floor to midnight

    # ── County → FIPS ─────────────────────────────────────────────────────────
    df["fips"] = df["CNTY_NM"].map(county_to_fips)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        unmapped = df.loc[df["fips"].isna(), "CNTY_NM"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped county: %s", year, n_miss, unmapped)
    df = df.dropna(subset=["fips"])

    # ── Severity ──────────────────────────────────────────────────────────────
    df["fatals"]      = pd.to_numeric(df["TOT_FATAL_CNT"],     errors="coerce").fillna(0)
    df["serious_inj"] = pd.to_numeric(df["TOT_INJ_LVL_A_CNT"], errors="coerce").fillna(0)

    # ── Aggregate to county-day ────────────────────────────────────────────────
    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(
              or_fatals     =("fatals",      "sum"),
              or_serious_inj=("serious_inj", "sum"),
              or_crashes    =("fatals",      "count"),
          )
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  or_fatals=%.0f  or_serious_inj=%.0f  or_crashes=%d",
             year, len(agg),
             agg["or_fatals"].sum(), agg["or_serious_inj"].sum(), agg["or_crashes"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Oregon ODOT crash data (2019–2024) …")
log.info("Source: %s", MAP_SERVER)

session = requests.Session()
session.headers.update(HEADERS)
parts = []

for yr in YEARS:
    log.info("=== Year %d ===", yr)
    raw = fetch_year(session, yr)
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(1.0)   # be polite between years

session.close()

if not parts:
    log.error("No Oregon data downloaded — aborting.")
    sys.exit(1)

# ── Combine and final dedup ───────────────────────────────────────────────────
or_panel = pd.concat(parts, ignore_index=True)
or_panel["date"] = pd.to_datetime(or_panel["date"])

or_panel = (
    or_panel.groupby(["fips", "date"])
      .agg(
          or_fatals     =("or_fatals",      "sum"),
          or_serious_inj=("or_serious_inj", "sum"),
          or_crashes    =("or_crashes",     "sum"),
      )
      .reset_index()
)

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("")
log.info("Final Oregon ODOT panel:")
log.info("  Rows         : %d", len(or_panel))
log.info("  Counties     : %d", or_panel["fips"].nunique())
log.info("  Date range   : %s – %s",
         or_panel["date"].min().date(), or_panel["date"].max().date())
log.info("  or_fatals    : %.0f", or_panel["or_fatals"].sum())
log.info("  or_serious_inj: %.0f", or_panel["or_serious_inj"].sum())
log.info("  or_crashes   : %d", int(or_panel["or_crashes"].sum()))

or_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
