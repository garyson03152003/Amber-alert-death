"""
build_tennessee_tdot.py
=================================================================
Download Tennessee TDOT crash data from the ArcGIS FeatureServer
and build a county-day panel of crashes, fatalities, and injuries.

Source:
  https://services2.arcgis.com/nf3p7v7Zy4fTOh6M/arcgis/rest/services/
  Tennessee_Crashes_JAN_2021_JAN_2025/FeatureServer/0

Key fields:
  NBR_TENN_C  — county name (uppercase, e.g. "SHELBY")
  DATEOFCRAS  — crash date (epoch milliseconds)
  TOTALKILLE  — total killed (fatalities)
  TOTALINJUR  — total injured
  TOTAL_INCA  — incapacitating (serious) injuries
  YEAROFCRAS  — year string (for year-level queries)

Coverage: 2021–2024 (2025 partial)

Output: data/processed/tennessee_tdot_county_day.parquet
  columns: fips, date, tn_crashes, tn_fatals, tn_serious_inj
"""
import gc, sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("tennessee_tdot")

OUT_PATH = DATA_PROC / "tennessee_tdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FS_URL   = ("https://services2.arcgis.com/nf3p7v7Zy4fTOh6M/arcgis/rest/services/"
             "Tennessee_Crashes_JAN_2021_JAN_2025/FeatureServer/0/query")
HEADERS  = {"User-Agent": "amber-research/1.0 (academic)"}
YEARS    = list(range(2021, 2025))   # 2025 is partial; use 2021–2024
PAGE_SIZE = 2000
OUT_FIELDS = "OBJECTID,NBR_TENN_C,CNTY_SEQ,DATEOFCRAS,YEAROFCRAS,TOTALKILLE,TOTALINJUR,TOTAL_INCA"
SLEEP_PAGE = 0.25
SLEEP_YEAR = 3.0

# ── Tennessee county name → FIPS ──────────────────────────────────────────────
TN_COUNTY_FIPS = {
    "ANDERSON": "47001", "BEDFORD": "47003", "BENTON": "47005",
    "BLEDSOE": "47007", "BLOUNT": "47009", "BRADLEY": "47011",
    "CAMPBELL": "47013", "CANNON": "47015", "CARROLL": "47017",
    "CARTER": "47019", "CHEATHAM": "47021", "CHESTER": "47023",
    "CLAIBORNE": "47025", "CLAY": "47027", "COCKE": "47029",
    "COFFEE": "47031", "CROCKETT": "47033", "CUMBERLAND": "47035",
    "DAVIDSON": "47037", "DECATUR": "47039", "DEKALB": "47041",
    "DICKSON": "47043", "DYER": "47045", "FAYETTE": "47047",
    "FENTRESS": "47049", "FRANKLIN": "47051", "GIBSON": "47053",
    "GILES": "47055", "GRAINGER": "47057", "GREENE": "47059",
    "GRUNDY": "47061", "HAMBLEN": "47063", "HAMILTON": "47065",
    "HANCOCK": "47067", "HARDEMAN": "47069", "HARDIN": "47071",
    "HAWKINS": "47073", "HAYWOOD": "47075", "HENDERSON": "47077",
    "HENRY": "47079", "HICKMAN": "47081", "HOUSTON": "47083",
    "HUMPHREYS": "47085", "JACKSON": "47087", "JEFFERSON": "47089",
    "JOHNSON": "47091", "KNOX": "47093", "LAKE": "47095",
    "LAUDERDALE": "47097", "LAWRENCE": "47099", "LEWIS": "47101",
    "LINCOLN": "47103", "LOUDON": "47105", "MCMINN": "47107",
    "MCNAIRY": "47109", "MACON": "47111", "MADISON": "47113",
    "MARION": "47115", "MARSHALL": "47117", "MAURY": "47119",
    "MEIGS": "47121", "MONROE": "47123", "MONTGOMERY": "47125",
    "MOORE": "47127", "MORGAN": "47129", "OBION": "47131",
    "OVERTON": "47133", "PERRY": "47135", "PICKETT": "47137",
    "POLK": "47139", "PUTNAM": "47141", "RHEA": "47143",
    "ROANE": "47145", "ROBERTSON": "47147", "RUTHERFORD": "47149",
    "SCOTT": "47151", "SEQUATCHIE": "47153", "SEVIER": "47155",
    "SHELBY": "47157", "SMITH": "47159", "STEWART": "47161",
    "SULLIVAN": "47163", "SUMNER": "47165", "TIPTON": "47167",
    "TROUSDALE": "47169", "UNICOI": "47171", "UNION": "47173",
    "VAN BUREN": "47175", "WARREN": "47177", "WASHINGTON": "47179",
    "WAYNE": "47181", "WEAKLEY": "47183", "WHITE": "47185",
    "WILLIAMSON": "47187", "WILSON": "47189",
    # Aliases
    "DE KALB": "47041", "DEKALB": "47041",
    "MC MINN": "47107", "MC NAIRY": "47109",
}


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    where = f"YEAROFCRAS='{year}'"

    try:
        r = session.get(FS_URL, params={"where": where, "returnCountOnly": "true", "f": "json"},
                        timeout=30)
        r.raise_for_status()
        total = r.json().get("count", 0)
        log.info("  [%d] total records: %d", year, total)
    except Exception as e:
        log.warning("  [%d] count failed: %s", year, e)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skip", year)
        return None

    parts, offset = [], 0
    while offset < total:
        params = {
            "where": where,
            "outFields": OUT_FIELDS,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        try:
            r = session.get(FS_URL, params=params, timeout=120)
            r.raise_for_status()
            feats = r.json().get("features", [])
        except Exception as e:
            log.warning("  [%d] page offset=%d failed: %s — retry", year, offset, e)
            time.sleep(4)
            try:
                r = session.get(FS_URL, params=params, timeout=120)
                r.raise_for_status()
                feats = r.json().get("features", [])
            except Exception as e2:
                log.warning("  [%d] retry failed: %s — stopping year", year, e2)
                break

        if not feats:
            break
        parts.append(pd.DataFrame([f["attributes"] for f in feats]))
        offset += len(feats)

        if offset % 50_000 == 0 or offset >= total:
            log.info("  [%d] … %d / %d", year, offset, total)
        time.sleep(SLEEP_PAGE)

    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] fetched %d rows", year, len(df))
    return df


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()

    # Parse date (epoch ms)
    df["date"] = pd.to_datetime(df["DATEOFCRAS"], unit="ms", errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # Restrict to target year (service might include edge-months)
    df = df[df["date"].dt.year == year].copy()
    if df.empty:
        log.warning("  [%d] no records after year filter", year)
        return None

    # Map county to FIPS
    df["county_upper"] = df["NBR_TENN_C"].astype(str).str.strip().str.upper()
    df["fips"] = df["county_upper"].map(TN_COUNTY_FIPS)
    unmatched = df["fips"].isna().sum()
    if unmatched:
        bad = df.loc[df["fips"].isna(), "county_upper"].value_counts().head(5)
        log.warning("  [%d] %d rows unmatched county: %s", year, unmatched, bad.to_dict())
    df = df.dropna(subset=["fips"])

    # Outcomes
    for col in ["TOTALKILLE", "TOTALINJUR", "TOTAL_INCA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Aggregate to county-day
    agg = (
        df.groupby(["fips", "date"])
        .agg(
            tn_crashes     =("OBJECTID",    "count"),
            tn_fatals      =("TOTALKILLE",  "sum"),
            tn_serious_inj =("TOTAL_INCA",  "sum"),
        )
        .reset_index()
    )
    log.info(
        "  [%d] → %d county-days  crashes=%.0f  fatals=%.0f  serious_inj=%.0f",
        year, len(agg), agg["tn_crashes"].sum(),
        agg["tn_fatals"].sum(), agg["tn_serious_inj"].sum()
    )
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────
log.info("Downloading Tennessee TDOT crash data (2021–2024) …")
session = requests.Session()
session.headers.update(HEADERS)
parts = []

for yr in YEARS:
    log.info("Year %d …", yr)
    raw = fetch_year(session, yr)
    agg = process_year(raw, yr)
    if agg is not None:
        parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(SLEEP_YEAR)

session.close()

if not parts:
    log.error("No Tennessee data downloaded.")
    sys.exit(1)

panel = pd.concat(parts, ignore_index=True)
panel = (
    panel.groupby(["fips", "date"])
    .agg(tn_crashes=("tn_crashes","sum"), tn_fatals=("tn_fatals","sum"),
         tn_serious_inj=("tn_serious_inj","sum"))
    .reset_index()
)
panel = panel.sort_values(["fips", "date"]).reset_index(drop=True)
panel["date"] = pd.to_datetime(panel["date"])

log.info("\nFinal Tennessee TDOT panel:")
log.info("  Rows: %d  Counties: %d  %s – %s",
         len(panel), panel["fips"].nunique(),
         panel["date"].min().date(), panel["date"].max().date())
log.info("  tn_crashes=%.0f  tn_fatals=%.0f  tn_serious_inj=%.0f",
         panel["tn_crashes"].sum(), panel["tn_fatals"].sum(),
         panel["tn_serious_inj"].sum())

panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
