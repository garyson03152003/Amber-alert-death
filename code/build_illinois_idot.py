"""
build_illinois_idot.py
========================================================
Download Illinois IDOT crash data from the ArcGIS Open Data Hub
and build a county-day panel of serious injuries and fatalities.

Source: https://gis-idot.opendata.arcgis.com/
No authentication required; one dataset per year (2016–2024).

Illinois is the 9th most-alerted state (1,608 alerts in our sample).
The IDOT data uses KABCO severity with individual injury count columns:
  TotalFatals             ← K (Fatal)
  AInjuries               ← A (Suspected Serious Injury)
  BInjuries               ← B (Suspected Minor Injury)
  CInjuries               ← C (Possible Injury)
  TotalInjured            ← All injury types combined

County is coded as CountyCode (1–102, matching IL county FIPS suffix).
FIPS = "17" + str(CountyCode * 2 - 1).zfill(3)

Data access: opendata download API issues a pre-signed Azure CSV URL.
Base: https://gis-idot.opendata.arcgis.com/api/download/v1/items/{item_id}/csv?layers=0
Fallback: services2.arcgis.com FeatureServer pagination.

Output: data/processed/illinois_idot_county_day.parquet
"""
import sys, warnings, io, gc, time, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request, urllib.parse
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from state_dot_sources import strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise

warnings.filterwarnings("ignore")
log = get_logger("illinois_idot")

OUT_PATH = DATA_PROC / "illinois_idot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

# ── Item IDs and FeatureServer URLs (from gis-idot.opendata.arcgis.com DCAT catalog) ──
IDOT_ITEMS = {
    2016: {"item_id": "583e29335ef348c5990ff63dd0acd153",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/Crashes_2016_SDMExtract/FeatureServer/0"},
    2017: {"item_id": "3291ad89f44448e0be3b26e7b05ea90f",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/Crashes_2017_SDMExtract/FeatureServer/0"},
    2018: {"item_id": "5cb42e3eb58c4a2b8066ae1e374c24fc",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/Crashes_2018_SDMExtract/FeatureServer/0"},
    2019: {"item_id": "c47903b664164b719cce04a2e7584dac",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/Crashes2019/FeatureServer/0"},
    2020: {"item_id": "c286e23e9bf44af397a5da24aaeff8f8",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/Crashes_2020/FeatureServer/0"},
    2021: {"item_id": "bc11eb27849249c89868d6b4cd178613",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/CRASHES2021/FeatureServer/0"},
    2022: {"item_id": "f3a4623c8d14486a9947d29a966bbf9d",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/CRASHES_2022/FeatureServer/0"},
    2023: {"item_id": "ae1333c03cca42c8ae2014bf74666f15",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/CRASHES_2023/FeatureServer/0"},
    2024: {"item_id": "e765e485839f4573b882b06ad84376c9",
           "fs_url": "https://services2.arcgis.com/aIrBD8yn1TDTEXoz/arcgis/rest/services/CRASHES__2024/FeatureServer/0"},
}

# Illinois county FIPS (state 17, counties 001–203, every other)
IL_COUNTY_FIPS_BY_NAME = {
    "ADAMS": "17001", "ALEXANDER": "17003", "BOND": "17005", "BOONE": "17007",
    "BROWN": "17009", "BUREAU": "17011", "CALHOUN": "17013", "CARROLL": "17015",
    "CASS": "17017", "CHAMPAIGN": "17019", "CHRISTIAN": "17021", "CLARK": "17023",
    "CLAY": "17025", "CLINTON": "17027", "COLES": "17029", "COOK": "17031",
    "CRAWFORD": "17033", "CUMBERLAND": "17035", "DEKALB": "17037", "DE WITT": "17039",
    "DEWITT": "17039", "DOUGLAS": "17041", "DUPAGE": "17043", "EDGAR": "17045",
    "EDWARDS": "17047", "EFFINGHAM": "17049", "FAYETTE": "17051", "FORD": "17053",
    "FRANKLIN": "17055", "FULTON": "17057", "GALLATIN": "17059", "GREENE": "17061",
    "GRUNDY": "17063", "HAMILTON": "17065", "HANCOCK": "17067", "HARDIN": "17069",
    "HENDERSON": "17071", "HENRY": "17073", "IROQUOIS": "17075", "JACKSON": "17077",
    "JASPER": "17079", "JEFFERSON": "17081", "JERSEY": "17083", "JO DAVIESS": "17085",
    "JOHNSON": "17087", "KANE": "17089", "KANKAKEE": "17091", "KENDALL": "17093",
    "KNOX": "17095", "LAKE": "17097", "LASALLE": "17099", "LAWRENCE": "17101",
    "LEE": "17103", "LIVINGSTON": "17105", "LOGAN": "17107", "MCDONOUGH": "17109",
    "MCHENRY": "17111", "MCLEAN": "17113", "MACON": "17115", "MACOUPIN": "17117",
    "MADISON": "17119", "MARION": "17121", "MARSHALL": "17123", "MASON": "17125",
    "MASSAC": "17127", "MENARD": "17129", "MERCER": "17131", "MONROE": "17133",
    "MONTGOMERY": "17135", "MORGAN": "17137", "MOULTRIE": "17139", "OGLE": "17141",
    "PEORIA": "17143", "PERRY": "17145", "PIATT": "17147", "PIKE": "17149",
    "POPE": "17151", "PULASKI": "17153", "PUTNAM": "17155", "RANDOLPH": "17157",
    "RICHLAND": "17159", "ROCK ISLAND": "17161", "ST. CLAIR": "17163",
    "SALINE": "17165", "SANGAMON": "17167", "SCHUYLER": "17169", "SCOTT": "17171",
    "SHELBY": "17173", "STARK": "17175", "STEPHENSON": "17177", "TAZEWELL": "17179",
    "UNION": "17181", "VERMILION": "17183", "WABASH": "17185", "WARREN": "17187",
    "WASHINGTON": "17189", "WAYNE": "17191", "WHITE": "17193", "WHITESIDE": "17195",
    "WILL": "17197", "WILLIAMSON": "17199", "WINNEBAGO": "17201", "WOODFORD": "17203",
}
VALID_IL_FIPS = set(IL_COUNTY_FIPS_BY_NAME.values())
FETCH_FAILURES: dict[int, BaseException] = {}

# Each spelling below is accepted by ``process_idot_df``.  The source
# validation call uses this explicit contract rather than a fuzzy schema match.
IDOT_COLUMN_ALIASES = {
    "CRASH_DATE": ("CrashDate", "Crash Date", "CRASH_DATE", "CrashDateTime"),
    "CRASH_YEAR": ("CrashYr", "Crash Year"),
    "CRASH_MONTH": ("CrashMonth", "Crash Month"),
    "CRASH_DAY": ("CrashDay", "Crash Day"),
    "COUNTY_CODE": ("CountyCode", "County Code", "COUNTY", "County"),
    "TOTALFATALS": ("TotalFatals", "Total Fatals", "INJURIES_FATAL", "FATAL"),
    "AINJURIES": ("AInjuries", "Incapacitating Injuries", "INJURIES_INCAPACITATING"),
}


def idot_county_to_fips(value: object) -> str | None:
    """Map the documented one-based IDOT county code, failing closed."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or int(numeric) != numeric or not 1 <= int(numeric) <= 102:
        return None
    fips = f"17{2 * int(numeric) - 1:03d}"
    return fips if fips in VALID_IL_FIPS else None


def idot_validation_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Expose the alternate split-date schema to the raw-source validator."""
    validation = raw.copy()
    if not any(name in validation.columns for name in IDOT_COLUMN_ALIASES["CRASH_DATE"]):
        year_col = next((name for name in IDOT_COLUMN_ALIASES["CRASH_YEAR"] if name in validation.columns), None)
        month_col = next((name for name in IDOT_COLUMN_ALIASES["CRASH_MONTH"] if name in validation.columns), None)
        day_col = next((name for name in IDOT_COLUMN_ALIASES["CRASH_DAY"] if name in validation.columns), None)
        if year_col and month_col and day_col:
            years = pd.to_numeric(validation[year_col], errors="coerce")
            years = years.where(years >= 100, years + 2000)
            validation["CRASH_DATE"] = pd.to_datetime(
                {"year": years, "month": pd.to_numeric(validation[month_col], errors="coerce"),
                 "day": pd.to_numeric(validation[day_col], errors="coerce")}, errors="coerce")
    return validation


def fetch_via_download_api(item_id: str, year: int, retries: int = 2) -> pd.DataFrame | None:
    """
    Use the opendata download API to get a pre-signed CSV URL, then download it.
    The API redirects to a temporary Azure blob URL. Retries on empty result.
    """
    api_url = (f"https://gis-idot.opendata.arcgis.com/api/download/v1/items/"
               f"{item_id}/csv?layers=0")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(api_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
            # Strip UTF-8 BOM if present
            if content.startswith(b'\xef\xbb\xbf'):
                content = content[3:]
            df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")),
                             low_memory=False)
            df.attrs["source_checksum"] = hashlib.sha256(content).hexdigest()
            if not df.empty:
                log.info("  [download-API] %d rows for %d", len(df), year)
                return df
            log.warning("  [download-API] 0 rows for %d (attempt %d/%d) — retrying",
                        year, attempt + 1, retries + 1)
            time.sleep(3.0)
        except Exception as e:
            log.warning("  [download-API] failed for %d (attempt %d/%d): %s",
                        year, attempt + 1, retries + 1, e)
            if attempt < retries:
                time.sleep(3.0)
    return None


def fetch_via_featureserver(fs_url: str, year: int) -> pd.DataFrame | None:
    """
    Paginate the FeatureServer query endpoint (CSV output).
    services2.arcgis.com is reachable even when gis.idot.illinois.gov is not.
    """
    query_url = fs_url.rstrip("/") + "/query"

    # Get total count
    try:
        count_params = urllib.parse.urlencode({"where": "1=1", "returnCountOnly": "true", "f": "json"})
        req = urllib.request.Request(f"{query_url}?{count_params}", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.loads(r.read())
        total = info.get("count", 0)
        log.info("  [FeatureServer] %d total records for %d", total, year)
    except Exception as e:
        FETCH_FAILURES[year] = e
        log.warning("  [FeatureServer] count failed for %d: %s", year, e)
        return None

    if total == 0:
        return None

    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        return strict_arcgis_dataframe(session, url=query_url, where="1=1",
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields="*", page_size=2000)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [FeatureServer] strict pagination failed for %d: %s", year, exc)
        return None
    finally:
        session.close()

    # Paginate in chunks of 2000 using JSON (CSV returns 400 on some years)
    parts = []
    offset = 0
    page_size = 2000
    while offset < total:
        params = urllib.parse.urlencode({
            "where": "1=1",
            "outFields": "*",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        })
        try:
            req = urllib.request.Request(f"{query_url}?{params}", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                page_data = json.loads(r.read())
            features = page_data.get("features", [])
            if not features:
                break
            rows = [f["attributes"] for f in features]
            chunk_df = pd.DataFrame(rows)
            parts.append(chunk_df)
            offset += len(chunk_df)
            if len(chunk_df) < page_size:
                break
            time.sleep(0.3)
        except Exception as e:
            log.warning("  [FeatureServer] page offset=%d failed: %s", offset, e)
            break

    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    log.info("  [FeatureServer] fetched %d rows for %d", len(df), year)
    return df


def process_idot_df(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Convert raw IDOT CSV to county-day aggregation."""
    if df is None or df.empty:
        return None

    df.columns = [c.strip() for c in df.columns]
    log.info("  Columns (first 20): %s", list(df.columns)[:20])

    # ── Date ────────────────────────────────────────────────────────────────────
    # 2020+: CrashYr (4-digit), CrashMonth, CrashDay (camelCase)
    # 2016–2019: "Crash Date" column (string), or "Crash Year" (2-digit), "Crash Month", "Crash Day"
    date_col = next((c for c in df.columns
                     if c in ("CrashDate", "Crash Date", "CRASH_DATE", "CrashDateTime")), None)

    yr_col  = next((c for c in df.columns if c in ("CrashYr",   "Crash Year")),  None)
    mo_col  = next((c for c in df.columns if c in ("CrashMonth", "Crash Month")), None)
    day_col = next((c for c in df.columns if c in ("CrashDay",   "Crash Day")),   None)

    if date_col:
        df["crash_date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif yr_col and mo_col and day_col:
        yr_vals = pd.to_numeric(df[yr_col], errors="coerce")
        # 2-digit year (e.g. 16 → 2016); 4-digit year (e.g. 2020) use as-is
        yr_str = yr_vals.apply(lambda y: f"20{int(y):02d}" if pd.notna(y) and y < 100
                               else str(int(y)) if pd.notna(y) else "")
        df["crash_date"] = pd.to_datetime(
            yr_str + "-" +
            df[mo_col].astype(str).str.zfill(2) + "-" +
            df[day_col].astype(str).str.zfill(2),
            errors="coerce"
        )
    else:
        log.warning("  No date info for %d; cols=%s", year, list(df.columns)[:30])
        return None

    df = df.dropna(subset=["crash_date"])

    # ── County → FIPS ────────────────────────────────────────────────────────────
    # 2020+: CountyCode; 2016–2019: "County Code"
    county_col = next((c for c in df.columns
                       if c in ("CountyCode", "County Code", "COUNTY", "County")), None)
    if county_col is None:
        log.warning("  No county column for %d; cols=%s", year, list(df.columns)[:30])
        return None

    df["_county_num"] = pd.to_numeric(df[county_col], errors="coerce")
    df = df.dropna(subset=["_county_num"])
    df["_county_num"] = df["_county_num"].astype(int)
    df["fips"] = "17" + ((df["_county_num"] * 2 - 1).astype(str).str.zfill(3))
    invalid = ~df["fips"].isin(VALID_IL_FIPS)
    if invalid.mean() > 0.05:
        log.warning("  %.1f%% rows have unrecognised county code for %d",
                    invalid.mean() * 100, year)
    df = df[~invalid]

    # ── Severity counts ──────────────────────────────────────────────────────────
    # 2020+: TotalFatals, AInjuries, TotalInjured (camelCase)
    # 2016–2019: "Total Fatals", "Incapacitating Injuries", "Total Injured"
    fatal_col   = next((c for c in df.columns
                        if c in ("TotalFatals", "Total Fatals",
                                 "INJURIES_FATAL", "FATAL")), None)
    serious_col = next((c for c in df.columns
                        if c in ("AInjuries", "Incapacitating Injuries",
                                 "INJURIES_INCAPACITATING")), None)
    injured_col = next((c for c in df.columns
                        if c in ("TotalInjured", "Total Injured",
                                 "NUMBERINJURED")), None)
    if fatal_col is None or serious_col is None:
        log.error("  Missing required native outcome field(s) for %d", year)
        return None

    df["fatals"]     = pd.to_numeric(df[fatal_col],   errors="coerce").fillna(0)
    df["serious_inj"]= pd.to_numeric(df[serious_col], errors="coerce").fillna(0)
    df["all_injured"]= pd.to_numeric(df[injured_col], errors="coerce").fillna(0) if injured_col else 0

    log.info("  fatal_col=%s  serious_col=%s  injured_col=%s",
             fatal_col, serious_col, injured_col)

    # ── Aggregate to county-day ──────────────────────────────────────────────────
    agg = (df.groupby(["fips", "crash_date"])
              .agg(il_fatals     =("fatals",      "sum"),
                   il_serious_inj=("serious_inj", "sum"),
                   il_all_injured=("all_injured",  "sum"),
                   il_crashes    =("fatals",       "count"))
              .reset_index()
              .rename(columns={"crash_date": "date"}))
    log.info("  → %d county-days  fatals=%.0f  serious=%.0f",
             len(agg), agg["il_fatals"].sum(), agg["il_serious_inj"].sum())
    return agg


# ── Main download loop ────────────────────────────────────────────────────────
# Executed only as a script. Without this guard the whole download-and-
# write pipeline ran on *import*, so merely importing this module (from a
# test, a notebook, or another builder) silently re-downloaded the source
# and overwrote the Illinois panel on disk.
if __name__ == "__main__":
    log.info("Downloading Illinois IDOT crash data …")
    parts = []
    coverage_rows = []

    for yr in range(2016, 2025):
        log.info("Year %d …", yr)
        info = IDOT_ITEMS.get(yr, {})
        item_id = info.get("item_id")
        fs_url  = info.get("fs_url")

        raw = None
        if item_id:
            raw = fetch_via_download_api(item_id, yr)
        # Fall back to FeatureServer if download API fails or returns 0 rows
        if (raw is None or raw.empty) and fs_url:
            log.info("  Falling back to FeatureServer pagination …")
            raw = fetch_via_featureserver(fs_url, yr)

        validation_raw = None if raw is None else idot_validation_frame(raw)
        coverage_rows.append(validate_source_frame("IL", yr, validation_raw,
            required_columns={"CRASH_DATE", "COUNTY_CODE", "TOTALFATALS", "AINJURIES"},
            date_column="CRASH_DATE", outcome_columns={"TOTALFATALS", "AINJURIES"},
            column_aliases=IDOT_COLUMN_ALIASES,
            geography_column="COUNTY_CODE", geography_mapper=idot_county_to_fips,
            unresolvable_geography_values=frozenset({"0", "0.0"}),
            source_checksum=None if raw is None else raw.attrs.get("source_checksum"),
            terminal_error=FETCH_FAILURES.get(yr)))

        agg = process_idot_df(raw, yr)
        if agg is not None:
            parts.append(agg)
        else:
            log.warning("  Year %d: no data obtained", yr)

        time.sleep(2.0)
        gc.collect()

    write_state_manifest_or_raise("IL", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Illinois data downloaded. Check network access.")
        log.info("Manual alternative: visit https://gis-idot.opendata.arcgis.com/ "
                 "and download per-year CSV files.")
        sys.exit(1)

    il_panel = pd.concat(parts, ignore_index=True)
    il_panel["date"] = pd.to_datetime(il_panel["date"])
    il_panel = (il_panel.groupby(["fips", "date"])
                         .agg(il_fatals     =("il_fatals",      "sum"),
                              il_serious_inj=("il_serious_inj", "sum"),
                              il_all_injured=("il_all_injured", "sum"),
                              il_crashes    =("il_crashes",     "sum"))
                         .reset_index())

    log.info("\nFinal Illinois IDOT panel:")
    log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
             len(il_panel), il_panel["fips"].nunique(),
             il_panel["date"].min().date(), il_panel["date"].max().date())
    log.info("  Total fatals: %.0f  Total serious injuries: %.0f",
             il_panel["il_fatals"].sum(), il_panel["il_serious_inj"].sum())

    il_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
