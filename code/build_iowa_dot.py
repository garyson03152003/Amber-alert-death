"""
build_iowa_dot.py
========================================================
Download Iowa DOT crash data (SOR - Statewide Crash Data) and build
a county-day panel of fatalities and serious injuries.

Source: Iowa DOT Open Data Hub (data.iowadot.gov)
Item: Crash Data (SOR), item ID 7a1f786d55a2439a9bfa8f7a527936e8
FeatureServer: https://gis.iowadot.gov/agshost/rest/services/Traffic_Safety/Crash_Data/FeatureServer/0
Coverage: 2015–2024 (616k+ records)
No authentication required.

Key columns:
  CRASH_DATE      — date string '2015/01/18 00:00:00+00'
  COUNTY_NAME     — county name (uppercase, e.g. "JOHNSON")
  FATALITIES      — count of fatalities
  MAJINJURY       — suspected serious injuries (KABCO-A equivalent)
  INJURIES        — total injuries

Iowa has 99 counties. FIPS = "19" + str(county_num * 2 - 1).zfill(3)
(sequential alphabetical, 1-99).

Output: data/processed/iowa_dot_county_day.parquet
"""
import sys, warnings, gc, time, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from state_dot_sources import (filter_to_requested_years, strict_arcgis_dataframe,
                               validate_source_frame, write_state_manifest_or_raise)

warnings.filterwarnings("ignore")
log = get_logger("iowa_dot")

OUT_PATH = DATA_PROC / "iowa_dot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}
DOWNLOAD_URL = ("https://data.iowadot.gov/api/download/v1/items/"
                "7a1f786d55a2439a9bfa8f7a527936e8/csv?layers=0")
FS_BASE = ("https://gis.iowadot.gov/agshost/rest/services/"
           "Traffic_Safety/Crash_Data/FeatureServer/0/query")

# Iowa county FIPS — 99 counties, sequential alphabetical → FIPS 19001..19197
IA_COUNTY_FIPS = {
    "ADAIR": "19001", "ADAMS": "19003", "ALLAMAKEE": "19005", "APPANOOSE": "19007",
    "AUDUBON": "19009", "BENTON": "19011", "BLACK HAWK": "19013", "BOONE": "19015",
    "BREMER": "19017", "BUCHANAN": "19019", "BUENA VISTA": "19021", "BUTLER": "19023",
    "CALHOUN": "19025", "CARROLL": "19027", "CASS": "19029", "CEDAR": "19031",
    "CERRO GORDO": "19033", "CHEROKEE": "19035", "CHICKASAW": "19037", "CLARKE": "19039",
    "CLAY": "19041", "CLAYTON": "19043", "CLINTON": "19045", "CRAWFORD": "19047",
    "DALLAS": "19049", "DAVIS": "19051", "DECATUR": "19053", "DELAWARE": "19055",
    "DES MOINES": "19057", "DICKINSON": "19059", "DUBUQUE": "19061", "EMMET": "19063",
    "FAYETTE": "19065", "FLOYD": "19067", "FRANKLIN": "19069", "FREMONT": "19071",
    "GREENE": "19073", "GRUNDY": "19075", "GUTHRIE": "19077", "HAMILTON": "19079",
    "HANCOCK": "19081", "HARDIN": "19083", "HARRISON": "19085", "HENRY": "19087",
    "HOWARD": "19089", "HUMBOLDT": "19091", "IDA": "19093", "IOWA": "19095",
    "JACKSON": "19097", "JASPER": "19099", "JEFFERSON": "19101", "JOHNSON": "19103",
    "JONES": "19105", "KEOKUK": "19107", "KOSSUTH": "19109", "LEE": "19111",
    "LINN": "19113", "LOUISA": "19115", "LUCAS": "19117", "LYON": "19119",
    "MADISON": "19121", "MAHASKA": "19123", "MARION": "19125", "MARSHALL": "19127",
    "MILLS": "19129", "MITCHELL": "19131", "MONONA": "19133", "MONROE": "19135",
    "MONTGOMERY": "19137", "MUSCATINE": "19139", "O'BRIEN": "19141", "OBRIEN": "19141",
    "OSCEOLA": "19143", "PAGE": "19145", "PALO ALTO": "19147", "PLYMOUTH": "19149",
    "POCAHONTAS": "19151", "POLK": "19153", "POTTAWATTAMIE": "19155", "POWESHIEK": "19157",
    "RINGGOLD": "19159", "SAC": "19161", "SCOTT": "19163", "SHELBY": "19165",
    "SIOUX": "19167", "STORY": "19169", "TAMA": "19171", "TAYLOR": "19173",
    "UNION": "19175", "VAN BUREN": "19177", "WAPELLO": "19179", "WARREN": "19181",
    "WASHINGTON": "19183", "WAYNE": "19185", "WEBSTER": "19187", "WINNEBAGO": "19189",
    "WINNESHIEK": "19191", "WOODBURY": "19193", "WORTH": "19195", "WRIGHT": "19197",
}
VALID_IA_FIPS = set(IA_COUNTY_FIPS.values())
FETCH_FAILURES: dict[int, BaseException] = {}

OUT_FIELDS = "CRASH_DATE,COUNTY_NAME,FATALITIES,MAJINJURY,INJURIES,CRASH_MONTH,CRASH_DAY"


def fetch_via_download_api(retries: int = 2) -> pd.DataFrame | None:
    """Download the full SOR CSV via the opendata download API."""
    import io
    for attempt in range(retries + 1):
        try:
            log.info("  [download-API] attempt %d/%d …", attempt + 1, retries + 1)
            r = requests.get(DOWNLOAD_URL, headers=HEADERS, timeout=180,
                             allow_redirects=True)
            r.raise_for_status()
            content = r.content
            if content.startswith(b'\xef\xbb\xbf'):
                content = content[3:]
            df = pd.read_csv(
                io.StringIO(content.decode("utf-8", errors="replace")),
                low_memory=False,
                usecols=lambda c: c.strip() in (
                    "CRASH_DATE", "COUNTY_NAME", "FATALITIES",
                    "MAJINJURY", "INJURIES", "CRASH_MONTH", "CRASH_DAY",
                    "CSEV"
                )
            )
            df.attrs["source_checksum"] = hashlib.sha256(content).hexdigest()
            if not df.empty:
                log.info("  [download-API] %d rows downloaded", len(df))
                return df
            log.warning("  [download-API] 0 rows — retrying")
            time.sleep(5.0)
        except Exception as e:
            log.warning("  [download-API] failed (attempt %d): %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(5.0)
    return None


def fetch_via_featureserver(year: int, retries: int = 2) -> pd.DataFrame | None:
    """Paginate one year from the FeatureServer (JSON format)."""
    # Count
    where = (f"CRASH_DATE >= DATE '{year}-01-01' AND "
             f"CRASH_DATE < DATE '{year+1}-01-01'")
    try:
        r = requests.get(FS_BASE, params={
            "where": where,
            "returnCountOnly": "true", "f": "json"
        }, timeout=20, headers=HEADERS)
        total = r.json().get("count", 0)
        log.info("  [FeatureServer] year %d: %d records", year, total)
    except Exception as e:
        FETCH_FAILURES[year] = e
        log.warning("  [FeatureServer] count failed for %d: %s", year, e)
        return None

    if total == 0:
        return None

    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        return strict_arcgis_dataframe(session, url=FS_BASE, where=where,
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields=OUT_FIELDS, page_size=2000)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [FeatureServer] strict pagination failed for %d: %s", year, exc)
        return None
    finally:
        session.close()

    parts = []
    offset = 0
    page_size = 2000
    while offset < total:
        for attempt in range(retries + 1):
            try:
                r = requests.get(FS_BASE, params={
                    "where": (f"CRASH_DATE >= DATE '{year}-01-01' AND "
                              f"CRASH_DATE < DATE '{year+1}-01-01'"),
                    "outFields": OUT_FIELDS,
                    "resultOffset": offset,
                    "resultRecordCount": page_size,
                    "f": "json",
                }, timeout=60, headers=HEADERS)
                r.raise_for_status()
                feats = r.json().get("features", [])
                if not feats:
                    offset = total  # break outer loop
                    break
                rows = [f["attributes"] for f in feats]
                parts.append(pd.DataFrame(rows))
                offset += len(feats)
                if len(feats) < page_size:
                    offset = total  # last page
                break
            except Exception as e:
                log.warning("  [FeatureServer] offset=%d attempt %d failed: %s",
                            offset, attempt + 1, e)
                if attempt < retries:
                    time.sleep(3.0)
                else:
                    offset = total  # give up on this page
        time.sleep(0.3)

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """Convert raw Iowa crash rows to county-day aggregation."""
    if df is None or df.empty:
        return None

    df.columns = [c.strip() for c in df.columns]
    log.info("  Columns: %s", list(df.columns)[:15])

    # ── Date ────────────────────────────────────────────────────────────────────
    date_col = next((c for c in df.columns if "CRASH_DATE" in c), None)
    if date_col:
        df["crash_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df["crash_date"] = df["crash_date"].dt.normalize().dt.tz_localize(None)
    else:
        # Fallback: reconstruct from CRASH_MONTH + CRASH_DAY + year
        log.warning("  No CRASH_DATE column; reconstructing from month/day")
        return None

    df = df.dropna(subset=["crash_date"])

    # The validated Iowa source contract begins in 2015.
    df = filter_to_requested_years(df, state="IA", date_column="crash_date")
    if df.empty:
        log.warning("  No rows in 2015-2024 range after date filter")
        return None

    # ── County → FIPS ────────────────────────────────────────────────────────────
    county_col = next((c for c in df.columns if "COUNTY_NAME" in c or c == "COUNTY_NAME"), None)
    if county_col is None:
        log.warning("  No county column")
        return None

    df["county_upper"] = df[county_col].astype(str).str.upper().str.strip()
    df["fips"] = df["county_upper"].map(IA_COUNTY_FIPS)
    invalid = df["fips"].isna()
    if invalid.mean() > 0.05:
        log.warning("  %.1f%% rows have unknown county: %s",
                    invalid.mean() * 100,
                    df.loc[invalid, "county_upper"].value_counts().head(5).to_dict())
    df = df[~invalid]

    # ── Severity counts ──────────────────────────────────────────────────────────
    fatal_col   = next((c for c in df.columns if c == "FATALITIES"), None)
    serious_col = next((c for c in df.columns if c == "MAJINJURY"), None)
    inj_col     = next((c for c in df.columns if c == "INJURIES"), None)
    if fatal_col is None or serious_col is None:
        log.error("  Required Iowa native fatal/serious outcome field missing")
        return None

    df["fatals"]      = pd.to_numeric(df[fatal_col],   errors="coerce").fillna(0)
    df["serious_inj"] = pd.to_numeric(df[serious_col], errors="coerce").fillna(0)
    df["all_injured"] = pd.to_numeric(df[inj_col],     errors="coerce").fillna(0) if inj_col     else 0

    # ── Aggregate ────────────────────────────────────────────────────────────────
    agg = (df.groupby(["fips", "crash_date"])
              .agg(ia_fatals     =("fatals",      "sum"),
                   ia_serious_inj=("serious_inj", "sum"),
                   ia_all_injured=("all_injured",  "sum"),
                   ia_crashes    =("fatals",       "count"))
              .reset_index()
              .rename(columns={"crash_date": "date"}))

    log.info("  → %d county-days  fatals=%.0f  serious=%.0f  years %s–%s",
             len(agg), agg["ia_fatals"].sum(), agg["ia_serious_inj"].sum(),
             agg["date"].dt.year.min(), agg["date"].dt.year.max())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Iowa DOT crash data (SOR) …")

# Try full download API first (all years in one shot)
raw = fetch_via_download_api()

if raw is None or raw.empty:
    log.info("Download API failed — falling back to FeatureServer year-by-year …")
    parts = []
    for yr in range(2015, 2025):
        log.info("Year %d …", yr)
        part = fetch_via_featureserver(yr)
        if part is not None:
            parts.append(part)
        time.sleep(2.0)
        gc.collect()
    raw = pd.concat(parts, ignore_index=True) if parts else None

if raw is None or raw.empty:
    log.error("No Iowa data obtained.")
    import sys; sys.exit(1)

coverage_rows = []
raw_years = pd.to_datetime(raw["CRASH_DATE"], errors="coerce").dt.year
for yr in range(2015, 2025):
    year_raw = raw.loc[raw_years.eq(yr)].copy()
    year_raw.attrs["source_checksum"] = raw.attrs.get("source_checksum")
    coverage_rows.append(validate_source_frame("IA", yr, None if year_raw.empty else year_raw,
        required_columns={"CRASH_DATE", "COUNTY_NAME", "FATALITIES", "MAJINJURY"},
        date_column="CRASH_DATE", outcome_columns={"FATALITIES", "MAJINJURY"},
        geography_column="COUNTY_NAME", geography_mapper=lambda value: IA_COUNTY_FIPS.get(str(value).strip().upper()),
        source_checksum=raw.attrs.get("source_checksum"), terminal_error=FETCH_FAILURES.get(yr)))
write_state_manifest_or_raise("IA", coverage_rows, output_dir=DATA_PROC / "coverage")

log.info("Processing %d raw rows …", len(raw))
ia_panel = process_df(raw)

if ia_panel is None or ia_panel.empty:
    log.error("Processing returned no data.")
    import sys; sys.exit(1)

# Final de-duplicate
ia_panel = (ia_panel.groupby(["fips", "date"])
                     .agg(ia_fatals     =("ia_fatals",      "sum"),
                          ia_serious_inj=("ia_serious_inj", "sum"),
                          ia_all_injured=("ia_all_injured", "sum"),
                          ia_crashes    =("ia_crashes",     "sum"))
                     .reset_index())

log.info("\nFinal Iowa DOT panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(ia_panel), ia_panel["fips"].nunique(),
         ia_panel["date"].min().date(), ia_panel["date"].max().date())
log.info("  Total fatals: %.0f  Total serious injuries: %.0f",
         ia_panel["ia_fatals"].sum(), ia_panel["ia_serious_inj"].sum())

ia_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
