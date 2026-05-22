"""
02_collect_amber_alerts.py — Compile AMBER Alert issuance records.

Data source hierarchy (use whichever is available):
  1. FEMA IPAWS public alert feed  (best: precise timestamps + CAP polygons)
  2. Researcher / FOIA-obtained CSVs placed in data/raw/amber/foia/
  3. GDELT news corpus             (fallback: recovers ~70-80% of alerts via
                                    news coverage; timing is ±hours)

Output: data/processed/amber_alerts_clean.parquet
    Columns:
        alert_id        unique identifier (source + id)
        state_fips      2-digit state FIPS
        county_fips     5-digit county FIPS list (comma-joined string for multi-county)
        issued_utc      pd.Timestamp (UTC)
        issued_local    pd.Timestamp (local time, inferred from state if tz unavailable)
        hour_local      int  [0–23]
        is_night        bool (hour_local in [22,23] or [0,4])
        night_band      str  {"early_night","deep_night","late_night",None}
        cancelled_utc   pd.Timestamp or NaT
        source          str  {"ipaws","foia","gdelt"}

Run: python code/02_collect_amber_alerts.py
     Optional env vars:
       NOAA_CDO_TOKEN — used only for weather; not needed here
"""

import os
import re
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    AMBER_RAW, DATA_PROC,
    IPAWS_BASE, GDELT_DOC_API,
    NIGHT_START_HOUR, NIGHT_END_HOUR, NIGHT_BANDS,
    STUDY_YEARS,
)
from utils import get_logger, download_file

log = get_logger("02_amber")

# Timezone offsets for each state (standard time, UTC offset in hours).
# Used to convert UTC → approximate local time when tz not embedded in alert.
STATE_UTC_OFFSET = {
    "01": -6, "02": -9, "04": -7, "05": -6, "06": -8, "08": -7, "09": -5,
    "10": -5, "11": -5, "12": -5, "13": -5, "15": -10, "16": -7, "17": -6,
    "18": -5, "19": -6, "20": -6, "21": -5, "22": -6, "23": -5, "24": -5,
    "25": -5, "26": -5, "27": -6, "28": -6, "29": -6, "30": -7, "31": -6,
    "32": -8, "33": -5, "34": -5, "35": -7, "36": -5, "37": -5, "38": -6,
    "39": -5, "40": -6, "41": -8, "42": -5, "44": -5, "45": -5, "46": -6,
    "47": -6, "48": -6, "49": -7, "50": -5, "51": -5, "53": -8, "54": -5,
    "55": -6, "56": -7,
}


# ===========================================================================
# Source 1: FEMA IPAWS public REST feed
# ===========================================================================

def fetch_ipaws_alerts(
    start_year: int,
    end_year: int,
    session: requests.Session,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Pull AMBER alerts from the FEMA IPAWS OPEN public API.

    The public endpoint exposes alerts sent through the Integrated Public Alert
    and Warning System; AMBER alerts have eventCode = "CAE".

    Docs: https://apps.fema.gov/IPAWSOPEN_EAS_SERVICE/rest/public (no auth required)
    """
    log.info("Fetching AMBER alerts from FEMA IPAWS...")
    records = []

    # IPAWS returns alerts paged by sent datetime range
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = f"{IPAWS_BASE}/alerts"
            params = {
                "type": "CAE",            # Child Abduction Emergency = AMBER
                "startTime": f"{year}-{month:02d}-01T00:00:00Z",
                "endTime":   f"{year}-{month:02d}-28T23:59:59Z",
                "limit": page_size,
                "offset": 0,
            }
            try:
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code == 404:
                    log.debug("IPAWS endpoint not found (month %d/%d)", month, year)
                    break
                resp.raise_for_status()
                data = resp.json()
                alerts = data.get("alerts", data) if isinstance(data, dict) else data
                for a in alerts:
                    records.append(_parse_ipaws_alert(a))
            except Exception as exc:
                log.warning("IPAWS fetch failed %d-%02d: %s", year, month, exc)

    if not records:
        log.warning("IPAWS returned no records — endpoint may require auth or be unavailable.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    log.info("IPAWS: %d alerts fetched", len(df))
    return df


def _parse_ipaws_alert(raw: dict) -> dict:
    """Extract key fields from a single IPAWS CAP alert dict."""
    info = raw.get("info", [{}])
    info = info[0] if info else {}
    area = info.get("area", [{}])
    area = area[0] if area else {}

    fips_list = []
    for code in area.get("geocode", []):
        if code.get("valueName") == "FIPS6":
            # FIPS6 is zero-padded 6-digit; first 5 are county FIPS
            fips_list.append(code["value"][:5])
        elif code.get("valueName") == "UGC":
            pass  # UGC codes need separate crosswalk

    sent_raw = raw.get("sent", "")
    issued_utc = pd.to_datetime(sent_raw, utc=True, errors="coerce")

    expires_raw = info.get("expires", "")
    cancelled_utc = pd.to_datetime(expires_raw, utc=True, errors="coerce")

    state_fips = fips_list[0][:2] if fips_list else None

    return {
        "alert_id":      f"ipaws_{raw.get('identifier', '')}",
        "state_fips":    state_fips,
        "county_fips":   ",".join(fips_list),
        "issued_utc":    issued_utc,
        "cancelled_utc": cancelled_utc,
        "source":        "ipaws",
    }


# ===========================================================================
# Source 2: FOIA / researcher-supplied CSV files
# ===========================================================================
# Place CSV/Excel files in data/raw/amber/foia/
# Required columns (flexible casing): date, time, state_fips or state, county_fips or county
# Optional: cancelled_date, cancelled_time

FOIA_DIR = AMBER_RAW / "foia"

FOIA_COLUMN_MAP = {
    # normalised name → possible raw names
    "issued_date": ["date", "alert_date", "issued_date", "activation_date"],
    "issued_time": ["time", "alert_time", "issued_time", "activation_time"],
    "state_fips":  ["state_fips", "state", "st_fips", "statefp"],
    "county_fips": ["county_fips", "county", "fips", "countyfp"],
}


def load_foia_files() -> pd.DataFrame:
    """Load all CSV/Excel files from the FOIA directory into a unified DataFrame."""
    FOIA_DIR.mkdir(parents=True, exist_ok=True)
    files = list(FOIA_DIR.glob("*.csv")) + list(FOIA_DIR.glob("*.xlsx"))

    if not files:
        log.info("No FOIA files found in %s — skipping.", FOIA_DIR)
        return pd.DataFrame()

    frames = []
    for f in files:
        log.info("Loading FOIA file: %s", f.name)
        try:
            df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
            df.columns = [c.strip().lower() for c in df.columns]
            df = _normalise_foia_df(df, source_name=f.stem)
            frames.append(df)
        except Exception as exc:
            log.warning("Failed to load %s: %s", f.name, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info("FOIA: %d alerts loaded from %d files", len(combined), len(frames))
    return combined


def _normalise_foia_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Rename columns to canonical form and parse timestamps."""
    # Map raw column names to canonical names
    rename = {}
    for canonical, candidates in FOIA_COLUMN_MAP.items():
        for c in candidates:
            if c in df.columns:
                rename[c] = canonical
                break

    df = df.rename(columns=rename)

    # Parse issued timestamp
    if "issued_date" in df.columns and "issued_time" in df.columns:
        df["issued_utc"] = pd.to_datetime(
            df["issued_date"].astype(str) + " " + df["issued_time"].astype(str),
            errors="coerce",
        )
    elif "issued_date" in df.columns:
        df["issued_utc"] = pd.to_datetime(df["issued_date"], errors="coerce")
    else:
        log.warning("No date column found in FOIA source %s", source_name)
        df["issued_utc"] = pd.NaT

    # Normalise county FIPS to 5-digit string
    if "county_fips" in df.columns:
        df["county_fips"] = df["county_fips"].astype(str).str.strip().str.zfill(5)
    else:
        df["county_fips"] = ""

    if "state_fips" not in df.columns:
        df["state_fips"] = df.get("county_fips", pd.Series("", index=df.index)).str[:2]

    df["cancelled_utc"] = pd.NaT
    df["alert_id"] = f"foia_{source_name}_" + df.index.astype(str)
    df["source"] = "foia"

    return df[["alert_id", "state_fips", "county_fips", "issued_utc", "cancelled_utc", "source"]]


# ===========================================================================
# Source 3: GDELT news corpus (fallback)
# ===========================================================================

def fetch_gdelt_alerts(
    start_year: int,
    end_year: int,
    session: requests.Session,
) -> pd.DataFrame:
    """
    Query the GDELT Document API for news articles about AMBER Alert issuances.

    GDELT captures publication timestamps; these proxy for alert issuance with a
    lag of minutes to hours.  Use IPAWS/FOIA data when available.

    The DOC API returns up to 250 results per query; we slice by month to
    maximise coverage.
    """
    log.info("Fetching AMBER alert proxies from GDELT DOC API...")
    records = []

    query = (
        '"AMBER Alert" OR "Amber Alert issued" OR "child abduction emergency"'
        ' sourcelang:english'
    )

    for year in tqdm(range(start_year, end_year + 1), desc="GDELT years"):
        for month in range(1, 13):
            # GDELT date range format: YYYYMMDDHHMMSS
            start_str = f"{year}{month:02d}01000000"
            if month == 12:
                end_str = f"{year}1231235959"
            else:
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                end_str = f"{year}{month:02d}{last_day:02d}235959"

            params = {
                "query":     query,
                "mode":      "artlist",
                "format":    "json",
                "startdatetime": start_str,
                "enddatetime":   end_str,
                "maxrecords": 250,
                "sort":      "DateDesc",
            }
            try:
                resp = session.get(GDELT_DOC_API, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                articles = data.get("articles", [])
                for art in articles:
                    rec = _parse_gdelt_article(art)
                    if rec:
                        records.append(rec)
                time.sleep(0.5)   # polite rate limiting
            except Exception as exc:
                log.warning("GDELT failed %d-%02d: %s", year, month, exc)

    if not records:
        log.warning("GDELT returned no records.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate: keep one record per (date, inferred state)
    df = df.drop_duplicates(subset=["issued_utc", "state_fips"])
    log.info("GDELT: %d unique alert-proxies after dedup", len(df))
    return df


def _parse_gdelt_article(art: dict) -> Optional[dict]:
    """
    Extract alert-relevant fields from a GDELT article record.

    GDELT does not have county-level geographic tagging, so we infer state from
    article URL / source domain / seendate.  County assignment happens in
    04_build_panel.py using the state + news content when possible.
    """
    url = art.get("url", "")
    title = art.get("title", "").lower()

    # Filter to articles that plausibly describe an alert issuance
    issuance_keywords = ["issued", "activated", "alert issued", "active amber", "new amber"]
    if not any(kw in title for kw in issuance_keywords):
        return None

    seendate = art.get("seendate", "")
    try:
        issued_utc = pd.to_datetime(seendate, format="%Y%m%dT%H%M%SZ", utc=True)
    except Exception:
        issued_utc = pd.to_datetime(seendate, utc=True, errors="coerce")

    if pd.isnull(issued_utc):
        return None

    # Infer state from source URL / content  (heuristic — improve with NER)
    state_fips = _infer_state_from_url(url)

    return {
        "alert_id":      f"gdelt_{hash(url + seendate) % 10**10}",
        "state_fips":    state_fips,
        "county_fips":   "",            # unknown at article level
        "issued_utc":    issued_utc,
        "cancelled_utc": pd.NaT,
        "source":        "gdelt",
    }


# Map state name fragments in URLs to 2-digit FIPS
_STATE_URL_MAP = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
    "california": "06", "colorado": "08", "connecticut": "09", "delaware": "10",
    "florida": "12", "georgia": "13", "hawaii": "15", "idaho": "16",
    "illinois": "17", "indiana": "18", "iowa": "19", "kansas": "20",
    "kentucky": "21", "louisiana": "22", "maine": "23", "maryland": "24",
    "massachusetts": "25", "michigan": "26", "minnesota": "27", "mississippi": "28",
    "missouri": "29", "montana": "30", "nebraska": "31", "nevada": "32",
    "new-hampshire": "33", "new-jersey": "34", "new-mexico": "35",
    "new-york": "36", "north-carolina": "37", "north-dakota": "38", "ohio": "39",
    "oklahoma": "40", "oregon": "41", "pennsylvania": "42", "rhode-island": "44",
    "south-carolina": "45", "south-dakota": "46", "tennessee": "47", "texas": "48",
    "utah": "49", "vermont": "50", "virginia": "51", "washington": "53",
    "west-virginia": "54", "wisconsin": "55", "wyoming": "56",
}

def _infer_state_from_url(url: str) -> Optional[str]:
    url_lower = url.lower()
    for name, fips in _STATE_URL_MAP.items():
        if name in url_lower:
            return fips
    return None


# ===========================================================================
# Common post-processing
# ===========================================================================

def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute local-time fields and night classification.

    Uses standard-time UTC offsets by state as a first-order approximation.
    For precision: load the full IANA tz → county crosswalk and adjust for DST.
    """
    df = df.copy()

    utc_offset = df["state_fips"].map(STATE_UTC_OFFSET).fillna(-6)  # default Central
    df["issued_local"] = df["issued_utc"] + pd.to_timedelta(utc_offset * 60, unit="m")
    df["hour_local"] = df["issued_local"].dt.hour

    # Night classification
    night_hours = set(range(NIGHT_START_HOUR, 24)) | set(range(0, NIGHT_END_HOUR))
    df["is_night"] = df["hour_local"].isin(night_hours)

    # Sub-band
    def classify_band(h):
        for band, (lo, hi) in NIGHT_BANDS.items():
            if lo <= h < hi:
                return band
            if lo > hi:  # wraps midnight — handled by separate range logic
                pass
        # early_night 22-24
        if 22 <= h < 24:
            return "early_night"
        # deep_night 0-3
        if 0 <= h < 3:
            return "deep_night"
        # late_night 3-5
        if 3 <= h < 5:
            return "late_night"
        return None

    df["night_band"] = df["hour_local"].apply(classify_band)
    return df


def explode_counties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where county_fips contains multiple counties (comma-joined), create one row
    per county.  GDELT rows with empty county_fips get a sentinel row for the
    state (used in 04_build_panel for state-level fallback).
    """
    df = df.copy()
    df["county_fips"] = df["county_fips"].fillna("").astype(str)

    # Split multi-county strings
    df["county_fips"] = df["county_fips"].str.split(",")
    df = df.explode("county_fips")
    df["county_fips"] = df["county_fips"].str.strip()

    # Drop empty county rows only if state_fips is also absent
    mask_empty = df["county_fips"].isin(["", "nan", "None"])
    df.loc[mask_empty, "county_fips"] = pd.NA

    return df.reset_index(drop=True)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    AMBER_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})

    start_year, end_year = min(STUDY_YEARS), max(STUDY_YEARS)

    frames = []

    # --- Source 1: IPAWS ---
    ipaws_df = fetch_ipaws_alerts(start_year, end_year, session)
    if not ipaws_df.empty:
        frames.append(ipaws_df)
        log.info("Using IPAWS as primary source (%d records)", len(ipaws_df))

    # --- Source 2: FOIA files ---
    foia_df = load_foia_files()
    if not foia_df.empty:
        frames.append(foia_df)
        log.info("Adding FOIA records (%d records)", len(foia_df))

    # --- Source 3: GDELT fallback ---
    if not frames:
        log.info("No primary sources available; falling back to GDELT.")
        gdelt_df = fetch_gdelt_alerts(start_year, end_year, session)
        if not gdelt_df.empty:
            frames.append(gdelt_df)

    if not frames:
        log.error(
            "No AMBER Alert data obtained from any source.\n"
            "  → Place FOIA CSV files in %s, or ensure IPAWS API is accessible.",
            FOIA_DIR,
        )
        sys.exit(1)

    # Combine sources, dedup on (alert_id)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["alert_id"])
    log.info("Combined: %d unique alerts before filtering", len(combined))

    # Drop records without a usable timestamp
    combined = combined.dropna(subset=["issued_utc"])

    # Study-period filter
    combined = combined[combined["issued_utc"].dt.year.isin(STUDY_YEARS)]

    # Add time fields and night classification
    combined = add_time_fields(combined)

    # Explode multi-county alerts into one row per county
    combined = explode_counties(combined)

    log.info("Final: %d alert-county rows (%d unique alert IDs)",
             len(combined), combined["alert_id"].nunique())

    out_path = DATA_PROC / "amber_alerts_clean.parquet"
    combined.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)

    # Print summary for inspection
    print("\n--- Alert source breakdown ---")
    print(combined["source"].value_counts())
    print("\n--- Night classification ---")
    print(combined["is_night"].value_counts())
    print("\n--- Night band breakdown ---")
    print(combined["night_band"].value_counts())


if __name__ == "__main__":
    main()
