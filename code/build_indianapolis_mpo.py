"""
build_indianapolis_mpo.py
========================================================
Download Fatal/SSI crash data for the 8-county Indianapolis MPO region from
a live ArcGIS FeatureServer and build a county-day panel.

Multi-county (8-county) addition: no Indiana statewide crash-level feed was
found. This MPO-published dataset draws from Indiana's statewide crash
database, restricted to the MPO's 8 member counties -- confirmed genuinely
county-wide (not a single-city-PD jurisdiction) by checking that Marion
County's own records include Lawrence, Speedway, Beech Grove, and Southport,
not only Indianapolis proper (a same-session Seattle/Kansas City candidate
was rejected for exactly this failure: single-city-PD data covering only a
fraction of its nominal county).

Source: Indianapolis MPO Fatal/SSI Crash Data (ArcGIS Online)
URL: https://services5.arcgis.com/qVN2o0aio8BMbwcJ/arcgis/rest/services/2015_2017_Crash_Data/FeatureServer/0
Coverage: 2018-2024 requested (source has 2018-Sep 2025; 2025 excluded as
partial).
No authentication required.

This source is **Fatal/SSI-only**: every row is a crash classified either
"Fatal" or "SSI" (suspected-serious-injury) via the categorical
`Incapacitated_Fatal` field -- there is no all-crash denominator, and no
per-crash numeric fatality/injury COUNT (just a crash-level severity flag).
Counting Fatal-flagged crashes tracked FARS's true person-fatality count
within 2-8% every year 2018-2024 (checked directly against the live source
before building this), an acceptable proxy given no true person count
exists.

Key fields (confirmed by probe):
  Date                 — epoch milliseconds (real calendar date)
  County                — county name, title-case (e.g. "Marion", "Boone")
  Incapacitated_Fatal   — "Fatal" or "SSI" (severity flag, not a count)

Output columns: fips, date, inmpo_fatals, inmpo_serious_inj
Output: data/processed/indianapolis_mpo_county_day.parquet
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
log = get_logger("indianapolis_mpo")

OUT_PATH = DATA_PROC / "indianapolis_mpo_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

QUERY_URL = "https://services5.arcgis.com/qVN2o0aio8BMbwcJ/arcgis/rest/services/2015_2017_Crash_Data/FeatureServer/0/query"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2018, 2025))
PAGE_SIZE = 2_000
FETCH_FAILURES: dict[int, BaseException] = {}
OUT_FIELDS = "Date,County,Incapacitated_Fatal"

# Indiana's 8 MPO-member counties, standard odd-suffix Census FIPS (state 18).
IN_COUNTY_FIPS = {
    "BOONE": "18011", "HAMILTON": "18057", "HANCOCK": "18059",
    "HENDRICKS": "18063", "JOHNSON": "18081", "MARION": "18097",
    "MORGAN": "18109", "SHELBY": "18145",
}


def county_to_fips(name: str) -> str | None:
    if name is None:
        return None
    return IN_COUNTY_FIPS.get(str(name).strip().upper())


def _count(session: requests.Session, where: str) -> int:
    r = session.get(QUERY_URL, params={"where": where, "returnCountOnly": "true", "f": "json"}, timeout=45)
    r.raise_for_status()
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(f"count query error: {resp['error']}")
    return resp.get("count", 0)


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame:
    where = f"Year = '{year}'"
    try:
        return strict_arcgis_dataframe(session, url=QUERY_URL, where=where,
                                        expected_count=_count(session, where),
                                        id_field="ObjectId", out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return pd.DataFrame()


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

    severity = df["Incapacitated_Fatal"].astype(str).str.strip().str.upper()
    df["is_fatal"] = severity.eq("FATAL")
    df["is_ssi"] = severity.eq("SSI")

    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(inmpo_fatals=("is_fatal", "sum"), inmpo_serious_inj=("is_ssi", "sum"))
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  inmpo_fatals=%d  inmpo_serious_inj=%d",
             year, len(agg), agg["inmpo_fatals"].sum(), agg["inmpo_serious_inj"].sum())
    return agg


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main ─────────────────────────────────────────────────────────────────────
    log.info("Downloading Indianapolis MPO Fatal/SSI crash data (2018–2024) …")
    log.info("Source: %s", QUERY_URL)

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    coverage_rows = []

    for yr in YEARS:
        log.info("=== Year %d ===", yr)
        raw = fetch_year(session, yr)
        coverage_rows.append(validate_source_frame("INMPO", yr, None if raw.empty else raw,
            required_columns={"Date", "County", "Incapacitated_Fatal"},
            date_column="Date", outcome_columns=set(), date_unit="ms",
            geography_column="County", geography_mapper=county_to_fips,
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(raw, yr)
        if agg is not None:
            parts.append(agg)
        del raw, agg
        gc.collect()
        time.sleep(0.5)

    session.close()
    write_state_manifest_or_raise("INMPO", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Indianapolis MPO data downloaded — aborting.")
        sys.exit(1)

    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = (
        panel.groupby(["fips", "date"])
          .agg(inmpo_fatals=("inmpo_fatals", "sum"), inmpo_serious_inj=("inmpo_serious_inj", "sum"))
          .reset_index()
    )

    log.info("")
    log.info("Final Indianapolis MPO panel:")
    log.info("  Rows              : %d", len(panel))
    log.info("  Counties          : %d", panel["fips"].nunique())
    log.info("  Date range        : %s – %s", panel["date"].min().date(), panel["date"].max().date())
    log.info("  inmpo_fatals      : %d", int(panel["inmpo_fatals"].sum()))
    log.info("  inmpo_serious_inj : %d", int(panel["inmpo_serious_inj"].sum()))

    panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
