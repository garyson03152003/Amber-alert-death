"""
extend_texas_txdot.py
=============================================================
Download Texas TxDOT crash data for 2023-2024 only and append
to the existing 2020-2022 parquet.

The original build_texas_txdot.py timed out on the ArcGIS
server for those two years; this script retries them.
"""
import gc, sys, time, warnings
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("tx_extend")

OUT_PATH = DATA_PROC / "texas_txdot_county_day.parquet"
FS_URL   = ("https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/"
             "TXDOT_Statewide_Bicyclist_Involved_Crashes/FeatureServer/0/query")
HEADERS  = {"User-Agent": "amber-research/1.0 (academic)"}
YEARS    = [2023, 2024]
PAGE_SIZE = 2000
OUT_FIELDS = "crash_id,cnty_id,crash_date,death_cnt,sus_serious_injry_cnt,crash_fatal_fl"
SLEEP_PAGE = 0.25
SLEEP_YEAR = 3.0


def cris_to_fips(cris_id: int) -> str:
    return f"48{2 * cris_id - 1:03d}"


def fetch_year(session, year):
    where = f"crash_date >= '{year}-01-01' AND crash_date <= '{year}-12-31'"
    try:
        r = session.get(FS_URL, params={"where": where, "returnCountOnly": "true", "f": "json"}, timeout=45)
        r.raise_for_status()
        total = r.json().get("count", 0)
        log.info("  [%d] total records: %d", year, total)
    except Exception as e:
        log.warning("  [%d] count failed: %s", year, e)
        return None

    if total == 0:
        return None

    parts, offset = [], 0
    while offset < total:
        params = {"where": where, "outFields": OUT_FIELDS,
                  "resultOffset": offset, "resultRecordCount": PAGE_SIZE, "f": "json"}
        for attempt in range(3):
            try:
                r = session.get(FS_URL, params=params, timeout=120)
                r.raise_for_status()
                feats = r.json().get("features", [])
                break
            except Exception as e:
                log.warning("  [%d] offset=%d attempt=%d failed: %s", year, offset, attempt, e)
                time.sleep(4 * (attempt + 1))
                feats = []

        if not feats:
            break
        parts.append(pd.DataFrame([f["attributes"] for f in feats]))
        offset += len(feats)
        if offset % 100_000 == 0 or offset >= total:
            log.info("  [%d] … %d / %d", year, offset, total)
        time.sleep(SLEEP_PAGE)

    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] fetched %d rows", year, len(df))
    return df


def process_year(df, year):
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["crash_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["cnty_id"] = pd.to_numeric(df["cnty_id"], errors="coerce")
    df = df[(df["cnty_id"] >= 1) & (df["cnty_id"] <= 254)].copy()
    df["fips"] = df["cnty_id"].apply(lambda x: cris_to_fips(int(x)))
    df["death_cnt"] = pd.to_numeric(df["death_cnt"], errors="coerce").fillna(0)
    df["sus_serious_injry_cnt"] = pd.to_numeric(df["sus_serious_injry_cnt"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["crash_id"]).copy()
    agg = (
        df.groupby(["fips", "date"])
        .agg(tx_crashes=("crash_id", "count"),
             tx_fatals=("death_cnt", "sum"),
             tx_serious_inj=("sus_serious_injry_cnt", "sum"))
        .reset_index()
    )
    log.info("  [%d] → %d county-days  crashes=%.0f  fatals=%.0f  serious_inj=%.0f",
             year, len(agg), agg["tx_crashes"].sum(),
             agg["tx_fatals"].sum(), agg["tx_serious_inj"].sum())
    return agg


# ── Load existing parquet ─────────────────────────────────────────────────────
if not OUT_PATH.exists():
    log.error("Existing TX parquet not found at %s", OUT_PATH)
    sys.exit(1)

existing = pd.read_parquet(OUT_PATH)
log.info("Existing TX panel: %d rows  %s – %s",
         len(existing),
         existing["date"].min().date(),
         existing["date"].max().date())

# Drop any existing rows for 2023-2024 (shouldn't be any, but be safe)
existing = existing[pd.to_datetime(existing["date"]).dt.year < 2023].copy()

# ── Download new years ────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)
new_parts = []

for yr in YEARS:
    log.info("Year %d …", yr)
    raw = fetch_year(session, yr)
    agg = process_year(raw, yr)
    if agg is not None:
        new_parts.append(agg)
    del raw, agg
    gc.collect()
    time.sleep(SLEEP_YEAR)

session.close()

if not new_parts:
    log.error("No new years downloaded — keeping existing 2020-2022 parquet.")
    sys.exit(0)

# ── Combine and save ──────────────────────────────────────────────────────────
new_data = pd.concat(new_parts, ignore_index=True)
log.info("New years total: %d county-days", len(new_data))

panel = pd.concat([existing, new_data], ignore_index=True)
panel = (
    panel.groupby(["fips", "date"])
    .agg(tx_crashes=("tx_crashes", "sum"),
         tx_fatals=("tx_fatals", "sum"),
         tx_serious_inj=("tx_serious_inj", "sum"))
    .reset_index()
)
panel = panel.sort_values(["fips", "date"]).reset_index(drop=True)
panel["date"] = pd.to_datetime(panel["date"])

log.info("\nFinal combined TX panel:")
log.info("  Rows: %d  Counties: %d  %s – %s",
         len(panel), panel["fips"].nunique(),
         panel["date"].min().date(), panel["date"].max().date())
log.info("  tx_crashes=%.0f  tx_fatals=%.0f  tx_serious_inj=%.0f",
         panel["tx_crashes"].sum(), panel["tx_fatals"].sum(),
         panel["tx_serious_inj"].sum())

panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
