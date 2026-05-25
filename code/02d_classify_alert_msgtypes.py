"""
02d_classify_alert_msgtypes.py — Fetch msgType for existing alert_ids from
the OpenFEMA IPAWS API and add a msg_type column to the local alert CSV.

Background
----------
The original fetch script (02c) captured only SAME codes and timestamps.
Approximately 46% of alert_ids are Cancel messages (case-resolved WEA) and
~2% are Updates; only ~52% are new Alert messages. All three types fire WEA
phone notifications and can disrupt sleep.

This script:
1. Reads existing openfema_ipaws_alerts_*.csv
2. For each unique alert_id, fetches msgType from the API (batched by 10 IDs)
3. Adds a msg_type column (Alert / Update / Cancel / unknown)
4. Saves the annotated CSV (overwrites in-place, backup kept as .bak)

Note: The API allows filtering by id, but batch OR filters can be large.
We query one ID per request to avoid URL length limits, using a thread pool
for concurrency. Rate-limited to ~4 req/s with exponential backoff on errors.

Run
---
python code/02d_classify_alert_msgtypes.py
"""

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import AMBER_RAW, STUDY_YEARS
from utils import get_logger

log = get_logger("02d_classify")

API_URL = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts"
SLEEP_S = 0.25           # between requests
RETRY_DELAYS = [2, 4, 8]

MSGTYPE_RE = re.compile(
    r"<msgType>\s*(Alert|Update|Cancel)\s*</msgType>",
    re.IGNORECASE,
)

session = requests.Session()
session.headers.update({
    "User-Agent": "amber-alert-research/1.0 (academic; contact: researcher@university.edu)",
    "Accept": "application/json",
})


def fetch_msgtype(alert_id: str, timeout: int = 20) -> str:
    """Fetch msgType for a single alert_id.  Returns 'Alert','Update','Cancel', or 'unknown'."""
    params = {
        "$filter": f"id eq '{alert_id}'",
        "$select": "id,msgType,originalMessage",
        "$top": 1,
    }
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            r = session.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            recs = data if isinstance(data, list) else data.get("IpawsArchivedAlerts", [])
            if not recs:
                return "unknown"
            rec = recs[0]
            # Try structured msgType field first
            mt = rec.get("msgType", "")
            if mt in ("Alert", "Update", "Cancel"):
                return mt
            # Fallback: parse from CAP XML
            orig = rec.get("originalMessage", "") or ""
            m = MSGTYPE_RE.search(orig)
            if m:
                return m.group(1).capitalize()
            return "unknown"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            log.warning("Attempt %d for %s failed: %s", attempt, alert_id[:8], exc)
        except requests.exceptions.HTTPError as exc:
            log.warning("HTTP %s for %s on attempt %d", exc, alert_id[:8], attempt)
    return "unknown"


def main() -> None:
    yr0, yr1 = min(STUDY_YEARS), max(STUDY_YEARS)
    in_path = AMBER_RAW / "foia" / f"openfema_ipaws_alerts_{yr0}_{yr1}.csv"
    if not in_path.exists():
        log.error("Alert CSV not found: %s", in_path)
        return

    df = pd.read_csv(in_path)
    log.info("Loaded %d rows, %d unique alert_ids", len(df), df["alert_id"].nunique())

    if "msg_type" in df.columns:
        already_known = df.loc[df["msg_type"].notna() & (df["msg_type"] != ""), "alert_id"].unique()
        to_classify = [aid for aid in df["alert_id"].unique() if aid not in set(already_known)]
        log.info("msg_type column exists; %d already classified, %d remaining",
                 len(already_known), len(to_classify))
    else:
        to_classify = df["alert_id"].unique().tolist()
        log.info("%d unique alert_ids to classify", len(to_classify))

    if not to_classify:
        log.info("Nothing to do — all alert_ids already have msg_type.")
        return

    # Backup original
    bak = in_path.with_suffix(".csv.bak")
    if not bak.exists():
        import shutil
        shutil.copy2(in_path, bak)
        log.info("Backup saved → %s", bak.name)

    # Fetch msgType sequentially (polite) with progress bar
    type_map: dict[str, str] = {}
    log.info("Fetching msgType for %d alert_ids (may take ~%.0f min)...",
             len(to_classify), len(to_classify) * (SLEEP_S + 0.3) / 60)

    for aid in tqdm(to_classify, desc="classify msgType"):
        mt = fetch_msgtype(aid)
        type_map[aid] = mt
        time.sleep(SLEEP_S)

    # Summary
    from collections import Counter
    counts = Counter(type_map.values())
    log.info("msgType distribution: %s", dict(counts))
    pct_alert = counts.get("Alert", 0) / len(to_classify) * 100
    log.info("  %.1f%% Alert, %.1f%% Cancel, %.1f%% Update, %.1f%% unknown",
             pct_alert,
             counts.get("Cancel", 0) / len(to_classify) * 100,
             counts.get("Update", 0) / len(to_classify) * 100,
             counts.get("unknown", 0) / len(to_classify) * 100)

    # Write back
    df["msg_type"] = df["alert_id"].map(type_map).fillna(
        df.get("msg_type", pd.Series(dtype=str))
    )
    df.to_csv(in_path, index=False)
    log.info("Saved annotated CSV → %s", in_path.name)

    # Quick filter check
    alert_only = df[df["msg_type"] == "Alert"]["alert_id"].nunique()
    log.info("Alert-only unique alert_ids: %d of %d total (%.1f%%)",
             alert_only, df["alert_id"].nunique(),
             alert_only / df["alert_id"].nunique() * 100)


if __name__ == "__main__":
    main()
