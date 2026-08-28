"""
02e_fetch_missing_person_text_screen.py — Screen the full OpenFEMA IPAWS
Archived Alerts archive for Silver-Alert / missing-vulnerable-person
language, 2013-2024.

Why this exists
----------------
Checked FEMA's official NWEM Event Code Descriptions fact sheet (current
for essentially this repo's entire study period): there is no
Silver-Alert-equivalent IPAWS event code before the "Missing and
Endangered Persons" (MEP) code, added September 2025 -- after this
repo's 2013-2024 window ends. So historical Silver Alerts sent through
IPAWS/WEA, if any, went out under a generic code (most plausibly CEM or
LAE) with no structured tag to identify them the way AMBER's `>CAE<`
eventCode does in 02c_fetch_openfema_ipaws.py. The only way to find them
is free-text screening of the message content. This is NOT a dead end:
a spot check found a real hit -- Camden County, GA issued an
Alzheimer's-related "Local Area Emergency" (LAE) Alert on 2019-06-05,
with a matching Cancel 43 minutes later (the classic resolved-case
Alert/Cancel pattern) -- confirming some jurisdictions really did push
these through IPAWS under a generic code, findable only via text.

Why the bulk .jsonl stream, not the paginated query API
-----------------------------------------------------------
The standard $top/$skip paginated API (used by 02c for AMBER) degrades
sharply under sustained use: a server-side contains() filter returned
503s/60s+ timeouts on every attempt in this session (including the
proven AMBER pattern, so it's a current backend issue with that
operation generally); even a bare date-only filter with no contains()
went from ~1-4s/request to 40-60s/request after roughly 6 consecutive
requests, consistent with request-count throttling rather than a
query-cost issue. FEMA's DataSets metadata
(https://www.fema.gov/api/open/v1/DataSets?$filter=name eq
'IpawsArchivedAlerts') advertises a separate bulk distribution:
IpawsArchivedAlerts.jsonl (newline-delimited JSON, one record per line).
Verified directly: requesting this endpoint with a $filter and no
$top/$skip streams EVERY matching record continuously (no 1000-row page
cap observed) at a sustained ~200 records/sec with no throttling
slowdown over a 2-minute test pull -- a fundamentally different, much
more reliable code path than the paginated query API.

Streaming, not save-then-filter
---------------------------------
The full archive is >10GB (4.88M records since June 2012, mostly NWS
weather products). Downloading a whole year to disk before filtering it
would mean tens of GB of transient raw data for no benefit -- this
streams the response line-by-line (requests' iter_lines(), one JSON
object per line) and only ever writes MATCHING rows to disk, discarding
each non-matching line immediately. One HTTP request per calendar year
(bounded by a `sent` date range), not per day/month, to minimize
connection-setup overhead and avoid whatever's triggering the paginated
API's per-request throttling.

Keywords
--------
Local (case-insensitive) substring match against known Silver-Alert-
program names and the descriptive language these alerts typically use
("last seen... has dementia", etc.), not just the literal phrase
"Silver Alert" -- since states use different program names (Georgia's
"Mattie's Call", some states' "Golden Alert" / "Senior Alert", Ohio's
"Missing Adult Alert", etc.) and the message body itself often names the
underlying condition even when the program-name phrase is absent.

Checkpointed per year so an interruption doesn't lose progress; safe to
re-run (skips years whose checkpoint file already exists).

Output
------
data/raw/amber/foia/missing_person_text_screen_2013_2024.csv
  Columns: year, id, sent, msgType, event_text, event_code, snippet, matched_keyword
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import AMBER_RAW, STUDY_YEARS
from utils import get_logger

log = get_logger("02e_missing_person")

BULK_URL = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts.jsonl"
RETRY_DELAYS = [10, 30, 60]  # a dropped multi-hour stream is expensive to restart

CHECKPOINT_DIR = AMBER_RAW / "foia" / "_missing_person_checkpoints"
OUT_PATH = AMBER_RAW / "foia" / "missing_person_text_screen_2013_2024.csv"

# Case-insensitive substrings (matched on a lowercased copy of
# originalMessage). Covers: the generic phrase and its common variants;
# known non-"Silver Alert"-named state program names; and the
# descriptive/medical language these alerts characteristically use even
# when the program-name phrase itself is absent from the message body.
KEYWORDS = [
    "silver alert", "silver amber", "senior alert", "golden alert",
    "gold alert", "critical missing", "missing endangered",
    "endangered missing", "missing senior", "missing elderly",
    "missing adult alert", "missing vulnerable", "endangered adult",
    "at-risk missing", "at risk missing",
    "mattie's call", "matties call",
    "dementia", "alzheimer",
]

EVENT_RE = re.compile(r"<event>(.*?)</event>", re.IGNORECASE | re.DOTALL)
EVENTCODE_RE = re.compile(
    r"<eventCode>\s*<valueName>.*?</valueName>\s*<value>(.*?)</value>\s*</eventCode>",
    re.IGNORECASE | re.DOTALL,
)


def screen_line(line: str, year: int) -> dict | None:
    try:
        rec = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    orig = rec.get("originalMessage", "") or ""
    low = orig.lower()
    hit = next((kw for kw in KEYWORDS if kw in low), None)
    if hit is None:
        return None
    ev_match = EVENT_RE.search(orig)
    code_match = EVENTCODE_RE.search(orig)
    return {
        "year": year,
        "id": rec.get("id") or rec.get("identifier", ""),
        "sent": rec.get("sent", ""),
        "msgType": rec.get("msgType", ""),
        "event_text": ev_match.group(1).strip() if ev_match else "",
        "event_code": code_match.group(1).strip() if code_match else "",
        "matched_keyword": hit,
        "snippet": orig[:400].replace("\n", " "),
    }


def fetch_year(year: int, session: requests.Session) -> pd.DataFrame:
    date_filter = f"sent gt '{year}-01-01T00:00:00Z' and sent lt '{year + 1}-01-01T00:00:00Z'"
    params = {
        "$filter": date_filter,
        "$select": "id,sent,msgType,originalMessage",
    }
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            log.warning("  [%d] retrying year %d in %ds...", attempt, year, delay)
            time.sleep(delay)
        rows = []
        n_scanned = 0
        t0 = time.time()
        try:
            with session.get(BULK_URL, params=params, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    n_scanned += 1
                    hit = screen_line(raw_line, year)
                    if hit is not None:
                        rows.append(hit)
                    if n_scanned % 50_000 == 0:
                        log.info("  year %d: %d scanned so far (%.0fs elapsed, %d hits)",
                                 year, n_scanned, time.time() - t0, len(rows))
            log.info("  year %d: DONE -- %d scanned, %d hits, %.0fs",
                     year, n_scanned, len(rows), time.time() - t0)
            return pd.DataFrame(rows)
        except Exception as exc:
            log.warning("  year %d: stream failed after %d records (%.0fs): %s",
                        year, n_scanned, time.time() - t0, exc)
    log.error("  year %d: giving up after %d attempts", year, len(RETRY_DELAYS) + 1)
    return pd.DataFrame()


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; contact: researcher@university.edu)",
        "Accept": "application/jsonl+json",
    })

    for year in STUDY_YEARS:
        ckpt = CHECKPOINT_DIR / f"{year}.parquet"
        if ckpt.exists():
            log.info("Year %d already done -- skipping", year)
            continue
        log.info("Streaming year %d ...", year)
        df_y = fetch_year(year, session)
        df_y.to_parquet(ckpt, index=False)

    parts = sorted(CHECKPOINT_DIR.glob("*.parquet"))
    if not parts:
        log.warning("No checkpoint files found.")
        return
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"])
    combined.to_csv(OUT_PATH, index=False)
    log.info("Saved -> %s (%d unique keyword-matched records)", OUT_PATH, len(combined))
    if len(combined):
        log.info("matched_keyword counts:\n%s", combined["matched_keyword"].value_counts().to_string())
        log.info("event_text top values:\n%s", combined["event_text"].value_counts().head(20).to_string())
        log.info("year counts:\n%s", combined.groupby("year")["id"].nunique().to_string())


if __name__ == "__main__":
    main()
