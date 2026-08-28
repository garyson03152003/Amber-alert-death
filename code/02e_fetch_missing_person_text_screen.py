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
is free-text screening of the message content.

Why local screening, not a server-side contains() filter
----------------------------------------------------------
02c/an earlier scout attempt used FEMA's OData `contains(originalMessage,
'...')` filter server-side (as 02c does for AMBER's `>CAE<` code). That
filter was observed to be extremely unreliable in this session -- 503s
and 60s+ timeouts on EVERY contains()-filtered request tried, including
the proven-good AMBER pattern itself (so this is a current backend
issue with that operation, not specific to any one query). A plain
date-only $filter, by contrast, returned in ~1.3 seconds. So this script
pages through the FULL archive by date only (reliable, fast) and does
the keyword screening locally in Python on the downloaded records --
which also sidesteps the case-sensitivity problem entirely (02c has to
OR together Title-Case and ALL-CAPS variants because FEMA's contains()
is case-sensitive; local matching just lowercases everything once).

Scale
-----
The full IpawsArchivedAlerts dataset is 4.88M records since June 2012
(nearly all NWS weather warnings, which dwarf the human-issued civil
alerts this is looking for) -- per FEMA's own DataSets metadata endpoint
(https://www.fema.gov/api/open/v1/DataSets?$filter=name eq
'IpawsArchivedAlerts'). This script does NOT retain non-matching
records (that would mean holding onto essentially the whole 4.88M-record,
>10GB archive) -- it screens each page in memory and only ever writes
matching rows to disk, so the persistent footprint stays small
regardless of how much of the archive gets paged through.

Keywords
--------
Local (case-insensitive) substring match against known Silver-Alert-
program names and the descriptive language these alerts typically use
("last seen... has dementia", etc.), not just the literal phrase
"Silver Alert" -- since states use different program names (Georgia's
"Mattie's Call", some states' "Golden Alert" / "Senior Alert", Ohio's
"Missing Adult Alert", etc.) and the message body itself often names the
underlying condition even when the program-name phrase is absent.

Checkpointed per month so an interruption doesn't lose progress; safe to
re-run (skips months whose checkpoint file already exists).

Output
------
data/raw/amber/foia/missing_person_text_screen_2013_2024.csv
  Columns: year, month, id, sent, msgType, event_text, event_code, snippet, matched_keyword
"""
import re
import sys
import time
from calendar import monthrange
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import AMBER_RAW, STUDY_YEARS
from utils import get_logger

log = get_logger("02e_missing_person")

API_URL = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts"
TOP = 1000
RETRY_DELAYS = [3, 6, 12, 24]
SLEEP_S = 0.3
MAX_SKIP = 200_000  # generous ceiling; a month should never approach this

CHECKPOINT_DIR = AMBER_RAW / "foia" / "_missing_person_checkpoints"
OUT_PATH = AMBER_RAW / "foia" / "missing_person_text_screen_2013_2024.csv"

# Case-insensitive substrings (matched on a lowercased copy of
# originalMessage). Covers: the generic phrase and its common variants;
# known non-"Silver Alert"-named state program names; and the
# descriptive/medical language these alerts characteristically use even
# when the program-name phrase itself is absent from the message body.
KEYWORDS = [
    # generic phrase variants
    "silver alert", "silver amber", "senior alert", "golden alert",
    "gold alert", "critical missing", "missing endangered",
    "endangered missing", "missing senior", "missing elderly",
    "missing adult alert", "missing vulnerable", "endangered adult",
    "at-risk missing", "at risk missing",
    # named state programs that don't use "silver"/"senior"/"golden"
    "mattie's call", "matties call",
    # descriptive/medical language common in the message body
    "dementia", "alzheimer",
]


def _get(session: requests.Session, params: dict, timeout: int = 60) -> dict | None:
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            r = session.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("    attempt %d failed (%s); retry in %ds", attempt, exc, delay or 0)
    log.error("    giving up for params=%s", params)
    return None


EVENT_RE = re.compile(r"<event>(.*?)</event>", re.IGNORECASE | re.DOTALL)
EVENTCODE_RE = re.compile(
    r"<eventCode>\s*<valueName>.*?</valueName>\s*<value>(.*?)</value>\s*</eventCode>",
    re.IGNORECASE | re.DOTALL,
)


def screen_record(rec: dict, year: int, month: int) -> dict | None:
    orig = rec.get("originalMessage", "") or ""
    low = orig.lower()
    hit = next((kw for kw in KEYWORDS if kw in low), None)
    if hit is None:
        return None
    ev_match = EVENT_RE.search(orig)
    code_match = EVENTCODE_RE.search(orig)
    return {
        "year": year, "month": month,
        "id": rec.get("id") or rec.get("identifier", ""),
        "sent": rec.get("sent", ""),
        "msgType": rec.get("msgType", ""),
        "event_text": ev_match.group(1).strip() if ev_match else "",
        "event_code": code_match.group(1).strip() if code_match else "",
        "matched_keyword": hit,
        "snippet": orig[:400].replace("\n", " "),
    }


def fetch_month(year: int, month: int, session: requests.Session) -> pd.DataFrame:
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    date_filter = (
        f"sent gt '{year}-{month:02d}-01T00:00:00Z' and "
        f"sent lt '{next_year}-{next_month:02d}-01T00:00:00Z'"
    )
    rows = []
    skip = 0
    n_scanned = 0
    while skip <= MAX_SKIP:
        params = {
            "$top": TOP, "$skip": skip, "$filter": date_filter,
            "$select": "id,sent,msgType,originalMessage",
            "$orderby": "sent asc",
        }
        data = _get(session, params)
        if data is None:
            log.error("  %d-%02d: aborting at skip=%d after repeated failures", year, month, skip)
            break
        records = data if isinstance(data, list) else data.get("IpawsArchivedAlerts", [])
        if not records:
            break
        n_scanned += len(records)
        for rec in records:
            hit = screen_record(rec, year, month)
            if hit is not None:
                rows.append(hit)
        time.sleep(SLEEP_S)
        if len(records) < TOP:
            break
        skip += TOP
    log.info("  %d-%02d: scanned %d records, %d keyword hits", year, month, n_scanned, len(rows))
    return pd.DataFrame(rows)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; contact: researcher@university.edu)",
        "Accept": "application/json",
    })

    months = [(y, m) for y in STUDY_YEARS for m in range(1, 13)]
    log.info("Screening %d months (%d-%d) for missing-person keywords...",
             len(months), min(STUDY_YEARS), max(STUDY_YEARS))

    for year, month in months:
        ckpt = CHECKPOINT_DIR / f"{year}_{month:02d}.parquet"
        if ckpt.exists():
            continue
        log.info("Scanning %d-%02d ...", year, month)
        df_m = fetch_month(year, month, session)
        df_m.to_parquet(ckpt, index=False)

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
