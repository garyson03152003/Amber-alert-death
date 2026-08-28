"""
02e_scout_missing_person_alerts.py — Feasibility scout for adding Silver
Alert / missing-vulnerable-adult alerts to the analysis via free-text
screening of the OpenFEMA IPAWS Archived Alerts API.

Why free-text, not an eventCode filter
---------------------------------------
02c_fetch_openfema_ipaws.py filters AMBER alerts on the structured CAP
eventCode tag (`>CAE<`), which is clean and reliable because Child
Abduction Emergency has had its own dedicated IPAWS event code for the
entire 2013-2024 study period. Checked against FEMA's official NWEM
Event Code Descriptions fact sheet (Nov 2020, current for essentially
this whole window): there is NO Silver-Alert-equivalent code in that
list. The "Missing and Endangered Persons" (MEP) code was not added
until September 2025 -- after this repo's study period ends. So a
Silver Alert (or equivalent) issued 2013-2024 through IPAWS, if it went
through IPAWS/WEA at all rather than another channel (highway signs,
opt-in text systems, direct broadcaster relationships), would have had
to be sent under a generic code (most plausibly CEM "Civil Emergency
Message" or LAE "Local Area Emergency") with no structured tag
distinguishing it from any other unrelated message under that code.

This script is a SCOUT, not a production fetcher: it screens a handful
of sample months (not a full 12-year pull) for free-text hits on
missing-senior/vulnerable-adult phrasing, to establish whether this
channel carries ANY meaningful volume of identifiable records before
investing in a full historical build. If yield is near-zero, that
confirms the (already strongly suspected) conclusion that most/all
historical Silver Alerts bypassed IPAWS entirely for this period.

Method
------
* OData contains() on originalMessage is case-sensitive (per 02c's own
  documented experience with AMBER's all-caps variants), so this
  ORs together multiple literal-cased phrase variants rather than
  relying on a single casing.
* Screens for: "Silver Alert" / "SILVER ALERT", "Missing Senior",
  "Missing Elderly", "Endangered Missing", "Missing Endangered",
  "Missing Adult", "Silver AMBER" (a few states used this hybrid name).
* Same retry/backoff and $top/$skip pagination pattern as 02c, since
  FEMA's OpenFEMA API is known to intermittently return 503s / timeouts
  on contains()-filtered queries (observed directly while building this
  script -- even the proven-good AMBER `>CAE<` filter pattern timed out
  repeatedly in the same session before recovering).
* Logs eventCode/event-text and a message snippet for every hit found,
  for manual review of whether they're real Silver-Alert-type messages
  or false positives (e.g. "missing" appearing in an unrelated warning).

This is NOT meant to be run unattended for a full historical pull --
inspect the output CSV before deciding whether a full 12-year fetch is
worth building.

Output: data/raw/amber/foia/missing_person_text_scout_{months}.csv
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import AMBER_RAW
from utils import get_logger

log = get_logger("02e_scout")

API_URL = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts"
TOP = 1000
RETRY_DELAYS = [3, 6, 12, 24, 48]  # more patient than 02c -- API observed flaky
SLEEP_S = 0.5

# Phrase variants, ORed together. contains() is case-sensitive on FEMA's
# backend, so both Title Case and ALL CAPS variants are included (mirrors
# the exact bug 02c documents for AMBER's "CHILD ABDUCTION EMERGENCY").
PHRASES = [
    "Silver Alert", "SILVER ALERT",
    "Missing Senior", "MISSING SENIOR",
    "Missing Elderly", "MISSING ELDERLY",
    "Endangered Missing", "ENDANGERED MISSING",
    "Missing Endangered", "MISSING ENDANGERED",
    "Silver AMBER", "SILVER AMBER",
    "Missing Vulnerable", "MISSING VULNERABLE",
]

EVENT_RE = re.compile(r"<event>(.*?)</event>", re.IGNORECASE | re.DOTALL)
EVENTCODE_RE = re.compile(
    r"<eventCode>\s*<valueName>.*?</valueName>\s*<value>(.*?)</value>\s*</eventCode>",
    re.IGNORECASE | re.DOTALL,
)

# A handful of representative sample months spanning the study period --
# not a full historical pull, just enough to gauge whether this channel
# carries any meaningful volume before investing further.
SAMPLE_MONTHS = [
    (2013, 6), (2014, 6), (2015, 6), (2016, 6), (2017, 6),
    (2018, 6), (2019, 6), (2020, 6), (2021, 6), (2022, 6),
    (2023, 6), (2024, 6),
]


def _get(session: requests.Session, params: dict, timeout: int = 90) -> dict | None:
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            r = session.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("  attempt %d failed (%s); retry in %ds", attempt, exc, delay or 0)
    log.error("  giving up after %d attempts for params=%s", len(RETRY_DELAYS) + 1, params)
    return None


def fetch_month(year: int, month: int, session: requests.Session) -> pd.DataFrame:
    from calendar import monthrange
    _, last_day = monthrange(year, month)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    date_filter = (
        f"sent gt '{year}-{month:02d}-01T00:00:00Z' and "
        f"sent lt '{next_year}-{next_month:02d}-01T00:00:00Z'"
    )
    text_filter = " or ".join(f"contains(originalMessage,'{p}')" for p in PHRASES)
    combined = f"({text_filter}) and ({date_filter})"

    rows = []
    skip = 0
    while skip <= 5000:
        params = {
            "$top": TOP, "$skip": skip, "$filter": combined,
            "$select": "id,sent,msgType,originalMessage",
            "$orderby": "sent asc",
        }
        data = _get(session, params)
        if data is None:
            break
        records = data if isinstance(data, list) else data.get("IpawsArchivedAlerts", [])
        if not records:
            break
        for rec in records:
            orig = rec.get("originalMessage", "") or ""
            ev_match = EVENT_RE.search(orig)
            code_match = EVENTCODE_RE.search(orig)
            rows.append({
                "year": year, "month": month,
                "id": rec.get("id") or rec.get("identifier", ""),
                "sent": rec.get("sent", ""),
                "msgType": rec.get("msgType", ""),
                "event_text": ev_match.group(1).strip() if ev_match else "",
                "event_code": code_match.group(1).strip() if code_match else "",
                "snippet": orig[:300].replace("\n", " "),
            })
        time.sleep(SLEEP_S)
        if len(records) < TOP:
            break
        skip += TOP
    return pd.DataFrame(rows)


def main():
    out_dir = AMBER_RAW / "foia"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "missing_person_text_scout.csv"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; contact: researcher@university.edu)",
        "Accept": "application/json",
    })

    frames = []
    for year, month in SAMPLE_MONTHS:
        log.info("Scouting %d-%02d ...", year, month)
        df_m = fetch_month(year, month, session)
        log.info("  %d-%02d: %d text-match rows", year, month, len(df_m))
        if not df_m.empty:
            frames.append(df_m)

    if not frames:
        log.warning("Zero hits across all %d sampled months. Free-text screening for "
                     "Silver-Alert-type language found nothing in the OpenFEMA IPAWS "
                     "archive for these sample points.", len(SAMPLE_MONTHS))
        pd.DataFrame(columns=["year", "month", "id", "sent", "msgType",
                               "event_text", "event_code", "snippet"]).to_csv(out_path, index=False)
        return

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    combined.to_csv(out_path, index=False)
    log.info("Saved -> %s (%d unique matching records across %d sampled months)",
             out_path, len(combined), len(SAMPLE_MONTHS))
    log.info("event_text value counts:\n%s", combined["event_text"].value_counts().head(20).to_string())
    log.info("event_code value counts:\n%s", combined["event_code"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
