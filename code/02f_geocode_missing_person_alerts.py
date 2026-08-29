"""
02f_geocode_missing_person_alerts.py — Attach county FIPS (and an
extracted age, where mentioned) to the Silver Alert / missing-
endangered-person text-screening hits found by
02e_fetch_missing_person_text_screen.py, and drop the event codes that
shouldn't be treated as new, genuine, elderly-relevant records.

Why this second pass is needed
--------------------------------
02e stored only a 400-character snippet of each hit's originalMessage
(enough to confirm the keyword match and pull the <event>/<eventCode>
tags near the top of the CAP message, but not enough to reach the
<info><area><geocode> block the SAME/county code lives in -- verified
directly: 0 of 1,428 kept snippets contain the string "SAME" -- or the
<parameter><valueName>CMAMlongtext</valueName> field the actual WEA
message text lives in for most of these records, needed for age
extraction below). Getting county geography and age requires the full
message body, which means re-streaming -- but only for the specific
(year, month) periods that actually contain a kept hit, not the full
132-month archive again.

Dropped before re-fetching (see missing_person_text_screen_2013_2024.csv
and its commit message for counts; all five confirmed by manually
pulling and reading the full message text, not assumed from the code
alone):
  - 'ADR' (Administrative Message): FEMA's own NWEM event code glossary
    marks this EAS & NWEM only -- not WEA-eligible, so it doesn't carry
    the phone-alert mechanism this repo's analyses are built around.
  - 'CAE' (Child Abduction Emergency): these are AMBER alerts that
    happen to also contain one of the missing-person keywords
    (plausible, since real AMBER text describes a missing child using
    similar language) -- they are already in 02c_fetch_openfema_ipaws.py's
    dataset, not a new population, and keeping them here would
    double-count the same events under a different label.
  - 'NWS', 'RWT', 'TOE': spot-checked individually (only 1-2 hits each)
    and confirmed as coincidental keyword collisions in unrelated
    messages -- a Wind Chill Advisory that happened to contain
    "dementia", Required Weekly Test (routine system test) messages
    that happened to contain "silver/golden alert", and a 911 Telephone
    Outage notice that happened to contain "silver alert".

NOT dropped, despite generic-sounding matched keywords: a manual review
of "missing endangered" / "endangered missing" / "endangered adult" /
"at risk missing" hits found a genuinely mixed population -- real
elderly/dementia cases (a 73yo Alzheimer's case, an 89yo dementia case)
alongside missing children (an 11yo, a 14yo autistic juvenile) and a
non-elderly adult (44yo). Rather than filter these broader keyword
categories by dropping rows, this pass extracts a mentioned age (see
AGE_RE below) and uses it to CLASSIFY, not exclude: a record with a
parsed age under 18 is labeled population='child_amber_adjacent'
(conceptually the same missing/endangered-minor population AMBER/CAE
covers, just issued under a different generic code instead of CAE) and
everything else -- including the 69% of records with no parsed age at
all, since there's no positive evidence they're a child -- keeps
population='missing_person'. Nothing is dropped from the dataset on
account of age; downstream analyses can filter on the `population`
column as needed.

Geocoding method
-----------------
Reuses 02c_fetch_openfema_ipaws.py's SAME-code extraction verbatim (via
import, not reimplementation): SAME_RE / AREADESC_RE regexes over the
raw CAP XML, plus _resolve_nonstandard_same() for the small share of
non-standard (P != 0) SAME prefixes that need areaDesc text matching
against county names rather than naive digit-stripping. Same 0SSCCC ->
FIPS convention, same "CCC=000 means statewide, expand downstream" as
02c/run_state_dot_analysis_fixed.py.

Only re-scans (year, month) periods that contain a kept hit (90 of the
132 months in the original 2013-2024 range), using the same one-request-
per-month bulk .jsonl streaming approach as 02e (see that script's
docstring for why: the paginated $top/$skip API is unreliable, and
per-year requests die around 68-80 minutes in, so per-month is the
proven-reliable chunk size). Checkpointed per month.

Output
------
data/raw/amber/foia/missing_person_alerts_geocoded_2013_2024.csv
  Columns: alert_id, sent_utc, fips, state_fips, msg_type, event_code,
  event_text, matched_keyword, mentioned_age (nullable int; no age
  reliably parsed out of the message text for that record), population
  ('child_amber_adjacent' if mentioned_age < 18, else 'missing_person' --
  a classification, not a filter; every row from the keyword screen is
  still present)
  (one row per alert x county, same shape as 02c's AMBER output --
  statewide alerts appear as a single COUNTY=000 row here, same as
  02c's raw output; expansion to individual counties is a downstream
  modeling choice via the same _expand_statewide_rows() convention.)
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

fetch_screen = import_module("02e_fetch_missing_person_text_screen")
ipaws_amber = import_module("02c_fetch_openfema_ipaws")

from config import AMBER_RAW
from utils import get_logger

log = get_logger("02f_geocode_missing_person")

BULK_URL = fetch_screen.BULK_URL
RETRY_DELAYS = fetch_screen.RETRY_DELAYS
KEYWORDS = fetch_screen.KEYWORDS
EVENT_RE = fetch_screen.EVENT_RE
EVENTCODE_RE = fetch_screen.EVENTCODE_RE

DROP_EVENT_CODES = {
    "ADR",  # Administrative Message: EAS/NWEM only per FEMA's glossary, not WEA
    "CAE",  # Child Abduction Emergency: these ARE existing AMBER alerts, not new records
    "NWS",  # spot-checked: a Wind Chill Advisory that coincidentally matched "dementia"
    "RWT",  # spot-checked: Required Weekly Test messages, not real alerts
    "TOE",  # spot-checked: a 911 Telephone Outage notice, unrelated to missing persons
}

# Loose age-mention extractor, run over the full CAP message text ("73YO",
# "89 YO", "67 years old", "44YEAR OLD", "73 y/o", "14 years old" all seen
# in real hits during manual review). Not meant to be exact -- it exists
# because a spot check of the initial keyword screen found the broader
# keywords ("missing endangered", "endangered adult", "at risk missing",
# etc.) catch a genuinely mixed population: real elderly/dementia cases,
# but also missing children (including a 14-year-old with autism, an
# 11-year-old child) and non-elderly adults (a 44-year-old woman). The
# narrower keywords (dementia, alzheimer, "silver/golden/senior alert",
# "missing elderly/senior") are much safer bets for being elderly-
# specific on their own, but this extracted age lets the broader
# categories be filtered to an actual elderly threshold too, rather than
# either keeping everything indiscriminately or discarding hits (like the
# 73yo Roberta Hart and 89yo Parma dementia cases) that the generic
# keywords legitimately did catch.
AGE_RE = re.compile(
    r"\b(\d{1,3})[\s-]*(?:y\s*/?\s*o\.?\b|yo\b|yrs?[\s.-]*old\b|years?[\s.-]*old\b)",
    re.IGNORECASE,
)

SCREEN_CSV = AMBER_RAW / "foia" / "missing_person_text_screen_2013_2024.csv"
CHECKPOINT_DIR = AMBER_RAW / "foia" / "_missing_person_geocode_checkpoints"
OUT_PATH = AMBER_RAW / "foia" / "missing_person_alerts_geocoded_2013_2024.csv"


def target_months() -> list[tuple[int, int]]:
    df = pd.read_csv(SCREEN_CSV)
    keep = df[~df["event_code"].isin(DROP_EVENT_CODES)].copy()
    keep["sent_dt"] = pd.to_datetime(keep["sent"], utc=True, errors="coerce")
    keep = keep.dropna(subset=["sent_dt"])
    pairs = sorted({(d.year, d.month) for d in keep["sent_dt"]})
    log.info("Kept %d of %d screened records (dropped %s) across %d target months",
             len(keep), len(df), sorted(DROP_EVENT_CODES), len(pairs))
    return pairs


def geocode_line(line: str) -> list[dict]:
    """Return [] if the line isn't a keyword hit; otherwise one row per
    resolved county SAME code (mirrors 02c._parse_record's county-geocode
    logic exactly, plus the keyword/event_text/event_code fields)."""
    import json
    try:
        rec = json.loads(line)
    except (ValueError, TypeError):
        return []
    orig = rec.get("originalMessage", "") or ""
    low = orig.lower()
    hit = next((kw for kw in KEYWORDS if kw in low), None)
    if hit is None:
        return []

    ev_match = EVENT_RE.search(orig)
    code_match = EVENTCODE_RE.search(orig)
    event_code = code_match.group(1).strip() if code_match else ""
    if event_code in DROP_EVENT_CODES:
        return []

    alert_id = rec.get("id") or rec.get("identifier", "")
    sent_raw = rec.get("sent", "")
    try:
        sent_utc = pd.to_datetime(sent_raw, utc=True)
    except Exception:
        sent_utc = pd.NaT
    mt_match = ipaws_amber.MSGTYPE_RE.search(orig)
    msg_type = mt_match.group(1).capitalize() if mt_match else rec.get("msgType", "")
    age_match = AGE_RE.search(orig)
    mentioned_age = int(age_match.group(1)) if age_match else None

    same_codes: set[str] = set(ipaws_amber.SAME_RE.findall(orig))
    for info in rec.get("info", []) or []:
        for area in info.get("areas", []) or []:
            for gc in area.get("geocode", []) or []:
                if gc.get("valueName", "").upper() == "SAME":
                    val = str(gc.get("value", "")).strip()
                    if re.fullmatch(r"\d{6}", val):
                        same_codes.add(val)

    rows = []
    for same in same_codes:
        if same.startswith("0") and len(same) == 6:
            fips = same[1:]
        else:
            area_descs = ipaws_amber.AREADESC_RE.findall(orig)
            fips = ipaws_amber._resolve_nonstandard_same(same, area_descs)
            if fips is None:
                continue
        rows.append({
            "alert_id": alert_id, "sent_utc": sent_utc, "fips": fips,
            "state_fips": fips[:2], "msg_type": msg_type,
            "event_code": event_code,
            "event_text": ev_match.group(1).strip() if ev_match else "",
            "matched_keyword": hit,
            "mentioned_age": mentioned_age,
        })
    if not rows:
        log.warning("  hit with no resolvable county geocode: id=%s event_code=%s", alert_id, event_code)
    return rows


def fetch_month_geocoded(year: int, month: int, session: requests.Session) -> pd.DataFrame:
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    date_filter = (
        f"sent gt '{year}-{month:02d}-01T00:00:00Z' and "
        f"sent lt '{next_year}-{next_month:02d}-01T00:00:00Z'"
    )
    params = {"$filter": date_filter, "$select": "id,sent,msgType,originalMessage,info"}
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            log.warning("    [%d] retrying %d-%02d in %ds...", attempt, year, month, delay)
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
                    rows.extend(geocode_line(raw_line))
            log.info("  %d-%02d: DONE -- %d scanned, %d county-rows, %.0fs",
                     year, month, n_scanned, len(rows), time.time() - t0)
            return pd.DataFrame(rows)
        except Exception as exc:
            log.warning("  %d-%02d: stream failed after %d records (%.0fs): %s",
                        year, month, n_scanned, time.time() - t0, exc)
    log.error("  %d-%02d: giving up after %d attempts", year, month, len(RETRY_DELAYS) + 1)
    return pd.DataFrame()


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    months = target_months()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; contact: researcher@university.edu)",
        "Accept": "application/jsonl+json",
    })

    for year, month in months:
        ckpt = CHECKPOINT_DIR / f"{year}_{month:02d}.parquet"
        if ckpt.exists():
            continue
        log.info("Streaming %d-%02d for geocoding ...", year, month)
        df_m = fetch_month_geocoded(year, month, session)
        df_m.to_parquet(ckpt, index=False)

    parts = sorted(CHECKPOINT_DIR.glob("*.parquet"))
    if not parts:
        log.warning("No checkpoint files found.")
        return
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined = combined.drop_duplicates(subset=["alert_id", "fips"])
    combined = combined.sort_values(["sent_utc", "alert_id", "fips"]).reset_index(drop=True)

    # Classification, not filtering: nothing is dropped here. A record with
    # a parsed age under 18 is a missing CHILD -- conceptually the same
    # population AMBER/CAE covers (missing/endangered minor), just issued
    # under a different generic event code instead of CAE, so it's labeled
    # accordingly rather than lumped into the elderly/Silver-Alert-style
    # "missing_person" population. Records with no parsed age (69% of the
    # total) default to "missing_person" -- there's no positive evidence
    # they're a child, and the elderly-specific keywords (dementia,
    # alzheimer, "silver/golden/senior alert", "missing elderly/senior")
    # are reliably adult/elderly on their own even without a numeric age.
    combined["population"] = "missing_person"
    combined.loc[combined["mentioned_age"] < 18, "population"] = "child_amber_adjacent"

    combined.to_csv(OUT_PATH, index=False)
    log.info("Saved -> %s (%d alert x county rows, %d unique alerts, %d counties)",
             OUT_PATH, len(combined), combined["alert_id"].nunique(), combined["fips"].nunique())
    log.info("event_code breakdown (unique alert_ids):\n%s",
              combined.groupby("event_code")["alert_id"].nunique().to_string())
    log.info("statewide (COUNTY=000) rows: %d", int(combined["fips"].str.endswith("000").sum()))
    log.info("population breakdown (unique alert_ids):\n%s",
              combined.drop_duplicates(subset=["alert_id"]).groupby("population").size().to_string())
    by_alert = combined.drop_duplicates(subset=["alert_id"])
    n_with_age = int(by_alert["mentioned_age"].notna().sum())
    log.info("mentioned_age parsed for %d/%d unique alerts (%.0f%%)",
              n_with_age, len(by_alert), 100 * n_with_age / max(len(by_alert), 1))
    if n_with_age:
        ages = by_alert["mentioned_age"].dropna()
        log.info("age distribution: min=%d, p25=%.0f, median=%.0f, p75=%.0f, max=%d; "
                  "%d (%.0f%%) are 60+, %d (%.0f%%) are under 18",
                  ages.min(), ages.quantile(.25), ages.median(), ages.quantile(.75), ages.max(),
                  int((ages >= 60).sum()), 100 * (ages >= 60).mean(),
                  int((ages < 18).sum()), 100 * (ages < 18).mean())


if __name__ == "__main__":
    main()
