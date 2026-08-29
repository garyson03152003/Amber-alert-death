"""
02f_geocode_missing_person_alerts.py — Attach county FIPS to the Silver
Alert / missing-endangered-person text-screening hits found by
02e_fetch_missing_person_text_screen.py, and drop the two categories of
hit that shouldn't be treated as new records.

Why this second pass is needed
--------------------------------
02e stored only a 400-character snippet of each hit's originalMessage
(enough to confirm the keyword match and pull the <event>/<eventCode>
tags near the top of the CAP message, but not enough to reach the
<info><area><geocode> block the SAME/county code lives in -- verified
directly: 0 of 1,428 kept snippets contain the string "SAME"). Getting
county geography requires the full message body, which means
re-streaming -- but only for the specific (year, month) periods that
actually contain a kept hit, not the full 132-month archive again.

Dropped before re-fetching (see missing_person_text_screen_2013_2024.csv
and its commit message for the full breakdown):
  - event_code == 'ADR' (Administrative Message): FEMA's own NWEM event
    code glossary marks this EAS & NWEM only -- not WEA-eligible, so it
    doesn't carry the phone-alert mechanism this repo's analyses are
    built around.
  - event_code == 'CAE' (Child Abduction Emergency): these are AMBER
    alerts that happen to also contain one of the missing-person
    keywords (plausible, since real AMBER text describes a missing
    child using similar language) -- they are already in
    02c_fetch_openfema_ipaws.py's dataset, not a new population, and
    keeping them here would double-count the same events under a
    different label.

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
  event_text, matched_keyword
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

DROP_EVENT_CODES = {"ADR", "CAE"}

SCREEN_CSV = AMBER_RAW / "foia" / "missing_person_text_screen_2013_2024.csv"
CHECKPOINT_DIR = AMBER_RAW / "foia" / "_missing_person_geocode_checkpoints"
OUT_PATH = AMBER_RAW / "foia" / "missing_person_alerts_geocoded_2013_2024.csv"


def target_months() -> list[tuple[int, int]]:
    df = pd.read_csv(SCREEN_CSV)
    keep = df[~df["event_code"].isin(DROP_EVENT_CODES)].copy()
    keep["sent_dt"] = pd.to_datetime(keep["sent"], utc=True, errors="coerce")
    keep = keep.dropna(subset=["sent_dt"])
    pairs = sorted({(d.year, d.month) for d in keep["sent_dt"]})
    log.info("Kept %d of %d screened records (dropped ADR/CAE) across %d target months",
             len(keep), len(df), len(pairs))
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
    combined.to_csv(OUT_PATH, index=False)
    log.info("Saved -> %s (%d alert x county rows, %d unique alerts, %d counties)",
             OUT_PATH, len(combined), combined["alert_id"].nunique(), combined["fips"].nunique())
    log.info("event_code breakdown (unique alert_ids):\n%s",
              combined.groupby("event_code")["alert_id"].nunique().to_string())
    log.info("statewide (COUNTY=000) rows: %d", int(combined["fips"].str.endswith("000").sum()))


if __name__ == "__main__":
    main()
