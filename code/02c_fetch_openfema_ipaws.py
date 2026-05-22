"""
02c_fetch_openfema_ipaws.py — Download real AMBER alert records from the
OpenFEMA IPAWS Archived Alerts API.

Strategy
--------
* Filter on `contains(originalMessage,'Child Abduction Emergency')` which
  selects CAP messages with EAS event code "CAE" (Child Abduction Emergency).
* Paginate month-by-month with $top=1000 to keep $skip values small and
  avoid server-side read timeouts that occur at skip ≥ 1000 on wide filters.
* Extract sent timestamp (UTC) and county SAME codes from each record.
* SAME codes are 6 digits (PSSCCC): P=0, SS=2-digit state FIPS, CCC=county.
  Strip leading zero to get 5-digit county FIPS.
* Deduplicate on alert id + fips pair.

Output
------
data/raw/amber/foia/openfema_ipaws_alerts_2013_2022.csv
  Columns: alert_id, sent_utc, fips, state_fips

Run
---
python code/02c_fetch_openfema_ipaws.py
"""

import re
import sys
import time
from calendar import monthrange
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import AMBER_RAW, STUDY_YEARS
from utils import get_logger

log = get_logger("02c_ipaws")

API_URL  = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts"
TOP      = 1000          # records per page (API maximum is likely 1000)
MAX_SKIP = 9000          # stop if a month somehow has >10k AMBER alerts
SLEEP_S  = 0.5          # polite pause between requests
RETRY_DELAYS = [2, 4, 8, 16]  # exponential backoff on failure

SAME_RE = re.compile(
    r"<valueName>SAME</valueName>\s*<value>(\d{6})</value>",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(session: requests.Session, params: dict, timeout: int = 40) -> dict:
    """GET with retry / exponential backoff."""
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            r = session.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            log.warning("Attempt %d failed (%s); retry in %ds", attempt, exc, delay or 0)
        except requests.exceptions.HTTPError as exc:
            log.warning("HTTP error %s on attempt %d", exc, attempt)
    raise RuntimeError(f"All retries exhausted for params={params}")


# ---------------------------------------------------------------------------
# Parse one record
# ---------------------------------------------------------------------------

def _parse_record(rec: dict) -> list[dict]:
    """
    Return a list of (alert_id, sent_utc, fips) rows — one per county in
    the SAME geocode list.  Returns [] if no SAME codes found.
    """
    alert_id = rec.get("id") or rec.get("identifier", "")
    sent_raw = rec.get("sent", "")

    # Parse sent timestamp — field is ISO-8601 UTC e.g. "2022-03-15T02:34:00Z"
    try:
        sent_utc = pd.to_datetime(sent_raw, utc=True)
    except Exception:
        sent_utc = pd.NaT

    # Extract SAME codes from raw CAP XML in originalMessage
    orig = rec.get("originalMessage", "") or ""

    # Also try the structured info array if present
    same_codes: set[str] = set(SAME_RE.findall(orig))

    # Fallback: info[].areas[].geocode[] list
    for info in rec.get("info", []) or []:
        for area in info.get("areas", []) or []:
            for gc in area.get("geocode", []) or []:
                if gc.get("valueName", "").upper() == "SAME":
                    val = str(gc.get("value", "")).strip()
                    if re.fullmatch(r"\d{6}", val):
                        same_codes.add(val)

    rows = []
    for same in same_codes:
        # SAME county code: 0SSCCC  →  FIPS = SSCCC
        if same.startswith("0") and len(same) == 6:
            fips = same[1:]           # 5-digit county FIPS
        else:
            fips = same.zfill(5)[-5:]
        state_fips = fips[:2]
        rows.append({
            "alert_id":   alert_id,
            "sent_utc":   sent_utc,
            "fips":       fips,
            "state_fips": state_fips,
        })
    return rows


# ---------------------------------------------------------------------------
# Month-level fetch
# ---------------------------------------------------------------------------

def fetch_month(year: int, month: int, session: requests.Session) -> pd.DataFrame:
    """
    Download all AMBER alert records for one calendar month.
    Returns a DataFrame with columns [alert_id, sent_utc, fips, state_fips].
    """
    _, last_day = monthrange(year, month)
    next_month  = month + 1 if month < 12 else 1
    next_year   = year if month < 12 else year + 1

    date_filter = (
        f"sent gt '{year}-{month:02d}-01T00:00:00Z' and "
        f"sent lt '{next_year}-{next_month:02d}-01T00:00:00Z'"
    )
    amber_filter = "contains(originalMessage,'Child Abduction Emergency')"
    combined = f"({amber_filter}) and ({date_filter})"

    all_rows = []
    skip = 0

    while skip <= MAX_SKIP:
        params = {
            "$top":    TOP,
            "$skip":   skip,
            "$filter": combined,
            "$select": "id,sent,originalMessage,info",
            "$orderby": "sent asc",
        }
        try:
            data = _get(session, params)
        except RuntimeError as exc:
            log.error("Giving up on %d-%02d skip=%d: %s", year, month, skip, exc)
            break

        records = data if isinstance(data, list) else data.get("IpawsArchivedAlerts", [])
        if not records:
            break

        for rec in records:
            all_rows.extend(_parse_record(rec))

        time.sleep(SLEEP_S)

        if len(records) < TOP:
            break           # last page
        skip += TOP

    if all_rows:
        return pd.DataFrame(all_rows)
    return pd.DataFrame(columns=["alert_id", "sent_utc", "fips", "state_fips"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = AMBER_RAW / "foia"
    out_dir.mkdir(parents=True, exist_ok=True)
    yr0, yr1 = min(STUDY_YEARS), max(STUDY_YEARS)
    out_path = out_dir / f"openfema_ipaws_alerts_{yr0}_{yr1}.csv"

    # Backward-compat: migrate old fixed filename if present and new one absent
    old_path = out_dir / "openfema_ipaws_alerts_2013_2022.csv"
    if old_path.exists() and not out_path.exists():
        old_path.rename(out_path)
        log.info("Renamed %s → %s", old_path.name, out_path.name)

    # Allow resuming: load already-fetched months if file exists
    already: set[tuple] = set()
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path, usecols=["sent_utc"])
            existing["sent_utc"] = pd.to_datetime(existing["sent_utc"], utc=True)
            already = {
                (d.year, d.month)
                for d in existing["sent_utc"].dropna()
            }
            log.info("Resuming — %d months already downloaded.", len(already))
        except Exception:
            pass

    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; contact: researcher@university.edu)",
        "Accept":     "application/json",
    })

    frames: list[pd.DataFrame] = []
    months = [
        (y, m)
        for y in STUDY_YEARS
        for m in range(1, 13)
        if (y, m) not in already
    ]

    log.info("Fetching %d month-batches from OpenFEMA IPAWS...", len(months))

    for year, month in tqdm(months, desc="IPAWS months"):
        df_m = fetch_month(year, month, session)
        if not df_m.empty:
            n_alerts = df_m["alert_id"].nunique()
            n_rows   = len(df_m)
            log.info("  %d-%02d: %d unique alerts, %d county rows",
                     year, month, n_alerts, n_rows)
            frames.append(df_m)
        else:
            log.info("  %d-%02d: 0 records", year, month)

    if not frames:
        log.warning("No records fetched. Check API connectivity.")
        return

    new_df = pd.concat(frames, ignore_index=True)

    # Append to existing if resuming
    if out_path.exists() and already:
        old_df = pd.read_csv(out_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate on alert_id + fips
    before = len(combined)
    combined = combined.drop_duplicates(subset=["alert_id", "fips"])
    log.info("Dedup: %d → %d rows", before, len(combined))

    combined = combined.sort_values(["sent_utc", "alert_id", "fips"]).reset_index(drop=True)
    combined.to_csv(out_path, index=False)

    n_alerts = combined["alert_id"].nunique()
    n_fips   = combined["fips"].nunique()
    log.info("Saved %s — %d records, %d unique alerts, %d counties",
             out_path, len(combined), n_alerts, n_fips)
    log.info("Year breakdown:\n%s",
             combined.assign(year=pd.to_datetime(combined["sent_utc"], utc=True).dt.year)
                     .groupby("year")["alert_id"].nunique().to_string())


if __name__ == "__main__":
    main()
