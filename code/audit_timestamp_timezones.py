"""Audit every state source for the UTC-date bug.

The bug
-------
A source that publishes an *instant* (ArcGIS epoch milliseconds, or ISO-8601
with a "Z") must be converted to local time before its calendar date is
taken. US zones are behind UTC, so an evening crash has already rolled past
UTC midnight: taking the date straight off the UTC value files it under the
following day. Confirmed in Delaware (16.9% of crashes shifted) and Utah
(32.8%).

Why field names cannot answer this
----------------------------------
Many builders read a column called CRASH_DATE, Crash_Date or Date and parse
it with unit="ms". That is only safe if the value is *date-only* -- i.e. an
epoch pinned to UTC midnight. If it carries a real time-of-day it is an
instant and the same bug applies. The two are indistinguishable by name, so
this audit inspects the values:

    every epoch is an exact multiple of 86,400,000 ms  ->  date-only, SAFE
    otherwise                                          ->  a real instant

A real instant then splits again, and this is the trap: an ArcGIS service may
publish genuine UTC, or it may publish *local wall-clock stored as if it were
UTC* -- a very common misconfiguration. The two are byte-identical. Blindly
converting the second kind to local time subtracts the offset a second time
and creates the very bug this audit is looking for.

They are told apart by the shape of the day. Crash counts always peak in the
afternoon/early evening and trough before dawn, so:

    raw UTC hours already look like crashes   ->  local-stored-as-UTC, SAFE
    converted-to-local hours look like crashes ->  genuine UTC, AFFECTED

Massachusetts is a live example of the first kind: its raw hours peak at
15-17 (a normal crash curve) and converting to Eastern would move the peak
to noon, which no real crash dataset shows. Its existing builder is correct
and must be left alone.

For genuinely affected sources the audit reports the exact share of rows
whose calendar date changes under correct conversion.

DST is handled throughout by converting with a real timezone
(``tz_convert``), never a fixed offset: the rollover hour differs between
standard and daylight time (e.g. Eastern shifts at 19:00 EST but 20:00 EDT),
so a fixed-offset correction would itself be wrong for half the year.

FARS is excluded by construction: it stores YEAR/MONTH/DAY/HOUR as separate
integer fields already in local time, so it has no instant to misinterpret.

Usage
-----
    python3 code/audit_timestamp_timezones.py --year 2019
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

log = get_logger("timestamp_audit")

MS_PER_DAY = 86_400_000

# source key -> (module, timestamp column candidates, IANA timezone)
AUDIT_TARGETS = {
    "CT":        ("build_connecticut_uconn",   ("CrashDate",),                    "America/New_York"),
    "FL":        ("build_florida_fdot",        ("CRASH_DATE",),                   "America/New_York"),
    "HI":        ("build_hawaii_dot",          ("Crash_Date",),                   "Pacific/Honolulu"),
    "INMPO":     ("build_indianapolis_mpo",    ("Date",),                         "America/Indiana/Indianapolis"),
    "NC":        ("build_northcarolina_ncdot", ("Date",),                         "America/New_York"),
    "OR":        ("build_oregon_odot",         ("CRASH_DT",),                     "America/Los_Angeles"),
    "VA":        ("build_virginia_vdot",       ("CRASH_DT",),                     "America/New_York"),
    "NV":        ("build_nevada_ndot",         ("Crash_Date",),                   "America/Los_Angeles"),
    "IDCOMPASS": ("build_idaho_compass",       ("accident_date",),                "America/Boise"),
    "MA":        ("build_massachusetts_massdot", ("CRASH_DATE", "CRASH_DATETIME"), "America/New_York"),
    "UT":        ("build_utah_udot",           ("CRASH_DATETIME",),               "America/Denver"),
}


def _diurnal_plausible(hours: pd.Series) -> tuple[bool, int, int]:
    """Does this hour distribution look like real crash data?"""
    counts = hours.value_counts().reindex(range(24), fill_value=0).sort_index()
    if counts.sum() == 0:
        return False, -1, -1
    peak, trough = int(counts.idxmax()), int(counts.idxmin())
    return (14 <= peak <= 19 and 1 <= trough <= 5), peak, trough


def classify_instant_timezone(values: pd.Series, tz: str, *, kind: str = "epoch_ms") -> dict:
    """Decide whether an instant column is genuine UTC or local-stored-as-UTC."""
    if kind == "epoch_ms":
        utc = pd.to_datetime(pd.to_numeric(values, errors="coerce"),
                             unit="ms", errors="coerce", utc=True)
    else:
        utc = pd.to_datetime(values, errors="coerce", utc=True)
    utc = utc.dropna()
    if utc.empty:
        return {"verdict": "no_data"}
    local = utc.dt.tz_convert(tz)
    raw_ok, raw_peak, raw_trough = _diurnal_plausible(utc.dt.hour)
    loc_ok, loc_peak, loc_trough = _diurnal_plausible(local.dt.hour)

    if raw_ok and not loc_ok:
        verdict = "local_stored_as_utc_SAFE"
    elif loc_ok and not raw_ok:
        verdict = "genuine_utc_AFFECTED"
    else:
        verdict = "AMBIGUOUS_needs_review"
    return {
        "verdict": verdict,
        "raw_utc_peak": raw_peak, "raw_utc_trough": raw_trough,
        "local_peak": loc_peak, "local_trough": loc_trough,
    }


# A date-only column may be anchored anywhere, not just UTC midnight. Two
# real examples: Florida pins every crash to 05:00 UTC (= EST midnight, and
# its service declares "Eastern Standard Time", DST off), and Oregon pins
# every crash to 12:00 UTC (noon). Testing only for `epoch % 86,400,000 == 0`
# therefore misses both and misreports them as instants.
#
# The property that actually matters is whether time-of-day carries any
# information. A handful of distinct values means it does not -- the column
# is a date wearing a timestamp's clothes. The allowance of a few values
# covers a source that anchored on *local* midnight and so alternates
# between two UTC offsets across DST.
MAX_DISTINCT_TIMES_FOR_DATE_ONLY = 3
# A handful of distinct times is only meaningful with enough rows behind it,
# and each anchor must carry a real share of the data. Otherwise a mostly-
# date-only column with a few genuinely-timed rows -- or a tiny sample --
# would be waved through. When in doubt this falls back to "instant", which
# flags the source for review rather than silently declaring it safe.
MIN_ROWS_FOR_DATE_ONLY = 50
MIN_ANCHOR_SHARE = 0.05


def classify_epoch_ms(values: pd.Series) -> dict:
    """Decide whether an epoch-ms column is date-only or a true instant."""
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return {"verdict": "no_data", "n": 0}
    times_of_day = nums % MS_PER_DAY
    n_distinct = int(times_of_day.nunique())
    shares = times_of_day.value_counts(normalize=True)
    date_only = (
        n_distinct <= MAX_DISTINCT_TIMES_FOR_DATE_ONLY
        and len(nums) >= MIN_ROWS_FOR_DATE_ONLY
        and float(shares.min()) >= MIN_ANCHOR_SHARE
    )
    anchors = sorted(round(v / 3_600_000, 2) for v in times_of_day.unique()[:5])
    return {
        "verdict": "date_only_SAFE" if date_only else "instant_AFFECTED",
        "n": int(len(nums)),
        "n_distinct_times_of_day": n_distinct,
        "anchor_hours_utc": anchors if date_only else None,
        "share_at_utc_midnight": round(float((times_of_day == 0).mean()), 6),
    }


def date_shift_share(values: pd.Series, tz: str, *, kind: str = "epoch_ms") -> float:
    """Share of rows whose calendar date changes under correct local conversion."""
    if kind == "epoch_ms":
        utc = pd.to_datetime(pd.to_numeric(values, errors="coerce"),
                             unit="ms", errors="coerce", utc=True)
    else:
        utc = pd.to_datetime(values, errors="coerce", utc=True)
    utc = utc.dropna()
    if utc.empty:
        return 0.0
    # tz_convert applies the correct DST offset per timestamp.
    local = utc.dt.tz_convert(tz)
    return float((utc.dt.date != local.dt.date).mean())


def audit_source(key: str, year: int) -> dict:
    module, columns, tz = AUDIT_TARGETS[key]
    try:
        mod = importlib.import_module(module)
    except Exception as exc:                                   # noqa: BLE001
        return {"source": key, "verdict": "import_failed", "detail": str(exc)}

    fetch = getattr(mod, "fetch_year", None)
    if fetch is None:
        return {"source": key, "verdict": "no_fetch_year"}

    session = requests.Session()
    try:
        raw = fetch(session, year)
    except Exception as exc:                                   # noqa: BLE001
        return {"source": key, "verdict": "fetch_failed", "detail": str(exc)}
    finally:
        session.close()

    if raw is None or not isinstance(raw, pd.DataFrame) or len(raw) == 0:
        return {"source": key, "verdict": "no_usable_frame",
                "detail": type(raw).__name__}

    col = next((c for c in columns if c in raw.columns), None)
    if col is None:
        return {"source": key, "verdict": "column_missing",
                "detail": f"none of {columns} in {list(raw.columns)[:12]}"}

    info = classify_epoch_ms(raw[col])
    out = {"source": key, "year": year, "column": col, "tz": tz, **info}
    if info["verdict"] == "instant_AFFECTED":
        tzinfo = classify_instant_timezone(raw[col], tz)
        out.update({k: v for k, v in tzinfo.items() if k != "verdict"})
        out["verdict"] = tzinfo["verdict"]
        if tzinfo["verdict"] == "genuine_utc_AFFECTED":
            out["share_dates_shifted"] = round(date_shift_share(raw[col], tz), 4)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--sources", nargs="*", default=list(AUDIT_TARGETS),
                        choices=list(AUDIT_TARGETS))
    args = parser.parse_args(argv)

    rows = []
    for key in args.sources:
        log.info("=== auditing %s (%d) ===", key, args.year)
        row = audit_source(key, args.year)
        rows.append(row)
        log.info("  %s", row)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "output" / "tables" / "timestamp_timezone_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)
    affected = df[df.get("verdict", pd.Series(dtype=str)).isin(
        ["genuine_utc_AFFECTED", "AMBIGUOUS_needs_review"])]
    if len(affected):
        log.warning("AFFECTED sources:\n%s", affected.to_string(index=False))
    else:
        log.info("No affected sources in this run.")


if __name__ == "__main__":
    main()
