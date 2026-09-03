"""Build the combined AMBER + reviewed missing-person/Silver treatment file.

The monthly WEA cache is the source for the additional treatment records.
``fetch_other_wea_controls.classify_record`` is the single classification
boundary used for both sides of the design: explicit missing-person/Silver
records are promoted into treatment, while the same records are excluded from
the non-AMBER WEA control.  The existing CAE AMBER rows are retained exactly
as fetched, including their message type.

Unlike ``load_verified_alerts`` (whose historical AMBER-only default is
Alert/Update), this combined file deliberately preserves Cancel messages.
The phone-delivered cancellation is an observable WEA exposure and can be
included by the combined analysis loader with ``include_cancel=True``.

Outputs
-------
``data/raw/amber/foia/openfema_ipaws_alerts_amber_missing_2013_2024.csv``
    One row per alert x SAME geography, with source/family metadata.
``output/tables/amber_missing_alert_selection_summary.csv``
    Record- and county-row counts by source, family, and message type.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fetch_other_wea_controls as other_wea
from config import AMBER_RAW, OUTPUT_TABS
from utils import get_logger

log = get_logger("build_amber_missing_dataset")

AMBER_PATH = AMBER_RAW / "foia" / "openfema_ipaws_alerts_2013_2024.csv"
COMBINED_PATH = AMBER_RAW / "foia" / "openfema_ipaws_alerts_amber_missing_2013_2024.csv"
SUMMARY_PATH = OUTPUT_TABS / "amber_missing_alert_selection_summary.csv"


def iter_cached_records(cache_dir: Path = other_wea.MONTH_CACHE_DIR) -> Iterator[dict]:
    """Yield raw WEA archive records from completed month checkpoints."""
    for path in sorted(cache_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _event_code(classified: dict[str, object]) -> str:
    codes = classified.get("event_codes") or {}
    values = codes.get("EVENTCODE", set()) if isinstance(codes, dict) else set()
    values = sorted(str(value) for value in values if str(value).strip())
    return values[0] if values else ""


def _event_text(classified: dict[str, object]) -> str:
    codes = classified.get("event_codes") or {}
    values = codes.get("EVENT", set()) if isinstance(codes, dict) else set()
    values = sorted(str(value) for value in values if str(value).strip())
    return values[0] if values else ""


def person_rows_from_records(records: Iterable[dict]) -> pd.DataFrame:
    """Return reviewed missing-person/Silver rows, preserving cancellations.

    The returned grain is alert x numeric SAME geography.  A repeated raw
    record (as can occur when the API cache is the union of several indexed
    WEA queries) is collapsed to one alert/fips pair.
    """
    rows: list[dict[str, object]] = []
    for rec in records:
        classified = other_wea.classify_record(rec)
        if classified.get("reason") != "person_treatment":
            continue
        same_rows = other_wea.parse_same_rows(rec)
        if not same_rows:
            continue
        alert_id = str(rec.get("id") or rec.get("identifier") or "")
        if not alert_id:
            continue
        msg_type = str(classified.get("msg_type") or rec.get("msgType") or "").capitalize()
        family = str(classified.get("alert_family") or "")
        for parsed in same_rows:
            fips = str(parsed.get("fips") or "").zfill(5)
            if not fips.isdigit() or len(fips) != 5:
                continue
            rows.append({
                "alert_id": alert_id,
                "sent_utc": classified.get("sent"),
                "fips": fips,
                "state_fips": fips[:2],
                "msg_type": msg_type,
                "event_code": _event_code(classified),
                "event_text": _event_text(classified),
                "source": "ipaws_wea_screen",
                "alert_family": family,
            })
    columns = [
        "alert_id", "sent_utc", "fips", "state_fips", "msg_type",
        "event_code", "event_text", "source", "alert_family",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out = out.drop_duplicates(["alert_id", "fips"]).sort_values(
        ["sent_utc", "alert_id", "fips"]
    ).reset_index(drop=True)
    return out


def _read_amber_rows(path: Path = AMBER_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"AMBER alert file not found: {path}")
    amber = pd.read_csv(path)
    required = {"alert_id", "sent_utc", "fips", "state_fips", "msg_type"}
    missing = required - set(amber.columns)
    if missing:
        raise ValueError(f"AMBER alert file missing columns: {sorted(missing)}")
    amber = amber.copy()
    amber["fips"] = amber["fips"].astype(str).str.zfill(5)
    amber["state_fips"] = amber["state_fips"].astype(str).str.zfill(2)
    amber["event_code"] = "CAE"
    amber["event_text"] = "Child Abduction Emergency"
    amber["source"] = "amber_cae"
    amber["alert_family"] = "amber"
    return amber[
        ["alert_id", "sent_utc", "fips", "state_fips", "msg_type",
         "event_code", "event_text", "source", "alert_family"]
    ]


def build_combined_dataset(
    *,
    amber_path: Path = AMBER_PATH,
    cache_dir: Path = other_wea.MONTH_CACHE_DIR,
    output_path: Path = COMBINED_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and write the combined raw treatment dataset plus audit summary."""
    amber = _read_amber_rows(amber_path)
    person = person_rows_from_records(iter_cached_records(cache_dir))
    combined = pd.concat([amber, person], ignore_index=True)
    combined["alert_id"] = combined["alert_id"].astype(str)
    combined["fips"] = combined["fips"].astype(str).str.zfill(5)
    combined["state_fips"] = combined["state_fips"].astype(str).str.zfill(2)
    combined["sent_utc"] = pd.to_datetime(combined["sent_utc"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["sent_utc"])
    combined = combined.drop_duplicates(["alert_id", "fips"])
    combined = combined.sort_values(["sent_utc", "alert_id", "fips"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    audit = combined.copy()
    audit["msg_type"] = audit["msg_type"].astype(str).str.capitalize()
    summary = (
        audit.groupby(["source", "alert_family", "msg_type"], as_index=False)
        .agg(county_rows=("fips", "size"), unique_alerts=("alert_id", "nunique"))
        .sort_values(["source", "alert_family", "msg_type"])
        .reset_index(drop=True)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    log.info(
        "Saved combined treatment %s (%d rows, %d alerts) and audit %s",
        output_path, len(combined), combined["alert_id"].nunique(), summary_path,
    )
    return combined, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild existing output")
    args = parser.parse_args(argv)
    if COMBINED_PATH.exists() and not args.force:
        log.info("Combined treatment already exists at %s; use --force to rebuild", COMBINED_PATH)
        return
    build_combined_dataset()


if __name__ == "__main__":
    main()
