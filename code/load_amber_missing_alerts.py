"""Load the combined AMBER + missing-person/Silver treatment dataset.

The transformation mirrors ``run_state_dot_analysis_fixed.load_verified_alerts``
for timezone conversion, statewide expansion, and night-date assignment, but
reads the reviewed combined file produced by ``build_amber_missing_dataset``.
The loader defaults to retaining Alert, Update, and Cancel because all three
message types are phone-delivered for the records in this treatment file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import AMBER_RAW, DATA_PROC, DATA_RAW
from utils import get_logger

log = get_logger("amber_missing_alerts")

COMBINED_PATH = AMBER_RAW / "foia" / "openfema_ipaws_alerts_amber_missing_2013_2024.csv"
AMBER_SOURCE_PATH = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2024.csv"


def _apply_amber_routing_audit(alerts: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing CMAS audit to AMBER rows when its sidecar exists."""
    if alerts.empty or "source" not in alerts.columns:
        return alerts
    amber = alerts.loc[alerts["source"].eq("amber_cae")].copy()
    other = alerts.loc[~alerts["source"].eq("amber_cae")].copy()
    if not amber.empty:
        amber = base._exclude_cmas_blocked_alerts(amber, AMBER_SOURCE_PATH)
    return pd.concat([amber, other], ignore_index=True)


def _prepare_combined_rows(path: Path | None = None, *, include_cancel: bool = True) -> pd.DataFrame:
    if path is None:
        path = COMBINED_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"combined AMBER/missing-person treatment not found at {path}; "
            "run code/build_amber_missing_dataset.py first"
        )
    alerts = pd.read_csv(path)
    required = {"alert_id", "sent_utc", "fips", "state_fips", "msg_type"}
    missing = required - set(alerts.columns)
    if missing:
        raise ValueError(f"combined treatment missing columns: {sorted(missing)}")
    alerts = alerts.copy()
    if "source" not in alerts.columns:
        alerts["source"] = "amber_cae"
    if "alert_family" not in alerts.columns:
        alerts["alert_family"] = "amber"
    alerts = _apply_amber_routing_audit(alerts)
    allowed = {"Alert", "Update", "Cancel"} if include_cancel else {"Alert", "Update"}
    alerts["msg_type"] = alerts["msg_type"].astype(str).str.capitalize()
    alerts = alerts[alerts["msg_type"].isin(allowed)].copy()

    alerts["fips"] = alerts["fips"].astype(str).str.zfill(5)
    alerts = alerts[alerts["fips"].str.fullmatch(r"\d{5}")].copy()
    alerts["original_fips"] = alerts["fips"]
    alerts["geo_scope"] = np.where(
        alerts["fips"].str.fullmatch(r"\d{2}000"), "statewide_same", "county_same"
    )
    alerts["state_fips"] = alerts["fips"].str[:2]
    alerts = base._expand_statewide_rows(alerts)
    alerts = alerts[alerts["state_fips"].isin(base.NATIONWIDE_STATE_TIMEZONE)].copy()
    return alerts


def load_combined_alerts(
    *,
    window: str = "night",
    detail: bool = False,
    include_cancel: bool = True,
    night_start: int = base.NIGHT_START_HOUR,
    night_end: int = base.NIGHT_END_HOUR,
) -> pd.DataFrame:
    """Return combined treatment exposure in a local-time window.

    ``window="night"`` keeps [``night_start``, 24) U [0, ``night_end``) and
    shifts evening alerts to the following driving date.  ``window="day"``
    keeps the complement and assigns alerts to their same calendar date.
    With ``detail=True`` one row per alert x county is returned; otherwise the
    result is one row per county/effective date with a treatment flag.
    """
    if window not in {"night", "day"}:
        raise ValueError(f"window must be 'night' or 'day', got {window!r}")
    if not night_end < night_start <= 23:
        raise ValueError(
            f"night_start must be in ({night_end}, 23], got {night_start!r}"
        )

    alerts = _prepare_combined_rows(include_cancel=include_cancel)
    timezone_map = base.county_timezone_map(DATA_PROC / "county_pop_centroids.parquet")
    alerts["tz_name"] = alerts["fips"].map(timezone_map)
    missing_timezone = alerts.loc[alerts["tz_name"].isna(), "fips"].drop_duplicates().tolist()
    if missing_timezone:
        raise ValueError(
            "county-level IANA timezone missing; refusing state fallback for FIPS: "
            + ", ".join(missing_timezone[:10])
        )

    utc = pd.to_datetime(alerts["sent_utc"], utc=True, errors="coerce")
    alerts["sent_local"] = pd.NaT
    alerts["hour_local"] = pd.NA
    for tz_name, idx in alerts.groupby("tz_name").groups.items():
        local = utc.loc[idx].dt.tz_convert(pytz.timezone(tz_name))
        alerts.loc[idx, "sent_local"] = local.dt.tz_localize(None)
        alerts.loc[idx, "hour_local"] = local.dt.hour

    alerts = alerts.dropna(subset=["sent_local", "hour_local"]).copy()
    alerts["hour_local"] = alerts["hour_local"].astype(int)
    is_night = (alerts["hour_local"] >= night_start) | (alerts["hour_local"] < night_end)
    alerts = alerts[is_night if window == "night" else ~is_night].copy()
    alerts["alert_date"] = pd.to_datetime(alerts["sent_local"]).dt.normalize()
    alerts["effective_crash_date"] = alerts["alert_date"]
    if window == "night":
        evening = alerts["hour_local"] >= night_start
        alerts.loc[evening, "effective_crash_date"] += pd.Timedelta(days=1)

    if detail:
        return alerts.reset_index(drop=True)

    flag = "night_alert" if window == "night" else "day_alert"
    out = alerts.groupby(["fips", "effective_crash_date"], as_index=False).agg(
        n_alerts=("alert_id", "nunique"),
        geo_scopes=("geo_scope", lambda values: "+".join(sorted(set(values)))),
        alert_families=("alert_family", lambda values: "+".join(sorted(set(values)))),
    )
    out[flag] = 1
    log.info(
        "Combined county-level %s-alert county-dates: %s (include_cancel=%s)",
        window, f"{len(out):,}", include_cancel,
    )
    return out


def load_combined_night_alerts(*, detail: bool = False, include_cancel: bool = True) -> pd.DataFrame:
    """Thin wrapper for the combined night window."""
    return load_combined_alerts(window="night", detail=detail, include_cancel=include_cancel)


if __name__ == "__main__":
    result = load_combined_alerts(window="night", detail=False)
    print(result.head().to_string(index=False))
