"""
load_missing_person_alerts.py
=============================================================
Adapter that converts data/raw/amber/foia/missing_person_alerts_geocoded_2013_2024.csv
(built by 02f_geocode_missing_person_alerts.py -- the Silver-Alert-type
missing-person data found by free-text screening the OpenFEMA IPAWS
archive, since no dedicated event code existed for this population
before September 2025) into the same (fips, sent_local, hour_local,
msg_type) shape run_state_dot_analysis_fixed.load_verified_alerts(detail=True)
produces for AMBER -- so the two sources can be concatenated into one
combined treatment definition for the same-hour case-crossover design.

Reuses the AMBER pipeline's exact UTC -> local-hour conversion logic
(NATIONWIDE_STATE_TIMEZONE / COUNTY_TIMEZONE_OVERRIDE state/county
timezone maps, DST-aware per-alert tz_convert) and statewide-row
expansion (_expand_statewide_rows), imported directly rather than
reimplemented, so a combined AMBER+missing-person exposure is computed
identically to how AMBER alone already is.

Excludes Cancel messages (kept as Alert/Update only), matching
load_verified_alerts's own treatment of AMBER msg_type.

population: which of 02f's population labels to include --
'missing_person' (the elderly/adult Silver-Alert-equivalent population),
'child_amber_adjacent' (missing/endangered minors caught by generic
event codes -- conceptually the same population AMBER/CAE covers), or
both (the default, matching the instruction to treat the child cases as
AMBER-equivalent exposure).
"""
import sys
from pathlib import Path

import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_RAW
from utils import get_logger

log = get_logger("missing_person_alerts")

MISSING_PERSON_PATH = DATA_RAW / "amber" / "foia" / "missing_person_alerts_geocoded_2013_2024.csv"


def load_missing_person_alert_hours(
    populations: tuple[str, ...] = ("missing_person", "child_amber_adjacent"),
) -> pd.DataFrame:
    """One row per (fips, sent_local, hour_local, msg_type) for the
    requested population(s) -- same shape as load_verified_alerts(detail=True)
    produces for AMBER, no night/day windowing (that shift is specific to
    the next-day exposure definitions in run_state_dot_analysis_fixed /
    run_night_to_morning_window; the same-hour design needs every alert's
    own local date+hour, unshifted)."""
    alerts = pd.read_csv(MISSING_PERSON_PATH, parse_dates=["sent_utc"])
    alerts = alerts[alerts["population"].isin(populations)].copy()
    n_before_msgtype = len(alerts)
    if "msg_type" in alerts.columns:
        alerts = alerts[alerts["msg_type"].isin(["Alert", "Update"])].copy()
    log.info("Missing-person alerts: %d rows (populations=%s), %d after excluding Cancel",
             n_before_msgtype, populations, len(alerts))

    alerts["fips"] = alerts["fips"].astype(str).str.zfill(5)
    alerts = alerts[alerts["fips"].str.match(r"^\d{5}$")].copy()
    alerts["state_fips"] = alerts["fips"].str[:2]
    alerts = base._expand_statewide_rows(alerts)
    alerts = alerts[alerts["state_fips"].isin(base.NATIONWIDE_STATE_TIMEZONE)].copy()
    alerts["tz_name"] = alerts["fips"].map(base.COUNTY_TIMEZONE_OVERRIDE).fillna(
        alerts["state_fips"].map(base.NATIONWIDE_STATE_TIMEZONE)
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
    log.info("Missing-person alert-hours after geography/timezone validation: %d "
             "(%d unique alert_ids, %d counties)",
             len(alerts), alerts["alert_id"].nunique(), alerts["fips"].nunique())
    return alerts[["fips", "sent_local", "hour_local", "msg_type"]].reset_index(drop=True)


def load_missing_person_night_alert_dates(
    night_start: int = base.NIGHT_START_HOUR, night_end: int = base.NIGHT_END_HOUR,
    populations: tuple[str, ...] = ("missing_person", "child_amber_adjacent"),
) -> pd.DataFrame:
    """(fips, effective_crash_date) pairs for missing-person alerts sent in
    the night window [night_start, 24) U [0, night_end) -- same semantics
    as load_verified_alerts(window="night"): an evening alert (hour_local
    >= night_start) belongs to the FOLLOWING day's driving, so its
    effective_crash_date is shifted forward one day; an early-morning
    alert (hour_local < night_end) already carries the correct date."""
    ev = load_missing_person_alert_hours(populations=populations)
    is_night = (ev["hour_local"] >= night_start) | (ev["hour_local"] < night_end)
    ev = ev[is_night].copy()
    ev["effective_crash_date"] = pd.to_datetime(ev["sent_local"]).dt.normalize()
    evening = ev["hour_local"] >= night_start
    ev.loc[evening, "effective_crash_date"] += pd.Timedelta(days=1)
    out = ev[["fips", "effective_crash_date"]].drop_duplicates().reset_index(drop=True)
    log.info("Missing-person NIGHT-window alert-dates: %d unique (fips, effective_crash_date) pairs",
             len(out))
    return out
