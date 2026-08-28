"""Attach the verified nighttime AMBER-alert treatment to the TMAS
station-hour panel (Task 2 of TRAFFIC_VOLUME_INSTRUCTIONS.md).

Reuses -- rather than reimplements -- the same verified treatment logic
already validated for the crash-panel analysis:

  - ``load_verified_night_alerts`` (run_state_dot_analysis_fixed.py) for the
    local-time, DST-aware alert timing and the "verified night alert" county
    definition (>=22:00 or <06:00 local, Cancel messages excluded).
  - ``build_commuter_spillover`` / ``add_spillover_classes``
    (state_dot_analysis_core.py) for the commuter-flow spillover share and
    the direct/spillover/clean_control exposure labeling.

Each station is attached to its county (``county_fips``, from the station
panel). Direct county exposure and commuter spillover are kept as separate
columns per the instructions -- the main traffic-volume treatment is direct
exposure; spillover is analyzed separately, never silently combined.

Output: data/processed/traffic/tmas_station_day_treatment.parquet
  one row per (station_id, date) with:
    night_alert_ct       -- 1 if this county had a verified night alert whose
                             effective_crash_date is this date, else 0
    n_alerts_ct          -- count of verified night alerts that night
    alert_time_local     -- local timestamp of the first verified night alert
                             that night (NaT if none)
    alert_hour_local     -- local hour (22-23 or 0-5) of that first alert
    spillover_share_ct   -- commuter-weighted spillover exposure share
    exposure_class       -- "direct" / "spillover" / "clean_control"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger
from state_dot_analysis_core import build_commuter_spillover, add_spillover_classes
from run_state_dot_analysis_fixed import load_verified_night_alerts

log = get_logger("merge_alert_treatment")

ROOT = Path(__file__).resolve().parent.parent.parent
STATION_HOUR_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_hour.parquet"
FLOWS_PATH = ROOT / "data" / "processed" / "commuting" / "county_commuting_weights.parquet"
OUT_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_day_treatment.parquet"


def build_station_day_treatment(station_hour: pd.DataFrame) -> pd.DataFrame:
    """One row per (station_id, county_fips, date) with the alert treatment attached."""
    station_days = (
        station_hour[["state_fips", "station_id", "county_fips", "date"]]
        .drop_duplicates()
        .copy()
    )
    station_days["date"] = pd.to_datetime(station_days["date"]).dt.normalize()

    alert_detail = load_verified_night_alerts(detail=True)
    if alert_detail.empty:
        station_days["night_alert_ct"] = 0
        station_days["n_alerts_ct"] = 0
        station_days["alert_time_local"] = pd.NaT
        station_days["alert_hour_local"] = pd.NA
    else:
        first_alert = (
            alert_detail.sort_values("sent_local")
            .groupby(["fips", "effective_crash_date"], as_index=False)
            .agg(
                n_alerts_ct=("alert_id", "nunique"),
                alert_time_local=("sent_local", "first"),
                alert_hour_local=("hour_local", "first"),
            )
        )
        first_alert = first_alert.rename(columns={
            "fips": "county_fips", "effective_crash_date": "date",
        })
        first_alert["date"] = pd.to_datetime(first_alert["date"]).dt.normalize()
        station_days = station_days.merge(
            first_alert, on=["county_fips", "date"], how="left"
        )
        station_days["night_alert_ct"] = station_days["n_alerts_ct"].notna().astype(int)
        station_days["n_alerts_ct"] = station_days["n_alerts_ct"].fillna(0).astype(int)

    flows = pd.read_parquet(FLOWS_PATH) if FLOWS_PATH.is_file() else pd.DataFrame()
    night_alerts_daily = load_verified_night_alerts(detail=False)
    spillover = build_commuter_spillover(night_alerts_daily, flows)
    if not spillover.empty:
        spillover = spillover.rename(columns={"fips": "county_fips"})
        station_days = station_days.merge(
            spillover[["county_fips", "effective_crash_date", "spillover_share"]]
            .rename(columns={"effective_crash_date": "date"}),
            on=["county_fips", "date"], how="left",
        )
    else:
        station_days["spillover_share"] = 0.0

    # add_spillover_classes expects a `night_alert` 0/1 column; reuse it
    # verbatim, then rename outputs back to the traffic-volume schema so the
    # direct-vs-spillover distinction stays explicit.
    station_days["night_alert"] = station_days["night_alert_ct"]
    station_days = add_spillover_classes(station_days)
    station_days = station_days.rename(columns={"spillover_share": "spillover_share_ct"})
    station_days = station_days.drop(columns=["night_alert", "spillover_commuters", "log_spillover_commuters", "clean_control"])

    return station_days


def main() -> None:
    if not STATION_HOUR_PATH.is_file():
        raise FileNotFoundError(
            f"station-hour panel not found at {STATION_HOUR_PATH}; run build_station_hour_panel.py first"
        )
    station_hour = pd.read_parquet(STATION_HOUR_PATH)
    treatment = build_station_day_treatment(station_hour)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    treatment.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s station-day treatment rows -> %s", f"{len(treatment):,}", OUT_PATH)
    log.info(
        "Exposure classes: %s",
        treatment["exposure_class"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
