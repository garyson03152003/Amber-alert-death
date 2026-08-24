import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "traffic_volume"))

from build_traffic_outcomes import build_station_day_outcomes, build_event_study_hours


def _row(hour, volume, date="2024-01-02"):
    return {
        "state_fips": "01", "station_id": "000001", "county_fips": "01001",
        "date": date, "hour": hour, "traffic_volume": volume,
        "latitude": 30.0, "longitude": -87.0, "source": "TMAS",
    }


def test_station_day_outcomes_windows_and_missing_hours():
    station_hour = pd.DataFrame([
        _row(4, 100), _row(5, 10), _row(6, 20), _row(9, 30),
        _row(10, 40), _row(15, 50), _row(16, 60), _row(18, 70), _row(23, 5),
    ])
    out = build_station_day_outcomes(station_hour)
    row = out.iloc[0]
    assert row["total_volume"] == sum([100, 10, 20, 30, 40, 50, 60, 70, 5])
    assert row["vol_05_10"] == 10 + 20 + 30  # hours 5,6,9
    assert row["vol_07_10"] == 30  # hour 9 only
    assert row["vol_10_16"] == 40 + 50  # hours 10, 15
    assert row["vol_16_19"] == 60 + 70  # hours 16, 18
    assert row["n_hours_reported"] == 9


def test_station_day_outcomes_all_missing_window_is_nan():
    station_hour = pd.DataFrame([_row(0, 5), _row(1, 6)])
    out = build_station_day_outcomes(station_hour)
    row = out.iloc[0]
    assert pd.isna(row["vol_05_10"])
    assert row["total_volume"] == 11


def test_event_study_hours_filters_to_exposed_stations_within_window():
    station_hour = pd.DataFrame([_row(h, h) for h in range(24)])
    treatment = pd.DataFrame([{
        "state_fips": "01", "station_id": "000001", "county_fips": "01001",
        "date": "2024-01-02", "night_alert_ct": 1, "alert_hour_local": 23,
        "exposure_class": "direct",
    }, {
        "state_fips": "01", "station_id": "000002", "county_fips": "01003",
        "date": "2024-01-02", "night_alert_ct": 0, "alert_hour_local": pd.NA,
        "exposure_class": "clean_control",
    }])
    out = build_event_study_hours(station_hour, treatment)
    assert set(out["station_id"]) == {"000001"}
    assert (out["hour"] - 23 == out["event_hour"]).all()
    assert out["event_hour"].abs().max() <= 12
