import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "traffic_volume"))

import run_state_dot_analysis_fixed as fixed_runner
import merge_alert_treatment as mat


def _station_hour(rows):
    return pd.DataFrame(rows)


def test_build_station_day_treatment_direct_exposure(monkeypatch):
    station_hour = _station_hour([
        {"state_fips": "01", "station_id": "000001", "county_fips": "01001",
         "date": "2024-01-02", "hour": 0, "traffic_volume": 10},
        {"state_fips": "01", "station_id": "000001", "county_fips": "01001",
         "date": "2024-01-03", "hour": 0, "traffic_volume": 12},
    ])

    def fake_load(*, detail=False):
        if detail:
            return pd.DataFrame({
                "alert_id": ["a1"],
                "fips": ["01001"],
                "state_fips": ["01"],
                "tz_name": ["America/Chicago"],
                "sent_local": [pd.Timestamp("2024-01-01 23:30:00")],
                "hour_local": [23],
                "effective_crash_date": [pd.Timestamp("2024-01-02")],
            })
        return pd.DataFrame({
            "fips": ["01001"], "effective_crash_date": [pd.Timestamp("2024-01-02")],
            "n_alerts": [1], "night_alert": [1],
        })

    monkeypatch.setattr(mat, "load_verified_night_alerts", fake_load)
    monkeypatch.setattr(mat.pd, "read_parquet", lambda p: pd.DataFrame())
    monkeypatch.setattr(mat, "FLOWS_PATH", Path("/nonexistent/flows.parquet"))

    out = mat.build_station_day_treatment(station_hour)
    exposed = out[out["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    clean = out[out["date"] == pd.Timestamp("2024-01-03")].iloc[0]

    assert exposed["night_alert_ct"] == 1
    assert exposed["n_alerts_ct"] == 1
    assert exposed["alert_hour_local"] == 23
    assert exposed["exposure_class"] == "direct"

    assert clean["night_alert_ct"] == 0
    assert clean["exposure_class"] == "clean_control"


def test_build_station_day_treatment_no_alerts(monkeypatch):
    station_hour = _station_hour([
        {"state_fips": "01", "station_id": "000001", "county_fips": "01001",
         "date": "2024-01-02", "hour": 0, "traffic_volume": 10},
    ])

    def fake_load(*, detail=False):
        return pd.DataFrame()

    monkeypatch.setattr(mat, "load_verified_night_alerts", fake_load)
    monkeypatch.setattr(mat.pd, "read_parquet", lambda p: pd.DataFrame())
    monkeypatch.setattr(mat, "FLOWS_PATH", Path("/nonexistent/flows.parquet"))

    out = mat.build_station_day_treatment(station_hour)
    assert (out["night_alert_ct"] == 0).all()
    assert (out["exposure_class"] == "clean_control").all()
