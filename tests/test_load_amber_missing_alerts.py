import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import load_amber_missing_alerts as loader


def test_combined_loader_can_include_phone_delivered_cancellations(monkeypatch, tmp_path):
    path = tmp_path / "combined.csv"
    pd.DataFrame(
        {
            "alert_id": ["a", "cancel"],
            "sent_utc": ["2024-01-02T08:00:00Z", "2024-01-02T09:00:00Z"],
            "fips": ["06001", "06001"],
            "state_fips": ["06", "06"],
            "msg_type": ["Alert", "Cancel"],
            "event_code": ["CAE", "LAE"],
            "event_text": ["Child Abduction Emergency", "Missing Person"],
            "source": ["amber_cae", "ipaws_wea_screen"],
            "alert_family": ["amber", "missing_person"],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(loader, "COMBINED_PATH", path)
    monkeypatch.setattr(loader, "AMBER_SOURCE_PATH", tmp_path / "amber.csv")
    monkeypatch.setattr(loader, "DATA_PROC", tmp_path)
    monkeypatch.setattr(loader.base, "county_timezone_map", lambda _path: {"06001": "America/Los_Angeles"})
    monkeypatch.setattr(loader.base, "_expand_statewide_rows", lambda alerts: alerts)

    with_cancel = loader.load_combined_alerts(window="night", detail=True, include_cancel=True)
    without_cancel = loader.load_combined_alerts(window="night", detail=True, include_cancel=False)

    assert set(with_cancel["msg_type"]) == {"Alert", "Cancel"}
    assert set(without_cancel["msg_type"]) == {"Alert"}


def test_combined_loader_retains_family_metadata_in_detail(monkeypatch, tmp_path):
    path = tmp_path / "combined.csv"
    pd.DataFrame(
        {
            "alert_id": ["silver"],
            "sent_utc": ["2024-01-02T08:00:00Z"],
            "fips": ["06001"],
            "state_fips": ["06"],
            "msg_type": ["Alert"],
            "event_code": ["LAE"],
            "event_text": ["Silver Alert"],
            "source": ["ipaws_wea_screen"],
            "alert_family": ["silver_alert"],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(loader, "COMBINED_PATH", path)
    monkeypatch.setattr(loader, "DATA_PROC", tmp_path)
    monkeypatch.setattr(loader.base, "county_timezone_map", lambda _path: {"06001": "America/Los_Angeles"})
    monkeypatch.setattr(loader.base, "_expand_statewide_rows", lambda alerts: alerts)

    out = loader.load_combined_alerts(window="night", detail=True)

    assert out.loc[0, "alert_family"] == "silver_alert"
    assert out.loc[0, "source"] == "ipaws_wea_screen"
