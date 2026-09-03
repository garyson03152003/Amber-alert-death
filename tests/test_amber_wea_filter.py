import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_state_dot_analysis_fixed as fixed_runner


def test_loader_excludes_alerts_blocked_from_wea(monkeypatch, tmp_path):
    raw = tmp_path / "amber" / "foia"
    raw.mkdir(parents=True)
    pd.DataFrame([
        {
            "alert_id": "kept",
            "sent_utc": "2024-06-03T18:00:00Z",
            "fips": "01001",
            "state_fips": "01",
            "msg_type": "Alert",
        },
        {
            "alert_id": "blocked",
            "sent_utc": "2024-06-03T19:00:00Z",
            "fips": "01001",
            "state_fips": "01",
            "msg_type": "Alert",
        },
    ]).to_csv(raw / "openfema_ipaws_alerts_2013_2024.csv", index=False)
    pd.DataFrame([
        {"alert_id": "kept", "cmas_blocked": False},
        {"alert_id": "blocked", "cmas_blocked": True},
    ]).to_csv(raw / "openfema_ipaws_alerts_2013_2024_wea_vetting.csv", index=False)
    monkeypatch.setattr(fixed_runner, "DATA_RAW", tmp_path)
    monkeypatch.setattr(
        fixed_runner,
        "county_timezone_map",
        lambda _path: {"01001": "America/Chicago"},
    )

    out = fixed_runner.load_verified_alerts(window="day", detail=True)

    assert out["alert_id"].tolist() == ["kept"]
