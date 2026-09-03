import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import build_other_wea_hour_controls as builder
import run_same_hour_wea_sensitivity as sensitivity


def _record(alert_id, sent, *, category="Safety"):
    return {
        "id": alert_id,
        "sent": sent,
        "status": "Actual",
        "msgType": "Alert",
        "scope": "Public",
        "originalMessage": (
            "<alert><msgType>Alert</msgType><category>"
            f"{category}</category><parameter><valueName>CMAMtext</valueName>"
            "<value>Public warning</value></parameter></alert>"
        ),
    }


def test_aggregate_hour_controls_keeps_all_and_non_weather_counts(monkeypatch):
    monkeypatch.setattr(
        builder,
        "parse_same_rows",
        lambda rec: [{"fips": "06001"}],
    )
    records = [
        _record("safety-1", "2024-01-02T07:30:00Z"),
        _record("met-1", "2024-01-02T07:30:00Z", category="Met"),
    ]

    out, summary = builder.aggregate_hour_controls(
        records,
        {"06001": "America/New_York"},
        {"06": ["06001"]},
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-03", tz="UTC"),
    )

    hit = out.loc[(out["fips"] == "06001") & (out["hour"] == 2)].iloc[0]
    assert hit["all_wea_hour_count"] == 2
    assert hit["non_weather_wea_hour_count"] == 1
    assert hit["all_wea_hour_alert"] == 1
    assert hit["non_weather_wea_hour_alert"] == 1
    assert summary.set_index("reason").loc["weather_excluded", "count"] == 1


def test_attach_hour_wea_controls_adds_direct_and_commuter_spillover():
    panel = pd.DataFrame(
        {
            "fips": ["06003", "06001"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "hour": [2, 2],
        }
    )
    controls = pd.DataFrame(
        {
            "fips": ["06001"],
            "date": pd.to_datetime(["2024-01-02"]),
            "hour": [2],
            "all_wea_hour_alert": [1],
            "all_wea_hour_count": [2],
            "non_weather_wea_hour_alert": [1],
            "non_weather_wea_hour_count": [1],
        }
    )
    weights = pd.DataFrame(
        {
            "fips_home": [6001, 6003],
            "fips_work": [6003, 6001],
            "weight": [0.4, 0.2],
        }
    )

    out = sensitivity.attach_hour_wea_controls(panel, controls, weights=weights)

    work_county = out[out["fips"] == "06003"].iloc[0]
    home_county = out[out["fips"] == "06001"].iloc[0]
    assert work_county["all_wea_hour_alert"] == 0
    assert work_county["all_wea_hour_spillover"] == 0.4
    assert work_county["non_weather_wea_hour_spillover"] == 0.4
    assert home_county["all_wea_hour_spillover"] == 0
