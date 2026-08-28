import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "traffic_volume"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from estimate_traffic_effects import run_station_day_model


def _synthetic_panel(true_beta=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(50):
        county = f"{s % 10:05d}"
        base = rng.uniform(4, 8)
        for d in range(30):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            alert = 1 if (s % 10 == 0 and d == 15) else 0
            vol = np.exp(base + true_beta * alert + rng.normal(0, 0.1))
            rows.append({
                "state_fips": "01", "station_id": f"{s:06d}", "county_fips": county,
                "date": date, "total_volume": vol,
                "night_alert_ct": alert, "spillover_share_ct": 0.0,
                "exposure_class": "direct" if alert else "clean_control",
            })
    return pd.DataFrame(rows)


def test_station_day_model_recovers_direct_effect_sign_and_significance():
    panel = _synthetic_panel(true_beta=0.05)
    rows = run_station_day_model(panel, "total_volume")
    est = next(r for r in rows if r["record_type"] == "estimate" and r["term"] == "night_alert_ct")
    assert est["status"] == "ok"
    assert est["beta"] > 0
    assert est["pvalue"] < 0.05


def test_station_day_model_skips_when_no_variation():
    panel = _synthetic_panel(true_beta=0.0)
    panel["night_alert_ct"] = 0
    rows = run_station_day_model(panel, "total_volume")
    assert rows[0]["status"] == "skipped"


def test_station_day_model_handles_missing_outcome_column_values():
    panel = _synthetic_panel(true_beta=0.05)
    panel.loc[panel.index[:10], "total_volume"] = np.nan
    rows = run_station_day_model(panel, "total_volume")
    est = next(r for r in rows if r["record_type"] == "fit_status")
    assert est["input_n"] <= len(panel) - 10


def test_station_day_model_drops_rows_with_missing_county_fips():
    panel = _synthetic_panel(true_beta=0.05)
    panel.loc[panel.index[:5], "county_fips"] = None
    rows = run_station_day_model(panel, "total_volume")
    status_row = next(r for r in rows if r["record_type"] == "fit_status")
    assert status_row["input_n"] <= len(panel) - 5
    assert status_row["status"] in {"ok", "partial"}
