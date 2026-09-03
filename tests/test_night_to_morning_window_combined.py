import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_night_to_morning_window_combined as combined


def test_attach_combined_night_alert_deduplicates_county_dates(monkeypatch):
    monkeypatch.setattr(
        combined.combined_loader,
        "load_combined_alerts",
        lambda **_: pd.DataFrame(
            {
                "fips": ["06001", "06001"],
                "effective_crash_date": ["2024-01-02", "2024-01-02"],
                "night_alert": [1, 1],
            }
        ),
    )
    grid = pd.DataFrame(
        {
            "fips": ["06001", "06001", "06001"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )

    out = combined.attach_combined_night_alert(grid)

    assert out["night_alert"].tolist() == [0, 1, 0]
    assert out["night_alert_lag1"].tolist() == [0, 0, 1]
