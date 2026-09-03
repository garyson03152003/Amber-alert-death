import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_night_to_morning_window as ntm


def test_other_wea_control_specs_include_binary_and_dose_variants():
    assert ntm.OTHER_WEA_CONTROL_SPECS == (
        ("binary", ("other_wea_night_alert",)),
        ("dose", ("other_wea_night_count",)),
    )


def test_attach_other_wea_control_merges_binary_and_count(monkeypatch, tmp_path):
    path = tmp_path / "other_wea_night_controls.parquet"
    pd.DataFrame(
        {
            "fips": ["01001"],
            "date": ["2024-01-02"],
            "other_wea_night_alert": [1],
            "other_wea_night_count": [3],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr(ntm, "OTHER_WEA_CONTROL_PATH", path)
    grid = pd.DataFrame({"fips": ["01001", "01003"], "date": ["2024-01-02", "2024-01-02"]})

    out = ntm.attach_other_wea_control(grid)

    assert out["other_wea_night_alert"].tolist() == [1, 0]
    assert out["other_wea_night_count"].tolist() == [3, 0]


def test_attach_other_wea_control_accepts_an_explicit_sensitivity_path(monkeypatch, tmp_path):
    path = tmp_path / "other_wea_no_weather.parquet"
    pd.DataFrame(
        {
            "fips": ["01001"],
            "date": ["2024-01-02"],
            "other_wea_night_alert": [1],
            "other_wea_night_count": [2],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr(ntm, "OTHER_WEA_CONTROL_PATH", tmp_path / "does_not_exist.parquet")
    grid = pd.DataFrame({"fips": ["01001"], "date": ["2024-01-02"]})

    out = ntm.attach_other_wea_control(grid, path=path)

    assert out.loc[0, "other_wea_night_count"] == 2
