import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_night_to_morning_window as ntm


DATE = pd.Timestamp("2013-06-01")


def _pair_dosage():
    return {
        ("2015", 2013): pd.DataFrame(
            {
                "fips_home": ["01001", "01003", "01003", "01005", "01005"],
                "fips_work": ["01001", "01001", "01003", "01001", "01005"],
                "commuter_car_miles": [2.4, 2.5, 2.4, 3.0, 4.0],
            }
        )
    }


def test_year_matched_exposure_uses_self_loop_for_own_and_alerted_origins_for_cross():
    grid = pd.DataFrame(
        {
            "fips": ["01001", "01003", "01005"],
            "date": [DATE, DATE, DATE],
            "night_alert": [1, 1, 0],
        }
    )

    out = ntm.attach_year_matched_commuter_exposure(
        grid, pair_dosages=_pair_dosage()
    ).set_index("fips")

    assert out.loc["01001", "own_driver_distance"] == pytest.approx(2.4)
    assert out.loc["01001", "cross_driver_distance"] == pytest.approx(2.5)
    assert out.loc["01003", "own_driver_distance"] == pytest.approx(2.4)
    assert out.loc["01003", "cross_driver_distance"] == 0.0
    assert out.loc["01005", "own_driver_distance"] == 0.0
    assert out.loc["01005", "cross_driver_distance"] == 0.0


def test_year_matched_exposure_preserves_grid_rows_and_existing_columns():
    grid = pd.DataFrame(
        {
            "fips": ["01005", "01001"],
            "date": [DATE, DATE],
            "night_alert": [0, 1],
            "fatals_0623": [3, 4],
        }
    )

    out = ntm.attach_year_matched_commuter_exposure(
        grid, pair_dosages=_pair_dosage()
    )

    assert out[["fips", "date"]].equals(grid[["fips", "date"]])
    assert out["fatals_0623"].tolist() == [3, 4]
    assert out["own_driver_distance"].tolist() == pytest.approx([0.0, 2.4])


def test_year_matched_exposure_selects_the_pair_dosage_for_each_crash_year():
    second_regime = {
        ("2015", 2018): _pair_dosage()[("2015", 2013)].assign(
            commuter_car_miles=lambda frame: frame["commuter_car_miles"] * 10
        )
    }
    pair_dosages = {**_pair_dosage(), **second_regime}
    grid = pd.DataFrame(
        {
            "fips": ["01001", "01001"],
            "date": [DATE, pd.Timestamp("2016-06-01")],
            "night_alert": [1, 1],
        }
    )

    out = ntm.attach_year_matched_commuter_exposure(
        grid, pair_dosages=pair_dosages
    )

    assert out["own_driver_distance"].tolist() == pytest.approx([2.4, 24.0])
