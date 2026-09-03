import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_time_block_outcomes_use_average_hour_rates_and_fixed_boundaries():
    from run_symmetric_commuter_fatigue import build_time_block_outcomes

    date = pd.Timestamp("2024-03-05")
    hourly = pd.DataFrame(
        {
            "fips": ["01001"] * 18,
            "date": [date] * 18,
            "hour": list(range(6, 24)),
            "person_fatals": [float(hour - 5) for hour in range(6, 24)],
        }
    )

    got = build_time_block_outcomes(hourly).iloc[0]

    assert got["fatals_avg_0609"] == pytest.approx(2.5)
    assert got["fatals_avg_1014"] == pytest.approx(7.0)
    assert got["fatals_avg_1519"] == pytest.approx(12.0)
    assert got["fatals_avg_2023"] == pytest.approx(16.5)
    assert got["fatals_late_minus_morning"] == pytest.approx(14.0)


def test_time_block_outcomes_treat_absent_sparse_hours_as_zero():
    from run_symmetric_commuter_fatigue import build_time_block_outcomes

    hourly = pd.DataFrame(
        {
            "fips": ["01001"],
            "date": [pd.Timestamp("2024-03-05")],
            "hour": [6],
            "person_fatals": [4.0],
        }
    )

    got = build_time_block_outcomes(hourly).iloc[0]

    assert got["fatals_avg_0609"] == pytest.approx(1.0)
    assert got["fatals_avg_1014"] == 0.0
    assert got["fatals_avg_1519"] == 0.0
    assert got["fatals_avg_2023"] == 0.0
    assert got["fatals_late_minus_morning"] == pytest.approx(-1.0)

