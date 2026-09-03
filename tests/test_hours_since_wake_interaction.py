import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_hours_since_wake_interaction as interaction
import run_night_to_morning_window as ntm


def test_hour_window_is_six_to_midnight_exclusive():
    assert interaction.HOURS == tuple(range(6, 24))
    assert len(interaction.HOURS) == 18
    assert interaction.HOURS[-1] == 23  # 23:00--24:00 is included.


def test_interaction_exposes_binary_and_dose_control_variants():
    assert interaction.CONTROL_SPECS == ntm.OTHER_WEA_CONTROL_SPECS


def test_interaction_terms_use_clock_hours_and_both_exposures():
    panel = pd.DataFrame(
        {
            "hour": [6, 10, 23],
            "own_alert": [0, 1, 1],
            "cross_spillover": [0.0, 0.25, 0.5],
        }
    )

    out = interaction.add_interaction_terms(panel)

    assert out["hours_since_wake"].tolist() == [0, 4, 17]
    assert out["own_alert_x_hours_since_wake"].tolist() == [0, 4, 17]
    assert out["cross_spillover_x_hours_since_wake"].tolist() == [0.0, 1.0, 8.5]
