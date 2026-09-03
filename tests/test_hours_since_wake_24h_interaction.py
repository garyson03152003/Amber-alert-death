import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_hours_since_wake_24h_interaction as interaction
import run_night_to_morning_window as ntm


def test_24h_interaction_exposes_binary_and_dose_control_variants():
    assert interaction.CONTROL_SPECS == ntm.OTHER_WEA_CONTROL_SPECS


def test_relative_hours_span_wake_to_next_wake():
    window = interaction.relative_hour_window()

    assert len(window) == 24
    assert window.iloc[0].to_dict() == {
        "hours_since_wake": 0,
        "outcome_day_offset": 0,
        "hour": 6,
    }
    assert window.iloc[17].to_dict() == {
        "hours_since_wake": 17,
        "outcome_day_offset": 0,
        "hour": 23,
    }
    assert window.iloc[18].to_dict() == {
        "hours_since_wake": 18,
        "outcome_day_offset": 1,
        "hour": 0,
    }
    assert window.iloc[-1].to_dict() == {
        "hours_since_wake": 23,
        "outcome_day_offset": 1,
        "hour": 5,
    }


def test_interaction_terms_use_elapsed_hours_for_next_day_bins():
    import pandas as pd

    panel = pd.DataFrame(
        {
            "hours_since_wake": [0, 18, 23],
            "own_alert": [0, 1, 1],
            "cross_spillover": [0.0, 0.25, 0.5],
        }
    )

    out = interaction.add_interaction_terms(panel)

    assert out["own_alert_x_hours_since_wake"].tolist() == [0, 18, 23]
    assert out["cross_spillover_x_hours_since_wake"].tolist() == [0.0, 4.5, 11.5]
