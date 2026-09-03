import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_window_shift_sensitivity as sensitivity


def test_shifted_window_maps_clock_hours_and_wake_time():
    window = sensitivity.relative_window(start_hour=7, end_hour=1)

    assert len(window) == 18
    assert window.iloc[0].to_dict() == {
        "elapsed_hour": 0,
        "hours_since_wake": 1,
        "outcome_day_offset": 0,
        "hour": 7,
    }
    assert window.iloc[-1].to_dict() == {
        "elapsed_hour": 17,
        "hours_since_wake": 18,
        "outcome_day_offset": 1,
        "hour": 0,
    }


def test_shifted_window_crosses_midnight_only_when_needed():
    window = sensitivity.relative_window(start_hour=4, end_hour=22)
    assert window["outcome_day_offset"].tolist() == [0] * 18

    wake_to_wake = sensitivity.relative_window(start_hour=6, end_hour=6)
    assert wake_to_wake["outcome_day_offset"].tolist() == [0] * 18 + [1] * 6
    assert wake_to_wake["hour"].tolist()[-6:] == [0, 1, 2, 3, 4, 5]


def test_explicit_endpoints_include_requested_04_to_23_and_24_windows():
    to_23 = sensitivity.relative_window(start_hour=4, end_hour=23)
    to_24 = sensitivity.relative_window(start_hour=4, end_hour=24)

    assert len(to_23) == 19
    assert to_23.iloc[-1]["hour"] == 22
    assert len(to_24) == 20
    assert to_24.iloc[-1]["hour"] == 23
    assert sensitivity.window_label(4, 23) == "04:00-23:00"
    assert sensitivity.window_label(4, 24) == "04:00-24:00"


def test_window_grid_contains_all_start_end_combinations():
    starts = {start for start, _ in sensitivity.WINDOW_SPECS}
    ends = {end for _, end in sensitivity.WINDOW_SPECS}

    assert starts == {4, 5, 6, 7, 8}
    assert ends == {22, 23, 24, 1, 2}
    assert len(sensitivity.WINDOW_SPECS) == 25
    assert len(set(sensitivity.WINDOW_SPECS)) == 25
