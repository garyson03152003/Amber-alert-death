import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_fars_night_window_compare as runner


def test_night_window_compare_runs_ordered_cutoff_gradient():
    assert runner.NIGHT_STARTS == (20, 21, 22, 23)
