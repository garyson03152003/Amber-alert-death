"""Run the combined sleep/spillover models with weather WEA removed from control.

This sensitivity keeps the treatment definition and model specifications fixed,
but replaces the all-non-AMBER-WEA control with a control containing only
non-weather CAP categories.  Weather is identified from CAP's structured
``category=Met`` field; fire, evacuation, earthquake, and other non-Met public
safety alerts remain eligible controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window_combined as combined
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

log = get_logger("night_to_morning_weather_sensitivity")

WEATHER_EXCLUDED_CONTROL_PATH = DATA_PROC / "other_wea_night_controls_no_weather.parquet"
OUTPUT_PATH = OUTPUT_TABS / "reg_night_to_morning_window_weather_sensitivity.csv"


def main() -> None:
    if not WEATHER_EXCLUDED_CONTROL_PATH.is_file():
        raise FileNotFoundError(
            f"weather-excluded control not found at {WEATHER_EXCLUDED_CONTROL_PATH}; "
            "run code/fetch_other_wea_controls.py --exclude-weather first"
        )

    headline = combined.run_combined(
        include_cancel=True, control_path=WEATHER_EXCLUDED_CONTROL_PATH
    )
    sensitivity = combined.run_combined(
        include_cancel=False, control_path=WEATHER_EXCLUDED_CONTROL_PATH
    )
    out = pd.concat([headline, sensitivity], ignore_index=True)
    out["sensitivity"] = "exclude_weather"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved weather-excluded estimates -> %s (%d rows)", OUTPUT_PATH, len(out))


if __name__ == "__main__":
    main()
