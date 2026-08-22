"""Distraction-vs-sleep-disruption mechanism test: WHEN do crashes happen
relative to a night AMBER alert, not just whether the next day's total moves.

Five time windows relative to the alert night D (an alerted county-date):

  W0 same-night   D 20:00 -> D+1 06:00   (H1: immediate driving distraction)
  W1 morning      D+1 06:00 -> 10:00     (H2: sleep-disruption commute)
  W2 midday       D+1 10:00 -> 16:00     (control)
  W3 evening      D+1 16:00 -> 20:00     (control)
  W4 placebo      D+2 06:00 -> 10:00     (same commute window, 24h later)

Uses the same crosswalk-validated FARS geography/date rules as
``build_fars_county_day.py`` (via ``build_fars_hourly.py``) and the same
zero-preserving PPML + log(population) offset, county + calendar-date fixed
effects, commuter-share spillover design as the rest of the validated FARS
pipeline (``run_validated_fars_share.py``). A missing hourly county-day
combination is a genuine zero (FARS is a census of all fatal crashes), not a
missing observation -- unlike the state-DOT panels, so it is zero-filled here.

Output: output/tables/fars_time_window_share.csv
"""
from __future__ import annotations

import pandas as pd

import run_state_dot_analysis_share as share
from config import DATA_PROC, OUTPUT_TABS
from run_validated_fars_share import build_panel as build_base_fars_panel
from state_dot_analysis_core import summarize_fit_statuses

HOURLY = DATA_PROC / "fars_hourly_county_day.parquet"

WINDOWS = {
    # (date_offset_from_alert_night, hour_lo, hour_hi)
    "w0": [(0, 20, 23), (1, 0, 5)],
    "w1": [(1, 6, 9)],
    "w2": [(1, 10, 15)],
    "w3": [(1, 16, 19)],
    "w4": [(2, 6, 9)],
}


def _window_totals(hourly: pd.DataFrame, offset: int, hour_lo: int, hour_hi: int) -> pd.DataFrame:
    """Return county x alert-night-date totals for one (offset, hour range)."""
    sub = hourly.loc[hourly["hour"].between(hour_lo, hour_hi)].copy()
    # An alert issued on panel date D produces a crash on (D + offset); invert
    # to recover the panel date each hourly row would be attributed to.
    sub["date"] = sub["date"] - pd.Timedelta(days=offset)
    return sub.groupby(["fips", "date"], as_index=False)[["person_fatals", "serious_inj"]].sum()


def build_window_panel() -> pd.DataFrame:
    if not HOURLY.is_file():
        raise FileNotFoundError(f"hourly validated FARS panel not found: {HOURLY}; run build_fars_hourly.py first")
    hourly = pd.read_parquet(HOURLY)
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.normalize()

    panel = build_base_fars_panel(direct_only=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    for window, parts in WINDOWS.items():
        totals = pd.concat(
            [_window_totals(hourly, offset, lo, hi) for offset, lo, hi in parts],
            ignore_index=True,
        ).groupby(["fips", "date"], as_index=False)[["person_fatals", "serious_inj"]].sum()
        totals = totals.rename(columns={
            "person_fatals": f"{window}_fatals", "serious_inj": f"{window}_serious",
        })
        panel = panel.merge(totals, on=["fips", "date"], how="left")
        # A FARS census with no matching hourly row for this county-date-window
        # combination is a genuine zero count, not a missing observation.
        panel[f"{window}_fatals"] = panel[f"{window}_fatals"].fillna(0.0)
        panel[f"{window}_serious"] = panel[f"{window}_serious"].fillna(0.0)
        panel[f"{window}_fatals_per_100k"] = 100_000 * panel[f"{window}_fatals"] / panel["population"]
        panel[f"{window}_serious_per_100k"] = 100_000 * panel[f"{window}_serious"] / panel["population"]
    return panel


def main() -> None:
    panel = build_window_panel()
    rows: list[dict] = []
    for window in WINDOWS:
        for outcome in ("fatals", "serious"):
            count_col = f"{window}_{outcome}"
            rate_col = f"{count_col}_per_100k"
            label = f"FARS_WINDOW_{window.upper()}_{outcome.upper()}"
            rows.extend(share.run_wls(panel, rate_col, label))
            rows.extend(share.run_ppml(panel, count_col, label))
            rows.extend(share.run_wls(panel, rate_col, label, clean_controls=True))
            rows.extend(share.run_ppml(panel, count_col, label, clean_controls=True))
    all_rows = pd.DataFrame(rows)
    estimates = all_rows.loc[all_rows["record_type"].eq("estimate")].copy()
    statuses = all_rows.loc[all_rows["record_type"].eq("fit_status")].copy()
    statuses = pd.concat([statuses, pd.DataFrame([{
        "record_type": "model_count_summary", **summarize_fit_statuses(statuses.to_dict("records")),
    }])], ignore_index=True)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUTPUT_TABS / "fars_time_window_share.csv", index=False)
    statuses.to_csv(OUTPUT_TABS / "fars_time_window_share_status.csv", index=False)
    print(f"Saved {len(estimates)} estimates and {len(statuses)} fit diagnostics -> {OUTPUT_TABS}")


if __name__ == "__main__":
    main()
