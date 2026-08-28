"""FARS hourly window analysis with alignment fixed and exposure reported.

Two corrections over run_fars_time_window_share.py:

1. Offsets are measured from D = ``effective_crash_date``, which is the day
   whose DRIVING the alert affects (a 23:00 alert on the 12th has D = the
   13th; a 02:00 alert on the 20th has D = the 20th). The morning commute
   after an alert is therefore D hours 06-09, i.e. OFFSET 0. The original
   module used offset 1 for that window -- the second morning after -- and
   offset 2 for its placebo.

2. Every estimate is reported next to the number of treated county-days that
   actually contain a fatality. FARS is a census of a rare event: 99.4% of
   non-empty county-hours hold exactly one fatal crash, and the full
   county x date x hour grid is 99.875% true zeros. A PPML coefficient can
   look precise while resting on a couple of dozen events, so the exposure
   count is part of the result, not a footnote.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_share as share
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from run_validated_fars_share import build_panel as build_base_fars_panel

HOURLY = DATA_PROC / "fars_hourly_county_day.parquet"

# label -> list of (offset_from_D, hour_lo, hour_hi) inclusive
WINDOWS = {
    "same_day_00_06":   [(0, 0, 5)],    # requested: overnight of the affected day
    "next_day_10_12":   [(1, 10, 11)],  # requested: mid-morning, following day
    "same_day_06_10":   [(0, 6, 9)],    # correctly aligned morning commute
    "OLD_next_day_06_10": [(1, 6, 9)],  # what the original module called "w1"
}


def _totals(hourly, offset, lo, hi):
    sub = hourly.loc[hourly["hour"].between(lo, hi)].copy()
    sub["date"] = sub["date"] - pd.Timedelta(days=offset)
    return sub.groupby(["fips", "date"], as_index=False)[["person_fatals", "serious_inj"]].sum()


def main() -> None:
    hourly = pd.read_parquet(HOURLY)
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.normalize()
    panel = build_base_fars_panel(direct_only=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    rows = []
    for label, parts in WINDOWS.items():
        tot = pd.concat([_totals(hourly, o, lo, hi) for o, lo, hi in parts],
                        ignore_index=True).groupby(["fips", "date"], as_index=False).sum()
        tot = tot.rename(columns={"person_fatals": f"{label}_fatals",
                                  "serious_inj": f"{label}_serious"})
        p = panel.merge(tot, on=["fips", "date"], how="left")
        for c in (f"{label}_fatals", f"{label}_serious"):
            p[c] = p[c].fillna(0.0)
            p[f"{c}_per_100k"] = 100_000 * p[c] / p["population"]

        treated = p["night_alert"] == 1
        for outcome in ("fatals", "serious"):
            col = f"{label}_{outcome}"
            n_treated_days = int(treated.sum())
            n_treated_events = int((p.loc[treated, col] > 0).sum())
            sum_treated = float(p.loc[treated, col].sum())
            mean_t = float(p.loc[treated, col].mean())
            mean_c = float(p.loc[~treated, col].mean())
            print(f"[{label} / {outcome}] treated county-days={n_treated_days:,} "
                  f"with>0={n_treated_events} sum={sum_treated:.0f} "
                  f"raw_mean_ratio={mean_t/mean_c if mean_c else float('nan'):.3f}", flush=True)

            est = share.run_ppml(p, col, f"FARS_{label.upper()}_{outcome.upper()}")
            for r in est:
                if r.get("record_type") == "estimate":
                    r.update({"window": label, "n_treated_county_days": n_treated_days,
                              "n_treated_with_event": n_treated_events,
                              "treated_event_sum": sum_treated,
                              "raw_mean_ratio": mean_t / mean_c if mean_c else np.nan})
                    rows.append(r)

    out = pd.DataFrame(rows)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_TABS / "fars_window_custom.csv", index=False)
    keep = ["window", "outcome", "term", "pct_change", "pvalue",
            "n_treated_with_event", "raw_mean_ratio"]
    keep = [c for c in keep if c in out.columns]
    print("\n" + out[out["term"] == "night_alert"][keep].to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
