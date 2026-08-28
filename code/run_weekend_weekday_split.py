"""
run_weekend_weekday_split.py
=============================================================
Does the commuting-spillover (and own-alert) effect shrink on weekends,
when people can sleep in and partly compensate for a disrupted night --
versus weekdays, when a fixed work-start time forces people onto the
road regardless of how much sleep debt they're carrying?

This is a second, independent test of the same time-awake/sleep-debt
mechanism probed by run_hours_since_wake_dose_response.py: if the
effect is really about accumulated sleep debt forcing itself onto the
road, weekday mornings (fixed wake-up, no compensation option) should
show it more than weekends (flexible wake-up, natural compensation).

Splits the existing 06:00-23:59 headline outcome window
(fatals_0623, from run_night_to_morning_window.py's spec) by the
OUTCOME date's day of week -- Mon-Fri vs Sat-Sun -- and re-runs the
same own-controlled cross_spillover / night_alert regressions
separately on each subsample. FE drops fips_dow's mid-week variation
within each split (2-3 dow categories per county instead of 7), which is
fine -- it's still absorbing each county's own weekday-specific baseline
within whichever subsample it's estimated on.

Output: output/tables/reg_weekend_weekday_split.csv
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("weekend_split")

FE = "fips_year + fips_dow + month_str"


def fit(grid, label, treat, extra_controls, results):
    controls = [treat] + extra_controls
    sub = grid.dropna(subset=controls + ["fatals_0623"]).copy()
    formula = f"fatals_0623 ~ {' + '.join(controls)} | {FE}"
    fit_ = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
    log.info("[%s] beta=%+.6f se=%.6f p=%.4f %s n=%d", label, coef, se, pval, sig, int(fit_._N))
    results.append({"label": label, "treatment": treat, "coef": coef, "se": se,
                    "pval": pval, "nobs": int(fit_._N)})


def main():
    grid = ntm.build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid = ntm.attach_cross_spillover(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow_num"] = grid["date"].dt.dayofweek
    grid["dow"] = grid["dow_num"].astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    grid["is_weekend"] = grid["dow_num"].isin([5, 6])

    weekday = grid[~grid["is_weekend"]].copy()
    weekend = grid[grid["is_weekend"]].copy()
    log.info("Weekday (Mon-Fri) rows: %d, weekend (Sat-Sun) rows: %d", len(weekday), len(weekend))

    results = []
    for label, sub in [("WEEKDAY (Mon-Fri)", weekday), ("WEEKEND (Sat-Sun)", weekend)]:
        log.info("=== %s ===", label)
        fit(sub, f"{label}: OWN night_alert (spillover-controlled)", "night_alert",
            ["cross_spillover"], results)
        fit(sub, f"{label}: CROSS_SPILLOVER (own-controlled)", "cross_spillover",
            ["night_alert"], results)

    out = pd.DataFrame(results)
    out_path = ntm.OUTPUT_TABS / "reg_weekend_weekday_split.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
