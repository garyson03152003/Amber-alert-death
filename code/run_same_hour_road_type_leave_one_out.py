"""
run_same_hour_road_type_leave_one_out.py
=============================================================
Is the headline same-hour highway-fatals result (reg_same_hour_road_type_split.csv:
beta=-0.000175, p=6.7e-06, national) driven by a small number of states?

Motivation: alert county-date-hour events are heavily concentrated by state
-- Texas alone supplies ~32% of all alert-hour observations in the matched-
referent grid, and Texas + Georgia + North Carolina together supply ~53%.
That concentration comes from how statewide AMBER alert campaigns expand
into county-level rows (run_state_dot_analysis_fixed.load_verified_alerts),
not from genuine geographic dispersion, so a national estimate could in
principle be dominated by a handful of large, alert-heavy states rather than
reflecting a broadly shared response. County-level concentration is checked
too but is much milder (top 10 of 1,676 alert-touched counties are only
~2.6% of alert-hours), since a single statewide campaign fans out across
many counties in that state.

Design: exact same matched-referent case-crossover grid and specification
as run_same_hour_road_type_split.py (fips_hour_dow + fips_year + year_month
FE, two-way state+date clustering), for both highway_fatals and
nonhighway_fatals. Three leave-one-out checks:

  1. Leave-one-state-out: drop each state with >=1 alert-hour in turn,
     re-estimate on the rest, and report how far the coefficient moves.
  2. Drop the top-3 alert-hour states (TX, GA, NC) together, since they
     jointly supply over half the treated variation.
  3. Leave-one-county-out for the top 10 alert-hour counties, as a check
     that no single county is doing outsized work (expected to matter far
     less than state, given the diffuse county distribution).

Output: output/tables/reg_same_hour_road_type_leave_one_out.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_same_hour_event_study as base
import run_same_hour_road_type_split as rt
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("same_hour_road_type_loo")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_hour_dow + fips_year + year_month"
CLUSTER_VARS = base.CLUSTER_VARS
OUTCOMES = ["highway_fatals", "nonhighway_fatals"]


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def fit_one(sub, outcome, label, results, *, dropped=""):
    sub = sub.dropna(subset=["is_alert_hour", outcome])
    formula = f"{outcome} ~ is_alert_hour | {FE}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": CLUSTER_VARS}, lean=True)
    td = fit.tidy()
    row = td.loc["is_alert_hour"]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    n_alert = int(sub["is_alert_hour"].sum())
    log.info("  [%-9s] %-40s beta=%+.6f se=%.6f p=%.3g n=%d n_alert=%d %s",
             outcome, label, coef, se, pval, int(fit._N), n_alert, _sig(pval))
    results.append({"check": label, "dropped": dropped, "outcome": outcome,
                     "coef": coef, "se": se, "pval": pval,
                     "nobs": int(fit._N), "n_alert_hours": n_alert})
    del fit, sub, td
    gc.collect()


def main():
    active = base.active_counties()
    log.info("Active (>=%d fatals/yr) counties: %d", base.MIN_FATALS_PER_YEAR, len(active))
    ev = base.load_any_time_alert_hours(active)

    grid = rt.build_road_type_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    alert_hours_by_state = (grid.loc[grid["is_alert_hour"] == 1]
                             .groupby("state_code").size().sort_values(ascending=False))
    alert_hours_by_county = (grid.loc[grid["is_alert_hour"] == 1]
                              .groupby("fips").size().sort_values(ascending=False))
    total_alert_hours = int(grid["is_alert_hour"].sum())
    log.info("Total alert-hour rows: %d across %d states, %d counties",
             total_alert_hours, alert_hours_by_state.shape[0], alert_hours_by_county.shape[0])
    log.info("Top state shares: %s", {s: f"{n / total_alert_hours:.1%}"
                                       for s, n in alert_hours_by_state.head(5).items()})

    results = []

    log.info("\n=== Baseline (full national sample) ===")
    for outcome in OUTCOMES:
        fit_one(grid, outcome, "baseline (all states)", results)

    log.info("\n=== Leave-one-state-out (%d states with alert-hours) ===",
              alert_hours_by_state.shape[0])
    for state in alert_hours_by_state.index:
        share = alert_hours_by_state[state] / total_alert_hours
        sub = grid[grid["state_code"] != state]
        for outcome in OUTCOMES:
            fit_one(sub, outcome, f"drop state {state} ({share:.1%} of alert-hours)",
                    results, dropped=state)
        del sub
        gc.collect()

    log.info("\n=== Drop top-3 alert-hour states together (TX+GA+NC) ===")
    top3 = list(alert_hours_by_state.index[:3])
    sub = grid[~grid["state_code"].isin(top3)]
    for outcome in OUTCOMES:
        fit_one(sub, outcome, f"drop top-3 states {top3} (jointly)", results, dropped="+".join(top3))
    del sub
    gc.collect()

    log.info("\n=== Leave-one-county-out (top 10 alert-hour counties) ===")
    for fips in alert_hours_by_county.index[:10]:
        share = alert_hours_by_county[fips] / total_alert_hours
        sub = grid[grid["fips"] != fips]
        for outcome in OUTCOMES:
            fit_one(sub, outcome, f"drop county {fips} ({share:.2%} of alert-hours)",
                    results, dropped=fips)
        del sub
        gc.collect()

    out = pd.DataFrame(results)
    baseline = out[out["check"] == "baseline (all states)"].set_index("outcome")["coef"]
    out["coef_shift_from_baseline"] = out.apply(
        lambda r: r["coef"] - baseline[r["outcome"]], axis=1)
    out_path = OUTPUT_TABS / "reg_same_hour_road_type_leave_one_out.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
