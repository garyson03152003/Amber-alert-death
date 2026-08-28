"""
run_night_to_morning_leave_one_out.py
=============================================================
Is the headline H2 sleep/commuting-spillover result driven by a small
number of states or work-counties?

Motivation: run_night_to_morning_window.py's headline term is
CROSS_SPILLOVER (own-controlled) -> fatals under the robust spec
(fips_year + fips_dow + month_str FE, two-way state+date clustering):
  coef=+0.03046, se=0.01164, p=0.0117, n=7,348,614
(interpreted as the sleep-disruption-via-commuters channel: a work
county's fatal-crash count rises with the commuting-weighted share of
its workforce living in a county that had a nighttime AMBER alert in the
last two nights).

Both sides of that exposure are concentrated in the same few states that
drove the H1 same-hour highway-fatals result (see
reg_same_hour_road_type_leave_one_out.csv / SAME_HOUR_ROAD_TYPE_LOO_RESULTS.md):
  - home-alert county-days (the source of spillover exposure): Texas,
    Georgia, and North Carolina together supply ~51% of all night-alert
    county-days.
  - work-state cross_spillover mass (the sum of exposure actually
    received): the same three states (Georgia, Texas, North Carolina)
    together receive ~51% of total cross_spillover mass, mostly via
    within-state commuting flows into other counties in the same alert-
    heavy state.

Design: exact same national county-day panel and specification as
run_night_to_morning_window.py's robust spec (night_alert + cross_spillover
jointly, fips_year + fips_dow + month_str FE, state+date two-way
clustering). Reports the cross_spillover (and, for reference, night_alert)
coefficient while:
  1. Leave-one-state-out: drop each state's counties (as work-county
     observations) one at a time, keeping the exposure construction
     (which home counties alerted, and the commuting weights) computed on
     the full national data untouched -- this isolates whether that
     state's own crash outcomes/observations are necessary to the result,
     not whether removing it as an alert source matters.
  2. Drop the top-3 work-state-mass states (GA, TX, NC) together.
  3. Leave-one-county-out for the 10 work-counties with the largest total
     cross_spillover mass.

Output: output/tables/reg_night_to_morning_leave_one_out.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning_loo")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
CLUSTER_VARS = ntm.CLUSTER_VARS
TREAT = "cross_spillover"
EXTRA = ["night_alert"]
OUTCOME = "fatals_0623"


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def fit_one(sub, label, results, *, dropped=""):
    controls = [TREAT] + EXTRA
    sub = sub.dropna(subset=controls + [OUTCOME])
    formula = f"{OUTCOME} ~ {' + '.join(controls)} | {FE}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": CLUSTER_VARS}, lean=True)
    td = fit.tidy()
    row = td.loc[TREAT]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    n_nonzero = int((sub[TREAT] > 0).sum())
    log.info("  %-55s beta=%+.6f se=%.6f p=%.4g n=%d n_nonzero_spillover=%d %s",
             label, coef, se, pval, int(fit._N), n_nonzero, _sig(pval))
    results.append({"check": label, "dropped": dropped, "outcome": OUTCOME, "treatment": TREAT,
                     "coef": coef, "se": se, "pval": pval,
                     "nobs": int(fit._N), "n_nonzero_spillover": n_nonzero})
    del fit, sub, td
    gc.collect()
    # Checkpoint after every regression -- the memory-hungry sibling script
    # (run_night_to_morning_window.py) OOM-killed once already on this same
    # national grid, so don't risk losing 60+ regressions' worth of progress.
    pd.DataFrame(results).to_csv(OUTPUT_TABS / "reg_night_to_morning_leave_one_out.csv", index=False)


def main():
    # NOTE on memory: run_night_to_morning_window.py's own grid construction
    # leaves fips/fips_year/fips_dow/state_code/date_str as raw object-dtype
    # strings across 7.3M rows (~1.3GB) and re-copies the full grid on every
    # one of its 12 sequential regressions -- fine for a single run, but this
    # script re-estimates 60+ times over leave-one-out subsets and OOM-killed
    # (13.9GB RSS in a 15GB container) when first tried with that approach.
    # Casting the fixed-effect/cluster columns to pandas categoricals and
    # downcasting numeric dtypes cuts the grid to ~140MB and peak fit RSS to
    # ~4GB, with identical point estimates (verified against the committed
    # reg_night_to_morning_window.csv baseline: coef=0.030464, p=0.011666).
    grid = ntm.build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid = ntm.attach_cross_spillover(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype("category")
    grid["fips_dow"] = (grid["fips"] + "_" + grid["dow"]).astype("category")
    grid["fips_year"] = (grid["fips"] + "_" + grid["year_str"]).astype("category")
    grid["state_code"] = grid["fips"].str[:2].astype("category")
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d").astype("category")
    grid["fips"] = grid["fips"].astype("category")
    grid["night_alert"] = grid["night_alert"].astype("int8")
    grid["cross_spillover"] = grid["cross_spillover"].astype("float32")
    grid["fatals_0623"] = grid["fatals_0623"].astype("float32")
    grid = grid.drop(columns=["serious_0623", "night_alert_lag1", "night_alert_lead1",
                               "alert_last2nights_any", "alert_last2nights_dose",
                               "year_str", "dow", "date"])
    gc.collect()

    mass_by_state = grid.groupby("state_code", observed=True)[TREAT].sum().sort_values(ascending=False)
    total_mass = float(mass_by_state.sum())
    mass_by_county = grid.groupby("fips", observed=True)[TREAT].sum().sort_values(ascending=False)
    log.info("Total cross_spillover mass: %.2f across %d states, %d counties",
             total_mass, mass_by_state.shape[0], mass_by_county.shape[0])
    log.info("Top state shares of work-side spillover mass: %s",
             {s: f"{m / total_mass:.1%}" for s, m in mass_by_state.head(5).items()})

    results = []

    log.info("\n=== Baseline (full national sample) ===")
    fit_one(grid, "baseline (all states)", results)

    log.info("\n=== Leave-one-state-out (%d states) ===", mass_by_state.shape[0])
    for state in mass_by_state.index:
        share = mass_by_state[state] / total_mass
        sub = grid[grid["state_code"] != state]
        fit_one(sub, f"drop state {state} ({share:.1%} of spillover mass)", results, dropped=state)
        del sub
        gc.collect()

    log.info("\n=== Drop top-3 spillover-mass states together ===")
    top3 = list(mass_by_state.index[:3])
    sub = grid[~grid["state_code"].isin(top3)]
    fit_one(sub, f"drop top-3 states {top3} (jointly)", results, dropped="+".join(top3))
    del sub
    gc.collect()

    log.info("\n=== Leave-one-county-out (top 10 spillover-mass work-counties) ===")
    for fips in mass_by_county.index[:10]:
        share = mass_by_county[fips] / total_mass
        sub = grid[grid["fips"] != fips]
        fit_one(sub, f"drop county {fips} ({share:.2%} of spillover mass)", results, dropped=fips)
        del sub
        gc.collect()

    out = pd.DataFrame(results)
    baseline_coef = out.loc[out["check"] == "baseline (all states)", "coef"].iloc[0]
    out["coef_shift_from_baseline"] = out["coef"] - baseline_coef
    out_path = OUTPUT_TABS / "reg_night_to_morning_leave_one_out.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
