"""
run_night_to_morning_state_dot_severity.py
=============================================================
Severity-separated version of the sleep-channel (H2) test: does the
commuting-spillover / own-alert result hold up against ALL crash
severities (not just FARS fatalities), using the validated multi-state
DOT crash panels (run_state_dot_analysis_fixed.load_validated_state_crashes)
which carry a total-crash count alongside fatals and serious injuries?

Coverage: 19 states/MPO regions with reviewed, FARS-cross-validated
daily county crash panels (CA, CT, DE, FL, HI, ID(COMPASS), IL, IN(MPO),
IA, MD(Montgomery Co.), MA, NC, NY, OR, TN, TX, UT, VA, WI) --
2,823,754 county-date rows, 2013-2025.

Design differences from the FARS-based run_night_to_morning_window.py:
  - Outcome is the FULL calendar day on effective_crash_date (this data
    has no hour column, so the 06:00-23:59-only restriction FARS allows
    isn't available here -- this trades a small amount of window
    precision for ~19x the geographic coverage and, for the "crashes"
    outcome, orders of magnitude more events than fatal-only).
  - Same FE (county x year + county x weekday + month) and two-way
    (state+date) clustering as the FARS headline spec.
  - Restricted to counties actually present in the validated panel
    (only a fraction of these states' counties are covered/reviewed --
    see config/accepted_state_years.csv).

Output: output/tables/reg_night_to_morning_state_dot_severity.csv
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
import run_state_dot_analysis_fixed as base
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning_state_dot")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
OUTCOMES = ["crashes", "fatals", "serious_inj"]


def build_outcome_grid():
    flows = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    crashes = base.load_validated_state_crashes(flows=flows)
    crashes["date"] = pd.to_datetime(crashes["date"]).dt.normalize()
    active = sorted(crashes["fips"].unique())
    log.info("Validated state-DOT counties: %d (states: %s)",
             len(active), sorted(crashes["fips"].str[:2].unique()))

    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")
    grid = pd.MultiIndex.from_product([active, dates], names=["fips", "date"]).to_frame(index=False)
    grid = grid.merge(crashes[["fips", "date"] + OUTCOMES], on=["fips", "date"], how="left")
    for col in OUTCOMES:
        grid[col] = grid[col].fillna(0)
    return grid, active


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def run(grid, label, outcome, treat, extra_controls, results):
    controls = [treat] + extra_controls
    sub = grid.dropna(subset=controls + [outcome]).copy()
    formula = f"{outcome} ~ {' + '.join(controls)} | {FE}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit.tidy().loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    log.info("  %-70s beta=%+.6f se=%.6f p=%.3f n=%d %s",
             label, coef, se, pval, int(fit._N), _sig(pval))
    results.append({"label": label, "outcome": outcome, "treatment": treat,
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N)})


def main():
    grid, active = build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid = ntm.attach_cross_spillover(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    results = []
    log.info("\n=== State-DOT panel, severity-separated (robust FE, two-way clustering) ===")
    for outcome in OUTCOMES:
        run(grid, f"{outcome}: OWN night_alert (spillover-controlled)", outcome,
            "night_alert", ["cross_spillover"], results)
        run(grid, f"{outcome}: CROSS_SPILLOVER (own-controlled)", outcome,
            "cross_spillover", ["night_alert"], results)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_night_to_morning_state_dot_severity.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
