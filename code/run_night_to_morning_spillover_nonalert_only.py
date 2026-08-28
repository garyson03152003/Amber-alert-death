"""
run_night_to_morning_spillover_nonalert_only.py
=============================================================
Does the H2 commuting-spillover effect hold up when restricted to
county-days where the RECIPIENT county was not itself directly alerted
that night -- i.e. a "clean" spillover estimate, isolating the case a
commuter drives INTO a county from an alerted home county, without that
work county having simultaneously issued (or been geo-expanded into) an
alert of its own?

Motivation: run_night_to_morning_window.py's headline joint model
regresses fatals_0623 on cross_spillover while linearly controlling for
night_alert, on the full national panel (coef=+0.030464, p=0.0117). But
~65% of AMBER alerts are broadcast statewide and
run_state_dot_analysis_fixed._expand_statewide_rows() geo-expands each one
to every county in that state -- so on a statewide-campaign night, most or
all counties in that state are simultaneously flagged night_alert=1 AND
receive large cross_spillover mass from their many other alerted in-state
neighbors via commuting. The two regressors are not just correlated but
structurally entangled on exactly the highest-dose cross_spillover
observations: of the 196,174 county-days with cross_spillover > 0, the
20,489 that ALSO have night_alert==1 average cross_spillover=0.222,
versus 0.0037 for the other 175,685 -- a ~60x difference in the exposure
variable's own scale. A linear night_alert control cannot fully separate
"driving in from an alerted county" from "living in a county that was
itself swept into the same statewide campaign" when almost all of the
variable's identifying magnitude sits on days where both are true at once.

This script re-estimates cross_spillover -> fatals_0623 (and, for
reference, -> serious_0623) restricted to night_alert==0 county-days only
-- i.e. days the recipient county itself was NOT under any alert -- to see
whether a detectable effect survives once that compound-exposure mass is
removed and only the low-dose, "purely a neighbor's problem" spillover
variation remains.

Data/spec: identical to run_night_to_morning_window.py (FARS hourly
06:00-23:59 window, night_alert/cross_spillover construction, robust
fips_year+fips_dow+month_str FE and naive fips+year+dow+month FE, two-way
state+date clustering). Fixed-effect/cluster columns are cast to pandas
categoricals (see run_night_to_morning_leave_one_out.py) to keep this
comfortably under the memory ceiling that OOM-killed a naive rerun of the
full national grid.

Output: output/tables/reg_night_to_morning_spillover_nonalert_only.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning_spillover_nonalert")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

CLUSTER_VARS = ntm.CLUSTER_VARS
NAIVE_FE = "fips + year_str + dow + month_str"
ROBUST_FE = "fips_year + fips_dow + month_str"


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def fit_one(sub, label, outcome, fe, results, *, extra_controls=None):
    controls = ["cross_spillover"] + (extra_controls or [])
    formula = f"{outcome} ~ {' + '.join(controls)} | {fe}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": CLUSTER_VARS}, lean=True)
    row = fit.tidy().loc["cross_spillover"]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    log.info("  %-55s beta=%+.6f se=%.6f p=%.4g n=%d %s", label, coef, se, pval, int(fit._N), _sig(pval))
    results.append({"check": label, "outcome": outcome, "fe": fe, "coef": coef, "se": se,
                     "pval": pval, "nobs": int(fit._N)})
    del fit
    gc.collect()


def main():
    grid = ntm.build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid = ntm.attach_cross_spillover(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype("category")
    grid["fips_dow"] = (grid["fips"] + "_" + grid["dow"]).astype("category")
    grid["fips_year"] = (grid["fips"] + "_" + grid["year_str"]).astype("category")
    grid["fips"] = grid["fips"].astype("category")
    grid["state_code"] = grid["fips"].astype(str).str[:2].astype("category")
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d").astype("category")
    grid["cross_spillover"] = grid["cross_spillover"].astype("float32")
    grid["fatals_0623"] = grid["fatals_0623"].astype("float32")
    grid["serious_0623"] = grid["serious_0623"].astype("float32")
    gc.collect()

    n_total_nonzero = int((grid["cross_spillover"] > 0).sum())
    own_alert_nonzero = grid.loc[(grid["night_alert"] == 1) & (grid["cross_spillover"] > 0), "cross_spillover"]
    other_nonzero = grid.loc[(grid["night_alert"] == 0) & (grid["cross_spillover"] > 0), "cross_spillover"]
    log.info("Nonzero cross_spillover rows: %d total; %d (%.1f%%) co-occur with the recipient's OWN "
             "night_alert==1 (mean %.4f there vs %.4f elsewhere -- a %.0fx gap)",
             n_total_nonzero, len(own_alert_nonzero), 100 * len(own_alert_nonzero) / n_total_nonzero,
             own_alert_nonzero.mean(), other_nonzero.mean(), own_alert_nonzero.mean() / other_nonzero.mean())

    results = []

    log.info("\n=== Pooled joint model (own night_alert controlled), full national sample -- reference ===")
    fit_one(grid, "pooled (night_alert==0 or 1), robust FE", "fatals_0623", ROBUST_FE, results,
            extra_controls=["night_alert"])
    fit_one(grid, "pooled (night_alert==0 or 1), naive FE", "fatals_0623", NAIVE_FE, results,
            extra_controls=["night_alert"])

    log.info("\n=== Restricted to night_alert==0 (recipient county NOT itself alerted that night) ===")
    sub = grid[grid["night_alert"] == 0]
    log.info("Restricted sample: %d rows (dropped %d), %d nonzero-cross_spillover rows (dropped %d)",
              len(sub), len(grid) - len(sub), int((sub["cross_spillover"] > 0).sum()),
              n_total_nonzero - int((sub["cross_spillover"] > 0).sum()))
    fit_one(sub, "night_alert==0 only, robust FE", "fatals_0623", ROBUST_FE, results)
    fit_one(sub, "night_alert==0 only, naive FE", "fatals_0623", NAIVE_FE, results)
    fit_one(sub, "night_alert==0 only, robust FE, serious injuries", "serious_0623", ROBUST_FE, results)
    del sub
    gc.collect()

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_night_to_morning_spillover_nonalert_only.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
