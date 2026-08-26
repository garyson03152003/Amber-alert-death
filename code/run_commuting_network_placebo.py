"""
run_commuting_network_placebo.py
=============================================================
Falsification test for the commuting-spillover result: does the effect
depend on the REAL commuting network, or would any random county-to-county
network produce a similarly "significant" spillover coefficient?

The temporal placebo (run_night_to_morning_window.py's backward-causal
check) rules out reverse causation on the time axis, but says nothing
about the network axis -- it still uses the true, real commuting weights.
This test instead scrambles WHICH county each edge's home-side represents,
while holding every edge's weight and its work-side county fixed:

    for each of N_PERM draws, pick a uniform-random relabeling (bijection)
    pi: home-county-set -> home-county-set, replace weights.fips_home with
    pi(weights.fips_home), and rebuild cross_spillover from the SAME real
    night_alert events joined through this fake network.

This preserves every structural feature of the commuting matrix (in/out
degree distribution, edge weights, self-loop exclusion) while destroying
the actual home/work correspondence -- so if the real effect is genuinely
carried by real commuting flows (rather than, say, some spurious
correlate of "how well-connected is this county in ANY network"), the
permuted coefficients should cluster near zero and the real estimate
should sit in the tail of that null distribution.

Reports an empirical (permutation) p-value: the fraction of permuted
coefficients >= the real coefficient (one-sided, since the maintained
hypothesis is a positive spillover effect).

Output: output/tables/reg_commuting_network_placebo.csv
  (one row per permutation draw, plus the real estimate for reference)
"""
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
log = get_logger("commuting_placebo")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

N_PERM = 100
SEED = 20240826
FE = "fips_year + fips_dow + month_str"


def fit_spillover(grid: pd.DataFrame, col: str):
    sub = grid.dropna(subset=[col, "night_alert", "fatals_0623"]).copy()
    formula = f"fatals_0623 ~ {col} + night_alert | {FE}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit.tidy().loc[col]
    return float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])


def main():
    grid = ntm.build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    weights = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    log.info("Commuting weights: %d edges, %d home counties, %d work counties",
             len(weights), weights["fips_home"].nunique(), weights["fips_work"].nunique())

    # Real (unpermuted) network -> the headline estimate
    real_grid = ntm.attach_cross_spillover(grid.copy(), out_col="cross_spillover", weights=weights)
    real_coef, real_se, real_p = fit_spillover(real_grid, "cross_spillover")
    log.info("REAL commuting network:   beta=%+.5f se=%.5f p=%.4f", real_coef, real_se, real_p)

    home_counties = weights["fips_home"].unique()
    rng = np.random.default_rng(SEED)

    results = [{"draw": "real", "coef": real_coef, "se": real_se, "pval": real_p}]
    perm_coefs = []
    for i in range(1, N_PERM + 1):
        perm_map = dict(zip(home_counties, rng.permutation(home_counties)))
        w_perm = weights.copy()
        w_perm["fips_home"] = w_perm["fips_home"].map(perm_map)

        g_perm = ntm.attach_cross_spillover(grid.copy(), out_col="cross_spillover_perm", weights=w_perm)
        coef, se, pval = fit_spillover(g_perm, "cross_spillover_perm")
        perm_coefs.append(coef)
        results.append({"draw": i, "coef": coef, "se": se, "pval": pval})
        if i % 10 == 0:
            log.info("  permutation %3d/%d done (running mean beta so far: %+.5f)",
                     i, N_PERM, float(np.mean(perm_coefs)))

    perm_coefs = np.array(perm_coefs)
    p_perm_onesided = (np.sum(perm_coefs >= real_coef) + 1) / (N_PERM + 1)
    p_perm_twosided = (np.sum(np.abs(perm_coefs) >= abs(real_coef)) + 1) / (N_PERM + 1)

    log.info("\n=== Network-permutation placebo summary ===")
    log.info("Real beta:            %+.5f (p=%.4f, analytic 2-way-clustered SE)", real_coef, real_p)
    log.info("Permuted-null beta:   mean=%+.5f sd=%.5f min=%+.5f max=%+.5f",
             perm_coefs.mean(), perm_coefs.std(), perm_coefs.min(), perm_coefs.max())
    log.info("Permutation p-value:  one-sided=%.4f  two-sided=%.4f  (n=%d draws)",
             p_perm_onesided, p_perm_twosided, N_PERM)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_commuting_network_placebo.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
