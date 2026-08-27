"""
run_own_county_dosage.py
=============================================================
Applies the SAME refinements built for the cross-county commuting
spillover (car-mode share, LODES worker-weighted distance) to the
OWN-county alert effect, which so far has only ever been tested as a
raw binary (night_alert). That's an apples-to-oranges comparison: the
spillover measure is a continuous dosage (commuting weight x car share
x distance), while the own-county measure is just "did this county have
an alert, yes/no" -- so the widely different p-values (own: null,
spillover: significant) could partly reflect measurement granularity
rather than a real difference in mechanism.

This rebuilds an own-county dosage using the same three ingredients,
just drawn from the SELF-loop (fips_home == fips_work) rows that the
cross-county analysis deliberately excludes:
  - own_weight_c:  share of county c's workforce that both lives AND
    works in c (county_commuting_weights.parquet self-loop row) --
    mean 72.7% nationally, i.e. most commuters don't cross county lines
    at all, so this is the dominant piece of any county's "local
    driving population."
  - car_share_c:   same ACS B08301 car-commute share used for the
    cross-county car dosage (county_car_commuters.parquet).
  - own_dist_c:    average INTRA-county commute distance from the same
    LODES pipeline used for cross-county distance
    (county_pair_lodes_distance.parquet self-loop row) -- mean 4.6
    miles nationally (much shorter than cross-county commutes, as
    expected).

All variants are tested jointly controlling for cross_spillover (as in
run_night_to_morning_window.py's headline spec), so this isolates
whether a more granular OWN-county exposure measure recovers an effect
that the raw binary missed -- if it stays null even with the same kind
of dosage weighting that made the spillover term significant, that is
real evidence the own/spillover asymmetry isn't just a measurement
artifact.

Output: output/tables/reg_own_county_dosage.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("own_county_dosage")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"


def load_own_county_factors() -> pd.DataFrame:
    w = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    self_w = w[w["fips_home"] == w["fips_work"]].copy()
    self_w["fips"] = self_w["fips_home"].astype(str).str.zfill(5)
    self_w = self_w[["fips", "weight"]].rename(columns={"weight": "own_weight"})

    car = pd.read_parquet(DATA_PROC / "county_car_commuters.parquet",
                          columns=["fips", "car_total", "total_workers"])
    car["fips"] = car["fips"].astype(str).str.zfill(5)
    car["car_share"] = car["car_total"] / car["total_workers"].clip(lower=1)

    lodes = pd.read_parquet(DATA_PROC / "commuting" / "county_pair_lodes_distance.parquet")
    self_d = lodes[lodes["fips_home"] == lodes["fips_work"]].copy()
    self_d = self_d.rename(columns={"fips_home": "fips", "avg_dist_mi": "own_dist_mi"})[["fips", "own_dist_mi"]]

    out = self_w.merge(car[["fips", "car_share"]], on="fips", how="outer") \
                .merge(self_d, on="fips", how="outer")
    log.info("Own-county factors: %d counties, own_weight mean=%.3f, car_share mean=%.3f, "
             "own_dist_mi mean=%.2f", len(out), out["own_weight"].mean(),
             out["car_share"].mean(), out["own_dist_mi"].mean())
    return out


def fit(grid, label, treat, results):
    controls = [treat, "cross_spillover"]
    sub = grid.dropna(subset=controls + ["fatals_0623"]).copy()
    formula = f"fatals_0623 ~ {' + '.join(controls)} | {FE}"
    fit_ = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
    log.info("[%s] beta=%+.6g se=%.6g p=%.4f %s n=%d", label, coef, se, pval, sig, int(fit_._N))
    results.append({"label": label, "treatment": treat, "coef": coef, "se": se,
                    "pval": pval, "nobs": int(fit_._N)})
    del fit_, sub
    gc.collect()


def main():
    grid = ntm.build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid = ntm.attach_cross_spillover(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    factors = load_own_county_factors()
    grid = grid.merge(factors, on="fips", how="left")

    # Fill missing factors with the national mean so a handful of
    # uncovered counties don't drop out of the FE regression entirely.
    for col in ["own_weight", "car_share", "own_dist_mi"]:
        grid[col] = grid[col].fillna(grid[col].mean())

    grid["own_weight_dosage"] = grid["own_weight"] * grid["night_alert"]
    grid["own_weight_car_dosage"] = grid["own_weight"] * grid["car_share"] * grid["night_alert"]
    grid["own_dist_only_dosage"] = grid["own_dist_mi"] * grid["night_alert"]
    grid["own_full_dosage"] = grid["own_weight"] * grid["car_share"] * grid["own_dist_mi"] * grid["night_alert"]

    results = []
    log.info("=== Own-county dosage variants (all jointly controlled for cross_spillover) ===")
    fit(grid, "raw binary night_alert (headline reference)", "night_alert", results)
    fit(grid, "own_weight x night_alert", "own_weight_dosage", results)
    fit(grid, "own_weight x car_share x night_alert", "own_weight_car_dosage", results)
    fit(grid, "own_dist_mi x night_alert (distance only)", "own_dist_only_dosage", results)
    fit(grid, "own_weight x car_share x own_dist_mi x night_alert (full dosage)", "own_full_dosage", results)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_own_county_dosage.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
