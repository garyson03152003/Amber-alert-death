"""
run_hours_since_wake_distance_nonalert.py
=============================================================
The most demanding combined robustness test for the H2 sleep/spillover
mechanism yet: TRACT-level distance x TRACT-level car share x
hours-since-wake x non-alert-affected-counties, all four dimensions at
once.

Prior checks each raised a distinct concern about the pooled
CROSS_SPILLOVER result:
  1. Distance split (reg_commuting_distance_robustness.csv): the LEVEL
     effect requires short commuting pairs (<21mi, beta=+0.034, p=0.008)
     and is null for long ones (>=21mi, beta=+0.0021, p=0.921).
  2. Non-alert-affected-counties restriction
     (reg_night_to_morning_spillover_nonalert_only.csv): the LEVEL effect
     is undetectable (p=0.46) once restricted to night_alert==0 days,
     because ~65% of alerts are statewide broadcasts that geo-expand to
     every county in the state, entangling "directly alerted" with "high
     spillover exposure" on the highest-dose observations.
  3. Road-type x hours-since-wake (reg_road_type_hours_since_wake.csv):
     the pooled dose-response's rising SLOPE (+0.000461, p=0.018) is not
     statistically distinguishable between highway and non-highway
     (interaction p=0.183) -- inconclusive on road-type attribution.
  4. TRUE-JOINT tract-preserved car x distance dosage
     (reg_commuting_car_distance_dosage.csv): build_lodes_tract_car_
     dosage.py computes E[car_share x dist] directly from TRACT-level
     LODES home->work flows before ever collapsing to county pairs (so
     it reflects the true within-pair correlation between which tracts
     are short/high-car-share vs long/low-car-share, not the product of
     two separately-averaged county marginals). Its pooled-day level
     effect is significant: coef=+0.000463, se=0.000107, p=6.9e-05.

This script crosses check 4's tract-preserved car x distance dosage with
checks 2 and 3's dimensions: for each of the 18 single-hour bins
(06:00-23:59, matching run_hours_since_wake_dose_response.py), restrict
the sample to night_alert==0 county-days (check 2) and use the
weight x avg_car_x_dist quantity (check 4, "weight_x_TRUE_car_x_dist" in
run_commuting_car_distance_dosage.py) as the spillover regressor, then
run the same inverse-variance-weighted meta-regression slope test used
throughout. The plain (unweighted) cross_spillover, also night_alert==0-
restricted, is estimated alongside as a same-sample reference point.

If a genuine, uncontaminated commuting mechanism exists -- a driver who
actually lives far away, actually commutes by car, from a home county
that alerted, into a work county that was not itself swept into the same
statewide campaign -- this is the most targeted combination in this repo
to detect it: a positive, rising-with-hours-since-wake dose-response in
the tract-preserved car x distance measure, restricted to night_alert==0.

Design/memory: same fips_year + fips_dow + month_str FE, two-way
state+date clustering, and categorical-dtype grid construction as
run_night_to_morning_leave_one_out.py (needed to safely re-estimate many
times over the ~7.3M-row national grid without repeating the earlier OOM).

Output: output/tables/reg_hours_since_wake_distance_nonalert.csv
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
import run_commuting_distance_robustness as dist_mod
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("hsw_distance_nonalert")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
WAKE_HOUR = 6
HOURS = list(range(6, 24))
LODES_CAR_PATH = DATA_PROC / "commuting" / "county_pair_lodes_car_dosage.parquet"


def build_hour_outcomes(active) -> dict:
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly = hourly[hourly["fips"].isin(active)]
    out = {}
    for h in HOURS:
        window = hourly[hourly["hour"] == h]
        agg = (window.groupby(["fips", "date"])
               .agg(fatals=("person_fatals", "sum")).reset_index())
        out[h] = agg
    log.info("Built %d single-hour outcome tables (hours %d-%d)", len(out), HOURS[0], HOURS[-1])
    return out


def build_true_joint_weights():
    """weight_x_TRUE_car_x_dist: tract-preserved car-share x distance
    dosage, same construction as run_commuting_car_distance_dosage.py."""
    weights_dist, median_dist = dist_mod.build_weights_with_distance()
    lodes_car = pd.read_parquet(LODES_CAR_PATH)
    weights_dist = weights_dist.merge(
        lodes_car[["fips_home", "fips_work", "avg_car_x_dist"]].rename(
            columns={"fips_home": "fips_home_s", "fips_work": "fips_work_s"}),
        on=["fips_home_s", "fips_work_s"], how="left")
    n_missing = weights_dist["avg_car_x_dist"].isna().sum()
    log.info("TRUE joint car_x_dist coverage: %d/%d edges missing (%.1f%%)",
             n_missing, len(weights_dist), 100 * n_missing / len(weights_dist))
    from build_nhts_car_share_by_distance import car_share_from_distance
    fallback = car_share_from_distance(weights_dist["dist_mi"]) * weights_dist["dist_mi"]
    weights_dist["avg_car_x_dist"] = weights_dist["avg_car_x_dist"].fillna(fallback)
    weights_dist["weight_x_TRUE_car_x_dist"] = weights_dist["weight"] * weights_dist["avg_car_x_dist"]
    return weights_dist


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def fit_one(sub, hours_since_wake, variant, results):
    s = sub.dropna(subset=["_spill", "fatals"])
    formula = "fatals ~ _spill | " + FE
    fit_ = pf.feols(formula, data=s, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc["_spill"]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    log.info("[%-22s] hours_since_wake=%2d beta=%+.6g se=%.6g p=%.4f %s n=%d",
              variant, hours_since_wake, coef, se, pval, _sig(pval), int(fit_._N))
    results.append({"variant": variant, "hour": hours_since_wake + WAKE_HOUR,
                    "hours_since_wake": hours_since_wake, "coef": coef, "se": se, "pval": pval,
                    "nobs": int(fit_._N)})
    del fit_, s
    gc.collect()


def weighted_slope_test(df: pd.DataFrame, label: str):
    w = 1.0 / (df["se"] ** 2)
    x = df["hours_since_wake"].to_numpy()
    y = df["coef"].to_numpy()
    W = w.to_numpy()
    xbar = np.average(x, weights=W)
    ybar = np.average(y, weights=W)
    slope = np.sum(W * (x - xbar) * (y - ybar)) / np.sum(W * (x - xbar) ** 2)
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = len(x) - 2
    sigma2 = np.sum(W * resid ** 2) / dof
    se_slope = np.sqrt(sigma2 / np.sum(W * (x - xbar) ** 2))
    from scipy import stats
    tstat = slope / se_slope
    pval = 2 * stats.t.sf(abs(tstat), dof)
    log.info("[%s] weighted meta-regression: slope=%+.6g se=%.6g p=%.4f (n=%d hours, dof=%d)",
             label, slope, se_slope, pval, len(x), dof)
    return slope, se_slope, pval


def main():
    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (fars.assign(year=fars["date"].dt.year)
                   .groupby(["fips", "year"])[fatals_col].sum().groupby("fips").mean())
    active = set(mean_annual[mean_annual >= ntm.MIN_FATALS_PER_YEAR].index)
    log.info("Active (>=%d fatals/yr) counties: %d", ntm.MIN_FATALS_PER_YEAR, len(active))

    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")

    # Alert SOURCE events (home counties/dates) must come from the full,
    # unrestricted panel -- built once, before the night_alert==0 row
    # restriction below (which applies to RECIPIENT rows only).
    full_alert_grid = pd.MultiIndex.from_product([sorted(active), dates], names=["fips", "date"]).to_frame(index=False)
    full_alert_grid = ntm.attach_night_alert(full_alert_grid)
    alert_events = full_alert_grid.loc[full_alert_grid["night_alert"] > 0, ["fips", "date"]].copy()
    alert_events["fips_home"] = alert_events["fips"].astype(int)
    del full_alert_grid
    gc.collect()

    base_grid = pd.MultiIndex.from_product([sorted(active), dates], names=["fips", "date"]).to_frame(index=False)
    base_grid = ntm.attach_night_alert(base_grid)
    base_grid["year_str"] = base_grid["date"].dt.year.astype(str)
    base_grid["dow"] = base_grid["date"].dt.dayofweek.astype(str)
    base_grid["month_str"] = base_grid["date"].dt.month.astype("category")
    base_grid["fips_dow"] = (base_grid["fips"] + "_" + base_grid["dow"]).astype("category")
    base_grid["fips_year"] = (base_grid["fips"] + "_" + base_grid["year_str"]).astype("category")
    base_grid["state_code"] = base_grid["fips"].str[:2].astype("category")
    base_grid["date_str"] = base_grid["date"].dt.strftime("%Y-%m-%d").astype("category")

    n_before = len(base_grid)
    base_grid = base_grid[base_grid["night_alert"] == 0].drop(columns=["night_alert", "year_str", "dow"])
    log.info("Restricted to night_alert==0 (recipient NOT itself alerted): %d rows (dropped %d)",
             len(base_grid), n_before - len(base_grid))
    gc.collect()

    fips_in_sample = set(base_grid["fips"].unique())
    grid_index = pd.MultiIndex.from_frame(base_grid[["fips", "date"]])

    plain_weights = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    plain_weights = plain_weights[plain_weights["fips_home"] != plain_weights["fips_work"]].copy()
    true_joint_weights = build_true_joint_weights()

    hour_outcomes = build_hour_outcomes(active)

    results = []
    for h in HOURS:
        hsw = h - WAKE_HOUR
        merged = base_grid.merge(hour_outcomes[h], on=["fips", "date"], how="left")
        merged["fatals"] = merged["fatals"].fillna(0).astype("float32")

        for variant_label, w_subset, value_col in [
            ("plain_cross_spillover", plain_weights, "weight"),
            ("TRUE_car_x_dist_tract", true_joint_weights, "weight_x_TRUE_car_x_dist"),
        ]:
            values, n_pairs, w_sum = dist_mod.spillover_series(
                grid_index, alert_events, w_subset, fips_in_sample, value_col=value_col)
            merged["_spill"] = values
            fit_one(merged, hsw, variant_label, results)
            merged.drop(columns=["_spill"], inplace=True)
        del merged
        gc.collect()
        pd.DataFrame(results).to_csv(OUTPUT_TABS / "reg_hours_since_wake_distance_nonalert.csv", index=False)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_hours_since_wake_distance_nonalert.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    log.info("\n=== Dose-response slope tests, night_alert==0 only ===")
    weighted_slope_test(out[out["variant"] == "plain_cross_spillover"], "PLAIN cross_spillover, night_alert==0")
    weighted_slope_test(out[out["variant"] == "TRUE_car_x_dist_tract"], "TRUE car x dist (tract), night_alert==0")


if __name__ == "__main__":
    main()
