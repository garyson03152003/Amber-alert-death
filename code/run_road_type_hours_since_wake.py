"""
run_road_type_hours_since_wake.py
=============================================================
The pooled hours-since-wake dose-response test (run_hours_since_wake_
dose_response.py) finds a rising CROSS_SPILLOVER effect across the day:
weighted meta-regression slope=+0.000461, p=0.018, across 18 single-hour
bins (06:00-23:59). run_road_type_split.py crossed road type with only
FOUR coarse time-of-day windows and found the effect concentrated outside
the actual morning/evening commute windows (evening/night non-commute,
19:00-24:00, was the one nominally significant cell) -- hard to reconcile
with a cross-county highway-commuting mechanism, and too coarse to see
the dose-response SHAPE by road type.

This re-runs the fine-grained 18-single-hour dose-response separately for
highway_fatals and nonhighway_fatals (fars_road_type_county_day.parquet,
the same FUNC_SYS-based split used in run_road_type_split.py and
run_same_hour_road_type_split.py -- fatals only, no serious_inj column
available at this split), to see whether the rising slope specifically
lives in highway crashes (consistent with a driving/highway-exposure
channel) or is present regardless of road type (more consistent with a
generic time-of-day confound riding on all crash types together).

Design: identical to run_hours_since_wake_dose_response.py (fips_year +
fips_dow + month_str FE, two-way state+date clustering, inverse-variance-
weighted meta-regression of the 18 per-hour point estimates on
hours_since_wake) -- only the outcome source changes to the road-type
split, and only CROSS_SPILLOVER is estimated per hour (OWN night_alert
was null in every slice checked so far in this repo and is not the
question here). Fixed-effect/cluster columns are cast to categoricals
(as in run_night_to_morning_leave_one_out.py) to stay well under the
memory ceiling across 36 sequential regressions.

Output: output/tables/reg_road_type_hours_since_wake.csv
  (per-hour, per-road-type point estimates)
  + weighted slope-test summary logged at the end (not written to CSV,
  same convention as run_hours_since_wake_dose_response.py)
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
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("road_type_hours_since_wake")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
WAKE_HOUR = 6
HOURS = list(range(6, 24))
ROAD_TYPE_PATH = DATA_PROC / "fars_road_type_county_day.parquet"


def build_hour_road_outcomes(active) -> dict:
    """{(hour, road_label): fips/date-indexed fatals} for highway/non-highway."""
    road = pd.read_parquet(ROAD_TYPE_PATH)
    road["date"] = pd.to_datetime(road["date"])
    road = road[road["fips"].isin(active)]

    out = {}
    for h in HOURS:
        window = road[road["hour"] == h]
        for road_label, is_hw in [("highway", True), ("non_highway", False)]:
            sub = window[window["is_highway"] == is_hw]
            agg = (sub.groupby(["fips", "date"])
                   .agg(fatals=("person_fatals", "sum")).reset_index())
            out[(h, road_label)] = agg
    log.info("Built %d (hour x road_type) outcome tables", len(out))
    return out


def fit(grid, label, hours_since_wake, road_label, results):
    controls = ["cross_spillover", "night_alert"]
    sub = grid.dropna(subset=controls + ["fatals"])
    formula = f"fatals ~ {' + '.join(controls)} | {FE}"
    fit_ = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc["cross_spillover"]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
    log.info("[%-11s] hours_since_wake=%2d beta=%+.6f se=%.6f p=%.4f %s",
             road_label, hours_since_wake, coef, se, pval, sig)
    results.append({"label": label, "road_type": road_label, "hour": hours_since_wake + WAKE_HOUR,
                    "hours_since_wake": hours_since_wake, "coef": coef, "se": se, "pval": pval,
                    "nobs": int(fit_._N)})
    del fit_, sub
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
    n = len(x)
    dof = n - 2
    sigma2 = np.sum(W * resid ** 2) / dof
    se_slope = np.sqrt(sigma2 / np.sum(W * (x - xbar) ** 2))
    from scipy import stats
    tstat = slope / se_slope
    pval = 2 * stats.t.sf(abs(tstat), dof)
    log.info("[%s] weighted meta-regression: slope=%+.6f se=%.6f p=%.4f (n=%d hours, dof=%d)",
             label, slope, se_slope, pval, n, dof)
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
    base_grid = pd.MultiIndex.from_product([sorted(active), dates], names=["fips", "date"]).to_frame(index=False)
    base_grid = ntm.attach_night_alert(base_grid)
    base_grid = ntm.attach_cross_spillover(base_grid)
    base_grid["year_str"] = base_grid["date"].dt.year.astype(str)
    base_grid["dow"] = base_grid["date"].dt.dayofweek.astype(str)
    base_grid["month_str"] = base_grid["date"].dt.month.astype("category")
    base_grid["fips_dow"] = (base_grid["fips"] + "_" + base_grid["dow"]).astype("category")
    base_grid["fips_year"] = (base_grid["fips"] + "_" + base_grid["year_str"]).astype("category")
    base_grid["state_code"] = base_grid["fips"].str[:2].astype("category")
    base_grid["date_str"] = base_grid["date"].dt.strftime("%Y-%m-%d").astype("category")
    base_grid["night_alert"] = base_grid["night_alert"].astype("int8")
    base_grid["cross_spillover"] = base_grid["cross_spillover"].astype("float32")
    base_grid = base_grid.drop(columns=["year_str", "dow"])
    gc.collect()

    hour_outcomes = build_hour_road_outcomes(active)

    results = []
    for h in HOURS:
        hsw = h - WAKE_HOUR
        for road_label in ("highway", "non_highway"):
            grid = base_grid.merge(hour_outcomes[(h, road_label)], on=["fips", "date"], how="left")
            grid["fatals"] = grid["fatals"].fillna(0).astype("float32")
            fit(grid, f"{road_label} CROSS_SPILLOVER (own-controlled)", hsw, road_label, results)
            del grid
            gc.collect()
        pd.DataFrame(results).to_csv(OUTPUT_TABS / "reg_road_type_hours_since_wake.csv", index=False)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_road_type_hours_since_wake.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    log.info("\n=== Dose-response slope tests, by road type ===")
    weighted_slope_test(out[out["road_type"] == "highway"], "HIGHWAY CROSS_SPILLOVER")
    weighted_slope_test(out[out["road_type"] == "non_highway"], "NON-HIGHWAY CROSS_SPILLOVER")


if __name__ == "__main__":
    main()
