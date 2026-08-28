"""
run_hours_since_wake_dose_response.py
=============================================================
Direct dose-response test of the time-awake fatigue-accumulation
mechanism: does the commuting-spillover (and own-alert) effect on
fatal crashes grow with hours since a typical wake time (06:00), rather
than being flat across the day or tied to a specific commute window?

This replaces run_commute_hour_window_split.py's four coarse windows
with 18 single-hour bins (06:00-06:59 through 23:00-23:59), each
labeled by hours_since_wake = hour - 6 (0..17), then:
  1. reports the own-controlled cross_spillover (and night_alert) effect
     separately for each single hour -- the raw dose-response curve.
  2. runs an inverse-variance-weighted meta-regression of those 18
     point estimates on hours_since_wake, testing whether the slope is
     significantly positive -- a formal test of "does the effect grow
     with time awake" rather than eyeballing 18 numbers.

Why 18 separate regressions instead of one pooled hourly panel: a
single fips x date x hour panel restricted to active counties would be
~130M rows (1,677 counties x ~4,380 dates x 18 hours) -- this session
already hit OOM kills on hourly panels at a fraction of that size, so
this keeps each regression on the same ~7.3M-row daily-aggregate scale
that has run reliably throughout, at the cost of a meta-regression
rather than one pooled interaction term.

Output: output/tables/reg_hours_since_wake_dose_response.csv
  (per-hour point estimates)
  + a slope-test summary logged at the end (not written to CSV, since
  it is two numbers derived from the table above, not a new regression)
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
log = get_logger("hours_since_wake")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
WAKE_HOUR = 6
HOURS = list(range(6, 24))  # 06:00-23:59, single-hour bins


def build_hour_outcomes(active) -> dict:
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly = hourly[hourly["fips"].isin(active)]

    out = {}
    for h in HOURS:
        window = hourly[hourly["hour"] == h]
        agg = (window.groupby(["fips", "date"])
               .agg(fatals=("person_fatals", "sum"))
               .reset_index())
        out[h] = agg
    log.info("Built %d single-hour outcome tables (hours %d-%d)", len(out), HOURS[0], HOURS[-1])
    return out


def fit(grid, label, treat, extra_controls, results, hours_since_wake):
    controls = [treat] + extra_controls
    sub = grid.dropna(subset=controls + ["fatals"]).copy()
    formula = f"fatals ~ {' + '.join(controls)} | {FE}"
    fit_ = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
    log.info("[%s] hours_since_wake=%2d beta=%+.6f se=%.6f p=%.4f %s",
             label, hours_since_wake, coef, se, pval, sig)
    results.append({"label": label, "hour": hours_since_wake + WAKE_HOUR,
                    "hours_since_wake": hours_since_wake, "coef": coef, "se": se, "pval": pval,
                    "nobs": int(fit_._N)})
    del fit_, sub
    gc.collect()


def weighted_slope_test(df: pd.DataFrame, label: str):
    """Inverse-variance-weighted OLS of coef ~ hours_since_wake across
    the per-hour point estimates -- a meta-regression testing the
    dose-response slope."""
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
    base_grid["month_str"] = base_grid["date"].dt.month.astype(str)
    base_grid["fips_dow"] = base_grid["fips"] + "_" + base_grid["dow"]
    base_grid["fips_year"] = base_grid["fips"] + "_" + base_grid["year_str"]
    base_grid["state_code"] = base_grid["fips"].str[:2]
    base_grid["date_str"] = base_grid["date"].dt.strftime("%Y-%m-%d")

    hour_outcomes = build_hour_outcomes(active)

    results = []
    for h in HOURS:
        grid = base_grid.merge(hour_outcomes[h], on=["fips", "date"], how="left")
        grid["fatals"] = grid["fatals"].fillna(0)
        hsw = h - WAKE_HOUR

        fit(grid, "OWN night_alert (spillover-controlled)", "night_alert",
            ["cross_spillover"], results, hsw)
        fit(grid, "CROSS_SPILLOVER (own-controlled)", "cross_spillover",
            ["night_alert"], results, hsw)

        del grid
        gc.collect()
        # Checkpoint after every hour so a crash doesn't lose all progress
        # (this script OOM-killed once already without this).
        pd.DataFrame(results).to_csv(OUTPUT_TABS / "reg_hours_since_wake_dose_response.csv", index=False)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_hours_since_wake_dose_response.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    log.info("\n=== Dose-response slope tests (does the effect grow with hours since wake?) ===")
    weighted_slope_test(out[out["label"] == "CROSS_SPILLOVER (own-controlled)"], "CROSS_SPILLOVER")
    weighted_slope_test(out[out["label"] == "OWN night_alert (spillover-controlled)"], "OWN night_alert")


if __name__ == "__main__":
    main()
