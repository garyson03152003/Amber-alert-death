"""
run_road_type_split.py
=============================================================
Tests the road-type hypothesis for the own-county null, CROSSED with
time-of-day (per the follow-up request to add a time-of-day control to
the sleep-channel analysis): does the commuting-spillover effect
concentrate on HIGHWAY crashes (the sustained, monotonous driving
classic drowsy-driving research points to) rather than local-road
crashes -- AND does the hours-since-wake rise (run_hours_since_wake_
dose_response.py: cross_spillover slope +0.000461, p=0.018) hold up
specifically WITHIN highway crashes, or does it wash out once road type
is held fixed?

Rather than one pooled panel with both a road-type and an hour-window
dimension (which would revisit the ~130M-row scale that already OOM-
killed a simpler version of this analysis once), this crosses
build_fars_road_type.py's highway/non-highway split with
run_commute_hour_window_split.py's four time-of-day windows as 8
SEPARATE daily-aggregate outcomes, each on the same ~7.3M-row grid that
has run reliably throughout -- 8 small regressions instead of 1 giant one.

Output: output/tables/reg_road_type_split.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from run_commute_hour_window_split import WINDOWS
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("road_type_split")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
ROAD_TYPE_PATH = DATA_PROC / "fars_road_type_county_day.parquet"


def build_outcomes(active) -> dict:
    df = pd.read_parquet(ROAD_TYPE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["fips"].isin(active)]

    out = {}
    for road_label, is_hw in [("highway", True), ("non_highway", False)]:
        road_sub = df[df["is_highway"] == is_hw]
        for window_label, (lo, hi) in WINDOWS.items():
            hour_sub = road_sub[road_sub["hour"].between(lo, hi)]
            agg = (hour_sub.groupby(["fips", "date"])
                   .agg(fatals=("person_fatals", "sum")).reset_index())
            key = f"{road_label} | {window_label}"
            out[key] = agg
            log.info("%-45s %d county-dates, total fatals=%d", key, len(agg), agg["fatals"].sum())
    return out


def fit(grid, label, treat, extra_controls, results):
    controls = [treat] + extra_controls
    sub = grid.dropna(subset=controls + ["fatals"]).copy()
    formula = f"fatals ~ {' + '.join(controls)} | {FE}"
    fit_ = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
    log.info("[%s] beta=%+.6f se=%.6f p=%.4f %s n=%d", label, coef, se, pval, sig, int(fit_._N))
    results.append({"label": label, "coef": coef, "se": se, "pval": pval, "nobs": int(fit_._N)})
    del fit_, sub
    gc.collect()


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

    outcomes = build_outcomes(active)

    results = []
    for key, agg in outcomes.items():
        grid = base_grid.merge(agg, on=["fips", "date"], how="left")
        grid["fatals"] = grid["fatals"].fillna(0)

        fit(grid, f"{key} -> OWN night_alert (spillover-controlled)", "night_alert",
            ["cross_spillover"], results)
        fit(grid, f"{key} -> CROSS_SPILLOVER (own-controlled)", "cross_spillover",
            ["night_alert"], results)

        del grid
        gc.collect()
        pd.DataFrame(results).to_csv(OUTPUT_TABS / "reg_road_type_split.csv", index=False)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_road_type_split.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
