"""
run_commute_hour_window_split.py
=============================================================
Does the commuting-spillover effect concentrate in actual commute hours
(morning rush, when a sleep-disrupted commuter from an alerted county
would be driving in) rather than being spread flat across the whole
06:00-23:59 outcome window used so far?

This is the sharper version of the mechanism test: "sleep-disrupted
commuter drives to work the next morning" specifically predicts an
effect concentrated in the morning commute, and a weaker echo in the
evening commute (residual fatigue driving home) -- not a uniform effect
across midday and late evening, when the person doing the alleged
commuting-linked driving mostly isn't on the road for that reason.

Splits the existing 06:00-23:59 outcome window into four non-overlapping
bins:
    06:00-09:59  morning commute
    10:00-15:59  midday (non-commute)
    16:00-18:59  evening commute
    19:00-23:59  evening/night (non-commute)

Same headline spec as run_night_to_morning_window.py: own-controlled
cross_spillover -> fatals, county x year + county x weekday + month FE,
two-way (state+date) clustering.

Output: output/tables/reg_commute_hour_window_split.csv
"""
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
log = get_logger("commute_hour_split")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
WINDOWS = {
    "morning_commute (06-10)": (6, 9),
    "midday_non_commute (10-16)": (10, 15),
    "evening_commute (16-19)": (16, 18),
    "evening_night_non_commute (19-24)": (19, 23),
}


def build_window_outcomes(active) -> dict:
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly = hourly[hourly["fips"].isin(active)]

    out = {}
    for label, (lo, hi) in WINDOWS.items():
        window = hourly[hourly["hour"].between(lo, hi)]
        agg = (window.groupby(["fips", "date"])
               .agg(fatals=("person_fatals", "sum"), serious=("serious_inj", "sum"))
               .reset_index())
        out[label] = agg
        log.info("%-35s hours %02d-%02d: %d county-dates with data",
                 label, lo, hi, len(agg))
    return out


def fit(grid, label, outcome, treat, extra_controls, results):
    controls = [treat] + extra_controls
    sub = grid.dropna(subset=controls + [outcome]).copy()
    formula = f"{outcome} ~ {' + '.join(controls)} | {FE}"
    fit_ = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit_.tidy().loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
    log.info("[%s] beta=%+.6f se=%.6f p=%.4f %s n=%d", label, coef, se, pval, sig, int(fit_._N))
    results.append({"window": label, "coef": coef, "se": se, "pval": pval, "nobs": int(fit_._N)})


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

    window_outcomes = build_window_outcomes(active)

    results = []
    for label, agg in window_outcomes.items():
        grid = base_grid.merge(agg, on=["fips", "date"], how="left")
        grid["fatals"] = grid["fatals"].fillna(0)
        grid["serious"] = grid["serious"].fillna(0)

        fit(grid, f"{label} -> fatals: OWN night_alert (spillover-controlled)",
            "fatals", "night_alert", ["cross_spillover"], results)
        fit(grid, f"{label} -> fatals: CROSS_SPILLOVER (own-controlled)",
            "fatals", "cross_spillover", ["night_alert"], results)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_commute_hour_window_split.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
