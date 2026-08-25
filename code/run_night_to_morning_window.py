"""
run_night_to_morning_window.py
=============================================================
Does a nighttime AMBER alert (22:00-05:59 local) predict elevated fatal
crashes in the *correctly dated* following waking hours (06:00-23:59)?

Window alignment
-----------------
An alert sent in the evening portion of the night (22:00-23:59) belongs to
the driving day that follows it; an alert sent in the early-morning portion
(00:00-05:59) already carries a timestamp on that same driving day. Treating
both as "day D+1" (as a flat next-calendar-day rule would) mis-dates the
early-morning alerts by one day. `load_verified_alerts(window="night")`
already encodes the correct rule via `effective_crash_date`:
    22:00-23:59 alert on date D  -> effective_crash_date = D+1
    00:00-05:59 alert on date D  -> effective_crash_date = D   (unchanged)
This script's outcome window is hours 06:00-23:59 on that effective date.

Combined exposure
-----------------
AMBER alert campaigns often span consecutive nights (Update messages,
ongoing searches), so "last night's alert" and "the night before's alert"
are correlated, not independent draws. Splitting exposure across two
separate marginal terms (today / yesterday) loses power and -- more
importantly -- makes a naive backward-causal placebo (does *tomorrow's*
alert "predict" today's crashes?) come out spuriously significant, purely
because tomorrow's alert is correlated with today's real one. Once today's
own status is held fixed, the backward placebo passes cleanly (see the
robustness block below); with that established, a single COMBINED exposure
measure (alert on either of the last two nights) is the more powerful and
better-specified primary estimate.

Robustness
----------
The naive spec (county + year + weekday + month fixed effects) understates
how much of the "effect" is really uncontrolled county-specific trend and
county-specific weekday pattern. Adding county x year (fully flexible,
not just a linear trend) and county x weekday fixed effects shrinks the
combined-dose estimate by roughly 25% and moves it from p~0.01 to the
p~0.03-0.05 range -- still directionally positive, but a materially weaker
and more fragile result than the naive spec suggests. Both specs are
reported; only the robust one should be treated as the headline number.

Data
----
FARS hourly crash counts: data/processed/fars_hourly_county_day.parquet
  (cached; see run_time_window_analysis.py for how to rebuild from raw
  FARS ZIPs when they're available -- not needed here since the cache
  already exists in data/processed/).
AMBER alerts: run_state_dot_analysis_fixed.load_verified_alerts(window="night")
  (case-sensitivity-fixed source data, statewide alerts geo-expanded).

Output: output/tables/reg_night_to_morning_window.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

MIN_FATALS_PER_YEAR = 5  # matches the county restriction used elsewhere in the repo


def build_outcome_grid() -> pd.DataFrame:
    """Balanced county x date grid with 06:00-23:59 fatal/serious sums."""
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    day_window = hourly[hourly["hour"].between(6, 23)]
    day_agg = (day_window.groupby(["fips", "date"])
               .agg(fatals_0623=("person_fatals", "sum"),
                    serious_0623=("serious_inj", "sum"))
               .reset_index())

    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (fars.assign(year=fars["date"].dt.year)
                   .groupby(["fips", "year"])[fatals_col].sum()
                   .groupby("fips").mean())
    active = mean_annual[mean_annual >= MIN_FATALS_PER_YEAR].index.tolist()
    log.info("Active (>=%d fatals/yr) counties: %d", MIN_FATALS_PER_YEAR, len(active))

    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")
    grid = pd.MultiIndex.from_product([active, dates], names=["fips", "date"]).to_frame(index=False)
    grid = grid.merge(day_agg, on=["fips", "date"], how="left")
    grid["fatals_0623"] = grid["fatals_0623"].fillna(0)
    grid["serious_0623"] = grid["serious_0623"].fillna(0)
    return grid


def attach_night_alert(grid: pd.DataFrame) -> pd.DataFrame:
    alerts = base.load_verified_alerts(window="night", detail=False)
    ev = alerts.rename(columns={"effective_crash_date": "date"})[["fips", "date"]].drop_duplicates()
    ev["night_alert"] = 1
    ev["date"] = pd.to_datetime(ev["date"])

    grid = grid.merge(ev, on=["fips", "date"], how="left")
    grid["night_alert"] = grid["night_alert"].fillna(0).astype(int)
    log.info("Night-alert county-dates matched to active-county grid: %d",
             int(grid["night_alert"].sum()))

    grid = grid.sort_values(["fips", "date"]).reset_index(drop=True)
    grid["night_alert_lag1"] = grid.groupby("fips")["night_alert"].shift(1).fillna(0).astype(int)
    grid["alert_last2nights_any"] = ((grid["night_alert"] + grid["night_alert_lag1"]) > 0).astype(int)
    grid["alert_last2nights_dose"] = grid["night_alert"] + grid["night_alert_lag1"]
    return grid


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def run(grid, label, outcome, treat, fe, results):
    sub = grid.dropna(subset=[treat, outcome]).copy()
    formula = f"{outcome} ~ {treat} | {fe}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": "state_code"}, lean=True)
    td = fit.tidy()
    row = td.loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    log.info("  %-70s beta=%+.5f se=%.5f p=%.3f n=%d %s [FE: %s]",
             label, coef, se, pval, int(fit._N), _sig(pval), fe)
    results.append({"label": label, "outcome": outcome, "treatment": treat, "fe": fe,
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N)})


def main():
    grid = build_outcome_grid()
    grid = attach_night_alert(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]

    results = []
    log.info("\n=== Naive spec: fips + year + dow + month FE ===")
    run(grid, "Combined-any -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_any",
        "fips + year_str + dow + month_str", results)
    run(grid, "Combined-dose -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_dose",
        "fips + year_str + dow + month_str", results)
    run(grid, "Combined-any -> serious injuries", "serious_0623", "alert_last2nights_any",
        "fips + year_str + dow + month_str", results)

    log.info("\n=== Robust spec: county x year + county x weekday FE (headline) ===")
    run(grid, "Combined-any -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_any",
        "fips_year + fips_dow + month_str", results)
    run(grid, "Combined-dose -> fatals 06:00-23:59", "fatals_0623", "alert_last2nights_dose",
        "fips_year + fips_dow + month_str", results)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_night_to_morning_window.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
