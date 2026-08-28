"""
run_night_to_morning_cutoff_sensitivity.py
=============================================================
How sensitive is the night-to-morning sleep-disruption result
(run_night_to_morning_window.py) to the exact hour used to define "night"?

The sleep-disruption hypothesis needs two boundary choices that are
substantively motivated but not sharp:
  - night_start: the evening hour after which an alert counts as
    "overnight" (and is dated to the FOLLOWING day's driving). Swept over
    20, 21, 22, 23.
  - night_end: the morning hour at which "night" ends and the outcome
    window of interest (waking-hours crashes) begins. Swept over
    4, 5, 6, 7.

  (A night_start of 0/midnight is not a coherent choice under this design:
  "night" is defined as [night_start, 24) U [0, night_end), so night_start=0
  would make every hour "night" -- there would be no evening boundary left
  to define. load_verified_alerts rejects it explicitly. The swept grid
  below is therefore the full set of substantively defensible cutoffs.)

For each of the 4 x 4 = 16 (night_start, night_end) combinations this
re-derives the exposure and outcome variables from scratch (the outcome
window shifts with night_end -- e.g. night_end=4 means the outcome window
is 04:00-23:59) and re-runs the two headline robust-FE specifications from
run_night_to_morning_window.py:
  1. Own night_alert -> fatals, controlling for cross_spillover
  2. Cross_spillover (commuting-weighted exposure to a neighboring county's
     alert) -> fatals, controlling for own night_alert
both with county x year + county x weekday + month fixed effects.

Output: output/tables/reg_night_to_morning_cutoff_sensitivity.csv
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("cutoff_sensitivity")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

NIGHT_STARTS = [20, 21, 22, 23]
NIGHT_ENDS = [4, 5, 6, 7]


def active_counties_and_dates():
    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (fars.assign(year=fars["date"].dt.year)
                   .groupby(["fips", "year"])[fatals_col].sum()
                   .groupby("fips").mean())
    active = mean_annual[mean_annual >= ntm.MIN_FATALS_PER_YEAR].index.tolist()
    log.info("Active (>=%d fatals/yr) counties: %d", ntm.MIN_FATALS_PER_YEAR, len(active))
    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")
    return active, dates


def build_day_agg_by_night_end(hourly: pd.DataFrame) -> dict:
    """One outcome aggregate per candidate night_end: sum of fatals/serious
    injuries in hours [night_end, 23] on each (fips, date)."""
    out = {}
    for ne in NIGHT_ENDS:
        window = hourly[hourly["hour"].between(ne, 23)]
        agg = (window.groupby(["fips", "date"])
               .agg(fatals=("person_fatals", "sum"), serious=("serious_inj", "sum"))
               .reset_index())
        out[ne] = agg
        log.info("night_end=%d: outcome window hours %d-23, %d county-dates with data",
                 ne, ne, len(agg))
    return out


def run_one(active, dates, day_agg_by_ne, weights, night_start, night_end, results):
    label_ne = f"{night_end:02d}"
    day_agg = day_agg_by_ne[night_end]
    grid = pd.MultiIndex.from_product([active, dates], names=["fips", "date"]).to_frame(index=False)
    grid = grid.merge(day_agg, on=["fips", "date"], how="left")
    grid["fatals"] = grid["fatals"].fillna(0)
    grid["serious"] = grid["serious"].fillna(0)

    alerts = base.load_verified_alerts(window="night", night_start=night_start, night_end=night_end)
    ev = alerts.rename(columns={"effective_crash_date": "date"})[["fips", "date"]].drop_duplicates()
    ev["night_alert"] = 1
    ev["date"] = pd.to_datetime(ev["date"])
    grid = grid.merge(ev, on=["fips", "date"], how="left")
    grid["night_alert"] = grid["night_alert"].fillna(0).astype(int)
    n_events = int(grid["night_alert"].sum())

    grid = ntm.attach_cross_spillover(grid, home_alert_col="night_alert",
                                      out_col="cross_spillover", weights=weights)

    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    fe = "fips_year + fips_dow + month_str"

    combo_label = f"night_start={night_start}, night_end={night_end}"
    log.info("=== %s (outcome window %s:00-23:59, %d night-alert county-dates) ===",
             combo_label, label_ne, n_events)

    for treat, extra, outcome, tag in [
        ("night_alert", ["cross_spillover"], "fatals", "own_alert"),
        ("cross_spillover", ["night_alert"], "fatals", "cross_spillover"),
    ]:
        controls = [treat] + extra
        sub = grid.dropna(subset=controls + [outcome]).copy()
        formula = f"{outcome} ~ {' + '.join(controls)} | {fe}"
        fit = pf.feols(formula, data=sub, vcov={"CRV1": "state_code"}, lean=True)
        td = fit.tidy()
        row = td.loc[treat]
        coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
        log.info("  [%s] beta=%+.5f se=%.5f p=%.3f n=%d %s",
                 tag, coef, se, pval, int(fit._N), ntm._sig(pval))
        results.append({
            "night_start": night_start, "night_end": night_end, "term": tag,
            "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N),
            "n_night_alert_events": n_events,
        })


def main():
    active, dates = active_counties_and_dates()
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    day_agg_by_ne = build_day_agg_by_night_end(hourly)

    ntm._ensure_commuting_weights()
    weights = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    log.info("Commuting weights: %d OD pairs, %d work counties",
             len(weights), weights["fips_work"].nunique())

    results = []
    for night_start in NIGHT_STARTS:
        for night_end in NIGHT_ENDS:
            run_one(active, dates, day_agg_by_ne, weights, night_start, night_end, results)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_night_to_morning_cutoff_sensitivity.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
