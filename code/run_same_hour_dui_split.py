"""
run_same_hour_dui_split.py
=============================================================
Does the same-hour ("immediate distraction") highway-fatals effect
(reg_same_hour_road_type_split.csv: beta=-0.000175, se=0.000035,
p=6.7e-06, n=948,423) concentrate on alcohol-involved crashes rather
than sober ones?

Motivation: an active AMBER alert search plausibly brings heightened
police presence, traffic stops, and public vigilance to an area --
exactly the conditions that would suppress DUI-related crashes
specifically, via deterrence/enforcement, rather than via any driver-
attention or distraction mechanism. If the same-hour effect is really a
DUI-deterrence artifact, it should concentrate in drunk_fatals and be
weak or absent in sober_fatals; if it holds (or is stronger) for sober
driving, that argues against a DUI-specific channel and toward a
broader attention/distraction explanation instead.

Uses the exact same matched-referent case-crossover design validated in
run_same_hour_event_study.py (exact hour + exact day-of-week referents at
+/-7/14/21/28-day offsets, fips_hour_dow + fips_year + year_month FE,
two-way state+date clustering) -- only the outcome source changes, from
fars_hourly_county_day.parquet (all crashes pooled) to
fars_dui_county_day.parquet (split by vehicle-level DR_DRINK==1, built in
build_fars_dui.py). That file has no serious_inj column and no road-type
split, so this covers person_fatals by alcohol involvement only.

Output: output/tables/reg_same_hour_dui_split.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_same_hour_event_study as base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("same_hour_dui")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

DUI_PATH = DATA_PROC / "fars_dui_county_day.parquet"
CLUSTER_VARS = base.CLUSTER_VARS


def build_dui_referent_grid(ev: pd.DataFrame, *, day_match: str) -> pd.DataFrame:
    """Same referent construction as base.build_matched_referent_grid, but
    merges drunk_fatals/sober_fatals from the DUI-split FARS extract
    instead of person_fatals/serious_inj from the pooled hourly file."""
    dui = pd.read_parquet(DUI_PATH)
    dui["date"] = pd.to_datetime(dui["date"])
    wide = (dui.pivot_table(index=["fips", "date", "hour"], columns="is_drunk",
                             values="person_fatals", fill_value=0)
               .reset_index())
    wide.columns = ["fips", "date", "hour"] + [
        "drunk_fatals" if c is True else "sober_fatals" for c in wide.columns[3:]
    ]
    for col in ("drunk_fatals", "sober_fatals"):
        if col not in wide.columns:
            wide[col] = 0.0

    ev = ev.copy()
    ev["dow"] = ev["date"].dt.dayofweek.astype("int8")
    ev["weekend"] = (ev["dow"] >= 5).astype("int8")
    full_lo, full_hi = pd.Timestamp("2013-01-01"), pd.Timestamp("2024-12-30")

    if day_match == "dow":
        offsets = [-28, -21, -14, -7, 0, 7, 14, 21, 28]
    elif day_match == "weekend":
        offsets = list(range(-base.WINDOW_DAYS, base.WINDOW_DAYS + 1))
    else:
        raise ValueError(day_match)

    frames = []
    for off in offsets:
        tmp = ev[["fips", "hour", "weekend", "date"]].copy()
        tmp["date"] = tmp["date"] + pd.Timedelta(days=off)
        frames.append(tmp)
    referent = pd.concat(frames, ignore_index=True)
    referent = referent[(referent["date"] >= full_lo) & (referent["date"] <= full_hi)]
    referent["dow"] = referent["date"].dt.dayofweek.astype("int8")

    if day_match == "weekend":
        cand_weekend = (referent["dow"] >= 5).astype("int8")
        referent = referent[cand_weekend == referent["weekend"]]

    referent = referent.drop(columns="weekend").drop_duplicates(subset=["fips", "hour", "date"])
    referent["month"] = referent["date"].dt.month.astype("int8")
    referent["year"] = referent["date"].dt.year.astype("int16")

    referent = referent.merge(wide[["fips", "date", "hour", "drunk_fatals", "sober_fatals"]],
                               on=["fips", "date", "hour"], how="left")
    referent["drunk_fatals"] = referent["drunk_fatals"].fillna(0).astype("float32")
    referent["sober_fatals"] = referent["sober_fatals"].fillna(0).astype("float32")

    alert_set = ev[["fips", "date", "hour"]].drop_duplicates()
    alert_set["_flag"] = 1
    referent = referent.merge(alert_set, on=["fips", "date", "hour"], how="left")
    referent["is_alert_hour"] = referent["_flag"].fillna(0).astype("int8")
    referent = referent.drop(columns="_flag")

    placebo = alert_set[["fips", "hour", "date"]].copy()
    placebo["date"] = placebo["date"] - pd.Timedelta(days=1)
    placebo["_flag3"] = 1
    referent = referent.merge(placebo, on=["fips", "hour", "date"], how="left")
    referent["is_alert_hour_tomorrow"] = referent["_flag3"].fillna(0).astype("int8")
    referent = referent.drop(columns="_flag3")

    log.info("DUI matched-referent grid (day_match=%s): %d rows, %d alert-hour rows",
             day_match, len(referent), int(referent["is_alert_hour"].sum()))
    return referent


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def run(grid, label, outcome, treat, fe, results, extra_controls=None):
    controls = [treat] + (extra_controls or [])
    sub = grid.dropna(subset=controls + [outcome]).copy()
    formula = f"{outcome} ~ {' + '.join(controls)} | {fe}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": CLUSTER_VARS}, lean=True)
    td = fit.tidy()
    row = td.loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    log.info("  %-55s beta=%+.6f se=%.6f p=%.3f n=%d %s",
             label, coef, se, pval, int(fit._N), _sig(pval))
    results.append({"label": label, "outcome": outcome, "fe": fe,
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N)})
    del fit, sub, td
    gc.collect()


def main():
    active = base.active_counties()
    log.info("Active (>=%d fatals/yr) counties: %d", base.MIN_FATALS_PER_YEAR, len(active))
    ev = base.load_any_time_alert_hours(active)

    grid = build_dui_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    fe = "fips_hour_dow + fips_year + year_month"

    results = []
    log.info("\n=== Same-hour case-crossover, split by alcohol involvement (robust FE) ===")
    run(grid, "drunk fatals", "drunk_fatals", "is_alert_hour", fe, results)
    run(grid, "sober fatals", "sober_fatals", "is_alert_hour", fe, results)

    log.info("\n=== Backward-causal placebo (tomorrow's alert, controlling for today's real one) ===")
    run(grid, "drunk fatals, placebo", "drunk_fatals", "is_alert_hour_tomorrow", fe, results,
        extra_controls=["is_alert_hour"])
    run(grid, "sober fatals, placebo", "sober_fatals", "is_alert_hour_tomorrow", fe, results,
        extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_dui_split.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
