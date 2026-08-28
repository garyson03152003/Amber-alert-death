"""
run_same_hour_road_type_split.py
=============================================================
Does the same-hour ("immediate distraction") null result hold up when
fatal crashes are split by road type -- highway (Interstate + freeway/
expressway) versus everything else (arterials, collectors, local roads)?

Motivation: sustained highway driving is the classic drowsy/distraction
crash scenario in the literature (long monotonous stretches, higher
speed, less frequent braking/attention demand than stop-and-go local
roads). If an immediate-distraction reaction to an AMBER alert exists at
all, it might plausibly be concentrated on highways rather than local
roads -- or the reverse could hold if the mechanism is more about
glancing at a phone in slow traffic. Either way, pooling both road types
together (as the main same-hour test does) could mask a real effect
concentrated in one road type.

Uses the exact same matched-referent case-crossover design validated in
run_same_hour_event_study.py (exact hour + exact day-of-week referents at
+/-7/14/21/28-day offsets, fips_hour_dow + fips_year + year_month FE,
two-way state+date clustering) -- only the outcome source changes, from
fars_hourly_county_day.parquet (all roads pooled) to
fars_road_type_county_day.parquet (already split by FARS's own FUNC_SYS
field, built in build_fars_road_type.py for the H2 commuting-spillover
road-type check). That file has no serious_inj column (FARS road-type
build only carries person_fatals), so this script covers fatals only.

Output: output/tables/reg_same_hour_road_type_split.csv
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
log = get_logger("same_hour_road_type")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

ROAD_TYPE_PATH = DATA_PROC / "fars_road_type_county_day.parquet"
CLUSTER_VARS = base.CLUSTER_VARS


def build_road_type_referent_grid(ev: pd.DataFrame, *, day_match: str) -> pd.DataFrame:
    """Same referent construction as base.build_matched_referent_grid, but
    merges highway_fatals/nonhighway_fatals from the road-type FARS extract
    instead of person_fatals/serious_inj from the pooled hourly file."""
    road = pd.read_parquet(ROAD_TYPE_PATH)
    road["date"] = pd.to_datetime(road["date"])
    wide = (road.pivot_table(index=["fips", "date", "hour"], columns="is_highway",
                              values="person_fatals", fill_value=0)
                 .reset_index())
    wide.columns = ["fips", "date", "hour"] + [
        "highway_fatals" if c is True else "nonhighway_fatals" for c in wide.columns[3:]
    ]
    for col in ("highway_fatals", "nonhighway_fatals"):
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

    referent = referent.merge(wide[["fips", "date", "hour", "highway_fatals", "nonhighway_fatals"]],
                               on=["fips", "date", "hour"], how="left")
    referent["highway_fatals"] = referent["highway_fatals"].fillna(0).astype("float32")
    referent["nonhighway_fatals"] = referent["nonhighway_fatals"].fillna(0).astype("float32")

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

    log.info("Road-type matched-referent grid (day_match=%s): %d rows, %d alert-hour rows",
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

    grid = build_road_type_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    fe = "fips_hour_dow + fips_year + year_month"

    results = []
    log.info("\n=== Same-hour case-crossover, split by road type (robust FE) ===")
    run(grid, "highway fatals", "highway_fatals", "is_alert_hour", fe, results)
    run(grid, "non-highway fatals", "nonhighway_fatals", "is_alert_hour", fe, results)

    log.info("\n=== Backward-causal placebo (tomorrow's alert, controlling for today's real one) ===")
    run(grid, "highway fatals, placebo", "highway_fatals", "is_alert_hour_tomorrow", fe, results,
        extra_controls=["is_alert_hour"])
    run(grid, "non-highway fatals, placebo", "nonhighway_fatals", "is_alert_hour_tomorrow", fe, results,
        extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_road_type_split.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
