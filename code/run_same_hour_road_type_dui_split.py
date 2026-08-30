"""
run_same_hour_road_type_dui_split.py
=============================================================
Crosses the H1 same-hour highway-fatals effect
(reg_same_hour_road_type_split.csv: beta=-0.000175, se=0.000035,
p=6.7e-06, n=948,423) with alcohol involvement, WITHIN highway crashes
specifically -- the plain DUI split (reg_same_hour_dui_split.csv, no
road-type control) pools highway+non-highway together and came back with
BOTH drunk_fatals and sober_fatals showing the wrong (positive) sign
relative to the highway-only headline number, because the negative
highway effect was diluted/offset by the positive (non-significant)
non-highway coefficient found in the road-type split. This isolates
highway crashes first, then splits THOSE by alcohol involvement, to test
whether the highway-specific effect is a DUI-deterrence artifact
(heightened police presence/enforcement during an active search) or
holds regardless of alcohol involvement.

Uses the exact same matched-referent case-crossover design validated in
run_same_hour_event_study.py / run_same_hour_road_type_split.py (exact
hour + exact day-of-week referents at +/-7/14/21/28-day offsets,
fips_hour_dow + fips_year + year_month FE, two-way state+date
clustering) -- only the outcome source changes to
fars_road_type_dui_county_day.parquet (built in
build_fars_road_type_dui.py), which crosses FARS's FUNC_SYS road-type
field with vehicle-level DR_DRINK==1 alcohol involvement.

Output: output/tables/reg_same_hour_road_type_dui_split.csv
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
log = get_logger("same_hour_road_type_dui")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

SRC_PATH = DATA_PROC / "fars_road_type_dui_county_day.parquet"
CLUSTER_VARS = base.CLUSTER_VARS

CELLS = [
    ("highway_drunk", True, True),
    ("highway_sober", True, False),
    ("nonhighway_drunk", False, True),
    ("nonhighway_sober", False, False),
]


def build_referent_grid(ev: pd.DataFrame, *, day_match: str) -> pd.DataFrame:
    src = pd.read_parquet(SRC_PATH)
    src["date"] = pd.to_datetime(src["date"])
    wide = (src.pivot_table(index=["fips", "date", "hour"], columns=["is_highway", "is_drunk"],
                             values="person_fatals", fill_value=0)
               .reset_index())
    wide.columns = ["fips", "date", "hour"] + [
        {(True, True): "highway_drunk", (True, False): "highway_sober",
         (False, True): "nonhighway_drunk", (False, False): "nonhighway_sober"}[c]
        for c in wide.columns[3:]
    ]
    for label, _, _ in CELLS:
        if label not in wide.columns:
            wide[label] = 0.0

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

    merge_cols = ["fips", "date", "hour"] + [label for label, _, _ in CELLS]
    referent = referent.merge(wide[merge_cols], on=["fips", "date", "hour"], how="left")
    for label, _, _ in CELLS:
        referent[label] = referent[label].fillna(0).astype("float32")

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

    log.info("Road-type x DUI matched-referent grid (day_match=%s): %d rows, %d alert-hour rows",
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

    grid = build_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    fe = "fips_hour_dow + fips_year + year_month"

    results = []
    log.info("\n=== Same-hour case-crossover, road type x alcohol involvement (robust FE) ===")
    for label, _, _ in CELLS:
        run(grid, label, label, "is_alert_hour", fe, results)

    log.info("\n=== Backward-causal placebo (tomorrow's alert, controlling for today's real one) ===")
    for label, _, _ in CELLS:
        run(grid, f"{label}, placebo", label, "is_alert_hour_tomorrow", fe, results,
            extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_road_type_dui_split.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
