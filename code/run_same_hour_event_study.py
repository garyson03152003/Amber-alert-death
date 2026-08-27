"""
run_same_hour_event_study.py
=============================================================
Does the exact clock hour an AMBER alert is issued show elevated fatal
crashes (the "immediate distraction" / H1 mechanism), compared to the
same county/hour/weekday on nearby dates without an alert?

Design: proper time-stratified case-crossover referent sampling. For
every alert-hour event, the control/referent set is the SAME county and
SAME exact hour-of-day, on OTHER nearby dates that match on day type, one
of two ways:

  day_match="dow": the exact same day-of-week, at offsets of
    +/-7,+/-14,+/-21,+/-28 days (the classic case-crossover "time-
    stratified" referent scheme -- up to 8 referent dates per event).
    This is the primary spec (fips_hour_dow FE).
  day_match="weekend": any date in the +/-28-day window whose
    weekend/weekday classification matches the event's own -- a looser
    day match (many more referent dates per event) that trades exact
    weekday matching for a larger, still-relevant control pool. Reported
    as a secondary spec (fips_hour_weekend FE).

Both hold the HOUR fixed exactly; only the day-matching criterion differs
between them. A third possibility -- coarsening the HOUR dimension too
(e.g. to a PEAK/MID/LOW tercile) -- was tried in an earlier round of this
analysis and dropped: it does not hold "same time" in any meaningful
sense, made precision worse rather than better (hour-of-day carries far
more signal than day-of-week -- PEAK hours run ~7x LOW-hour crash volume),
and a version combining it with a coarsened day match held NEITHER
dimension exact. What that earlier round is kept for is the tier
BREAKDOWN below: splitting the (exact-hour, exact-weekday-matched) sample
by which tier the alert hour itself falls into, to test whether the
effect differs by time of day -- this preserves exact hour+weekday
matching throughout; only the sample is subset, not the FE.

Why this replaced the original grid construction
--------------------------------------------------
The original version crossed every date in a county's +/-28-day alert
window with ALL 24 hours, then kept every alert-hour row plus a random
20% subsample of everything else, to keep the national grid from
exceeding memory (13.2M rows). But under fips_hour_dow FE, only cells
that actually contain an alert can identify the treatment coefficient --
checking directly, just 25.7% of that grid's (fips, hour, dow) cells
(72,323 of 281,254) ever contained one. The other ~74% of cells, and most
of the rows in them, were dead weight for this coefficient: present only
because of the "cross with all 24 hours" construction, not because they
were ever a relevant comparison. Building referent dates directly --
only for the (fips, hour) pairs and day-types that alerts actually
touched -- removes that dead weight and the need for random subsampling
entirely, while also being the textbook-correct case-crossover design
rather than an approximation of one.

Robustness additions (matching the standard set for the sleep-channel
analysis elsewhere in this repo):
  1. Two-way (state + date) clustering, not just state -- state-only
     clustering absorbs within-state correlation from statewide alert
     campaigns, but misses same-day correlation across DIFFERENT states.
  2. fips_year FE added alongside fips_hour_dow + year_month, so each
     county can trend independently of the national year-month effect.
  3. "First alert of a campaign only" as a sharper treatment definition
     (msg_type == "Alert", excluding "Update" follow-ups) alongside the
     original "any alert message" definition.
  4. A backward-causal placebo: does a FUTURE alert hour "predict" THIS
     hour's crashes, jointly controlling for whether this hour's own
     alert status is real (alert campaigns are serially correlated
     across hours/days, so a naive placebo without that control can look
     spuriously significant).

Not attempted here: same-day weather matching, DUI-crash exclusion, and
a station-hour traffic-volume control (the last would require building
an entirely new FHWA TMAS pipeline per TRAFFIC_VOLUME_INSTRUCTIONS.md).

Reports both OLS (linear probability/count) and PPML (Poisson) on the
primary spec -- PPML is the more standard model for rare-count crash
outcomes, but suffers heavy "separation" data loss here, so OLS is the
primary, better-powered estimate and PPML a check.

Data
----
FARS hourly crash counts: data/processed/fars_hourly_county_day.parquet
AMBER alerts (any time of day): run_state_dot_analysis_fixed.load_verified_alerts,
  called once for window="night" and once for window="day" and combined,
  since the two windows are mutually exclusive by construction.

Output: output/tables/reg_same_hour_event_study.csv
"""
import gc
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
log = get_logger("same_hour_event_study")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 28
MIN_FATALS_PER_YEAR = 5
CLUSTER_VARS = "state_code + date_str"

# PEAK/MID/LOW hour tercile, from output/tables/national_hourly_volume_profile.csv
# (national crash-volume share by hour): LOW = 22:00-05:00, MID = 06:00-10:00
# & 19:00-21:00, PEAK = 11:00-18:00 -- each exactly 8 hours.
LOW_HOURS = {0, 1, 2, 3, 4, 5, 22, 23}
MID_HOURS = {6, 7, 8, 9, 10, 19, 20, 21}


def active_counties() -> set[str]:
    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (fars.assign(year=fars["date"].dt.year)
                   .groupby(["fips", "year"])[fatals_col].sum()
                   .groupby("fips").mean())
    return set(mean_annual[mean_annual >= MIN_FATALS_PER_YEAR].index)


def load_any_time_alert_hours(active: set[str]) -> pd.DataFrame:
    """One row per (fips, date, hour) for any alert, day or night, with
    msg_type retained so "first alert of a campaign" can be isolated."""
    frames = []
    for window in ("night", "day"):
        a = base.load_verified_alerts(window=window, detail=True)
        cols = ["fips", "sent_local", "hour_local"] + (["msg_type"] if "msg_type" in a.columns else [])
        frames.append(a[cols])
    alerts = pd.concat(frames, ignore_index=True)
    alerts["date"] = pd.to_datetime(alerts["sent_local"]).dt.normalize()
    alerts = alerts.rename(columns={"hour_local": "hour"})
    keep_cols = ["fips", "date", "hour"] + (["msg_type"] if "msg_type" in alerts.columns else [])
    alerts = alerts[keep_cols].drop_duplicates()
    alerts = alerts[alerts["fips"].isin(active)]
    log.info("Alert county-date-hour events (active counties): %d", len(alerts))
    if "msg_type" in alerts.columns:
        log.info("  of which msg_type=='Alert' (first-of-campaign): %d",
                 int((alerts["msg_type"] == "Alert").sum()))
    return alerts


def build_matched_referent_grid(ev: pd.DataFrame, *, day_match: str) -> pd.DataFrame:
    """Time-stratified case-crossover referent grid: same county, same
    exact hour, on OTHER dates matching the event's day type.

    day_match="dow": referent dates are the exact same day-of-week, at
      offsets of +/-7,+/-14,+/-21,+/-28 days (8 referents/event).
    day_match="weekend": referent dates are any date in +/-28 days whose
      weekend/weekday classification matches the event's own.
    """
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])

    ev = ev.copy()
    ev["dow"] = ev["date"].dt.dayofweek.astype("int8")
    ev["weekend"] = (ev["dow"] >= 5).astype("int8")
    full_lo, full_hi = pd.Timestamp("2013-01-01"), pd.Timestamp("2024-12-30")

    if day_match == "dow":
        offsets = [-28, -21, -14, -7, 0, 7, 14, 21, 28]
    elif day_match == "weekend":
        offsets = list(range(-WINDOW_DAYS, WINDOW_DAYS + 1))
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
    # For day_match=="dow" the offsets are multiples of 7, so every
    # candidate date automatically shares the event's exact weekday --
    # no filter needed.

    referent = referent.drop(columns="weekend").drop_duplicates(subset=["fips", "hour", "date"])
    referent["month"] = referent["date"].dt.month.astype("int8")
    referent["year"] = referent["date"].dt.year.astype("int16")

    referent = referent.merge(hourly[["fips", "date", "hour", "person_fatals", "serious_inj"]],
                               on=["fips", "date", "hour"], how="left")
    referent["person_fatals"] = referent["person_fatals"].fillna(0).astype("float32")
    referent["serious_inj"] = referent["serious_inj"].fillna(0).astype("float32")

    alert_set = ev[["fips", "date", "hour"]].drop_duplicates()
    alert_set["_flag"] = 1
    referent = referent.merge(alert_set, on=["fips", "date", "hour"], how="left")
    referent["is_alert_hour"] = referent["_flag"].fillna(0).astype("int8")
    referent = referent.drop(columns="_flag")

    if "msg_type" in ev.columns:
        first_only = ev.loc[ev["msg_type"] == "Alert", ["fips", "date", "hour"]].drop_duplicates()
        first_only["_flag2"] = 1
        referent = referent.merge(first_only, on=["fips", "date", "hour"], how="left")
        referent["is_first_alert_hour"] = referent["_flag2"].fillna(0).astype("int8")
        referent = referent.drop(columns="_flag2")
    else:
        referent["is_first_alert_hour"] = referent["is_alert_hour"]

    placebo = alert_set[["fips", "hour", "date"]].copy()
    placebo["date"] = placebo["date"] - pd.Timedelta(days=1)
    placebo["_flag3"] = 1
    referent = referent.merge(placebo, on=["fips", "hour", "date"], how="left")
    referent["is_alert_hour_tomorrow"] = referent["_flag3"].fillna(0).astype("int8")
    referent = referent.drop(columns="_flag3")

    log.info("Matched-referent grid (day_match=%s, +/-%d days): %d rows, "
             "%d alert-hour rows, %d first-alert-hour rows",
             day_match, WINDOW_DAYS, len(referent), int(referent["is_alert_hour"].sum()),
             int(referent["is_first_alert_hour"].sum()))
    return referent


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def run(grid, label, outcome, treat, fe, model, results, extra_controls=None):
    controls = [treat] + (extra_controls or [])
    sub = grid.dropna(subset=controls + [outcome]).copy()
    formula = f"{outcome} ~ {' + '.join(controls)} | {fe}"
    if model == "ols":
        fit = pf.feols(formula, data=sub, vcov={"CRV1": CLUSTER_VARS}, lean=True)
    else:
        fit = pf.fepois(formula, data=sub, vcov={"CRV1": CLUSTER_VARS})
    td = fit.tidy()
    row = td.loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    pct = 100 * (np.exp(coef) - 1) if model == "ppml" else None
    log.info("  [%s] %-65s beta=%+.6f se=%.6f p=%.3f n=%d %s%s",
             model, label, coef, se, pval, int(fit._N), _sig(pval),
             f" pct_change={pct:+.2f}%" if pct is not None else "")
    results.append({"label": label, "outcome": outcome, "model": model, "fe": fe,
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N),
                    "pct_change": pct})
    del fit, sub, td
    gc.collect()


def main():
    active = active_counties()
    log.info("Active (>=%d fatals/yr) counties: %d", MIN_FATALS_PER_YEAR, len(active))
    ev = load_any_time_alert_hours(active)

    results = []

    # ------------------------------------------------------------------
    # Primary spec: exact hour + exact day-of-week referent matching.
    # ------------------------------------------------------------------
    grid = build_matched_referent_grid(ev, day_match="dow")
    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    grid["hour_tier"] = np.where(grid["hour"].isin(LOW_HOURS), "LOW",
                          np.where(grid["hour"].isin(MID_HOURS), "MID", "PEAK"))
    log.info("FE cell count: fips_hour_dow=%d", grid["fips_hour_dow"].nunique())

    log.info("\n=== Same-hour case-crossover (exact hour + exact dow referents), any alert message ===")
    run(grid, "fatals: fips_hour_dow + year_month FE", "person_fatals", "is_alert_hour",
        "fips_hour_dow + year_month", "ols", results)
    run(grid, "fatals: fips_hour_dow + year_month FE", "person_fatals", "is_alert_hour",
        "fips_hour_dow + year_month", "ppml", results)
    run(grid, "serious: fips_hour_dow + year_month FE", "serious_inj", "is_alert_hour",
        "fips_hour_dow + year_month", "ols", results)

    log.info("\n=== Robust spec: fips_hour_dow + fips_year + year_month FE ===")
    run(grid, "fatals: robust FE", "person_fatals", "is_alert_hour",
        "fips_hour_dow + fips_year + year_month", "ols", results)
    run(grid, "serious: robust FE", "serious_inj", "is_alert_hour",
        "fips_hour_dow + fips_year + year_month", "ols", results)

    log.info("\n=== First-alert-of-campaign only (msg_type=='Alert'), robust FE ===")
    run(grid, "fatals: first-alert-hour only", "person_fatals", "is_first_alert_hour",
        "fips_hour_dow + fips_year + year_month", "ols", results)
    run(grid, "serious: first-alert-hour only", "serious_inj", "is_first_alert_hour",
        "fips_hour_dow + fips_year + year_month", "ols", results)

    log.info("\n=== Backward-causal placebo: does TOMORROW's same-hour alert "
             "predict TODAY's crashes, controlling for today's real alert? ===")
    run(grid, "fatals: placebo (tomorrow's alert), robust FE", "person_fatals",
        "is_alert_hour_tomorrow", "fips_hour_dow + fips_year + year_month", "ols", results,
        extra_controls=["is_alert_hour"])

    log.info("\n=== Heterogeneity: does the alert-hour effect differ across PEAK/MID/LOW "
             "hours? (subsample by hour_tier; exact hour+dow matching preserved) ===")
    for tier in ("PEAK", "MID", "LOW"):
        sub_grid = grid[grid["hour_tier"] == tier]
        n_alert = int(sub_grid["is_alert_hour"].sum())
        log.info("  [%s] %d rows, %d alert-hours", tier, len(sub_grid), n_alert)
        run(sub_grid, f"fatals: {tier}-hour subsample, robust FE", "person_fatals", "is_alert_hour",
            "fips_hour_dow + fips_year + year_month", "ols", results)
        run(sub_grid, f"serious: {tier}-hour subsample, robust FE", "serious_inj", "is_alert_hour",
            "fips_hour_dow + fips_year + year_month", "ols", results)
        del sub_grid
        gc.collect()

    del grid
    gc.collect()

    # ------------------------------------------------------------------
    # Secondary spec: exact hour + weekend/weekday referent matching
    # (looser day match, larger referent pool per event).
    # ------------------------------------------------------------------
    grid_wk = build_matched_referent_grid(ev, day_match="weekend")
    grid_wk["weekend"] = (grid_wk["dow"] >= 5).astype(int)
    grid_wk["fips_hour_weekend"] = (grid_wk["fips"] + "_" + grid_wk["hour"].astype(str) + "_"
                                     + grid_wk["weekend"].astype(str))
    grid_wk["year_month"] = grid_wk["year"].astype(str) + "_" + grid_wk["month"].astype(str)
    grid_wk["fips_year"] = grid_wk["fips"] + "_" + grid_wk["year"].astype(str)
    grid_wk["state_code"] = grid_wk["fips"].str[:2]
    grid_wk["date_str"] = grid_wk["date"].dt.strftime("%Y-%m-%d")
    log.info("FE cell count: fips_hour_weekend=%d", grid_wk["fips_hour_weekend"].nunique())

    log.info("\n=== Secondary spec: exact hour + weekend/weekday referents, robust FE ===")
    run(grid_wk, "fatals: weekend-matched FE", "person_fatals", "is_alert_hour",
        "fips_hour_weekend + fips_year + year_month", "ols", results)
    run(grid_wk, "serious: weekend-matched FE", "serious_inj", "is_alert_hour",
        "fips_hour_weekend + fips_year + year_month", "ols", results)
    run(grid_wk, "fatals: placebo, weekend-matched FE", "person_fatals", "is_alert_hour_tomorrow",
        "fips_hour_weekend + fips_year + year_month", "ols", results,
        extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_event_study.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
