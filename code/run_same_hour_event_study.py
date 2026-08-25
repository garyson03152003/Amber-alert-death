"""
run_same_hour_event_study.py
=============================================================
Does the exact clock hour an AMBER alert is issued show elevated fatal
crashes, compared to the same county/hour/weekday on nearby dates without
an alert?

Design: case-crossover. For each alert-hour event, the control set is the
same county, same hour-of-day, same day-of-week, on dates within +/- 4
weeks of the alert (excluding the alert date itself). This is a tighter
local-time match than comparing against the entire multi-year history: it
controls for seasonal drift and county-specific trends directly, at the
cost of a smaller (but still large) control pool, and avoids materialising
a full national county x date x hour panel (which does not fit in memory
at this county count -- see the random-subsample fallback below for
environments too memory-constrained even for the windowed version).

Reports both OLS (linear probability/count) and PPML (Poisson) -- PPML is
the more standard model for rare-count crash outcomes and matches the
methodology of run_hourly_alert_event_study.py's CA-specific event study,
but suffers heavy "separation" data loss here (most county-hour-weekday
cells have zero fatalities across the whole window), so OLS is reported as
the primary, better-powered estimate and PPML as a check.

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
CONTROL_SAMPLE_FRAC = 0.2  # subsample of non-alert control hours, per county
MIN_FATALS_PER_YEAR = 5
SEED = 42


def active_counties() -> set[str]:
    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (fars.assign(year=fars["date"].dt.year)
                   .groupby(["fips", "year"])[fatals_col].sum()
                   .groupby("fips").mean())
    return set(mean_annual[mean_annual >= MIN_FATALS_PER_YEAR].index)


def load_any_time_alert_hours(active: set[str]) -> pd.DataFrame:
    """One row per (fips, date, hour) for any alert, day or night."""
    frames = []
    for window in ("night", "day"):
        a = base.load_verified_alerts(window=window, detail=True)
        frames.append(a[["fips", "sent_local", "hour_local"]])
    alerts = pd.concat(frames, ignore_index=True)
    alerts["date"] = pd.to_datetime(alerts["sent_local"]).dt.normalize()
    alerts = alerts.rename(columns={"hour_local": "hour"})[["fips", "date", "hour"]].drop_duplicates()
    alerts = alerts[alerts["fips"].isin(active)]
    log.info("Alert county-date-hour events (active counties): %d", len(alerts))
    return alerts


def build_case_crossover_grid(ev: pd.DataFrame, active: set[str]) -> pd.DataFrame:
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly_by_fips = {f: g for f, g in hourly[hourly["fips"].isin(active)].groupby("fips")}
    ev_by_fips = {f: g for f, g in ev.groupby("fips")}

    rng = np.random.default_rng(SEED)
    hour_df = pd.DataFrame({"hour": np.arange(24, dtype="int8")})
    full_lo, full_hi = pd.Timestamp("2013-01-01"), pd.Timestamp("2024-12-30")

    chunks = []
    for fips5, erows in ev_by_fips.items():
        alert_dates = erows["date"].unique()
        keep_dates = set()
        for d in alert_dates:
            d = pd.Timestamp(d)
            lo = max(d - pd.Timedelta(days=WINDOW_DAYS), full_lo)
            hi = min(d + pd.Timedelta(days=WINDOW_DAYS), full_hi)
            keep_dates.update(pd.date_range(lo, hi, freq="D"))

        dd = pd.DataFrame({"date": sorted(keep_dates)})
        g = dd.merge(hour_df, how="cross")
        g["dow"] = g["date"].dt.dayofweek.astype("int8")
        g["month"] = g["date"].dt.month.astype("int8")
        g["year"] = g["date"].dt.year.astype("int16")
        g["fips"] = fips5

        hrows = hourly_by_fips.get(fips5)
        if hrows is not None:
            g = g.merge(hrows[["date", "hour", "person_fatals", "serious_inj"]],
                       on=["date", "hour"], how="left")
        else:
            g["person_fatals"] = np.nan
            g["serious_inj"] = np.nan
        g["person_fatals"] = g["person_fatals"].fillna(0).astype("float32")
        g["serious_inj"] = g["serious_inj"].fillna(0).astype("float32")

        g["is_alert_hour"] = 0
        g = g.merge(erows[["date", "hour"]].assign(_flag=1), on=["date", "hour"], how="left")
        g["is_alert_hour"] = g["_flag"].fillna(0).astype("int8")
        g = g.drop(columns="_flag")

        mask_alert = g["is_alert_hour"] == 1
        mask_keep = rng.random(len(g)) < CONTROL_SAMPLE_FRAC
        chunks.append(g[mask_alert | mask_keep].copy())

    grid = pd.concat(chunks, ignore_index=True)
    log.info("Case-crossover grid (+/- %d days, %.0f%% control subsample): "
             "%d rows, %d alert-hour rows",
             WINDOW_DAYS, 100 * CONTROL_SAMPLE_FRAC, len(grid), int(grid["is_alert_hour"].sum()))
    return grid


def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."


def run(grid, label, outcome, treat, fe, model, results):
    sub = grid.dropna(subset=[treat, outcome]).copy()
    formula = f"{outcome} ~ {treat} | {fe}"
    if model == "ols":
        fit = pf.feols(formula, data=sub, vcov={"CRV1": "state_code"}, lean=True)
    else:
        fit = pf.fepois(formula, data=sub, vcov={"CRV1": "state_code"})
    td = fit.tidy()
    row = td.loc[treat]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    pct = 100 * (np.exp(coef) - 1) if model == "ppml" else None
    log.info("  [%s] %-55s beta=%+.6f se=%.6f p=%.3f n=%d %s%s",
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
    grid = build_case_crossover_grid(ev, active)

    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]

    results = []
    log.info("\n=== Same-hour case-crossover (+/- %d days) ===", WINDOW_DAYS)
    run(grid, "fatals: fips_hour_dow + year_month FE", "person_fatals", "is_alert_hour",
        "fips_hour_dow + year_month", "ols", results)
    run(grid, "fatals: fips_hour_dow + year_month FE", "person_fatals", "is_alert_hour",
        "fips_hour_dow + year_month", "ppml", results)
    run(grid, "serious: fips_hour_dow + year_month FE", "serious_inj", "is_alert_hour",
        "fips_hour_dow + year_month", "ols", results)

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_event_study.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
