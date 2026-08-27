"""
run_same_hour_ca_severity.py
=============================================================
Severity-separated version of the same-hour (H1, immediate-distraction)
case-crossover test: does the FARS-only fatal-crash result hold up when
tested against ALL crash severities, not just fatalities?

FARS is a fatal-crash-only census -- it structurally cannot show whether
a same-hour effect exists in the vastly larger population of non-fatal
crashes. California's CHP CCRS data (already built and validated
elsewhere in this repo, data/processed/california_ccrs_county_hour.parquet)
is the one hourly-resolution state crash source in this project that
already carries a severity breakdown (ca_crashes = all severities,
ca_fatals, ca_serious_inj), covering 2016-2022 -- a shorter window than
FARS's 2013-2024, but ~110x more total crash events (3.16M vs 28,868
fatals) and ~60x more serious-injury events than fatal ones in the same
county-hours, i.e. a real power increase for this specific test, at the
cost of external validity beyond California.

Design and robustness set identical to run_same_hour_event_study.py: same
+/-28-day case-crossover, two-way (state+date) clustering, fips_hour_dow
+ fips_year + year_month FE, first-alert-of-campaign-only variant, and
the same jointly-controlled backward-causal placebo -- just re-run
against CA's three severity outcomes instead of FARS's one.

Other states with hourly crash data (MA, UT, CT, DE, IA -- see
build_state_hourly_panels.py) only have a total crash count in their
cached hourly panels; extracting severity for them would require
extending each state's own extraction script and was not attempted here.

Output: output/tables/reg_same_hour_ca_severity.csv
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
log = get_logger("same_hour_ca")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 28
CONTROL_SAMPLE_FRAC = 0.5  # CA alone is much smaller than the national grid
SEED = 42
CLUSTER_VARS = "fips + date_str"  # state_code is constant (CA-only) -- cluster by
# county instead of state for the "spatial" dimension, keeping date as the
# second dimension for same-day cross-county correlation.
CA_HOURLY_PATH = DATA_PROC / "california_ccrs_county_hour.parquet"
OUTCOMES = ["ca_crashes", "ca_fatals", "ca_serious_inj"]


def ca_counties() -> set[str]:
    hourly = pd.read_parquet(CA_HOURLY_PATH, columns=["fips"])
    return set(hourly["fips"].unique())


def load_ca_alert_hours(active: set[str], date_lo, date_hi) -> pd.DataFrame:
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
    alerts = alerts[alerts["fips"].isin(active) & alerts["date"].between(date_lo, date_hi)]
    log.info("CA alert county-date-hour events: %d", len(alerts))
    if "msg_type" in alerts.columns:
        log.info("  of which msg_type=='Alert' (first-of-campaign): %d",
                 int((alerts["msg_type"] == "Alert").sum()))
    return alerts


def build_case_crossover_grid(ev: pd.DataFrame, active: set[str], date_lo, date_hi) -> pd.DataFrame:
    hourly = pd.read_parquet(CA_HOURLY_PATH)
    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly_by_fips = {f: g for f, g in hourly[hourly["fips"].isin(active)].groupby("fips")}
    ev_by_fips = {f: g for f, g in ev.groupby("fips")}

    rng = np.random.default_rng(SEED)
    hour_df = pd.DataFrame({"hour": np.arange(24, dtype="int8")})

    chunks = []
    for fips5, erows in ev_by_fips.items():
        alert_dates = erows["date"].unique()
        keep_dates = set()
        for d in alert_dates:
            d = pd.Timestamp(d)
            lo = max(d - pd.Timedelta(days=WINDOW_DAYS), date_lo)
            hi = min(d + pd.Timedelta(days=WINDOW_DAYS), date_hi)
            keep_dates.update(pd.date_range(lo, hi, freq="D"))

        dd = pd.DataFrame({"date": sorted(keep_dates)})
        g = dd.merge(hour_df, how="cross")
        g["dow"] = g["date"].dt.dayofweek.astype("int8")
        g["month"] = g["date"].dt.month.astype("int8")
        g["year"] = g["date"].dt.year.astype("int16")
        g["fips"] = fips5

        hrows = hourly_by_fips.get(fips5)
        if hrows is not None:
            g = g.merge(hrows[["date", "hour"] + OUTCOMES], on=["date", "hour"], how="left")
        else:
            for col in OUTCOMES:
                g[col] = np.nan
        for col in OUTCOMES:
            g[col] = g[col].fillna(0).astype("float32")

        g["is_alert_hour"] = 0
        g = g.merge(erows[["date", "hour"]].assign(_flag=1).drop_duplicates(["date", "hour"]),
                   on=["date", "hour"], how="left")
        g["is_alert_hour"] = g["_flag"].fillna(0).astype("int8")
        g = g.drop(columns="_flag")

        if "msg_type" in erows.columns:
            first_only = erows.loc[erows["msg_type"] == "Alert", ["date", "hour"]].drop_duplicates()
            first_only["_flag2"] = 1
            g = g.merge(first_only, on=["date", "hour"], how="left")
            g["is_first_alert_hour"] = g["_flag2"].fillna(0).astype("int8")
            g = g.drop(columns="_flag2")
        else:
            g["is_first_alert_hour"] = g["is_alert_hour"]

        placebo_dates = (erows[["date", "hour"]].drop_duplicates()
                        .assign(date=lambda d: d["date"] - pd.Timedelta(days=1), _flag3=1))
        g = g.merge(placebo_dates, on=["date", "hour"], how="left")
        g["is_alert_hour_tomorrow"] = g["_flag3"].fillna(0).astype("int8")
        g = g.drop(columns="_flag3")

        mask_alert = g["is_alert_hour"] == 1
        mask_keep = rng.random(len(g)) < CONTROL_SAMPLE_FRAC
        chunks.append(g[mask_alert | mask_keep].copy())

    grid = pd.concat(chunks, ignore_index=True)
    log.info("CA case-crossover grid (+/- %d days, %.0f%% control subsample): "
             "%d rows, %d alert-hour rows", WINDOW_DAYS, 100 * CONTROL_SAMPLE_FRAC,
             len(grid), int(grid["is_alert_hour"].sum()))
    return grid


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
    log.info("  %-70s beta=%+.6f se=%.6f p=%.3f n=%d %s",
             label, coef, se, pval, int(fit._N), _sig(pval))
    results.append({"label": label, "outcome": outcome, "treatment": treat, "fe": fe,
                    "coef": coef, "se": se, "pval": pval, "nobs": int(fit._N)})
    del fit, sub, td
    gc.collect()


def main():
    active = ca_counties()
    log.info("CA counties with hourly crash data: %d", len(active))
    date_lo, date_hi = pd.Timestamp("2016-01-01"), pd.Timestamp("2022-12-31")

    ev = load_ca_alert_hours(active, date_lo, date_hi)
    grid = build_case_crossover_grid(ev, active, date_lo, date_hi)

    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    fe = "fips_hour_dow + fips_year + year_month"

    results = []
    log.info("\n=== CA same-hour case-crossover, severity-separated (any alert message) ===")
    for outcome in OUTCOMES:
        run(grid, f"{outcome}: any alert message", outcome, "is_alert_hour", fe, results)

    log.info("\n=== First-alert-of-campaign only ===")
    for outcome in OUTCOMES:
        run(grid, f"{outcome}: first-alert-hour only", outcome, "is_first_alert_hour", fe, results)

    log.info("\n=== Backward-causal placebo (tomorrow's alert, controlling for today's real one) ===")
    for outcome in OUTCOMES:
        run(grid, f"{outcome}: placebo (tomorrow's alert)", outcome, "is_alert_hour_tomorrow", fe,
            results, extra_controls=["is_alert_hour"])

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_same_hour_ca_severity.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
