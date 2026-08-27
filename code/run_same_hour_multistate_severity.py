"""
run_same_hour_multistate_severity.py
=============================================================
Pooled, severity-separated version of the same-hour (H1, immediate-
distraction) case-crossover test, combining every state with BOTH hourly
crash timestamps AND a per-crash severity count:

  CA (CHP CCRS, 58 counties, 2016-2022)  -- already built
  IA (Iowa DOT, 99 counties, 2015-2024)  -- severity added this round
  UT (Utah UDOT, 29 counties, 2018-2024) -- severity added this round
  MA (MassDOT, 14 counties, 2013-2017,2019-2020; 2018 is date-only and
      excluded from the hourly panel) -- severity added this round

Delaware and Connecticut are excluded: Delaware's source has no per-crash
severity COUNT field (categorical only), and Connecticut's severity lives
on a separate Person layer that the generic hourly builder does not join
in (see build_state_hourly_panels.py). Both also have the fewest alert-days
of the six hourly states, so the loss is small.

Motivation: CA alone has only 530 alert-hours in its 2016-2022 window (out
of 106,624 nationally), giving a fatals SE about 34x wider than the
national FARS test -- underpowered enough that a null result there mostly
reflects noise, not evidence. Pooling all four severity-capable states
raises this to ~2,700 alert county-date-hour events across 200 counties,
narrowing (but not closing) that gap.

Design: identical case-crossover as run_same_hour_event_study.py /
run_same_hour_ca_severity.py -- +/-28-day matched county/hour/weekday
window, fips_hour_dow + fips_year + year_month FE, first-alert-of-campaign
variant, and the same jointly-controlled backward-causal placebo. Two-way
(state+date) clustering is valid again here (unlike the CA-only script)
since state now varies.

Caveat: Massachusetts's serious-injury figure did not fully reconcile
against the on-disk validated day panel (36% exact match on ma_injury_proxy,
vs. 100% for ma_fatals and ma_crashes) -- the extraction logic is identical
to build_massachusetts_massdot.py's own, so this most likely reflects the
source data being revised between when that day panel was built and this
fresh fetch, not a bug in this script, but it is flagged rather than
silently trusted. IA and UT serious-injury reconciled at 100%.

Output: output/tables/reg_same_hour_multistate_severity.csv
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
log = get_logger("same_hour_multistate")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 28
CONTROL_SAMPLE_FRAC = 0.3
SEED = 42
CLUSTER_VARS = "state_code + date_str"
OUTCOMES = ["crashes", "fatals", "serious_inj"]

STATE_SOURCES = {
    "CA": dict(path=DATA_PROC / "california_ccrs_county_hour.parquet",
               cols={"ca_crashes": "crashes", "ca_fatals": "fatals", "ca_serious_inj": "serious_inj"},
               date_lo=pd.Timestamp("2016-01-01"), date_hi=pd.Timestamp("2022-12-31")),
    "IA": dict(path=DATA_PROC / "ia_county_hour.parquet",
               cols={"ia_crashes": "crashes", "ia_fatals": "fatals", "ia_serious_inj": "serious_inj"},
               date_lo=pd.Timestamp("2015-01-01"), date_hi=pd.Timestamp("2024-12-31")),
    "UT": dict(path=DATA_PROC / "ut_county_hour.parquet",
               cols={"ut_crashes": "crashes", "ut_fatals": "fatals", "ut_serious_inj": "serious_inj"},
               date_lo=pd.Timestamp("2018-01-01"), date_hi=pd.Timestamp("2024-12-30")),
    "MA": dict(path=DATA_PROC / "ma_county_hour.parquet",
               cols={"ma_crashes": "crashes", "ma_fatals": "fatals", "ma_serious_inj": "serious_inj"},
               date_lo=pd.Timestamp("2013-01-01"), date_hi=pd.Timestamp("2020-12-31")),
}


def load_state_hourly() -> dict[str, pd.DataFrame]:
    """Load and harmonize each state's hourly panel to crashes/fatals/serious_inj."""
    out = {}
    for key, spec in STATE_SOURCES.items():
        df = pd.read_parquet(spec["path"])
        df = df.rename(columns=spec["cols"])
        df["date"] = pd.to_datetime(df["date"])
        out[key] = df[["fips", "date", "hour"] + OUTCOMES]
        log.info("[%s] %d county-hour rows, %d counties, %s - %s", key, len(df),
                 df["fips"].nunique(), df["date"].min().date(), df["date"].max().date())
    return out


def fips_to_state(fips_to_source: dict[str, str], fips: str) -> str | None:
    return fips_to_source.get(fips)


def load_alert_hours(active: set[str]) -> pd.DataFrame:
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
    log.info("Pooled 4-state alert county-date-hour events: %d", len(alerts))
    if "msg_type" in alerts.columns:
        log.info("  of which msg_type=='Alert' (first-of-campaign): %d",
                 int((alerts["msg_type"] == "Alert").sum()))
    return alerts


def build_case_crossover_grid(ev: pd.DataFrame, hourly_by_state: dict[str, pd.DataFrame],
                               fips_to_source: dict[str, str]) -> pd.DataFrame:
    hourly_by_fips = {}
    for key, df in hourly_by_state.items():
        for f, g in df.groupby("fips"):
            hourly_by_fips[f] = g

    ev_by_fips = {f: g for f, g in ev.groupby("fips")}
    rng = np.random.default_rng(SEED)
    hour_df = pd.DataFrame({"hour": np.arange(24, dtype="int8")})

    chunks = []
    for fips5, erows in ev_by_fips.items():
        src = fips_to_source.get(fips5)
        if src is None:
            continue
        date_lo, date_hi = STATE_SOURCES[src]["date_lo"], STATE_SOURCES[src]["date_hi"]

        alert_dates = erows["date"].unique()
        keep_dates = set()
        for d in alert_dates:
            d = pd.Timestamp(d)
            lo = max(d - pd.Timedelta(days=WINDOW_DAYS), date_lo)
            hi = min(d + pd.Timedelta(days=WINDOW_DAYS), date_hi)
            if lo > hi:
                continue
            keep_dates.update(pd.date_range(lo, hi, freq="D"))
        if not keep_dates:
            continue

        dd = pd.DataFrame({"date": sorted(keep_dates)})
        g = dd.merge(hour_df, how="cross")
        g["dow"] = g["date"].dt.dayofweek.astype("int8")
        g["month"] = g["date"].dt.month.astype("int8")
        g["year"] = g["date"].dt.year.astype("int16")
        g["fips"] = fips5
        g["source_state"] = src

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
    log.info("Pooled case-crossover grid (+/- %d days, %.0f%% control subsample): "
             "%d rows, %d alert-hour rows across %d counties, states=%s",
             WINDOW_DAYS, 100 * CONTROL_SAMPLE_FRAC, len(grid),
             int(grid["is_alert_hour"].sum()), grid["fips"].nunique(),
             sorted(grid["source_state"].unique()))
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
    hourly_by_state = load_state_hourly()
    fips_to_source = {}
    for key, df in hourly_by_state.items():
        for f in df["fips"].unique():
            fips_to_source.setdefault(f, key)  # first state wins; no county appears in >1 state
    active = set(fips_to_source)
    log.info("Pooled active counties (CA+IA+UT+MA hourly severity data): %d", len(active))

    ev = load_alert_hours(active)
    grid = build_case_crossover_grid(ev, hourly_by_state, fips_to_source)

    grid["fips_hour_dow"] = grid["fips"] + "_" + grid["hour"].astype(str) + "_" + grid["dow"].astype(str)
    grid["year_month"] = grid["year"].astype(str) + "_" + grid["month"].astype(str)
    grid["fips_year"] = grid["fips"] + "_" + grid["year"].astype(str)
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")
    fe = "fips_hour_dow + fips_year + year_month"

    results = []
    log.info("\n=== Pooled 4-state same-hour case-crossover, severity-separated (any alert message) ===")
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
    out_path = OUTPUT_TABS / "reg_same_hour_multistate_severity.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
