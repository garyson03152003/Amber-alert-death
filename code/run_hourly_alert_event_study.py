"""Hourly event study: do crashes spike in the hour an AMBER alert is issued,
or the hours immediately after?

Why this design
---------------
The county-DAY analysis dilutes any distraction effect across all 24 hours,
including hours before the alert existed. This uses California CCRS
county-HOUR crashes (2016-2022, all 58 counties validated against FARS) to
compare the alert hour itself and each following hour against the hour
*before* the alert.

Specification
-------------
    crashes_cdh ~ sum_k 1{event_hour = k}  |  county x hour-of-day x day-of-week
                                           +  county x date
                                           +  hour-of-day x year-month

``event_hour`` is hours since alert issuance in local time: 0 = the clock
hour the alert was sent, +1 = the next hour, -1 = the hour before (the
omitted reference).

The fixed effects are what make this sharp:

  * county x date absorbs everything about that county on that day --
    weather, holidays, local events. Identification therefore comes only
    from *within-day* variation across hours.
  * county x hour-of-day x day-of-week absorbs each county's typical
    diurnal crash profile *separately by day type*, so the weekday
    commuting peak is not mistaken for an alert effect.
  * hour-of-day x year-month absorbs seasonal shifts in the shape of the
    driving day (daylight, holiday travel).

Why the day-of-week interaction is not optional
-----------------------------------------------
Controlling only for a pooled county x hour-of-day profile fails, and
visibly so. Commuting peaks are a weekday phenomenon -- in CA, 15:00-17:00
carries ~7.5% of weekday crashes but ~6.0% of weekend crashes -- while
alerts are very unevenly distributed across weekdays (Mon 130, Sat 113,
Sun 19). The event-time dummies then absorb "this is a weekday rush hour"
rather than "an alert was sent". Under the pooled spec the pre-period
placebo below fails at p=0.0006; the ``basic`` FE option is retained only so
that failure can be reproduced rather than quietly forgotten.

Pre-alert coefficients (-6..-2) are placebo checks: alerts cannot affect
hours before they were sent, so those must be jointly flat. The joint Wald
test on the leads -- not the individual p-values, which invite a false
positive across a dozen coefficients -- is the arbiter. If the leads are
jointly nonzero, the post-period estimates are not causal and must not be
reported as such.

Balanced-panel requirement
--------------------------
The source county-hour file is sparse -- county-hours with no crash are
absent. They are treated as true zeros and the panel is explicitly expanded
to the full county x date x hour grid and filled with zeros. Fitting on the
sparse file directly would condition on having had a crash and bias
everything.

That zero-fill is licensed by coverage, not assumed: every CA county-year in
2016-2022 is validated in config/accepted_state_years.csv (FARS ratio
1.017-1.027, county-date agreement >= 0.984), so CCRS is reporting
continuously and an absent hour means "nothing was reported", not "we have
no data here". This is exactly the sparse-source trap the crash panel guards
against elsewhere; it is safe here only because coverage is established
first.

Definitional caveat: a zero means no crash meeting California's *reportable*
crash threshold -- it is not a claim that no vehicle contact occurred.
Sub-threshold fender-benders are invisible to this outcome. That is the
right denominator for a crash-severity question but means the design cannot
speak to very minor incidents.

Outputs
-------
output/tables/hourly_alert_event_study.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from state_dot_analysis_core import extract_finite_coefficients, fit_status_row

log = base.log

CA_HOURLY = DATA_PROC / "california_ccrs_county_hour.parquet"
CA_YEARS = (2016, 2022)          # every year validated in accepted_state_years.csv
EVENT_MIN, EVENT_MAX = -6, 6     # hours around issuance
REFERENCE_EVENT_HOUR = -1        # omitted category

OUTCOMES = ["ca_crashes", "ca_serious_inj", "ca_fatals"]


def build_balanced_hourly_panel(
    sparse: pd.DataFrame, *, years: tuple[int, int] = CA_YEARS,
) -> pd.DataFrame:
    """Expand the sparse county-hour file to a full county x date x hour grid.

    Absent county-hours are real zero-crash hours, not missing data, so they
    must be materialised before estimation.
    """
    sparse = sparse.copy()
    sparse["date"] = pd.to_datetime(sparse["date"]).dt.normalize()
    lo, hi = years
    sparse = sparse[sparse["date"].dt.year.between(lo, hi)]
    if sparse.empty:
        raise ValueError(f"no county-hour rows within {lo}-{hi}")

    counties = np.sort(sparse["fips"].unique())
    dates = pd.date_range(f"{lo}-01-01", f"{hi}-12-31", freq="D")
    grid = pd.MultiIndex.from_product(
        [counties, dates, range(24)], names=["fips", "date", "hour"]
    ).to_frame(index=False)

    panel = grid.merge(sparse, on=["fips", "date", "hour"], how="left")
    count_cols = [c for c in panel.columns if c not in {"fips", "date", "hour"}]
    panel[count_cols] = panel[count_cols].fillna(0.0)
    return panel


def attach_event_hours(
    panel: pd.DataFrame, alerts: pd.DataFrame, *,
    event_min: int = EVENT_MIN, event_max: int = EVENT_MAX,
) -> pd.DataFrame:
    """Label each county-hour with hours-since-issuance of the nearest alert.

    Built by expanding each alert into its own [event_min, event_max] window
    and merging on the exact local timestamp, so windows that cross midnight
    are handled naturally. Where windows from two alerts overlap, the closest
    alert wins (smallest |event_hour|).
    """
    panel = panel.copy()
    panel["ts"] = panel["date"] + pd.to_timedelta(panel["hour"], unit="h")

    if alerts.empty:
        panel["event_hour"] = np.nan
        return panel.drop(columns=["ts"])

    a = alerts[["fips", "sent_local"]].copy()
    a["alert_ts"] = pd.to_datetime(a["sent_local"]).dt.floor("h")
    a = a[["fips", "alert_ts"]].drop_duplicates()

    offsets = np.arange(event_min, event_max + 1)
    windows = a.loc[a.index.repeat(len(offsets))].copy()
    windows["event_hour"] = np.tile(offsets, len(a))
    windows["ts"] = windows["alert_ts"] + pd.to_timedelta(windows["event_hour"], unit="h")

    # Closest alert wins where two alert windows overlap.
    windows["abs_event"] = windows["event_hour"].abs()
    windows = (
        windows.sort_values("abs_event")
        .drop_duplicates(subset=["fips", "ts"], keep="first")
        [["fips", "ts", "event_hour"]]
    )

    panel = panel.merge(windows, on=["fips", "ts"], how="left")
    return panel.drop(columns=["ts"])


def add_event_dummies(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One dummy per event hour except the omitted reference."""
    out = panel.copy()
    terms: list[str] = []
    for k in range(EVENT_MIN, EVENT_MAX + 1):
        if k == REFERENCE_EVENT_HOUR:
            continue
        name = f"ev_{'m' if k < 0 else 'p'}{abs(k)}"
        out[name] = (out["event_hour"] == k).astype(int)
        terms.append(name)
    return out, terms


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    dt = pd.to_datetime(out["date"])
    fips = out["fips"].astype(str)
    hour = out["hour"].astype(str)

    out["_county_hourofday"] = fips + "_" + hour
    # Commuting peaks are a weekday phenomenon: in CA, 15:00-17:00 carries
    # ~7.5% of weekday crashes but only ~6.0% on weekends. Pooling all days
    # into one diurnal profile therefore leaves a day-type artifact that the
    # event-time dummies absorb -- alerts are far from uniform across
    # weekdays (Mon 130, Sat 113, Sun 19 in CA), so that artifact lands
    # squarely on the treatment. Interacting hour-of-day with day-of-week
    # compares each hour only against the same hour on the same weekday.
    out["_county_hour_dow"] = out["_county_hourofday"] + "_" + dt.dt.dayofweek.astype(str)
    # Daylight and seasonal travel shift the shape of the day across the
    # year; absorb that separately from the county's level.
    out["_hour_yearmonth"] = hour + "_" + dt.dt.strftime("%Y%m")

    out["_county_date"] = fips + "_" + dt.dt.strftime("%Y%m%d")
    out["_county_str"] = fips
    out["_date_str"] = dt.dt.strftime("%Y-%m-%d")
    return out


# Fixed-effect specifications. "rich" is the preferred one; "basic" is kept
# only so the day-type confound it suffers from can be shown explicitly.
FE_SPECS = {
    "basic": "_county_hourofday + _county_date",
    "rich": "_county_hour_dow + _county_date + _hour_yearmonth",
}


def joint_wald(fit, live: list[str], selected: list[str], label: str) -> dict | None:
    """Jointly test that a subset of event-time coefficients are all zero.

    Testing pre-period leads one at a time invites a false positive: with a
    dozen coefficients, one crossing p<0.05 is expected by chance. The
    placebo check that actually matters is whether the leads are *jointly*
    indistinguishable from zero, which is this Wald test.
    """
    idx = [live.index(t) for t in selected if t in live]
    if not idx:
        return None
    R = np.zeros((len(idx), len(live)))
    for row, col in enumerate(idx):
        R[row, col] = 1.0
    try:
        fit.wald_test(R=R, q=np.zeros(len(idx)))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("joint Wald test (%s) failed: %s", label, exc)
        return None
    stat, pvalue = fit._wald_statistic, fit._p_value
    if not np.isfinite(stat) or not np.isfinite(pvalue):
        return None
    return {
        "record_type": "joint_test", "status": "ok", "test": label,
        "terms": "|".join(selected), "n_restrictions": len(idx),
        "wald_stat": float(stat), "pvalue": float(pvalue),
    }


def run_event_study(panel: pd.DataFrame, outcome: str, terms: list[str],
                    *, fe_spec: str = "rich") -> list[dict]:
    """PPML event study; see FE_SPECS for the fixed-effect structures."""
    sub = _prepare(panel.dropna(subset=[outcome]))
    live = [t for t in terms if sub[t].nunique() >= 2]
    if not live or len(sub) < 1000:
        return [fit_status_row(
            status="skipped", input_n=len(sub), fitted_n=0, zero_share=None,
            terms_requested=tuple(terms), error_reason="insufficient_estimable_sample",
        ) | {"outcome": outcome, "fe_spec": fe_spec, "model": "PPML_hourly_event_study"}]

    zero_share = float((sub[outcome] == 0).mean())
    formula = f"{outcome} ~ {' + '.join(live)} | {FE_SPECS[fe_spec]}"
    log.info("[%s/%s] fitting %s rows; %.1f%% zero hours", outcome, fe_spec,
             f"{len(sub):,}", 100 * zero_share)
    try:
        fit = pf.fepois(formula, data=sub, vcov={"CRV1": "_county_str + _date_str"})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("[%s] event study failed: %s", outcome, exc)
        return [fit_status_row(
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=tuple(terms), error_reason=str(exc),
        ) | {"outcome": outcome, "fe_spec": fe_spec, "model": "PPML_hourly_event_study"}]

    coefficients, produced, errors = extract_finite_coefficients(fit, tuple(live))
    rows = []
    for c in coefficients:
        k = int(c["term"].split("_")[1].replace("m", "-").replace("p", ""))
        rows.append({
            "record_type": "estimate", "status": "ok",
            "model": "PPML_hourly_event_study", "outcome": outcome,
            "term": c["term"], "event_hour": k,
            "beta": c["beta"], "se": c["se"], "pvalue": c["pvalue"],
            "pct_change": float(100 * (np.exp(c["beta"]) - 1)),
            "ci_low_pct": float(100 * (np.exp(c["beta"] - 1.96 * c["se"]) - 1)),
            "ci_high_pct": float(100 * (np.exp(c["beta"] + 1.96 * c["se"]) - 1)),
            "n_obs": int(fit._N), "cluster": "county+date", "fe_spec": fe_spec,
            "reference": f"event_hour={REFERENCE_EVENT_HOUR}",
        })
    # Placebo: leads must be jointly zero, else the design is picking up
    # something other than the alert and the post-period is not credible.
    pre_terms = [f"ev_m{abs(k)}" for k in range(EVENT_MIN, REFERENCE_EVENT_HOUR)]
    post_terms = [f"ev_p{k}" for k in range(0, EVENT_MAX + 1)]
    for selected, label in [(pre_terms, "pre_period_placebo"),
                            (post_terms, "post_period_joint")]:
        test = joint_wald(fit, live, selected, label)
        if test is not None:
            rows.append(test | {"outcome": outcome, "fe_spec": fe_spec,
                                "model": "PPML_hourly_event_study"})

    status = "ok" if not errors else ("partial" if produced else "failed")
    rows.append(fit_status_row(
        status=status, input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=tuple(terms), terms_produced=produced,
        error_reason="; ".join(f"{t}:{r}" for t, r in errors.items()) or None,
    ) | {"outcome": outcome, "fe_spec": fe_spec,
         "model": "PPML_hourly_event_study"})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="day", choices=["day", "night"],
                        help="alert time-of-day window to study")
    parser.add_argument("--outcomes", nargs="*", default=["ca_crashes"],
                        help=f"subset of {OUTCOMES}")
    parser.add_argument("--fe", nargs="*", default=["rich"],
                        choices=list(FE_SPECS), help="fixed-effect specification(s)")
    args = parser.parse_args(argv)

    if not CA_HOURLY.is_file():
        raise FileNotFoundError(f"CA county-hour file not found: {CA_HOURLY}")

    sparse = pd.read_parquet(CA_HOURLY)
    panel = build_balanced_hourly_panel(sparse)
    log.info("Balanced CA county-hour panel: %s rows (%s counties)",
             f"{len(panel):,}", panel["fips"].nunique())

    alerts = base.load_verified_alerts(window=args.window, detail=True)
    alerts = alerts[alerts["state_fips"] == "06"].copy()
    log.info("CA %s alerts in window: %s", args.window, f"{len(alerts):,}")

    panel = attach_event_hours(panel, alerts)
    panel, terms = add_event_dummies(panel)
    covered = panel["event_hour"].notna().sum()
    log.info("County-hours inside an event window: %s (%.3f%% of panel)",
             f"{covered:,}", 100 * covered / len(panel))

    results: list[dict] = []
    for fe_spec in args.fe:
        for outcome in args.outcomes:
            results.extend(run_event_study(panel, outcome, terms, fe_spec=fe_spec))

    out = pd.DataFrame(results)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_TABS / "hourly_alert_event_study.csv", index=False)
    log.info("Saved -> %s", OUTPUT_TABS / "hourly_alert_event_study.csv")

    est = out[out["record_type"] == "estimate"]
    if len(est):
        log.info("\n%s", est.sort_values(["fe_spec", "outcome", "event_hour"])[
            ["fe_spec", "outcome", "event_hour", "pct_change", "ci_low_pct", "ci_high_pct", "pvalue"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
