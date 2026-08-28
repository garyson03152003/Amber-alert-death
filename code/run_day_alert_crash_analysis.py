"""Daytime AMBER-alert effect on crashes: the driver-distraction hypothesis.

Motivation
----------
The night-alert design asks whether an alert issued 22:00-05:59 changes the
*following* day's driving outcomes. That timing is wrong for a distraction
mechanism: a distraction effect should show up while drivers are actually on
the road reading the alert. Daytime alerts (06:00-21:59 local) are assigned
to the *same* calendar date for exactly that reason, and are 4x more common
than night alerts, so this design also has considerably more power.

Specification
-------------
    crashes_cd ~ day_alert_cd + night_alert_cd | county + calendar-date

Both windows enter jointly. A county-day can receive both a daytime and a
late-night alert; attributing such a day to one window alone would confound
the two exposures. The night coefficient is a nuisance control here, not the
estimand.

PPML on raw counts with a true log(population) exposure offset is the
preferred model (retains valid zero-crash county-days); WLS on rates per
100k is reported alongside as a linear cross-check. Inference is two-way
CRV1 clustered on county + calendar-date, matching the night-alert analysis.

Interpretation limit
--------------------
Outcomes are county-DAY counts, so a same-day design necessarily includes
hours *before* the alert was issued. That attenuates any true post-alert
distraction effect toward zero: the estimate is a lower bound on an
hours-after-issuance effect, not a direct measure of one. A sharper test
needs crash timestamps, which the county-day panel does not carry.

Outputs
-------
output/tables/day_alert_crash_analysis.csv
output/tables/day_alert_crash_analysis_status.csv
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
from config import OUTPUT_TABS
from state_dot_analysis_core import (
    prepare_ppml_sample,
    extract_finite_coefficients,
    fit_status_row,
    summarize_fit_statuses,
)

log = base.log

TREATMENTS = ("day_alert", "night_alert")
OUTCOMES = [
    ("crashes_per_100k", "crashes"),
    ("fatals_per_100k", "fatals"),
    ("serious_per_100k", "serious_inj"),
]


def _diagnostic(*, label, model, outcome, status, input_n, fitted_n,
                zero_share, terms_produced=(), error_reason=None) -> dict:
    return {
        "state": label, "model": model, "outcome": outcome, "sample": "day_vs_night_joint",
        **fit_status_row(
            status=status, input_n=input_n, fitted_n=fitted_n, zero_share=zero_share,
            terms_requested=TREATMENTS, terms_produced=terms_produced,
            error_reason=error_reason,
        ),
    }


def _estimable(sub: pd.DataFrame) -> bool:
    """Guard the same degenerate cases the night-alert runners screen for.

    A single-county sample cannot support county+date two-way FE, and
    pyfixest's Rust backend aborts the process rather than raising on that
    input, so it must be caught before the fit call.
    """
    return (
        len(sub) >= 100
        and sub["fips"].nunique() >= 2
        and sub["day_alert"].nunique() >= 2
    )


def _present_treatments(sub: pd.DataFrame) -> tuple[str, ...]:
    """Drop a treatment with no variation in this subsample (e.g. a state-year
    slice with no night alerts) so it cannot silently absorb the fit."""
    return tuple(t for t in TREATMENTS if sub[t].nunique() >= 2)


def run_ppml(panel: pd.DataFrame, count_col: str, label: str) -> list[dict]:
    sub = prepare_ppml_sample(panel, count_col, treatment_cols=TREATMENTS)
    zero_share = float((sub[count_col] == 0).mean()) if len(sub) else None
    terms = _present_treatments(sub) if len(sub) else ()
    if not _estimable(sub) or not terms:
        return [_diagnostic(
            label=label, model="PPML_raw_count", outcome=count_col, status="skipped",
            input_n=len(sub), fitted_n=0, zero_share=zero_share,
            error_reason="insufficient_estimable_sample",
        )]

    formula = f"{count_col} ~ {' + '.join(terms)} | _fips_str + _date_str"
    try:
        fit = pf.fepois(formula, data=sub, offset="_log_population",
                        vcov={"CRV1": "_fips_str + _date_str"})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("Skipping %s %s PPML: %s", label, count_col, exc)
        return [_diagnostic(
            label=label, model="PPML_raw_count", outcome=count_col, status="failed",
            input_n=len(sub), fitted_n=0, zero_share=zero_share, error_reason=str(exc),
        )]

    coefficients, produced, errors = extract_finite_coefficients(fit, terms)
    rows = [{
        "record_type": "estimate", "status": "ok", "sample": "day_vs_night_joint",
        "state": label, "model": "PPML_raw_count", "outcome": count_col,
        "term": c["term"], "beta": c["beta"], "se": c["se"], "pvalue": c["pvalue"],
        "irr": float(np.exp(c["beta"])),
        "pct_change": float(100 * (np.exp(c["beta"]) - 1)),
        "ci_low_pct": float(100 * (np.exp(c["beta"] - 1.96 * c["se"]) - 1)),
        "ci_high_pct": float(100 * (np.exp(c["beta"] + 1.96 * c["se"]) - 1)),
        "n_obs": int(fit._N), "zero_share_input": zero_share,
        "exposure_mode": "log_population_offset", "cluster": "county+date",
    } for c in coefficients]

    status = "ok" if not errors else ("partial" if produced else "failed")
    rows.append(_diagnostic(
        label=label, model="PPML_raw_count", outcome=count_col, status=status,
        input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_produced=produced,
        error_reason="; ".join(f"{t}:{r}" for t, r in errors.items()) or None,
    ))
    return rows


def run_wls(panel: pd.DataFrame, rate_col: str, label: str) -> list[dict]:
    sub = panel.dropna(subset=[rate_col, "population", *TREATMENTS]).copy()
    zero_share = float((sub[rate_col] == 0).mean()) if len(sub) else None
    terms = _present_treatments(sub) if len(sub) else ()
    if not _estimable(sub) or not terms:
        return [_diagnostic(
            label=label, model="WLS_TWFE", outcome=rate_col, status="skipped",
            input_n=len(sub), fitted_n=0, zero_share=zero_share,
            error_reason="insufficient_estimable_sample",
        )]

    sub["_fips_str"] = sub["fips"].astype(str)
    sub["_date_str"] = pd.to_datetime(sub["date"]).astype(str)
    sub["_pop"] = sub["population"].astype(float)
    formula = f"{rate_col} ~ {' + '.join(terms)} | _fips_str + _date_str"
    try:
        fit = pf.feols(formula, data=sub, weights="_pop",
                       vcov={"CRV1": "_fips_str + _date_str"})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("Skipping %s %s WLS: %s", label, rate_col, exc)
        return [_diagnostic(
            label=label, model="WLS_TWFE", outcome=rate_col, status="failed",
            input_n=len(sub), fitted_n=0, zero_share=zero_share, error_reason=str(exc),
        )]

    coefficients, produced, errors = extract_finite_coefficients(fit, terms)
    rows = [{
        "record_type": "estimate", "status": "ok", "sample": "day_vs_night_joint",
        "state": label, "model": "WLS_TWFE", "outcome": rate_col,
        "term": c["term"], "beta": c["beta"], "se": c["se"], "pvalue": c["pvalue"],
        "ci_low": float(c["beta"] - 1.96 * c["se"]),
        "ci_high": float(c["beta"] + 1.96 * c["se"]),
        "n_obs": int(fit._N),
        "exposure_mode": "rate_per_100k_population_weighted", "cluster": "county+date",
    } for c in coefficients]

    status = "ok" if not errors else ("partial" if produced else "failed")
    rows.append(_diagnostic(
        label=label, model="WLS_TWFE", outcome=rate_col, status=status,
        input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_produced=produced,
        error_reason="; ".join(f"{t}:{r}" for t, r in errors.items()) or None,
    ))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-state", action="store_true",
                        help="also estimate each validated source separately")
    args = parser.parse_args(argv)

    panel = base.build_panel(direct_only=True)
    log.info(
        "Panel: %s county-days; %s day-alert days; %s night-alert days; %s both",
        f"{len(panel):,}",
        f"{int(panel['day_alert'].sum()):,}",
        f"{int(panel['night_alert'].sum()):,}",
        f"{int(((panel['day_alert'] == 1) & (panel['night_alert'] == 1)).sum()):,}",
    )

    labels = [None] + (sorted(panel["state"].unique().tolist()) if args.by_state else [])
    results = []
    for state_filter in labels:
        label = "ALL" if state_filter is None else state_filter
        sub = panel if state_filter is None else panel[panel["state"] == state_filter]
        for rate_col, count_col in OUTCOMES:
            results.extend(run_ppml(sub, count_col, label))
            results.extend(run_wls(sub, rate_col, label))

    all_rows = pd.DataFrame(results)
    kind = all_rows.get("record_type", pd.Series(dtype=str))
    estimates = all_rows.loc[kind.eq("estimate")].copy()
    statuses = all_rows.loc[kind.eq("fit_status")].copy()
    statuses = pd.concat([statuses, pd.DataFrame([{
        "record_type": "model_count_summary",
        **summarize_fit_statuses(statuses.to_dict("records")),
    }])], ignore_index=True)

    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUTPUT_TABS / "day_alert_crash_analysis.csv", index=False)
    statuses.to_csv(OUTPUT_TABS / "day_alert_crash_analysis_status.csv", index=False)
    log.info("Saved %d estimates, %d diagnostics -> %s",
             len(estimates), len(statuses) - 1, OUTPUT_TABS)

    headline = estimates[
        (estimates["state"] == "ALL") & (estimates["model"] == "PPML_raw_count")
    ]
    if len(headline):
        log.info("\n%s", headline[
            ["outcome", "term", "pct_change", "ci_low_pct", "ci_high_pct", "pvalue", "n_obs"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
