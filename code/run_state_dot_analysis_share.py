"""Share-based corrected state-DOT AMBER-alert analysis.

This is the preferred runner after the log(1+x) spillover specification was
rejected. It reuses the corrected data construction from
``run_state_dot_analysis_fixed.py`` but models cross-county interference as the
share of a destination county's commuters whose home county received the alert.

Interpretation
--------------
``spillover_share_10pp`` = spillover commuter share / 0.10, so its coefficient
is the effect of a 10 percentage-point increase in the destination workforce
coming from alerted home counties.

Models
------
1. WLS TWFE on rates per 100k, population weighted.
2. PPML on raw counts with a true log(population) exposure offset.
3. Joint direct + spillover-share specification.
4. Direct-vs-clean-control sensitivity excluding all predicted spillover days.

All models absorb county and calendar-date fixed effects and use multiway
county + calendar-date CRV1 clustering.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import pyfixest as pf

import run_state_dot_analysis_fixed as base
from config import OUTPUT_TABS
from state_dot_analysis_core import prepare_ppml_sample
from state_dot_analysis_core import extract_finite_coefficients, fit_status_row, summarize_fit_statuses

log = base.log


def _diagnostic(
    *, label: str, model: str, outcome: str, sample: str, status: str,
    input_n: int, fitted_n: int, zero_share: float | None,
    terms_requested: tuple[str, ...], terms_produced: tuple[str, ...] = (),
    error_reason: str | None = None,
) -> dict[str, object]:
    return {
        "state": label, "model": model, "outcome": outcome, "sample": sample,
        **fit_status_row(
            status=status, input_n=input_n, fitted_n=fitted_n,
            zero_share=zero_share, terms_requested=terms_requested,
            terms_produced=terms_produced, error_reason=error_reason,
        ),
    }


def build_panel(*, direct_only: bool = False) -> pd.DataFrame:
    panel = base.build_panel(direct_only=direct_only).copy()
    if "spillover_share" not in panel.columns:
        panel["spillover_share"] = 0.0
    panel["spillover_share"] = panel["spillover_share"].fillna(0.0).clip(0.0, 1.0)
    panel["spillover_share_10pp"] = panel["spillover_share"] / 0.10
    return panel


def _treatments(panel: pd.DataFrame) -> tuple[str, ...]:
    if panel["spillover_share_10pp"].fillna(0).gt(0).any():
        return ("night_alert", "spillover_share_10pp")
    return ("night_alert",)


def _vcov(sub: pd.DataFrame) -> dict:
    return {"CRV1": "_fips_str + _date_str"}


def run_wls(
    panel: pd.DataFrame,
    rate_col: str,
    label: str,
    *,
    clean_controls: bool = False,
    direct_only: bool = False,
) -> list[dict]:
    sub = panel.dropna(subset=[rate_col, "population", "night_alert"]).copy()
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    treatments = ("night_alert",) if (clean_controls or direct_only) else _treatments(sub)
    sample = "direct_vs_clean" if clean_controls else ("direct_only" if direct_only else "spillover_joint")
    zero_share = float((sub[rate_col] == 0).mean()) if len(sub) else None
    # A single-county sample (a sub-state source like Montgomery County, MD)
    # cannot support a county+date two-way FE fit: with one county, every
    # date is its own singleton cell after demeaning, which is degenerate
    # rather than merely underpowered -- pyfixest's Rust backend panics
    # (uncatchable process abort, not a Python exception) instead of raising
    # cleanly on this input, so it must be skipped before the fit call.
    if len(sub) < 100 or sub["night_alert"].nunique() < 2 or sub["fips"].nunique() < 2:
        return [_diagnostic(
            label=label, model="WLS_TWFE", outcome=rate_col, sample=sample,
            status="skipped", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason="insufficient_estimable_sample",
        )]

    sub["_fips_str"] = sub["fips"].astype(str)
    sub["_date_str"] = pd.to_datetime(sub["date"]).astype(str)
    sub["_pop"] = sub["population"].astype(float)
    formula = f"{rate_col} ~ {' + '.join(treatments)} | _fips_str + _date_str"
    try:
        fit = pf.feols(formula, data=sub, weights="_pop", vcov=_vcov(sub))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning(
            "Skipping %s %s WLS_TWFE %s: %s",
            label, rate_col, sample, exc,
        )
        return [_diagnostic(
            label=label, model="WLS_TWFE", outcome=rate_col, sample=sample,
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason=str(exc),
        )]

    coefficients, produced, errors = extract_finite_coefficients(fit, treatments)
    rows = []
    for coefficient in coefficients:
        b, se, p = coefficient["beta"], coefficient["se"], coefficient["pvalue"]
        rows.append({
            "record_type": "estimate", "status": "ok",
            "sample": sample,
            "state": label,
            "model": "WLS_TWFE",
            "outcome": rate_col,
            "term": coefficient["term"],
            "beta": b,
            "se": se,
            "pvalue": p,
            "n_obs": int(fit._N),
            "exposure_mode": "rate_per_100k_population_weighted",
            "cluster": "county+date",
        })
    status = "ok" if not errors else ("partial" if produced else "failed")
    error_reason = next(iter(set(errors.values()))) if len(set(errors.values())) == 1 else "; ".join(
        f"{term}:{reason}" for term, reason in errors.items()
    )
    rows.append(_diagnostic(
        label=label, model="WLS_TWFE", outcome=rate_col, sample=sample,
        status=status, input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=treatments, terms_produced=produced, error_reason=error_reason,
    ))
    return rows


def run_ppml(
    panel: pd.DataFrame,
    count_col: str,
    label: str,
    *,
    clean_controls: bool = False,
    direct_only: bool = False,
) -> list[dict]:
    treatments = ("night_alert",) if (clean_controls or direct_only) else _treatments(panel)
    sub = prepare_ppml_sample(panel, count_col, treatment_cols=treatments)
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    sample = "direct_vs_clean" if clean_controls else ("direct_only" if direct_only else "spillover_joint")
    zero_share = float((sub[count_col] == 0).mean()) if len(sub) else None
    # A single-county sample (a sub-state source like Montgomery County, MD)
    # cannot support a county+date two-way FE fit: with one county, every
    # date is its own singleton cell after demeaning, which is degenerate
    # rather than merely underpowered -- pyfixest's Rust backend panics
    # (uncatchable process abort, not a Python exception) instead of raising
    # cleanly on this input, so it must be skipped before the fit call.
    if len(sub) < 100 or sub["night_alert"].nunique() < 2 or sub["fips"].nunique() < 2:
        return [_diagnostic(
            label=label, model="PPML_raw_count", outcome=count_col, sample=sample,
            status="skipped", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason="insufficient_estimable_sample",
        )]

    formula = f"{count_col} ~ {' + '.join(treatments)} | _fips_str + _date_str"
    log.info(
        "[%s %s] PPML %s obs; %.1f%% zero outcomes; log(population) offset",
        label, count_col, f"{len(sub):,}", 100 * zero_share,
    )
    try:
        fit = pf.fepois(
            formula,
            data=sub,
            offset="_log_population",
            vcov=_vcov(sub),
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning(
            "Skipping %s %s PPML_raw_count %s: %s",
            label, count_col, sample, exc,
        )
        return [_diagnostic(
            label=label, model="PPML_raw_count", outcome=count_col, sample=sample,
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason=str(exc),
        )]

    coefficients, produced, errors = extract_finite_coefficients(fit, treatments)
    rows = []
    for coefficient in coefficients:
        b, se, p = coefficient["beta"], coefficient["se"], coefficient["pvalue"]
        rows.append({
            "record_type": "estimate", "status": "ok",
            "sample": sample,
            "state": label,
            "model": "PPML_raw_count",
            "outcome": count_col,
            "term": coefficient["term"],
            "beta": b,
            "se": se,
            "pvalue": p,
            "irr": float(np.exp(b)),
            "pct_change": float(100 * (np.exp(b) - 1)),
            "n_obs": int(fit._N),
            "zero_share_input": zero_share,
            "exposure_mode": "log_population_offset",
            "cluster": "county+date",
        })
    status = "ok" if not errors else ("partial" if produced else "failed")
    error_reason = next(iter(set(errors.values()))) if len(set(errors.values())) == 1 else "; ".join(
        f"{term}:{reason}" for term, reason in errors.items()
    )
    rows.append(_diagnostic(
        label=label, model="PPML_raw_count", outcome=count_col, sample=sample,
        status=status, input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=treatments, terms_produced=produced, error_reason=error_reason,
    ))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-only", action="store_true",
        help="run the explicitly requested direct-alert-only sensitivity; do not relabel it as spillover_joint",
    )
    args = parser.parse_args(argv)
    panel = build_panel(direct_only=args.direct_only)
    spill = panel.loc[panel["spillover_share"] > 0, "spillover_share"]
    log.info(
        "Panel: %s rows; %s direct-alert days; %s spillover-only days; %s clean controls",
        f"{len(panel):,}",
        f"{int((panel['exposure_class'] == 'direct').sum()):,}",
        f"{int((panel['exposure_class'] == 'spillover').sum()):,}",
        f"{int((panel['exposure_class'] == 'clean_control').sum()):,}",
    )
    if len(spill):
        log.info(
            "Positive spillover share: median %.3f, p90 %.3f, max %.3f",
            float(spill.median()), float(spill.quantile(0.90)), float(spill.max()),
        )

    desc = panel.groupby("state", as_index=False).agg(
        county_days=("fips", "size"),
        counties=("fips", "nunique"),
        direct_alert_days=("night_alert", "sum"),
        spillover_only_days=("exposure_class", lambda s: int((s == "spillover").sum())),
        mean_spillover_share=("spillover_share", "mean"),
        max_spillover_share=("spillover_share", "max"),
        crash_rows_available=("crashes", lambda s: int(s.notna().sum())),
        fatal_rows_available=("fatals", lambda s: int(s.notna().sum())),
        serious_rows_available=("serious_inj", lambda s: int(s.notna().sum())),
    )
    desc.to_csv(OUTPUT_TABS / "state_dot_descriptives_share.csv", index=False)

    outcomes = [
        ("crashes_per_100k", "crashes"),
        ("fatals_per_100k", "fatals"),
        ("serious_per_100k", "serious_inj"),
    ]
    results = []
    for state_filter in [None] + sorted(panel["state"].unique().tolist()):
        label = "ALL" if state_filter is None else state_filter
        sub = panel if state_filter is None else panel[panel["state"] == state_filter]
        for rate_col, count_col in outcomes:
            results.extend(run_wls(sub, rate_col, label, clean_controls=False, direct_only=args.direct_only))
            results.extend(run_ppml(sub, count_col, label, clean_controls=False, direct_only=args.direct_only))
            results.extend(run_wls(sub, rate_col, label, clean_controls=True))
            results.extend(run_ppml(sub, count_col, label, clean_controls=True))

    all_rows = pd.DataFrame(results)
    out = all_rows.loc[all_rows.get("record_type", pd.Series(dtype=str)).eq("estimate")].copy()
    statuses = all_rows.loc[all_rows.get("record_type", pd.Series(dtype=str)).eq("fit_status")].copy()
    statuses = pd.concat([statuses, pd.DataFrame([{
        "record_type": "model_count_summary", **summarize_fit_statuses(statuses.to_dict("records")),
    }])], ignore_index=True)
    out.to_csv(OUTPUT_TABS / "state_dot_analysis_share.csv", index=False)
    statuses.to_csv(OUTPUT_TABS / "state_dot_analysis_share_status.csv", index=False)
    log.info("Saved %d estimates and %d fit diagnostics -> %s", len(out), len(statuses) - 1, OUTPUT_TABS)


if __name__ == "__main__":
    main()
