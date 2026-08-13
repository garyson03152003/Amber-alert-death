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

import numpy as np
import pandas as pd
import pyfixest as pf

import run_state_dot_analysis_fixed as base
from config import OUTPUT_TABS
from state_dot_analysis_core import prepare_ppml_sample

log = base.log


def build_panel() -> pd.DataFrame:
    panel = base.build_panel().copy()
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
) -> list[dict]:
    sub = panel.dropna(subset=[rate_col, "population", "night_alert"]).copy()
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    if len(sub) < 100 or sub["night_alert"].nunique() < 2:
        return []

    sub["_fips_str"] = sub["fips"].astype(str)
    sub["_date_str"] = pd.to_datetime(sub["date"]).astype(str)
    sub["_pop"] = sub["population"].astype(float)

    treatments = ("night_alert",) if clean_controls else _treatments(sub)
    formula = f"{rate_col} ~ {' + '.join(treatments)} | _fips_str + _date_str"
    fit = pf.feols(formula, data=sub, weights="_pop", vcov=_vcov(sub))

    rows = []
    for term in treatments:
        vals = base._coef_row(fit, term)
        if vals is None:
            continue
        b, se, p = vals
        rows.append({
            "sample": "direct_vs_clean" if clean_controls else "spillover_joint",
            "state": label,
            "model": "WLS_TWFE",
            "outcome": rate_col,
            "term": term,
            "beta": b,
            "se": se,
            "pvalue": p,
            "n_obs": int(fit._N),
            "exposure_mode": "rate_per_100k_population_weighted",
            "cluster": "county+date",
        })
    return rows


def run_ppml(
    panel: pd.DataFrame,
    count_col: str,
    label: str,
    *,
    clean_controls: bool = False,
) -> list[dict]:
    treatments = ("night_alert",) if clean_controls else _treatments(panel)
    sub = prepare_ppml_sample(panel, count_col, treatment_cols=treatments)
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    if len(sub) < 100 or sub["night_alert"].nunique() < 2:
        return []

    formula = f"{count_col} ~ {' + '.join(treatments)} | _fips_str + _date_str"
    zero_share = float((sub[count_col] == 0).mean())
    log.info(
        "[%s %s] PPML %,d obs; %.1f%% zero outcomes; log(population) offset",
        label, count_col, len(sub), 100 * zero_share,
    )
    fit = pf.fepois(
        formula,
        data=sub,
        offset="_log_population",
        vcov=_vcov(sub),
    )

    rows = []
    for term in treatments:
        vals = base._coef_row(fit, term)
        if vals is None:
            continue
        b, se, p = vals
        rows.append({
            "sample": "direct_vs_clean" if clean_controls else "spillover_joint",
            "state": label,
            "model": "PPML_raw_count",
            "outcome": count_col,
            "term": term,
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
    return rows


def main() -> None:
    panel = build_panel()
    spill = panel.loc[panel["spillover_share"] > 0, "spillover_share"]
    log.info(
        "Panel: %,d rows; %,d direct-alert days; %,d spillover-only days; %,d clean controls",
        len(panel),
        int((panel["exposure_class"] == "direct").sum()),
        int((panel["exposure_class"] == "spillover").sum()),
        int((panel["exposure_class"] == "clean_control").sum()),
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
        if sub["night_alert"].sum() < 10:
            continue
        for rate_col, count_col in outcomes:
            if sub[count_col].notna().sum() < 100:
                continue
            results.extend(run_wls(sub, rate_col, label, clean_controls=False))
            results.extend(run_ppml(sub, count_col, label, clean_controls=False))
            results.extend(run_wls(sub, rate_col, label, clean_controls=True))
            results.extend(run_ppml(sub, count_col, label, clean_controls=True))

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_TABS / "state_dot_analysis_share.csv", index=False)
    log.info("Saved %d result rows -> %s", len(out), OUTPUT_TABS / "state_dot_analysis_share.csv")


if __name__ == "__main__":
    main()
