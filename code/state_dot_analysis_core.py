"""Core helpers for the corrected state-DOT AMBER-alert analysis.

These functions are deliberately side-effect free so the treatment construction
and estimator samples can be unit tested independently of the large data build.
"""
from __future__ import annotations

import inspect
import numpy as np
import pandas as pd


def _sum_preserve_missing(series: pd.Series) -> float:
    """Sum observed values while returning NaN when the outcome is unavailable."""
    return series.sum(min_count=1)


def normalize_state_outcomes(
    df: pd.DataFrame,
    *,
    crashes_col: str | None,
    fatals_col: str | None,
    serious_col: str | None,
    fatals_comparable: bool = True,
) -> pd.DataFrame:
    """Map one state's native columns to comparable county-day outcomes.

    Missing or non-comparable outcomes are represented as NaN, never structural
    zeros. This prevents states without a measure from entering pooled models as
    if they had zero events on every day.
    """
    out = df.copy()
    rename = {}
    if crashes_col and crashes_col in out.columns:
        rename[crashes_col] = "crashes"
    if fatals_col and fatals_col in out.columns and fatals_comparable:
        rename[fatals_col] = "fatals"
    if serious_col and serious_col in out.columns:
        rename[serious_col] = "serious_inj"
    out = out.rename(columns=rename)

    for col in ("crashes", "fatals", "serious_inj"):
        if col not in out.columns:
            out[col] = np.nan

    if not fatals_comparable:
        out["fatals"] = np.nan

    out = out[["fips", "date", "crashes", "fatals", "serious_inj"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["fips"] = out["fips"].astype(str).str.zfill(5)

    return (
        out.groupby(["fips", "date"], as_index=False)
        .agg(
            crashes=("crashes", _sum_preserve_missing),
            fatals=("fatals", _sum_preserve_missing),
            serious_inj=("serious_inj", _sum_preserve_missing),
        )
    )


def prepare_ppml_sample(
    df: pd.DataFrame,
    count_col: str,
    *,
    treatment_cols: tuple[str, ...] = ("night_alert",),
) -> pd.DataFrame:
    """Return a raw-count PPML sample while retaining valid zero outcomes."""
    required = [count_col, "population", "fips", "date", "year", *treatment_cols]
    sub = df.dropna(subset=required).copy()
    sub[count_col] = pd.to_numeric(sub[count_col], errors="coerce")
    sub = sub.dropna(subset=[count_col])
    sub = sub[sub[count_col] >= 0].copy()
    sub = sub[sub["population"] > 0].copy()

    sub["_fips_str"] = sub["fips"].astype(str).str.zfill(5)
    sub["_date_str"] = pd.to_datetime(sub["date"]).astype(str)
    sub["_year_str"] = sub["year"].astype(str)
    sub["_log_population"] = np.log(sub["population"].astype(float))
    return sub


def build_ppml_call_spec(fepois_func, *, count_col: str,
                         treatment_cols: tuple[str, ...]) -> dict:
    """Build a PPML formula/call compatible with the installed pyfixest API.

    If fepois exposes an ``offset`` argument we use log(population) as a true
    exposure offset. Otherwise log(population) enters explicitly with an
    estimated coefficient; this is not mathematically identical to an offset,
    but is transparent and avoids the previous incorrect rate+weight shortcut.
    """
    rhs = " + ".join(treatment_cols)
    try:
        has_offset = "offset" in inspect.signature(fepois_func).parameters
    except (TypeError, ValueError):
        has_offset = False

    if has_offset:
        formula = f"{count_col} ~ {rhs} | _fips_str + _date_str"
        return {
            "formula": formula,
            "offset": "_log_population",
            "exposure_mode": "log_population_offset",
        }

    formula = f"{count_col} ~ {rhs} + _log_population | _fips_str + _date_str"
    return {
        "formula": formula,
        "offset": None,
        "exposure_mode": "log_population_control_no_offset_support",
    }


def build_commuter_spillover(
    night_alerts: pd.DataFrame,
    flows: pd.DataFrame,
) -> pd.DataFrame:
    """Predict destination-county exposure from commuters in alerted counties.

    Main spillover intensity is the share of all commuters working in county i
    whose home county j received the alert. This is bounded in [0, 1] and is
    invariant to the arbitrary units of the raw commuter count. Own-county flows
    are excluded because ``night_alert`` captures direct phone exposure.

    ``spillover_commuters`` is retained as a descriptive quantity only. The
    legacy log-count column is kept for backwards compatibility with the earlier
    exploratory runner; the share-based runner does not use it as a regressor.
    """
    empty = pd.DataFrame(
        columns=["fips", "effective_crash_date", "spillover_commuters",
                 "spillover_share", "log_spillover_commuters"]
    )
    if night_alerts.empty or flows.empty:
        return empty

    alerts = night_alerts[["fips", "effective_crash_date"]].drop_duplicates().copy()
    alerts["fips_home"] = alerts["fips"].astype(str).str.zfill(5)
    alerts["effective_crash_date"] = pd.to_datetime(
        alerts["effective_crash_date"]
    ).dt.normalize()

    needed = ["fips_home", "fips_work", "workers"]
    missing = [c for c in needed if c not in flows.columns]
    if missing:
        raise ValueError(f"commuting-flow data missing required columns: {missing}")

    keep = needed + (["weight"] if "weight" in flows.columns else [])
    fl = flows[keep].copy()
    fl["fips_home"] = fl["fips_home"].astype(str).str.zfill(5)
    fl["fips_work"] = fl["fips_work"].astype(str).str.zfill(5)
    fl["workers"] = pd.to_numeric(fl["workers"], errors="coerce").fillna(0.0)
    fl = fl[fl["workers"] > 0].copy()

    # The existing commuting-weight builder defines weight as the fraction of
    # all workers in the destination/work county coming from each home county.
    # Recompute it if an older flow file lacks the column.
    if "weight" not in fl.columns:
        totals = fl.groupby("fips_work")["workers"].transform("sum")
        fl["weight"] = fl["workers"] / totals
    else:
        fl["weight"] = pd.to_numeric(fl["weight"], errors="coerce").fillna(0.0)

    fl = fl[fl["fips_home"] != fl["fips_work"]].copy()
    exposed = alerts.merge(fl, on="fips_home", how="inner")
    if exposed.empty:
        return empty

    out = (
        exposed.groupby(["fips_work", "effective_crash_date"], as_index=False)
        .agg(
            spillover_commuters=("workers", "sum"),
            spillover_share=("weight", "sum"),
        )
        .rename(columns={"fips_work": "fips"})
    )
    out["spillover_share"] = out["spillover_share"].clip(lower=0.0, upper=1.0)
    out["log_spillover_commuters"] = np.log1p(out["spillover_commuters"])
    return out


def add_spillover_classes(panel: pd.DataFrame) -> pd.DataFrame:
    """Label direct, spillover-only, and uncontaminated control observations."""
    out = panel.copy()
    if "spillover_share" not in out.columns:
        out["spillover_share"] = 0.0
    out["spillover_share"] = out["spillover_share"].fillna(0.0).clip(0.0, 1.0)

    if "spillover_commuters" not in out.columns:
        out["spillover_commuters"] = 0.0
    out["spillover_commuters"] = out["spillover_commuters"].fillna(0.0)

    # Backwards compatibility only; new specifications use spillover_share.
    if "log_spillover_commuters" not in out.columns:
        out["log_spillover_commuters"] = np.log1p(out["spillover_commuters"])
    else:
        out["log_spillover_commuters"] = out["log_spillover_commuters"].fillna(0.0)

    direct = out["night_alert"].fillna(0).astype(int).eq(1)
    spill = (~direct) & out["spillover_share"].gt(0)
    out["exposure_class"] = np.select(
        [direct, spill], ["direct", "spillover"], default="clean_control"
    )
    out["clean_control"] = (out["exposure_class"] == "clean_control").astype(int)
    return out
