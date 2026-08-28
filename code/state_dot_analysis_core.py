"""Core helpers for the corrected state-DOT AMBER-alert analysis.

These functions are deliberately side-effect free so the treatment construction
and estimator samples can be unit tested independently of the large data build.
"""
from __future__ import annotations

import inspect
import numpy as np
import pandas as pd


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if value is not None and not pd.isna(value) else False


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


def build_car_weighted_spillover(
    night_alerts: pd.DataFrame,
    flows: pd.DataFrame,
    car_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Spillover exposure weighted by who actually drives.

    ``build_commuter_spillover`` treats every commuter alike, so a rail
    commuter from an alerted county counts the same as a driver. The
    mechanism under test is a driver carrying alert exposure onto the road,
    so the flow is reweighted by each origin county's ACS B08301 car share
    (drove alone + carpooled, over all workers 16+):

        spillover_driver_share_it =
            sum_j workers(j->i) * car_share_j * 1{j alerted on t}
            / sum_j workers(j->i) * car_share_j

    Own-county flows are excluded, as in the unweighted version, because
    direct exposure is already captured by ``night_alert``.

    The denominator is the county's whole inbound driving workforce, so the
    result stays bounded in [0, 1] and remains comparable in scale to
    ``spillover_share`` -- the difference is composition, not units.

    Note this is a *commuting* mode share standing in for driving generally,
    and a single ACS vintage applied across all years. Counties with no car
    share available are dropped from the weighting rather than treated as
    zero, since a missing share is not a claim that nobody drives.
    """
    empty = pd.DataFrame(
        columns=["fips", "effective_crash_date",
                 "spillover_driver_share", "spillover_drivers"]
    )
    if night_alerts.empty or flows.empty or car_shares.empty:
        return empty

    needed = ["fips_home", "fips_work", "workers"]
    missing = [c for c in needed if c not in flows.columns]
    if missing:
        raise ValueError(f"commuting-flow data missing required columns: {missing}")
    if not {"fips", "car_share"}.issubset(car_shares.columns):
        raise ValueError("car_shares must contain fips and car_share")

    fl = flows[needed].copy()
    fl["fips_home"] = fl["fips_home"].astype(str).str.zfill(5)
    fl["fips_work"] = fl["fips_work"].astype(str).str.zfill(5)
    fl["workers"] = pd.to_numeric(fl["workers"], errors="coerce").fillna(0.0)
    fl = fl[(fl["workers"] > 0) & (fl["fips_home"] != fl["fips_work"])].copy()

    cs = car_shares[["fips", "car_share"]].copy()
    cs["fips"] = cs["fips"].astype(str).str.zfill(5)
    cs["car_share"] = pd.to_numeric(cs["car_share"], errors="coerce")
    cs = cs.dropna(subset=["car_share"])

    fl = fl.merge(cs.rename(columns={"fips": "fips_home"}), on="fips_home", how="inner")
    fl["driving_workers"] = fl["workers"] * fl["car_share"]
    denom = fl.groupby("fips_work")["driving_workers"].sum().rename("total_driving_inflow")
    if denom.empty:
        return empty

    alerts = night_alerts[["fips", "effective_crash_date"]].drop_duplicates().copy()
    alerts["fips_home"] = alerts["fips"].astype(str).str.zfill(5)
    alerts["effective_crash_date"] = pd.to_datetime(
        alerts["effective_crash_date"]
    ).dt.normalize()

    exposed = alerts.merge(fl, on="fips_home", how="inner")
    if exposed.empty:
        return empty

    out = (
        exposed.groupby(["fips_work", "effective_crash_date"], as_index=False)
        .agg(spillover_drivers=("driving_workers", "sum"))
        .rename(columns={"fips_work": "fips"})
    )
    out = out.merge(denom, left_on="fips", right_index=True, how="left")
    out["spillover_driver_share"] = (
        out["spillover_drivers"] / out["total_driving_inflow"]
    ).clip(lower=0.0, upper=1.0)
    return out[["fips", "effective_crash_date",
                "spillover_driver_share", "spillover_drivers"]]


def validate_analysis_inputs(
    panel: pd.DataFrame,
    manifest: pd.DataFrame,
    review: pd.DataFrame | None,
    *,
    flows: pd.DataFrame | None,
    direct_only: bool = False,
    require_review: bool = True,
) -> None:
    """Fail closed before a validated panel reaches an estimator.

    A legacy sparse source file cannot establish that an absent county-day is a
    true zero.  The runner therefore requires the balanced-panel provenance,
    its coverage manifest, a reviewed state-year decision, and (unless the
    caller explicitly requests a direct-only model) commuter weights.
    """
    required_panel = {"fips", "date", "year", "coverage_valid", "structural_zero", "source"}
    missing_panel = required_panel - set(panel.columns)
    if missing_panel:
        raise ValueError(f"validated panel missing required columns: {sorted(missing_panel)}")
    if panel.empty:
        raise ValueError("validated panel is empty")
    if not panel["coverage_valid"].map(_as_bool).all():
        raise ValueError("validated panel contains invalid coverage units")

    required_manifest = {"year", "coverage_valid", "source"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"coverage manifest missing required columns: {sorted(missing_manifest)}")
    if manifest.empty:
        raise ValueError("coverage manifest is empty")
    if not manifest["coverage_valid"].map(_as_bool).all():
        raise ValueError("coverage manifest contains invalid reporting units")
    _validate_panel_manifest_binding(panel, manifest)

    if not direct_only:
        if flows is None:
            raise ValueError("commuting weights are required for spillover_joint analysis")
        required_flows = {"fips_home", "fips_work", "workers", "weight"}
        missing_flows = required_flows - set(flows.columns)
        if missing_flows or flows.empty:
            rendered = sorted(missing_flows) if missing_flows else ["no rows"]
            raise ValueError(f"commuting weights are missing or incomplete: {rendered}")

    if not require_review:
        return
    if review is None or review.empty:
        raise ValueError("reviewed accepted state-years are required")
    required_review = {"state", "year", "review_status"}
    missing_review = required_review - set(review.columns)
    if missing_review:
        raise ValueError(f"review table missing required columns: {sorted(missing_review)}")
    accepted = review.loc[
        review["review_status"].astype(str).str.lower().eq("accepted"), ["state", "year"]
    ].copy()
    accepted["state"] = accepted["state"].astype(str).str.upper()
    accepted["year"] = pd.to_numeric(accepted["year"], errors="coerce")
    if accepted["year"].isna().any():
        raise ValueError("reviewed accepted state-years contain invalid years")
    if "state" not in panel.columns:
        raise ValueError("validated state panel is missing state labels for review")
    observed = panel.loc[:, ["state", "year"]].drop_duplicates().copy()
    observed["state"] = observed["state"].astype(str).str.upper()
    observed["year"] = pd.to_numeric(observed["year"], errors="coerce")
    required_keys = set(map(tuple, observed.to_records(index=False)))
    accepted_keys = set(map(tuple, accepted.to_records(index=False)))
    missing = sorted(required_keys - accepted_keys)
    if missing:
        rendered = ", ".join(f"{state} {int(year)}" for state, year in missing)
        raise ValueError(f"validated panel includes state-years not reviewed accepted: {rendered}")


def _validate_panel_manifest_binding(panel: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Require every balanced panel reporting key to originate in its manifest.

    State/year matching alone is not provenance: a panel row must also carry a
    source that the corresponding manifest authorizes.  County-year sources
    (Wisconsin) additionally require a matching county reporting unit.
    """
    panel_keys = panel.loc[:, ["year", "source"]].copy()
    panel_keys["year"] = pd.to_numeric(panel_keys["year"], errors="coerce")
    panel_keys["source"] = panel_keys["source"].astype(str)
    manifest_keys = manifest.loc[:, ["year", "source"]].copy()
    manifest_keys["year"] = pd.to_numeric(manifest_keys["year"], errors="coerce")
    manifest_keys["source"] = manifest_keys["source"].astype(str)
    if "state" in panel.columns and "state" in manifest.columns:
        panel_keys["state"] = panel["state"].astype(str).str.upper()
        manifest_keys["state"] = manifest["state"].astype(str).str.upper()
    join_keys = list(panel_keys.columns)
    allowed = manifest_keys.drop_duplicates()
    unmatched = panel_keys.drop_duplicates().merge(allowed, on=join_keys, how="left", indicator=True)
    if unmatched["_merge"].ne("both").any():
        bad = unmatched.loc[unmatched["_merge"].ne("both"), join_keys].to_dict("records")
        raise ValueError(f"validated panel source/reporting keys are absent from coverage manifest: {bad}")

    if "county_fips" not in manifest.columns or not manifest["county_fips"].notna().any():
        return
    county_manifest = manifest.loc[manifest["county_fips"].notna()].copy()
    county_sources = set(county_manifest["source"].astype(str))
    county_panel = panel.loc[panel["source"].astype(str).isin(county_sources)].copy()
    if county_panel.empty:
        return
    county_panel["fips"] = county_panel["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    county_manifest["county_fips"] = county_manifest["county_fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    county_panel["year"] = pd.to_numeric(county_panel["year"], errors="coerce")
    county_manifest["year"] = pd.to_numeric(county_manifest["year"], errors="coerce")
    county_panel["source"] = county_panel["source"].astype(str)
    county_manifest["source"] = county_manifest["source"].astype(str)
    keys = ["source", "year"]
    if "state" in county_panel.columns and "state" in county_manifest.columns:
        county_panel["state"] = county_panel["state"].astype(str).str.upper()
        county_manifest["state"] = county_manifest["state"].astype(str).str.upper()
        keys.append("state")
    observed = county_panel.loc[:, [*keys, "fips"]].drop_duplicates()
    expected = county_manifest.loc[:, [*keys, "county_fips"]].drop_duplicates()
    unmatched_counties = observed.merge(expected, left_on=[*keys, "fips"], right_on=[*keys, "county_fips"], how="left", indicator=True)
    if unmatched_counties["_merge"].ne("both").any():
        bad = unmatched_counties.loc[unmatched_counties["_merge"].ne("both"), [*keys, "fips"]].to_dict("records")
        raise ValueError(f"validated county-year panel keys are absent from coverage manifest: {bad}")


def extract_finite_coefficients(
    fit,
    terms: tuple[str, ...],
) -> tuple[list[dict[str, float | str]], tuple[str, ...], dict[str, str]]:
    """Extract only interpretable estimates from a pyfixest fit.

    Nonfinite estimates, standard errors, and p-values are model failures, not
    values to render as an apparently valid result.
    """
    table = fit.tidy()
    rows: list[dict[str, float | str]] = []
    errors: dict[str, str] = {}
    for term in terms:
        if term not in table.index:
            errors[term] = "missing_coefficient"
            continue
        try:
            beta = float(table.loc[term, "Estimate"])
            se = float(table.loc[term, "Std. Error"])
            pvalue = float(table.loc[term, "Pr(>|t|)"])
        except (KeyError, TypeError, ValueError):
            errors[term] = "invalid_coefficient_schema"
            continue
        if not np.isfinite([beta, se, pvalue]).all():
            errors[term] = "nonfinite_coefficient"
            continue
        rows.append({"term": term, "beta": beta, "se": se, "pvalue": pvalue})
    return rows, tuple(row["term"] for row in rows), errors


def fit_status_row(
    *,
    status: str,
    input_n: int,
    fitted_n: int,
    zero_share: float | None,
    terms_requested: tuple[str, ...],
    terms_produced: tuple[str, ...] = (),
    error_reason: str | None = None,
) -> dict[str, object]:
    """Return a machine-readable diagnostic for one expected model fit."""
    return {
        "record_type": "fit_status",
        "status": status,
        "input_n": int(input_n),
        "fitted_n": int(fitted_n),
        "zero_share_input": zero_share,
        "terms_requested": "|".join(terms_requested),
        "terms_produced": "|".join(terms_produced),
        "error_reason": error_reason or "",
    }


def summarize_fit_statuses(rows: list[dict[str, object]]) -> dict[str, int]:
    """Report expected and successful fit/term counts from status diagnostics."""
    statuses = [row for row in rows if row.get("record_type", "fit_status") == "fit_status"]
    return {
        "expected_fits": len(statuses),
        "produced_fits": sum(row.get("status") in {"ok", "partial"} for row in statuses),
        "expected_terms": sum(bool(term) for row in statuses for term in str(row.get("terms_requested", "")).split("|")),
        "produced_terms": sum(bool(term) for row in statuses for term in str(row.get("terms_produced", "")).split("|")),
    }


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
