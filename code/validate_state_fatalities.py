"""Compare validated state-DOT person-fatality counts with canonical FARS.

The comparison produces review evidence; it never turns a numerical threshold
into an automatic acceptance decision.  A state-year is accepted only when it
has valid coverage *and* appears in the reviewed, version-controlled allowlist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_ALLOWLIST_COLUMNS = {"state", "year"}


def _normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if value is not None and not pd.isna(value) else False


def load_review_allowlist(path: str | Path | None) -> pd.DataFrame:
    """Read explicit reviewed decisions, returning no accepted units if absent.

    An absent allowlist is deliberate during candidate-report generation: all
    otherwise-valid units stay ``pending``.  A present malformed allowlist is
    an operator error and fails loudly rather than accepting by accident.
    Only rows whose ``review_status`` is ``accepted`` (or that omit the
    column entirely, for backward compatibility with allowlists that only
    ever recorded acceptances) are returned: a reviewed file must be able to
    record an explicit rejection without every other caller misreading mere
    presence in the file as acceptance.
    """
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=["state", "year", "reason"])
    allowlist = pd.read_csv(path)
    missing = REQUIRED_ALLOWLIST_COLUMNS - set(allowlist.columns)
    if missing:
        raise ValueError(f"allowlist missing required columns: {sorted(missing)}")
    columns = ["state", "year", "reason"]
    if "review_status" in allowlist.columns:
        columns.append("review_status")
    result = allowlist.loc[:, [column for column in columns if column in allowlist.columns]].copy()
    result["state"] = result["state"].astype(str).str.strip().str.upper()
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    if "reason" not in result.columns:
        result["reason"] = ""
    if result.duplicated(["state", "year"]).any():
        raise ValueError("allowlist contains duplicate state-year decisions")
    if "review_status" in result.columns:
        result = result.loc[result["review_status"].astype(str).str.strip().str.lower().eq("accepted")]
        result = result.drop(columns="review_status")
    return result.reset_index(drop=True)


def _event_totals(events: pd.DataFrame, *, county_fips: frozenset[str], year: int) -> pd.DataFrame:
    """Return one row per county-date, with absent event rows interpreted as zero.

    This operation is used only after coverage validity has been assessed in
    the manifest.  It is comparison bookkeeping, not panel balancing.

    Filtering by the source's exact ``expected_county_fips`` set (rather than
    a 2-digit state-FIPS prefix) is equivalent for every full-state source --
    each one's contract already claims its state's entire county set -- and
    is required for a sub-state source (e.g. a single county) whose claimed
    geography is a strict subset of its state's counties.
    """
    if events.empty or not {"fips", "date"}.issubset(events.columns):
        return pd.DataFrame(columns=["fips", "date", "person_fatals"])
    out = events.loc[:, [column for column in ("fips", "date", "person_fatals")
                         if column in events.columns]].copy()
    if "person_fatals" not in out.columns:
        out["person_fatals"] = np.nan
    out["fips"] = _normalize_fips(out["fips"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.loc[out["date"].dt.year.eq(int(year)) & out["fips"].isin(county_fips)]
    out["person_fatals"] = pd.to_numeric(out["person_fatals"], errors="coerce")
    return out.groupby(["fips", "date"], as_index=False)["person_fatals"].sum(min_count=1)


def _pearson(left: pd.Series, right: pd.Series) -> float:
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 2 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return float("nan")
    return float(paired["left"].corr(paired["right"], method="pearson"))


def _comparison_metrics(dot: pd.DataFrame, fars: pd.DataFrame) -> dict[str, float]:
    dot_total = dot["person_fatals"].sum(min_count=1)
    fars_total = fars["person_fatals"].sum(min_count=1)
    ratio = (float(dot_total) / float(fars_total)
             if pd.notna(dot_total) and pd.notna(fars_total) and fars_total != 0 else float("nan"))

    county = (dot.groupby("fips", as_index=False)["person_fatals"].sum(min_count=1)
              .merge(fars.groupby("fips", as_index=False)["person_fatals"].sum(min_count=1),
                     on="fips", how="outer", suffixes=("_dot", "_fars"))
              .fillna(0))
    county_pearson = _pearson(county["person_fatals_dot"], county["person_fatals_fars"])

    daily = dot.merge(fars, on=["fips", "date"], how="outer", suffixes=("_dot", "_fars")).fillna(0)
    agreement = (float(daily["person_fatals_dot"].eq(daily["person_fatals_fars"]).mean())
                 if len(daily) else float("nan"))
    return {
        "dot_person_fatals": float(dot_total) if pd.notna(dot_total) else float("nan"),
        "fars_person_fatals": float(fars_total) if pd.notna(fars_total) else float("nan"),
        "dot_fars_ratio": ratio,
        "county_year_pearson": county_pearson,
        "county_date_agreement": agreement,
    }


def validate_state_fatalities(
    state_events: pd.DataFrame,
    fars_events: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    allowlist_path: str | Path | None = "config/accepted_state_years.csv",
    states: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create review rows for every state-year represented by a DOT manifest."""
    required = {"state", "year", "coverage_valid"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    coverage = manifest.copy()
    coverage["state"] = coverage["state"].astype(str).str.strip().str.upper()
    coverage["year"] = pd.to_numeric(coverage["year"], errors="raise").astype(int)
    coverage["coverage_valid"] = coverage["coverage_valid"].map(_as_bool)
    coverage["invalid_geography_count"] = pd.to_numeric(
        coverage.get("invalid_geography_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    # A combined manifest also includes national FARS rows.  Those rows are
    # the comparison reference, never candidates for state-DOT review.
    from state_dot_sources import STATE_SOURCE_SPECS
    coverage = coverage.loc[coverage["state"].isin(STATE_SOURCE_SPECS)]
    if states is not None:
        selected = {str(state).upper() for state in states}
        coverage = coverage.loc[coverage["state"].isin(selected)]
    if coverage.empty:
        return pd.DataFrame(columns=[
            "state", "year", "dot_person_fatals", "fars_person_fatals", "dot_fars_ratio",
            "county_year_pearson", "county_date_agreement", "invalid_geography_count",
            "coverage_valid", "review_status", "review_reason",
        ])

    allowlist = load_review_allowlist(allowlist_path)
    allow_map = {(row.state, int(row.year)): row.reason
                 for row in allowlist.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for (state, year), unit in coverage.groupby(["state", "year"], sort=True):
        # A state-year cannot be balanced if any constituent reporting unit
        # failed (notably Wisconsin's county-year requests).
        coverage_valid = bool(unit["coverage_valid"].all())
        invalid_geography_count = int(unit["invalid_geography_count"].sum())
        county_fips = _expected_county_fips(state)
        metrics = _comparison_metrics(
            _event_totals(state_events, county_fips=county_fips, year=year),
            _event_totals(fars_events, county_fips=county_fips, year=year),
        )
        decision = allow_map.get((state, int(year)))
        if not coverage_valid:
            review_status, review_reason = "rejected_coverage", "coverage manifest is invalid"
        elif decision is None:
            review_status, review_reason = "pending", "not listed in reviewed allowlist"
        else:
            review_status, review_reason = "accepted", str(decision)
        rows.append({
            "state": state, "year": int(year), **metrics,
            "invalid_geography_count": invalid_geography_count,
            "coverage_valid": coverage_valid,
            "review_status": review_status, "review_reason": review_reason,
        })
    return pd.DataFrame(rows)


def _expected_county_fips(state: str) -> frozenset[str]:
    """Resolve a source's claimed county universe from its contract."""
    from state_dot_sources import get_spec

    return get_spec(state).expected_county_fips


def write_validation_report(report: pd.DataFrame, path: str | Path) -> Path:
    """Write a deterministic review CSV without changing source data."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    report.sort_values(["state", "year"], kind="mergesort").to_csv(target, index=False, lineterminator="\n")
    return target
