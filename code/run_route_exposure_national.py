"""Build year-matched national route exposures and run the established models.

The runner consumes already-routed county-segment partitions.  It deliberately
does not download LODES/ACS data or start OSRM: those bounded build stages are
owned by the national flow, network, and segment commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from config import DATA_PROC, OUTPUT_TABS
from run_route_exposure_pilot import (
    SAME_TRACT_MODES,
    build_alert_date_exposures,
    load_reviewed_route_alerts,
)


NATIONAL_LABEL = "route_exposure_national"
SEGMENT_SCHEMA_V1 = "route_national.segments.v1"
LEGACY_SEGMENT_SCHEMA = "route_national.segments.legacy.v0"
SUPPORTED_SEGMENT_SCHEMAS = {SEGMENT_SCHEMA_V1, LEGACY_SEGMENT_SCHEMA}
TASK5_PROVENANCE_COLUMNS = (
    "analysis_year",
    "lodes_source_year",
    "acs_car_share_vintage",
    "source_manifest_id",
    "network_manifest_id",
    "source_partition_id",
)
ROUTE_TREATMENTS = (
    "own_affected_share",
    "cross_affected_share",
    "pass_through_affected_share",
)
ROUTE_ZERO_COLUMNS = (
    "affected_commuter_car_miles",
    "own_affected_car_miles",
    "cross_affected_car_miles",
    "pass_through_affected_car_miles",
    "affected_route_share",
    *ROUTE_TREATMENTS,
    "affected_commuter_car_miles_per_10000",
)
ROUTE_DENOMINATOR_COLUMNS = (
    "total_commuter_car_miles",
    "own_commuter_car_miles",
    "cross_commuter_car_miles",
    "pass_through_commuter_car_miles",
    "failed_route_commuter_car_weight",
    "unallocated_commuter_car_weight",
    "omitted_coordinate_worker_weight",
    "omitted_car_share_worker_weight",
)
ROUTE_COVERAGE_INCLUDED = "included_positive_denominator"
ROUTE_COVERAGE_EXCLUDED = "excluded_missing_or_nonpositive_denominator"
GATE_SCHEMA_VERSION = "route_national.gates.v1"


def _gate_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number) and number in {0.0, 1.0}:
            return bool(number)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError("gate boolean must be true/false or 1/0")


def _gate_number(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("gate number must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("gate number must be finite")
    return number


def evaluate_national_gates(metrics: Mapping[str, object]) -> dict[str, object]:
    """Evaluate the national acceptance contract, failing closed on omissions."""
    definitions = (
        ("coverage", "successful_commuter_car_share", lambda value: _gate_number(value) >= 0.99, ">= 0.99"),
        ("route_conservation", "maximum_route_conservation_error", lambda value: _gate_number(value) <= 0.005, "<= 0.005"),
        ("aggregate_conservation", "aggregate_conservation_error", lambda value: _gate_number(value) <= 0.001, "<= 0.001"),
        ("denominators", "positive_denominators", _gate_bool, "true"),
        ("partition_availability", "all_partitions_available", _gate_bool, "true"),
        ("alerted_origin_bias", "no_alerted_origin_bias", _gate_bool, "true"),
        ("same_tract_stability", "same_tract_stable", _gate_bool, "true"),
        ("explicit_accounting", "omissions_and_failures_explicit", _gate_bool, "true"),
        ("route_destination_comparison", "route_comparison_complete", _gate_bool, "true"),
        ("runtime_footprint", "runtime_seconds", lambda value: _gate_number(value) >= 0, ">= 0 seconds"),
        ("storage_footprint", "storage_bytes", lambda value: _gate_number(value) >= 0, ">= 0 bytes"),
        ("restart_footprint", "restart_reused_share", lambda value: _gate_number(value) >= 0.99, ">= 0.99"),
    )
    rows: list[dict[str, object]] = []
    for gate, metric, check, threshold in definitions:
        value = metrics.get(metric)
        try:
            passed = value is not None and not pd.isna(value) and bool(check(value))
        except (TypeError, ValueError, OverflowError):
            passed = False
        rows.append(
            {
                "gate": gate,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "passed": passed,
            }
        )
    failed = [str(row["gate"]) for row in rows if not row["passed"]]
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "accepted": not failed,
        "failed_gates": failed,
        "metrics": dict(metrics),
        "gates": rows,
    }


def validate_requested_partitions(
    manifest: pd.DataFrame,
    *,
    analysis_years: Sequence[int],
    states: Sequence[str],
) -> dict[str, object]:
    """Validate the requested state-by-year grid before any model is fitted."""
    frame = manifest.copy()
    if "state" not in frame and "work_state" in frame:
        frame = frame.rename(columns={"work_state": "state"})
    required = {"analysis_year", "state", "status"}
    if missing_columns := required - set(frame):
        return {
            "complete": False,
            "missing_columns": sorted(missing_columns),
            "missing_partitions": [],
            "unavailable_partitions": [],
            "duplicate_partitions": [],
        }
    years = sorted({_integer(year, "analysis year") for year in analysis_years})
    requested_states = sorted({str(state).strip().lower() for state in states})
    frame["analysis_year"] = frame["analysis_year"].map(
        lambda value: _integer(value, "manifest analysis_year")
    )
    frame["state"] = frame["state"].astype(str).str.strip().str.lower()
    requested = {(year, state) for year in years for state in requested_states}
    observed = set(
        frame.loc[
            frame["analysis_year"].isin(years) & frame["state"].isin(requested_states),
            ["analysis_year", "state"],
        ].itertuples(index=False, name=None)
    )
    missing = sorted(requested - observed)
    counts = frame.groupby(["analysis_year", "state"], dropna=False).size()
    duplicates = sorted(
        (int(year), str(state))
        for (year, state), count in counts.items()
        if (year, state) in requested and int(count) != 1
    )
    successful = frame["status"].astype(str).str.lower().isin(
        {"success", "ok", "complete", "reused"}
    )
    unavailable = sorted(
        set(
            frame.loc[
                ~successful
                & frame["analysis_year"].isin(years)
                & frame["state"].isin(requested_states),
                ["analysis_year", "state"],
            ].itertuples(index=False, name=None)
        )
    )

    def records(values: Sequence[tuple[int, str]]) -> list[dict[str, object]]:
        return [
            {"analysis_year": int(year), "state": str(state)}
            for year, state in values
        ]

    return {
        "complete": not missing and not unavailable and not duplicates,
        "missing_columns": [],
        "missing_partitions": records(missing),
        "unavailable_partitions": records(unavailable),
        "duplicate_partitions": records(duplicates),
    }


def evaluate_same_tract_results(
    model_results: pd.DataFrame,
    *,
    same_tract_commuter_car_weight_share: object,
    provenance: Mapping[str, object] | None = None,
    analysis_scope: str = "pooled",
) -> dict[str, object]:
    """Require comparable three-mode estimates and reject any sign reversal."""
    frame = model_results.copy()
    provenance_result = dict(provenance or {"valid": True, "status": "not_assessed"})
    required = {"same_tract_mode", "term"}
    result_column = "coef" if "coef" in frame else "estimate" if "estimate" in frame else None
    if result_column is None:
        required.add("estimate")
    if missing := required - set(frame):
        return {
            "complete": False,
            "stable": False,
            "missing_columns": sorted(missing),
            "missing_modes": sorted(SAME_TRACT_MODES),
            "missing_terms": list(ROUTE_TREATMENTS),
            "sign_reversal_terms": [],
            "dominance_acceptable": False,
            "provenance": provenance_result,
        }
    # The established symmetric commuter runner emits ``coef`` while older
    # sensitivity tables used ``estimate``.  Normalize at this boundary so
    # the gate validates the actual model output without accepting a table
    # that contains neither numeric result column.  If both are present,
    # ``coef`` is authoritative; an invalid established result must not be
    # masked by a stale compatibility column.
    frame["estimate"] = frame[result_column]
    if "analysis_scope" in frame:
        frame = frame.loc[
            frame["analysis_scope"].astype(str).eq(str(analysis_scope))
        ].copy()
    frame["same_tract_mode"] = frame["same_tract_mode"].astype(str)
    present_modes = set(frame["same_tract_mode"])
    missing_modes = sorted(SAME_TRACT_MODES - present_modes)
    missing_terms = sorted(set(ROUTE_TREATMENTS) - set(frame["term"].astype(str)))
    estimate = pd.to_numeric(frame["estimate"], errors="coerce")
    frame["estimate"] = estimate
    identity_candidates = (
        "term",
        "outcome",
        "spec",
        "fixed_effect_spec",
        "control_spec",
        "other_wea_control",
        "inference_spec",
    )
    identity = [column for column in identity_candidates if column in frame]
    duplicate = bool(frame.duplicated([*identity, "same_tract_mode"]).any())
    if identity and not frame.empty and not duplicate:
        comparison = frame.pivot(index=identity, columns="same_tract_mode", values="estimate")
    else:
        comparison = pd.DataFrame()
    complete = bool(
        not missing_modes
        and not missing_terms
        and not duplicate
        and not comparison.empty
        and set(SAME_TRACT_MODES).issubset(comparison.columns)
        and comparison[list(SAME_TRACT_MODES)].notna().all().all()
    )
    reversals: list[str] = []
    if complete:
        for index, row in comparison.iterrows():
            signs = set(np.sign(row[list(SAME_TRACT_MODES)].astype(float))) - {0.0}
            if signs == {-1.0, 1.0}:
                term = index[identity.index("term")] if isinstance(index, tuple) else index
                reversals.append(str(term))
    try:
        dominance_share = _gate_number(same_tract_commuter_car_weight_share)
        dominance_acceptable = 0 <= dominance_share < 0.5
    except (TypeError, ValueError, OverflowError):
        dominance_share = None
        dominance_acceptable = False
    provenance_valid = provenance_result.get("valid") is True
    return {
        "complete": complete,
        "stable": bool(
            complete and dominance_acceptable and not reversals and provenance_valid
        ),
        "missing_columns": [],
        "missing_modes": missing_modes,
        "missing_terms": missing_terms,
        "duplicate_comparisons": duplicate,
        "sign_reversal_terms": sorted(set(reversals)),
        "same_tract_commuter_car_weight_share": dominance_share,
        "dominance_acceptable": dominance_acceptable,
        "provenance": provenance_result,
    }


def _same_tract_provenance(
    model_results: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    network_year: int | None,
    analysis_scope: str = "pooled",
) -> dict[str, object]:
    """Require sensitivity estimates to identify the exact evaluated manifest."""
    required = {
        "analysis_scope",
        "analysis_years",
        "states",
        "network_year",
        "source_manifest_ids",
        "source_partition_ids",
        "network_manifest_ids",
    }
    missing = sorted(required - set(model_results))
    if missing:
        return {"valid": False, "missing_columns": missing, "mismatches": {}}
    scoped = model_results.loc[
        model_results["analysis_scope"].astype(str).eq(str(analysis_scope))
    ].copy()
    if scoped.empty or network_year is None:
        return {
            "valid": False,
            "missing_columns": [],
            "mismatches": {
                "scope": analysis_scope,
                "scope_rows": "missing" if scoped.empty else "present",
                "network_year": network_year,
            },
        }

    state_column = "state" if "state" in manifest else "work_state"
    expected = {
        "analysis_years": ",".join(
            map(str, sorted(pd.to_numeric(manifest["analysis_year"]).astype(int).unique()))
        ),
        "states": ",".join(
            sorted(manifest[state_column].astype(str).str.strip().str.lower().unique())
        ),
        "source_manifest_ids": ",".join(
            sorted(manifest["source_manifest_id"].astype(str).str.strip().unique())
        ),
        "source_partition_ids": ",".join(
            sorted(manifest["source_partition_id"].astype(str).str.strip().unique())
        ),
        "network_manifest_ids": ",".join(
            sorted(manifest["network_manifest_id"].astype(str).str.strip().unique())
        ),
        "network_year": str(int(network_year)),
    }
    mismatches: dict[str, object] = {}
    for column, wanted in expected.items():
        if column == "network_year":
            actual = pd.to_numeric(scoped[column], errors="coerce")
            actual_values = sorted(
                {str(int(value)) for value in actual.dropna() if float(value).is_integer()}
            )
        else:
            actual_values = sorted(
                {
                    ",".join(
                        sorted(
                            token.strip().lower() if column == "states" else token.strip()
                            for token in str(value).split(",")
                            if token.strip()
                        )
                    )
                    for value in scoped[column].dropna()
                }
            )
        if actual_values != [wanted]:
            mismatches[column] = {"expected": wanted, "actual": actual_values}
    return {
        "valid": not mismatches,
        "missing_columns": [],
        "mismatches": mismatches,
        "expected": expected,
    }


@dataclass(frozen=True)
class RouteModelSpec:
    """One established control, fixed-effect, and inference specification."""

    label: str
    fixed_effect_label: str
    fixed_effect_cols: tuple[str, ...]
    controls: tuple[str, ...]
    other_wea_control: str
    inference: str
    wild_kind: str
    randomization: bool = False


DEFAULT_MANIFEST = (
    DATA_PROC / "commuting" / "route_national" / "segments" / "segment_manifest.csv"
)


def _integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{label} must be an integer")
    return parsed


def _fips(values: pd.Series) -> pd.Series:
    missing = values.isna()
    out = values.astype("string").str.strip().str.removesuffix(".0").str.zfill(5)
    if (missing | ~out.str.fullmatch(r"\d{5}", na=False)).any():
        raise ValueError("invalid or missing county FIPS")
    return out


def _nonblank(value: object, label: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{label} must be nonblank")
    return str(value).strip()


def _joined_unique(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    return ",".join(sorted({str(value).strip() for value in frame[column].dropna() if str(value).strip()}))


def _manifest_state_label(rows: pd.DataFrame) -> str:
    for column in ("state", "work_state"):
        if column in rows:
            return ",".join(
                sorted(rows[column].astype(str).str.strip().str.lower().unique())
            )
    return ""


def _apply_same_tract_mode(segments: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "primary_calibrated":
        return segments
    if "same_tract" in segments:
        same = segments["same_tract"].fillna(False).astype(bool)
    elif {"home_tract", "work_tract"}.issubset(segments):
        same = segments["home_tract"].notna() & segments["home_tract"].eq(segments["work_tract"])
    else:
        raise ValueError(
            "zero/exclude same-tract sensitivity requires same_tract or tract identifiers"
        )
    if mode == "exclude":
        return segments.loc[~same].copy()
    out = segments.copy()
    for column in ("route_miles_in_county", "route_miles_total", "unallocated_miles"):
        if column in out:
            out.loc[same, column] = 0.0
    return out


def reconcile_route_audit_weights(
    county_segments: pd.DataFrame,
    route_audits: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Reconcile unresolved commuter-car weight once per route.

    County segments contain successful county allocations and, for some
    legacy artifacts, explicit ``failed_route`` rows.  The national segment
    runner records failed routes only in its audit table.  This helper unions
    those two representations by route identity, preferring the audit weight
    when present, and never turns an unresolved route into county exposure.
    """

    def identity_column(frame: pd.DataFrame) -> str | None:
        return next(
            (column for column in ("route_id", "route_signature") if column in frame),
            None,
        )

    def route_weights(frame: pd.DataFrame, identity: str) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype="float64")
        if "commuter_car_weight" in frame:
            weight = pd.to_numeric(frame["commuter_car_weight"], errors="coerce")
        elif {"workers", "home_car_share"}.issubset(frame):
            weight = pd.to_numeric(frame["workers"], errors="coerce") * pd.to_numeric(
                frame["home_car_share"], errors="coerce"
            )
        else:
            return pd.Series(dtype="float64")
        work = pd.DataFrame({identity: frame[identity].astype(str), "weight": weight})
        work = work.loc[work[identity].str.strip().ne("")].copy()
        work["weight"] = work["weight"].fillna(0.0)
        return work.groupby(identity, sort=False)["weight"].max()

    segments = county_segments.copy()
    segment_identity = identity_column(segments)
    segment_weights = (
        route_weights(segments, segment_identity) if segment_identity else pd.Series(dtype="float64")
    )
    segment_types = (
        segments.get(
            "segment_type", pd.Series("county", index=segments.index)
        )
        .astype(str)
        .str.strip()
        .str.lower()
    )
    segment_ids = (
        segments[segment_identity].astype(str)
        if segment_identity
        else pd.Series(dtype="string")
    )
    failed_ids = set(segment_ids.loc[segment_types.isin({"failed_route", "failedroute"})])
    unallocated_ids = set(segment_ids.loc[segment_types.eq("unallocated")])

    audit_weights = pd.Series(dtype="float64")
    if route_audits is not None and not route_audits.empty:
        audits = route_audits.copy()
        audit_identity = identity_column(audits)
        if audit_identity is None:
            raise ValueError("route audits require route_id or route_signature")
        if "status" not in audits:
            raise ValueError("route audits require status")
        audit_weights = route_weights(audits, audit_identity)
        audit_ids = audits[audit_identity].astype(str)
        eligible = audits.get(
            "routing_eligible", pd.Series(True, index=audits.index)
        ).fillna(False).astype(bool)
        status = audits["status"].astype(str).str.strip().str.lower()
        failed_ids.update(set(audit_ids.loc[eligible & ~status.eq("ok")]))
        unallocated_miles = pd.to_numeric(
            audits.get("unallocated_miles", pd.Series(0.0, index=audits.index)),
            errors="coerce",
        ).fillna(0.0)
        unallocated_ids.update(
            set(audit_ids.loc[eligible & status.eq("ok") & unallocated_miles.gt(0)])
        )

    # A failed route's full weight is accounted for under failure; its audit
    # mileage is not also counted as an unallocated successful route.
    unallocated_ids.difference_update(failed_ids)
    weights = audit_weights.combine_first(segment_weights)

    def total(ids: set[str]) -> float:
        if not ids or weights.empty:
            return 0.0
        return float(weights.reindex(sorted(ids)).fillna(0.0).sum())

    return {
        "failed_route_commuter_car_weight": total(failed_ids),
        "unallocated_commuter_car_weight": total(unallocated_ids),
        "failed_route_ids": sorted(failed_ids),
        "unallocated_route_ids": sorted(unallocated_ids),
    }


def _validate_route_audits_for_exposure(
    county_segments: pd.DataFrame, route_audits: pd.DataFrame | None
) -> None:
    """Fail closed when a non-successful audit is represented as exposure."""
    if route_audits is None:
        return
    if not isinstance(route_audits, pd.DataFrame):
        raise TypeError("route audits must be a pandas DataFrame")
    if route_audits.empty:
        if not county_segments.empty:
            raise ValueError("route audits are empty while county segments are present")
        return
    identity = next(
        (
            column
            for column in ("route_id", "route_signature")
            if column in county_segments.columns and column in route_audits.columns
        ),
        None,
    )
    if identity is None:
        raise ValueError(
            "route audits and county segments require a shared route_id or route_signature"
        )
    if "status" not in route_audits.columns:
        raise ValueError("route audits require status")

    segment_ids = county_segments[identity].astype("string").str.strip()
    audit_ids = route_audits[identity].astype("string").str.strip()
    if segment_ids.isna().any() or segment_ids.eq("").any():
        raise ValueError(f"county segments contain blank {identity}")
    if audit_ids.isna().any() or audit_ids.eq("").any():
        raise ValueError(f"route audits contain blank {identity}")
    if audit_ids.duplicated().any():
        raise ValueError(f"route audits contain duplicate {identity}")

    status = route_audits["status"].astype("string").str.strip().str.lower()
    if status.isna().any() or status.eq("").any():
        raise ValueError("route audits contain blank status")
    non_ok_ids = set(audit_ids.loc[~status.eq("ok")].astype(str))
    conflicting_ids = sorted(
        non_ok_ids.intersection(set(segment_ids.astype(str)))
    )
    if conflicting_ids:
        raise ValueError(
            "non-Ok route audit has county segments: "
            f"{conflicting_ids}"
        )


def build_national_exposure(
    county_segments: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    analysis_year: int,
    flow_source_year: int,
    car_share_vintage: str,
    same_tract_mode: str = "primary_calibrated",
    route_audits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build county-date route dosage with complete source metadata."""
    year = _integer(analysis_year, "analysis_year")
    source_year = _integer(flow_source_year, "flow_source_year")
    vintage = _nonblank(car_share_vintage, "car_share_vintage")
    if same_tract_mode not in SAME_TRACT_MODES:
        raise ValueError("same_tract_mode must be primary_calibrated, zero, or exclude")

    segments = _apply_same_tract_mode(county_segments.copy(), same_tract_mode)
    _validate_route_audits_for_exposure(segments, route_audits)
    for column, expected in (
        ("analysis_year", year),
        ("lodes_source_year", source_year),
        ("acs_car_share_vintage", vintage),
    ):
        if column not in segments:
            continue
        values = segments[column].dropna().astype(str).str.strip().unique()
        if len(values) and set(values) != {str(expected)}:
            raise ValueError(f"county_segments {column} conflicts with requested vintage")

    selected_alerts = alerts.copy()
    date_column = "alert_date" if "alert_date" in selected_alerts else "effective_crash_date"
    if date_column in selected_alerts:
        parsed_dates = pd.to_datetime(selected_alerts[date_column], errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError("alerts has invalid alert date")
        selected_alerts = selected_alerts.loc[parsed_dates.dt.year.eq(year)].copy()

    metadata = {
        "analysis_year": year,
        "lodes_source_year": source_year,
        "acs_car_share_vintage": vintage,
        "source_manifest_ids": _joined_unique(segments, "source_manifest_id"),
        "source_partition_ids": _joined_unique(segments, "source_partition_id"),
    }
    exposure = build_alert_date_exposures(
        segments,
        selected_alerts,
        same_tract_mode,
        label=NATIONAL_LABEL,
        vintage_columns=metadata,
    )
    audit_weights = reconcile_route_audit_weights(segments, route_audits)
    for column in (
        "failed_route_commuter_car_weight",
        "unallocated_commuter_car_weight",
    ):
        if not exposure.empty:
            exposure[column] = float(audit_weights[column])
    return exposure


def required_flow_partitions(
    analysis_years: Sequence[int], mapping: Mapping[int, int]
) -> list[int]:
    """Return sorted unique source flow years needed by the analysis years."""
    years = [_integer(year, "analysis year") for year in analysis_years]
    missing = [year for year in years if year not in mapping]
    if missing:
        raise ValueError(f"missing flow mapping for analysis years: {sorted(set(missing))}")
    return sorted({_integer(mapping[year], f"flow mapping for {year}") for year in years})


def _read_table(value: Path | str | pd.DataFrame, label: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported {label} format: {path.suffix}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_partition_value(
    frame: pd.DataFrame, column: str, *, require_complete: bool = False
) -> object:
    if column not in frame:
        raise ValueError(f"segment partition provenance missing {column}")
    series = frame[column]
    if require_complete:
        blank = series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if series.isna().any() or blank.any():
            raise ValueError(
                f"segment partition provenance {column} has missing or blank rows"
            )
        values = series.drop_duplicates()
    else:
        values = series.dropna().drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"segment partition provenance {column} must have one value")
    return values.iloc[0]


def _validate_segment_partition(
    frame: pd.DataFrame,
    manifest_row: pd.Series,
    *,
    path: Path,
    checksum: str,
) -> None:
    schema = str(manifest_row["schema_version"]).strip()
    if "segment_sha256" in manifest_row.index and pd.notna(manifest_row["segment_sha256"]):
        expected_checksum = str(manifest_row["segment_sha256"]).strip().lower()
        if expected_checksum != checksum:
            raise ValueError(f"segment partition checksum mismatch: {path}")

    expected: dict[str, object] = {
        "lodes_source_year": int(manifest_row["lodes_source_year"]),
        "acs_car_share_vintage": str(manifest_row["acs_car_share_vintage"]),
    }
    if schema == SEGMENT_SCHEMA_V1:
        expected.update({
            "schema_version": SEGMENT_SCHEMA_V1,
            "analysis_year": int(manifest_row["analysis_year"]),
            "source_manifest_id": str(manifest_row["source_manifest_id"]),
            "network_manifest_id": str(manifest_row["network_manifest_id"]),
            "source_partition_id": str(manifest_row["source_partition_id"]),
        })
    elif "analysis_year" in frame:
        expected["analysis_year"] = int(
            manifest_row.get("segment_analysis_year", manifest_row["analysis_year"])
        )
    for column, wanted in expected.items():
        actual = _unique_partition_value(
            frame, column, require_complete=schema == SEGMENT_SCHEMA_V1
        )
        if column.endswith("year"):
            try:
                matches = _integer(actual, f"segment {column}") == int(wanted)
            except ValueError:
                matches = False
        else:
            matches = str(actual).strip() == str(wanted).strip()
        if not matches:
            raise ValueError(
                f"segment partition provenance mismatch for {column}: "
                f"manifest={wanted!r}, partition={actual!r}, path={path}"
            )

    if schema == LEGACY_SEGMENT_SCHEMA and "schema_version" in frame:
        actual_schema = str(_unique_partition_value(frame, "schema_version")).strip()
        if actual_schema != LEGACY_SEGMENT_SCHEMA:
            raise ValueError(
                "segment partition provenance mismatch for schema_version: "
                f"manifest={LEGACY_SEGMENT_SCHEMA!r}, partition={actual_schema!r}, "
                f"path={path}"
            )

    for column in ("source_manifest_id", "network_manifest_id", "source_partition_id"):
        if schema == SEGMENT_SCHEMA_V1:
            continue
        if column in manifest_row.index and pd.notna(manifest_row[column]):
            actual = str(_unique_partition_value(frame, column)).strip()
            wanted = str(manifest_row[column]).strip()
            if not wanted or actual != wanted:
                raise ValueError(
                    f"segment partition provenance mismatch for {column}: "
                    f"manifest={wanted!r}, partition={actual!r}, path={path}"
                )


def _manifest_frame(value: Path | str | pd.DataFrame) -> pd.DataFrame:
    manifest = _read_table(value, "segment manifest")
    aliases = {
        "flow_source_year": "lodes_source_year",
        "car_share_vintage": "acs_car_share_vintage",
        "path": "segment_path",
    }
    for source, target in aliases.items():
        if target not in manifest and source in manifest:
            manifest = manifest.rename(columns={source: target})
    required = {
        "schema_version",
        "analysis_year",
        "lodes_source_year",
        "acs_car_share_vintage",
        "segment_path",
    }
    if missing := required - set(manifest):
        raise ValueError(f"segment manifest missing required columns: {sorted(missing)}")
    manifest["schema_version"] = manifest["schema_version"].map(
        lambda value: _nonblank(value, "manifest schema_version")
    )
    unsupported = sorted(set(manifest["schema_version"]) - SUPPORTED_SEGMENT_SCHEMAS)
    if unsupported:
        raise ValueError(f"unsupported segment schema versions: {unsupported}")
    v1 = manifest["schema_version"].eq(SEGMENT_SCHEMA_V1)
    if v1.any():
        identity_columns = set(TASK5_PROVENANCE_COLUMNS) - {
            "analysis_year", "lodes_source_year", "acs_car_share_vintage"
        }
        if missing := identity_columns - set(manifest):
            raise ValueError(
                f"v1 segment manifest missing required columns: {sorted(missing)}"
            )
        for column in sorted(identity_columns):
            manifest.loc[v1, column] = manifest.loc[v1, column].map(
                lambda value, name=column: _nonblank(value, f"manifest {name}")
            )
    manifest["analysis_year"] = manifest["analysis_year"].map(
        lambda value: _integer(value, "manifest analysis_year")
    )
    manifest["lodes_source_year"] = manifest["lodes_source_year"].map(
        lambda value: _integer(value, "manifest lodes_source_year")
    )
    manifest["acs_car_share_vintage"] = manifest["acs_car_share_vintage"].map(
        lambda value: _nonblank(value, "manifest acs_car_share_vintage")
    )
    manifest["segment_path"] = manifest["segment_path"].map(str)
    return manifest.sort_values(
        ["analysis_year", "lodes_source_year", "acs_car_share_vintage", "segment_path"]
    ).reset_index(drop=True)


def _combine_year_segments(
    rows: pd.DataFrame, cache: dict[str, tuple[pd.DataFrame, str]]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if rows["segment_path"].duplicated().any():
        raise ValueError("segment manifest repeats a partition within an analysis year")
    for _, manifest_row in rows.iterrows():
        raw_path = manifest_row["segment_path"]
        path = str(Path(raw_path).expanduser().resolve())
        if path not in cache:
            path_object = Path(path)
            cache[path] = (
                _read_table(path_object, "county-segment partition"),
                _file_sha256(path_object),
            )
        frame, checksum = cache[path]
        _validate_segment_partition(
            frame, manifest_row, path=Path(path), checksum=checksum
        )
        frames.append(frame)
    if not frames:
        raise ValueError("analysis year has no county-segment partitions")
    return pd.concat(frames, ignore_index=True)


def _combine_year_audits(
    rows: pd.DataFrame, cache: dict[str, tuple[pd.DataFrame, str]]
) -> pd.DataFrame | None:
    """Load optional route audits used to account for failed national routes."""
    if "audit_path" not in rows:
        return None
    paths = rows["audit_path"]
    if paths.isna().any() or paths.astype(str).str.strip().eq("").any():
        raise ValueError("national segment manifest has missing audit_path")
    if rows.get("audit_sha256", pd.Series(index=rows.index, dtype=object)).isna().any():
        raise ValueError("national segment manifest has missing audit_sha256")
    frames: list[pd.DataFrame] = []
    for _, manifest_row in rows.iterrows():
        path = str(Path(manifest_row["audit_path"]).expanduser().resolve())
        if path not in cache:
            path_object = Path(path)
            if not path_object.is_file():
                raise FileNotFoundError(f"missing route-audit partition: {path_object}")
            expected = _nonblank(manifest_row["audit_sha256"], "manifest audit_sha256")
            checksum = _file_sha256(path_object)
            if checksum != expected:
                raise ValueError(f"route-audit partition checksum mismatch: {path_object}")
            cache[path] = (_read_table(path_object, "route-audit partition"), checksum)
        frames.append(cache[path][0])
    return pd.concat(frames, ignore_index=True) if frames else None


def _year_exposure(
    rows: pd.DataFrame,
    segments: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    same_tract_mode: str,
    route_audits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    year = int(rows["analysis_year"].iloc[0])
    source_years = sorted(rows["lodes_source_year"].unique())
    vintages = sorted(rows["acs_car_share_vintage"].unique())
    # A real national year may contain state-specific nearest vintages.  Keep
    # their set explicitly rather than pretending all states used one source.
    selected_alerts = alerts.copy()
    date_column = "alert_date" if "alert_date" in selected_alerts else "effective_crash_date"
    if date_column in selected_alerts:
        dates = pd.to_datetime(selected_alerts[date_column], errors="coerce")
        if dates.isna().any():
            raise ValueError("alerts has invalid alert date")
        selected_alerts = selected_alerts.loc[dates.dt.year.eq(year)].copy()
    segments = _apply_same_tract_mode(segments.copy(), same_tract_mode)
    _validate_route_audits_for_exposure(segments, route_audits)
    metadata = {
        "analysis_year": year,
        "lodes_source_year": source_years[0] if len(source_years) == 1 else pd.NA,
        "lodes_source_years": ",".join(map(str, source_years)),
        "acs_car_share_vintage": vintages[0] if len(vintages) == 1 else "mixed",
        "acs_car_share_vintages": ",".join(vintages),
        "source_manifest_ids": _joined_unique(segments, "source_manifest_id"),
        "source_partition_ids": _joined_unique(segments, "source_partition_id"),
        "segment_partition_paths": ",".join(sorted(set(rows["segment_path"]))),
    }
    exposure = build_alert_date_exposures(
        segments,
        selected_alerts,
        same_tract_mode,
        label=NATIONAL_LABEL,
        vintage_columns=metadata,
    )
    audit_weights = reconcile_route_audit_weights(segments, route_audits)
    for column in (
        "failed_route_commuter_car_weight",
        "unallocated_commuter_car_weight",
    ):
        if not exposure.empty:
            exposure[column] = float(audit_weights[column])
    return exposure


def _merge_destination_measure(
    panel: pd.DataFrame, destination: pd.DataFrame | None
) -> pd.DataFrame:
    if destination is None:
        return panel
    existing = destination.copy()
    if "outcome_fips" in existing and "fips" not in existing:
        existing = existing.rename(columns={"outcome_fips": "fips"})
    if "alert_date" in existing and "date" not in existing:
        existing = existing.rename(columns={"alert_date": "date"})
    if not {"fips", "date"}.issubset(existing):
        raise ValueError("destination exposure requires fips/outcome_fips and date/alert_date")
    existing["fips"] = _fips(existing["fips"])
    existing["date"] = pd.to_datetime(existing["date"], errors="raise").dt.normalize()
    if existing.duplicated(["fips", "date"]).any():
        raise ValueError("destination exposure has duplicate county-date rows")
    collisions = (set(existing) & set(panel)) - {"fips", "date"}
    existing = existing.rename(columns={column: f"destination_{column}" for column in collisions})
    return panel.merge(existing, on=["fips", "date"], how="left", validate="one_to_one")


def _attach_exposure_to_panel(
    panel: pd.DataFrame, exposure: pd.DataFrame, metadata: Mapping[str, object]
) -> pd.DataFrame:
    out = panel.copy()
    if "outcome_fips" in out and "fips" not in out:
        out = out.rename(columns={"outcome_fips": "fips"})
    if "alert_date" in out and "date" not in out:
        out = out.rename(columns={"alert_date": "date"})
    if not {"fips", "date"}.issubset(out):
        raise ValueError("analysis panel requires fips/outcome_fips and date/alert_date")
    out["fips"] = _fips(out["fips"])
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if out.duplicated(["fips", "date"]).any():
        raise ValueError("analysis panel has duplicate county-date rows")

    route = exposure.rename(columns={"outcome_fips": "fips", "alert_date": "date"}).copy()
    route["fips"] = _fips(route["fips"])
    route["date"] = pd.to_datetime(route["date"], errors="raise").dt.normalize()
    denominator_columns = [column for column in ROUTE_DENOMINATOR_COLUMNS if column in route]
    denominators = (
        route[["fips", *denominator_columns]].drop_duplicates("fips")
        if denominator_columns
        else pd.DataFrame({"fips": route["fips"].drop_duplicates()})
    )
    out = out.merge(route, on=["fips", "date"], how="left", validate="one_to_one")
    if denominator_columns:
        out = out.merge(
            denominators,
            on="fips",
            how="left",
            suffixes=("", "_year_denominator"),
            validate="many_to_one",
        )
        for column in denominator_columns:
            fallback = f"{column}_year_denominator"
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(
                pd.to_numeric(out.pop(fallback), errors="coerce")
            )
    if "total_commuter_car_miles" in out:
        denominator = pd.to_numeric(out["total_commuter_car_miles"], errors="coerce")
    else:
        denominator = pd.Series(np.nan, index=out.index, dtype=float)
    covered = denominator.gt(0) & np.isfinite(denominator)
    out["route_coverage_status"] = np.where(
        covered, ROUTE_COVERAGE_INCLUDED, ROUTE_COVERAGE_EXCLUDED
    )
    for column in ROUTE_ZERO_COLUMNS:
        if column not in out:
            out[column] = np.nan
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out.loc[covered, column] = out.loc[covered, column].fillna(0.0)
        out.loc[~covered, column] = np.nan
    if NATIONAL_LABEL not in out:
        out[NATIONAL_LABEL] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out[NATIONAL_LABEL] = pd.to_numeric(out[NATIONAL_LABEL], errors="coerce").astype("Int64")
    out.loc[covered, NATIONAL_LABEL] = out.loc[covered, NATIONAL_LABEL].fillna(0)
    out.loc[~covered, NATIONAL_LABEL] = pd.NA
    for column, value in metadata.items():
        out[column] = value
    return out


def _prepare_established_ids(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["year"] = out["date"].dt.year
    out["fips_id"] = pd.factorize(out["fips"], sort=True)[0]
    out["year_id"] = out["year"].astype(int)
    out["dow_id"] = out["date"].dt.dayofweek.astype(int)
    out["fips_year_id"] = pd.factorize(out["fips"] + "|" + out["year"].astype(str), sort=True)[0]
    out["fips_dow_id"] = pd.factorize(out["fips"] + "|" + out["date"].dt.dayofweek.astype(str), sort=True)[0]
    out["month_id"] = out["date"].dt.month.astype(int)
    out["state_cluster_id"] = pd.factorize(out["fips"].str[:2], sort=True)[0]
    out["date_cluster_id"] = pd.factorize(out["date"], sort=True)[0]
    out["state_date_id"] = pd.factorize(
        out["fips"].str[:2] + "|" + out["date"].dt.strftime("%Y-%m-%d"), sort=True
    )[0]
    out["state_month_cluster_id"] = pd.factorize(
        out["fips"].str[:2] + "|" + out["date"].dt.to_period("M").astype(str), sort=True
    )[0]
    return out


def established_route_model_specs(panel: pd.DataFrame) -> tuple[RouteModelSpec, ...]:
    """Return the established control/FE/inference ladder for route dosage.

    Controls are retained when supplied by the established county-day panel;
    the route integration never silently synthesizes or drops them.
    """
    common_candidates = (
        "is_holiday",
        "is_day_after_holiday",
        "prcp_mm",
        "tmax_c",
        "fatals_tm1",
        "day_alert",
    )
    common = tuple(column for column in common_candidates if column in panel)
    other_wea = []
    if "other_wea_night_alert" in panel:
        other_wea.append(("binary", "other_wea_night_alert"))
    if "other_wea_night_count" in panel:
        other_wea.append(("dose", "other_wea_night_count"))
    if not other_wea:
        other_wea.append(("none", None))

    fixed_effects = (
        ("baseline_calendar", ("fips_id", "year_id", "dow_id", "month_id")),
        ("county_year_weekday", ("fips_year_id", "fips_dow_id", "month_id")),
        ("state_date", ("fips_year_id", "fips_dow_id", "state_date_id")),
    )
    specs: list[RouteModelSpec] = []
    for wea_label, wea_column in other_wea:
        controls = (*common, *((wea_column,) if wea_column else ()))
        for fe_label, fe_columns in fixed_effects:
            specs.append(RouteModelSpec(
                label=f"{fe_label}__other_wea_{wea_label}__webb_state",
                fixed_effect_label=fe_label,
                fixed_effect_cols=fe_columns,
                controls=controls,
                other_wea_control=wea_label,
                inference="webb_state",
                wild_kind="webb",
            ))
        specs.append(RouteModelSpec(
            label=f"county_year_weekday__other_wea_{wea_label}__rademacher_state_month",
            fixed_effect_label="county_year_weekday",
            fixed_effect_cols=("fips_year_id", "fips_dow_id", "month_id"),
            controls=controls,
            other_wea_control=wea_label,
            inference="rademacher_state_month",
            wild_kind="rademacher",
            randomization=True,
        ))
    return tuple(specs)


def _established_model_runner(
    panel: pd.DataFrame,
    *,
    scope: str,
    treatment_columns: Sequence[str],
    bootstrap_reps: int = 9_999,
) -> pd.DataFrame:
    """Use the current symmetric commuter model's FE/inference implementation."""
    from run_symmetric_commuter_robustness import _fit_analytic

    prepared = _prepare_established_ids(panel)
    outcomes = [
        column
        for column in ("fatal_crashes", "total_fatals", "person_fatals", "combined_rate")
        if column in prepared
    ]
    if not outcomes:
        raise ValueError("analysis panel has no established fatal-count or combined-rate outcome")
    rows: list[dict[str, object]] = []
    specifications = established_route_model_specs(prepared)
    for spec_index, model_spec in enumerate(specifications):
        terms = [*treatment_columns, *model_spec.controls]
        for outcome_index, outcome in enumerate(outcomes):
            fitted = _fit_analytic(
                prepared,
                outcome,
                terms,
                spec=f"route_{scope}_{model_spec.label}",
                wild_kind=model_spec.wild_kind,
                bootstrap_reps=int(bootstrap_reps),
                bootstrap_seed=20260828 + spec_index * 100 + outcome_index * 10,
                randomization=model_spec.randomization,
                fixed_effect_cols=model_spec.fixed_effect_cols,
                prefer_pyfixest=False,
            )
            for row in fitted:
                if row.get("term") not in treatment_columns:
                    continue
                row.update({
                    "fixed_effect_spec": model_spec.fixed_effect_label,
                    "control_spec": ",".join(model_spec.controls),
                    "other_wea_control": model_spec.other_wea_control,
                    "inference_spec": model_spec.inference,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def _run_models(
    runner: Callable[..., pd.DataFrame],
    panel: pd.DataFrame,
    *,
    scope: str,
    bootstrap_reps: int,
    labels: Mapping[str, object],
) -> pd.DataFrame:
    result = runner(
        panel,
        scope=scope,
        treatment_columns=ROUTE_TREATMENTS,
        bootstrap_reps=bootstrap_reps,
    )
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)
    result = result.copy()
    result["analysis_scope"] = scope
    for column, value in labels.items():
        result[column] = value
    return result


def _provenance_labels(panel: pd.DataFrame) -> dict[str, object]:
    labels: dict[str, object] = {}
    for column in (
        "analysis_years",
        "states",
        "network_year",
        "lodes_source_years",
        "acs_car_share_vintages",
        "source_manifest_ids",
        "source_partition_ids",
        "network_manifest_ids",
        "segment_partition_paths",
        "same_tract_mode",
    ):
        labels[column] = _joined_unique(panel, column)
    source_years = sorted(
        pd.to_numeric(panel.get("lodes_source_year", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    labels["lodes_source_year"] = source_years[0] if len(source_years) == 1 else pd.NA
    vintages = sorted(
        value
        for value in panel.get("acs_car_share_vintage", pd.Series(dtype=str)).dropna().astype(str).unique()
        if value != "mixed"
    )
    labels["acs_car_share_vintage"] = vintages[0] if len(vintages) == 1 else "mixed"
    return labels


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.parquet" if path.suffix.lower() == ".parquet" else ".tmp.csv"
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if path.suffix.lower() == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp.json", dir=path.parent,
        mode="w", encoding="utf-8", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent,
        mode="w", encoding="utf-8", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _invocation_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "analysis_years": [int(year) for year in args.analysis_years],
        "states": sorted(str(state).lower() for state in (getattr(args, "states", None) or [])),
        "network_year": getattr(args, "network_year", None),
        "same_tract_mode": getattr(args, "same_tract_mode", None),
        "chunk_rows": getattr(args, "chunk_rows", None),
        "route_workers": getattr(args, "route_workers", None),
        "checkpoint_every": getattr(args, "checkpoint_every", None),
        "geometry_sample_rate": getattr(args, "geometry_sample_rate", None),
        "segment_manifest": str(getattr(args, "segment_manifest", "")),
    }


def _write_national_gate_report(
    result: dict[str, object], output_dir: Path, *, title: str, note: str
) -> dict[str, object]:
    gates_dir = Path(output_dir) / "gates"
    result["paths"] = {
        **dict(result.get("paths", {})),
        "gate_report": str(gates_dir / "national_gate_report.json"),
        "gate_table": str(gates_dir / "national_gate_table.csv"),
        "partition_gate_table": str(gates_dir / "national_partition_gate_table.csv"),
    }
    partition_results = list(result.get("partition_results", []))
    partition_gate_rows = [
        {"source_partition_id": partition["source_partition_id"], **row}
        for partition in partition_results
        for row in partition.get("gates", [])
    ]
    _atomic_write_json(result, gates_dir / "national_gate_report.json")
    _atomic_write(pd.DataFrame(result.get("gates", [])), gates_dir / "national_gate_table.csv")
    _atomic_write(
        pd.DataFrame(
            partition_gate_rows,
            columns=[
                "source_partition_id", "gate", "metric", "value", "threshold", "passed"
            ],
        ),
        gates_dir / "national_partition_gate_table.csv",
    )
    partition_summary = pd.DataFrame(
        {
            "source_partition_id": [item["source_partition_id"] for item in partition_results],
            "accepted": [item["accepted"] for item in partition_results],
            "failed_gates": [",".join(item["failed_gates"]) for item in partition_results],
        }
    )
    lines = [
        f"# {title}",
        "",
        f"Status: **{'ACCEPTED' if result['accepted'] else 'REJECTED'}**",
        "",
        note,
        "",
        pd.DataFrame(result.get("gates", [])).to_markdown(index=False),
        "",
        "## Partition status",
        "",
        partition_summary.to_markdown(index=False),
        "",
    ]
    _atomic_write_text("\n".join(lines), gates_dir / "NATIONAL_ROUTE_GATE_REPORT.md")
    return result


def _synthetic_fixture_inputs(
    analysis_year: int, state: str, state_index: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return deterministic in-memory LODES, ACS, and county fixtures."""
    from build_route_national_flows import FIPS_TO_STATE

    state_fips = next(
        (fips for fips, abbreviation in FIPS_TO_STATE.items() if abbreviation == state),
        None,
    )
    if state_fips is None:
        raise ValueError(f"unsupported national state: {state}")
    home_county = f"{state_fips}001"
    pass_county = f"{state_fips}003"
    work_county = f"{state_fips}005"
    home_tract = f"{home_county}000100"
    work_tract = f"{work_county}000100"
    same_tract = f"{pass_county}000200"
    blocks = {
        "home": f"{home_tract}1001",
        "work": f"{work_tract}1001",
        "same_home": f"{same_tract}1001",
        "same_work": f"{same_tract}1002",
    }
    base_lon = -160.0 + state_index * 2.5
    latitude = 25.0 + (state_index % 6) * 4.0
    crosswalk = pd.DataFrame(
        {
            "tabblk2020": list(blocks.values()),
            "cty": [home_county, work_county, pass_county, pass_county],
            "trct": [home_tract, work_tract, same_tract, same_tract],
            "blklatdd": [latitude, latitude, latitude + 0.4, latitude + 0.4],
            "blklondd": [base_lon, base_lon + 1.5, base_lon + 0.05, base_lon + 0.25],
        }
    )
    flows = pd.DataFrame(
        {
            "h_geocode": [blocks["home"], blocks["same_home"]],
            "w_geocode": [blocks["work"], blocks["same_work"]],
            "S000": [10.0, 1.0],
            "file_type": ["main", "main"],
        }
    )
    car_share = pd.Series(
        [0.8, 0.5],
        index=pd.Index([home_tract, same_tract], name="tract"),
    )
    boundaries = pd.DataFrame(
        {
            "county_fips": [home_county, pass_county, work_county],
            "geometry": [
                {
                    "type": "Polygon",
                    "coordinates": [[[base_lon - 0.2, latitude - 0.2], [base_lon + 0.5, latitude - 0.2], [base_lon + 0.5, latitude + 0.7], [base_lon - 0.2, latitude + 0.7], [base_lon - 0.2, latitude - 0.2]]],
                },
                {
                    "type": "Polygon",
                    "coordinates": [[[base_lon + 0.5, latitude - 0.2], [base_lon + 1.0, latitude - 0.2], [base_lon + 1.0, latitude + 0.7], [base_lon + 0.5, latitude + 0.7], [base_lon + 0.5, latitude - 0.2]]],
                },
                {
                    "type": "Polygon",
                    "coordinates": [[[base_lon + 1.0, latitude - 0.2], [base_lon + 1.7, latitude - 0.2], [base_lon + 1.7, latitude + 0.7], [base_lon + 1.0, latitude + 0.7], [base_lon + 1.0, latitude - 0.2]]],
                },
            ],
        }
    )
    return flows, crosswalk, car_share, boundaries


def _synthetic_model_runner(
    panel: pd.DataFrame,
    *,
    scope: str,
    treatment_columns: Sequence[str],
    bootstrap_reps: int,
) -> pd.DataFrame:
    """Exercise Task 6's model-runner boundary without fitting tiny-fixture FEs."""
    return pd.DataFrame(
        {
            "term": list(treatment_columns),
            "estimate": [-0.01] * len(treatment_columns),
            "status": ["synthetic_dry_run"] * len(treatment_columns),
            "model_scope_received": [scope] * len(treatment_columns),
            "bootstrap_reps_received": [int(bootstrap_reps)] * len(treatment_columns),
            "model_rows_received": [len(panel)] * len(treatment_columns),
        }
    )


def _national_gate_metrics(
    audits: pd.DataFrame,
    segments: pd.DataFrame,
    model_panel: pd.DataFrame,
    *,
    all_partitions_available: bool,
    restart_reused_share: object,
    runtime_seconds: object,
    storage_bytes: int,
    alerted_origin_fips: set[str] | None = None,
    same_tract_evaluation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _validate_route_audits_for_exposure(segments, audits)
    weights = pd.to_numeric(audits["commuter_car_weight"], errors="coerce").fillna(0.0)
    eligible = audits["routing_eligible"].fillna(False).astype(bool)
    successful = audits["status"].astype(str).eq("Ok") & eligible
    selected_weight = float(weights.loc[eligible].sum())
    successful_weight = float(weights.loc[successful].sum())
    coverage = successful_weight / selected_weight if selected_weight > 0 else 0.0

    allocated = (
        segments.loc[segments["county_fips"].notna()]
        .groupby("route_signature")["route_miles_in_county"]
        .sum()
    )
    route_rows = audits.loc[audits["status"].astype(str).eq("Ok")].copy()
    route_rows["allocated_miles"] = route_rows["route_signature"].map(allocated).fillna(0.0)
    total = pd.to_numeric(route_rows["route_miles_total"], errors="coerce")
    conserved = (
        route_rows["allocated_miles"]
        + pd.to_numeric(route_rows["unallocated_miles"], errors="coerce").fillna(0.0)
    )
    relative = ((conserved - total).abs() / total.replace(0, np.nan)).fillna(0.0)
    total_sum = float(total.fillna(0.0).sum())
    aggregate_error = (
        abs(float(conserved.sum()) - total_sum) / total_sum if total_sum > 0 else 0.0
    )
    alerted_origins = alerted_origin_fips or set()
    audit_origins = audits.get("home_fips", pd.Series(index=audits.index, dtype=str)).astype(str)
    alerted = audit_origins.isin(alerted_origins) & eligible
    nonalerted = ~audit_origins.isin(alerted_origins) & eligible
    alerted_selected = float(weights.loc[alerted].sum())
    alerted_success = float(weights.loc[alerted & successful].sum())
    nonalerted_selected = float(weights.loc[nonalerted].sum())
    nonalerted_success = float(weights.loc[nonalerted & successful].sum())
    comparable = bool(
        np.isfinite(alerted_selected)
        and alerted_selected > 0
        and np.isfinite(nonalerted_selected)
        and nonalerted_selected > 0
    )
    alerted_share = alerted_success / alerted_selected if alerted_selected > 0 else None
    nonalerted_share = (
        nonalerted_success / nonalerted_selected if nonalerted_selected > 0 else None
    )
    required_accounting = {
        "omitted_coordinate_worker_weight",
        "omitted_car_share_worker_weight",
        "status",
        "error_message",
        "unallocated_miles",
    }
    denominator_columns = {"route_coverage_status", "total_commuter_car_miles"}
    positive_denominators = bool(
        not model_panel.empty
        and denominator_columns.issubset(model_panel.columns)
        and model_panel["route_coverage_status"].eq(ROUTE_COVERAGE_INCLUDED).all()
        and pd.to_numeric(model_panel["total_commuter_car_miles"], errors="coerce")
        .map(lambda value: bool(np.isfinite(value) and value > 0))
        .all()
    )
    comparison_columns = {"destination_dosage", "affected_route_share"}
    route_comparison_complete = bool(
        not model_panel.empty
        and comparison_columns.issubset(model_panel.columns)
        and model_panel[list(comparison_columns)].notna().all().all()
    )
    audit_weight_summary = reconcile_route_audit_weights(segments, audits)
    exposure_weight_columns = {
        "failed_route_commuter_car_weight",
        "unallocated_commuter_car_weight",
    }
    accounting_reconciled = exposure_weight_columns.issubset(model_panel.columns)
    for column in exposure_weight_columns:
        expected = float(audit_weight_summary[column])
        if column not in model_panel:
            continue
        observed = pd.to_numeric(model_panel[column], errors="coerce")
        accounting_reconciled &= bool(
            observed.notna().all()
            and np.isclose(observed.to_numpy(dtype=float), expected).all()
        )
    same_tract_weight = float(
        weights.loc[audits.get("same_tract", pd.Series(False, index=audits.index)).fillna(False).astype(bool)].sum()
    )
    model_panel_analysis_years = sorted(
        pd.to_numeric(
            model_panel.get("analysis_year", pd.Series(dtype=float)), errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return {
        "successful_commuter_car_share": coverage,
        "maximum_route_conservation_error": float(relative.max()) if not relative.empty else 0.0,
        "aggregate_conservation_error": aggregate_error,
        "positive_denominators": positive_denominators,
        "all_partitions_available": bool(all_partitions_available),
        "no_alerted_origin_bias": bool(
            comparable
            and abs(float(alerted_share) - float(nonalerted_share)) <= 0.01
        ),
        "alerted_origin_comparable": comparable,
        "same_tract_stable": bool(
            same_tract_evaluation is not None
            and same_tract_evaluation.get("stable") is True
        ),
        "omissions_and_failures_explicit": bool(
            required_accounting.issubset(audits.columns) and accounting_reconciled
        ),
        "route_comparison_complete": route_comparison_complete,
        "runtime_seconds": runtime_seconds,
        "storage_bytes": int(storage_bytes),
        "restart_reused_share": restart_reused_share,
        "selected_commuter_car_weight": selected_weight,
        "successful_commuter_car_weight": successful_weight,
        "alerted_origin_successful_share": alerted_share,
        "nonalerted_origin_successful_share": nonalerted_share,
        "same_tract_commuter_car_weight_share": same_tract_weight / selected_weight if selected_weight else None,
        "model_panel_analysis_years": model_panel_analysis_years,
        "failed_route_count": int((eligible & ~successful).sum()),
        "input_omission_count": int((~eligible).sum()),
        "unallocated_miles": float(pd.to_numeric(audits["unallocated_miles"], errors="coerce").fillna(0.0).sum()),
        "failed_route_commuter_car_weight": float(
            audit_weight_summary["failed_route_commuter_car_weight"]
        ),
        "unallocated_commuter_car_weight": float(
            audit_weight_summary["unallocated_commuter_car_weight"]
        ),
        "route_audit_weights_reconciled": bool(accounting_reconciled),
    }


def _partition_same_tract_evaluation(
    model_results: pd.DataFrame,
    manifest: pd.DataFrame,
    audits: pd.DataFrame,
    *,
    network_year: int | None,
) -> dict[str, object]:
    """Evaluate only three-mode estimates produced for one source partition."""
    partition_ids = {
        str(value).strip()
        for value in manifest.get("source_partition_id", pd.Series(dtype=str)).dropna()
        if str(value).strip()
    }
    if len(partition_ids) != 1 or "source_partition_ids" not in model_results:
        scoped = model_results.iloc[0:0].copy()
    else:
        partition_id = next(iter(partition_ids))
        scoped = model_results.loc[
            model_results["source_partition_ids"].astype(str).eq(partition_id)
        ].copy()
    weights = pd.to_numeric(
        audits.loc[
            audits["routing_eligible"].fillna(False), "commuter_car_weight"
        ],
        errors="coerce",
    ).fillna(0.0)
    same_weights = pd.to_numeric(
        audits.loc[
            audits["routing_eligible"].fillna(False)
            & audits["same_tract"].fillna(False),
            "commuter_car_weight",
        ],
        errors="coerce",
    ).fillna(0.0)
    same_share = (
        float(same_weights.sum() / weights.sum()) if weights.sum() > 0 else None
    )
    provenance = _same_tract_provenance(
        scoped,
        manifest,
        network_year=network_year,
        analysis_scope="partition",
    )
    return evaluate_same_tract_results(
        scoped,
        same_tract_commuter_car_weight_share=same_share,
        provenance=provenance,
        analysis_scope="partition",
    )


def _alerted_origins_for_years(
    alerts: pd.DataFrame, analysis_years: set[int]
) -> set[str]:
    """Return only origins whose alert date belongs to the evaluated partition."""
    date_column = "alert_date" if "alert_date" in alerts else "effective_crash_date"
    if date_column not in alerts or "home_fips" not in alerts:
        return set()
    dates = pd.to_datetime(alerts[date_column], errors="coerce")
    if dates.isna().any():
        return set()
    selected = alerts.loc[dates.dt.year.isin(analysis_years), "home_fips"]
    try:
        return set(_fips(selected))
    except ValueError:
        return set()


def _partition_gate_panel(
    rows: pd.DataFrame,
    segments: pd.DataFrame,
    alerts: pd.DataFrame,
    raw_panel: pd.DataFrame,
    destination: pd.DataFrame | None,
    *,
    same_tract_mode: str,
    network_year: int | None,
    route_audits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebuild the exact partition sample used by data-dependent gates."""
    if raw_panel.empty:
        return pd.DataFrame()
    year = int(rows["analysis_year"].iloc[0])
    exposure = _year_exposure(
        rows,
        segments,
        alerts,
        same_tract_mode=same_tract_mode,
        route_audits=route_audits,
    )
    if exposure.empty:
        return pd.DataFrame()
    panel = raw_panel.copy()
    if "outcome_fips" in panel and "fips" not in panel:
        panel = panel.rename(columns={"outcome_fips": "fips"})
    if "alert_date" in panel and "date" not in panel:
        panel = panel.rename(columns={"alert_date": "date"})
    if not {"fips", "date"}.issubset(panel):
        raise ValueError("analysis panel requires fips/outcome_fips and date/alert_date")
    panel["fips"] = _fips(panel["fips"])
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    outcome_fips = set(_fips(exposure["outcome_fips"]))
    panel = panel.loc[
        panel["date"].dt.year.eq(year) & panel["fips"].isin(outcome_fips)
    ].copy()
    if panel.empty:
        return pd.DataFrame()
    metadata = {
        "analysis_year": year,
        "analysis_years": str(year),
        "states": _manifest_state_label(rows),
        "network_year": network_year,
        "lodes_source_year": int(rows["lodes_source_year"].iloc[0]),
        "lodes_source_years": ",".join(
            map(str, sorted(rows["lodes_source_year"].unique()))
        ),
        "acs_car_share_vintage": str(rows["acs_car_share_vintage"].iloc[0]),
        "acs_car_share_vintages": ",".join(
            sorted(rows["acs_car_share_vintage"].astype(str).unique())
        ),
        "same_tract_mode": same_tract_mode,
        "source_manifest_ids": _joined_unique(segments, "source_manifest_id"),
        "source_partition_ids": (
            _joined_unique(rows, "source_partition_id")
            or _joined_unique(segments, "source_partition_id")
        ),
        "network_manifest_ids": _joined_unique(segments, "network_manifest_id"),
        "segment_partition_paths": ",".join(sorted(set(rows["segment_path"]))),
    }
    return _merge_destination_measure(
        _attach_exposure_to_panel(panel, exposure, metadata), destination
    )


def run_synthetic_national_dry_run(args: argparse.Namespace) -> dict[str, object]:
    """Run every national stage on deterministic fixtures without network I/O."""
    from build_route_national_flows import (
        LODES_CANDIDATE_YEARS,
        NATIONAL_STATES,
        build_flow_partition_from_chunks,
    )
    from build_route_national_segments import route_partition_to_segments
    from route_vintages import resolve_acs_window, resolve_nearest_year, write_vintage_manifest

    started = time.perf_counter()
    root = Path(args.output_dir)
    years = sorted({_integer(year, "analysis year") for year in args.analysis_years})
    states = sorted({str(state).strip().lower() for state in args.states})
    if not years:
        raise ValueError("analysis_years must not be empty")
    invalid_states = sorted(set(states) - set(NATIONAL_STATES))
    if not states or invalid_states:
        raise ValueError(f"states must be 50-state-plus-DC abbreviations; invalid={invalid_states}")
    if int(args.network_year) <= 0:
        raise ValueError("network_year must be positive")
    if int(args.chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")

    # The dry run may route only a small requested subset, but the network
    # artifact it exercises must still satisfy the production national-network
    # contract (all 50 states plus DC).  This keeps the dry run useful for
    # testing the same gate that will protect a full production analysis.
    network_states = sorted(NATIONAL_STATES)
    network_payload = {
        "schema_version": "route_national.network.synthetic.v1",
        "network_year": int(args.network_year),
        "states": network_states,
        "scope": "national",
        "profile": "car",
        "source": "deterministic synthetic dry-run fixture",
        "source_url": "synthetic://openstreetmap/common-network",
        "retrieved_at_utc": "synthetic-fixture",
    }
    network_checksum = hashlib.sha256(
        json.dumps(network_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    network_manifest_id = f"sha256:{network_checksum}"
    network_manifest = {
        **network_payload,
        "manifest_id": network_manifest_id,
        "manifest_payload_sha256": network_checksum,
    }
    network_manifest_path = root / "manifests" / "network_manifest.json"
    _atomic_write_json(network_manifest, network_manifest_path)
    vintage_rows: list[dict[str, object]] = []
    segment_manifest_rows: list[dict[str, object]] = []
    all_audits: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    alert_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    destination_rows: list[dict[str, object]] = []
    restart_reused: list[float] = []

    for year in years:
        available_lodes = [candidate for candidate in LODES_CANDIDATE_YEARS if abs(candidate - year) == 1]
        if not available_lodes:
            available_lodes = list(LODES_CANDIDATE_YEARS)
        lodes_choice = resolve_nearest_year(year, available_lodes)
        acs_choice = resolve_acs_window(year, [(year - 4, year, f"{year - 4}-{year}")])
        if acs_choice.vintage is None:
            raise RuntimeError("synthetic ACS vintage unexpectedly unavailable")
        for state_index, state in enumerate(states):
            flows, crosswalk, car_share, boundaries = _synthetic_fixture_inputs(
                year, state, state_index
            )
            chunks = (
                flows.iloc[start : start + int(args.chunk_rows)].copy()
                for start in range(0, len(flows), int(args.chunk_rows))
            )
            pairs, diagnostics = build_flow_partition_from_chunks(
                chunks,
                crosswalk,
                car_share,
                analysis_year=year,
                lodes_source_year=int(lodes_choice.source_year),
                work_state=state,
            )
            provenance_payload = (
                f"synthetic-lodes8-jt00|{year}|{lodes_choice.source_year}|{state}|"
                f"{acs_choice.vintage}"
            )
            provenance_sha256 = hashlib.sha256(provenance_payload.encode("utf-8")).hexdigest()
            lodes_source_id = f"synthetic-lodes8-jt00-{lodes_choice.source_year}-{state}"
            pairs["lodes_source_id"] = lodes_source_id
            pairs["provenance_sha256"] = provenance_sha256
            pairs["acs_car_share_vintage"] = acs_choice.vintage
            flow_path = (
                root / "flows" / "partitions" / f"analysis_year={year}"
                / f"lodes_source_year={lodes_choice.source_year}"
                / f"work_state={state}.parquet"
            )
            _atomic_write(pairs, flow_path)
            diagnostic_path = (
                root / "flows" / "diagnostics" / f"analysis_year={year}"
                / f"work_state={state}.csv"
            )
            _atomic_write(diagnostics, diagnostic_path)
            source_manifest_id = f"{lodes_source_id}|{provenance_sha256}"
            vintage_rows.append(
                {
                    "analysis_year": year,
                    "state": state,
                    "lodes_source_year": int(lodes_choice.source_year),
                    "lodes_year_gap": int(lodes_choice.gap),
                    "lodes_status": lodes_choice.status,
                    "acs_car_share_vintage": acs_choice.vintage,
                    "acs_window_start": acs_choice.window_start,
                    "acs_window_end": acs_choice.window_end,
                    "acs_status": acs_choice.status,
                    "selection_rule": "nearest source year/window; earlier year wins ties",
                    "source_url": f"synthetic://lodes8/JT00/{lodes_choice.source_year}/{state}",
                    "retrieved_at_utc": "synthetic-fixture",
                    "source_bytes": flow_path.stat().st_size,
                    "source_sha256": _file_sha256(flow_path),
                    "source_manifest_id": source_manifest_id,
                    "partition_status": "success",
                    "retained_worker_weight": float(pairs["workers"].sum()),
                    "omitted_worker_weight": float(
                        pairs[["omitted_coordinate_worker_weight", "omitted_car_share_worker_weight"]]
                        .sum().sum()
                    ),
                }
            )

            segment_dir = (
                root / "segments" / f"analysis_year={year}"
                / f"lodes_source_year={lodes_choice.source_year}"
                / f"work_state={state}"
            )
            calls = {"count": 0}

            def route_client(home_lon, home_lat, work_lon, work_lat, route_id, base_url):
                calls["count"] += 1
                distance_m = max(abs(float(work_lon) - float(home_lon)) * 100_000.0, 10_000.0)
                return {
                    "route_id": route_id,
                    "status": "Ok",
                    "distance_m": distance_m,
                    "duration_s": distance_m / 20.0,
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[home_lon, home_lat], [work_lon, work_lat]],
                    },
                    "error_message": None,
                }

            audits, segments = route_partition_to_segments(
                pairs,
                boundaries,
                base_url="synthetic://osrm",
                cache_dir=segment_dir,
                route_client=route_client,
                network_manifest_id=network_manifest_id,
                max_workers=int(args.route_workers),
                checkpoint_every=int(args.checkpoint_every),
                geometry_sample_rate=float(args.geometry_sample_rate),
            )
            calls_before_restart = calls["count"]
            route_partition_to_segments(
                pairs,
                boundaries,
                base_url="synthetic://osrm",
                cache_dir=segment_dir,
                route_client=route_client,
                network_manifest_id=network_manifest_id,
                max_workers=int(args.route_workers),
                checkpoint_every=int(args.checkpoint_every),
                geometry_sample_rate=float(args.geometry_sample_rate),
            )
            restart_calls = calls["count"] - calls_before_restart
            restart_reused.append(
                (len(audits) - restart_calls) / len(audits) if len(audits) else 0.0
            )
            segments["home_fips"] = segments["home_fips"].astype(str).str.zfill(5)
            audits = audits.merge(
                pairs[["route_id", "home_county", "same_tract"]],
                on="route_id",
                how="left",
                validate="one_to_one",
            ).rename(columns={"home_county": "home_fips"})
            all_audits.append(audits)
            all_segments.append(segments)
            segment_path = segment_dir / "county_segments.parquet"
            audit_path = segment_dir / "route_audits.parquet"
            source_partition_id = str(pairs["source_partition_id"].iloc[0])
            segment_manifest_rows.append(
                {
                    "schema_version": SEGMENT_SCHEMA_V1,
                    "analysis_year": year,
                    "state": state,
                    "lodes_source_year": int(lodes_choice.source_year),
                    "acs_car_share_vintage": acs_choice.vintage,
                    "source_manifest_id": source_manifest_id,
                    "network_manifest_id": network_manifest_id,
                    "source_partition_id": source_partition_id,
                    "flow_path": str(flow_path.resolve()),
                    "flow_sha256": _file_sha256(flow_path),
                    "segment_path": str(segment_path.resolve()),
                    "segment_sha256": _file_sha256(segment_path),
                    "audit_path": str(audit_path.resolve()),
                    "audit_sha256": _file_sha256(audit_path),
                    "status": "success",
                }
            )
            alert_date = pd.Timestamp(year=year, month=6, day=15)
            alert_rows.append(
                {
                    "home_fips": str(pairs.loc[~pairs["same_tract"], "home_county"].iloc[0]),
                    "alert_date": alert_date,
                    "geo_scope": "county_same",
                }
            )
            outcome_counties = sorted(set(segments["outcome_fips"].dropna().astype(str)))
            for county_index, county in enumerate(outcome_counties):
                panel_rows.append(
                    {"fips": county, "date": alert_date, "fatal_crashes": float(county_index % 2)}
                )
                destination_rows.append(
                    {"fips": county, "date": alert_date, "destination_dosage": float(county_index + 1) / 10.0}
                )

    manifest_path = root / "manifests" / "national_vintage_manifest.csv"
    write_vintage_manifest(vintage_rows, manifest_path)
    segment_manifest_path = root / "segments" / "segment_manifest.csv"
    segment_manifest = pd.DataFrame(segment_manifest_rows)
    _atomic_write(segment_manifest, segment_manifest_path)
    alerts = pd.DataFrame(alert_rows).drop_duplicates()
    panel = pd.DataFrame(panel_rows).drop_duplicates(["fips", "date"])
    destination = pd.DataFrame(destination_rows).drop_duplicates(["fips", "date"])
    analysis_dir = root / "analysis"
    same_tract_model_frames: list[pd.DataFrame] = []
    for mode in sorted(SAME_TRACT_MODES):
        mode_output_dir = (
            analysis_dir
            if mode == args.same_tract_mode
            else analysis_dir / "same_tract_modes" / f"mode={mode}"
        )
        mode_results = run_national_route_analysis(
            argparse.Namespace(
                segment_manifest=segment_manifest,
                alerts=alerts,
                panel=panel,
                destination_exposure=destination,
                analysis_years=years,
                states=states,
                network_year=int(args.network_year),
                same_tract_mode=mode,
                bootstrap_reps=1,
                model_runner=_synthetic_model_runner,
                output_dir=mode_output_dir,
            )
        )
        mode_results["same_tract_mode"] = mode
        same_tract_model_frames.append(mode_results)
    same_tract_model_results = pd.concat(same_tract_model_frames, ignore_index=True)
    _atomic_write(
        same_tract_model_results,
        analysis_dir / "same_tract_model_results.csv",
    )
    model_panel = pd.read_parquet(analysis_dir / "national_route_model_panel.parquet")
    audits = pd.concat(all_audits, ignore_index=True)
    segments = pd.concat(all_segments, ignore_index=True)
    from run_route_exposure_pilot import compare_destination_and_route_exposure

    route_comparison = compare_destination_and_route_exposure(
        model_panel.rename(columns={"fips": "outcome_fips", "date": "alert_date"}),
        destination.rename(columns={"fips": "outcome_fips", "date": "alert_date"}),
    )
    _atomic_write(route_comparison, analysis_dir / "route_vs_destination_comparison.csv")
    same_tract_rows: list[dict[str, object]] = []
    for mode in sorted(SAME_TRACT_MODES):
        mode_exposures = []
        for year, rows in segment_manifest.groupby("analysis_year", sort=True):
            year_segments = segments.loc[
                pd.to_numeric(segments["analysis_year"], errors="coerce").eq(int(year))
            ].copy()
            mode_exposures.append(
                _year_exposure(rows, year_segments, alerts, same_tract_mode=mode)
            )
        mode_frame = pd.concat(mode_exposures, ignore_index=True)
        denominator = pd.to_numeric(
            mode_frame["total_commuter_car_miles"], errors="coerce"
        )
        same_tract_rows.append(
            {
                "same_tract_mode": mode,
                "county_date_rows": len(mode_frame),
                "minimum_denominator": float(denominator.min()),
                "total_affected_commuter_car_miles": float(
                    pd.to_numeric(
                        mode_frame["affected_commuter_car_miles"], errors="coerce"
                    ).sum()
                ),
                "mean_affected_route_share": float(
                    pd.to_numeric(mode_frame["affected_route_share"], errors="coerce").mean()
                ),
            }
        )
    same_tract_summary = pd.DataFrame(same_tract_rows)
    _atomic_write(same_tract_summary, analysis_dir / "same_tract_mode_summary.csv")
    selected_weights = pd.to_numeric(
        audits.loc[audits["routing_eligible"].fillna(False), "commuter_car_weight"],
        errors="coerce",
    ).fillna(0.0)
    same_tract_weights = pd.to_numeric(
        audits.loc[
            audits["routing_eligible"].fillna(False)
            & audits["same_tract"].fillna(False),
            "commuter_car_weight",
        ],
        errors="coerce",
    ).fillna(0.0)
    same_tract_share = (
        float(same_tract_weights.sum() / selected_weights.sum())
        if selected_weights.sum() > 0
        else None
    )
    same_tract_evaluation = evaluate_same_tract_results(
        same_tract_model_results,
        same_tract_commuter_car_weight_share=same_tract_share,
    )
    runtime_seconds = time.perf_counter() - started
    storage_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    metrics = _national_gate_metrics(
        audits,
        segments,
        model_panel,
        all_partitions_available=bool(segment_manifest["status"].eq("success").all()),
        restart_reused_share=min(restart_reused) if restart_reused else 0.0,
        runtime_seconds=runtime_seconds,
        storage_bytes=storage_bytes,
        alerted_origin_fips=set(alerts["home_fips"].astype(str)),
        same_tract_evaluation=same_tract_evaluation,
    )
    result = evaluate_national_gates(metrics)
    partition_results: list[dict[str, object]] = []
    partition_gate_rows: list[dict[str, object]] = []
    for source_partition_id, partition_audits in audits.groupby(
        "source_partition_id", sort=True
    ):
        partition_segments = segments.loc[
            segments["source_partition_id"].astype(str).eq(str(source_partition_id))
        ]
        partition_manifest = segment_manifest.loc[
            segment_manifest["source_partition_id"].astype(str).eq(str(source_partition_id))
        ].copy()
        manifest_row = partition_manifest.iloc[0]
        partition_years = set(
            pd.to_numeric(partition_manifest["analysis_year"]).astype(int)
        )
        partition_storage = sum(
            Path(manifest_row[column]).stat().st_size
            for column in ("segment_path", "audit_path")
        )
        partition_panel = _partition_gate_panel(
            partition_manifest,
            partition_segments,
            alerts,
            panel,
            destination,
            same_tract_mode=args.same_tract_mode,
            network_year=int(args.network_year),
            route_audits=partition_audits,
        )
        partition_same_evaluation = _partition_same_tract_evaluation(
            same_tract_model_results,
            partition_manifest,
            partition_audits,
            network_year=int(args.network_year),
        )
        partition_metrics = _national_gate_metrics(
            partition_audits,
            partition_segments,
            partition_panel,
            all_partitions_available=str(manifest_row["status"]).lower() == "success",
            restart_reused_share=min(restart_reused) if restart_reused else 0.0,
            runtime_seconds=runtime_seconds / max(len(segment_manifest), 1),
            storage_bytes=partition_storage,
            alerted_origin_fips=_alerted_origins_for_years(alerts, partition_years),
            same_tract_evaluation=partition_same_evaluation,
        )
        partition_result = evaluate_national_gates(partition_metrics)
        partition_result["source_partition_id"] = str(source_partition_id)
        partition_result["same_tract_evaluation"] = partition_same_evaluation
        partition_results.append(partition_result)
        partition_gate_rows.extend(
            {"source_partition_id": str(source_partition_id), **row}
            for row in partition_result["gates"]
        )
    result["partition_results"] = partition_results
    failed_partitions = [
        item["source_partition_id"]
        for item in partition_results
        if not item["accepted"]
    ]
    if failed_partitions:
        result["accepted"] = False
        result["failed_gates"] = [
            *result["failed_gates"],
            *[f"partition:{partition}" for partition in failed_partitions],
        ]
    gates_dir = root / "gates"
    result["paths"] = {
        "vintage_manifest": str(manifest_path),
        "network_manifest": str(network_manifest_path),
        "segment_manifest": str(segment_manifest_path),
        "gate_report": str(gates_dir / "national_gate_report.json"),
        "gate_table": str(gates_dir / "national_gate_table.csv"),
        "partition_gate_table": str(gates_dir / "national_partition_gate_table.csv"),
        "model_panel": str(analysis_dir / "national_route_model_panel.parquet"),
    }
    _atomic_write_json(result, gates_dir / "national_gate_report.json")
    _atomic_write(pd.DataFrame(result["gates"]), gates_dir / "national_gate_table.csv")
    _atomic_write(
        pd.DataFrame(partition_gate_rows),
        gates_dir / "national_partition_gate_table.csv",
    )
    report_lines = [
        "# Synthetic national route-exposure gate report",
        "",
        f"Status: **{'ACCEPTED' if result['accepted'] else 'REJECTED'}**",
        "",
        "This status applies only to the deterministic synthetic dry run; it is not a real-data result.",
        "",
        pd.DataFrame(result["gates"]).to_markdown(index=False),
        "",
        "## Partition status",
        "",
        pd.DataFrame(
            {
                "source_partition_id": [item["source_partition_id"] for item in partition_results],
                "accepted": [item["accepted"] for item in partition_results],
                "failed_gates": [",".join(item["failed_gates"]) for item in partition_results],
            }
        ).to_markdown(index=False),
        "",
    ]
    report_path = gates_dir / "NATIONAL_ROUTE_GATE_REPORT.md"
    _atomic_write_text("\n".join(report_lines), report_path)
    return result


def _load_production_gate_artifacts(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if "audit_path" not in manifest or "audit_sha256" not in manifest:
        raise ValueError("segment manifest requires audit_path and audit_sha256 for gates")
    audits: list[pd.DataFrame] = []
    segments: list[pd.DataFrame] = []
    storage_bytes = 0
    for _, row in manifest.iterrows():
        segment_path = Path(str(row["segment_path"])).expanduser().resolve()
        audit_path = Path(str(row["audit_path"])).expanduser().resolve()
        for path, checksum_column in (
            (segment_path, "segment_sha256"),
            (audit_path, "audit_sha256"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing gate artifact: {path}")
            expected = _nonblank(row.get(checksum_column), f"manifest {checksum_column}")
            if _file_sha256(path) != expected:
                raise ValueError(f"gate artifact checksum mismatch: {path}")
            storage_bytes += path.stat().st_size
        segment = _read_table(segment_path, "county-segment partition")
        _validate_segment_partition(
            segment,
            row,
            path=segment_path,
            checksum=_file_sha256(segment_path),
        )
        audit = _read_table(audit_path, "route-audit partition")
        for column, wanted in (
            ("source_partition_id", row["source_partition_id"]),
            ("source_manifest_id", row["source_manifest_id"]),
            ("network_manifest_id", row["network_manifest_id"]),
        ):
            actual = _unique_partition_value(audit, column, require_complete=True)
            if str(actual).strip() != str(wanted).strip():
                raise ValueError(f"route-audit provenance mismatch for {column}: {audit_path}")
        route_origin = segment[["route_id", "home_fips", "same_tract"]].drop_duplicates()
        if "flow_path" in row.index and pd.notna(row.get("flow_path")):
            flow_path = Path(str(row["flow_path"])).expanduser().resolve()
            if not flow_path.is_file():
                raise FileNotFoundError(f"missing flow artifact for audit reconciliation: {flow_path}")
            expected_flow = _nonblank(row.get("flow_sha256"), "manifest flow_sha256")
            if _file_sha256(flow_path) != expected_flow:
                raise ValueError(f"flow artifact checksum mismatch: {flow_path}")
            storage_bytes += flow_path.stat().st_size
            flow_origin = _read_table(flow_path, "flow partition")[
                ["route_id", "home_county", "same_tract"]
            ].rename(columns={"home_county": "home_fips"})
            route_origin = pd.concat([route_origin, flow_origin], ignore_index=True).drop_duplicates()
        if route_origin["route_id"].duplicated().any():
            raise ValueError(f"segment route origins are inconsistent: {segment_path}")
        audit = audit.merge(
            route_origin,
            on="route_id",
            how="left",
            validate="one_to_one",
        )
        if audit[["home_fips", "same_tract"]].isna().any().any():
            raise ValueError(f"route audit cannot be reconciled to segment origins: {audit_path}")
        audits.append(audit)
        segments.append(segment)
    if not audits or not segments:
        raise ValueError("no successful route partitions available for gate evaluation")
    return pd.concat(audits, ignore_index=True), pd.concat(segments, ignore_index=True), storage_bytes


def _network_manifest_valid(
    path: Path | None,
    *,
    network_year: int | None,
    segments: pd.DataFrame,
) -> tuple[bool, dict[str, object]]:
    if path is None or network_year is None or not Path(path).is_file():
        return False, {"error": "network manifest and network year are required"}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest_id = _nonblank(payload.get("manifest_id"), "network manifest_id")
        actual_year = _integer(payload.get("network_year"), "network manifest year")
        from build_route_national_network import NATIONAL_STATES

        required_states = {str(state).strip().lower() for state in NATIONAL_STATES}
        raw_states = payload.get("states")
        if not isinstance(raw_states, (list, tuple)):
            raise ValueError("network manifest states must be a list")
        manifest_states = [str(state).strip().lower() for state in raw_states]
        manifest_state_set = set(manifest_states)
        scope = str(payload.get("scope", "")).strip().lower()
        missing_states = sorted(required_states - manifest_state_set)
        extra_states = sorted(manifest_state_set - required_states)
        state_coverage_valid = (
            len(manifest_states) == len(required_states)
            and not missing_states
            and not extra_states
        )
        segment_ids = {
            str(value).strip() for value in segments["network_manifest_id"].dropna()
        }
        valid = bool(
            actual_year == int(network_year)
            and segment_ids == {manifest_id}
            and scope == "national"
            and state_coverage_valid
        )
        return valid, {
            "path": str(Path(path)),
            "manifest_id": manifest_id,
            "network_year": actual_year,
            "scope": scope or None,
            "state_count": len(manifest_states),
            "required_state_count": len(required_states),
            "missing_states": missing_states,
            "extra_states": extra_states,
            "state_coverage_valid": state_coverage_valid,
            "segment_network_manifest_ids": sorted(segment_ids),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, {"path": str(path), "error": str(exc)}


def build_production_gate_report(
    args: argparse.Namespace,
    *,
    partition_validation: Mapping[str, object],
) -> dict[str, object]:
    """Reconcile production artifacts and write fail-closed pooled/partition gates."""
    manifest = _manifest_frame(args.segment_manifest)
    requested_years = {_integer(year, "analysis year") for year in args.analysis_years}
    requested_states = {str(state).strip().lower() for state in args.states}
    state_column = "state" if "state" in manifest else "work_state"
    manifest = manifest.loc[
        manifest["analysis_year"].isin(requested_years)
        & manifest[state_column].astype(str).str.lower().isin(requested_states)
    ].copy()
    errors: list[str] = []
    try:
        audits, segments, storage_bytes = _load_production_gate_artifacts(manifest)
    except (OSError, TypeError, ValueError) as exc:
        audits, segments, storage_bytes = pd.DataFrame(), pd.DataFrame(), 0
        errors.append(str(exc))
    try:
        run_metrics = json.loads(Path(args.run_metrics).read_text(encoding="utf-8"))
        runtime_seconds = run_metrics.get("runtime_seconds")
        restart_reused_share = run_metrics.get("restart_reused_share")
    except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        run_metrics = {}
        runtime_seconds = None
        restart_reused_share = None
        errors.append(f"run metrics unavailable: {exc}")
    try:
        same_tract_results = _read_table(args.same_tract_results, "same-tract model results")
    except (AttributeError, OSError, ValueError) as exc:
        same_tract_results = pd.DataFrame()
        errors.append(f"same-tract model results unavailable: {exc}")
    model_panel_path = Path(args.output_dir) / "national_route_model_panel.parquet"
    try:
        model_panel = _read_table(model_panel_path, "national model panel")
    except (OSError, ValueError) as exc:
        model_panel = pd.DataFrame()
        errors.append(str(exc))
    try:
        raw_panel = _read_table(args.panel, "analysis panel")
    except (AttributeError, OSError, ValueError) as exc:
        raw_panel = pd.DataFrame()
        errors.append(f"analysis panel unavailable for partition gates: {exc}")
    destination_value = getattr(args, "destination_exposure", None)
    try:
        destination = (
            None
            if destination_value is None
            else _read_table(destination_value, "destination exposure")
        )
    except (OSError, ValueError) as exc:
        destination = None
        errors.append(f"destination exposure unavailable for partition gates: {exc}")
    alerts_value = getattr(args, "alerts", None)
    alerts = load_reviewed_route_alerts() if alerts_value is None else _read_table(alerts_value, "alerts")
    alerted_origins = _alerted_origins_for_years(alerts, requested_years)
    network_valid, network_details = _network_manifest_valid(
        getattr(args, "network_manifest", None),
        network_year=getattr(args, "network_year", None),
        segments=segments,
    ) if not segments.empty else (False, {"error": "segments unavailable"})
    selected_weights = (
        pd.to_numeric(
            audits.loc[audits["routing_eligible"].fillna(False), "commuter_car_weight"],
            errors="coerce",
        ).fillna(0.0)
        if not audits.empty
        else pd.Series(dtype=float)
    )
    same_weights = (
        pd.to_numeric(
            audits.loc[
                audits["routing_eligible"].fillna(False)
                & audits["same_tract"].fillna(False),
                "commuter_car_weight",
            ],
            errors="coerce",
        ).fillna(0.0)
        if not audits.empty
        else pd.Series(dtype=float)
    )
    same_share = (
        float(same_weights.sum() / selected_weights.sum())
        if selected_weights.sum() > 0
        else None
    )
    same_provenance = _same_tract_provenance(
        same_tract_results,
        manifest,
        network_year=getattr(args, "network_year", None),
    )
    same_evaluation = evaluate_same_tract_results(
        same_tract_results,
        same_tract_commuter_car_weight_share=same_share,
        provenance=same_provenance,
    )
    if audits.empty or segments.empty or model_panel.empty:
        result = evaluate_national_gates(
            {
                "all_partitions_available": bool(partition_validation.get("complete"))
                and network_valid,
                "runtime_seconds": runtime_seconds,
                "storage_bytes": storage_bytes,
                "restart_reused_share": restart_reused_share,
                "same_tract_stable": same_evaluation["stable"],
            }
        )
        result["partition_results"] = []
    else:
        metrics = _national_gate_metrics(
            audits,
            segments,
            model_panel,
            all_partitions_available=bool(partition_validation.get("complete"))
            and network_valid,
            restart_reused_share=restart_reused_share,
            runtime_seconds=runtime_seconds,
            storage_bytes=storage_bytes,
            alerted_origin_fips=alerted_origins,
            same_tract_evaluation=same_evaluation,
        )
        result = evaluate_national_gates(metrics)
        partition_results = []
        for partition_id, partition_audits in audits.groupby("source_partition_id", sort=True):
            partition_segments = segments.loc[
                segments["source_partition_id"].astype(str).eq(str(partition_id))
            ]
            partition_manifest = manifest.loc[
                manifest["source_partition_id"].astype(str).eq(str(partition_id))
            ]
            partition_years = set(
                pd.to_numeric(partition_manifest["analysis_year"]).astype(int)
            )
            try:
                partition_panel = _partition_gate_panel(
                    partition_manifest,
                    partition_segments,
                    alerts,
                    raw_panel,
                    destination,
                    same_tract_mode=getattr(
                        args, "same_tract_mode", "primary_calibrated"
                    ),
                    network_year=getattr(args, "network_year", None),
                    route_audits=partition_audits,
                )
            except (KeyError, TypeError, ValueError) as exc:
                partition_panel = pd.DataFrame()
                errors.append(f"partition {partition_id} gate panel unavailable: {exc}")
            partition_same_evaluation = _partition_same_tract_evaluation(
                same_tract_results,
                partition_manifest,
                partition_audits,
                network_year=getattr(args, "network_year", None),
            )
            partition_metrics = _national_gate_metrics(
                partition_audits,
                partition_segments,
                partition_panel,
                all_partitions_available=True,
                restart_reused_share=restart_reused_share,
                runtime_seconds=runtime_seconds,
                storage_bytes=storage_bytes,
                alerted_origin_fips=_alerted_origins_for_years(
                    alerts, partition_years
                ),
                same_tract_evaluation=partition_same_evaluation,
            )
            partition_result = evaluate_national_gates(partition_metrics)
            partition_result["source_partition_id"] = str(partition_id)
            partition_result["same_tract_evaluation"] = partition_same_evaluation
            partition_results.append(partition_result)
        result["partition_results"] = partition_results
        failed_partitions = [
            item["source_partition_id"] for item in partition_results if not item["accepted"]
        ]
        if failed_partitions:
            result["accepted"] = False
            result["failed_gates"] = [
                *result["failed_gates"],
                *[f"partition:{partition}" for partition in failed_partitions],
            ]
    result["partition_validation"] = dict(partition_validation)
    result["same_tract_evaluation"] = same_evaluation
    result["network_manifest"] = network_details
    result["run_metrics"] = run_metrics
    result["invocation"] = _invocation_metadata(args)
    result["errors"] = errors
    return _write_national_gate_report(
        result,
        args.output_dir,
        title="National route-exposure gate report",
        note="Acceptance requires every requested state-year partition and every pooled and partition gate.",
    )


def run_national_route_analysis(args: argparse.Namespace) -> pd.DataFrame:
    """Load partitioned exposures and run the established model specifications."""
    manifest = _manifest_frame(args.segment_manifest)
    requested = [_integer(year, "analysis year") for year in args.analysis_years]
    requested_states = getattr(args, "states", None)
    if requested_states:
        partition_validation = validate_requested_partitions(
            manifest,
            analysis_years=requested,
            states=requested_states,
        )
        if not partition_validation["complete"]:
            raise ValueError(
                "segment manifest does not cover the requested state-by-year grid: "
                f"{partition_validation}"
            )
    selected = manifest["analysis_year"].isin(requested)
    if requested_states:
        state_column = "state" if "state" in manifest else "work_state"
        states = {str(state).strip().lower() for state in requested_states}
        selected &= manifest[state_column].astype(str).str.strip().str.lower().isin(states)
    manifest = manifest.loc[selected].copy()
    missing_years = sorted(set(requested) - set(manifest["analysis_year"]))
    if missing_years:
        raise ValueError(f"segment manifest missing analysis years: {missing_years}")
    if "status" in manifest:
        successful = manifest["status"].astype(str).str.lower().isin({"success", "ok", "complete"})
        if not successful.all():
            failed_years = sorted(manifest.loc[~successful, "analysis_year"].unique())
            raise ValueError(f"segment manifest contains unavailable partitions for years: {failed_years}")
    same_tract_mode = getattr(args, "same_tract_mode", "primary_calibrated")
    if same_tract_mode not in SAME_TRACT_MODES:
        raise ValueError(f"unsupported same-tract mode: {same_tract_mode}")

    alerts_value = getattr(args, "alerts", None)
    alerts = load_reviewed_route_alerts() if alerts_value is None else _read_table(alerts_value, "alerts")
    panel = _read_table(args.panel, "analysis panel")
    if "alert_date" in panel and "date" not in panel:
        panel = panel.rename(columns={"alert_date": "date"})
    if "date" not in panel:
        raise ValueError("analysis panel requires date or alert_date")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    destination_value = getattr(args, "destination_exposure", None)
    destination = None if destination_value is None else _read_table(destination_value, "destination exposure")
    cache: dict[str, tuple[pd.DataFrame, str]] = {}
    audit_cache: dict[str, tuple[pd.DataFrame, str]] = {}
    exposure_frames: list[pd.DataFrame] = []
    panel_frames: list[pd.DataFrame] = []
    partition_panel_frames: list[tuple[str, pd.DataFrame]] = []

    for year, rows in manifest.groupby("analysis_year", sort=True):
        segments = _combine_year_segments(rows, cache)
        route_audits = _combine_year_audits(rows, audit_cache)
        exposure = _year_exposure(
            rows,
            segments,
            alerts,
            same_tract_mode=same_tract_mode,
            route_audits=route_audits,
        )
        exposure_frames.append(exposure)
        source_years = sorted(rows["lodes_source_year"].unique())
        vintages = sorted(rows["acs_car_share_vintage"].unique())
        metadata = {
            "analysis_year": int(year),
            "analysis_years": str(int(year)),
            "states": _manifest_state_label(rows),
            "network_year": getattr(args, "network_year", None),
            "lodes_source_year": source_years[0] if len(source_years) == 1 else pd.NA,
            "lodes_source_years": ",".join(map(str, source_years)),
            "acs_car_share_vintage": vintages[0] if len(vintages) == 1 else "mixed",
            "acs_car_share_vintages": ",".join(vintages),
            "same_tract_mode": same_tract_mode,
            "source_manifest_ids": _joined_unique(segments, "source_manifest_id"),
            "source_partition_ids": (
                _joined_unique(rows, "source_partition_id")
                or _joined_unique(segments, "source_partition_id")
            ),
            "network_manifest_ids": _joined_unique(segments, "network_manifest_id"),
            "segment_partition_paths": ",".join(sorted(set(rows["segment_path"]))),
        }
        year_panel = panel.loc[panel["date"].dt.year.eq(int(year))]
        panel_frames.append(_attach_exposure_to_panel(year_panel, exposure, metadata))
        if "source_partition_id" in rows:
            for partition_id, partition_rows in rows.groupby(
                "source_partition_id", sort=True
            ):
                partition_segments = _combine_year_segments(partition_rows, cache)
                partition_audits = _combine_year_audits(partition_rows, audit_cache)
                partition_panel_frames.append(
                    (
                        str(partition_id),
                        _partition_gate_panel(
                            partition_rows,
                            partition_segments,
                            alerts,
                            panel,
                            destination,
                            same_tract_mode=same_tract_mode,
                            network_year=getattr(args, "network_year", None),
                            route_audits=partition_audits,
                        ),
                    )
                )

    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    model_panel = pd.concat(panel_frames, ignore_index=True)
    model_panel = _merge_destination_measure(model_panel, destination)
    estimation_panel = model_panel.loc[
        model_panel["route_coverage_status"].eq(ROUTE_COVERAGE_INCLUDED)
    ].copy()
    if estimation_panel.empty:
        raise ValueError("no analysis rows have a positive route-exposure denominator")
    runner = getattr(args, "model_runner", None) or _established_model_runner
    bootstrap_reps = int(getattr(args, "bootstrap_reps", 9_999))
    results: list[pd.DataFrame] = []

    for year, subset in estimation_panel.groupby("analysis_year", sort=True):
        results.append(_run_models(
            runner, subset, scope="year", bootstrap_reps=bootstrap_reps,
            labels={
                **_provenance_labels(subset),
                "analysis_year": int(year),
            },
        ))
    vintage_columns = ["lodes_source_years", "acs_car_share_vintages"]
    for keys, subset in estimation_panel.groupby(vintage_columns, sort=True, dropna=False):
        results.append(_run_models(
            runner, subset, scope="vintage", bootstrap_reps=bootstrap_reps,
            labels={
                **_provenance_labels(subset),
                "analysis_year": ",".join(map(str, sorted(subset["analysis_year"].unique()))),
                "lodes_source_years": keys[0],
                "acs_car_share_vintages": keys[1],
            },
        ))
    for partition_id, partition_panel in partition_panel_frames:
        partition_sample = partition_panel.loc[
            partition_panel["route_coverage_status"].eq(ROUTE_COVERAGE_INCLUDED)
        ].copy()
        if partition_sample.empty:
            continue
        results.append(_run_models(
            runner,
            partition_sample,
            scope="partition",
            bootstrap_reps=bootstrap_reps,
            labels={
                **_provenance_labels(partition_sample),
                "analysis_year": int(partition_sample["analysis_year"].iloc[0]),
                "source_partition_id": partition_id,
            },
        ))
    results.append(_run_models(
        runner, estimation_panel, scope="pooled", bootstrap_reps=bootstrap_reps,
        labels={
            **_provenance_labels(estimation_panel),
            "analysis_year": "all",
            "lodes_source_years": ",".join(map(str, sorted(manifest["lodes_source_year"].unique()))),
            "acs_car_share_vintages": ",".join(sorted(manifest["acs_car_share_vintage"].unique())),
        },
    ))
    output = pd.concat(results, ignore_index=True)

    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        directory = Path(output_dir)
        _atomic_write(exposures, directory / "national_route_county_date_exposures.parquet")
        _atomic_write(model_panel, directory / "national_route_model_panel.parquet")
        _atomic_write(output, directory / "national_route_model_results.csv")
        for scope in ("year", "vintage", "partition", "pooled"):
            _atomic_write(
                output.loc[output["analysis_scope"].eq(scope)].copy(),
                directory / f"national_route_model_results_{scope}.csv",
            )
    return output


def _parser() -> argparse.ArgumentParser:
    from build_route_national_flows import NATIONAL_STATES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--alerts", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--destination-exposure", type=Path)
    parser.add_argument("--analysis-years", type=int, nargs="+", default=list(range(2013, 2025)))
    parser.add_argument("--states", nargs="+", default=list(NATIONAL_STATES))
    parser.add_argument("--network-year", type=int)
    parser.add_argument("--network-manifest", type=Path)
    parser.add_argument("--run-metrics", type=Path)
    parser.add_argument("--same-tract-results", type=Path)
    parser.add_argument("--same-tract-mode", choices=sorted(SAME_TRACT_MODES), default="primary_calibrated")
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--route-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--geometry-sample-rate", type=float, default=0.001)
    parser.add_argument("--dry-run-fixture", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=9_999)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_TABS / "route_national")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run_fixture:
        if args.states is None:
            raise ValueError("--dry-run-fixture requires --states")
        if args.network_year is None:
            raise ValueError("--dry-run-fixture requires --network-year")
        result = run_synthetic_national_dry_run(args)
        status = "ACCEPTED" if result["accepted"] else "REJECTED"
        print(f"Synthetic national gate status: {status}")
        for label, path in result["paths"].items():
            print(f"{label}: {path}")
        return 0 if result["accepted"] else 2
    if args.panel is None:
        raise ValueError("--panel is required unless --dry-run-fixture is used")
    manifest = _manifest_frame(args.segment_manifest)
    partition_validation = validate_requested_partitions(
        manifest,
        analysis_years=args.analysis_years,
        states=args.states,
    )
    if not partition_validation["complete"]:
        result = evaluate_national_gates({"all_partitions_available": False})
        result["partition_validation"] = partition_validation
        result["partition_results"] = []
        result["invocation"] = _invocation_metadata(args)
        _write_national_gate_report(
            result,
            args.output_dir,
            title="National route-exposure gate report",
            note="The requested state-by-year grid is incomplete; modeling was not run.",
        )
        print("National gate status: REJECTED")
        return 2
    run_national_route_analysis(args)
    result = build_production_gate_report(
        args,
        partition_validation=partition_validation,
    )
    status = "ACCEPTED" if result["accepted"] else "REJECTED"
    print(f"National gate status: {status}")
    for label, path in result["paths"].items():
        print(f"{label}: {path}")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
