"""Build, diagnose, and gate the Wisconsin route-exposure pilot."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from config import ROOT, ROUTE_PILOT_CACHE
from load_amber_missing_alerts import load_combined_alerts

LABEL = "route_exposure_2022"
SAME_TRACT_MODES = {"primary_calibrated", "zero", "exclude"}
SUCCESS_STATUSES = {"ok", "success", "routed", "sametractimputed", "sametractzero"}
AUDIT_SEGMENT_TYPES = {"failed_route", "failedroute", "unallocated"}


def _codes(values: pd.Series, *, allow_missing: bool = False) -> pd.Series:
    missing = values.isna()
    if missing.any() and not allow_missing:
        raise ValueError("missing FIPS identifier")
    out = values.astype("string").str.strip().str.removesuffix(".0").str.zfill(5)
    if ((~missing) & (~out.str.fullmatch(r"\d{5}", na=False))).any():
        raise ValueError("invalid FIPS identifier")
    return out.mask(missing, pd.NA)


def _require(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    out = frame[columns].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any() or not np.isfinite(out.to_numpy()).all() or (out < 0).any().any():
        raise ValueError(f"{label} has nonfinite or negative fields")
    return out


def _route_audit_weight(segments: pd.DataFrame, segment_types: set[str]) -> float:
    route = segments.copy()
    route["_segment_type"] = route.get("segment_type", pd.Series("county", index=route.index)).astype(str).str.lower()
    route = route.loc[route["_segment_type"].isin(segment_types)].sort_values("route_id").drop_duplicates("route_id")
    return float((route["workers"] * route["home_car_share"]).sum()) if not route.empty else 0.0


def build_alert_date_exposures(
    county_segments: pd.DataFrame,
    alerts: pd.DataFrame,
    same_tract_mode: str,
    *,
    label: str = LABEL,
    vintage_columns: Mapping[str, object] | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Construct county-date dosage while retaining failed/unallocated audit weight."""
    if same_tract_mode not in SAME_TRACT_MODES:
        raise ValueError("same_tract_mode must be primary_calibrated, zero, or exclude")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("exposure label must be a nonblank string")
    label = label.strip()
    _require(county_segments, ["route_id", "outcome_fips", "home_fips", "work_fips", "workers", "home_car_share", "route_miles_in_county"], "county_segments")
    if "home_fips" not in alerts.columns and "fips" in alerts.columns:
        alerts = alerts.rename(columns={"fips": "home_fips"})
    if "alert_date" not in alerts.columns and "effective_crash_date" in alerts.columns:
        alerts = alerts.rename(columns={"effective_crash_date": "alert_date"})
    _require(alerts, ["home_fips", "alert_date", "geo_scope"], "alerts")

    seg = county_segments.copy()
    seg["_segment_type"] = seg.get("segment_type", pd.Series("county", index=seg.index)).astype(str).str.lower()
    missing_outcome = seg["outcome_fips"].isna()
    if (missing_outcome & ~seg["_segment_type"].isin(AUDIT_SEGMENT_TYPES)).any():
        raise ValueError("missing outcome FIPS is permitted only for failed or unallocated records")
    seg["outcome_fips"] = _codes(seg["outcome_fips"], allow_missing=True)
    seg["home_fips"] = _codes(seg["home_fips"])
    seg["work_fips"] = _codes(seg["work_fips"])
    numeric = _numeric(seg, ["workers", "route_miles_in_county"], "county_segments")
    seg[numeric.columns] = numeric
    seg["home_car_share"] = pd.to_numeric(seg["home_car_share"], errors="coerce")
    allocated_mask = seg["outcome_fips"].notna() & ~seg["_segment_type"].isin(AUDIT_SEGMENT_TYPES)
    invalid_share = seg["home_car_share"].notna() & ~seg["home_car_share"].between(0, 1)
    if invalid_share.any() or seg.loc[allocated_mask, "home_car_share"].isna().any():
        raise ValueError("county_segments home_car_share must be within [0, 1]")
    seg["commuter_car_miles"] = seg["workers"] * seg["home_car_share"] * seg["route_miles_in_county"]
    if not np.isfinite(seg.loc[allocated_mask, "commuter_car_miles"]).all():
        raise ValueError("county_segments commuter-car miles overflowed")

    failed_weight = _route_audit_weight(seg, {"failed_route", "failedroute"})
    unallocated_weight = _route_audit_weight(seg, {"unallocated"})
    def omitted_workers(column: str) -> float:
        if column not in seg:
            return 0.0
        route = seg[["route_id", column]].copy()
        route[column] = pd.to_numeric(route[column], errors="coerce").fillna(0.0)
        return float(route.groupby("route_id")[column].max().sum())
    omitted_coordinate_workers = omitted_workers("omitted_coordinate_worker_weight")
    omitted_car_share_workers = omitted_workers("omitted_car_share_worker_weight")
    metadata: dict[str, object] = {}
    if vintage_columns is not None:
        if isinstance(vintage_columns, Mapping):
            metadata = {str(column): value for column, value in vintage_columns.items()}
        else:
            for column in vintage_columns:
                if column not in seg.columns:
                    raise ValueError(f"county_segments missing vintage column: {column}")
                values = seg[column].dropna().drop_duplicates()
                if len(values) != 1:
                    raise ValueError(
                        f"county_segments vintage column {column} must have one nonmissing value"
                    )
                metadata[str(column)] = values.iloc[0]
        invalid_metadata = [column for column in metadata if not column.strip() or column == label]
        if invalid_metadata:
            raise ValueError(f"invalid vintage columns: {invalid_metadata}")
    allocated = seg.loc[allocated_mask].copy()
    totals = allocated.groupby("outcome_fips", as_index=False)["commuter_car_miles"].sum().rename(columns={"commuter_car_miles": "total_commuter_car_miles"})
    if totals.empty or (totals["total_commuter_car_miles"] <= 0).any():
        raise ValueError("zero denominator")

    a = alerts.copy()
    a["home_fips"] = _codes(a["home_fips"])
    if a["geo_scope"].isna().any():
        raise ValueError("alerts has missing geo_scope")
    invalid_scope = sorted(set(a["geo_scope"].astype(str)) - {"county_same", "statewide_same"})
    if invalid_scope:
        raise ValueError(f"unknown geo_scope: {invalid_scope}")
    a["alert_date"] = pd.to_datetime(a["alert_date"], errors="coerce").dt.normalize()
    if a["alert_date"].isna().any():
        raise ValueError("alerts has invalid alert_date")
    if a.empty:
        return pd.DataFrame(
            columns=["outcome_fips", "alert_date", label, *metadata]
        )

    dates = a[["alert_date"]].drop_duplicates()
    grid = totals[["outcome_fips"]].assign(_key=1).merge(dates.assign(_key=1), on="_key").drop(columns="_key")
    allocated = allocated.merge(totals, on="outcome_fips", how="left", validate="many_to_one")
    allocated["origin_class"] = np.select(
        [allocated["home_fips"].eq(allocated["outcome_fips"]), allocated["work_fips"].eq(allocated["outcome_fips"])],
        ["own", "cross"], default="pass_through"
    )
    network_ids = ",".join(sorted(set(seg.get("network_manifest_id", pd.Series(dtype=str)).dropna().astype(str))))
    rows: list[dict] = []
    for _, key in grid.iterrows():
        county, date = key["outcome_fips"], key["alert_date"]
        day_origins = set(a.loc[a["alert_date"].eq(date), "home_fips"].astype(str))
        county_rows = allocated.loc[allocated["outcome_fips"].eq(county)].copy()
        county_rows["affected_miles"] = county_rows["commuter_car_miles"].where(county_rows["home_fips"].isin(day_origins), 0.0)
        by_class = county_rows.groupby("origin_class")["affected_miles"].sum()
        total = float(county_rows["total_commuter_car_miles"].iloc[0])
        affected_total = float(county_rows["affected_miles"].sum())
        row = {
            "outcome_fips": county, "alert_date": date,
            "total_commuter_car_miles": total,
            "affected_commuter_car_miles": affected_total,
            "own_affected_car_miles": float(by_class.get("own", 0.0)),
            "cross_affected_car_miles": float(by_class.get("cross", 0.0)),
            "pass_through_affected_car_miles": float(by_class.get("pass_through", 0.0)),
            "own_commuter_car_miles": float(county_rows.loc[county_rows["origin_class"].eq("own"), "commuter_car_miles"].sum()),
            "cross_commuter_car_miles": float(county_rows.loc[county_rows["origin_class"].eq("cross"), "commuter_car_miles"].sum()),
            "pass_through_commuter_car_miles": float(county_rows.loc[county_rows["origin_class"].eq("pass_through"), "commuter_car_miles"].sum()),
            "affected_route_share": affected_total / total,
            "own_affected_share": float(by_class.get("own", 0.0)) / total,
            "cross_affected_share": float(by_class.get("cross", 0.0)) / total,
            "pass_through_affected_share": float(by_class.get("pass_through", 0.0)) / total,
            "failed_route_commuter_car_weight": failed_weight,
            "unallocated_commuter_car_weight": unallocated_weight,
            "omitted_coordinate_worker_weight": omitted_coordinate_workers,
            "omitted_car_share_worker_weight": omitted_car_share_workers,
            label: 1, "same_tract_mode": same_tract_mode, "network_manifest_id": network_ids,
            **metadata,
        }
        row["affected_commuter_car_miles_per_10000"] = affected_total / total * 10_000
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["outcome_fips", "alert_date"]).reset_index(drop=True)


def build_route_pilot_diagnostics(
    pairs: pd.DataFrame, route_results: pd.DataFrame, county_segments: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Compute route-level coverage once per route and segment-level conservation."""
    _require(pairs, ["route_id", "workers", "home_car_share", "commuter_car_miles"], "pairs")
    _require(route_results, ["route_id", "workers", "home_car_share", "commuter_car_miles", "status"], "route_results")
    for frame, label in ((pairs, "pairs"), (route_results, "route_results")):
        if frame["route_id"].isna().any() or frame["route_id"].duplicated().any():
            raise ValueError(f"route_id must be present and unique in {label}")
    if set(pairs["route_id"].astype(str)) != set(route_results["route_id"].astype(str)):
        raise ValueError("pairs and route_results route_id sets differ")
    p = pairs.copy().sort_values("route_id").reset_index(drop=True)
    r = route_results.copy().sort_values("route_id").reset_index(drop=True)
    p["workers"] = _numeric(p, ["workers"], "pairs")["workers"]
    r["workers"] = _numeric(r, ["workers"], "route_results")["workers"]
    for frame, label in ((p, "pairs"), (r, "route_results")):
        frame["home_car_share"] = pd.to_numeric(frame["home_car_share"], errors="coerce")
        invalid = frame["home_car_share"].notna() & ~frame["home_car_share"].between(0, 1)
        if invalid.any():
            raise ValueError(f"{label} has invalid home_car_share")
    eligible = p.get("routing_eligible", pd.Series(True, index=p.index)).fillna(False).astype(bool)
    if p.loc[eligible, "home_car_share"].isna().any():
        raise ValueError("routing-eligible pairs require home_car_share")
    joined = r[["route_id", "status", "workers", "home_car_share", "commuter_car_miles"]].merge(
        p[["route_id", "workers", "home_car_share", "commuter_car_miles"]], on="route_id", suffixes=("_route", "_pair"), validate="one_to_one"
    )
    for column in ("workers", "home_car_share", "commuter_car_miles"):
        if not np.allclose(pd.to_numeric(joined[f"{column}_route"], errors="coerce"), pd.to_numeric(joined[f"{column}_pair"], errors="coerce"), rtol=1e-8, atol=1e-8, equal_nan=True):
            raise ValueError(f"route_results {column} does not match pairs")
    p["commuter_car_weight"] = p["workers"] * p["home_car_share"]
    statuses = r.set_index("route_id").loc[p["route_id"], "status"].astype(str).str.lower()
    success = statuses.isin(SUCCESS_STATUSES).to_numpy() & eligible.to_numpy()
    selected_workers = float(p.loc[eligible, "workers"].sum())
    successful_workers = float(p.loc[success, "workers"].sum())
    selected_car = float(p.loc[eligible, "commuter_car_weight"].sum())
    successful_car = float(p.loc[success, "commuter_car_weight"].sum())
    def optional_sum(column: str) -> float:
        return float(pd.to_numeric(p[column], errors="coerce").fillna(0).sum()) if column in p else 0.0
    coverage = pd.DataFrame([
        {"metric": "selected_worker_weight", "value": selected_workers},
        {"metric": "selected_commuter_car_weight", "value": selected_car},
        {"metric": "successful_worker_weight", "value": successful_workers},
        {"metric": "successful_commuter_car_weight", "value": successful_car},
        {"metric": "successful_worker_share", "value": successful_workers / selected_workers if selected_workers else 0.0},
        {"metric": "successful_commuter_car_share", "value": successful_car / selected_car if selected_car else 0.0},
        {"metric": "failed_route_worker_weight", "value": selected_workers - successful_workers},
        {"metric": "failed_route_commuter_car_weight", "value": selected_car - successful_car},
        {"metric": "omitted_external_endpoint_worker_weight", "value": optional_sum("external_endpoint_worker_weight")},
        {"metric": "omitted_external_endpoint_car_weight", "value": optional_sum("external_endpoint_car_weight")},
        {"metric": "omitted_coordinate_worker_weight", "value": optional_sum("omitted_coordinate_worker_weight")},
        {"metric": "omitted_car_share_worker_weight", "value": optional_sum("omitted_car_share_worker_weight")},
    ])
    status = r["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
    same = p.get("same_tract", pd.Series(False, index=p.index)).fillna(False).astype(bool)
    route_rows: list[dict] = [
        {"metric": "failed_route_count", "value": float((eligible.to_numpy() & ~statuses.isin(SUCCESS_STATUSES).to_numpy()).sum())},
        {"metric": "input_omission_route_count", "value": float((~eligible).sum())},
        {"metric": "same_tract_pair_count", "value": float(same.sum())},
        {"metric": "same_tract_worker_share", "value": float(p.loc[same & eligible, "workers"].sum() / selected_workers) if selected_workers else 0.0},
        {"metric": "same_tract_commuter_car_weight_share", "value": float(p.loc[same & eligible, "commuter_car_weight"].sum() / selected_car) if selected_car else 0.0},
    ]
    seg = county_segments.copy()
    if not seg.empty and {"route_miles_total", "route_miles_in_county", "unallocated_miles"}.issubset(seg.columns):
        values = _numeric(seg, ["route_miles_total", "route_miles_in_county", "unallocated_miles"], "county_segments")
        seg[values.columns] = values
        totals = seg.groupby("route_id").agg(
            total=("route_miles_total", "max"), allocated=("route_miles_in_county", "sum"), unallocated=("unallocated_miles", "sum"),
            failed=("segment_type", lambda x: any(str(v).lower() in {"failed_route", "failedroute"} for v in x)),
        )
        evaluable = totals.loc[~totals["failed"]]
        relative_gap = ((evaluable["allocated"] + evaluable["unallocated"] - evaluable["total"]).abs() / evaluable["total"].replace(0, np.nan)).fillna(0)
        aggregate_total = float(evaluable["total"].sum())
        aggregate_gap = abs(float((evaluable["allocated"] + evaluable["unallocated"]).sum()) - aggregate_total) / aggregate_total if aggregate_total else 0.0
        row_pass = bool((relative_gap <= 0.005).all())
        route_rows.extend([
            {"metric": "mileage_conservation_max_relative_gap", "value": float(relative_gap.max()) if not relative_gap.empty else 0.0},
            {"metric": "mileage_conservation_aggregate_gap", "value": aggregate_gap},
            {"metric": "mileage_conservation_row_pass", "value": float(row_pass)},
            {"metric": "mileage_conservation_pass", "value": float(aggregate_gap <= 0.001)},
            {"metric": "mileage_conservation_accepted", "value": float(row_pass and aggregate_gap <= 0.001)},
            {"metric": "unallocated_miles", "value": float(totals["unallocated"].sum())},
        ])
    if {"home_fips", "work_fips", "outcome_fips", "workers", "home_car_share", "route_miles_in_county"}.issubset(seg.columns):
        allocated = seg.loc[seg["outcome_fips"].notna()].copy()
        allocated["car_miles"] = allocated["workers"] * allocated["home_car_share"] * allocated["route_miles_in_county"]
        own = allocated["home_fips"].astype(str).eq(allocated["outcome_fips"].astype(str))
        passthrough = ~own & ~allocated["work_fips"].astype(str).eq(allocated["outcome_fips"].astype(str))
        route_rows.extend([
            {"metric": "own_total_car_miles", "value": float(allocated.loc[own, "car_miles"].sum())},
            {"metric": "cross_total_car_miles", "value": float(allocated.loc[~own & ~passthrough, "car_miles"].sum())},
            {"metric": "pass_through_total_car_miles", "value": float(allocated.loc[passthrough, "car_miles"].sum())},
        ])
    return {"coverage": coverage, "route": pd.DataFrame(route_rows), "statuses": status, "pairs": p, "county_segments": seg}


def compare_destination_and_route_exposure(route_exposures: pd.DataFrame, existing_exposure: pd.DataFrame) -> pd.DataFrame:
    """Return percentile and correlation evidence using the existing dosage column."""
    r, e = route_exposures.copy(), existing_exposure.copy()
    keys = [column for column in ("outcome_fips", "alert_date") if column in r and column in e]
    existing_column = next((column for column in ("existing_destination_dosage", "existing_commuter_car_miles", "destination_dosage", "commuter_car_miles") if column in e), None)
    if existing_column:
        e = e.rename(columns={existing_column: "existing_destination_dosage"})
    if keys:
        r = r.merge(e.drop_duplicates(keys), on=keys, how="left", suffixes=("", "_existing"))
    measures = [column for column in ("affected_route_share", "affected_commuter_car_miles", "existing_destination_dosage", "simple_commuter_share", "straight_line_allocation", "current_county_denominator") if column in r]
    rows: list[dict] = []
    for column in measures:
        values = pd.to_numeric(r[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            rows.append({"metric": column, LABEL: 1, "n": len(values), "mean": float(values.mean()), "p50": float(values.quantile(.5)), "p90": float(values.quantile(.9)), "min": float(values.min()), "max": float(values.max())})
    if {"affected_route_share", "existing_destination_dosage"}.issubset(r.columns):
        pair = r[["affected_route_share", "existing_destination_dosage"]].apply(pd.to_numeric, errors="coerce").dropna()
        defined = len(pair) > 1 and pair.nunique().gt(1).all()
        correlation = float(pair.corr().iloc[0, 1]) if defined else 0.0
        rows.append({"metric": "correlation_affected_route_share_with_existing_destination_dosage", LABEL: 1, "correlation_defined": bool(defined), "n": len(pair), "mean": correlation, "p50": correlation, "p90": correlation, "min": correlation, "max": correlation})
    return pd.DataFrame(rows)


def load_reviewed_route_alerts(*, loader: Callable[..., pd.DataFrame] = load_combined_alerts) -> pd.DataFrame:
    detail = loader(window="night", detail=True)
    _require(detail, ["fips", "effective_crash_date", "geo_scope"], "reviewed combined alerts")
    columns = ["fips", "effective_crash_date", "geo_scope"] + (["original_fips"] if "original_fips" in detail else [])
    out = detail[columns].rename(columns={"fips": "home_fips", "effective_crash_date": "alert_date"}).copy()
    out["home_fips"] = _codes(out["home_fips"])
    out["alert_date"] = pd.to_datetime(out["alert_date"], errors="raise").dt.normalize()
    return out.drop_duplicates().sort_values(["alert_date", "home_fips"]).reset_index(drop=True)


def evaluate_pilot_gate(evidence: Mapping[str, object]) -> tuple[bool, pd.DataFrame]:
    def strict_bool(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in {0.0, 1.0}:
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("boolean gate evidence must be true/false or 1/0")
    definitions = [
        ("routing_coverage", "routing_coverage", lambda v: float(v) >= .99, ">= 0.99"),
        ("row_conservation", "row_conservation_pass", strict_bool, "all route gaps <= 0.005"),
        ("aggregate_conservation", "aggregate_conservation_gap", lambda v: float(v) <= .001, "gap <= 0.001"),
        ("tract_aggregation_bias", "tract_aggregation_bias_acceptable", strict_bool, "acceptable and not alerted-origin biased"),
        ("same_tract_dominance", "same_tract_dominance_acceptable", strict_bool, "imputation not dominant"),
        ("same_tract_sign_stability", "same_tract_sign_stable", strict_bool, "sign stable across modes"),
        ("denominator_stability", "denominator_stable", strict_bool, "positive and stable across modes"),
        ("route_destination_materiality", "route_destination_material", strict_bool, "material or clearer mechanism"),
        ("computational_feasibility", "computationally_feasible", strict_bool, "national build feasible"),
    ]
    rows = []
    for criterion, key, check, threshold in definitions:
        value = evidence.get(key)
        try:
            passed = value is not None and not pd.isna(value) and bool(check(value))
        except (TypeError, ValueError):
            passed = False
        rows.append({"criterion": criterion, "evidence_key": key, "value": value, "threshold": threshold, "passed": passed})
    table = pd.DataFrame(rows)
    return bool(table["passed"].all()), table


def default_output_paths(root: Path, year: int, mode: str) -> dict:
    if mode not in SAME_TRACT_MODES:
        raise ValueError(f"unsupported same-tract mode: {mode}")
    table_dir = Path(root) / "output" / "tables"
    suffix = "" if mode == "primary_calibrated" else f"_{mode}"
    return {
        "tables": {
            "input": table_dir / f"route_pilot_input_diagnostics{suffix}.csv",
            "route": table_dir / f"route_pilot_route_diagnostics{suffix}.csv",
            "county_exposure": table_dir / f"route_pilot_county_exposure_summary{suffix}.csv",
            "comparison": table_dir / f"route_pilot_exposure_comparison{suffix}.csv",
            "gate": table_dir / f"route_pilot_gate{suffix}.csv",
        },
        "report": Path(root) / "output" / f"ROUTE_EXPOSURE_PILOT_REPORT{suffix}.md",
        "rejected_report": Path(root) / "output" / f"ROUTE_EXPOSURE_PILOT_REPORT{suffix}_REJECTED.md",
    }


def write_pilot_report(diagnostics: Mapping[str, pd.DataFrame], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Wisconsin route exposure pilot", "", f"Label: `{LABEL}`", "", "The pilot uses one county commuter-car-mile denominator and preserves failed/unallocated route weight as audit diagnostics.", "", "## Diagnostics", ""]
    for name, original in diagnostics.items():
        frame = original.replace([np.inf, -np.inf], np.nan).fillna(0)
        lines.extend([f"### {name}", "", frame.head(20).to_markdown(index=False) if not frame.empty else "No observations.", ""])
    lines.extend(["## Labels", "", "`route_exposure_2022`, `same_tract_mode`, and `network_manifest_id` identify model inputs and provenance.", "", "## Limitations and source URLs", "", "Coverage is a Wisconsin pilot. External endpoints, input omissions, route failures, unallocated mileage, and same-tract imputation are reported separately.", "", "Source URLs: [Census LODES](https://lehd.ces.census.gov/data/), [Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html), [Geofabrik/OpenStreetMap](https://download.geofabrik.de/), and [OSRM](https://project-osrm.org/).", "", "Exact command: `python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode primary_calibrated --write-report`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required pilot artifact at {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--same-tract-mode", choices=sorted(SAME_TRACT_MODES), default="primary_calibrated")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROUTE_PILOT_CACHE)
    parser.add_argument("--county-segments", type=Path)
    parser.add_argument("--alerts", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--route-results", type=Path)
    parser.add_argument("--existing-exposure", type=Path)
    parser.add_argument("--flow-diagnostics", type=Path)
    parser.add_argument("--gate-evidence", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    cache = Path(args.cache_dir)
    segment_suffix = (
        str(args.year)
        if args.same_tract_mode == "primary_calibrated"
        else f"{args.year}_{args.same_tract_mode}"
    )
    try:
        pairs = _read_table(args.pairs or cache / f"pilot_tract_pairs_{args.year}.parquet")
        routes = _read_table(args.route_results or cache / f"route_results_{args.year}.parquet")
        segments = _read_table(
            args.county_segments or cache / f"route_county_segments_{segment_suffix}.parquet"
        )
    except FileNotFoundError as exc:
        parser.error(f"{exc}. Run the flow, network, and county-mile commands first; defaults are mode-specific.")
    alerts = _read_table(args.alerts) if args.alerts else load_reviewed_route_alerts()
    exposures = build_alert_date_exposures(segments, alerts, args.same_tract_mode)
    diagnostics = build_route_pilot_diagnostics(pairs, routes, segments)
    flow_diagnostics_path = args.flow_diagnostics or cache / f"pilot_flow_diagnostics_{args.year}.csv"
    if flow_diagnostics_path.exists():
        flow_diagnostics = _read_table(flow_diagnostics_path)
        for source_column, metric in (
            ("external_endpoint_worker_weight", "omitted_external_endpoint_worker_weight"),
            ("missing_coordinate_worker_weight", "omitted_coordinate_worker_weight"),
            ("missing_home_car_share_worker_weight", "omitted_car_share_worker_weight"),
        ):
            if source_column in flow_diagnostics:
                diagnostics["coverage"] = diagnostics["coverage"].loc[
                    ~diagnostics["coverage"]["metric"].eq(metric)
                ]
                diagnostics["coverage"] = pd.concat(
                    [
                        diagnostics["coverage"],
                        pd.DataFrame(
                            [{"metric": metric, "value": float(pd.to_numeric(flow_diagnostics[source_column], errors="coerce").sum())}]
                        ),
                    ],
                    ignore_index=True,
                )
    comparison = compare_destination_and_route_exposure(exposures, _read_table(args.existing_exposure)) if args.existing_exposure else pd.DataFrame()
    coverage = diagnostics["coverage"].set_index("metric")["value"]
    route_diag = diagnostics["route"].set_index("metric")["value"]
    evidence: dict[str, object] = {
        "routing_coverage": coverage.get("successful_commuter_car_share"),
        "row_conservation_pass": bool(route_diag.get("mileage_conservation_row_pass", 0)),
        "aggregate_conservation_gap": route_diag.get("mileage_conservation_aggregate_gap"),
    }
    if args.gate_evidence:
        evidence_rows = _read_table(args.gate_evidence)
        _require(evidence_rows, ["metric", "value"], "gate evidence")
        evidence.update(dict(zip(evidence_rows["metric"], evidence_rows["value"], strict=True)))
    if comparison.empty:
        evidence["route_destination_material"] = False
    accepted, gate = evaluate_pilot_gate(evidence)
    outputs = default_output_paths(args.output_root, args.year, args.same_tract_mode)
    for path in outputs["tables"].values():
        path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics["coverage"].to_csv(outputs["tables"]["input"], index=False)
    diagnostics["route"].to_csv(outputs["tables"]["route"], index=False)
    gate.to_csv(outputs["tables"]["gate"], index=False)
    report_diagnostics = {"coverage": diagnostics["coverage"], "route": diagnostics["route"], "gate": gate}
    if accepted:
        exposures.to_csv(outputs["tables"]["county_exposure"], index=False)
        comparison.to_csv(outputs["tables"]["comparison"], index=False)
        if args.write_report:
            write_pilot_report({**report_diagnostics, "comparison": comparison}, outputs["report"])
    else:
        outputs["tables"]["county_exposure"].unlink(missing_ok=True)
        outputs["tables"]["comparison"].unlink(missing_ok=True)
        outputs["report"].unlink(missing_ok=True)
        if args.write_report:
            write_pilot_report(report_diagnostics, outputs["rejected_report"])
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
