"""Stream one national route-flow partition directly into county segments."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from build_route_pilot_county_miles import _base_segment_row, _straight_line_miles
from build_route_pilot_network import _derive_route_id, _route_signature_from_row
from route_exposure_core import MILES_PER_METER, allocate_route_miles


SCHEMA_VERSION = "route_national.segments.v1"
AUDIT_FILE_NAME = "route_audits.parquet"
SEGMENT_FILE_NAME = "county_segments.parquet"
CHECKPOINT_COMMIT_FILE_NAME = "checkpoint_commit.json"
QA_GEOMETRY_DIR_NAME = "qa_geometries"

AUDIT_COLUMNS = [
    "schema_version",
    "route_signature",
    "route_id",
    "status",
    "distance_m",
    "duration_s",
    "route_miles_total",
    "unallocated_miles",
    "workers",
    "home_car_share",
    "commuter_car_weight",
    "omitted_coordinate_worker_weight",
    "omitted_car_share_worker_weight",
    "routing_eligible",
    "error_message",
    "source_manifest_id",
    "network_manifest_id",
    "source_partition_id",
    "routed_at_utc",
]


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{next(tempfile._get_candidate_names())}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp.json", dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _empty_audits() -> pd.DataFrame:
    return pd.DataFrame(columns=AUDIT_COLUMNS)


def _empty_segments() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "schema_version", "route_signature", "route_id", "county_fips", "outcome_fips",
            "route_miles_total", "route_miles_in_county", "unallocated_miles", "segment_type",
            "analysis_year", "lodes_source_year", "acs_car_share_vintage",
            "source_partition_id", "source_manifest_id", "network_manifest_id",
        ]
    )


def _load_checkpoint(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, list(dict.fromkeys([*columns, *frame.columns]))].copy()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_committed_checkpoint(
    audit_path: Path, segment_path: Path, commit_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a checkpoint only when both artifacts match its commit marker."""
    try:
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        valid = (
            commit.get("schema_version") == SCHEMA_VERSION
            and commit.get("audit_sha256") == _file_sha256(audit_path)
            and commit.get("segment_sha256") == _file_sha256(segment_path)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    if not valid:
        return _empty_audits(), _empty_segments()
    return (
        _load_checkpoint(audit_path, AUDIT_COLUMNS),
        _load_checkpoint(segment_path, _empty_segments().columns.tolist()),
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _route_miles(distance_m: object) -> float:
    value = pd.to_numeric(distance_m, errors="coerce")
    return float(value) * MILES_PER_METER if pd.notna(value) and float(value) >= 0 else 0.0


def _sample_signature(route_signature: str, rate: float) -> bool:
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    value = int(hashlib.sha256(str(route_signature).encode("utf-8")).hexdigest(), 16)
    return value < rate * (1 << 256)


def _route_client_result(row: pd.Series, base_url: str, route_client: Callable) -> dict:
    """Call the configured OSRM client and convert client exceptions into audits."""
    route_id = str(row["route_id"])
    try:
        result = route_client(
            float(row["home_lon"]), float(row["home_lat"]),
            float(row["work_lon"]), float(row["work_lat"]), route_id, base_url,
        )
    except Exception as exc:
        return {
            "route_id": route_id,
            "status": "RouteClientError",
            "distance_m": None,
            "duration_s": None,
            "geometry": None,
            "error_message": str(exc),
        }
    if not isinstance(result, Mapping):
        return {
            "route_id": route_id,
            "status": "MalformedResponse",
            "distance_m": None,
            "duration_s": None,
            "geometry": None,
            "error_message": "route client returned a non-mapping result",
        }
    return {
        "route_id": str(result.get("route_id", route_id)),
        "status": str(result.get("status") or "MalformedResponse"),
        "distance_m": result.get("distance_m"),
        "duration_s": result.get("duration_s"),
        "geometry": result.get("geometry"),
        "error_message": result.get("error_message"),
    }


def _ineligible_result(row: pd.Series) -> dict:
    missing_coordinates = any(pd.isna(row.get(column)) for column in ("home_lon", "home_lat", "work_lon", "work_lat"))
    return {
        "route_id": str(row["route_id"]),
        "status": "Ineligible",
        "distance_m": None,
        "duration_s": None,
        "geometry": None,
        "error_message": "pair omitted before routing because endpoint coordinates are missing" if missing_coordinates else "pair omitted before routing because routing_eligible is false",
    }


def _is_routing_eligible(row: pd.Series) -> bool:
    declared = row.get("routing_eligible", True)
    if pd.isna(declared) or not bool(declared):
        return False
    return all(pd.notna(row.get(column)) for column in ("home_lon", "home_lat", "work_lon", "work_lat"))


def _nonblank_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _adapt_partition_provenance(
    frame: pd.DataFrame, *, network_manifest_id: str | None
) -> None:
    """Map Task 3 partition provenance onto the national routing contract."""
    if "source_manifest_id" not in frame.columns:
        frame["source_manifest_id"] = pd.NA

    task3_columns = ("lodes_source_id", "provenance_sha256")
    if any(column in frame.columns for column in task3_columns):
        missing = [column for column in task3_columns if column not in frame.columns]
        blank = [
            column
            for column in task3_columns
            if column in frame.columns and frame[column].map(_nonblank_text).isna().any()
        ]
        if missing or blank:
            details = sorted({*missing, *blank})
            raise ValueError(
                f"pairs missing complete Task 3 provenance fields: {details}"
            )
        derived = frame.apply(
            lambda row: (
                f"{_nonblank_text(row['lodes_source_id'])}|"
                f"{_nonblank_text(row['provenance_sha256'])}"
            ),
            axis=1,
        )
        explicit = frame["source_manifest_id"].map(_nonblank_text)
        conflicts = explicit.notna() & explicit.ne(derived)
        if conflicts.any():
            raise ValueError(
                "source_manifest_id does not match the Task 3 partition provenance"
            )
        frame["source_manifest_id"] = derived
    else:
        frame["source_manifest_id"] = frame["source_manifest_id"].map(_nonblank_text)

    if network_manifest_id is None:
        return
    if not isinstance(network_manifest_id, str):
        raise ValueError("network_manifest_id must be a nonblank string")
    network_id = _nonblank_text(network_manifest_id)
    if network_id is None:
        raise ValueError("network_manifest_id must be nonblank")
    if "network_manifest_id" in frame.columns:
        existing_network_ids = frame["network_manifest_id"].map(_nonblank_text)
        if existing_network_ids.isna().any():
            raise ValueError("network_manifest_id in the partition input must be nonblank")
        conflicts = existing_network_ids.ne(network_id)
        if conflicts.any():
            raise ValueError("network_manifest_id does not match the partition input")
    frame["network_manifest_id"] = network_id


def _validate_provenance(frame: pd.DataFrame) -> None:
    required = ("source_manifest_id", "network_manifest_id", "source_partition_id")
    missing = [column for column in required if column not in frame.columns]
    blank = [
        column
        for column in required
        if column in frame.columns
        and frame[column].map(lambda value: pd.isna(value) or not str(value).strip()).any()
    ]
    if missing or blank:
        details = [*missing, *[column for column in blank if column not in missing]]
        raise ValueError(f"pairs missing complete provenance fields: {sorted(details)}")


def _national_route_signature(row: pd.Series) -> str:
    """Extend the shared route signature with the national partition identity."""
    payload = {
        "route_signature": _route_signature_from_row(row),
        "source_partition_id": str(row["source_partition_id"]).strip(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audit_row(row: pd.Series, result: Mapping[str, object], *, total: float, unallocated: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "route_signature": str(row["route_signature"]),
        "route_id": str(row["route_id"]),
        "status": str(result.get("status") or "MalformedResponse"),
        "distance_m": result.get("distance_m"),
        "duration_s": result.get("duration_s"),
        "route_miles_total": float(total),
        "unallocated_miles": float(unallocated),
        "workers": row.get("workers"),
        "home_car_share": row.get("home_car_share"),
        "commuter_car_weight": row.get("commuter_car_weight"),
        "omitted_coordinate_worker_weight": row.get("omitted_coordinate_worker_weight", 0.0),
        "omitted_car_share_worker_weight": row.get("omitted_car_share_worker_weight", 0.0),
        "routing_eligible": row.get("routing_eligible", True),
        "error_message": result.get("error_message"),
        "source_manifest_id": row.get("source_manifest_id"),
        "network_manifest_id": row.get("network_manifest_id"),
        "source_partition_id": row.get("source_partition_id"),
        "routed_at_utc": _now_utc(),
    }


def _segment_rows(row: pd.Series, allocated: pd.DataFrame) -> list[dict]:
    base = _base_segment_row(row)
    block_distance = pd.to_numeric(row.get("block_pair_straight_line_miles"), errors="coerce")
    if pd.notna(block_distance):
        base["straight_line_miles"] = float(block_distance)
    else:
        base["straight_line_miles"] = _straight_line_miles(
            row["home_lon"], row["home_lat"], row["work_lon"], row["work_lat"]
        )
    out = []
    for _, segment in allocated.iterrows():
        out.append(
            {
                **base,
                "schema_version": SCHEMA_VERSION,
                "route_signature": str(row["route_signature"]),
                "county_fips": segment.get("county_fips"),
                "outcome_fips": segment.get("county_fips"),
                "route_miles_total": float(segment["route_miles_total"]),
                "route_miles_in_county": float(segment["route_miles_in_county"]),
                "unallocated_miles": float(segment["unallocated_miles"]),
                "segment_type": str(segment["segment_type"]),
                "analysis_year": row.get("analysis_year"),
                "lodes_source_year": row.get("lodes_source_year"),
                "acs_car_share_vintage": row.get("acs_car_share_vintage"),
                "source_partition_id": row.get("source_partition_id"),
                "source_manifest_id": row.get("source_manifest_id"),
                "network_manifest_id": row.get("network_manifest_id"),
            }
        )
    return out


def _qa_document(row: pd.Series, result: Mapping[str, object]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "route_id": str(row["route_id"]),
        "route_signature": str(row["route_signature"]),
        "source_manifest_id": row.get("source_manifest_id"),
        "network_manifest_id": row.get("network_manifest_id"),
        "status": result.get("status"),
        "distance_m": result.get("distance_m"),
        "duration_s": result.get("duration_s"),
        "error_message": result.get("error_message"),
        "geometry": result.get("geometry"),
    }


def route_partition_to_segments(
    pairs: pd.DataFrame,
    county_boundaries: object,
    *,
    base_url: str,
    cache_dir: Path,
    route_client: Callable,
    network_manifest_id: str | None = None,
    max_workers: int = 8,
    checkpoint_every: int = 10_000,
    geometry_sample_rate: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route one flow partition and immediately emit county-mile segments.

    ``route_client`` is the existing OSRM retry/status-parsing client (or an
    equivalent test double) and accepts endpoint coordinates, route ID, and
    base URL.  Successful response geometry is allocated during this call and
    discarded unless its full provenance-aware signature enters the QA sample.
    """
    if int(max_workers) < 1 or int(checkpoint_every) < 1:
        raise ValueError("max_workers and checkpoint_every must be positive")
    if not isinstance(geometry_sample_rate, (int, float)) or not math.isfinite(float(geometry_sample_rate)) or not 0 <= float(geometry_sample_rate) <= 1:
        raise ValueError("geometry_sample_rate must be between zero and one")
    required = {"home_lon", "home_lat", "work_lon", "work_lat"}
    if "route_id" not in pairs.columns:
        required.update({"home_tract", "work_tract"})
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing required columns: {sorted(missing)}")

    cache_dir = Path(cache_dir)
    audit_path = cache_dir / AUDIT_FILE_NAME
    segment_path = cache_dir / SEGMENT_FILE_NAME
    commit_path = cache_dir / CHECKPOINT_COMMIT_FILE_NAME
    frame = pairs.copy()
    _adapt_partition_provenance(frame, network_manifest_id=network_manifest_id)
    _validate_provenance(frame)
    frame["route_id"] = frame.apply(_derive_route_id, axis=1)
    frame["route_signature"] = frame.apply(_national_route_signature, axis=1)
    frame = frame.sort_values(["route_id", "route_signature"]).drop_duplicates("route_signature", keep="first").reset_index(drop=True)

    existing_audits, existing_segments = _load_committed_checkpoint(audit_path, segment_path, commit_path)
    current_signatures = set(frame["route_signature"].astype(str))
    existing_audits = existing_audits.loc[existing_audits["route_signature"].astype(str).isin(current_signatures)].copy()
    existing_segments = existing_segments.loc[existing_segments["route_signature"].astype(str).isin(current_signatures)].copy()
    successful_signatures = set(
        existing_audits.loc[existing_audits["status"].astype(str).eq("Ok"), "route_signature"].astype(str)
    )
    segment_signatures = set(existing_segments["route_signature"].astype(str))
    incomplete_successes = successful_signatures - segment_signatures
    if incomplete_successes:
        existing_audits = existing_audits.loc[
            ~existing_audits["route_signature"].astype(str).isin(incomplete_successes)
        ].copy()
        existing_segments = existing_segments.loc[
            ~existing_segments["route_signature"].astype(str).isin(incomplete_successes)
        ].copy()
    completed = set(existing_audits["route_signature"].astype(str))

    audit_records = existing_audits.to_dict("records")
    segment_records = existing_segments.to_dict("records")
    pending = [row for _, row in frame.iterrows() if str(row["route_signature"]) not in completed]

    def write_checkpoint() -> None:
        audits = pd.DataFrame.from_records(audit_records)
        segments = pd.DataFrame.from_records(segment_records)
        for column in AUDIT_COLUMNS:
            if column not in audits.columns:
                audits[column] = pd.NA
        audits = audits.loc[:, AUDIT_COLUMNS].sort_values(["route_id", "route_signature"]).reset_index(drop=True)
        if segments.empty:
            segments = _empty_segments()
        else:
            segments = segments.sort_values(["route_id", "route_signature", "segment_type", "county_fips"], na_position="last").reset_index(drop=True)
        _atomic_write_parquet(audits, audit_path)
        _atomic_write_parquet(segments, segment_path)
        _atomic_write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "audit_sha256": _file_sha256(audit_path),
                "segment_sha256": _file_sha256(segment_path),
            },
            commit_path,
        )

    def prune_qa_geometries() -> None:
        qa_dir = cache_dir / QA_GEOMETRY_DIR_NAME
        if not qa_dir.exists():
            return
        retained_signatures = {
            str(record["route_signature"])
            for record in audit_records
            if str(record.get("status")) == "Ok"
            and _sample_signature(str(record["route_signature"]), float(geometry_sample_rate))
        }
        for geometry_path in qa_dir.glob("*.geojson"):
            if geometry_path.stem not in retained_signatures:
                geometry_path.unlink()

    for start in range(0, len(pending), int(checkpoint_every)):
        batch = pending[start:start + int(checkpoint_every)]
        eligible = [row for row in batch if _is_routing_eligible(row)]
        results_by_signature = {
            str(row["route_signature"]): _ineligible_result(row)
            for row in batch if not _is_routing_eligible(row)
        }
        if int(max_workers) == 1:
            results_by_signature.update({
                str(row["route_signature"]): _route_client_result(row, base_url, route_client)
                for row in eligible
            })
        elif eligible:
            with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
                results_by_signature.update(
                    zip(
                        (str(row["route_signature"]) for row in eligible),
                        executor.map(lambda row: _route_client_result(row, base_url, route_client), eligible),
                        strict=True,
                    )
                )

        for row in batch:
            result = results_by_signature[str(row["route_signature"])]
            total = _route_miles(result.get("distance_m"))
            unallocated = total
            if result["status"] == "Ok":
                if not isinstance(result.get("geometry"), Mapping):
                    result = {
                        **result,
                        "status": "MalformedResponse",
                        "error_message": "successful route response is missing a GeoJSON geometry mapping",
                    }
                else:
                    try:
                        geometry = dict(result["geometry"])
                        geometry["properties"] = {"distance_m": result.get("distance_m")}
                        allocated = allocate_route_miles(geometry, county_boundaries, str(row["route_id"]))
                        if not allocated.empty:
                            total = float(allocated["route_miles_total"].max())
                            unallocated = float(allocated["unallocated_miles"].sum())
                            segment_records.extend(_segment_rows(row, allocated))
                        else:
                            result = {**result, "status": "AllocationError", "error_message": "county allocator returned no segments"}
                    except (TypeError, ValueError) as exc:
                        result = {**result, "status": "AllocationError", "error_message": str(exc)}
            audit_records.append(_audit_row(row, result, total=total, unallocated=unallocated))
            if result["status"] == "Ok" and isinstance(result.get("geometry"), Mapping) and _sample_signature(str(row["route_signature"]), float(geometry_sample_rate)):
                _atomic_write_json(_qa_document(row, result), cache_dir / QA_GEOMETRY_DIR_NAME / f"{row['route_signature']}.geojson")
        write_checkpoint()

    if not pending:
        write_checkpoint()
    prune_qa_geometries()
    return _load_checkpoint(audit_path, AUDIT_COLUMNS).loc[:, AUDIT_COLUMNS], _load_checkpoint(segment_path, _empty_segments().columns.tolist())
