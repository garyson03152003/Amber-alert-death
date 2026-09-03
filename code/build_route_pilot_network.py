"""Prepare the Wisconsin route-pilot OSRM network and cached tract-pair routes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import pandas as pd
import requests

from config import ROUTE_PILOT_CACHE
from crash_download import download_bulk_file, sha256_file
from route_exposure_core import parse_osrm_route

DEFAULT_YEAR = 2022
PILOT_STATES = ("wi", "il", "ia", "mn", "mi")
STATE_SLUGS = {
    "wi": "wisconsin",
    "il": "illinois",
    "ia": "iowa",
    "mn": "minnesota",
    "mi": "michigan",
}
GEOFABRIK_BASE = "https://download.geofabrik.de/north-america/us"
OSRM_VERSION = "5.27.1"
OSRM_PROFILE = "car"
OSRM_ALGORITHM = "mld"
OSRM_IMAGE = f"ghcr.io/project-osrm/osrm-backend:{OSRM_VERSION}"
OSM_MERGE_IMAGE = "ghcr.io/osmcode/osmium-tool:v1.18.0"
ROUTE_TIMEOUT = (5, 30)
ROUTE_RETRY_DELAYS = (1.0, 2.0)
PERMANENT_OSRM_STATUSES = {"NoRoute", "NoSegment", "TooBig"}
_ROUTE_SESSION_LOCAL = threading.local()
CHECKPOINT_COLUMNS = [
    "route_signature",
    "route_id",
    "home_lon",
    "home_lat",
    "work_lon",
    "work_lat",
    "home_tract",
    "work_tract",
    "home_county",
    "work_county",
    "workers",
    "home_car_share",
    "commuter_car_weight",
    "block_pair_straight_line_miles",
    "commuter_car_miles",
    "same_tract",
    "routing_eligible",
    "omitted_coordinate_worker_weight",
    "omitted_car_share_worker_weight",
    "status",
    "distance_m",
    "duration_s",
    "geometry_path",
    "error_message",
    "source_manifest_id",
    "network_manifest_id",
    "routed_at_utc",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _geofabrik_url(state: str, year: int) -> str:
    state_key = str(state).lower()
    if state_key not in STATE_SLUGS:
        raise ValueError(f"unsupported pilot state: {state}")
    historical_suffix = f"{int(year) % 100:02d}0101"
    return f"{GEOFABRIK_BASE}/{STATE_SLUGS[state_key]}-{historical_suffix}.osm.pbf"


def _network_manifest_id(source_sha256: list[str]) -> str:
    joined = "|".join(sorted(source_sha256)) + f"|{OSRM_PROFILE}|{OSRM_VERSION}|{OSRM_ALGORITHM}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{next(tempfile._get_candidate_names())}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _empty_checkpoint_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CHECKPOINT_COLUMNS)


def _load_checkpoint_rows(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return _empty_checkpoint_frame()
    frame = pd.read_parquet(path)
    for column in CHECKPOINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["route_signature"] = frame.apply(_route_signature_from_row, axis=1)
    return frame.loc[:, CHECKPOINT_COLUMNS].copy()


def _write_checkpoint_rows(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in CHECKPOINT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["route_signature"] = out.apply(_route_signature_from_row, axis=1)
    out = out.loc[:, CHECKPOINT_COLUMNS].sort_values("route_id").reset_index(drop=True)
    _atomic_write_parquet(out, Path(path))


def download_geofabrik_extract(state: str, year: int, cache_dir: Path, session: requests.Session) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_key = str(state).lower()
    target = cache_dir / f"{state_key}-{int(year)}.osm.pbf"
    sidecar = target.with_name(target.name + ".manifest.json")
    if target.exists() and target.stat().st_size > 0:
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            valid = (
                record.get("url") == _geofabrik_url(state_key, year)
                and int(record.get("bytes", -1)) == target.stat().st_size
                and record.get("sha256") == sha256_file(target)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise RuntimeError(
                f"cached Geofabrik input failed manifest/checksum validation at {target}; "
                "remove or replace the cache and its .manifest.json sidecar, then rerun"
            )
        return target
    download_bulk_file(session, _geofabrik_url(state_key, year), target, timeout=1200)
    _atomic_write_json(
        {
            "path": str(target),
            "url": _geofabrik_url(state_key, year),
            "retrieved_at_utc": _now_utc(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "attribution": "© OpenStreetMap contributors",
            "license": "ODbL-1.0",
        },
        sidecar,
    )
    return target


def prepare_osrm_network(pbf_paths: list[Path], network_dir: Path, docker_runner: Callable) -> dict:
    if not pbf_paths:
        raise ValueError("pbf_paths must contain at least one state extract")
    network_dir = Path(network_dir)
    network_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = network_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    merged_path = network_dir / f"pilot-network-{DEFAULT_YEAR}.osm.pbf"
    manifest_path = network_dir / "network_manifest.json"
    source_paths = sorted((Path(path) for path in pbf_paths), key=lambda path: path.name)
    source_sha256 = [sha256_file(path) for path in source_paths]
    manifest_id = _network_manifest_id(source_sha256)
    staged_paths: list[Path] = []
    for source_path, source_digest in zip(source_paths, source_sha256, strict=True):
        staged_path = inputs_dir / source_path.name
        if not staged_path.exists() or sha256_file(staged_path) != source_digest:
            shutil.copy2(source_path, staged_path)
        staged_paths.append(staged_path)

    mount_arg = f"{network_dir.resolve()}:/data"
    merged_container_path = "/data/pilot-network-2022.osm.pbf"
    routed_base_path = "/data/pilot-network-2022.osrm"

    commands = [
        [
            "docker",
            "run",
            "--rm",
            "-v",
            mount_arg,
            OSM_MERGE_IMAGE,
            "osmium",
            "merge",
            "--overwrite",
            *[f"/data/inputs/{path.name}" for path in staged_paths],
            "-o",
            merged_container_path,
        ],
        [
            "docker",
            "run",
            "--rm",
            "-v",
            mount_arg,
            OSRM_IMAGE,
            "osrm-extract",
            "-p",
            "/opt/car.lua",
            merged_container_path,
        ],
        ["docker", "run", "--rm", "-v", mount_arg, OSRM_IMAGE, "osrm-partition", routed_base_path],
        ["docker", "run", "--rm", "-v", mount_arg, OSRM_IMAGE, "osrm-customize", routed_base_path],
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            "5000:5000",
            "-v",
            mount_arg,
            OSRM_IMAGE,
            "osrm-routed",
            "--algorithm",
            OSRM_ALGORITHM,
            routed_base_path,
        ],
    ]

    executed: list[dict[str, Any]] = []
    for command in commands:
        try:
            result = docker_runner(command, network_dir)
        except RuntimeError:
            raise
        except (FileNotFoundError, PermissionError, subprocess.CalledProcessError) as exc:
            raise _actionable_docker_error(command, exc) from exc
        executed.append({"command": command, "result": result})

    manifest = {
        "manifest_id": manifest_id,
        "created_at_utc": _now_utc(),
        "source_count": len(source_paths),
        "sources": [
            {
                "path": str(path),
                "staged_path": str(inputs_dir / path.name),
                "url": _geofabrik_url(path.name.split("-")[0].split(".")[0], DEFAULT_YEAR),
                "retrieved_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
            for path, digest in zip(source_paths, source_sha256, strict=True)
        ],
        "merged_pbf_path": str(merged_path),
        "osrm_profile": OSRM_PROFILE,
        "osrm_profile_path": "/opt/car.lua",
        "osrm_version": OSRM_VERSION,
        "osrm_algorithm": OSRM_ALGORITHM,
        "merge_tool": "osmium",
        "merge_image": OSM_MERGE_IMAGE,
        "osrm_image": OSRM_IMAGE,
        "attribution": "© OpenStreetMap contributors",
        "license": "ODbL-1.0",
        "commands": commands,
    }
    _atomic_write_json(manifest, manifest_path)
    return {"manifest": manifest, "manifest_path": str(manifest_path), "commands": executed}


def _status_result(
    route_id: str,
    *,
    status: str,
    error_message: str | None,
    geometry: dict | None = None,
    distance_m: float | None = None,
    duration_s: float | None = None,
) -> dict:
    return {
        "route_id": str(route_id),
        "status": status,
        "distance_m": distance_m,
        "duration_s": duration_s,
        "geometry": geometry,
        "error_message": error_message,
    }


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(str(base_url))
    return parsed.scheme == "http" and (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}


def _normalize_manifest_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _format_signature_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    return f"{float(value):.12f}"


def _derive_route_id(row: pd.Series) -> str:
    route_id = row.get("route_id")
    if route_id is not None and not pd.isna(route_id) and str(route_id).strip():
        return str(route_id)
    home_tract = row.get("home_tract")
    work_tract = row.get("work_tract")
    if home_tract is None or work_tract is None or pd.isna(home_tract) or pd.isna(work_tract):
        raise ValueError("pairs missing route_id and home_tract/work_tract required to derive one")
    return f"{str(home_tract)}__{str(work_tract)}"


def _route_signature(route_id: str, home_lon: object, home_lat: object, work_lon: object, work_lat: object, source_manifest_id: object, network_manifest_id: object, workers: object = None, home_car_share: object = None, commuter_car_miles: object = None) -> str:
    payload = {
        "route_id": str(route_id),
        "home_lon": _format_signature_float(home_lon),
        "home_lat": _format_signature_float(home_lat),
        "work_lon": _format_signature_float(work_lon),
        "work_lat": _format_signature_float(work_lat),
        "source_manifest_id": _normalize_manifest_id(source_manifest_id),
        "network_manifest_id": _normalize_manifest_id(network_manifest_id),
        "workers": _format_signature_float(workers),
        "home_car_share": _format_signature_float(home_car_share),
        "commuter_car_miles": _format_signature_float(commuter_car_miles),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _route_signature_from_row(row: pd.Series) -> str:
    return _route_signature(
        row.get("route_id"),
        row.get("home_lon"),
        row.get("home_lat"),
        row.get("work_lon"),
        row.get("work_lat"),
        row.get("source_manifest_id"),
        row.get("network_manifest_id"),
        row.get("workers"),
        row.get("home_car_share"),
        row.get("commuter_car_miles"),
    )


def _actionable_docker_error(command: list[str], exc: BaseException) -> RuntimeError:
    stdout = ""
    stderr = ""
    if isinstance(exc, subprocess.CalledProcessError):
        stdout = (exc.output or "").strip()
        stderr = (exc.stderr or "").strip()
    detail = stderr or stdout or str(exc)
    message = (
        "Docker failed while preparing the local OSRM network.\n"
        f"Command: {' '.join(command)}\n"
        f"Detail: {detail}"
    )
    if isinstance(exc, FileNotFoundError):
        message = (
            "Docker is required to prepare the local OSRM network. Install Docker and rerun "
            "`python code/build_route_pilot_network.py --year 2022 --states wi il ia mn mi`."
        )
    elif isinstance(exc, PermissionError):
        message += "\nCheck Docker permissions for the current user and rerun."
    elif "Cannot connect to the Docker daemon" in detail:
        message += "\nStart the Docker daemon and rerun."
    elif "permission denied" in detail.lower():
        message += "\nCheck Docker permissions for the current user and rerun."
    elif "pull access denied" in detail.lower() or "manifest unknown" in detail.lower() or "not found" in detail.lower():
        message += "\nVerify that the required Docker images are available and can be pulled on this machine."
    return RuntimeError(message)


def route_pair(
    home_lon: float,
    home_lat: float,
    work_lon: float,
    work_lat: float,
    route_id: str,
    base_url: str,
    session: requests.Session,
) -> dict:
    if not _is_local_base_url(base_url):
        raise ValueError(f"route_pair requires a local OSRM endpoint, got {base_url}")

    url = f"{str(base_url).rstrip('/')}/route/v1/driving/{home_lon},{home_lat};{work_lon},{work_lat}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}

    for attempt in range(len(ROUTE_RETRY_DELAYS) + 1):
        try:
            response = session.get(url, params=params, timeout=ROUTE_TIMEOUT)
            status_code = getattr(response, "status_code", 200)
            if status_code >= 500:
                raise requests.HTTPError(f"status {status_code}", response=response)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("code") in PERMANENT_OSRM_STATUSES:
                return _status_result(
                    route_id,
                    status=str(payload.get("code")),
                    error_message=str(payload.get("message") or payload.get("error") or payload.get("code")),
                )
            try:
                parsed = parse_osrm_route(payload, route_id)
            except ValueError as exc:
                return _status_result(route_id, status="MalformedResponse", error_message=str(exc))
            return {
                "route_id": parsed["route_id"],
                "status": "Ok",
                "distance_m": parsed["distance_m"],
                "duration_s": parsed["duration_s"],
                "geometry": parsed["geometry"],
                "error_message": None,
            }
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= len(ROUTE_RETRY_DELAYS):
                return _status_result(route_id, status="TransportError", error_message=str(exc))
            time.sleep(ROUTE_RETRY_DELAYS[attempt])
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is not None and getattr(response, "status_code", 0) >= 500 and attempt < len(ROUTE_RETRY_DELAYS):
                time.sleep(ROUTE_RETRY_DELAYS[attempt])
                continue
            return _status_result(route_id, status="HttpError", error_message=str(exc))


def _route_geometry_document(result: dict) -> dict[str, Any]:
    return {
        "route_id": result["route_id"],
        "status": result["status"],
        "distance_m": result.get("distance_m"),
        "duration_s": result.get("duration_s"),
        "error_message": result.get("error_message"),
        "geometry": result.get("geometry"),
    }


def _route_row_result(row: pd.Series, base_url: str, session: requests.Session) -> dict:
    """Route one pair or record why it was ineligible without contacting OSRM."""
    route_id = str(row["route_id"])
    routing_eligible = bool(row.get("routing_eligible", True))
    missing_coordinates = any(
        pd.isna(row.get(column)) for column in ("home_lon", "home_lat", "work_lon", "work_lat")
    )
    missing_car_share = "home_car_share" in row.index and pd.isna(row.get("home_car_share"))
    if not routing_eligible or missing_coordinates or missing_car_share:
        status = "MissingCoordinates" if missing_coordinates else "MissingCarShare"
        return _status_result(
            route_id,
            status=status,
            error_message=(
                "pair omitted before routing because endpoint coordinates are missing"
                if missing_coordinates
                else "pair omitted before routing because home car share is missing"
            ),
        )
    return route_pair(
        float(row["home_lon"]),
        float(row["home_lat"]),
        float(row["work_lon"]),
        float(row["work_lat"]),
        route_id,
        base_url,
        session,
    )


def _parallel_route_row_result(args: tuple[pd.Series, str]) -> dict:
    """Thread-pool adapter with one requests session per worker thread."""
    row, base_url = args
    session = getattr(_ROUTE_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _ROUTE_SESSION_LOCAL.session = session
    return _route_row_result(row, base_url, session)


def route_pairs_with_checkpoints(
    pairs: pd.DataFrame,
    cache_path: Path,
    base_url: str,
    session: requests.Session,
    *,
    max_workers: int = 1,
    checkpoint_every: int = 1,
    geometry_callback: Callable[[pd.Series, dict], str | Path | None] | None = None,
) -> pd.DataFrame:
    if int(max_workers) < 1 or int(checkpoint_every) < 1:
        raise ValueError("max_workers and checkpoint_every must be positive")
    max_workers = int(max_workers)
    checkpoint_every = int(checkpoint_every)
    required = {"route_id", "home_lon", "home_lat", "work_lon", "work_lat"}
    if "route_id" not in pairs.columns:
        required = {"home_lon", "home_lat", "work_lon", "work_lat", "home_tract", "work_tract"}
    else:
        required = {"route_id", "home_lon", "home_lat", "work_lon", "work_lat"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing required columns: {sorted(missing)}")

    cache_path = Path(cache_path)
    geometry_dir = cache_path.parent / "geometries"
    if geometry_callback is None:
        geometry_dir.mkdir(parents=True, exist_ok=True)

    frame = pairs.copy()
    frame["route_id"] = frame.apply(_derive_route_id, axis=1)
    frame["route_signature"] = frame.apply(_route_signature_from_row, axis=1)
    frame = frame.sort_values(["route_id", "route_signature"]).drop_duplicates("route_signature", keep="first").reset_index(drop=True)

    existing = _load_checkpoint_rows(cache_path)
    completed_signatures = set(existing["route_signature"].astype(str)) if not existing.empty else set()

    pending_rows: list[pd.Series] = []

    def _checkpoint_batch(batch: list[pd.Series], executor: ThreadPoolExecutor | None) -> None:
        nonlocal existing
        if executor is None:
            results = [_route_row_result(row, base_url, session) for row in batch]
        else:
            results = list(executor.map(_parallel_route_row_result, ((row, base_url) for row in batch)))

        checkpoint_records = []
        route_ids = set()
        for row, result in zip(batch, results, strict=True):
            route_id = str(row["route_id"])
            route_signature = str(row["route_signature"])
            route_ids.add(route_id)
            if geometry_callback is None:
                geometry_path: str | Path | None = geometry_dir / f"{route_id}.geojson"
                _atomic_write_json(_route_geometry_document(result), geometry_path)
            else:
                geometry_path = geometry_callback(row, result)
            checkpoint_records.append(
                {
                    "route_signature": route_signature,
                    "route_id": route_id,
                    "home_lon": row.get("home_lon"),
                    "home_lat": row.get("home_lat"),
                    "work_lon": row.get("work_lon"),
                    "work_lat": row.get("work_lat"),
                    "home_tract": row.get("home_tract"),
                    "work_tract": row.get("work_tract"),
                    "home_county": row.get("home_county"),
                    "work_county": row.get("work_county"),
                    "workers": row.get("workers"),
                    "home_car_share": row.get("home_car_share"),
                    "commuter_car_weight": row.get("commuter_car_weight"),
                    "block_pair_straight_line_miles": row.get("block_pair_straight_line_miles"),
                    "commuter_car_miles": row.get("commuter_car_miles"),
                    "same_tract": row.get("same_tract"),
                    "routing_eligible": bool(row.get("routing_eligible", True)),
                    "omitted_coordinate_worker_weight": row.get("omitted_coordinate_worker_weight"),
                    "omitted_car_share_worker_weight": row.get("omitted_car_share_worker_weight"),
                    "status": result["status"],
                    "distance_m": result.get("distance_m"),
                    "duration_s": result.get("duration_s"),
                    "geometry_path": str(geometry_path) if geometry_path is not None else pd.NA,
                    "error_message": result.get("error_message"),
                    "source_manifest_id": row.get("source_manifest_id"),
                    "network_manifest_id": row.get("network_manifest_id"),
                    "routed_at_utc": _now_utc(),
                }
            )
            completed_signatures.add(route_signature)

        checkpoint_rows = pd.DataFrame.from_records(checkpoint_records, columns=CHECKPOINT_COLUMNS)
        if not existing.empty:
            existing = existing.loc[~existing["route_id"].astype(str).isin(route_ids)].copy()
        if existing.empty:
            existing = checkpoint_rows
        else:
            existing = pd.concat([existing.loc[:, CHECKPOINT_COLUMNS], checkpoint_rows], ignore_index=True)
        _write_checkpoint_rows(existing, cache_path)

    executor = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    try:
        for _, row in frame.iterrows():
            if str(row["route_signature"]) in completed_signatures:
                continue
            pending_rows.append(row)
            if len(pending_rows) >= checkpoint_every:
                _checkpoint_batch(pending_rows, executor)
                pending_rows = []
        if pending_rows:
            _checkpoint_batch(pending_rows, executor)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return _load_checkpoint_rows(cache_path)


def _docker_runner(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError) as exc:
        raise _actionable_docker_error(command, exc) from exc
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--states", nargs="+", default=list(PILOT_STATES))
    parser.add_argument("--cache-dir", type=Path, default=ROUTE_PILOT_CACHE)
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    input_dir = cache_dir / "osm" / str(int(args.year))
    network_dir = cache_dir / "network" / str(int(args.year))
    session = requests.Session()
    pbf_paths = [download_geofabrik_extract(state, int(args.year), input_dir, session) for state in args.states]
    prepare_osrm_network(pbf_paths, network_dir, _docker_runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
