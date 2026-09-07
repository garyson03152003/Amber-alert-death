"""Route national flow partitions through a local OSRM server in bounded chunks.

The flow builder writes one parquet file per workplace state.  This runner
never loads that whole file into memory: it iterates Arrow record batches,
routes one batch, and commits compact audit/segment artifacts before moving to
the next batch.  A completed chunk is identified by its checksum manifest and
is skipped on restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq
import requests

from build_route_national_segments import route_partition_to_segments
from build_route_pilot_county_miles import load_county_boundaries
from build_route_pilot_network import ROUTE_TIMEOUT, route_pair
from build_route_national_network import NATIONAL_STATES
from route_exposure_core import build_county_spatial_index


_ROUTE_SESSION_LOCAL = threading.local()
_PARTITION_RE = re.compile(
    r"analysis_year=(?P<analysis>\d+)/lodes_source_year=(?P<source>\d+)/work_state=(?P<state>[a-z]{2})\.parquet$"
)


def wait_for_osrm(
    base_url: str,
    *,
    session: requests.Session | None = None,
    timeout_seconds: float = 900.0,
    poll_seconds: float = 2.0,
) -> None:
    """Wait until the local OSRM server is accepting requests.

    ``osrm-routed`` is started detached by the network builder and may take
    several minutes to load a national graph after its container is created.
    Waiting here prevents the first route batch from being recorded as
    transport failures while the server is still starting.
    """
    timeout_seconds = float(timeout_seconds)
    poll_seconds = float(poll_seconds)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be nonnegative")
    client = session or requests.Session()
    url = (
        f"{str(base_url).rstrip('/')}/route/v1/driving/"
        "-89.4,43.0;-89.3,43.1?overview=false"
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            response = client.get(url, timeout=ROUTE_TIMEOUT)
            status_code = int(getattr(response, "status_code", 0))
            if 0 < status_code < 500:
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"OSRM did not become ready at {base_url} within {timeout_seconds:g}s")
        time.sleep(min(poll_seconds, remaining))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def iter_flow_batches(path: Path, *, batch_rows: int = 25_000) -> Iterator[pd.DataFrame]:
    """Yield pandas batches from a flow parquet without reading it all."""
    if int(batch_rows) <= 0:
        raise ValueError("batch_rows must be positive")
    parquet = pq.ParquetFile(Path(path))
    for batch in parquet.iter_batches(batch_size=int(batch_rows), use_threads=True):
        yield batch.to_pandas()


def write_chunk_manifest(
    path: Path,
    *,
    analysis_year: int,
    source_year: int,
    work_state: str,
    chunk_index: int,
    segment_path: Path,
    audit_path: Path,
    route_count: int,
) -> Path:
    """Atomically record a completed chunk and both artifact checksums."""
    segment_path = Path(segment_path)
    audit_path = Path(audit_path)
    if not segment_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("chunk artifacts must exist before writing a manifest")
    payload = {
        "schema_version": "route_national.chunk_manifest.v1",
        "status": "success",
        "analysis_year": int(analysis_year),
        "lodes_source_year": int(source_year),
        "work_state": str(work_state).strip().lower(),
        "chunk_index": int(chunk_index),
        "route_count": int(route_count),
        "segment_path": str(segment_path.resolve()),
        "segment_sha256": _sha256(segment_path),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "completed_at_utc": _utc_now(),
    }
    return _atomic_json(payload, Path(path))


def _chunk_is_valid(path: Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        segment = Path(str(payload["segment_path"]))
        audit = Path(str(payload["audit_path"]))
        return (
            payload.get("status") == "success"
            and segment.is_file()
            and audit.is_file()
            and payload.get("segment_sha256") == _sha256(segment)
            and payload.get("audit_sha256") == _sha256(audit)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _local_route_client(
    home_lon: float,
    home_lat: float,
    work_lon: float,
    work_lat: float,
    route_id: str,
    base_url: str,
) -> dict[str, object]:
    session = getattr(_ROUTE_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _ROUTE_SESSION_LOCAL.session = session
    return route_pair(
        home_lon,
        home_lat,
        work_lon,
        work_lat,
        route_id,
        base_url,
        session,
    )


def _partition_metadata(path: Path) -> tuple[int, int, str]:
    match = _PARTITION_RE.search(str(Path(path)))
    if match is None:
        raise ValueError(
            "flow partition must end with analysis_year=YYYY/"
            "lodes_source_year=YYYY/work_state=xx.parquet"
        )
    return int(match.group("analysis")), int(match.group("source")), match.group("state")


def route_flow_partition_chunks(
    partition_path: Path,
    county_boundaries: pd.DataFrame,
    *,
    output_root: Path,
    network_manifest_id: str,
    base_url: str = "http://127.0.0.1:5000",
    batch_rows: int = 25_000,
    route_workers: int = 8,
    checkpoint_every: int = 5_000,
    geometry_sample_rate: float = 0.001,
    allocation_workers: int = 1,
    chunk_start: int = 0,
    chunk_end: int | None = None,
    spatial_index: dict[str, object] | None = None,
) -> list[Path]:
    """Route one state partition, returning valid per-chunk manifests."""
    if int(chunk_start) < 0:
        raise ValueError("chunk_start must be nonnegative")
    if chunk_end is not None and int(chunk_end) <= int(chunk_start):
        raise ValueError("chunk_end must be greater than chunk_start")
    analysis_year, source_year, work_state = _partition_metadata(Path(partition_path))
    output_dir = (
        Path(output_root)
        / f"analysis_year={analysis_year}"
        / f"lodes_source_year={source_year}"
        / f"work_state={work_state}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    for chunk_index, pairs in enumerate(iter_flow_batches(partition_path, batch_rows=batch_rows)):
        if chunk_index < int(chunk_start):
            continue
        if chunk_end is not None and chunk_index >= int(chunk_end):
            break
        chunk_dir = output_dir / f"chunk={chunk_index:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = chunk_dir / "chunk_manifest.json"
        if _chunk_is_valid(manifest_path):
            manifests.append(manifest_path)
            continue
        audits, _segments = route_partition_to_segments(
            pairs,
            county_boundaries,
            base_url=base_url,
            cache_dir=chunk_dir,
            route_client=_local_route_client,
            network_manifest_id=network_manifest_id,
            spatial_index=spatial_index,
            max_workers=int(route_workers),
            checkpoint_every=int(checkpoint_every),
            geometry_sample_rate=float(geometry_sample_rate),
            allocation_workers=int(allocation_workers),
        )
        manifest_path = write_chunk_manifest(
            manifest_path,
            analysis_year=analysis_year,
            source_year=source_year,
            work_state=work_state,
            chunk_index=chunk_index,
            segment_path=chunk_dir / "county_segments.parquet",
            audit_path=chunk_dir / "route_audits.parquet",
            route_count=len(audits),
        )
        manifests.append(manifest_path)
        print(
            json.dumps(
                {
                    "event": "chunk_complete",
                    "analysis_year": analysis_year,
                    "lodes_source_year": source_year,
                    "work_state": work_state,
                    "chunk_index": chunk_index,
                    "route_count": len(audits),
                    "manifest": str(manifest_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return manifests


def _load_network_manifest(path: Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest_id = str(payload.get("manifest_id") or "").strip()
    if not manifest_id:
        raise ValueError(f"network manifest missing manifest_id: {path}")
    return manifest_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/commuting/route_national/flows",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/commuting/route_national/segments_chunks",
    )
    parser.add_argument("--county-boundaries-path", type=Path, required=True)
    parser.add_argument("--network-manifest", type=Path, required=True)
    parser.add_argument("--analysis-year", type=int, default=2013)
    parser.add_argument("--states", nargs="+", default=list(NATIONAL_STATES))
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--batch-rows", type=int, default=25_000)
    parser.add_argument("--route-workers", type=int, default=8)
    parser.add_argument(
        "--allocation-workers",
        type=int,
        default=1,
        help="parallel county-mile allocation workers per route batch (default: 1)",
    )
    parser.add_argument(
        "--allocation-crs",
        default=None,
        help="optional fixed CRS for reusable county/route projection (e.g. EPSG:5070)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--geometry-sample-rate", type=float, default=0.001)
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="inclusive chunk index to start for each selected state (default: 0)",
    )
    parser.add_argument(
        "--chunk-end",
        type=int,
        default=None,
        help="exclusive chunk index at which to stop (default: all chunks)",
    )
    parser.add_argument("--osrm-ready-timeout", type=float, default=900.0)
    parser.add_argument("--osrm-poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    states = sorted({str(state).strip().lower() for state in args.states})
    invalid = sorted(set(states) - set(NATIONAL_STATES))
    if invalid:
        raise ValueError(f"unsupported national states: {invalid}")
    network_manifest_id = _load_network_manifest(args.network_manifest)
    wait_for_osrm(
        args.base_url,
        timeout_seconds=float(args.osrm_ready_timeout),
        poll_seconds=float(args.osrm_poll_seconds),
    )
    boundaries = load_county_boundaries(args.county_boundaries_path)
    spatial_index = build_county_spatial_index(
        boundaries,
        projection_crs=args.allocation_crs,
    )
    flow_root = Path(args.flow_root)
    partition_root = flow_root / "partitions" / f"analysis_year={int(args.analysis_year)}"
    partitions = []
    for path in sorted(partition_root.glob("lodes_source_year=*/work_state=*.parquet")):
        _analysis, _source, state = _partition_metadata(path)
        if state in states:
            partitions.append(path)
    if not partitions:
        raise FileNotFoundError(f"no requested flow partitions found below {partition_root}")

    run_status = Path(args.output_root) / "execution" / f"national_route_chunks_{int(args.analysis_year)}.json"
    _atomic_json(
        {
            "status": "running",
            "started_at_utc": _utc_now(),
            "analysis_year": int(args.analysis_year),
            "states": states,
            "network_manifest_id": network_manifest_id,
            "batch_rows": int(args.batch_rows),
            "route_workers": int(args.route_workers),
            "allocation_workers": int(args.allocation_workers),
            "allocation_crs": args.allocation_crs,
            "checkpoint_every": int(args.checkpoint_every),
            "geometry_sample_rate": float(args.geometry_sample_rate),
            "chunk_start": int(args.chunk_start),
            "chunk_end": args.chunk_end,
        },
        run_status,
    )
    completed = 0
    try:
        for partition in partitions:
            manifests = route_flow_partition_chunks(
                partition,
                boundaries,
                output_root=args.output_root,
                network_manifest_id=network_manifest_id,
                base_url=args.base_url,
                batch_rows=int(args.batch_rows),
                route_workers=int(args.route_workers),
                checkpoint_every=int(args.checkpoint_every),
                geometry_sample_rate=float(args.geometry_sample_rate),
                allocation_workers=int(args.allocation_workers),
                chunk_start=int(args.chunk_start),
                chunk_end=args.chunk_end,
                spatial_index=spatial_index,
            )
            completed += len(manifests)
            _atomic_json(
                {
                    "status": "running",
                    "updated_at_utc": _utc_now(),
                    "analysis_year": int(args.analysis_year),
                    "states": states,
                    "partitions_completed": completed,
                    "last_partition": str(partition),
                },
                run_status,
            )
    except BaseException as exc:
        _atomic_json(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "analysis_year": int(args.analysis_year),
                "states": states,
                "partitions_completed": completed,
                "error": repr(exc),
            },
            run_status,
        )
        raise
    _atomic_json(
        {
            "status": "completed",
            "finished_at_utc": _utc_now(),
            "analysis_year": int(args.analysis_year),
            "states": states,
            "partitions_completed": completed,
            "network_manifest_id": network_manifest_id,
        },
        run_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
