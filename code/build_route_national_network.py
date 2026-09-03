"""Build a reproducible OSRM network from all U.S. states and DC.

The pilot network remains intentionally separate.  This module shares its
download/checksum and Docker conventions while using its own national manifest
and year-specific artifact names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import requests

from build_route_pilot_network import (
    OSRM_ALGORITHM,
    OSRM_IMAGE,
    OSRM_PROFILE,
    OSM_MERGE_IMAGE,
    _actionable_docker_error,
    _docker_runner,
    download_bulk_file,
    sha256_file,
)
from config import ROUTE_PILOT_CACHE

DEFAULT_YEAR = 2022
MANIFEST_SCHEMA_VERSION = "route_national.network.v1"
NATIONAL_STATES: tuple[str, ...] = (
    "ak", "al", "ar", "az", "ca", "co", "ct", "dc", "de", "fl", "ga",
    "hi", "ia", "id", "il", "in", "ks", "ky", "la", "ma", "md", "me",
    "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm",
    "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx",
    "ut", "va", "vt", "wa", "wi", "wv", "wy",
)
_STATE_SLUGS = {
    "ak": "alaska", "al": "alabama", "ar": "arkansas", "az": "arizona", "ca": "california",
    "co": "colorado", "ct": "connecticut", "dc": "district-of-columbia", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "ia": "iowa", "id": "idaho",
    "il": "illinois", "in": "indiana", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "ma": "massachusetts", "md": "maryland", "me": "maine", "mi": "michigan", "mn": "minnesota",
    "mo": "missouri", "ms": "mississippi", "mt": "montana", "nc": "north-carolina", "nd": "north-dakota",
    "ne": "nebraska", "nh": "new-hampshire", "nj": "new-jersey", "nm": "new-mexico", "nv": "nevada",
    "ny": "new-york", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode-island", "sc": "south-carolina", "sd": "south-dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "va": "virginia", "vt": "vermont", "wa": "washington", "wi": "wisconsin",
    "wv": "west-virginia", "wy": "wyoming",
}
GEOFABRIK_BASE = "https://download.geofabrik.de/north-america/us"


def _validate_states(states: Sequence[str]) -> list[str]:
    normalized = [str(state).strip().lower() for state in states]
    invalid = sorted(set(normalized) - set(NATIONAL_STATES))
    if invalid:
        raise ValueError(f"unsupported national state(s): {', '.join(invalid)}; expected 50 states plus DC")
    if len(set(normalized)) != len(normalized):
        raise ValueError("national states must not contain duplicates")
    return sorted(normalized)


def _geofabrik_url(state: str, year: int) -> str:
    state = str(state).lower()
    if state not in _STATE_SLUGS:
        raise ValueError(f"unsupported national state: {state}")
    return f"{GEOFABRIK_BASE}/{_STATE_SLUGS[state]}-{int(year) % 100:02d}0101.osm.pbf"


def build_network_manifest(
    source_records: Sequence[Mapping[str, object]], *, network_year: int, network_id: str | None = None
) -> dict[str, object]:
    """Return a deterministic, provenance-rich national network manifest."""
    year = int(network_year)
    records = []
    for source in source_records:
        state = str(source.get("state", "")).strip().lower()
        _validate_states([state])
        record = dict(source)
        record["state"] = state
        record.setdefault("url", _geofabrik_url(state, year))
        records.append(record)
    records.sort(key=lambda row: (str(row["state"]), str(row.get("path", ""))))
    if network_id is None:
        identity = "|".join(
            f"{row['state']}:{row.get('sha256', '')}" for row in records
        ) + f"|{OSRM_PROFILE}|{OSRM_ALGORITHM}|{OSRM_IMAGE}|{OSM_MERGE_IMAGE}|{year}"
        network_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    if not str(network_id).strip():
        raise ValueError("network_id must not be empty")
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": str(network_id),
        "network_year": year,
        "states": [str(row["state"]) for row in records],
        "scope": "national" if set(row["state"] for row in records) == set(NATIONAL_STATES) else "scoped",
        "source_count": len(records),
        "sources": records,
        "osrm_profile": OSRM_PROFILE,
        "osrm_algorithm": OSRM_ALGORITHM,
        "osrm_image": OSRM_IMAGE,
        "merge_image": OSM_MERGE_IMAGE,
        "attribution": "© OpenStreetMap contributors",
        "license": "ODbL-1.0",
    }


def download_geofabrik_extract(state: str, year: int, cache_dir: Path, session: requests.Session) -> Path:
    state = _validate_states([state])[0]
    target = Path(cache_dir) / f"{state}-{int(year)}.osm.pbf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        sidecar = target.with_name(target.name + ".manifest.json")
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            valid = (
                record.get("url") == _geofabrik_url(state, year)
                and int(record.get("bytes", -1)) == target.stat().st_size
                and record.get("sha256") == sha256_file(target)
                and bool(record.get("retrieved_at_utc"))
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise RuntimeError(f"cache failed manifest/checksum validation at {target}; remove stale extract and sidecar, then rerun")
        return target
    try:
        digest = download_bulk_file(session, _geofabrik_url(state, year), target, timeout=1200)
    except Exception as exc:
        raise RuntimeError(f"missing Geofabrik extract for {state} ({year}) at {_geofabrik_url(state, year)}: {exc}") from exc
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"missing Geofabrik extract for {state} ({year}): downloader produced no file at {target}")
    _atomic_write_json({"path": str(target), "url": _geofabrik_url(state, year), "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "bytes": target.stat().st_size, "sha256": digest}, target.with_name(target.name + ".manifest.json"))
    return target


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _network_output_checksums(network_dir: Path) -> dict[str, str]:
    outputs = {}
    for path in sorted(Path(network_dir).rglob("*")):
        if not path.is_file() or "inputs" in path.parts or path.name.startswith("."):
            continue
        outputs[str(path.relative_to(network_dir))] = sha256_file(path)
    return outputs


def _outputs_valid(network_dir: Path, outputs: object) -> bool:
    if not isinstance(outputs, dict) or not outputs:
        return False
    root = Path(network_dir).resolve()
    for relative, digest in outputs.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        path = Path(network_dir) / relative_path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            return False
        if not resolved.is_file() or sha256_file(resolved) != digest:
            return False
    return True


def prepare_national_osrm_network(
    states: Sequence[str], network_year: int, cache_dir: Path, docker_runner: Callable, *, allow_partial: bool = False
) -> dict[str, object]:
    """Download all requested extracts, merge them, and prepare the OSRM graph."""
    ordered_states = _validate_states(states)
    if not allow_partial and set(ordered_states) != set(NATIONAL_STATES):
        raise ValueError("national graph requires all 51 states; pass allow_partial=True for a scoped graph")
    year = int(network_year)
    root = Path(cache_dir)
    input_dir = root / "osm" / str(year)
    network_dir = root / "network" / str(year)
    session = requests.Session()
    paths = [download_geofabrik_extract(state, year, input_dir, session) for state in ordered_states]
    missing = [str(path) for path in paths if not Path(path).is_file() or Path(path).stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"missing national Geofabrik extracts: {', '.join(missing)}")
    network_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = network_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    records = []
    for state, path in zip(ordered_states, paths, strict=True):
        source = Path(path)
        digest = sha256_file(source)
        try:
            retrieved_at = json.loads(source.with_name(source.name + ".manifest.json").read_text(encoding="utf-8"))["retrieved_at_utc"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"missing retrieval provenance for national extract {source}") from exc
        destination = inputs_dir / source.name
        if not destination.exists() or sha256_file(destination) != digest:
            temporary = destination.with_name(f".{destination.name}.staging")
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"staged national extract checksum mismatch for {source}")
            os.replace(temporary, destination)
        staged.append(destination)
        records.append({"state": state, "path": str(source), "staged_path": str(destination), "url": _geofabrik_url(state, year), "retrieved_at_utc": retrieved_at, "bytes": source.stat().st_size, "sha256": digest})
    merged_name = f"national-network-{year}.osm.pbf"
    base = f"/data/national-network-{year}.osrm"
    mount = f"{network_dir.resolve()}:/data"
    commands = [
        ["docker", "run", "--rm", "-v", mount, OSM_MERGE_IMAGE, "osmium", "merge", "--overwrite", *[f"/data/inputs/{p.name}" for p in staged], "-o", f"/data/{merged_name}"],
        ["docker", "run", "--rm", "-v", mount, OSRM_IMAGE, "osrm-extract", "-p", "/opt/car.lua", f"/data/{merged_name}"],
        ["docker", "run", "--rm", "-v", mount, OSRM_IMAGE, "osrm-partition", base],
        ["docker", "run", "--rm", "-v", mount, OSRM_IMAGE, "osrm-customize", base],
        ["docker", "run", "--rm", "-d", "-p", "5000:5000", "-v", mount, OSRM_IMAGE, "osrm-routed", "--algorithm", OSRM_ALGORITHM, base],
    ]
    executed = []
    stage_path = network_dir / ".stage_manifest.json"
    command_ids = [hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest() for command in commands]
    completed = {}
    if stage_path.exists():
        try:
            prior = json.loads(stage_path.read_text())
            if not isinstance(prior, Mapping):
                raise ValueError("stage manifest root must be an object")
            candidate = prior.get("completed", {})
            if prior.get("network_id") == build_network_manifest(records, network_year=year)["manifest_id"] and prior.get("commands") == command_ids and isinstance(candidate, dict):
                completed = candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            completed = {}
    for index, command in enumerate(commands):
        marker = completed.get(str(index))
        if isinstance(marker, dict) and marker.get("id") == command_ids[index] and _outputs_valid(network_dir, marker.get("outputs")):
            executed.append({"command": command, "result": "reused"})
            continue
        try:
            executed.append({"command": command, "result": docker_runner(command, network_dir)})
        except Exception as exc:
            if isinstance(exc, (FileNotFoundError, PermissionError, subprocess.CalledProcessError)):
                raise _actionable_docker_error(command, exc) from exc
            raise RuntimeError(f"Docker failed while preparing the national OSRM network. Command: {' '.join(command)}. Detail: {exc}") from exc
        outputs = _network_output_checksums(network_dir)
        if index < 4 and not outputs:
            raise RuntimeError(f"Docker stage {index} completed without producing a verifiable output")
        completed[str(index)] = {"id": command_ids[index], "outputs": outputs}
        _atomic_write_json({"network_id": build_network_manifest(records, network_year=year)["manifest_id"], "commands": command_ids, "completed": completed}, stage_path)
    manifest = build_network_manifest(records, network_year=year)
    manifest.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "merged_pbf_path": str(network_dir / merged_name), "commands": commands})
    manifest_path = network_dir / "network_manifest.json"
    _atomic_write_json(manifest, manifest_path)
    return {"manifest": manifest, "manifest_path": str(manifest_path), "commands": executed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--states", nargs="+", default=list(NATIONAL_STATES))
    parser.add_argument("--cache-dir", type=Path, default=ROUTE_PILOT_CACHE)
    args = parser.parse_args()
    prepare_national_osrm_network(args.states, args.year, args.cache_dir, _docker_runner)
