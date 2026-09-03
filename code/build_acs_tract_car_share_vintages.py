"""Build vintage-matched ACS B08301 tract car-share partitions.

Each partition is a state-level, atomically written Parquet file beneath the
national route cache.  The primary loader deliberately keeps unavailable tract
shares missing so downstream flow diagnostics can account for their weight.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from config import DATA_PROC
from build_acs_tract_car_share import parse_legacy_tract_archive
from route_vintages import resolve_acs_window

CAR_SHARE_ROOT = DATA_PROC / "commuting" / "route_national" / "car_share"
TRACT_PREFIX = "1400000US"
TABLE_BASED_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/{year}/"
    "table-based-SF/data/5YRData/acsdt5y{year}-b08301.dat"
)
LEGACY_STATE_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/{year}/data/"
    "5_year_by_state/{state}_Tracts_Block_Groups_Only.zip"
)
NATIONAL_STATE_CODES = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
    "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
)

# The legacy Summary File archive uses concatenated title-case names rather
# than postal abbreviations.  Keep this mapping explicit because DC is named
# ``DistrictOfColumbia`` (not ``DistrictOf-Columbia`` or ``DC``) in the
# published filename.
LEGACY_STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "DistrictOfColumbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "NewHampshire", "34": "NewJersey", "35": "NewMexico",
    "36": "NewYork", "37": "NorthCarolina", "38": "NorthDakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "RhodeIsland",
    "45": "SouthCarolina", "46": "SouthDakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "WestVirginia", "55": "Wisconsin", "56": "Wyoming",
}
# Census's fixed-width archives use a vintage-specific sequence number for
# each table.  In particular, B08301 is sequence 0028 in 2015 but 0027 in
# 2020.  Resolve the location from the publisher's lookup rather than
# silently parsing a same-shaped, unrelated sequence.
LEGACY_LOOKUP_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/{year}/"
    "documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt"
)
# The fixed-width/sequence-file Summary File archives cover the 2009--2020
# vintages.  The table-based B08301 files begin with the 2021 vintage.
LEGACY_VINTAGE_RANGE = range(2009, 2021)
TABLE_BASED_VINTAGE_RANGE = range(2021, 2025)


def build_car_share_frame(
    raw: pd.DataFrame, *, return_diagnostics: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Normalize one published B08301 table to valid tract-level car shares."""
    required = {"GEO_ID", "B08301_E001", "B08301_E002"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"B08301 table missing required columns: {sorted(missing)}")

    frame = raw.loc[:, ["GEO_ID", "B08301_E001", "B08301_E002"]].copy()
    geoid = frame["GEO_ID"].astype("string").str.strip()
    valid_tract = geoid.str.startswith(TRACT_PREFIX, na=False) & geoid.str.len().eq(len(TRACT_PREFIX) + 11)
    workers = pd.to_numeric(frame["B08301_E001"], errors="coerce")
    cars = pd.to_numeric(frame["B08301_E002"], errors="coerce")
    malformed = ~valid_tract | workers.isna() | cars.isna()
    zero_workers = valid_tract & workers.notna() & workers.le(0)
    share = cars / workers.where(workers.ne(0))
    out_of_range = valid_tract & workers.gt(0) & cars.notna() & ~share.between(0, 1, inclusive="both")
    keep = valid_tract & workers.gt(0) & cars.notna() & share.between(0, 1, inclusive="both")

    result = pd.DataFrame({
        "tract": geoid.loc[keep].str[-11:].to_numpy(),
        "total_workers": workers.loc[keep].astype(float).to_numpy(),
        "car_total": cars.loc[keep].astype(float).to_numpy(),
        "car_share": share.loc[keep].astype(float).to_numpy(),
    }).sort_values("tract").drop_duplicates("tract", keep="last").reset_index(drop=True)
    diagnostics = {
        "input_rows": int(len(frame)),
        "retained_rows": int(len(result)),
        "zero_worker_rows": int(zero_workers.sum()),
        "malformed_rows": int(malformed.sum()),
        "out_of_range_share_rows": int(out_of_range.sum()),
        "omitted_rows": int((~keep).sum()),
    }
    return (result, diagnostics) if return_diagnostics else result


@lru_cache(maxsize=None)
def resolve_legacy_table_location(
    vintage: int, table_id: str = "B08301"
) -> tuple[str, int]:
    """Return ``(sequence_number, start_position)`` from Census metadata."""
    year = int(vintage)
    url = LEGACY_LOOKUP_URL.format(year=year)
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    raw = response.content.decode("latin-1")
    lookup = pd.read_csv(io.StringIO(raw), dtype=str, low_memory=False)
    lookup.columns = [str(column).strip() for column in lookup.columns]
    required = {"Table ID", "Sequence Number", "Start Position"}
    missing = required.difference(lookup.columns)
    if missing:
        raise ValueError(f"ACS {year} lookup missing columns: {sorted(missing)}")
    matches = lookup[lookup["Table ID"].astype(str).str.strip().eq(str(table_id))]
    matches = matches[matches["Sequence Number"].notna() & matches["Start Position"].notna()]
    if matches.empty:
        raise ValueError(f"{table_id} not present in ACS {year} sequence lookup")
    sequence = str(matches.iloc[0]["Sequence Number"]).strip().zfill(4)
    start_position = int(str(matches.iloc[0]["Start Position"]).strip())
    return sequence, start_position


def _atomic_write_parquet(
    frame: pd.DataFrame, path: Path, *, provenance: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.parquet", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"route_national.provenance"] = json.dumps(
            provenance, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_state_partition(
    raw: pd.DataFrame, *, vintage: int | str, state: str, source_url: str, source_bytes: bytes,
    cache_dir: Path = CAR_SHARE_ROOT, window_start: int | None = None, window_end: int | None = None,
) -> tuple[Path, dict[str, int]]:
    """Write one state/vintage partition and provenance sidecar atomically."""
    vintage_label = str(vintage)
    state_code = str(state).upper()
    frame, diagnostics = build_car_share_frame(raw, return_diagnostics=True)
    partition_dir = Path(cache_dir) / f"acs_{vintage_label}"
    partition = partition_dir / f"state={state_code}.parquet"
    metadata_path = partition_dir / "metadata.json"
    metadata = {
        "schema_version": "route_national.acs_car_share.v1", "acs_vintage": vintage_label,
        "window_start": int(window_start if window_start is not None else int(vintage_label) - 4),
        "window_end": int(window_end if window_end is not None else vintage_label),
        "expected_states": list(NATIONAL_STATE_CODES),
    }
    # This metadata is vintage-level only.  State-specific provenance is kept
    # in the atomically replaced Parquet file, preventing a shared manifest
    # from drifting from a partition when writers run concurrently.
    _atomic_write_json(metadata, metadata_path)
    provenance = {
        "schema_version": "route_national.acs_car_share.v1", "acs_vintage": vintage_label,
        "window_start": metadata["window_start"], "window_end": metadata["window_end"],
        "state": state_code, "url": source_url,
        "sha256": hashlib.sha256(source_bytes).hexdigest(), "bytes": len(source_bytes),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_parquet(frame, partition, provenance=provenance)
    return partition, diagnostics


def load_car_share_for_analysis_year(
    year: int, cache_dir: Path, *, expected_states: tuple[str, ...] = NATIONAL_STATE_CODES
) -> tuple[pd.Series, str, dict[str, float]]:
    """Resolve a cached ACS window and return its tract-indexed share series."""
    root = Path(cache_dir)
    candidates: list[tuple[int, int, str, Path, dict[str, object], set[str]]] = []
    for directory in sorted(root.glob("acs_*")):
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        try:
            start, end = int(metadata["window_start"]), int(metadata["window_end"])
            vintage = str(metadata["acs_vintage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid ACS metadata at {metadata_path}") from exc
        present_states = {path.stem.removeprefix("state=").upper() for path in directory.glob("state=*.parquet")}
        if present_states:
            candidates.append((start, end, vintage, directory, metadata, present_states))
    choice = resolve_acs_window(int(year), [(start, end, vintage) for start, end, vintage, _, _, _ in candidates])
    if choice.status == "unavailable":
        raise FileNotFoundError(f"no valid ACS tract car-share partitions beneath {root}")
    selected = next(
        item for item in candidates
        if (item[0], item[1], item[2]) == (choice.window_start, choice.window_end, choice.vintage)
    )
    missing_states = sorted(set(expected_states).difference(selected[5]))
    if missing_states:
        raise FileNotFoundError(
            "incomplete ACS tract car-share coverage for "
            f"{selected[2]}: missing state partitions {missing_states}"
        )
    frames = [pd.read_parquet(path, columns=["tract", "car_share"]) for path in sorted(selected[3].glob("state=*.parquet"))]
    merged = pd.concat(frames, ignore_index=True)
    merged["tract"] = merged["tract"].astype("string").str.zfill(11)
    merged["car_share"] = pd.to_numeric(merged["car_share"], errors="coerce")
    merged = merged[merged["car_share"].between(0, 1, inclusive="both")].drop_duplicates("tract", keep="last")
    series = merged.set_index("tract")["car_share"].sort_index()
    diagnostics = {
        "acs_window_start": float(selected[0]), "acs_window_end": float(selected[1]),
        "selected_midpoint_gap": float(choice.gap or 0), "tracts_with_share": float(len(series)),
        "missing_share_fill_count": 0.0,
        "expected_state_partitions": float(len(expected_states)),
        "present_state_partitions": float(len(selected[5])),
    }
    return series, selected[2], diagnostics


def fetch_table_based_state(vintage: int, state_fips: str) -> tuple[pd.DataFrame, bytes, str]:
    """Download one table-based ACS file and retain one published state slice."""
    url = TABLE_BASED_URL.format(year=int(vintage))
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    raw_bytes = response.content
    raw = pd.read_csv(io.StringIO(raw_bytes.decode("latin-1")), sep="|", dtype=str, low_memory=False)
    prefix = f"{TRACT_PREFIX}{str(state_fips).zfill(2)}"
    return raw.loc[raw["GEO_ID"].astype(str).str.startswith(prefix)].copy(), raw_bytes, url


def fetch_legacy_state(
    vintage: int, state_name: str, *, sequence_number: str | None = None,
    start_position: int | None = None,
) -> tuple[pd.DataFrame, bytes, str]:
    """Download a legacy archive using its vintage-specific table location."""
    if sequence_number is None or start_position is None:
        resolved_sequence, resolved_start = resolve_legacy_table_location(vintage)
        sequence_number = resolved_sequence if sequence_number is None else sequence_number
        start_position = resolved_start if start_position is None else start_position
    url = LEGACY_STATE_URL.format(year=int(vintage), state=str(state_name))
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    payload = response.content
    parser_kwargs = {"sequence_number": str(sequence_number)}
    if int(start_position) != 157:
        parser_kwargs["start_position"] = int(start_position)
    parsed = parse_legacy_tract_archive(payload, **parser_kwargs)
    if parsed is None:
        raise ValueError(f"could not parse B08301 legacy archive: {url}")
    raw = pd.DataFrame({
        "GEO_ID": TRACT_PREFIX + parsed["tract"].astype("string").str.zfill(11),
        "B08301_E001": parsed["total_workers"],
        "B08301_E002": parsed["car_total"],
    })
    return raw, payload, url


def legacy_state_name(state_fips: str) -> str:
    """Return the exact legacy archive name for a two-digit state FIPS."""
    key = str(state_fips).strip().zfill(2)
    try:
        return LEGACY_STATE_NAMES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported state FIPS for ACS legacy archive: {state_fips}") from exc


def resolve_source(vintage: int, source: str = "auto") -> str:
    """Select the published ACS source family for one tract vintage."""
    year = int(vintage)
    choice = str(source).strip().lower()
    if choice not in {"auto", "legacy", "table"}:
        raise ValueError("ACS source must be auto, legacy, or table")
    if choice == "auto":
        if year in LEGACY_VINTAGE_RANGE:
            return "legacy"
        if year in TABLE_BASED_VINTAGE_RANGE:
            return "table"
        raise ValueError(
            f"no supported ACS tract source for vintage {year}; expected 2009-2024"
        )
    if choice == "legacy" and year not in LEGACY_VINTAGE_RANGE:
        raise ValueError("legacy ACS tract archives are supported for vintages 2009-2020")
    if choice == "table" and year not in TABLE_BASED_VINTAGE_RANGE:
        raise ValueError("table-based ACS tract files are supported for vintages 2021-2024")
    return choice


def fetch_state_for_vintage(
    vintage: int, state_fips: str, *, source: str = "auto"
) -> tuple[pd.DataFrame, bytes, str]:
    """Fetch one state using the archive family that contains ``vintage``."""
    selected = resolve_source(vintage, source)
    if selected == "legacy":
        return fetch_legacy_state(vintage, legacy_state_name(state_fips))
    return fetch_table_based_state(vintage, state_fips)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vintage", type=int)
    parser.add_argument("state_fips", help="two-digit state FIPS")
    parser.add_argument("--cache-dir", type=Path, default=CAR_SHARE_ROOT)
    parser.add_argument(
        "--source", choices=("auto", "legacy", "table"), default="auto",
        help="ACS archive family; auto selects legacy 2009-2020 or table-based 2021-2024",
    )
    args = parser.parse_args(argv)
    raw, payload, url = fetch_state_for_vintage(
        args.vintage, args.state_fips, source=args.source
    )
    partition, diagnostics = write_state_partition(
        raw, vintage=args.vintage, state=args.state_fips, cache_dir=args.cache_dir,
        source_url=url, source_bytes=payload,
    )
    print(json.dumps({"partition": str(partition), **diagnostics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
