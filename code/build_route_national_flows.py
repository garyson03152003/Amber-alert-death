"""Build resumable, workplace-state LODES tract-pair flow partitions.

Unlike the five-state pilot, this builder keeps every home state represented in
each workplace-state LODES source.  It writes a separate artifact for each
analysis year, selected LODES vintage, and workplace state so source vintages
can never be mixed silently downstream.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from build_acs_tract_car_share_vintages import CAR_SHARE_ROOT, load_car_share_for_analysis_year
from build_route_pilot_flows import (
    LODES_SOURCE_NAME,
    _load_lodes_crosswalks,
    _parse_flow_path,
    download_lodes_input,
    lodes_url,
    manifest_record,
)
from config import DATA_PROC
from route_exposure_core import prepare_crosswalk_for_pairs, weighted_tract_pairs
from route_vintages import resolve_nearest_year, write_vintage_manifest


NATIONAL_FLOW_ROOT = DATA_PROC / "commuting" / "route_national" / "flows"
# LODES8 releases differ by state.  The downloader validates actual
# availability rather than treating this catalogue as a promise.
LODES_CANDIDATE_YEARS = tuple(range(2002, 2024))
FIPS_TO_STATE = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co",
    "09": "ct", "10": "de", "11": "dc", "12": "fl", "13": "ga", "15": "hi",
    "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks", "21": "ky",
    "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn",
    "28": "ms", "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
    "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd", "39": "oh",
    "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd",
    "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa",
    "54": "wv", "55": "wi", "56": "wy",
}
NATIONAL_STATES = tuple(sorted(FIPS_TO_STATE.values()))
PARTITION_SCHEMA_VERSION = "route_national.lodes_flow.v2"
STREAMING_INPUT_BYTES = 20_000_000


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "home_tract", "work_tract", "home_county", "work_county", "workers",
        "valid_endpoint_workers", "missing_endpoint_workers", "home_lat", "home_lon",
        "work_lat", "work_lon", "home_car_share", "missing_home_car_share_workers",
        "block_pair_straight_line_miles", "home_state_fips", "work_state_fips",
        "home_state", "work_state", "same_tract", "route_id", "commuter_car_weight",
        "commuter_car_miles", "routing_eligible", "omitted_coordinate_worker_weight",
        "omitted_car_share_worker_weight", "analysis_year", "lodes_source_year",
        "lodes_source_name", "lodes_file_types", "source_partition_id",
    ])


def _positive_flow_metrics(flows: pd.DataFrame) -> tuple[int, float]:
    if flows.empty or "S000" not in flows:
        return 0, 0.0
    weights = pd.to_numeric(flows["S000"], errors="coerce")
    valid = weights.notna() & weights.gt(0)
    return int(valid.sum()), float(weights.loc[valid].sum())


def _flow_file_types(flows: pd.DataFrame) -> str:
    if "file_type" not in flows:
        return "unknown"
    values = sorted({str(value).lower() for value in flows["file_type"].dropna()})
    return ",".join(values) if values else "unknown"


def _decorate_partition(
    pairs: pd.DataFrame,
    *,
    analysis_year: int,
    lodes_source_year: int,
    work_state: str,
    input_row_count: int,
    input_worker_weight: float,
    lodes_file_types: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach source-safe fields and calculate aggregate diagnostics."""
    state = str(work_state).lower()
    analysis = int(analysis_year)
    source_year = int(lodes_source_year)
    if pairs.empty:
        out = _empty_pairs()
    else:
        out = pairs.copy(deep=True)
        out["home_state_fips"] = out["home_county"].astype(str).str[:2]
        out["work_state_fips"] = out["work_county"].astype(str).str[:2]
        out["home_state"] = out["home_state_fips"].map(FIPS_TO_STATE)
        # The LODES partition, not an inferred row value, is authoritative for
        # a workplace-state source.  This retains interstate auxiliary rows.
        out["work_state"] = state
        out["same_tract"] = out["home_tract"].eq(out["work_tract"])
        out["home_car_share"] = pd.to_numeric(out["home_car_share"], errors="coerce")
        out["commuter_car_weight"] = out["workers"] * out["home_car_share"]
        out["commuter_car_miles"] = out["commuter_car_weight"] * out["block_pair_straight_line_miles"]
        out["routing_eligible"] = (
            out["missing_endpoint_workers"].eq(0)
            & out[["home_lat", "home_lon", "work_lat", "work_lon"]].notna().all(axis=1)
            & out["home_car_share"].between(0, 1, inclusive="both")
        )
        out["omitted_coordinate_worker_weight"] = out["missing_endpoint_workers"]
        out["omitted_car_share_worker_weight"] = out["missing_home_car_share_workers"]
        out["analysis_year"] = analysis
        out["lodes_source_year"] = source_year
        out["lodes_source_name"] = LODES_SOURCE_NAME
        out["lodes_file_types"] = lodes_file_types
        out["source_partition_id"] = f"{analysis}__{source_year}__{state}"
        out["route_id"] = (
            out["source_partition_id"] + "__" + out["home_tract"].astype(str) + "__" + out["work_tract"].astype(str)
        )
        out = out.sort_values("route_id").reset_index(drop=True)

    diagnostics = pd.DataFrame([{
        "partition_status": "success",
        "analysis_year": analysis,
        "lodes_source_year": source_year,
        "work_state": state,
        "lodes_file_types": lodes_file_types,
        "input_row_count": input_row_count,
        "input_worker_weight": input_worker_weight,
        "retained_pair_count": int(len(out)),
        "retained_worker_weight": float(pd.to_numeric(out.get("workers", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "missing_coordinate_worker_weight": float(pd.to_numeric(out.get("missing_endpoint_workers", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "missing_home_car_share_worker_weight": float(pd.to_numeric(out.get("missing_home_car_share_workers", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "same_tract_worker_weight": float(out.loc[out.get("same_tract", pd.Series(dtype=bool)).fillna(False), "workers"].sum()) if not out.empty else 0.0,
        "same_tract_pair_count": int(out.get("same_tract", pd.Series(dtype=bool)).fillna(False).sum()),
    }])
    return out, diagnostics


def build_flow_partition(
    block_flows: pd.DataFrame,
    crosswalks: pd.DataFrame,
    tract_car_share: pd.Series,
    *,
    analysis_year: int,
    lodes_source_year: int,
    work_state: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one validated national flow partition from in-memory block flows."""
    row_count, worker_weight = _positive_flow_metrics(block_flows)
    pairs = weighted_tract_pairs(block_flows, crosswalks, tract_car_share)
    return _decorate_partition(
        pairs, analysis_year=analysis_year, lodes_source_year=lodes_source_year,
        work_state=work_state, input_row_count=row_count, input_worker_weight=worker_weight,
        lodes_file_types=_flow_file_types(block_flows),
    )


def iter_lodes_flow_chunks(paths: Sequence[Path], *, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Yield required LODES OD columns with their source file metadata."""
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    for path in paths:
        state, file_type, year = _parse_flow_path(Path(path))
        for chunk in pd.read_csv(
            path, compression="gzip", usecols=["h_geocode", "w_geocode", "S000"],
            dtype={"h_geocode": "string", "w_geocode": "string"}, chunksize=int(chunk_rows),
        ):
            chunk["state"] = state
            chunk["file_type"] = file_type
            chunk["year"] = year
            chunk["source_path"] = str(path)
            yield chunk


def _combine_chunk_pairs(chunk_pairs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in chunk_pairs if not frame.empty]
    if not frames:
        return weighted_tract_pairs(pd.DataFrame(columns=["h_geocode", "w_geocode", "S000"]), pd.DataFrame(columns=["tabblk2020", "cty", "trct", "blklatdd", "blklondd"]), pd.Series(dtype=float))
    merged = pd.concat(frames, ignore_index=True)
    group_cols = ["home_tract", "work_tract", "home_county", "work_county"]
    weight = pd.to_numeric(merged["valid_endpoint_workers"], errors="coerce").fillna(0.0)
    merged["_endpoint_weight"] = weight
    for column in ("home_lat", "home_lon", "work_lat", "work_lon", "block_pair_straight_line_miles"):
        merged[f"_{column}_sum"] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0) * weight
    aggregations: dict[str, tuple[str, str]] = {
        "workers": ("workers", "sum"),
        "valid_endpoint_workers": ("valid_endpoint_workers", "sum"),
        "missing_endpoint_workers": ("missing_endpoint_workers", "sum"),
        "_endpoint_weight": ("_endpoint_weight", "sum"),
        "home_car_share": ("home_car_share", "first"),
        "missing_home_car_share_workers": ("missing_home_car_share_workers", "sum"),
    }
    for column in ("home_lat", "home_lon", "work_lat", "work_lon", "block_pair_straight_line_miles"):
        aggregations[f"_{column}_sum"] = (f"_{column}_sum", "sum")
    combined = merged.groupby(group_cols, sort=True, as_index=False).agg(**aggregations)
    for column in ("home_lat", "home_lon", "work_lat", "work_lon", "block_pair_straight_line_miles"):
        combined[column] = (combined.pop(f"_{column}_sum") / combined["_endpoint_weight"]).where(combined["_endpoint_weight"].gt(0))
    return combined.drop(columns="_endpoint_weight").sort_values(["home_tract", "work_tract"]).reset_index(drop=True)


def _write_partial_pair(frame: pd.DataFrame, path: Path) -> None:
    """Persist one small aggregate before reading the next input chunk."""
    frame.to_parquet(path, index=False)


def _reduce_spilled_pairs(paths: list[Path], spill_dir: Path, max_open: int) -> pd.DataFrame:
    """Reduce spill files in bounded batches, retaining no chunk frames globally."""
    if not paths:
        return _combine_chunk_pairs(())
    generation = 0
    while len(paths) > 1:
        next_paths: list[Path] = []
        for start in range(0, len(paths), max_open):
            batch = paths[start:start + max_open]
            combined = _combine_chunk_pairs(pd.read_parquet(path) for path in batch)
            target = spill_dir / f"reduce-{generation:04d}-{len(next_paths):08d}.parquet"
            _write_partial_pair(combined, target)
            next_paths.append(target)
            for path in batch:
                path.unlink(missing_ok=True)
        paths = next_paths
        generation += 1
    try:
        return pd.read_parquet(paths[0])
    finally:
        paths[0].unlink(missing_ok=True)


def build_flow_partition_from_chunks(
    chunks: Iterable[pd.DataFrame], crosswalks: pd.DataFrame, tract_car_share: pd.Series, *,
    analysis_year: int, lodes_source_year: int, work_state: str,
    max_open_chunk_partitions: int = 4, spill_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate bounded flow chunks without changing weighted tract-pair results."""
    if int(max_open_chunk_partitions) < 2:
        raise ValueError("max_open_chunk_partitions must be at least two")
    prepared_crosswalk = prepare_crosswalk_for_pairs(crosswalks)
    row_count = 0
    worker_weight = 0.0
    file_types: set[str] = set()
    temporary_directory = None
    if spill_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="route-national-flow-")
        work_dir = Path(temporary_directory.name)
    else:
        work_dir = Path(spill_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        partial_paths: list[Path] = []
        for index, chunk in enumerate(chunks):
            rows, workers = _positive_flow_metrics(chunk)
            row_count += rows
            worker_weight += workers
            file_types.update(filter(None, _flow_file_types(chunk).split(",")))
            partial = weighted_tract_pairs(
                chunk, crosswalks, tract_car_share,
                prepared_crosswalk=prepared_crosswalk,
            )
            if not partial.empty:
                path = work_dir / f"chunk-{index:08d}.parquet"
                _write_partial_pair(partial, path)
                partial_paths.append(path)
        pairs = _reduce_spilled_pairs(partial_paths, work_dir, int(max_open_chunk_partitions))
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    return _decorate_partition(
        pairs, analysis_year=analysis_year, lodes_source_year=lodes_source_year,
        work_state=work_state, input_row_count=row_count, input_worker_weight=worker_weight,
        lodes_file_types=",".join(sorted(file_types)) or "unknown",
    )


def build_flow_partition_to_parquet(
    chunks: Iterable[pd.DataFrame], crosswalks: pd.DataFrame, tract_car_share: pd.Series, *,
    analysis_year: int, lodes_source_year: int, work_state: str, output_path: Path,
    extra_metadata: Mapping[str, object] | None = None, bucket_count: int = 64,
    max_open_chunk_partitions: int = 4,
) -> pd.DataFrame:
    """Hash-partition, reduce, and write a large flow partition incrementally."""
    if int(bucket_count) < 2:
        raise ValueError("bucket_count must be at least two")
    if int(max_open_chunk_partitions) < 2:
        raise ValueError("max_open_chunk_partitions must be at least two")
    prepared_crosswalk = prepare_crosswalk_for_pairs(crosswalks)
    temporary_directory = tempfile.TemporaryDirectory(prefix="route-national-stream-")
    work_dir = Path(temporary_directory.name)
    bucket_paths: list[list[Path]] = [[] for _ in range(int(bucket_count))]
    row_count = 0
    worker_weight = 0.0
    file_types: set[str] = set()
    try:
        for index, chunk in enumerate(chunks):
            rows, workers = _positive_flow_metrics(chunk)
            row_count += rows
            worker_weight += workers
            file_types.update(filter(None, _flow_file_types(chunk).split(",")))
            partial = weighted_tract_pairs(
                chunk, crosswalks, tract_car_share,
                prepared_crosswalk=prepared_crosswalk,
            )
            if partial.empty:
                continue
            group_cols = ["home_tract", "work_tract", "home_county", "work_county"]
            buckets = (
                pd.util.hash_pandas_object(partial[group_cols], index=False)
                .astype("uint64")
                .mod(int(bucket_count))
                .to_numpy()
            )
            for bucket in np.unique(buckets):
                selected = partial.loc[buckets == bucket]
                target = work_dir / f"bucket-{int(bucket):04d}-{index:08d}.parquet"
                selected.to_parquet(target, index=False)
                bucket_paths[int(bucket)].append(target)

        totals = {
            "retained_pair_count": 0,
            "retained_worker_weight": 0.0,
            "missing_coordinate_worker_weight": 0.0,
            "missing_home_car_share_worker_weight": 0.0,
            "same_tract_worker_weight": 0.0,
            "same_tract_pair_count": 0,
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.", suffix=".tmp.parquet", dir=output.parent,
            delete=False,
        ) as handle:
            temporary_output = Path(handle.name)
        writer: pq.ParquetWriter | None = None
        writer_schema: pa.Schema | None = None
        metadata_values = dict(extra_metadata or {})
        try:
            for paths in bucket_paths:
                if not paths:
                    continue
                reduced = _reduce_spilled_pairs(
                    paths, work_dir, int(max_open_chunk_partitions)
                )
                if reduced.empty:
                    continue
                decorated, bucket_diagnostics = _decorate_partition(
                    reduced, analysis_year=analysis_year,
                    lodes_source_year=lodes_source_year, work_state=work_state,
                    input_row_count=0, input_worker_weight=0.0,
                    lodes_file_types=",".join(sorted(file_types)) or "unknown",
                )
                for key, value in metadata_values.items():
                    decorated[key] = value
                table = pa.Table.from_pandas(
                    decorated, schema=writer_schema, preserve_index=False, safe=False
                ) if writer_schema is not None else pa.Table.from_pandas(
                    decorated, preserve_index=False
                )
                if writer is None:
                    writer_schema = table.schema
                    writer = pq.ParquetWriter(temporary_output, writer_schema)
                writer.write_table(table)
                row = bucket_diagnostics.iloc[0]
                totals["retained_pair_count"] += int(row["retained_pair_count"])
                for key in (
                    "retained_worker_weight", "missing_coordinate_worker_weight",
                    "missing_home_car_share_worker_weight", "same_tract_worker_weight",
                ):
                    totals[key] += float(row[key])
                totals["same_tract_pair_count"] += int(row["same_tract_pair_count"])
            if writer is None:
                empty = _empty_pairs()
                for key, value in metadata_values.items():
                    empty[key] = value
                empty.to_parquet(temporary_output, index=False)
            else:
                writer.close()
                writer = None
            os.replace(temporary_output, output)
        finally:
            if writer is not None:
                writer.close()
            temporary_output.unlink(missing_ok=True)
    finally:
        temporary_directory.cleanup()

    diagnostics = pd.DataFrame([{
        "partition_status": "success", "analysis_year": int(analysis_year),
        "lodes_source_year": int(lodes_source_year), "work_state": str(work_state).lower(),
        "lodes_file_types": ",".join(sorted(file_types)) or "unknown",
        "input_row_count": row_count, "input_worker_weight": worker_weight,
        **totals,
    }])
    return diagnostics


def _atomic_write_frame(frame: pd.DataFrame, path: Path, *, csv: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.csv" if csv else ".tmp.parquet"
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if csv:
            frame.to_csv(temporary, index=False)
        else:
            frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".tmp.json", dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
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


def _download_nearest_state_flows(
    state: str, analysis_year: int, input_dir: Path, session: requests.Session
) -> tuple[int | None, list[Path], str]:
    remaining = list(LODES_CANDIDATE_YEARS)
    errors: list[str] = []
    while remaining:
        choice = resolve_nearest_year(int(analysis_year), remaining)
        assert choice.source_year is not None
        year = choice.source_year
        try:
            paths = [download_lodes_input(state, kind, year, input_dir, session) for kind in ("main", "aux")]
            return year, paths, choice.reason
        except Exception as exc:
            errors.append(f"{year}: {exc}")
            remaining.remove(year)
    return None, [], "; ".join(errors) or "no available source years"


def _source_record(path: Path, url: str, state: str, file_type: str, year: int) -> dict[str, object]:
    """Use the immutable download sidecar when present, otherwise record a fixture input."""
    sidecar = Path(path).with_name(Path(path).name + ".manifest.json")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            record.get("url") == url
            and record.get("state") == str(state).lower()
            and record.get("file_type") == str(file_type).lower()
        ):
            return record
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    record = manifest_record(path, url, state, file_type, year)
    _atomic_write_json(record, sidecar)
    return record


def _canonical_provenance(records: Sequence[dict[str, object]]) -> tuple[str, str]:
    canonical = json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_crosswalks_for_states(
    states: Sequence[str], input_dir: Path, session: requests.Session
) -> tuple[list[Path], list[dict[str, object]]]:
    paths = [download_lodes_input(state, "xwalk", 0, input_dir, session) for state in states]
    records = [
        _source_record(path, lodes_url(state, "xwalk", 0), state, "xwalk", 0)
        for state, path in zip(states, paths, strict=True)
    ]
    return paths, records


def _normalize_block_ids(values: pd.Series) -> pd.Series:
    """Normalize LODES block identifiers for set membership and joins."""
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(15)


def _collect_aux_home_block_ids(paths: Sequence[Path], *, chunk_rows: int) -> set[str]:
    """Collect only auxiliary home blocks needed by a batch."""
    block_ids: set[str] = set()
    for path in paths:
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=["h_geocode", "w_geocode"],
            dtype={"h_geocode": "string", "w_geocode": "string"},
            chunksize=int(chunk_rows),
        ):
            values = _normalize_block_ids(chunk["h_geocode"]).dropna()
            block_ids.update(values.tolist())
    return block_ids


def _load_crosswalk_rows_for_blocks(
    paths: Sequence[Path], block_ids: set[str] | None = None, *, chunk_rows: int = 1_000_000
) -> pd.DataFrame:
    """Read only crosswalk rows referenced by a flow batch.

    National auxiliary files contain interstate home blocks, so loading only
    the workplace state's crosswalk is incorrect.  Filtering each source while
    streaming retains those auxiliary endpoints without materializing all
    8-million-plus national crosswalk rows at once.
    """
    wanted = (
        {str(value).strip().replace(".0", "").zfill(15) for value in block_ids}
        if block_ids is not None
        else None
    )
    columns = ["tabblk2020", "cty", "trct", "blklatdd", "blklondd"]
    if wanted is not None and not wanted:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    for path in paths:
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=columns,
            dtype={"tabblk2020": "string", "cty": "string", "trct": "string"},
            chunksize=int(chunk_rows),
        ):
            normalized = _normalize_block_ids(chunk["tabblk2020"])
            selected = (
                chunk.loc[normalized.isin(wanted)].copy()
                if wanted is not None
                else chunk
            )
            if not selected.empty:
                selected["tabblk2020"] = normalized.loc[selected.index]
                frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _crosswalk_state(path: Path) -> str:
    return Path(path).name.split("_", 1)[0].lower()


def _load_batch_crosswalks(
    batch: Sequence[tuple[str, int, list[Path], str, dict[str, object]]],
    crosswalk_paths: Sequence[Path],
    *,
    chunk_rows: int,
) -> pd.DataFrame:
    """Load complete workplace crosswalks plus filtered auxiliary origins."""
    by_state = {_crosswalk_state(path): path for path in crosswalk_paths}
    frames: list[pd.DataFrame] = []
    aux_paths: list[Path] = []
    for state, _, paths, _, _ in batch:
        work_path = by_state.get(state)
        if work_path is None:
            raise ValueError(f"missing crosswalk path for workplace state {state}")
        # Main files have both endpoints in the workplace state.  They need
        # the complete workplace crosswalk; auxiliary files add interstate
        # home endpoints and are filtered below.
        frames.append(_load_crosswalk_rows_for_blocks([work_path], chunk_rows=chunk_rows))
        aux_paths.extend(path for path in paths if "_od_aux_" in Path(path).name)

    aux_home_ids = _collect_aux_home_block_ids(aux_paths, chunk_rows=chunk_rows)
    if aux_home_ids:
        aux_states = {FIPS_TO_STATE.get(block[:2]) for block in aux_home_ids}
        aux_states.discard(None)
        frames.append(
            _load_crosswalk_rows_for_blocks(
                [by_state[state] for state in sorted(aux_states) if state in by_state],
                aux_home_ids,
                chunk_rows=chunk_rows,
            )
        )
    if not frames:
        return pd.DataFrame(columns=["tabblk2020", "cty", "trct", "blklatdd", "blklondd"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("tabblk2020", keep="first")


def _load_car_share(analysis_year: int) -> tuple[pd.Series, str, dict[str, float]]:
    return load_car_share_for_analysis_year(int(analysis_year), CAR_SHARE_ROOT)


def _acs_provenance(vintage: str) -> tuple[str, str]:
    """Collect state-specific ACS provenance embedded in the selected partitions."""
    records: list[dict[str, object]] = []
    directory = CAR_SHARE_ROOT / f"acs_{vintage}"
    for path in sorted(directory.glob("state=*.parquet")):
        try:
            metadata = pq.read_metadata(path).metadata or {}
            raw = metadata.get(b"route_national.provenance")
            if raw is not None:
                records.append(json.loads(raw.decode("utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return _canonical_provenance(records)


def _partition_path(root: Path, analysis_year: int, source_year: int, state: str) -> Path:
    return root / "partitions" / f"analysis_year={analysis_year}" / f"lodes_source_year={source_year}" / f"work_state={state}.parquet"


def _partition_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _valid_partition(path: Path, expected: dict[str, object]) -> bool:
    """Accept a completed partition only when its sidecar and row keys agree."""
    sidecar = _partition_sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        actual = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(actual.get(key) != value for key, value in expected.items()):
            return False
        if actual.get("parquet_sha256") != _file_sha256(path):
            return False
        table = pd.read_parquet(path, columns=["route_id", "provenance_sha256"])
        return table["route_id"].is_unique and table["provenance_sha256"].eq(
            expected["provenance_sha256"]
        ).all()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _metadata_for_partition(
    *, analysis_year: int, source_year: int, state: str,
    main: dict[str, object], aux: dict[str, object],
    crosswalk_records: Sequence[dict[str, object]], acs_vintage: str,
    car_share_diagnostics: dict[str, float], acs_provenance: str, acs_source_id: str,
) -> dict[str, object]:
    crosswalk_provenance, crosswalk_source_id = _canonical_provenance(crosswalk_records)
    source_partition_id = f"{analysis_year}__{source_year}__{state}"
    metadata: dict[str, object] = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "analysis_year": int(analysis_year), "lodes_source_year": int(source_year),
        "lodes_year_gap": abs(int(source_year) - int(analysis_year)), "work_state": state,
        "source_partition_id": source_partition_id, "lodes_source_id": f"lodes8:{source_partition_id}",
        "lodes_source_name": LODES_SOURCE_NAME,
        "main_url": main["url"], "main_sha256": main["sha256"], "main_bytes": main["bytes"],
        "main_retrieved_at_utc": main["retrieved_at_utc"], "aux_url": aux["url"],
        "aux_sha256": aux["sha256"], "aux_bytes": aux["bytes"],
        "aux_retrieved_at_utc": aux["retrieved_at_utc"],
        "crosswalk_source_provenance": crosswalk_provenance,
        "crosswalk_source_id": crosswalk_source_id,
        "acs_car_share_vintage": acs_vintage,
        "acs_window_start": car_share_diagnostics.get("acs_window_start"),
        "acs_window_end": car_share_diagnostics.get("acs_window_end"),
        "acs_source_provenance": acs_provenance, "acs_source_id": acs_source_id,
    }
    _, metadata["provenance_sha256"] = _canonical_provenance([metadata])
    return metadata


def build_national_flow_year(
    analysis_year: int,
    states: Sequence[str] | None = None,
    cache_dir: Path = NATIONAL_FLOW_ROOT,
    *,
    chunk_rows: int = 250_000,
    origin_states: Sequence[str] | None = None,
    crosswalk_batch_states: int = 4,
    stream_bucket_count: int = 16,
) -> pd.DataFrame:
    """Download, validate, process, and manifest all workplace states for one year."""
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    if int(crosswalk_batch_states) <= 0:
        raise ValueError("crosswalk_batch_states must be positive")
    if int(stream_bucket_count) < 2:
        raise ValueError("stream_bucket_count must be at least two")
    analysis = int(analysis_year)
    requested = tuple(sorted({str(state).lower() for state in (NATIONAL_STATES if states is None else states)}))
    if not requested or any(len(state) != 2 or not state.isalpha() for state in requested):
        raise ValueError("states must be non-empty two-letter abbreviations")
    origins = tuple(sorted({str(state).lower() for state in (NATIONAL_STATES if origin_states is None else origin_states)}))
    if not origins or any(len(state) != 2 or not state.isalpha() for state in origins):
        raise ValueError("origin_states must be non-empty two-letter abbreviations")
    root = Path(cache_dir)
    input_dir = root / "inputs"
    session = requests.Session()
    manifest_rows: list[dict[str, object]] = []

    # Resolve flow years before loading national geography or car shares.  A
    # fully unavailable request still emits useful evidence without requiring
    # unrelated input downloads.
    selected: dict[str, tuple[int, list[Path], str]] = {}
    for state in requested:
        source_year, paths, reason = _download_nearest_state_flows(state, analysis, input_dir, session)
        if source_year is None:
            diagnostic = pd.DataFrame([{
                "partition_status": "unavailable", "analysis_year": analysis, "lodes_source_year": pd.NA,
                "work_state": state, "lodes_reason": reason, "input_row_count": 0,
                "input_worker_weight": 0.0, "retained_pair_count": 0, "retained_worker_weight": 0.0,
            }])
            _atomic_write_frame(diagnostic, root / "diagnostics" / f"analysis_year={analysis}" / f"work_state={state}.csv", csv=True)
            manifest_rows.append({
                "analysis_year": analysis, "state": state, "lodes_source_year": None,
                "lodes_year_gap": None, "lodes_status": "unavailable", "lodes_reason": reason,
                "partition_status": "unavailable", "selection_rule": "nearest source year; earlier year wins ties",
            })
        else:
            selected[state] = (source_year, paths, reason)

    if selected:
        tract_car_share, acs_vintage, car_share_diagnostics = _load_car_share(analysis)
        acs_provenance, acs_source_id = _acs_provenance(acs_vintage)
        source_records: dict[str, dict[str, dict[str, object]]] = {}
        for state, (source_year, paths, _) in selected.items():
            source_records[state] = {
                kind: _source_record(path, lodes_url(state, kind, source_year), state, kind, source_year)
                for kind, path in zip(("main", "aux"), paths, strict=True)
            }

        # Geography is intentionally delayed until a partition needs rebuilding.
        # A valid resumed partition still validates all flow and ACS source ids,
        # but avoids the crosswalk load and block-to-tract aggregation.
        crosswalk_paths, crosswalk_records = _download_crosswalks_for_states(origins, input_dir, session)
        pending: list[tuple[str, int, list[Path], str, dict[str, object]]] = []
        for state, (source_year, paths, reason) in selected.items():
            metadata = _metadata_for_partition(
                analysis_year=analysis, source_year=source_year, state=state,
                main=source_records[state]["main"], aux=source_records[state]["aux"],
                crosswalk_records=crosswalk_records, acs_vintage=acs_vintage,
                car_share_diagnostics=car_share_diagnostics, acs_provenance=acs_provenance,
                acs_source_id=acs_source_id,
            )
            partition = _partition_path(root, analysis, source_year, state)
            if _valid_partition(partition, metadata):
                manifest_rows.append({
                    **metadata, "state": state, "lodes_status": "exact" if source_year == analysis else "nearest",
                    "lodes_reason": reason, "partition_status": "reused",
                })
            else:
                pending.append((state, source_year, paths, reason, metadata))

        # A batch keeps the crosswalk bounded: a national crosswalk is several
        # gigabytes in pandas, while each batch needs only its flow endpoints.
        batch_size = int(crosswalk_batch_states)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            crosswalks = _load_batch_crosswalks(
                batch,
                crosswalk_paths,
                chunk_rows=min(int(chunk_rows), 1_000_000),
            )
            for state, source_year, paths, reason, metadata in batch:
                partition = _partition_path(root, analysis, source_year, state)
                if sum(path.stat().st_size for path in paths) >= STREAMING_INPUT_BYTES:
                    diagnostics = build_flow_partition_to_parquet(
                        iter_lodes_flow_chunks(paths, chunk_rows=chunk_rows),
                        crosswalks, tract_car_share, analysis_year=analysis,
                        lodes_source_year=source_year, work_state=state,
                        output_path=partition,
                        extra_metadata={
                            "lodes_source_paths": "|".join(str(path) for path in paths),
                            **metadata,
                        },
                        bucket_count=int(stream_bucket_count),
                    )
                else:
                    pairs, diagnostics = build_flow_partition_from_chunks(
                        iter_lodes_flow_chunks(paths, chunk_rows=chunk_rows), crosswalks, tract_car_share,
                        analysis_year=analysis, lodes_source_year=source_year, work_state=state,
                    )
                    pairs["lodes_source_paths"] = "|".join(str(path) for path in paths)
                    for key, value in metadata.items():
                        pairs[key] = value
                    _atomic_write_frame(pairs, partition)
                metadata["parquet_sha256"] = _file_sha256(partition)
                _atomic_write_json(metadata, _partition_sidecar(partition))
                for key, value in metadata.items():
                    diagnostics[key] = value
                diagnostics["car_share_tracts"] = car_share_diagnostics.get("tracts_with_share", np.nan)
                _atomic_write_frame(diagnostics, root / "diagnostics" / f"analysis_year={analysis}" / f"work_state={state}.csv", csv=True)
                manifest_rows.append({
                    **metadata, "state": state, "lodes_status": "exact" if source_year == analysis else "nearest",
                    "lodes_reason": reason, "partition_status": "success",
                    "selection_rule": "nearest source year; earlier year wins ties",
                })

    write_vintage_manifest(manifest_rows, root / "national_vintage_manifest.csv")
    return pd.DataFrame(manifest_rows).sort_values(["analysis_year", "state"], kind="stable").reset_index(drop=True) if manifest_rows else pd.DataFrame()
