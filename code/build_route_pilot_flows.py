"""Prepare 2022 LODES inputs for the requested Wisconsin route-pilot region.

This module keeps the acquisition and aggregation steps separate from the later
routing work:

* download the 2022 JT00 OD files and LODES geography crosswalk for every
  requested state;
* write a small manifest for every downloaded input;
* load the OD rows at block level, filter to the five-state pilot region, and
  collapse them to representative tract pairs with worker-weighted endpoints;
* persist a compact pair table and flow diagnostics beneath the route-pilot
  cache.

The module is import-safe. Network access only happens through ``main()`` or an
explicit call to the downloader helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import DATA_PROC, ROUTE_PILOT_CACHE
from route_exposure_core import weighted_tract_pairs

LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"
TRACT_CAR_SHARE_PATH = DATA_PROC / "tract_car_share.parquet"
DEFAULT_YEAR = 2022
PILOT_STATES = ("wi", "il", "ia", "mn", "mi")
PILOT_STATE_FIPS = {
    "wi": "55",
    "il": "17",
    "ia": "19",
    "mn": "27",
    "mi": "26",
}
STATE_FIPS_TO_ABBR = {fips: abbr for abbr, fips in PILOT_STATE_FIPS.items()}
OD_FILE_RE = re.compile(r"^(?P<state>[a-z]{2})_od_(?P<file_type>main|aux)_JT00_(?P<year>\d{4})\.csv\.gz$")
XWALK_FILE_RE = re.compile(r"^(?P<state>[a-z]{2})_xwalk\.csv\.gz$")

PAIR_OUTPUT_NAME = "pilot_tract_pairs_2022.parquet"
DIAGNOSTICS_OUTPUT_NAME = "pilot_flow_diagnostics_2022.csv"
MANIFEST_OUTPUT_NAME = "pilot_input_manifest_2022.csv"
LODES_SOURCE_NAME = "U.S. Census Bureau LEHD Origin-Destination Employment Statistics (LODES8)"


def lodes_url(state: str, file_type: str, year: int) -> str:
    state = str(state).lower()
    file_type = str(file_type).lower()
    if file_type == "xwalk":
        return f"{LODES_BASE}/{state}/{state}_xwalk.csv.gz"
    if file_type not in {"main", "aux"}:
        raise ValueError(f"unsupported LODES file_type: {file_type}")
    return f"{LODES_BASE}/{state}/od/{state}_od_{file_type}_JT00_{int(year)}.csv.gz"


def _download_target_path(state: str, file_type: str, year: int) -> str:
    state = str(state).lower()
    file_type = str(file_type).lower()
    if file_type == "xwalk":
        return f"{state}_xwalk.csv.gz"
    return f"{state}_od_{file_type}_JT00_{int(year)}.csv.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_record(src: Path, url: str, state: str, file_type: str, year: int) -> dict:
    src = Path(src)
    return {
        "path": str(src),
        "url": str(url),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": _sha256(src),
        "bytes": int(src.stat().st_size),
        "state": str(state).lower(),
        "file_type": str(file_type).lower(),
        "year": int(year),
        "source_name": LODES_SOURCE_NAME,
        "attribution": "U.S. Census Bureau LODES8",
        "license": "U.S. government public data; see Census terms of service",
    }


def write_input_manifest(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        frame = pd.DataFrame(
            columns=["path", "url", "retrieved_at_utc", "sha256", "bytes", "state", "file_type", "year", "source_name", "attribution", "license"]
        )
    else:
        desired = ["path", "url", "retrieved_at_utc", "sha256", "bytes", "state", "file_type", "year", "source_name", "attribution", "license"]
        for col in desired:
            if col not in frame.columns:
                frame[col] = pd.NA
        frame = frame.loc[:, desired].sort_values(["state", "file_type", "year", "path"]).reset_index(drop=True)
    frame.to_csv(path, index=False)


def _download_with_session(url: str, dest: Path, session: requests.Session, attempts: int = 4) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            with session.get(url, stream=True, timeout=(30, 1800)) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if not tmp.exists() or tmp.stat().st_size <= 0:
                raise RuntimeError(f"download returned no bytes for {url}")
            os.replace(tmp, dest)
            return
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    assert last_exc is not None
    raise last_exc


def download_lodes_input(
    state: str,
    file_type: str,
    year: int,
    cache_dir: Path,
    session: requests.Session,
) -> Path:
    state = str(state).lower()
    file_type = str(file_type).lower()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target = cache_dir / _download_target_path(state, file_type, year)
    sidecar = target.with_name(target.name + ".manifest.json")
    if target.exists() and target.stat().st_size > 0:
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            valid = (
                cached.get("url") == lodes_url(state, file_type, year)
                and int(cached.get("bytes", -1)) == target.stat().st_size
                and cached.get("sha256") == _sha256(target)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise RuntimeError(
                f"cached LODES input failed manifest/checksum validation at {target}; "
                "remove or replace the cache and its .manifest.json sidecar, then rerun"
            )
        return target

    url = lodes_url(state, file_type, year)
    _download_with_session(url, target, session)
    record = manifest_record(target, url, state, file_type, year)
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _parse_flow_path(path: Path) -> tuple[str, str, int]:
    match = OD_FILE_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"unrecognized LODES flow filename: {path}")
    return match.group("state"), match.group("file_type"), int(match.group("year"))


def _parse_xwalk_path(path: Path) -> str:
    match = XWALK_FILE_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"unrecognized LODES crosswalk filename: {path}")
    return match.group("state")


def load_lodes_block_flows(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        state, file_type, year = _parse_flow_path(path)
        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=["h_geocode", "w_geocode", "S000"],
            dtype={"h_geocode": "string", "w_geocode": "string"},
        )
        frame["state"] = state
        frame["file_type"] = file_type
        frame["year"] = int(year)
        frame["source_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["h_geocode", "w_geocode", "S000", "state", "file_type", "year", "source_path"])
    return pd.concat(frames, ignore_index=True)


def _load_lodes_crosswalks(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        state = _parse_xwalk_path(path)
        wanted = {"tabblk2020", "cty", "trct", "blklatdd", "blklondd", "urban_rural_class", "urban", "ur"}
        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=lambda column: column in wanted,
            dtype={"tabblk2020": "string", "cty": "string", "trct": "string"},
        )
        urban_column = next((column for column in ("urban_rural_class", "urban", "ur") if column in frame), None)
        if urban_column and urban_column != "urban_rural_class":
            frame = frame.rename(columns={urban_column: "urban_rural_class"})
        frame["state"] = state
        frame["source_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["tabblk2020", "cty", "trct", "blklatdd", "blklondd", "state", "source_path"])
    return pd.concat(frames, ignore_index=True)


def _load_tract_car_share(path: Path = TRACT_CAR_SHARE_PATH) -> pd.Series:
    path = Path(path)
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            "Unable to read the packaged tract-car-share parquet at "
            f"{path}. This pilot requires a tract-level car-share "
            "series, and the cache must be regenerated with the repo's pinned "
            "pyarrow==25.0.1 stack. Run `python code/build_acs_tract_car_share.py` "
            "after installing the project requirements with "
            "`python -m pip install -r requirements-analysis.txt`."
        ) from exc
    if not {"tract", "car_share"}.issubset(frame.columns):
        raise ValueError(f"{path} missing tract or car_share columns")
    share = frame.set_index("tract")["car_share"]
    share.index = share.index.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    return pd.to_numeric(share, errors="coerce")


def build_pilot_tract_pairs(
    block_flows: pd.DataFrame,
    crosswalks: pd.DataFrame,
    tract_car_share: pd.Series,
    pilot_state_fips: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flows = block_flows.copy(deep=True)
    if flows.empty:
        empty_pairs = pd.DataFrame(
            columns=[
                "home_tract",
                "work_tract",
                "home_county",
                "work_county",
                "workers",
                "valid_endpoint_workers",
                "missing_endpoint_workers",
                "home_lat",
                "home_lon",
                "work_lat",
                "work_lon",
                "home_car_share",
                "missing_home_car_share_workers",
                "home_state_fips",
                "work_state_fips",
                "home_state",
                "work_state",
                "same_tract",
                "route_id",
                "commuter_car_weight",
                "block_pair_straight_line_miles",
                "commuter_car_miles",
                "routing_eligible",
                "omitted_coordinate_worker_weight",
                "omitted_car_share_worker_weight",
            ]
        )
        diagnostics = pd.DataFrame(
            [
                {
                    "input_row_count": 0,
                    "input_worker_weight": 0.0,
                    "external_endpoint_worker_weight": 0.0,
                    "retained_row_count": 0,
                    "retained_worker_weight": 0.0,
                    "retained_pair_count": 0,
                    "missing_coordinate_worker_weight": 0.0,
                    "missing_home_car_share_worker_weight": 0.0,
                    "same_tract_worker_weight": 0.0,
                    "same_tract_pair_count": 0,
                }
            ]
        )
        return empty_pairs, diagnostics

    required_flow_cols = {"h_geocode", "w_geocode", "S000"}
    missing_flow_cols = required_flow_cols - set(flows.columns)
    if missing_flow_cols:
        raise ValueError(f"block_flows missing required columns: {sorted(missing_flow_cols)}")

    flows["h_geocode"] = flows["h_geocode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(15)
    flows["w_geocode"] = flows["w_geocode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(15)
    flows["workers"] = pd.to_numeric(flows["S000"], errors="coerce")
    flows = flows.dropna(subset=["workers"]).copy()
    flows = flows.loc[flows["workers"] > 0].copy()
    flows["home_state_fips"] = flows["h_geocode"].str[:2]
    flows["work_state_fips"] = flows["w_geocode"].str[:2]
    flows["home_state"] = flows["home_state_fips"].map(STATE_FIPS_TO_ABBR)
    flows["work_state"] = flows["work_state_fips"].map(STATE_FIPS_TO_ABBR)

    input_row_count = int(len(flows))
    input_worker_weight = float(flows["workers"].sum())
    pilot_fips = (
        set(PILOT_STATE_FIPS.values())
        if pilot_state_fips is None
        else {str(fips).zfill(2) for fips in pilot_state_fips}
    )
    in_region_mask = flows["home_state_fips"].isin(pilot_fips) & flows["work_state_fips"].isin(pilot_fips)
    external_endpoint_worker_weight = float(flows.loc[~in_region_mask, "workers"].sum())
    retained_flows = flows.loc[in_region_mask, ["h_geocode", "w_geocode", "S000"]].copy()
    retained_row_count = int(len(retained_flows))
    retained_worker_weight = float(pd.to_numeric(retained_flows["S000"], errors="coerce").fillna(0.0).sum())

    crosswalk = crosswalks.copy(deep=True)
    required_crosswalk_cols = {"tabblk2020", "cty", "trct", "blklatdd", "blklondd"}
    missing_crosswalk_cols = required_crosswalk_cols - set(crosswalk.columns)
    if missing_crosswalk_cols:
        raise ValueError(f"crosswalks missing required columns: {sorted(missing_crosswalk_cols)}")

    crosswalk["tabblk2020"] = crosswalk["tabblk2020"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(15)
    crosswalk = crosswalk.loc[crosswalk["tabblk2020"].str[:2].isin(pilot_fips)].copy()
    crosswalk["cty"] = crosswalk["cty"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    crosswalk["trct"] = crosswalk["trct"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    crosswalk["blklatdd"] = pd.to_numeric(crosswalk["blklatdd"], errors="coerce")
    crosswalk["blklondd"] = pd.to_numeric(crosswalk["blklondd"], errors="coerce")

    pairs = weighted_tract_pairs(retained_flows, crosswalk, tract_car_share)
    if not pairs.empty:
        if "urban_rural_class" in crosswalk.columns:
            tract_class = (
                crosswalk[["trct", "urban_rural_class"]]
                .dropna()
                .sort_values(["trct", "urban_rural_class"])
                .drop_duplicates("trct")
                .rename(columns={"trct": "home_tract"})
            )
            pairs = pairs.merge(tract_class, on="home_tract", how="left", validate="many_to_one")
        pairs["home_state_fips"] = pairs["home_county"].astype(str).str[:2]
        pairs["work_state_fips"] = pairs["work_county"].astype(str).str[:2]
        pairs["home_state"] = pairs["home_state_fips"].map(STATE_FIPS_TO_ABBR)
        pairs["work_state"] = pairs["work_state_fips"].map(STATE_FIPS_TO_ABBR)
        pairs["same_tract"] = pairs["home_tract"].eq(pairs["work_tract"])
        pairs["route_id"] = pairs["home_tract"].astype(str) + "__" + pairs["work_tract"].astype(str)
        pairs["home_car_share"] = pd.to_numeric(pairs["home_car_share"], errors="coerce")
        pairs["commuter_car_weight"] = pairs["workers"] * pairs["home_car_share"]
        pairs["commuter_car_miles"] = (
            pairs["commuter_car_weight"] * pairs["block_pair_straight_line_miles"]
        )
        pairs["routing_eligible"] = (
            pairs["missing_endpoint_workers"].eq(0)
            & pairs[["home_lat", "home_lon", "work_lat", "work_lon"]].notna().all(axis=1)
            & pairs["home_car_share"].between(0, 1, inclusive="both")
        )
        pairs["omitted_coordinate_worker_weight"] = pairs["missing_endpoint_workers"]
        pairs["omitted_car_share_worker_weight"] = pairs["missing_home_car_share_workers"]
        pairs = pairs.sort_values("route_id").reset_index(drop=True)
    else:
        pairs["home_state_fips"] = pd.Series(dtype="string")
        pairs["work_state_fips"] = pd.Series(dtype="string")
        pairs["home_state"] = pd.Series(dtype="string")
        pairs["work_state"] = pd.Series(dtype="string")
        pairs["same_tract"] = pd.Series(dtype="bool")
        pairs["route_id"] = pd.Series(dtype="string")
        pairs["commuter_car_weight"] = pd.Series(dtype="float64")
        pairs["block_pair_straight_line_miles"] = pd.Series(dtype="float64")
        pairs["commuter_car_miles"] = pd.Series(dtype="float64")
        pairs["routing_eligible"] = pd.Series(dtype="bool")
        pairs["omitted_coordinate_worker_weight"] = pd.Series(dtype="float64")
        pairs["omitted_car_share_worker_weight"] = pd.Series(dtype="float64")

    diagnostics = pd.DataFrame(
        [
            {
                "input_row_count": input_row_count,
                "input_worker_weight": input_worker_weight,
                "external_endpoint_worker_weight": external_endpoint_worker_weight,
                "retained_row_count": retained_row_count,
                "retained_worker_weight": retained_worker_weight,
                "retained_pair_count": int(len(pairs)),
                "missing_coordinate_worker_weight": float(pd.to_numeric(pairs.get("missing_endpoint_workers", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not pairs.empty else 0.0,
                "missing_home_car_share_worker_weight": float(pd.to_numeric(pairs.get("missing_home_car_share_workers", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not pairs.empty else 0.0,
                "same_tract_worker_weight": float(pairs.loc[pairs["same_tract"], "workers"].sum()) if not pairs.empty else 0.0,
                "same_tract_pair_count": int(pairs["same_tract"].sum()) if not pairs.empty else 0,
            }
        ]
    )
    return pairs.reset_index(drop=True), diagnostics


def _output_paths(cache_dir: Path, year: int = DEFAULT_YEAR) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    year = int(year)
    return {
        "pairs": cache_dir / f"pilot_tract_pairs_{year}.parquet",
        "diagnostics": cache_dir / f"pilot_flow_diagnostics_{year}.csv",
        "manifest": cache_dir / f"pilot_input_manifest_{year}.csv",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--states", nargs="+", default=list(PILOT_STATES))
    parser.add_argument("--cache-dir", type=Path, default=ROUTE_PILOT_CACHE)
    parser.add_argument("--crosswalk-path", type=Path, default=None)
    parser.add_argument("--tract-car-share-path", type=Path, default=TRACT_CAR_SHARE_PATH)
    args = parser.parse_args(argv)

    year = int(args.year)
    states = [str(state).lower() for state in args.states]
    unknown_states = sorted(set(states) - set(PILOT_STATE_FIPS))
    if unknown_states:
        parser.error(f"unsupported pilot states: {unknown_states}")
    requested_state_fips = {PILOT_STATE_FIPS[state] for state in states}
    cache_dir = Path(args.cache_dir)
    input_dir = cache_dir / "inputs" / str(year)
    output_paths = _output_paths(cache_dir, year)
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    manifest_records: list[dict] = []
    flow_paths: list[Path] = []
    xwalk_paths: list[Path] = []
    for state in states:
        for file_type in ("main", "aux", "xwalk"):
            downloaded = download_lodes_input(state, file_type, year, input_dir, session)
            manifest_records.append(manifest_record(downloaded, lodes_url(state, file_type, year), state, file_type, year))
            if file_type == "xwalk":
                xwalk_paths.append(downloaded)
            else:
                flow_paths.append(downloaded)

    if args.crosswalk_path is not None:
        crosswalk_paths = [Path(args.crosswalk_path)]
    else:
        crosswalk_paths = xwalk_paths

    block_flows = load_lodes_block_flows(flow_paths)
    crosswalks = _load_lodes_crosswalks(crosswalk_paths)
    tract_car_share = _load_tract_car_share(args.tract_car_share_path)

    pairs, diagnostics = build_pilot_tract_pairs(
        block_flows,
        crosswalks,
        tract_car_share,
        pilot_state_fips=requested_state_fips,
    )

    output_paths["pairs"].parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_paths["pairs"], index=False)
    diagnostics.to_csv(output_paths["diagnostics"], index=False)
    write_input_manifest(manifest_records, output_paths["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
