"""Route Wisconsin pilot tract pairs and allocate their miles to counties."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import CRS
from shapely import wkt
from shapely.geometry import shape

from build_route_pilot_network import DEFAULT_YEAR, route_pairs_with_checkpoints
from config import DATA_RAW, ROUTE_PILOT_CACHE
from route_exposure_core import allocate_route_miles

PAIR_OUTPUT_NAME = "pilot_tract_pairs_2022.parquet"
INPUT_MANIFEST_NAME = "pilot_input_manifest_2022.csv"
ROUTE_CACHE_NAME = "route_results_2022.parquet"
COUNTY_SEGMENTS_NAME = "route_county_segments_2022.parquet"
COUNTY_DIAGNOSTICS_NAME = "route_county_diagnostics_2022.csv"
DEFAULT_BOUNDARY_PATH = DATA_RAW / "crosswalks" / "tl_2022_us_county.geojson"
SUPPORTED_SAME_TRACT_MODES = {"primary_calibrated", "zero", "exclude"}
SAME_TRACT_NEGLIGIBLE_MILES = 0.1
SHORT_TRIP_MAX_STRAIGHT_MILES = 25.0
LOGGER = logging.getLogger(__name__)


def _county_boundary_source_url(year: int) -> str:
    return f"https://www2.census.gov/geo/tiger/TIGER{int(year)}/COUNTY/tl_{int(year)}_us_county.zip"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_county_boundary_input(path: Path, year: int) -> Path:
    path = Path(path)
    if not path.is_file():
        url = _county_boundary_source_url(year)
        raise FileNotFoundError(
            f"missing 2022 TIGER/Line county boundaries at {path}. Download {url} and convert it "
            f"to CRS84 GeoJSON with `ogr2ogr -t_srs EPSG:4326 {path} /path/to/tl_{int(year)}_us_county.shp`, "
            "or pass --county-boundaries-path to an equivalent validated file."
        )
    return path


def ensure_county_boundary_manifest(
    path: Path,
    year: int,
    *,
    source_url: str | None = None,
    attribution: str | None = None,
    source_crs: str | None = None,
    official_default: bool = False,
) -> dict:
    path = _require_county_boundary_input(path, year)
    manifest_path = path.with_name(path.name + ".manifest.json")
    checksum = _file_sha256(path)
    if manifest_path.exists():
        try:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid county boundary manifest at {manifest_path}") from exc
        required_manifest_fields = {
            "source_url", "retrieved_at_utc", "bytes", "sha256", "attribution", "crs"
        }
        missing_manifest_fields = required_manifest_fields - set(record)
        if missing_manifest_fields:
            raise RuntimeError(
                f"county boundary manifest missing provenance fields: {sorted(missing_manifest_fields)}"
            )
        if record.get("sha256") != checksum or int(record.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"county boundary cache failed manifest/checksum validation at {path}")
        if source_url and record.get("source_url") != source_url:
            raise RuntimeError(f"county boundary manifest source URL does not match {path}")
        if attribution and record.get("attribution") != attribution:
            raise RuntimeError(f"county boundary manifest attribution does not match {path}")
        if source_crs:
            try:
                manifest_crs = CRS.from_user_input(record["crs"])
                actual_crs = CRS.from_user_input(source_crs)
            except Exception as exc:
                raise RuntimeError(f"invalid CRS in county boundary manifest at {manifest_path}") from exc
            if not manifest_crs.equals(actual_crs, ignore_axis_order=True):
                raise RuntimeError(
                    f"county boundary manifest CRS does not match file metadata at {path}"
                )
        return record
    if official_default:
        source_url = source_url or _county_boundary_source_url(year)
        attribution = attribution or "U.S. Census Bureau TIGER/Line"
    if not source_url or not attribution:
        raise ValueError(
            "a custom county boundary file requires its actual source URL and attribution; "
            "pass --county-boundaries-source-url and --county-boundaries-attribution"
        )
    if not source_crs:
        raise ValueError("county boundary manifest requires validated source CRS metadata")
    record = {
        "path": str(path),
        "source_url": source_url,
        "retrieved_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "bytes": path.stat().st_size,
        "sha256": checksum,
        "attribution": attribution,
        "year": int(year),
        "crs": source_crs,
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _mode_output_paths(cache_dir: Path, year: int, mode: str) -> dict[str, Path]:
    if mode not in SUPPORTED_SAME_TRACT_MODES:
        raise ValueError(f"unsupported same_tract mode: {mode}")
    suffix = str(int(year)) if mode == "primary_calibrated" else f"{int(year)}_{mode}"
    return {
        "segments": Path(cache_dir) / f"route_county_segments_{suffix}.parquet",
        "diagnostics": Path(cache_dir) / f"route_county_diagnostics_{suffix}.csv",
    }


def _flow_artifact_paths(cache_dir: Path, year: int = DEFAULT_YEAR) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    year = int(year)
    return {
        "pairs": cache_dir / f"pilot_tract_pairs_{year}.parquet",
        "manifest": cache_dir / f"pilot_input_manifest_{year}.csv",
        "routes": cache_dir / f"route_results_{year}.parquet",
    }


def _normalize_code(value: object, width: int | None = None) -> str:
    if pd.isna(value):
        raise ValueError("missing code")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if width is not None:
        text = text.zfill(width)
    return text


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{next(tempfile._get_candidate_names())}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{next(tempfile._get_candidate_names())}.tmp.csv")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_geometry(value: object):
    if value is None or pd.isna(value):
        raise ValueError("missing geometry")
    if hasattr(value, "__geo_interface__"):
        return shape(value.__geo_interface__)
    if isinstance(value, dict):
        return shape(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            return shape(json.loads(text))
        if text.upper().startswith(("POLYGON", "MULTIPOLYGON")):
            return wkt.loads(text)
    raise ValueError("unsupported geometry encoding")


def load_county_boundaries(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        crs_name = frame.attrs.get("crs")
        if not crs_name:
            raise ValueError(
                f"parquet county boundaries at {path} require CRS metadata; "
                "use EPSG:4326/OGC:CRS84 longitude-latitude coordinates"
            )
    elif suffix in {".json", ".geojson"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("type") == "FeatureCollection":
            crs_name = str(((payload.get("crs") or {}).get("properties") or {}).get("name") or "CRS84")
            rows = []
            for feature in payload.get("features", []):
                props = dict(feature.get("properties") or {})
                props["geometry"] = feature.get("geometry")
                rows.append(props)
            frame = pd.DataFrame(rows)
        else:
            raise ValueError(f"{path} must be a GeoJSON FeatureCollection")
    else:
        raise ValueError(
            f"unsupported county boundary format for {path}. "
            "Use a GeoJSON/JSON FeatureCollection or parquet with geometry and county FIPS columns."
        )

    try:
        source_crs = CRS.from_user_input(crs_name)
        is_lonlat = source_crs.to_epsg() == 4326 or source_crs.equals(
            CRS.from_user_input("OGC:CRS84"), ignore_axis_order=True
        )
    except Exception as exc:
        raise ValueError(f"invalid county boundary CRS metadata: {crs_name}") from exc
    if not is_lonlat:
        raise ValueError(
            f"county boundaries must use EPSG:4326 or CRS84 longitude/latitude coordinates; got {crs_name}"
        )

    county_col = next((col for col in ("county_fips", "fips", "GEOID", "geoid") if col in frame.columns), None)
    if county_col is None or "geometry" not in frame.columns:
        raise ValueError(f"{path} must contain county FIPS and geometry columns")

    out = frame.loc[:, [county_col, "geometry"]].copy()
    out = out.rename(columns={county_col: "county_fips"})
    out["county_fips"] = out["county_fips"].map(lambda value: _normalize_code(value, 5))
    out["geometry"] = out["geometry"].map(_parse_geometry)

    repaired = 0
    cleaned = []
    for geom in out["geometry"]:
        if geom.is_valid:
            cleaned.append(geom)
            continue
        fixed = geom.buffer(0)
        if not fixed.is_valid or fixed.is_empty:
            raise ValueError("county geometry could not be repaired with buffer(0)")
        cleaned.append(fixed)
        repaired += 1
    out["geometry"] = cleaned
    out["geometry_repaired"] = repaired > 0
    if repaired:
        LOGGER.info("repaired %s county geometries with buffer(0)", repaired)
    out = out.sort_values("county_fips").reset_index(drop=True)
    out.attrs["source_crs"] = str(crs_name)
    return out


def _derive_route_id(row: pd.Series) -> str:
    route_id = row.get("route_id")
    if route_id is not None and not pd.isna(route_id) and str(route_id).strip():
        return str(route_id)
    return f"{_normalize_code(row['home_tract'], 11)}__{_normalize_code(row['work_tract'], 11)}"


def _load_geometry_document(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coalesce(*values: object) -> object:
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None


def _straight_line_miles(home_lon: object, home_lat: object, work_lon: object, work_lat: object) -> float:
    if any(pd.isna(value) for value in (home_lon, home_lat, work_lon, work_lat)):
        return float("nan")
    lon1 = float(home_lon)
    lat1 = float(home_lat)
    lon2 = float(work_lon)
    lat2 = float(work_lat)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    meters = 2 * 6_371_008.8 * math.asin(math.sqrt(a))
    return meters / 1609.344


def _base_segment_row(route_row: pd.Series) -> dict:
    return {
        "route_id": str(route_row["route_id"]),
        "home_tract": _coalesce(route_row.get("home_tract"), pd.NA),
        "work_tract": _coalesce(route_row.get("work_tract"), pd.NA),
        "home_fips": _coalesce(route_row.get("home_county"), route_row.get("home_fips"), pd.NA),
        "work_fips": _coalesce(route_row.get("work_county"), route_row.get("work_fips"), pd.NA),
        "workers": float(pd.to_numeric(route_row.get("workers"), errors="coerce") or 0.0),
        "home_car_share": pd.to_numeric(route_row.get("home_car_share"), errors="coerce"),
        "commuter_car_weight": pd.to_numeric(route_row.get("commuter_car_weight"), errors="coerce"),
        "block_pair_straight_line_miles": pd.to_numeric(route_row.get("block_pair_straight_line_miles"), errors="coerce"),
        "commuter_car_miles": pd.to_numeric(route_row.get("commuter_car_miles"), errors="coerce"),
        "routing_eligible": bool(route_row.get("routing_eligible", True)),
        "omitted_coordinate_worker_weight": pd.to_numeric(route_row.get("omitted_coordinate_worker_weight", 0.0), errors="coerce"),
        "omitted_car_share_worker_weight": pd.to_numeric(route_row.get("omitted_car_share_worker_weight", 0.0), errors="coerce"),
        "same_tract": bool(route_row.get("same_tract", False)),
        "status": str(route_row.get("status") or ""),
        "distance_m": pd.to_numeric(route_row.get("distance_m"), errors="coerce"),
        "duration_s": pd.to_numeric(route_row.get("duration_s"), errors="coerce"),
        "geometry_path": _coalesce(route_row.get("geometry_path"), pd.NA),
        "source_manifest_id": _coalesce(route_row.get("source_manifest_id"), pd.NA),
        "network_manifest_id": _coalesce(route_row.get("network_manifest_id"), pd.NA),
        "home_lon": pd.to_numeric(route_row.get("home_lon"), errors="coerce"),
        "home_lat": pd.to_numeric(route_row.get("home_lat"), errors="coerce"),
        "work_lon": pd.to_numeric(route_row.get("work_lon"), errors="coerce"),
        "work_lat": pd.to_numeric(route_row.get("work_lat"), errors="coerce"),
    }


def allocate_cached_routes(route_cache: pd.DataFrame, county_boundaries: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    required = {"route_id", "status", "home_lon", "home_lat", "work_lon", "work_lat"}
    missing = required - set(route_cache.columns)
    if missing:
        raise ValueError(f"route_cache missing required columns: {sorted(missing)}")

    frame = route_cache.copy()
    frame["route_id"] = frame.apply(_derive_route_id, axis=1)

    rows: list[dict] = []
    for _, route_row in frame.iterrows():
        base = _base_segment_row(route_row)
        block_distance = pd.to_numeric(route_row.get("block_pair_straight_line_miles"), errors="coerce")
        base["straight_line_miles"] = (
            float(block_distance)
            if pd.notna(block_distance)
            else _straight_line_miles(route_row["home_lon"], route_row["home_lat"], route_row["work_lon"], route_row["work_lat"])
        )
        if str(route_row.get("status")) != "Ok":
            total = pd.to_numeric(route_row.get("distance_m"), errors="coerce")
            route_miles_total = float(total) / 1609.344 if pd.notna(total) else 0.0
            rows.append(
                {
                    **base,
                    "county_fips": pd.NA,
                    "outcome_fips": pd.NA,
                    "route_miles_total": route_miles_total,
                    "route_miles_in_county": 0.0,
                    "unallocated_miles": route_miles_total,
                    "segment_type": "failed_route",
                    "same_tract_imputed_miles": 0.0,
                    "same_tract_mode": "routed",
                }
            )
            continue

        geometry_path = route_row.get("geometry_path")
        if geometry_path is None or pd.isna(geometry_path):
            raise ValueError(f"route {route_row['route_id']} is Ok but missing geometry_path")
        document = _load_geometry_document(Path(geometry_path))
        geometry = document.get("geometry")
        if geometry is None:
            raise ValueError(f"route {route_row['route_id']} geometry document missing geometry")
        route_geojson = dict(geometry)
        route_geojson["properties"] = {
            "distance_m": float(pd.to_numeric(route_row.get("distance_m"), errors="coerce"))
            if pd.notna(pd.to_numeric(route_row.get("distance_m"), errors="coerce"))
            else None
        }
        allocated = allocate_route_miles(route_geojson, county_boundaries, str(route_row["route_id"]))
        for _, segment in allocated.iterrows():
            rows.append(
                {
                    **base,
                    "county_fips": segment.get("county_fips"),
                    "outcome_fips": segment.get("county_fips"),
                    "route_miles_total": float(segment["route_miles_total"]),
                    "route_miles_in_county": float(segment["route_miles_in_county"]),
                    "unallocated_miles": float(segment["unallocated_miles"]),
                    "segment_type": str(segment["segment_type"]),
                    "same_tract_imputed_miles": 0.0,
                    "same_tract_mode": "routed",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "route_id",
                "home_tract",
                "work_tract",
                "home_fips",
                "work_fips",
                "workers",
                "home_car_share",
                "commuter_car_weight",
                "block_pair_straight_line_miles",
                "commuter_car_miles",
                "routing_eligible",
                "omitted_coordinate_worker_weight",
                "omitted_car_share_worker_weight",
                "same_tract",
                "status",
                "distance_m",
                "duration_s",
                "geometry_path",
                "source_manifest_id",
                "network_manifest_id",
                "home_lon",
                "home_lat",
                "work_lon",
                "work_lat",
                "straight_line_miles",
                "county_fips",
                "outcome_fips",
                "route_miles_total",
                "route_miles_in_county",
                "unallocated_miles",
                "segment_type",
                "same_tract_imputed_miles",
                "same_tract_mode",
            ]
        )
    _atomic_write_parquet(out, output_path)
    return out


def calibrate_same_tract_distance(pairs: pd.DataFrame, routed_segments: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in SUPPORTED_SAME_TRACT_MODES:
        raise ValueError(f"unsupported same_tract mode: {mode}")

    seg = routed_segments.copy()
    seg["route_id"] = seg["route_id"].astype(str)
    seg["same_tract"] = seg["same_tract"].fillna(False).astype(bool)
    seg["same_tract_mode"] = mode
    seg["same_tract_imputed_miles"] = pd.to_numeric(seg["same_tract_imputed_miles"], errors="coerce").fillna(0.0)
    if "county_fips" not in seg.columns and "outcome_fips" in seg.columns:
        seg["county_fips"] = seg["outcome_fips"]
    if "outcome_fips" not in seg.columns and "county_fips" in seg.columns:
        seg["outcome_fips"] = seg["county_fips"]

    pairs_frame = pairs.copy()
    pairs_frame["route_id"] = pairs_frame.apply(_derive_route_id, axis=1)
    if "same_tract" not in pairs_frame.columns:
        pairs_frame["same_tract"] = pairs_frame["home_tract"].astype(str).eq(pairs_frame["work_tract"].astype(str))
    same_pairs = pairs_frame.loc[pairs_frame["same_tract"]].copy()
    if same_pairs.empty:
        return seg

    for coord_col in ("home_lon", "home_lat", "work_lon", "work_lat"):
        pairs_frame[coord_col] = pd.to_numeric(pairs_frame[coord_col], errors="coerce")
        same_pairs[coord_col] = pd.to_numeric(same_pairs[coord_col], errors="coerce")
    for frame in (pairs_frame, same_pairs):
        if "block_pair_straight_line_miles" in frame.columns:
            frame["straight_line_miles"] = pd.to_numeric(
                frame["block_pair_straight_line_miles"], errors="coerce"
            )
        else:
            frame["straight_line_miles"] = frame.apply(
                lambda row: _straight_line_miles(
                    row["home_lon"], row["home_lat"], row["work_lon"], row["work_lat"]
                ),
                axis=1,
            )

    route_summary = (
        seg.groupby("route_id", as_index=False)
        .agg(
            route_miles_total=("route_miles_total", "max"),
            allocated_miles=("route_miles_in_county", "sum"),
            status=("status", "first"),
        )
    )
    pairs_frame = pairs_frame.merge(route_summary, on="route_id", how="left")
    same_pairs = same_pairs.merge(route_summary, on="route_id", how="left")

    def _urban_rural_series(frame: pd.DataFrame) -> pd.Series:
        if "urban_rural_class" in frame.columns:
            return frame["urban_rural_class"].fillna("all")
        return pd.Series("all", index=frame.index)

    pairs_frame["urban_rural_class"] = _urban_rural_series(pairs_frame)
    same_pairs["urban_rural_class"] = _urban_rural_series(same_pairs)

    ratio_candidates = pairs_frame.loc[
        (~pairs_frame["same_tract"])
        & pairs_frame["route_miles_total"].notna()
        & (pairs_frame["route_miles_total"] > 0)
        & (pairs_frame["straight_line_miles"] > 0)
        & (pairs_frame["straight_line_miles"] <= SHORT_TRIP_MAX_STRAIGHT_MILES)
    ].copy()
    ratio_candidates["route_to_straight_ratio"] = (
        ratio_candidates["route_miles_total"] / ratio_candidates["straight_line_miles"]
    )
    ratio_candidates = ratio_candidates.loc[ratio_candidates["route_to_straight_ratio"].replace([math.inf, -math.inf], pd.NA).notna()]
    ratio_by_class = ratio_candidates.groupby("urban_rural_class")["route_to_straight_ratio"].median().to_dict()
    default_ratio = float(ratio_candidates["route_to_straight_ratio"].median()) if not ratio_candidates.empty else 1.2

    def _rebuilt_same_tract_row(
        row: pd.Series,
        *,
        miles: float,
        status: str,
        segment_type: str,
    ) -> dict:
        # Begin with the reviewed pair artifact so sensitivity reconstruction cannot
        # silently discard route weights, eligibility/omission fields, or provenance.
        rebuilt = row.to_dict()
        home_fips = _coalesce(row.get("home_county"), row.get("home_fips"))
        work_fips = _coalesce(row.get("work_county"), row.get("work_fips"))
        rebuilt.update(
            {
                "route_id": str(row["route_id"]),
                "home_fips": home_fips,
                "work_fips": work_fips,
                "same_tract": True,
                "status": status,
                "distance_m": miles * 1609.344,
                "duration_s": 0.0 if miles == 0 else pd.NA,
                "geometry_path": pd.NA,
                "straight_line_miles": row.get("straight_line_miles", pd.NA),
                "county_fips": home_fips,
                "outcome_fips": home_fips,
                "route_miles_total": miles,
                "route_miles_in_county": miles,
                "unallocated_miles": 0.0,
                "segment_type": segment_type,
                "same_tract_imputed_miles": miles if segment_type == "same_tract_imputed" else 0.0,
                "same_tract_mode": mode,
            }
        )
        return rebuilt

    same_route_ids = set(same_pairs["route_id"].astype(str))
    if mode == "exclude":
        seg = seg.loc[~seg["route_id"].isin(same_route_ids)].copy()
        excluded = [
            _rebuilt_same_tract_row(
                row,
                miles=0.0,
                status="SameTractExcluded",
                segment_type="same_tract_excluded",
            )
            for _, row in same_pairs.iterrows()
        ]
        return pd.concat([seg, pd.DataFrame(excluded)], ignore_index=True).reset_index(drop=True)

    if mode == "zero":
        seg = seg.loc[~seg["route_id"].isin(same_route_ids)].copy()
        appended = [
            _rebuilt_same_tract_row(
                row,
                miles=0.0,
                status="SameTractZero",
                segment_type="same_tract_zero",
            )
            for _, row in same_pairs.iterrows()
        ]
        if appended:
            seg = pd.concat([seg, pd.DataFrame(appended)], ignore_index=True)
        return seg.reset_index(drop=True)

    impute_route_ids = set(
        same_pairs.loc[
            same_pairs["status"].astype(str).str.lower().isin({"ok", "success", "routed"})
            & same_pairs["route_miles_total"].fillna(float("inf")).le(SAME_TRACT_NEGLIGIBLE_MILES),
            "route_id",
        ].astype(str)
    )
    if not impute_route_ids:
        return seg.reset_index(drop=True)
    same_pairs = same_pairs.loc[same_pairs["route_id"].isin(impute_route_ids)].copy()
    seg = seg.loc[~seg["route_id"].isin(impute_route_ids)].copy()
    appended = []
    for _, row in same_pairs.iterrows():
        ratio = float(ratio_by_class.get(row["urban_rural_class"], default_ratio))
        imputed_miles = float(row["straight_line_miles"]) * ratio
        appended.append(
            _rebuilt_same_tract_row(
                row,
                miles=imputed_miles,
                status="SameTractImputed",
                segment_type="same_tract_imputed",
            )
        )
    if appended:
        seg = pd.concat([seg, pd.DataFrame(appended)], ignore_index=True)
    return seg.reset_index(drop=True)


def validate_mileage_conservation(
    segments: pd.DataFrame,
    tolerance_row: float = 0.005,
    tolerance_total: float = 0.001,
) -> dict:
    def _rejected_metrics(**overrides: object) -> dict:
        result = {
            "accepted": False,
            "n_routes": 0,
            "n_failed_rows": 0,
            "n_failed_routes": 0,
            "n_failed_rows_row_threshold": 0,
            "aggregate_relative_gap": 0.0,
            "total_unallocated_miles": 0.0,
            "selected_worker_weight": 0.0,
            "successful_worker_weight": 0.0,
            "selected_commuter_car_weight": 0.0,
            "successful_commuter_car_weight": 0.0,
            "successful_commuter_car_share": 1.0,
            "coverage_accepted": False,
            "conservation_accepted": False,
            "same_tract_imputed_miles": 0.0,
            "route_ratio_p50": float("nan"),
            "route_ratio_p90": float("nan"),
            "omitted_coordinate_worker_weight": 0.0,
            "omitted_car_share_worker_weight": 0.0,
            "missing_required_column_count": 0,
            "invalid_numeric_count": 0,
            "duplicate_segment_count": 0,
            "unexplained_zero_count": 0,
            "route_total_conflict_count": 0,
        }
        result.update(overrides)
        return result

    if segments.empty:
        return _rejected_metrics()

    seg = segments.copy()
    required_columns = {
        "route_id",
        "segment_type",
        "route_miles_total",
        "route_miles_in_county",
        "unallocated_miles",
        "workers",
        "home_car_share",
    }
    missing_columns = required_columns - set(seg.columns)
    if missing_columns:
        return _rejected_metrics(missing_required_column_count=len(missing_columns))

    seg["route_id"] = seg["route_id"].astype(str)
    mileage_columns = ["route_miles_total", "route_miles_in_county", "unallocated_miles"]
    for column in mileage_columns:
        seg[column] = pd.to_numeric(seg[column], errors="coerce")
    mileage_values = seg[mileage_columns].to_numpy(dtype=float)
    invalid_numeric_count = int((~np.isfinite(mileage_values) | (mileage_values < 0)).sum())

    county_key = next((name for name in ("county_fips", "outcome_fips") if name in seg), None)
    duplicate_segment_count = int(
        seg.duplicated(
            ["route_id", "segment_type", county_key] if county_key else None,
            keep=False,
        ).sum()
    )

    segment_types = seg["segment_type"].astype(str).str.lower()
    zero_total_allowed = segment_types.isin(
        {"failed_route", "same_tract_zero", "same_tract_excluded"}
    )
    zero_allocated_allowed = segment_types.isin(
        {"failed_route", "unallocated", "same_tract_zero", "same_tract_excluded"}
    )
    unexplained_zero_count = int(
        ((seg["route_miles_total"] == 0) & ~zero_total_allowed).sum()
        + ((seg["route_miles_in_county"] == 0) & ~zero_allocated_allowed).sum()
        + ((segment_types == "unallocated") & (seg["unallocated_miles"] == 0)).sum()
    )
    route_total_conflict_count = int(
        seg.groupby("route_id")["route_miles_total"].nunique(dropna=False).gt(1).sum()
    )
    if (
        invalid_numeric_count
        or duplicate_segment_count
        or unexplained_zero_count
        or route_total_conflict_count
    ):
        return _rejected_metrics(
            n_routes=int(seg["route_id"].nunique()),
            invalid_numeric_count=invalid_numeric_count,
            duplicate_segment_count=duplicate_segment_count,
            unexplained_zero_count=unexplained_zero_count,
            route_total_conflict_count=route_total_conflict_count,
        )

    seg["workers"] = pd.to_numeric(seg.get("workers"), errors="coerce").fillna(0.0)
    seg["home_car_share"] = pd.to_numeric(seg.get("home_car_share"), errors="coerce")
    seg["routing_eligible"] = seg.get("routing_eligible", pd.Series(True, index=seg.index)).fillna(False).astype(bool)
    invalid_share = seg["routing_eligible"] & (
        seg["home_car_share"].isna() | ~seg["home_car_share"].between(0, 1)
    )
    if invalid_share.any():
        raise ValueError("routing-eligible segments require finite home_car_share in [0, 1]")
    seg["commuter_car_weight"] = pd.to_numeric(
        seg.get("commuter_car_weight", seg["workers"] * seg["home_car_share"]), errors="coerce"
    )
    seg["omitted_coordinate_worker_weight"] = pd.to_numeric(
        seg.get("omitted_coordinate_worker_weight", pd.Series(0.0, index=seg.index)), errors="coerce"
    ).fillna(0.0)
    seg["omitted_car_share_worker_weight"] = pd.to_numeric(
        seg.get("omitted_car_share_worker_weight", pd.Series(0.0, index=seg.index)), errors="coerce"
    ).fillna(0.0)
    seg["straight_line_miles"] = pd.to_numeric(seg.get("straight_line_miles"), errors="coerce")
    seg["same_tract_imputed_miles"] = pd.to_numeric(seg.get("same_tract_imputed_miles"), errors="coerce").fillna(0.0)

    route_level = (
        seg.groupby("route_id", as_index=False)
        .agg(
            route_miles_total=("route_miles_total", "max"),
            allocated_miles=("route_miles_in_county", "sum"),
            unallocated_miles=("unallocated_miles", "sum"),
            workers=("workers", "max"),
            home_car_share=("home_car_share", "max"),
            commuter_car_weight=("commuter_car_weight", "max"),
            routing_eligible=("routing_eligible", "max"),
            omitted_coordinate_worker_weight=("omitted_coordinate_worker_weight", "max"),
            omitted_car_share_worker_weight=("omitted_car_share_worker_weight", "max"),
            straight_line_miles=("straight_line_miles", "max"),
            has_failed_row=("segment_type", lambda values: any(value == "failed_route" for value in values)),
        )
    )
    route_level["observed_total_miles"] = route_level["allocated_miles"] + route_level["unallocated_miles"]
    route_level["relative_gap"] = 0.0
    positive_total = route_level["route_miles_total"] > 0
    route_level.loc[positive_total, "relative_gap"] = (
        (route_level.loc[positive_total, "observed_total_miles"] - route_level.loc[positive_total, "route_miles_total"])
        .abs()
        / route_level.loc[positive_total, "route_miles_total"]
    )
    route_level["row_failed"] = (~route_level["has_failed_row"]) & (route_level["relative_gap"] > tolerance_row)

    total_route_miles = float(route_level.loc[~route_level["has_failed_row"], "route_miles_total"].sum())
    total_observed_miles = float(route_level.loc[~route_level["has_failed_row"], "observed_total_miles"].sum())
    aggregate_relative_gap = abs(total_observed_miles - total_route_miles) / total_route_miles if total_route_miles > 0 else 0.0

    eligible = route_level["routing_eligible"]
    selected_worker_weight = float(route_level.loc[eligible, "workers"].sum())
    successful_worker_weight = float(route_level.loc[eligible & ~route_level["has_failed_row"], "workers"].sum())
    selected_car_weight = float(route_level.loc[eligible, "commuter_car_weight"].sum())
    successful_car_weight = float(route_level.loc[eligible & ~route_level["has_failed_row"], "commuter_car_weight"].sum())
    successful_car_share = successful_car_weight / selected_car_weight if selected_car_weight > 0 else 1.0

    route_ratios = route_level.loc[
        (~route_level["has_failed_row"])
        & route_level["straight_line_miles"].notna()
        & (route_level["straight_line_miles"] > 0)
        & (route_level["route_miles_total"] > 0),
        "route_miles_total",
    ] / route_level.loc[
        (~route_level["has_failed_row"])
        & route_level["straight_line_miles"].notna()
        & (route_level["straight_line_miles"] > 0)
        & (route_level["route_miles_total"] > 0),
        "straight_line_miles",
    ]

    conservation_accepted = bool(
        (route_level["row_failed"].sum() == 0) and (aggregate_relative_gap <= tolerance_total)
    )
    coverage_accepted = bool(successful_car_share >= 0.99)
    return {
        "accepted": bool(conservation_accepted and coverage_accepted),
        "n_routes": int(len(route_level)),
        "n_failed_rows": int(seg["segment_type"].eq("failed_route").sum()),
        "n_failed_routes": int(route_level["has_failed_row"].sum()),
        "n_failed_rows_row_threshold": int(route_level["row_failed"].sum()),
        "aggregate_relative_gap": float(aggregate_relative_gap),
        "total_unallocated_miles": float(route_level["unallocated_miles"].sum()),
        "selected_worker_weight": selected_worker_weight,
        "successful_worker_weight": successful_worker_weight,
        "selected_commuter_car_weight": selected_car_weight,
        "successful_commuter_car_weight": successful_car_weight,
        "successful_commuter_car_share": float(successful_car_share),
        "coverage_accepted": coverage_accepted,
        "conservation_accepted": conservation_accepted,
        "omitted_coordinate_worker_weight": float(route_level["omitted_coordinate_worker_weight"].sum()),
        "omitted_car_share_worker_weight": float(route_level["omitted_car_share_worker_weight"].sum()),
        "same_tract_imputed_miles": float(seg["same_tract_imputed_miles"].sum()),
        "route_ratio_p50": float(route_ratios.quantile(0.5)) if not route_ratios.empty else float("nan"),
        "route_ratio_p90": float(route_ratios.quantile(0.9)) if not route_ratios.empty else float("nan"),
        "missing_required_column_count": 0,
        "invalid_numeric_count": 0,
        "duplicate_segment_count": 0,
        "unexplained_zero_count": 0,
        "route_total_conflict_count": 0,
    }


def _source_manifest_id(cache_dir: Path, year: int = DEFAULT_YEAR) -> str:
    manifest_path = _flow_artifact_paths(cache_dir, year)["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing validated LODES input manifest at {manifest_path}; run build_route_pilot_flows.py first"
        )
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"LODES input manifest is empty at {manifest_path}")
    _require_manifest_columns = {"path", "sha256", "bytes", "url", "retrieved_at_utc"}
    missing = _require_manifest_columns - set(frame.columns)
    if missing:
        raise ValueError(f"LODES input manifest missing columns: {sorted(missing)}")
    for _, row in frame.iterrows():
        source = Path(str(row["path"]))
        if not source.is_file() or source.stat().st_size != int(row["bytes"]) or _file_sha256(source) != str(row["sha256"]):
            raise RuntimeError(f"LODES cache failed manifest/checksum validation at {source}")
    sha_parts = frame["sha256"].astype(str).sort_values().tolist()
    return "|".join(sha_parts)


def _network_manifest_id(cache_dir: Path, year: int) -> str:
    manifest_path = cache_dir / "network" / str(year) / "network_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing OSRM network manifest at {manifest_path}; run build_route_pilot_network.py first"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest_id = str(payload.get("manifest_id") or "")
    if not manifest_id:
        raise ValueError(f"OSRM network manifest missing manifest_id at {manifest_path}")
    for source in payload.get("sources", []):
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_file() or _file_sha256(source_path) != str(source.get("sha256", "")):
            raise RuntimeError(f"OSRM source cache failed manifest/checksum validation at {source_path}")
    return manifest_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--cache-dir", type=Path, default=ROUTE_PILOT_CACHE)
    parser.add_argument("--county-boundaries-path", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--county-boundaries-source-url")
    parser.add_argument("--county-boundaries-attribution")
    parser.add_argument("--same-tract-mode", choices=sorted(SUPPORTED_SAME_TRACT_MODES), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument(
        "--route-workers",
        type=int,
        default=8,
        help="bounded number of concurrent OSRM route requests (default: 8)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="write one atomic route checkpoint after this many pairs (default: 500)",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    flow_paths = _flow_artifact_paths(cache_dir, int(args.year))
    pair_path = flow_paths["pairs"]
    if not pair_path.exists():
        raise FileNotFoundError(f"missing tract-pair cache at {pair_path}")

    boundary_path = _require_county_boundary_input(args.county_boundaries_path, int(args.year))
    county_boundaries = load_county_boundaries(boundary_path)
    ensure_county_boundary_manifest(
        boundary_path,
        int(args.year),
        source_url=args.county_boundaries_source_url,
        attribution=args.county_boundaries_attribution,
        source_crs=county_boundaries.attrs["source_crs"],
        official_default=boundary_path.resolve() == DEFAULT_BOUNDARY_PATH.resolve(),
    )
    pairs = pd.read_parquet(pair_path)
    pairs["source_manifest_id"] = _source_manifest_id(cache_dir, int(args.year))
    pairs["network_manifest_id"] = _network_manifest_id(cache_dir, int(args.year))

    session = requests.Session()
    route_cache_path = flow_paths["routes"]
    route_results = route_pairs_with_checkpoints(
        pairs,
        route_cache_path,
        args.base_url,
        session,
        max_workers=args.route_workers,
        checkpoint_every=args.checkpoint_every,
    )
    required_route_columns = {
        "route_id", "home_tract", "work_tract", "home_county", "work_county",
        "workers", "home_car_share", "commuter_car_weight",
        "block_pair_straight_line_miles", "commuter_car_miles", "status",
    }
    missing_route_columns = required_route_columns - set(route_results.columns)
    if missing_route_columns:
        raise ValueError(
            f"route checkpoint missing Task 2 weight metadata: {sorted(missing_route_columns)}"
        )
    if set(route_results["route_id"].astype(str)) != set(pairs["route_id"].astype(str)):
        raise ValueError("route checkpoint and pair artifact route_id sets differ")
    route_cache = route_results.copy()

    mode_paths = _mode_output_paths(cache_dir, int(args.year), args.same_tract_mode)
    segment_path = mode_paths["segments"]
    staging_segment_path = segment_path.with_name(
        f".{segment_path.stem}.{next(tempfile._get_candidate_names())}.tmp.parquet"
    )
    segments = allocate_cached_routes(route_cache, county_boundaries, staging_segment_path)
    segments = calibrate_same_tract_distance(pairs, segments, args.same_tract_mode)

    diagnostics = validate_mileage_conservation(segments)
    diagnostics_path = mode_paths["diagnostics"]
    _atomic_write_csv(pd.DataFrame([diagnostics]), diagnostics_path)
    if not diagnostics["accepted"]:
        segment_path.unlink(missing_ok=True)
        staging_segment_path.unlink(missing_ok=True)
        raise RuntimeError(
            "route mileage conservation failed validation or coverage: "
            f"row_failures={diagnostics['n_failed_rows_row_threshold']}, "
            f"aggregate_relative_gap={diagnostics['aggregate_relative_gap']:.6f}, "
            f"invalid_numeric={diagnostics.get('invalid_numeric_count', 0)}, "
            f"duplicate_segments={diagnostics.get('duplicate_segment_count', 0)}, "
            f"unexplained_zero={diagnostics.get('unexplained_zero_count', 0)}"
        )

    _atomic_write_parquet(segments, segment_path)
    staging_segment_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
