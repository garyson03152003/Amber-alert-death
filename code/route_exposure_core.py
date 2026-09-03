"""Pure helpers for Wisconsin route-exposure construction.

The functions in this module are intentionally import-safe and side-effect
free. They accept pandas inputs, return pandas outputs, and keep the geometry
operations local so the later pipeline stages can build on them without
reaching into routing or download code.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape
from shapely.ops import transform as geom_transform

EARTH_RADIUS_M = 6_371_008.8
MILES_PER_METER = 1.0 / 1609.344
ROUTE_SAMPLE_STEP_M = 250.0


def _normalize_code(value: object, width: int | None = None) -> str:
    if pd.isna(value):
        raise ValueError("missing geocode or fips value")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if width is not None:
        text = text.zfill(width)
    return text


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _finite_series(df: pd.DataFrame, cols: tuple[str, str]) -> pd.Series:
    return np.isfinite(pd.to_numeric(df[cols[0]], errors="coerce")) & np.isfinite(
        pd.to_numeric(df[cols[1]], errors="coerce")
    )


def weighted_tract_pairs(
    block_flows: pd.DataFrame,
    block_crosswalk: pd.DataFrame,
    tract_car_share: pd.Series,
) -> pd.DataFrame:
    """Collapse block-level LODES flows to tract pairs with weighted endpoints."""
    if block_flows.empty:
        return pd.DataFrame(
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
                "block_pair_straight_line_miles",
            ]
        )

    flows = block_flows.copy(deep=True)
    crosswalk = block_crosswalk.copy(deep=True)
    shares = tract_car_share.copy(deep=True)

    _require_columns(flows, ["h_geocode", "w_geocode", "S000"], label="block_flows")
    _require_columns(
        crosswalk,
        ["tabblk2020", "cty", "trct", "blklatdd", "blklondd"],
        label="block_crosswalk",
    )

    flows["h_geocode"] = flows["h_geocode"].map(lambda v: _normalize_code(v))
    flows["w_geocode"] = flows["w_geocode"].map(lambda v: _normalize_code(v))
    flows["workers"] = pd.to_numeric(flows["S000"], errors="coerce")
    flows = flows.dropna(subset=["workers"]).copy()
    flows = flows[flows["workers"] > 0].copy()

    crosswalk = crosswalk.loc[
        :, ["tabblk2020", "cty", "trct", "blklatdd", "blklondd"]
    ].drop_duplicates("tabblk2020")
    crosswalk["tabblk2020"] = crosswalk["tabblk2020"].map(lambda v: _normalize_code(v))
    crosswalk["cty"] = crosswalk["cty"].map(lambda v: _normalize_code(v, 5))
    crosswalk["trct"] = crosswalk["trct"].map(lambda v: _normalize_code(v, 11))
    crosswalk["blklatdd"] = pd.to_numeric(crosswalk["blklatdd"], errors="coerce")
    crosswalk["blklondd"] = pd.to_numeric(crosswalk["blklondd"], errors="coerce")

    home = crosswalk.rename(
        columns={
            "tabblk2020": "h_geocode",
            "cty": "home_county",
            "trct": "home_tract",
            "blklatdd": "home_lat",
            "blklondd": "home_lon",
        }
    )
    work = crosswalk.rename(
        columns={
            "tabblk2020": "w_geocode",
            "cty": "work_county",
            "trct": "work_tract",
            "blklatdd": "work_lat",
            "blklondd": "work_lon",
        }
    )

    merged = flows.merge(home, on="h_geocode", how="left").merge(work, on="w_geocode", how="left")
    required_join_cols = [
        "home_county",
        "home_tract",
        "work_county",
        "work_tract",
    ]
    if merged[required_join_cols].isna().any().any():
        raise ValueError("block crosswalk is missing tract or county assignments for at least one block")

    merged["valid_endpoints"] = _finite_series(merged, ("home_lat", "home_lon")) & _finite_series(
        merged, ("work_lat", "work_lon")
    )
    merged["block_pair_straight_line_miles"] = np.nan
    valid = merged["valid_endpoints"]
    if valid.any():
        lat1 = np.radians(pd.to_numeric(merged.loc[valid, "home_lat"]))
        lat2 = np.radians(pd.to_numeric(merged.loc[valid, "work_lat"]))
        dlat = lat2 - lat1
        dlon = np.radians(
            pd.to_numeric(merged.loc[valid, "work_lon"])
            - pd.to_numeric(merged.loc[valid, "home_lon"])
        )
        hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        merged.loc[valid, "block_pair_straight_line_miles"] = (
            2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(hav)) * MILES_PER_METER
        )

    group_cols = ["home_tract", "work_tract", "home_county", "work_county"]
    value_cols = [
        "home_lat",
        "home_lon",
        "work_lat",
        "work_lon",
        "block_pair_straight_line_miles",
    ]
    weights = pd.to_numeric(merged["workers"], errors="coerce")
    finite_positive_weight = np.isfinite(weights) & (weights > 0)
    merged["_valid_endpoint_workers"] = weights.where(merged["valid_endpoints"], 0.0)
    named_aggregations: dict[str, tuple[str, str]] = {
        "workers": ("workers", "sum"),
        "valid_endpoint_workers": ("_valid_endpoint_workers", "sum"),
    }
    for value_col in value_cols:
        values = pd.to_numeric(merged[value_col], errors="coerce")
        mean_mask = merged["valid_endpoints"] & finite_positive_weight & np.isfinite(values)
        numerator_col = f"_{value_col}_weighted_sum"
        denominator_col = f"_{value_col}_weight_sum"
        merged[numerator_col] = (values * weights).where(mean_mask, 0.0)
        merged[denominator_col] = weights.where(mean_mask, 0.0)
        named_aggregations[numerator_col] = (numerator_col, "sum")
        named_aggregations[denominator_col] = (denominator_col, "sum")

    out = (
        merged.groupby(group_cols, sort=True, dropna=False, as_index=False)
        .agg(**named_aggregations)
    )
    out["missing_endpoint_workers"] = out["workers"] - out["valid_endpoint_workers"]
    for value_col in value_cols:
        numerator_col = f"_{value_col}_weighted_sum"
        denominator_col = f"_{value_col}_weight_sum"
        out[value_col] = (
            out[numerator_col] / out[denominator_col]
        ).where(out[denominator_col] > 0, np.nan)
        out = out.drop(columns=[numerator_col, denominator_col])
    out = out[
        group_cols
        + [
            "workers",
            "valid_endpoint_workers",
            "missing_endpoint_workers",
            *value_cols,
        ]
    ]
    shares.index = shares.index.map(lambda v: _normalize_code(v, 11))
    shares = pd.to_numeric(shares, errors="coerce")
    out["home_car_share"] = out["home_tract"].map(shares)
    out["missing_home_car_share_workers"] = np.where(
        out["home_car_share"].isna(), out["workers"], 0.0
    )
    return out.sort_values(["home_tract", "work_tract"]).reset_index(drop=True)


def parse_osrm_route(payload: dict, route_id: str) -> dict:
    """Validate and extract the first OSRM route from a JSON response."""
    if not isinstance(payload, dict):
        raise ValueError(f"OSRM route {route_id} response must be a dict")
    code = payload.get("code")
    message = payload.get("message") or payload.get("error") or ""
    if code != "Ok":
        raise ValueError(f"OSRM route {route_id} failed: {code}: {message}")

    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"OSRM route {route_id} response missing routes")
    route = routes[0]
    if not isinstance(route, dict):
        raise ValueError(f"OSRM route {route_id} response route must be a dict")

    geometry = route.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "LineString":
        raise ValueError(f"OSRM route {route_id} geometry must be a GeoJSON LineString")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError(f"OSRM route {route_id} geometry must contain at least two coordinates")

    try:
        distance_m = float(route["distance"])
        duration_s = float(route["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"OSRM route {route_id} missing distance or duration") from exc

    return {
        "route_id": str(route_id),
        "distance_m": distance_m,
        "duration_s": duration_s,
        "geometry": copy.deepcopy(geometry),
    }


def classify_route_origin(home_fips: str, work_fips: str, outcome_fips: str) -> str:
    home = _normalize_code(home_fips, 5)
    work = _normalize_code(work_fips, 5)
    outcome = _normalize_code(outcome_fips, 5)
    if home == outcome:
        return "own_origin"
    if work == outcome:
        return "cross_origin"
    return "pass_through"


def _lonlat_to_xy(
    lon: float,
    lat: float,
    *,
    lon0: float,
    lat0: float,
) -> tuple[float, float]:
    scale = math.cos(math.radians(lat0))
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * scale
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def _project_geometry(coords: list[list[float]], *, lon0: float, lat0: float) -> list[tuple[float, float]]:
    return [_lonlat_to_xy(float(lon), float(lat), lon0=lon0, lat0=lat0) for lon, lat in coords]


def _county_geometry_from_value(geometry: object):
    if hasattr(geometry, "__geo_interface__"):
        geometry = geometry.__geo_interface__
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if not isinstance(geometry, dict):
        raise ValueError("county geometry must be a GeoJSON-like mapping")
    gtype = geometry.get("type")
    if gtype not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"unsupported county geometry type: {gtype}")
    return shape(geometry)


def _route_transformer(route_coords: list[tuple[float, float]]) -> Transformer:
    lon0 = float(sum(lon for lon, _ in route_coords) / len(route_coords))
    lat0 = float(sum(lat for _, lat in route_coords) / len(route_coords))
    proj = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", proj, always_xy=True)


def _route_total_miles(route_geometry: dict) -> float:
    properties = route_geometry.get("properties", {})
    distance_m = properties.get("distance_m", properties.get("distance"))
    if distance_m is not None and pd.notna(distance_m):
        return float(distance_m) * MILES_PER_METER

    coords = route_geometry.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        raise ValueError("route geometry must contain at least two coordinates")
    total_m = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        total_m += _segment_length_m(float(lon1), float(lat1), float(lon2), float(lat2))
    return total_m * MILES_PER_METER


def _segment_length_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def allocate_route_miles(
    route_geojson: dict,
    counties: pd.DataFrame,
    route_id: str,
) -> pd.DataFrame:
    """Allocate one routed LineString across county polygons."""
    if not isinstance(route_geojson, dict):
        raise ValueError("route geometry must be a GeoJSON-like dict")
    if route_geojson.get("type") != "LineString":
        raise ValueError(f"route {route_id} must be a LineString geometry")

    coords = route_geojson.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        raise ValueError(f"route {route_id} must contain at least two coordinates")
    coord_pairs = [(float(lon), float(lat)) for lon, lat in coords]
    route_line = LineString(coord_pairs)
    if not route_line.is_valid or route_line.is_empty:
        raise ValueError(f"route {route_id} geometry is invalid")

    transformer = _route_transformer(coord_pairs)
    route_projected = geom_transform(transformer.transform, route_line)
    total_projected_m = float(route_projected.length)
    if total_projected_m <= 0:
        raise ValueError(f"route {route_id} has zero length")
    total_miles = _route_total_miles(route_geojson)

    county_id_col = next(
        (col for col in ("county_fips", "fips", "GEOID", "county") if col in counties.columns),
        None,
    )
    if county_id_col is None:
        raise ValueError("counties must contain a county identifier column")
    if "geometry" not in counties.columns:
        raise ValueError("counties must contain a geometry column")

    county_defs = []
    for _, row in counties.copy(deep=True).iterrows():
        county_fips = _normalize_code(row[county_id_col], 5)
        county_geom = _county_geometry_from_value(row["geometry"])
        county_projected = geom_transform(transformer.transform, county_geom)
        county_defs.append((county_fips, county_projected))

    county_miles: dict[str, float] = defaultdict(float)
    for county_fips, county_geom in county_defs:
        intersection = route_projected.intersection(county_geom)
        if intersection.is_empty:
            continue
        county_miles[county_fips] += float(intersection.length)

    projected_allocated_m = sum(county_miles.values())
    scale = total_miles / total_projected_m
    county_miles = {county_fips: miles * scale for county_fips, miles in county_miles.items()}
    allocated_miles = projected_allocated_m * scale
    unallocated_miles = max(total_miles - allocated_miles, 0.0)
    if unallocated_miles <= max(1e-6, total_miles * 1e-4) and county_miles:
        county_total = sum(county_miles.values())
        if county_total > 0:
            renorm = total_miles / county_total
            county_miles = {county_fips: miles * renorm for county_fips, miles in county_miles.items()}
        unallocated_miles = 0.0

    rows = []
    for county_fips, miles in sorted(county_miles.items()):
        if miles <= 0:
            continue
        rows.append(
            {
                "route_id": str(route_id),
                "county_fips": county_fips,
                "route_miles_total": total_miles,
                "route_miles_in_county": miles,
                "unallocated_miles": 0.0,
                "segment_type": "county",
            }
        )
    if unallocated_miles > 0:
        rows.append(
            {
                "route_id": str(route_id),
                "county_fips": None,
                "route_miles_total": total_miles,
                "route_miles_in_county": 0.0,
                "unallocated_miles": unallocated_miles,
                "segment_type": "unallocated",
            }
        )

    return pd.DataFrame(rows)


def build_county_exposure(
    route_segments: pd.DataFrame,
    alert_home_counties: pd.DataFrame,
    denominator_mode: str = "all_region_routes",
) -> pd.DataFrame:
    """Build county-date commuter-car exposure with a shared denominator."""
    _ = denominator_mode  # reserved for later same-tract-mode variations
    if route_segments.empty or alert_home_counties.empty:
        return pd.DataFrame(
            columns=[
                "outcome_fips",
                "alert_date",
                "total_commuter_car_miles",
                "own_commuter_car_miles",
                "cross_commuter_car_miles",
                "pass_through_commuter_car_miles",
                "affected_commuter_car_miles",
                "own_affected_car_miles",
                "cross_affected_car_miles",
                "pass_through_affected_car_miles",
                "affected_route_share",
                "affected_commuter_car_miles_per_10000",
            ]
        )

    seg = route_segments.copy(deep=True)
    alerts = alert_home_counties.copy(deep=True)
    _require_columns(
        seg,
        [
            "outcome_fips",
            "home_fips",
            "work_fips",
            "workers",
            "home_car_share",
            "route_miles_in_county",
        ],
        label="route_segments",
    )

    if "home_fips" not in alerts.columns and "fips" in alerts.columns:
        alerts = alerts.rename(columns={"fips": "home_fips"})
    if "alert_date" not in alerts.columns and "effective_crash_date" in alerts.columns:
        alerts = alerts.rename(columns={"effective_crash_date": "alert_date"})
    _require_columns(alerts, ["home_fips", "alert_date"], label="alert_home_counties")

    seg["outcome_fips"] = seg["outcome_fips"].map(lambda v: _normalize_code(v, 5))
    seg["home_fips"] = seg["home_fips"].map(lambda v: _normalize_code(v, 5))
    seg["work_fips"] = seg["work_fips"].map(lambda v: _normalize_code(v, 5))
    seg["workers"] = pd.to_numeric(seg["workers"], errors="coerce")
    seg["home_car_share"] = pd.to_numeric(seg["home_car_share"], errors="coerce")
    seg["route_miles_in_county"] = pd.to_numeric(seg["route_miles_in_county"], errors="coerce")
    if seg[["workers", "home_car_share", "route_miles_in_county"]].isna().any().any():
        raise ValueError("route_segments contains nonnumeric workers, home_car_share, or route_miles_in_county")

    seg["commuter_car_miles"] = seg["workers"] * seg["home_car_share"] * seg["route_miles_in_county"]
    seg["route_origin"] = seg.apply(
        lambda row: classify_route_origin(row["home_fips"], row["work_fips"], row["outcome_fips"]),
        axis=1,
    )

    county_totals = (
        seg.groupby("outcome_fips", as_index=False)
        .agg(total_commuter_car_miles=("commuter_car_miles", "sum"))
        .sort_values("outcome_fips")
        .reset_index(drop=True)
    )
    if county_totals.empty or (county_totals["total_commuter_car_miles"] <= 0).any():
        raise ValueError("zero denominator")

    county_class_totals = (
        seg.pivot_table(
            index="outcome_fips",
            columns="route_origin",
            values="commuter_car_miles",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(
            columns={
                "own_origin": "own_commuter_car_miles",
                "cross_origin": "cross_commuter_car_miles",
                "pass_through": "pass_through_commuter_car_miles",
            }
        )
        .reset_index()
    )
    for col in [
        "own_commuter_car_miles",
        "cross_commuter_car_miles",
        "pass_through_commuter_car_miles",
    ]:
        if col not in county_class_totals.columns:
            county_class_totals[col] = 0.0

    county_base = county_totals.merge(county_class_totals, on="outcome_fips", how="left").fillna(0.0)

    alerts["home_fips"] = alerts["home_fips"].map(lambda v: _normalize_code(v, 5))
    alerts["alert_date"] = pd.to_datetime(alerts["alert_date"]).dt.normalize()
    alerts = alerts.drop_duplicates(subset=["home_fips", "alert_date"]).copy()
    if alerts.empty:
        return pd.DataFrame(columns=["outcome_fips", "alert_date"])

    alert_dates = pd.DataFrame({"alert_date": sorted(alerts["alert_date"].unique())})
    panel = county_base.merge(alert_dates, how="cross")

    affected = seg.merge(alerts, on="home_fips", how="inner")
    if affected.empty:
        for col in [
            "affected_commuter_car_miles",
            "own_affected_car_miles",
            "cross_affected_car_miles",
            "pass_through_affected_car_miles",
        ]:
            panel[col] = 0.0
    else:
        affected["alert_date"] = pd.to_datetime(affected["alert_date"]).dt.normalize()
        affected_class = (
            affected.groupby(["outcome_fips", "alert_date", "route_origin"], as_index=False)
            .agg(affected_commuter_car_miles=("commuter_car_miles", "sum"))
        )
        affected_total = (
            affected.groupby(["outcome_fips", "alert_date"], as_index=False)
            .agg(affected_commuter_car_miles=("commuter_car_miles", "sum"))
        )
        affected_pivot = (
            affected_class.pivot_table(
                index=["outcome_fips", "alert_date"],
                columns="route_origin",
                values="affected_commuter_car_miles",
                aggfunc="sum",
                fill_value=0.0,
            )
            .rename(
                columns={
                    "own_origin": "own_affected_car_miles",
                    "cross_origin": "cross_affected_car_miles",
                    "pass_through": "pass_through_affected_car_miles",
                }
            )
            .reset_index()
        )
        panel = panel.merge(affected_total, on=["outcome_fips", "alert_date"], how="left")
        panel = panel.merge(affected_pivot, on=["outcome_fips", "alert_date"], how="left")
        for col in [
            "affected_commuter_car_miles",
            "own_affected_car_miles",
            "cross_affected_car_miles",
            "pass_through_affected_car_miles",
        ]:
            if col not in panel.columns:
                panel[col] = 0.0

    for col in [
        "affected_commuter_car_miles",
        "own_affected_car_miles",
        "cross_affected_car_miles",
        "pass_through_affected_car_miles",
    ]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)

    panel["affected_route_share"] = panel["affected_commuter_car_miles"] / panel["total_commuter_car_miles"]
    panel["affected_commuter_car_miles_per_10000"] = (
        panel["affected_route_share"] * 10000.0
    )
    panel["own_affected_route_share"] = panel["own_affected_car_miles"] / panel["total_commuter_car_miles"]
    panel["cross_affected_route_share"] = panel["cross_affected_car_miles"] / panel["total_commuter_car_miles"]
    panel["pass_through_affected_route_share"] = (
        panel["pass_through_affected_car_miles"] / panel["total_commuter_car_miles"]
    )
    panel = panel.sort_values(["outcome_fips", "alert_date"]).reset_index(drop=True)
    return panel
