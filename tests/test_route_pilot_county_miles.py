import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import build_route_pilot_county_miles as county_miles
from build_route_pilot_county_miles import (
    allocate_cached_routes,
    calibrate_same_tract_distance,
    load_county_boundaries,
    main,
    validate_mileage_conservation,
)


def _write_geometry(tmp_path: Path, route_id: str, coordinates: list[list[float]]) -> Path:
    path = tmp_path / f"{route_id}.geojson"
    payload = {
        "route_id": route_id,
        "status": "Ok",
        "distance_m": 80000.0,
        "duration_s": 3600.0,
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fixture_two_county_boundaries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_fips": ["55001", "55003"],
            "geometry": [
                {
                    "type": "Polygon",
                    "coordinates": [[[-90.0, 42.5], [-89.5, 42.5], [-89.5, 43.5], [-90.0, 43.5], [-90.0, 42.5]]],
                },
                {
                    "type": "Polygon",
                    "coordinates": [[[-89.5, 42.5], [-89.0, 42.5], [-89.0, 43.5], [-89.5, 43.5], [-89.5, 42.5]]],
                },
            ],
        }
    )


def _write_county_geojson(tmp_path: Path) -> Path:
    path = tmp_path / "counties.geojson"
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"GEOID": "55001"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-90.0, 42.5], [-89.5, 42.5], [-89.5, 43.5], [-90.0, 43.5], [-90.0, 42.5]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"GEOID": "55003"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-89.5, 42.5], [-89.0, 42.5], [-89.0, 43.5], [-89.5, 43.5], [-89.5, 42.5]]],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fixture_route_crossing_two_counties(tmp_path: Path) -> pd.DataFrame:
    geometry_path = _write_geometry(tmp_path, "r1", [[-90.0, 43.0], [-89.0, 43.0]])
    return pd.DataFrame(
        {
            "route_id": ["r1"],
            "status": ["Ok"],
            "geometry_path": [str(geometry_path)],
            "distance_m": [80000.0],
            "duration_s": [3600.0],
            "home_tract": ["55001000100"],
            "work_tract": ["55003000100"],
            "home_county": ["55001"],
            "work_county": ["55003"],
            "workers": [100.0],
            "home_car_share": [0.8],
            "same_tract": [False],
            "home_lon": [-90.0],
            "home_lat": [43.0],
            "work_lon": [-89.0],
            "work_lat": [43.0],
        }
    )


def fixture_segments_with_unallocated_miles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["r1", "r1"],
            "segment_type": ["county", "unallocated"],
            "route_miles_total": [10.0, 10.0],
            "route_miles_in_county": [8.0, 0.0],
            "unallocated_miles": [0.0, 2.0],
            "workers": [50.0, 50.0],
            "home_car_share": [0.7, 0.7],
            "straight_line_miles": [7.0, 7.0],
            "same_tract_imputed_miles": [0.0, 0.0],
        }
    )


def test_load_county_boundaries_reads_geojson_feature_collection(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"GEOID": "55001"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-90.0, 42.5], [-89.5, 42.5], [-89.5, 43.5], [-90.0, 43.5], [-90.0, 42.5]]],
                },
            }
        ],
    }
    path = tmp_path / "counties.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = load_county_boundaries(path)

    assert list(out["county_fips"]) == ["55001"]
    assert out.loc[0, "geometry"].geom_type == "Polygon"


def test_load_county_boundaries_rejects_non_lonlat_crs(tmp_path):
    path = _write_county_geojson(tmp_path)
    payload = json.loads(path.read_text())
    payload["crs"] = {"type": "name", "properties": {"name": "EPSG:3857"}}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="EPSG:4326 or CRS84"):
        load_county_boundaries(path)


def test_allocate_cached_routes_keeps_multi_county_segments(tmp_path):
    routes = fixture_route_crossing_two_counties(tmp_path)
    routes["commuter_car_weight"] = 80.0
    routes["block_pair_straight_line_miles"] = 40.0
    routes["commuter_car_miles"] = 3200.0
    counties = fixture_two_county_boundaries()

    out = allocate_cached_routes(routes, counties, tmp_path / "unused.parquet")

    assert set(out["county_fips"].dropna()) == {"55001", "55003"}
    assert set(out["outcome_fips"].dropna()) == {"55001", "55003"}
    assert out.groupby("route_id")["route_miles_in_county"].sum().iloc[0] == pytest.approx(
        routes.loc[0, "distance_m"] / 1609.344, rel=0.005
    )
    assert out[["route_id", "workers", "home_car_share", "commuter_car_weight", "block_pair_straight_line_miles", "commuter_car_miles"]].drop_duplicates().iloc[0].to_dict() == {
        "route_id": "r1",
        "workers": 100.0,
        "home_car_share": 0.8,
        "commuter_car_weight": 80.0,
        "block_pair_straight_line_miles": 40.0,
        "commuter_car_miles": 3200.0,
    }


def test_validate_mileage_conservation_flags_unallocated_route():
    result = validate_mileage_conservation(fixture_segments_with_unallocated_miles())

    assert result["n_failed_rows"] == 0
    assert result["total_unallocated_miles"] > 0
    assert result["accepted"] is True


def test_calibrate_same_tract_distance_primary_preserves_non_negligible_routed_same_tract():
    pairs = pd.DataFrame(
        {
            "home_tract": ["55001000100", "55001000200"],
            "work_tract": ["55003000100", "55001000200"],
            "home_county": ["55001", "55001"],
            "work_county": ["55003", "55001"],
            "workers": [100.0, 50.0],
            "home_car_share": [0.8, 0.7],
            "same_tract": [False, True],
            "home_lon": [-90.0, -89.5],
            "home_lat": [43.0, 43.0],
            "work_lon": [-89.0, -89.49],
            "work_lat": [43.0, 43.0],
        }
    )
    segments = pd.DataFrame(
        {
            "route_id": ["55001000100__55003000100", "55001000200__55001000200"],
            "home_tract": ["55001000100", "55001000200"],
            "work_tract": ["55003000100", "55001000200"],
            "home_fips": ["55001", "55001"],
            "work_fips": ["55003", "55001"],
            "workers": [100.0, 50.0],
            "home_car_share": [0.8, 0.7],
            "same_tract": [False, True],
            "status": ["Ok", "Ok"],
            "distance_m": [80000.0, 0.0],
            "duration_s": [3600.0, 0.0],
            "geometry_path": [pd.NA, pd.NA],
            "source_manifest_id": ["src", "src"],
            "network_manifest_id": ["net", "net"],
            "home_lon": [-90.0, -89.5],
            "home_lat": [43.0, 43.0],
            "work_lon": [-89.0, -89.49],
            "work_lat": [43.0, 43.0],
            "straight_line_miles": [45.0, 0.5],
            "outcome_fips": ["55001", "55001"],
            "route_miles_total": [49.7, 5.0],
            "route_miles_in_county": [49.7, 5.0],
            "unallocated_miles": [0.0, 0.0],
            "segment_type": ["county", "county"],
            "same_tract_imputed_miles": [0.0, 0.0],
            "same_tract_mode": ["routed", "routed"],
        }
    )

    out = calibrate_same_tract_distance(pairs, segments, "primary_calibrated")
    same = out.loc[out["route_id"] == "55001000200__55001000200"].iloc[0]

    assert same["outcome_fips"] == "55001"
    assert same["route_miles_in_county"] == pytest.approx(5.0)
    assert same["same_tract_imputed_miles"] == 0
    assert same["same_tract_mode"] == "primary_calibrated"


def test_calibrate_same_tract_distance_primary_imputes_negligible_from_block_pair_distance_and_class():
    pairs = pd.DataFrame(
        {
            "route_id": ["cal", "same"],
            "home_tract": ["55001000100", "55001000200"],
            "work_tract": ["55003000100", "55001000200"],
            "home_county": ["55001", "55001"],
            "work_county": ["55003", "55001"],
            "workers": [100.0, 50.0],
            "home_car_share": [0.8, 0.7],
            "same_tract": [False, True],
            "urban_rural_class": ["urban", "urban"],
            "block_pair_straight_line_miles": [10.0, 2.0],
            "home_lon": [-90.0, -89.5],
            "home_lat": [43.0, 43.0],
            "work_lon": [-89.9, -89.49],
            "work_lat": [43.0, 43.0],
        }
    )
    segments = pd.DataFrame(
        {
            "route_id": ["cal", "same"],
            "home_tract": pairs["home_tract"],
            "work_tract": pairs["work_tract"],
            "home_fips": ["55001", "55001"],
            "work_fips": ["55003", "55001"],
            "workers": [100.0, 50.0],
            "home_car_share": [0.8, 0.7],
            "same_tract": [False, True],
            "status": ["Ok", "Ok"],
            "straight_line_miles": [10.0, 2.0],
            "outcome_fips": ["55001", "55001"],
            "county_fips": ["55001", "55001"],
            "route_miles_total": [15.0, 0.05],
            "route_miles_in_county": [15.0, 0.05],
            "unallocated_miles": [0.0, 0.0],
            "segment_type": ["county", "county"],
            "same_tract_imputed_miles": [0.0, 0.0],
            "same_tract_mode": ["routed", "routed"],
        }
    )

    out = calibrate_same_tract_distance(pairs, segments, "primary_calibrated")
    same = out.loc[out["route_id"].eq("same")].iloc[0]

    assert same["route_miles_in_county"] == pytest.approx(3.0)
    assert same["same_tract_imputed_miles"] == pytest.approx(3.0)
    assert same["urban_rural_class"] == "urban"


@pytest.mark.parametrize(
    "mode,expected_type",
    [
        ("primary_calibrated", "same_tract_imputed"),
        ("zero", "same_tract_zero"),
        ("exclude", "same_tract_excluded"),
    ],
)
def test_same_tract_rebuilt_rows_preserve_route_weight_schema(mode, expected_type):
    pairs = pd.DataFrame(
        {
            "route_id": ["cal", "same"],
            "home_tract": ["55001000100", "55001000200"],
            "work_tract": ["55003000100", "55001000200"],
            "home_county": ["55001", "55001"],
            "work_county": ["55003", "55001"],
            "workers": [100.0, 10.0],
            "home_car_share": [0.8, 0.7],
            "commuter_car_weight": [80.0, 7.0],
            "block_pair_straight_line_miles": [10.0, 2.0],
            "commuter_car_miles": [800.0, 14.0],
            "routing_eligible": [True, True],
            "omitted_coordinate_worker_weight": [0.0, 0.0],
            "omitted_car_share_worker_weight": [0.0, 0.0],
            "same_tract": [False, True],
            "urban_rural_class": ["urban", "urban"],
            "source_manifest_id": ["src", "src"],
            "network_manifest_id": ["net", "net"],
            "home_lon": [-90.0, -89.5],
            "home_lat": [43.0, 43.0],
            "work_lon": [-89.9, -89.49],
            "work_lat": [43.0, 43.0],
        }
    )
    segments = pd.DataFrame(
        {
            "route_id": ["cal", "same"],
            "home_tract": pairs["home_tract"],
            "work_tract": pairs["work_tract"],
            "home_fips": ["55001", "55001"],
            "work_fips": ["55003", "55001"],
            "workers": [100.0, 10.0],
            "home_car_share": [0.8, 0.7],
            "commuter_car_weight": [80.0, 7.0],
            "block_pair_straight_line_miles": [10.0, 2.0],
            "commuter_car_miles": [800.0, 14.0],
            "routing_eligible": [True, True],
            "omitted_coordinate_worker_weight": [0.0, 0.0],
            "omitted_car_share_worker_weight": [0.0, 0.0],
            "same_tract": [False, True],
            "status": ["Ok", "Ok"],
            "source_manifest_id": ["src", "src"],
            "network_manifest_id": ["net", "net"],
            "straight_line_miles": [10.0, 2.0],
            "county_fips": ["55001", "55001"],
            "outcome_fips": ["55001", "55001"],
            "route_miles_total": [15.0, 0.05],
            "route_miles_in_county": [15.0, 0.05],
            "unallocated_miles": [0.0, 0.0],
            "segment_type": ["county", "county"],
            "same_tract_imputed_miles": [0.0, 0.0],
            "same_tract_mode": ["routed", "routed"],
        }
    )

    out = calibrate_same_tract_distance(pairs, segments, mode)
    rebuilt = out.loc[out["route_id"].eq("same")].iloc[0]

    assert rebuilt["segment_type"] == expected_type
    assert rebuilt[
        [
            "route_id",
            "workers",
            "home_car_share",
            "commuter_car_weight",
            "block_pair_straight_line_miles",
            "commuter_car_miles",
            "routing_eligible",
            "omitted_coordinate_worker_weight",
            "omitted_car_share_worker_weight",
            "source_manifest_id",
            "network_manifest_id",
        ]
    ].to_dict() == {
        "route_id": "same",
        "workers": 10.0,
        "home_car_share": 0.7,
        "commuter_car_weight": 7.0,
        "block_pair_straight_line_miles": 2.0,
        "commuter_car_miles": 14.0,
        "routing_eligible": True,
        "omitted_coordinate_worker_weight": 0.0,
        "omitted_car_share_worker_weight": 0.0,
        "source_manifest_id": "src",
        "network_manifest_id": "net",
    }
    coverage = validate_mileage_conservation(out)
    assert coverage["selected_commuter_car_weight"] == pytest.approx(87.0)


def test_conservation_gate_counts_selected_weight_once_per_route_and_requires_99_percent():
    segments = pd.DataFrame(
        {
            "route_id": ["ok", "ok", "failed"],
            "segment_type": ["county", "county", "failed_route"],
            "route_miles_total": [10.0, 10.0, 0.0],
            "route_miles_in_county": [4.0, 6.0, 0.0],
            "unallocated_miles": [0.0, 0.0, 0.0],
            "workers": [99.0, 99.0, 2.0],
            "home_car_share": [1.0, 1.0, 1.0],
            "straight_line_miles": [8.0, 8.0, 1.0],
            "same_tract_imputed_miles": [0.0, 0.0, 0.0],
        }
    )

    result = validate_mileage_conservation(segments)

    assert result["selected_commuter_car_weight"] == pytest.approx(101.0)
    assert result["successful_commuter_car_weight"] == pytest.approx(99.0)
    assert result["successful_commuter_car_share"] == pytest.approx(99 / 101)
    assert result["coverage_accepted"] is False
    assert result["accepted"] is False


def test_conservation_gate_excludes_pre_route_input_omissions_from_selected_weight():
    segments = pd.DataFrame(
        {
            "route_id": ["ok", "missing-car"],
            "segment_type": ["county", "failed_route"],
            "route_miles_total": [10.0, 0.0],
            "route_miles_in_county": [10.0, 0.0],
            "unallocated_miles": [0.0, 0.0],
            "workers": [99.0, 1000.0],
            "home_car_share": [1.0, pd.NA],
            "commuter_car_weight": [99.0, pd.NA],
            "routing_eligible": [True, False],
            "straight_line_miles": [8.0, pd.NA],
            "same_tract_imputed_miles": [0.0, 0.0],
            "omitted_car_share_worker_weight": [0.0, 1000.0],
        }
    )

    result = validate_mileage_conservation(segments)

    assert result["selected_worker_weight"] == 99
    assert result["selected_commuter_car_weight"] == 99
    assert result["successful_commuter_car_share"] == 1
    assert result["omitted_car_share_worker_weight"] == 1000
    assert result["accepted"] is True


@pytest.mark.parametrize(
    "mutation,expected_metric",
    [
        (lambda frame: frame.assign(route_miles_total=pd.NA), "invalid_numeric_count"),
        (lambda frame: frame.assign(route_miles_in_county=float("inf")), "invalid_numeric_count"),
        (lambda frame: frame.assign(unallocated_miles=-1.0), "invalid_numeric_count"),
        (lambda frame: pd.concat([frame, frame], ignore_index=True), "duplicate_segment_count"),
        (
            lambda frame: frame.assign(
                route_miles_total=0.0,
                route_miles_in_county=0.0,
                segment_type="county",
            ),
            "unexplained_zero_count",
        ),
        (
            lambda frame: frame.assign(
                route_miles_in_county=0.0,
                unallocated_miles=0.0,
                segment_type="unallocated",
            ),
            "unexplained_zero_count",
        ),
        (lambda frame: frame.drop(columns="unallocated_miles"), "missing_required_column_count"),
    ],
)
def test_conservation_rejects_invalid_duplicate_and_unexplained_zero_mileage(
    mutation, expected_metric
):
    valid = pd.DataFrame(
        {
            "route_id": ["r1"],
            "county_fips": ["55001"],
            "segment_type": ["county"],
            "route_miles_total": [10.0],
            "route_miles_in_county": [10.0],
            "unallocated_miles": [0.0],
            "workers": [10.0],
            "home_car_share": [0.8],
            "commuter_car_weight": [8.0],
            "routing_eligible": [True],
            "straight_line_miles": [8.0],
            "same_tract_imputed_miles": [0.0],
        }
    )

    result = validate_mileage_conservation(mutation(valid))

    assert result["accepted"] is False
    assert result[expected_metric] > 0


def test_mode_output_paths_are_distinct():
    paths = {
        mode: county_miles._mode_output_paths(Path("cache"), 2022, mode)
        for mode in ("primary_calibrated", "zero", "exclude")
    }
    assert len({value["segments"] for value in paths.values()}) == 3
    assert len({value["diagnostics"] for value in paths.values()}) == 3
    assert paths["primary_calibrated"]["segments"].name == "route_county_segments_2022.parquet"
    assert paths["primary_calibrated"]["diagnostics"].name == "route_county_diagnostics_2022.csv"
    assert "_zero" in paths["zero"]["segments"].stem


def test_county_mile_flow_inputs_follow_requested_year():
    default = county_miles._flow_artifact_paths(Path("cache"))
    prior = county_miles._flow_artifact_paths(Path("cache"), 2021)

    assert default["pairs"].name == county_miles.PAIR_OUTPUT_NAME
    assert default["manifest"].name == county_miles.INPUT_MANIFEST_NAME
    assert default["routes"].name == county_miles.ROUTE_CACHE_NAME
    assert set(default.values()).isdisjoint(set(prior.values()))


def test_missing_county_boundary_error_names_official_prerequisite(tmp_path):
    missing = tmp_path / "tl_2022_us_county.geojson"

    with pytest.raises(FileNotFoundError, match=r"www2\.census\.gov.*ogr2ogr"):
        county_miles._require_county_boundary_input(missing, 2022)


def test_county_boundary_manifest_has_checksum_size_time_and_attribution(tmp_path):
    path = _write_county_geojson(tmp_path)

    manifest = county_miles.ensure_county_boundary_manifest(
        path,
        2022,
        source_url="https://example.test/custom-counties.geojson",
        attribution="Fixture county provider",
        source_crs="OGC:CRS84",
    )

    assert manifest["source_url"] == "https://example.test/custom-counties.geojson"
    assert manifest["bytes"] == path.stat().st_size
    assert manifest["sha256"]
    assert manifest["retrieved_at_utc"]
    assert manifest["attribution"] == "Fixture county provider"
    assert manifest["crs"] == "OGC:CRS84"


def test_custom_county_boundary_requires_actual_source_provenance(tmp_path):
    path = _write_county_geojson(tmp_path)

    with pytest.raises(ValueError, match="source URL and attribution"):
        county_miles.ensure_county_boundary_manifest(
            path, 2022, source_crs="OGC:CRS84"
        )


def test_parquet_county_boundaries_require_lonlat_crs_metadata(tmp_path):
    path = tmp_path / "counties.parquet"
    frame = pd.DataFrame(
        {
            "county_fips": ["55001"],
            "geometry": [
                json.dumps(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[-90.0, 42.5], [-89.5, 42.5], [-89.5, 43.5], [-90.0, 43.5], [-90.0, 42.5]]
                        ],
                    }
                )
            ],
        }
    )
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="CRS metadata"):
        load_county_boundaries(path)

    frame.attrs["crs"] = "EPSG:4326"
    frame.to_parquet(path, index=False)
    out = load_county_boundaries(path)
    assert out.attrs["source_crs"] == "EPSG:4326"


@pytest.mark.parametrize(
    "mode,segment_type",
    [("zero", "same_tract_zero"), ("exclude", "same_tract_excluded")],
)
def test_calibrate_same_tract_distance_zero_and_exclude_modes(mode, segment_type):
    pairs = pd.DataFrame(
        {
            "home_tract": ["55001000100"],
            "work_tract": ["55001000100"],
            "home_county": ["55001"],
            "work_county": ["55001"],
            "workers": [10.0],
            "home_car_share": [0.6],
            "same_tract": [True],
            "home_lon": [-89.5],
            "home_lat": [43.0],
            "work_lon": [-89.49],
            "work_lat": [43.0],
        }
    )
    segments = pd.DataFrame(
        {
            "route_id": ["55001000100__55001000100"],
            "home_tract": ["55001000100"],
            "work_tract": ["55001000100"],
            "home_fips": ["55001"],
            "work_fips": ["55001"],
            "workers": [10.0],
            "home_car_share": [0.6],
            "same_tract": [True],
            "status": ["Ok"],
            "distance_m": [1609.344],
            "duration_s": [120.0],
            "geometry_path": [pd.NA],
            "source_manifest_id": ["src"],
            "network_manifest_id": ["net"],
            "home_lon": [-89.5],
            "home_lat": [43.0],
            "work_lon": [-89.49],
            "work_lat": [43.0],
            "straight_line_miles": [0.5],
            "outcome_fips": ["55001"],
            "route_miles_total": [1.0],
            "route_miles_in_county": [1.0],
            "unallocated_miles": [0.0],
            "segment_type": ["county"],
            "same_tract_imputed_miles": [0.0],
            "same_tract_mode": ["routed"],
        }
    )

    out = calibrate_same_tract_distance(pairs, segments, mode)

    assert set(out["segment_type"]) == {segment_type}
    assert out.iloc[0]["same_tract_mode"] == mode


def test_main_writes_diagnostics_and_removes_segment_output_on_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    pair_path = cache_dir / county_miles.PAIR_OUTPUT_NAME
    pair_frame = pd.DataFrame(
        {
            "route_id": ["55001000100__55003000100"],
            "home_tract": ["55001000100"],
            "work_tract": ["55003000100"],
            "home_county": ["55001"],
            "work_county": ["55003"],
            "workers": [100.0],
            "home_car_share": [0.8],
            "same_tract": [False],
            "home_lon": [-90.0],
            "home_lat": [43.0],
            "work_lon": [-89.0],
            "work_lat": [43.0],
            "commuter_car_weight": [80.0],
            "block_pair_straight_line_miles": [40.0],
            "commuter_car_miles": [3200.0],
            "routing_eligible": [True],
        }
    )
    pair_frame.to_parquet(pair_path, index=False)

    geometry_path = _write_geometry(tmp_path, "55001000100__55003000100", [[-90.0, 43.0], [-89.0, 43.0]])
    route_results = pair_frame.assign(
        status="Ok",
        distance_m=80000.0,
        duration_s=3600.0,
        geometry_path=str(geometry_path),
        source_manifest_id="src",
        network_manifest_id="net",
    )

    routing_options = {}

    def fake_route_pairs(*args, **kwargs):
        routing_options.update(kwargs)
        return route_results

    monkeypatch.setattr(county_miles, "route_pairs_with_checkpoints", fake_route_pairs)
    monkeypatch.setattr(county_miles, "_source_manifest_id", lambda *args: "src")
    monkeypatch.setattr(county_miles, "_network_manifest_id", lambda *args: "net")
    monkeypatch.setattr(
        county_miles,
        "validate_mileage_conservation",
        lambda segments: {
            "accepted": False,
            "n_failed_rows_row_threshold": 1,
            "aggregate_relative_gap": 0.1,
            "n_failed_rows": 0,
            "n_routes": 1,
            "n_failed_routes": 0,
            "total_unallocated_miles": 0.0,
            "selected_worker_weight": 100.0,
            "successful_worker_weight": 100.0,
            "selected_commuter_car_weight": 80.0,
            "successful_commuter_car_weight": 80.0,
            "successful_commuter_car_share": 1.0,
            "same_tract_imputed_miles": 0.0,
            "route_ratio_p50": 1.0,
            "route_ratio_p90": 1.0,
        },
    )

    with pytest.raises(RuntimeError, match="route mileage conservation failed"):
        main(
            [
                "--cache-dir",
                str(cache_dir),
                "--county-boundaries-path",
                str(_write_county_geojson(tmp_path)),
                "--county-boundaries-source-url",
                "https://example.test/counties.geojson",
                "--county-boundaries-attribution",
                "Fixture county provider",
                "--same-tract-mode",
                "primary_calibrated",
                "--route-workers",
                "3",
                "--checkpoint-every",
                "17",
            ]
        )

    mode_paths = county_miles._mode_output_paths(cache_dir, 2022, "primary_calibrated")
    assert mode_paths["diagnostics"].is_file()
    assert not mode_paths["segments"].exists()
    assert routing_options == {"max_workers": 3, "checkpoint_every": 17}
