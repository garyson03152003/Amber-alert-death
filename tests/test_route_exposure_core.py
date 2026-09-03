import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from route_exposure_core import weighted_tract_pairs
from route_exposure_core import (
    allocate_route_miles,
    build_county_exposure,
    classify_route_origin,
    parse_osrm_route,
)


def test_weighted_tract_pair_preserves_workers_and_weighted_endpoints():
    flows = pd.DataFrame({
        "h_geocode": ["550010001001001", "550010001001002"],
        "w_geocode": ["550010001002001", "550010001002002"],
        "S000": [3, 1],
    })
    crosswalk = pd.DataFrame({
        "tabblk2020": [
            "550010001001001",
            "550010001001002",
            "550010001002001",
            "550010001002002",
        ],
        "cty": ["55001", "55001", "55001", "55001"],
        "trct": ["55001000100", "55001000100", "55001000200", "55001000200"],
        "blklatdd": [43.00, 43.04, 43.10, 43.14],
        "blklondd": [-89.40, -89.36, -89.30, -89.26],
    })
    car = pd.Series({"55001000100": 0.8, "55001000200": 0.9})

    out = weighted_tract_pairs(flows, crosswalk, car)

    assert len(out) == 1
    assert out.loc[0, "workers"] == 4
    assert out.loc[0, "home_tract"] == "55001000100"
    assert out.loc[0, "work_tract"] == "55001000200"
    assert out.loc[0, "home_car_share"] == 0.8
    assert out.loc[0, "home_lat"] == pytest.approx(43.01)


def test_weighted_tract_pair_preserves_missing_endpoint_workers():
    flows = pd.DataFrame({
        "h_geocode": ["550010001001001", "550010001001002"],
        "w_geocode": ["550010001002001", "550010001002002"],
        "S000": [3, 1],
    })
    crosswalk = pd.DataFrame({
        "tabblk2020": [
            "550010001001001",
            "550010001001002",
            "550010001002001",
            "550010001002002",
        ],
        "cty": ["55001", "55001", "55001", "55001"],
        "trct": ["55001000100", "55001000100", "55001000200", "55001000200"],
        "blklatdd": [43.00, float("nan"), 43.10, 43.14],
        "blklondd": [-89.40, float("nan"), -89.30, -89.26],
    })
    car = pd.Series({"55001000100": 0.8, "55001000200": 0.9})

    out = weighted_tract_pairs(flows, crosswalk, car)

    assert len(out) == 1
    assert out.loc[0, "workers"] == 4
    assert out.loc[0, "valid_endpoint_workers"] == 3
    assert out.loc[0, "missing_endpoint_workers"] == 1
    assert out.loc[0, "home_lat"] == pytest.approx(43.0)


def test_weighted_tract_pairs_vectorizes_many_pair_aggregation():
    pair_count = 5_000
    flow_rows = []
    crosswalk_rows = []
    shares = {}
    for pair_index in range(pair_count):
        home_tract = f"55{pair_index:09d}"
        work_tract = f"55{pair_index + pair_count:09d}"
        home_blocks = [f"{home_tract}0001", f"{home_tract}0002"]
        work_blocks = [f"{work_tract}0001", f"{work_tract}0002"]
        flow_rows.extend(
            [
                {"h_geocode": home_blocks[0], "w_geocode": work_blocks[0], "S000": 3},
                {"h_geocode": home_blocks[1], "w_geocode": work_blocks[1], "S000": 1},
            ]
        )
        second_home_lat = float("nan") if pair_index % 2 else 43.04
        second_home_lon = float("nan") if pair_index % 2 else -89.36
        crosswalk_rows.extend(
            [
                {"tabblk2020": home_blocks[0], "cty": "55001", "trct": home_tract, "blklatdd": 43.0, "blklondd": -89.4},
                {"tabblk2020": home_blocks[1], "cty": "55001", "trct": home_tract, "blklatdd": second_home_lat, "blklondd": second_home_lon},
                {"tabblk2020": work_blocks[0], "cty": "55003", "trct": work_tract, "blklatdd": 43.1, "blklondd": -89.3},
                {"tabblk2020": work_blocks[1], "cty": "55003", "trct": work_tract, "blklatdd": 43.14, "blklondd": -89.26},
            ]
        )
        shares[home_tract] = 0.8

    started = time.perf_counter()
    out = weighted_tract_pairs(
        pd.DataFrame(flow_rows),
        pd.DataFrame(crosswalk_rows),
        pd.Series(shares),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert len(out) == pair_count
    assert list(out.columns) == [
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
        "block_pair_straight_line_miles",
        "home_car_share",
        "missing_home_car_share_workers",
    ]
    complete = out.loc[out["home_tract"].eq("55000000000")].iloc[0]
    incomplete = out.loc[out["home_tract"].eq("55000000001")].iloc[0]
    assert complete["workers"] == 4
    assert complete["valid_endpoint_workers"] == 4
    assert complete["missing_endpoint_workers"] == 0
    assert complete["home_lat"] == pytest.approx((43.0 * 3 + 43.04) / 4)
    assert incomplete["workers"] == 4
    assert incomplete["valid_endpoint_workers"] == 3
    assert incomplete["missing_endpoint_workers"] == 1
    assert incomplete["home_lat"] == pytest.approx(43.0)
    assert out["block_pair_straight_line_miles"].notna().all()
    assert out["home_car_share"].eq(0.8).all()


def test_parse_osrm_route_rejects_non_ok_response():
    with pytest.raises(ValueError, match="NoRoute"):
        parse_osrm_route({"code": "NoRoute", "message": "no path"}, "r1")


def test_parse_osrm_route_returns_geometry_and_metrics():
    payload = {
        "code": "Ok",
        "routes": [{
            "distance": 1234.5,
            "duration": 321.0,
            "geometry": {
                "type": "LineString",
                "coordinates": [[-90.0, 43.0], [-89.0, 43.0]],
            },
        }],
    }

    out = parse_osrm_route(payload, "r1")

    assert out["route_id"] == "r1"
    assert out["distance_m"] == pytest.approx(1234.5)
    assert out["duration_s"] == pytest.approx(321.0)
    assert out["geometry"]["type"] == "LineString"


def test_classify_route_origin():
    assert classify_route_origin("55001", "55001", "55001") == "own_origin"
    assert classify_route_origin("55003", "55001", "55001") == "cross_origin"
    assert classify_route_origin("55003", "55005", "55001") == "pass_through"


def _synthetic_county_boundaries_split_at_longitude(split_lon: float) -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": ["55001", "55003"],
        "geometry": [
            {
                "type": "Polygon",
                "coordinates": [[
                    [-90.0, 42.5],
                    [split_lon, 42.5],
                    [split_lon, 43.5],
                    [-90.0, 43.5],
                    [-90.0, 42.5],
                ]],
            },
            {
                "type": "Polygon",
                "coordinates": [[
                    [split_lon, 42.5],
                    [-89.0, 42.5],
                    [-89.0, 43.5],
                    [split_lon, 43.5],
                    [split_lon, 42.5],
                ]],
            },
        ],
    })


def test_allocate_route_miles_conserves_route_length():
    route = {
        "type": "LineString",
        "coordinates": [[-90.0, 43.0], [-89.0, 43.0]],
        "properties": {"distance_m": 80000.0},
    }
    counties = _synthetic_county_boundaries_split_at_longitude(-89.5)

    out = allocate_route_miles(route, counties, "r1")

    assert out["route_miles_in_county"].sum() == pytest.approx(49.7097, rel=0.005)
    assert out["route_miles_in_county"].ge(0).all()
    assert out["unallocated_miles"].sum() == pytest.approx(0.0, abs=1e-6)


def test_allocate_route_miles_splits_exactly_at_county_boundary():
    route = {
        "type": "LineString",
        "coordinates": [[-90.0, 43.0], [-89.0, 43.0]],
        "properties": {"distance_m": 80000.0},
    }
    counties = _synthetic_county_boundaries_split_at_longitude(-89.5)

    out = allocate_route_miles(route, counties, "r1").sort_values("county_fips").reset_index(drop=True)

    assert list(out["county_fips"]) == ["55001", "55003"]
    assert out.loc[0, "route_miles_in_county"] == pytest.approx(out.loc[1, "route_miles_in_county"], rel=1e-6)
    assert out["unallocated_miles"].sum() == pytest.approx(0.0, abs=1e-6)


def test_build_county_exposure_uses_one_denominator_for_own_and_cross():
    segments = pd.DataFrame({
        "outcome_fips": ["55001", "55001"],
        "home_fips": ["55001", "55003"],
        "work_fips": ["55001", "55001"],
        "workers": [100, 50],
        "home_car_share": [0.8, 0.8],
        "route_miles_in_county": [10.0, 20.0],
    })
    alerts = pd.DataFrame({
        "home_fips": ["55001"],
        "alert_date": [pd.Timestamp("2022-01-02")],
    })

    out = build_county_exposure(segments, alerts)
    row = out.iloc[0]

    assert row["total_commuter_car_miles"] == pytest.approx(1600.0)
    assert row["own_affected_car_miles"] == pytest.approx(800.0)
    assert row["cross_affected_car_miles"] == pytest.approx(0.0)
    assert row["affected_route_share"] == pytest.approx(0.5)
