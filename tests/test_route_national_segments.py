import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def fixture_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route_id": "route-a",
                "home_tract": "55001000100",
                "work_tract": "55003000100",
                "home_county": "55001",
                "work_county": "55003",
                "home_lon": -90.0,
                "home_lat": 43.0,
                "work_lon": -89.0,
                "work_lat": 43.0,
                "workers": 10.0,
                "home_car_share": 0.8,
                "commuter_car_weight": 8.0,
                "block_pair_straight_line_miles": 50.0,
                "commuter_car_miles": 400.0,
                "routing_eligible": True,
                "source_manifest_id": "flows-2022-wi",
                "network_manifest_id": "network-2022-national",
                "source_partition_id": "2022__2022__wi",
            },
            {
                "route_id": "route-c",
                "home_tract": "55001000200",
                "work_tract": "55003000200",
                "home_county": "55001",
                "work_county": "55003",
                "home_lon": -90.0,
                "home_lat": 43.1,
                "work_lon": -89.0,
                "work_lat": 43.1,
                "workers": 20.0,
                "home_car_share": 0.7,
                "commuter_car_weight": 14.0,
                "block_pair_straight_line_miles": 50.0,
                "commuter_car_miles": 700.0,
                "routing_eligible": True,
                "source_manifest_id": "flows-2022-wi",
                "network_manifest_id": "network-2022-national",
                "source_partition_id": "2022__2022__wi",
            },
        ]
    )


def fixture_boundaries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_fips": ["55001", "55003"],
            "geometry": [
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-90.0, 42.5], [-89.5, 42.5], [-89.5, 43.5], [-90.0, 43.5], [-90.0, 42.5]]
                    ],
                },
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-89.5, 42.5], [-89.0, 42.5], [-89.0, 43.5], [-89.5, 43.5], [-89.5, 42.5]]
                    ],
                },
            ],
        }
    )


def fixture_route_client():
    def route_client(home_lon, home_lat, work_lon, work_lat, route_id, base_url):
        return {
            "route_id": route_id,
            "status": "Ok",
            "distance_m": 80_000.0,
            "duration_s": 3_600.0,
            "geometry": {
                "type": "LineString",
                "coordinates": [[home_lon, home_lat], [work_lon, work_lat]],
            },
            "error_message": None,
        }

    return route_client


def fixture_route_client_that_fails_if_called():
    def route_client(*args, **kwargs):
        pytest.fail("completed route signatures must be resumed without calling OSRM")

    return route_client


def test_streaming_router_writes_segments_and_only_sampled_geometry(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    audits, segments = route_partition_to_segments(
        fixture_pairs(),
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        geometry_sample_rate=0.5,
        max_workers=1,
        route_client=fixture_route_client(),
    )

    assert len(audits) == 2
    allocated = segments.groupby("route_id")["route_miles_in_county"].sum().sum()
    unallocated = audits["unallocated_miles"].sum()
    assert allocated + unallocated == pytest.approx(audits["route_miles_total"].sum(), rel=1e-6)
    assert len(list((tmp_path / "qa_geometries").glob("*.geojson"))) == 1
    assert not (tmp_path / "geometries").exists()
    assert set(audits["schema_version"]) == {"route_national.segments.v1"}
    assert audits[["source_manifest_id", "network_manifest_id", "route_signature"]].notna().all().all()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.geojson")) == [
        next((tmp_path / "qa_geometries").glob("*.geojson")).relative_to(tmp_path)
    ]


def test_streaming_segments_retain_flow_vintage_provenance(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().assign(
        analysis_year=2022,
        lodes_source_year=2021,
        acs_car_share_vintage="2017-2021",
    )
    _, segments = route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        geometry_sample_rate=0,
        max_workers=1,
        route_client=fixture_route_client(),
    )

    assert segments["analysis_year"].eq(2022).all()
    assert segments["lodes_source_year"].eq(2021).all()
    assert segments["acs_car_share_vintage"].eq("2017-2021").all()
    assert segments["source_partition_id"].eq("2022__2022__wi").all()


def test_streaming_router_resumes_completed_route_signatures(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    first = route_partition_to_segments(
        fixture_pairs(),
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client(),
    )
    second = route_partition_to_segments(
        fixture_pairs(),
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client_that_fails_if_called(),
    )

    assert len(second[0]) == len(first[0])
    pd.testing.assert_frame_equal(second[0], first[0])
    pd.testing.assert_frame_equal(second[1], first[1])


def test_streaming_router_records_ineligible_and_failed_routes_as_audits(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].copy()
    pairs.loc[:, "routing_eligible"] = False
    pairs.loc[:, "omitted_coordinate_worker_weight"] = 10.0

    audits, segments = route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client_that_fails_if_called(),
    )

    assert audits.loc[0, ["status", "route_miles_total", "unallocated_miles"]].to_dict() == {
        "status": "Ineligible",
        "route_miles_total": 0.0,
        "unallocated_miles": 0.0,
    }
    assert audits.loc[0, [
        "workers",
        "home_car_share",
        "commuter_car_weight",
        "omitted_coordinate_worker_weight",
        "omitted_car_share_worker_weight",
    ]].to_dict() == {
        "workers": 10.0,
        "home_car_share": 0.8,
        "commuter_car_weight": 8.0,
        "omitted_coordinate_worker_weight": 10.0,
        "omitted_car_share_worker_weight": 0.0,
    }
    assert segments.empty


def test_streaming_router_never_routes_missing_coordinates_even_when_flagged_eligible(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].copy()
    pairs["home_lon"] = pd.Series([pd.NA], dtype="object")

    audits, _ = route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client_that_fails_if_called(),
    )

    assert audits.loc[0, "status"] == "Ineligible"
    assert "coordinates" in audits.loc[0, "error_message"]


def test_streaming_router_replaces_stale_qa_geometry_after_provenance_change(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].copy()
    route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        geometry_sample_rate=1.0,
        route_client=fixture_route_client(),
    )
    pairs.loc[:, "network_manifest_id"] = "network-2022-national-rebuilt"

    audits, _ = route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        geometry_sample_rate=1.0,
        route_client=fixture_route_client(),
    )

    qa_paths = list((tmp_path / "qa_geometries").glob("*.geojson"))
    assert len(qa_paths) == 1
    assert qa_paths[0].stem == audits.loc[0, "route_signature"]


def test_streaming_router_rejects_ok_result_without_geometry(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    def malformed_client(*args):
        return {
            "route_id": args[4],
            "status": "Ok",
            "distance_m": 80_000.0,
            "duration_s": 3_600.0,
            "geometry": None,
            "error_message": None,
        }

    audits, segments = route_partition_to_segments(
        fixture_pairs().iloc[[0]],
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=malformed_client,
    )

    assert audits.loc[0, "status"] == "MalformedResponse"
    assert "geometry" in audits.loc[0, "error_message"]
    assert audits.loc[0, "unallocated_miles"] == pytest.approx(audits.loc[0, "route_miles_total"])
    assert segments.empty


def test_streaming_router_reroutes_when_interrupted_checkpoint_lacks_segments(tmp_path):
    from build_route_national_segments import SEGMENT_FILE_NAME, route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]]
    route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client(),
    )
    (tmp_path / SEGMENT_FILE_NAME).unlink()
    calls = []

    def recording_client(*args):
        calls.append(args[4])
        return fixture_route_client()(*args)

    _, segments = route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=recording_client,
    )

    assert calls == ["route-a"]
    assert not segments.empty


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("source_manifest_id", pd.NA),
        ("network_manifest_id", ""),
        ("source_partition_id", "  "),
    ],
)
def test_streaming_router_requires_complete_provenance_before_routing(tmp_path, column, value):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].copy()
    pairs[column] = value

    with pytest.raises(ValueError, match="provenance"):
        route_partition_to_segments(
            pairs,
            fixture_boundaries(),
            base_url="http://osrm",
            cache_dir=tmp_path,
            route_client=fixture_route_client_that_fails_if_called(),
        )


def test_streaming_router_reroutes_when_source_partition_changes(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].copy()
    route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client(),
    )
    pairs.loc[:, "source_partition_id"] = "2022__2022__il"
    calls = []

    def recording_client(*args):
        calls.append(args[4])
        return fixture_route_client()(*args)

    route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=recording_client,
    )

    assert calls == ["route-a"]


def test_streaming_router_adapts_task3_partition_provenance_and_network_manifest(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].drop(
        columns=["source_manifest_id", "network_manifest_id"]
    ).copy()
    pairs["lodes_source_id"] = "lodes8:2022__2022__wi"
    pairs["provenance_sha256"] = "task3-provenance-sha256"

    audits, _ = route_partition_to_segments(
        pairs,
        fixture_boundaries(),
        base_url="http://osrm",
        cache_dir=tmp_path,
        route_client=fixture_route_client(),
        network_manifest_id="network-2022-national",
    )

    assert audits.loc[0, "source_manifest_id"] == (
        "lodes8:2022__2022__wi|task3-provenance-sha256"
    )
    assert audits.loc[0, "network_manifest_id"] == "network-2022-national"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("lodes_source_id", ""),
        ("provenance_sha256", pd.NA),
    ],
)
def test_streaming_router_rejects_incomplete_task3_partition_provenance(
    tmp_path, column, value
):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].drop(
        columns=["source_manifest_id", "network_manifest_id"]
    ).copy()
    pairs["lodes_source_id"] = "lodes8:2022__2022__wi"
    pairs["provenance_sha256"] = "task3-provenance-sha256"
    pairs[column] = value

    with pytest.raises(ValueError, match="provenance"):
        route_partition_to_segments(
            pairs,
            fixture_boundaries(),
            base_url="http://osrm",
            cache_dir=tmp_path,
            route_client=fixture_route_client_that_fails_if_called(),
            network_manifest_id="network-2022-national",
        )


@pytest.mark.parametrize("existing_network_id", ["", pd.NA])
def test_streaming_router_rejects_explicit_blank_network_provenance_even_with_argument(
    tmp_path, existing_network_id
):
    from build_route_national_segments import route_partition_to_segments

    pairs = fixture_pairs().iloc[[0]].copy()
    pairs["network_manifest_id"] = existing_network_id

    with pytest.raises(ValueError, match="network_manifest_id"):
        route_partition_to_segments(
            pairs,
            fixture_boundaries(),
            base_url="http://osrm",
            cache_dir=tmp_path,
            route_client=fixture_route_client_that_fails_if_called(),
            network_manifest_id="network-2022-national",
        )
