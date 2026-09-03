import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _routing_module():
    try:
        return importlib.import_module("build_route_pilot_network")
    except ModuleNotFoundError:
        pytest.fail("build_route_pilot_network module does not exist")


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def json(self):
        return self._payload


def fake_ok_osrm_response(self, url, *, params, timeout):
    assert url == "http://127.0.0.1:5000/route/v1/driving/-89.4,43.0;-89.3,43.1"
    assert params == {"overview": "full", "geometries": "geojson", "steps": "false"}
    assert timeout == (5, 30)
    return _FakeResponse(
        {
            "code": "Ok",
            "routes": [
                {
                    "distance": 12345.0,
                    "duration": 987.0,
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-89.4, 43.0], [-89.3, 43.1]],
                    },
                }
            ],
        }
    )


def fake_error_osrm_response(code):
    def _fake(self, url, *, params, timeout):
        return _FakeResponse({"code": code, "message": f"{code} happened"})

    return _fake


def test_route_pair_returns_distance_and_geometry(monkeypatch):
    routing = _routing_module()
    monkeypatch.setattr(requests.Session, "get", fake_ok_osrm_response)
    result = routing.route_pair(-89.4, 43.0, -89.3, 43.1, "r1", "http://127.0.0.1:5000", requests.Session())
    assert result["route_id"] == "r1"
    assert result["distance_m"] == 12345.0
    assert result["duration_s"] == 987.0
    assert result["status"] == "Ok"
    assert result["geometry"]["type"] == "LineString"


def test_route_pair_records_no_segment(monkeypatch):
    routing = _routing_module()
    monkeypatch.setattr(requests.Session, "get", fake_error_osrm_response("NoSegment"))
    result = routing.route_pair(0.0, 0.0, 1.0, 1.0, "r2", "http://127.0.0.1:5000", requests.Session())
    assert result["status"] == "NoSegment"
    assert result["error_message"] == "NoSegment happened"


def test_route_pair_retries_transport_then_succeeds(monkeypatch):
    routing = _routing_module()
    attempts = []

    def flaky(self, url, *, params, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise requests.ConnectionError("temporary")
        return _FakeResponse(
            {
                "code": "Ok",
                "routes": [
                    {
                        "distance": 10.0,
                        "duration": 5.0,
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0.0, 0.0], [1.0, 1.0]],
                        },
                    }
                ],
            }
        )

    sleeps = []
    monkeypatch.setattr(requests.Session, "get", flaky)
    monkeypatch.setattr(routing.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = routing.route_pair(0.0, 0.0, 1.0, 1.0, "r3", "http://127.0.0.1:5000", requests.Session())

    assert result["status"] == "Ok"
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]


def test_route_pair_rejects_lookalike_nonlocal_hosts():
    routing = _routing_module()
    with pytest.raises(ValueError, match="local OSRM endpoint"):
        routing.route_pair(0.0, 0.0, 1.0, 1.0, "r-bad", "http://127.0.0.1.example.com:5000", requests.Session())


def test_geofabrik_url_uses_historical_2022_suffix():
    routing = _routing_module()
    assert routing._geofabrik_url("wi", 2022) == (
        "https://download.geofabrik.de/north-america/us/wisconsin-220101.osm.pbf"
    )


def test_prepare_osrm_network_generates_expected_pipeline(tmp_path):
    routing = _routing_module()
    states = []
    for state in ("wi", "il", "ia", "mn", "mi"):
        path = tmp_path / f"{state}.osm.pbf"
        path.write_bytes(f"{state}-fixture".encode())
        states.append(path)

    calls = []

    def fake_runner(command, cwd):
        calls.append((command, cwd))
        return {"command": command, "cwd": str(cwd)}

    result = routing.prepare_osrm_network(states, tmp_path / "network", fake_runner)

    assert [command[:3] for command, _ in calls[:4]] == [
        ["docker", "run", "--rm"],
        ["docker", "run", "--rm"],
        ["docker", "run", "--rm"],
        ["docker", "run", "--rm"],
    ]
    assert calls[4][0][:4] == ["docker", "run", "--rm", "-d"]
    mount_arg = calls[0][0][4]
    assert mount_arg.startswith(str((tmp_path / "network").resolve()))
    assert mount_arg.endswith(":/data")
    assert calls[0][0][5] == routing.OSM_MERGE_IMAGE
    assert calls[0][0][6:10] == ["osmium", "merge", "--overwrite", "/data/inputs/ia.osm.pbf"]
    assert calls[0][0][-2:] == ["-o", "/data/pilot-network-2022.osm.pbf"]
    assert calls[1][0][5] == routing.OSRM_IMAGE
    assert calls[1][0][6:9] == ["osrm-extract", "-p", "/opt/car.lua"]
    assert calls[1][0][-1] == "/data/pilot-network-2022.osm.pbf"
    assert calls[2][0][6:] == ["osrm-partition", "/data/pilot-network-2022.osrm"]
    assert calls[3][0][6:] == ["osrm-customize", "/data/pilot-network-2022.osrm"]
    assert calls[4][0][2:6] == ["--rm", "-d", "-p", "5000:5000"]
    assert calls[4][0][8] == routing.OSRM_IMAGE
    assert calls[4][0][9:] == ["osrm-routed", "--algorithm", "mld", "/data/pilot-network-2022.osrm"]
    assert result["manifest"]["source_count"] == 5
    assert result["manifest"]["osrm_profile"] == "car"
    assert result["manifest"]["osrm_algorithm"] == "mld"
    assert ":latest" not in result["manifest"]["merge_image"]
    assert result["manifest"]["attribution"].startswith("© OpenStreetMap contributors")
    assert result["manifest"]["license"] == "ODbL-1.0"
    assert all({"url", "bytes", "retrieved_at_utc", "sha256"}.issubset(source) for source in result["manifest"]["sources"])
    assert Path(result["manifest_path"]).is_file()


def test_prepare_osrm_network_raises_actionable_docker_error(tmp_path):
    routing = _routing_module()
    pbf = tmp_path / "wi.osm.pbf"
    pbf.write_bytes(b"fixture")

    def missing_docker(command, cwd):
        raise FileNotFoundError("docker")

    with pytest.raises(RuntimeError, match="Docker is required to prepare the local OSRM network"):
        routing.prepare_osrm_network([pbf], tmp_path / "network", missing_docker)


def test_docker_runner_wraps_called_process_error(monkeypatch, tmp_path):
    routing = _routing_module()

    error = requests.HTTPError  # keep import usage stable for lint? no-op
    del error

    def fail_run(*args, **kwargs):
        raise routing.subprocess.CalledProcessError(
            125,
            ["docker", "run", "--rm", "image"],
            output="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        )

    monkeypatch.setattr(routing.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="Cannot connect to the Docker daemon"):
        routing._docker_runner(["docker", "run", "--rm", "image"], tmp_path)


def test_route_pairs_with_checkpoints_is_deterministic_and_resumable(monkeypatch, tmp_path):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [
            {
                "route_id": "r2",
                "home_tract": "55001000100",
                "work_tract": "55001000200",
                "home_lon": -89.3,
                "home_lat": 43.1,
                "work_lon": -89.2,
                "work_lat": 43.2,
                "source_manifest_id": "flows-2022",
                "network_manifest_id": "network-2022",
            },
            {
                "route_id": "r1",
                "home_tract": "55001000300",
                "work_tract": "55001000400",
                "home_lon": -89.4,
                "home_lat": 43.0,
                "work_lon": -89.1,
                "work_lat": 43.3,
                "source_manifest_id": "flows-2022",
                "network_manifest_id": "network-2022",
            },
        ]
    )

    cache_path = tmp_path / "routes.parquet"
    routing._write_checkpoint_rows(
        pd.DataFrame(
            [
                {
                    "route_id": "r1",
                    "home_lon": -89.4,
                    "home_lat": 43.0,
                    "work_lon": -89.1,
                    "work_lat": 43.3,
                    "status": "Ok",
                    "distance_m": 100.0,
                    "duration_s": 10.0,
                    "geometry_path": str(tmp_path / "geometries" / "r1.geojson"),
                    "error_message": None,
                    "source_manifest_id": "flows-2022",
                    "network_manifest_id": "network-2022",
                    "routed_at_utc": "2026-09-02T00:00:00+00:00",
                }
            ]
        ),
        cache_path,
    )
    (tmp_path / "geometries").mkdir(exist_ok=True)
    (tmp_path / "geometries" / "r1.geojson").write_text(json.dumps({"type": "LineString", "coordinates": [[0, 0], [1, 1]]}))

    calls = []

    def fake_route_pair(home_lon, home_lat, work_lon, work_lat, route_id, base_url, session):
        calls.append(route_id)
        return {
            "route_id": route_id,
            "home_lon": home_lon,
            "home_lat": home_lat,
            "work_lon": work_lon,
            "work_lat": work_lat,
            "status": "NoRoute",
            "distance_m": None,
            "duration_s": None,
            "geometry": None,
            "error_message": "topology gap",
        }

    monkeypatch.setattr(routing, "route_pair", fake_route_pair)

    result = routing.route_pairs_with_checkpoints(
        pairs,
        cache_path,
        "http://127.0.0.1:5000",
        requests.Session(),
    )

    assert calls == ["r2"]
    assert result["route_id"].tolist() == ["r1", "r2"]
    assert result["status"].tolist() == ["Ok", "NoRoute"]
    assert result["geometry_path"].notna().all()
    assert Path(result.loc[result["route_id"] == "r2", "geometry_path"].iloc[0]).is_file()
    stored = pd.read_parquet(cache_path).sort_values("route_id").reset_index(drop=True)
    assert stored["route_id"].tolist() == ["r1", "r2"]
    geometry_doc = json.loads(Path(stored.loc[1, "geometry_path"]).read_text())
    assert geometry_doc["route_id"] == "r2"
    assert geometry_doc["status"] == "NoRoute"


def test_route_pairs_with_checkpoints_reroutes_stale_signature(monkeypatch, tmp_path):
    routing = _routing_module()
    cache_path = tmp_path / "routes.parquet"
    geometry_dir = tmp_path / "geometries"
    geometry_dir.mkdir(exist_ok=True)
    stale_geometry = geometry_dir / "r1.geojson"
    stale_geometry.write_text(json.dumps({"route_id": "r1", "status": "Ok"}))

    routing._write_checkpoint_rows(
        pd.DataFrame(
            [
                {
                    "route_id": "r1",
                    "home_lon": -89.4,
                    "home_lat": 43.0,
                    "work_lon": -89.1,
                    "work_lat": 43.3,
                    "status": "Ok",
                    "distance_m": 100.0,
                    "duration_s": 10.0,
                    "geometry_path": str(stale_geometry),
                    "error_message": None,
                    "source_manifest_id": "flows-old",
                    "network_manifest_id": "network-2022",
                    "routed_at_utc": "2026-09-02T00:00:00+00:00",
                }
            ]
        ),
        cache_path,
    )

    pairs = pd.DataFrame(
        [
            {
                "route_id": "r1",
                "home_tract": "55001000300",
                "work_tract": "55001000400",
                "home_lon": -89.4,
                "home_lat": 43.0,
                "work_lon": -89.1,
                "work_lat": 43.3,
                "source_manifest_id": "flows-new",
                "network_manifest_id": "network-2022",
            }
        ]
    )

    calls = []

    def fake_route_pair(home_lon, home_lat, work_lon, work_lat, route_id, base_url, session):
        calls.append((route_id, home_lon, work_lon))
        return {
            "route_id": route_id,
            "status": "Ok",
            "distance_m": 200.0,
            "duration_s": 20.0,
            "geometry": {"type": "LineString", "coordinates": [[home_lon, home_lat], [work_lon, work_lat]]},
            "error_message": None,
        }

    monkeypatch.setattr(routing, "route_pair", fake_route_pair)

    result = routing.route_pairs_with_checkpoints(
        pairs,
        cache_path,
        "http://127.0.0.1:5000",
        requests.Session(),
    )

    assert calls == [("r1", -89.4, -89.1)]
    assert result["route_id"].tolist() == ["r1"]
    assert result.loc[0, "source_manifest_id"] == "flows-new"
    assert result.loc[0, "distance_m"] == 200.0


def test_route_pairs_with_checkpoints_derives_route_id_when_absent(monkeypatch, tmp_path):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [
            {
                "home_tract": "55001000300",
                "work_tract": "55001000400",
                "home_lon": -89.4,
                "home_lat": 43.0,
                "work_lon": -89.1,
                "work_lat": 43.3,
                "source_manifest_id": "flows-2022",
                "network_manifest_id": "network-2022",
            }
        ]
    )

    seen_ids = []

    def fake_route_pair(home_lon, home_lat, work_lon, work_lat, route_id, base_url, session):
        seen_ids.append(route_id)
        return {
            "route_id": route_id,
            "status": "Ok",
            "distance_m": 10.0,
            "duration_s": 5.0,
            "geometry": {"type": "LineString", "coordinates": [[home_lon, home_lat], [work_lon, work_lat]]},
            "error_message": None,
        }

    monkeypatch.setattr(routing, "route_pair", fake_route_pair)

    result = routing.route_pairs_with_checkpoints(
        pairs,
        tmp_path / "routes.parquet",
        "http://127.0.0.1:5000",
        requests.Session(),
    )

    assert seen_ids == ["55001000300__55001000400"]
    assert result.loc[0, "route_id"] == "55001000300__55001000400"


def test_route_pairs_with_checkpoints_can_delegate_geometry_persistence(monkeypatch, tmp_path):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [{
            "route_id": "callback-route",
            "home_lon": -89.4,
            "home_lat": 43.0,
            "work_lon": -89.3,
            "work_lat": 43.1,
            "source_manifest_id": "flows-2022",
            "network_manifest_id": "network-2022",
        }]
    )
    observed = []
    monkeypatch.setattr(
        routing,
        "route_pair",
        lambda *args, **kwargs: {
            "route_id": "callback-route",
            "status": "Ok",
            "distance_m": 10.0,
            "duration_s": 5.0,
            "geometry": {"type": "LineString", "coordinates": [[-89.4, 43.0], [-89.3, 43.1]]},
            "error_message": None,
        },
    )

    def capture_geometry(row, result):
        observed.append((row["route_id"], result["geometry"]))
        return None

    out = routing.route_pairs_with_checkpoints(
        pairs,
        tmp_path / "routes.parquet",
        "http://127.0.0.1:5000",
        requests.Session(),
        geometry_callback=capture_geometry,
    )

    assert observed == [("callback-route", {"type": "LineString", "coordinates": [[-89.4, 43.0], [-89.3, 43.1]]})]
    assert pd.isna(out.loc[0, "geometry_path"])
    assert not (tmp_path / "geometries").exists()


def test_route_pairs_with_checkpoints_parallelizes_and_batches_checkpoints(monkeypatch, tmp_path):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [
            {
                "route_id": f"r{i}",
                "home_tract": f"55001000{i}00",
                "work_tract": f"55001000{i}01",
                "home_lon": -89.4 + i / 100,
                "home_lat": 43.0,
                "work_lon": -89.3 + i / 100,
                "work_lat": 43.1,
                "source_manifest_id": "flows-2022",
                "network_manifest_id": "network-2022",
            }
            for i in range(1, 5)
        ]
    )
    active = 0
    max_active = 0
    call_count = 0
    lock = threading.Lock()
    first_pair_barrier = threading.Barrier(2)

    def fake_route_pair(home_lon, home_lat, work_lon, work_lat, route_id, base_url, session):
        nonlocal active, max_active, call_count
        with lock:
            call_count += 1
            ordinal = call_count
            active += 1
            max_active = max(max_active, active)
        if ordinal <= 2:
            first_pair_barrier.wait(timeout=2)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {
            "route_id": route_id,
            "status": "Ok",
            "distance_m": 100.0,
            "duration_s": 10.0,
            "geometry": {"type": "LineString", "coordinates": [[home_lon, home_lat], [work_lon, work_lat]]},
            "error_message": None,
        }

    monkeypatch.setattr(routing, "route_pair", fake_route_pair)
    checkpoint_writes = []
    original_write = routing._write_checkpoint_rows

    def recording_write(frame, path):
        checkpoint_writes.append(len(frame))
        return original_write(frame, path)

    monkeypatch.setattr(routing, "_write_checkpoint_rows", recording_write)
    cache_path = tmp_path / "routes.parquet"

    result = routing.route_pairs_with_checkpoints(
        pairs,
        cache_path,
        "http://127.0.0.1:5000",
        requests.Session(),
        max_workers=2,
        checkpoint_every=2,
    )

    assert max_active >= 2
    assert checkpoint_writes == [2, 4]
    assert result["route_id"].tolist() == ["r1", "r2", "r3", "r4"]

    routing.route_pairs_with_checkpoints(
        pairs,
        cache_path,
        "http://127.0.0.1:5000",
        requests.Session(),
        max_workers=2,
        checkpoint_every=2,
    )
    assert call_count == 4


@pytest.mark.parametrize(("max_workers", "checkpoint_every"), [(0, 1), (1, 0)])
def test_route_pairs_with_checkpoints_rejects_nonpositive_parallel_options(
    tmp_path, max_workers, checkpoint_every
):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [
            {
                "route_id": "r1",
                "home_lon": -89.4,
                "home_lat": 43.0,
                "work_lon": -89.3,
                "work_lat": 43.1,
            }
        ]
    )
    with pytest.raises(ValueError, match="positive"):
        routing.route_pairs_with_checkpoints(
            pairs,
            tmp_path / "routes.parquet",
            "http://127.0.0.1:5000",
            requests.Session(),
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
        )


def test_route_checkpoint_preserves_pair_weights_and_endpoint_metadata(monkeypatch, tmp_path):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [{
            "route_id": "55001000300__55001000400",
            "home_tract": "55001000300",
            "work_tract": "55001000400",
            "home_county": "55001",
            "work_county": "55001",
            "home_lon": -89.4,
            "home_lat": 43.0,
            "work_lon": -89.1,
            "work_lat": 43.3,
            "workers": 25.0,
            "home_car_share": 0.8,
            "commuter_car_weight": 20.0,
            "block_pair_straight_line_miles": 12.5,
            "commuter_car_miles": 250.0,
            "same_tract": False,
            "source_manifest_id": "flows-2022",
            "network_manifest_id": "network-2022",
        }]
    )

    monkeypatch.setattr(
        routing,
        "route_pair",
        lambda *args, **kwargs: {
            "route_id": "55001000300__55001000400",
            "status": "NoRoute",
            "distance_m": None,
            "duration_s": None,
            "geometry": None,
            "error_message": "topology gap",
        },
    )

    out = routing.route_pairs_with_checkpoints(
        pairs, tmp_path / "routes.parquet", "http://127.0.0.1:5000", requests.Session()
    )

    expected = {
        "route_id": "55001000300__55001000400",
        "home_tract": "55001000300",
        "work_tract": "55001000400",
        "home_county": "55001",
        "work_county": "55001",
        "workers": 25.0,
        "home_car_share": 0.8,
        "commuter_car_weight": 20.0,
        "block_pair_straight_line_miles": 12.5,
        "commuter_car_miles": 250.0,
        "status": "NoRoute",
    }
    assert out.loc[0, list(expected)].to_dict() == expected


def test_route_checkpoint_records_ineligible_pair_without_calling_osrm(monkeypatch, tmp_path):
    routing = _routing_module()
    pairs = pd.DataFrame(
        [{
            "route_id": "missing-coordinates",
            "home_tract": "55001000300",
            "work_tract": "55001000400",
            "home_county": "55001",
            "work_county": "55001",
            "home_lon": pd.NA,
            "home_lat": pd.NA,
            "work_lon": -89.1,
            "work_lat": 43.3,
            "workers": 25.0,
            "home_car_share": 0.8,
            "commuter_car_weight": 20.0,
            "block_pair_straight_line_miles": pd.NA,
            "commuter_car_miles": pd.NA,
            "same_tract": False,
            "routing_eligible": False,
            "omitted_coordinate_worker_weight": 25.0,
            "omitted_car_share_worker_weight": 0.0,
            "source_manifest_id": "flows-2022",
            "network_manifest_id": "network-2022",
        }]
    )
    monkeypatch.setattr(
        routing,
        "route_pair",
        lambda *args, **kwargs: pytest.fail("OSRM must not be called for ineligible pairs"),
    )

    out = routing.route_pairs_with_checkpoints(
        pairs, tmp_path / "routes.parquet", "http://127.0.0.1:5000", requests.Session()
    )

    assert out.loc[0, "status"] == "MissingCoordinates"
    assert out.loc[0, "workers"] == 25
    assert out.loc[0, "omitted_coordinate_worker_weight"] == 25
