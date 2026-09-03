import sys
import hashlib
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_national_state_universe_has_50_states_plus_dc():
    from build_route_national_network import NATIONAL_STATES

    assert len(NATIONAL_STATES) == 51
    assert "dc" in NATIONAL_STATES
    assert "pr" not in NATIONAL_STATES


def test_network_manifest_records_all_sources_and_network_year(tmp_path):
    from build_route_national_network import build_network_manifest

    manifest = build_network_manifest([{"state": "wi", "sha256": "a"}], network_year=2022, network_id="n1")
    assert manifest["network_year"] == 2022
    assert manifest["states"] == ["wi"]


def test_manifest_sources_are_sorted_and_metadata_is_explicit():
    from build_route_national_network import build_network_manifest

    manifest = build_network_manifest(
        [{"state": "wi", "sha256": "w"}, {"state": "al", "sha256": "a"}],
        network_year=2022,
        network_id="national-2022",
    )
    assert manifest["states"] == ["al", "wi"]
    assert [row["state"] for row in manifest["sources"]] == ["al", "wi"]
    assert manifest["manifest_schema_version"]
    assert manifest["osrm_profile"] == "car"
    assert manifest["osrm_algorithm"] == "mld"
    assert manifest["osrm_image"].endswith(":5.27.1")


def test_manifest_rejects_unknown_state():
    from build_route_national_network import build_network_manifest

    with pytest.raises(ValueError, match="unsupported national state"):
        build_network_manifest([{"state": "pr", "sha256": "x"}], network_year=2022, network_id="n1")


def test_prepare_rejects_partial_scope_without_explicit_opt_in(tmp_path):
    from build_route_national_network import prepare_national_osrm_network

    with pytest.raises(ValueError, match="all 51 states"):
        prepare_national_osrm_network(["wi"], 2022, tmp_path, lambda *_: None)


def test_cached_extract_requires_matching_sidecar_provenance(tmp_path):
    import build_route_national_network as national

    path = tmp_path / "wi-2022.osm.pbf"
    path.write_bytes(b"fixture")
    sidecar = path.with_name(path.name + ".manifest.json")
    sidecar.write_text(json.dumps({"url": "wrong", "bytes": 7, "sha256": "wrong", "retrieved_at_utc": "2022-01-01T00:00:00+00:00"}))
    with pytest.raises(RuntimeError, match="cache failed manifest/checksum validation"):
        national.download_geofabrik_extract("wi", 2022, tmp_path, object())


def test_manifest_id_depends_on_source_checksum():
    import build_route_national_network as national

    first = national.build_network_manifest([{"state": "wi", "sha256": "a"}], network_year=2022)
    second = national.build_network_manifest([{"state": "wi", "sha256": "b"}], network_year=2022)
    assert first["manifest_id"] != second["manifest_id"]


def test_docker_runtime_errors_are_actionable(tmp_path, monkeypatch):
    import build_route_national_network as national

    monkeypatch.setattr(national, "NATIONAL_STATES", ("wi",))
    pbf = tmp_path / "osm" / "2022" / "wi-2022.osm.pbf"
    pbf.parent.mkdir(parents=True)
    pbf.write_bytes(b"fixture")
    pbf.with_name(pbf.name + ".manifest.json").write_text(json.dumps({"url": national._geofabrik_url("wi", 2022), "bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest(), "retrieved_at_utc": "2022-01-01T00:00:00+00:00"}))
    with pytest.raises(RuntimeError, match="Docker failed.*osmium"):
        national.prepare_national_osrm_network(("wi",), 2022, tmp_path, lambda command, cwd: (_ for _ in ()).throw(RuntimeError("stderr details")))


def test_prepare_resumes_completed_stages(tmp_path, monkeypatch):
    import build_route_national_network as national

    monkeypatch.setattr(national, "NATIONAL_STATES", ("wi",))
    calls = []

    def fake_download(state, year, cache_dir, session):
        path = Path(cache_dir) / f"{state}-{year}.osm.pbf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        national._atomic_write_json({"url": national._geofabrik_url(state, year), "bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest(), "retrieved_at_utc": "2022-01-01T00:00:00+00:00"}, path.with_name(path.name + ".manifest.json"))
        return path

    monkeypatch.setattr(national, "download_geofabrik_extract", fake_download)
    def interrupt(command, cwd):
        calls.append(command)
        (Path(cwd) / f"stage-{len(calls)}.out").write_bytes(str(len(calls)).encode())
        if len(calls) == 2:
            raise RuntimeError("interrupt")

    with pytest.raises(RuntimeError):
        national.prepare_national_osrm_network(("wi",), 2022, tmp_path, interrupt)
    def resume(command, cwd):
        calls.append(command)
        (Path(cwd) / f"stage-{len(calls)}.out").write_bytes(str(len(calls)).encode())
    national.prepare_national_osrm_network(("wi",), 2022, tmp_path, resume, allow_partial=True)
    assert len(calls) == 6


def test_resume_rebuilds_when_recorded_stage_output_is_missing(tmp_path, monkeypatch):
    import build_route_national_network as national
    monkeypatch.setattr(national, "NATIONAL_STATES", ("wi",))
    def fake_download(state, year, cache_dir, session):
        path = Path(cache_dir) / f"{state}-{year}.osm.pbf"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
        national._atomic_write_json({"url": national._geofabrik_url(state, year), "bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest(), "retrieved_at_utc": "2022-01-01T00:00:00+00:00"}, path.with_name(path.name + ".manifest.json")); return path
    monkeypatch.setattr(national, "download_geofabrik_extract", fake_download)
    calls = []
    def runner(command, cwd):
        calls.append(command); (Path(cwd) / f"stage-{len(calls)}.out").write_bytes(f"{len(calls)}".encode())
    national.prepare_national_osrm_network(("wi",), 2022, tmp_path, runner)
    first_count = len(calls)
    (tmp_path / "network" / "2022" / "stage-1.out").unlink()
    national.prepare_national_osrm_network(("wi",), 2022, tmp_path, runner)
    assert len(calls) == first_count + 5


def test_malformed_stage_manifest_is_ignored(tmp_path, monkeypatch):
    import build_route_national_network as national
    monkeypatch.setattr(national, "NATIONAL_STATES", ("wi",))
    def fake_download(state, year, cache_dir, session):
        path = Path(cache_dir) / f"{state}-{year}.osm.pbf"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
        national._atomic_write_json({"url": national._geofabrik_url(state, year), "bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest(), "retrieved_at_utc": "2022-01-01T00:00:00+00:00"}, path.with_name(path.name + ".manifest.json")); return path
    monkeypatch.setattr(national, "download_geofabrik_extract", fake_download)
    network = tmp_path / "network" / "2022"; network.mkdir(parents=True); (network / ".stage_manifest.json").write_text(json.dumps({"completed": []}))
    calls = []
    def runner(command, cwd):
        calls.append(command); (Path(cwd) / f"stage-{len(calls)}.out").write_bytes(str(len(calls)).encode())
    national.prepare_national_osrm_network(("wi",), 2022, tmp_path, runner)
    assert len(calls) == 5


@pytest.mark.parametrize("root", [[], None])
def test_non_mapping_stage_manifest_root_is_ignored(tmp_path, monkeypatch, root):
    import build_route_national_network as national
    monkeypatch.setattr(national, "NATIONAL_STATES", ("wi",))
    def fake_download(state, year, cache_dir, session):
        path = Path(cache_dir) / f"{state}-{year}.osm.pbf"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
        national._atomic_write_json({"url": national._geofabrik_url(state, year), "bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest(), "retrieved_at_utc": "2022-01-01T00:00:00+00:00"}, path.with_name(path.name + ".manifest.json")); return path
    monkeypatch.setattr(national, "download_geofabrik_extract", fake_download)
    network = tmp_path / "network" / "2022"; network.mkdir(parents=True); (network / ".stage_manifest.json").write_text(json.dumps(root))
    calls = []
    def runner(command, cwd):
        calls.append(command); (Path(cwd) / f"stage-{len(calls)}.out").write_bytes(str(len(calls)).encode())
    national.prepare_national_osrm_network(("wi",), 2022, tmp_path, runner)
    assert len(calls) == 5


def test_outputs_valid_rejects_absolute_and_traversal_paths(tmp_path):
    import build_route_national_network as national
    outside = tmp_path / "outside"; outside.write_bytes(b"secret")
    digest = hashlib.sha256(b"secret").hexdigest()
    assert not national._outputs_valid(tmp_path, {"/outside": digest})
    assert not national._outputs_valid(tmp_path, {"../outside": digest})


def test_outputs_valid_rejects_symlink_escape(tmp_path):
    import build_route_national_network as national
    outside = tmp_path / "outside"; outside.write_bytes(b"secret")
    network = tmp_path / "network"; network.mkdir()
    (network / "escaped.out").symlink_to(outside)
    digest = hashlib.sha256(b"secret").hexdigest()
    assert not national._outputs_valid(network, {"escaped.out": digest})
