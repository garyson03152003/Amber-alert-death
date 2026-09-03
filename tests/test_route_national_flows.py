import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _national_module():
    try:
        return importlib.import_module("build_route_national_flows")
    except ModuleNotFoundError:
        pytest.fail("build_route_national_flows module does not exist")


def _fixture_main_and_aux_rows():
    # The auxiliary row has a Wisconsin home and a Michigan workplace.  It
    # must remain in Michigan's workplace partition.
    return pd.DataFrame(
        {
            "h_geocode": ["260010001001001", "550010001001001", "550010001001002"],
            "w_geocode": ["260010002001001", "260010002001001", "260010002001001"],
            "S000": [3, 5, 2],
            "file_type": ["main", "aux", "aux"],
        }
    )


def _fixture_crosswalk():
    return pd.DataFrame(
        {
            "tabblk2020": [
                "260010001001001", "260010002001001", "550010001001001", "550010001001002",
            ],
            "cty": ["26001", "26001", "55001", "55001"],
            "trct": ["26001000100", "26001000200", "55001000100", "55001000100"],
            "blklatdd": [43.0, 43.1, 44.0, 44.1],
            "blklondd": [-84.0, -84.1, -89.0, -89.1],
        }
    )


def _fixture_car_share():
    return pd.Series(
        [0.8, 0.7], index=pd.Index(["26001000100", "55001000100"], name="tract")
    )


def test_national_flow_partition_keeps_cross_state_aux_rows():
    """Dropping auxiliary cross-state origins must break this coverage test."""
    national = _national_module()

    pairs, diagnostics = national.build_flow_partition(
        block_flows=_fixture_main_and_aux_rows(),
        crosswalks=_fixture_crosswalk(),
        tract_car_share=_fixture_car_share(),
        analysis_year=2022,
        lodes_source_year=2021,
        work_state="mi",
    )

    assert set(pairs["work_state"]) == {"mi"}
    assert pairs.set_index("home_tract").loc["55001000100", "workers"] == 7
    assert diagnostics.loc[0, "lodes_source_year"] == 2021
    assert pairs["route_id"].is_unique


def test_national_flow_partition_route_ids_are_source_aware_and_stable():
    """Removing source metadata from route IDs would permit stale-route reuse."""
    national = _national_module()
    args = dict(
        block_flows=_fixture_main_and_aux_rows(), crosswalks=_fixture_crosswalk(),
        tract_car_share=_fixture_car_share(), work_state="mi",
    )

    first, _ = national.build_flow_partition(**args, analysis_year=2022, lodes_source_year=2021)
    second, _ = national.build_flow_partition(**args, analysis_year=2022, lodes_source_year=2021)
    changed, _ = national.build_flow_partition(**args, analysis_year=2022, lodes_source_year=2020)

    assert first["route_id"].tolist() == second["route_id"].tolist()
    assert set(first["route_id"]).isdisjoint(set(changed["route_id"]))
    assert set(first["analysis_year"]) == {2022}
    assert set(first["lodes_source_year"]) == {2021}
    assert set(first["lodes_file_types"]) == {"aux,main"}


def test_chunked_reader_and_aggregation_match_single_partition(tmp_path):
    """Changing chunk aggregation weights must not change tract-pair results."""
    national = _national_module()
    flow_path = tmp_path / "mi_od_main_JT00_2021.csv.gz"
    _fixture_main_and_aux_rows().drop(columns="file_type").to_csv(
        flow_path, index=False, compression="gzip"
    )

    chunks = list(national.iter_lodes_flow_chunks([flow_path], chunk_rows=1))
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]
    assert all(set(chunk.columns).issuperset({"h_geocode", "w_geocode", "S000", "file_type"}) for chunk in chunks)

    expected, _ = national.build_flow_partition(
        _fixture_main_and_aux_rows(), _fixture_crosswalk(), _fixture_car_share(),
        analysis_year=2022, lodes_source_year=2021, work_state="mi",
    )
    actual, _ = national.build_flow_partition_from_chunks(
        chunks, _fixture_crosswalk(), _fixture_car_share(),
        analysis_year=2022, lodes_source_year=2021, work_state="mi",
    )
    compare_columns = ["home_tract", "work_tract", "workers", "home_lat", "home_lon", "work_lat", "work_lon", "route_id"]
    pd.testing.assert_frame_equal(actual[compare_columns], expected[compare_columns])


def test_unavailable_state_is_manifested_without_an_empty_partition(tmp_path, monkeypatch):
    """Treating a failed LODES download as an empty success must fail this test."""
    national = _national_module()

    def unavailable(*args, **kwargs):
        raise OSError("404 fixture")

    monkeypatch.setattr(national, "download_lodes_input", unavailable)
    manifest = national.build_national_flow_year(2022, ["mi"], tmp_path)

    assert manifest.loc[0, "lodes_status"] == "unavailable"
    assert "404 fixture" in manifest.loc[0, "lodes_reason"]
    assert not list(tmp_path.rglob("work_state=mi.parquet"))
    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "analysis_year=2022" / "work_state=mi.csv")
    assert diagnostics.loc[0, "partition_status"] == "unavailable"


def test_two_state_build_writes_source_provenance_on_each_partition(tmp_path, monkeypatch):
    """Dropping input checksums or source paths must fail partition provenance."""
    national = _national_module()
    flows = {
        "mi": pd.DataFrame({
            "h_geocode": ["550010001001001"], "w_geocode": ["260010002001001"], "S000": [5],
        }),
        "wi": pd.DataFrame({
            "h_geocode": ["260010001001001"], "w_geocode": ["550010001001001"], "S000": [3],
        }),
    }
    crosswalks = _fixture_crosswalk().loc[lambda frame: frame["tabblk2020"].str[:2].isin(["26", "55"])]

    def download(state, file_type, year, cache_dir, session):
        filename = f"{state}_xwalk.csv.gz" if file_type == "xwalk" else f"{state}_od_{file_type}_JT00_{year}.csv.gz"
        path = Path(cache_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_type == "xwalk":
            state_fips = {"mi": "26", "wi": "55"}[state]
            crosswalks.loc[crosswalks["tabblk2020"].str.startswith(state_fips)].to_csv(
                path, index=False, compression="gzip"
            )
        else:
            flows[state].to_csv(path, index=False, compression="gzip")
        return path

    monkeypatch.setattr(national, "download_lodes_input", download)
    monkeypatch.setattr(
        national, "_load_car_share", lambda year: (_fixture_car_share(), "2020", {
            "tracts_with_share": 2.0, "acs_window_start": 2016.0, "acs_window_end": 2020.0,
        }),
    )

    manifest = national.build_national_flow_year(
        2022, ["wi", "mi"], tmp_path, origin_states=("mi", "wi"), chunk_rows=1
    )

    assert set(manifest["state"]) == {"mi", "wi"}
    assert manifest["main_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert manifest["aux_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    for state in ("mi", "wi"):
        partition = pd.read_parquet(
            tmp_path / "partitions" / "analysis_year=2022" / "lodes_source_year=2022" / f"work_state={state}.parquet"
        )
        assert set(partition["work_state"]) == {state}
        assert partition["lodes_source_paths"].str.contains("_od_main_JT00_2022.csv.gz").all()
        assert partition["main_url"].str.endswith("_od_main_JT00_2022.csv.gz").all()
        assert partition["aux_url"].str.endswith("_od_aux_JT00_2022.csv.gz").all()
        assert partition["main_bytes"].gt(0).all()
        assert partition["main_retrieved_at_utc"].notna().all()
        assert partition["lodes_year_gap"].eq(0).all()
        assert partition["acs_window_start"].notna().all()
        assert partition["acs_source_provenance"].notna().all()


def test_scoped_workplace_run_loads_explicit_origin_crosswalk_inventory(tmp_path, monkeypatch):
    """Loading only workplace crosswalks drops interstate auxiliary origins."""
    national = _national_module()
    requested_crosswalks = []

    def download(state, file_type, year, cache_dir, session):
        filename = f"{state}_xwalk.csv.gz" if file_type == "xwalk" else f"{state}_od_{file_type}_JT00_{year}.csv.gz"
        path = Path(cache_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_type == "xwalk":
            requested_crosswalks.append(state)
            state_fips = {"mi": "26", "wi": "55"}[state]
            _fixture_crosswalk().loc[lambda frame: frame["tabblk2020"].str.startswith(state_fips)].to_csv(
                path, index=False, compression="gzip"
            )
        else:
            pd.DataFrame({
                "h_geocode": ["550010001001001"], "w_geocode": ["260010002001001"], "S000": [5],
            }).to_csv(path, index=False, compression="gzip")
        return path

    monkeypatch.setattr(national, "download_lodes_input", download)
    monkeypatch.setattr(
        national, "_load_car_share", lambda year: (_fixture_car_share(), "2020", {
            "tracts_with_share": 2.0, "acs_window_start": 2016.0, "acs_window_end": 2020.0,
        }),
    )

    national.build_national_flow_year(
        2022, ["mi"], tmp_path, origin_states=("mi", "wi"), chunk_rows=1
    )

    assert requested_crosswalks == ["mi", "wi"]
    pairs = pd.read_parquet(
        tmp_path / "partitions" / "analysis_year=2022" / "lodes_source_year=2022" / "work_state=mi.parquet"
    )
    assert pairs.loc[0, "home_state"] == "wi"


def test_omitted_state_argument_defaults_to_the_national_workplace_inventory(tmp_path, monkeypatch):
    """A missing state argument must not silently select a one-state workload."""
    national = _national_module()
    monkeypatch.setattr(national, "_download_nearest_state_flows", lambda *args: (None, [], "fixture"))

    manifest = national.build_national_flow_year(2022, cache_dir=tmp_path)

    assert set(manifest["state"]) == set(national.NATIONAL_STATES)


def test_completed_partition_is_reused_and_stale_metadata_rebuilds(tmp_path, monkeypatch):
    """Ignoring a matching sidecar reprocesses work; trusting a stale one is unsafe."""
    national = _national_module()
    calls = {"aggregate": 0}

    def download(state, file_type, year, cache_dir, session):
        filename = f"{state}_xwalk.csv.gz" if file_type == "xwalk" else f"{state}_od_{file_type}_JT00_{year}.csv.gz"
        path = Path(cache_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_type == "xwalk":
            _fixture_crosswalk().loc[lambda frame: frame["tabblk2020"].str.startswith("26")].to_csv(
                path, index=False, compression="gzip"
            )
        else:
            pd.DataFrame({
                "h_geocode": ["260010001001001"], "w_geocode": ["260010002001001"], "S000": [3],
            }).to_csv(path, index=False, compression="gzip")
        return path

    monkeypatch.setattr(national, "download_lodes_input", download)
    monkeypatch.setattr(
        national, "_load_car_share", lambda year: (_fixture_car_share(), "2020", {
            "tracts_with_share": 2.0, "acs_window_start": 2016.0, "acs_window_end": 2020.0,
        }),
    )
    original = national.build_flow_partition_from_chunks

    def counting_aggregate(*args, **kwargs):
        calls["aggregate"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(national, "build_flow_partition_from_chunks", counting_aggregate)
    kwargs = {"origin_states": ("mi",), "chunk_rows": 1}
    national.build_national_flow_year(2022, ["mi"], tmp_path, **kwargs)
    assert calls["aggregate"] == 1

    national.build_national_flow_year(2022, ["mi"], tmp_path, **kwargs)
    assert calls["aggregate"] == 1

    sidecar = next(tmp_path.rglob("work_state=mi.parquet.manifest.json"))
    metadata = json.loads(sidecar.read_text())
    metadata["main_sha256"] = "stale"
    sidecar.write_text(json.dumps(metadata))
    national.build_national_flow_year(2022, ["mi"], tmp_path, **kwargs)
    assert calls["aggregate"] == 2

    # Keep the regenerated sidecar intact while corrupting a non-key data
    # value.  Partial column checks would incorrectly reuse this partition.
    partition_path = sidecar.with_name(sidecar.name.removesuffix(".manifest.json"))
    corrupted = pd.read_parquet(partition_path)
    corrupted.loc[0, "workers"] = 999
    corrupted.to_parquet(partition_path, index=False)
    national.build_national_flow_year(2022, ["mi"], tmp_path, **kwargs)
    assert calls["aggregate"] == 3
    assert pd.read_parquet(partition_path).loc[0, "workers"] == 6


def test_chunk_reducer_limits_open_partial_pair_frames(monkeypatch, tmp_path):
    """A reducer that receives every chunk frame at once is not bounded."""
    national = _national_module()
    observed_batch_sizes = []
    original = national._combine_chunk_pairs

    def bounded_combine(frames):
        frames = list(frames)
        observed_batch_sizes.append(len(frames))
        assert len(frames) <= 2
        return original(frames)

    monkeypatch.setattr(national, "_combine_chunk_pairs", bounded_combine)
    chunks = [
        pd.DataFrame({"h_geocode": ["260010001001001"], "w_geocode": ["260010002001001"], "S000": [1]})
        for _ in range(9)
    ]

    pairs, _ = national.build_flow_partition_from_chunks(
        chunks, _fixture_crosswalk(), _fixture_car_share(), analysis_year=2022,
        lodes_source_year=2022, work_state="mi", max_open_chunk_partitions=2,
    )

    assert pairs.loc[0, "workers"] == 9
    assert observed_batch_sizes
