import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _pilot_module():
    try:
        return importlib.import_module("build_route_pilot_flows")
    except ModuleNotFoundError:
        pytest.fail("build_route_pilot_flows module does not exist")


def test_lodes_url_uses_2022_jt00_file():
    pilot = _pilot_module()
    assert hasattr(pilot, "lodes_url")
    assert pilot.lodes_url("wi", "main", 2022).endswith("wi_od_main_JT00_2022.csv.gz")
    assert pilot.lodes_url("wi", "xwalk", 2022).endswith("wi_xwalk.csv.gz")


def test_manifest_records_checksum_and_source_metadata(tmp_path):
    pilot = _pilot_module()
    assert hasattr(pilot, "manifest_record")

    src = tmp_path / "input.csv.gz"
    src.write_bytes(b"fixture")
    record = pilot.manifest_record(src, "https://example.test/input.csv.gz", "wi", "main", 2022)

    assert set(["path", "url", "retrieved_at_utc", "sha256", "bytes", "state", "file_type", "year", "source_name", "attribution", "license"]).issubset(record)
    assert record["bytes"] == 7
    assert record["state"] == "wi"
    assert record["file_type"] == "main"
    assert record["year"] == 2022
    assert record["url"] == "https://example.test/input.csv.gz"
    assert record["source_name"].startswith("U.S. Census Bureau")


def test_cached_lodes_input_requires_matching_sidecar_checksum(tmp_path):
    pilot = _pilot_module()
    target = tmp_path / "wi_od_main_JT00_2022.csv.gz"
    target.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="manifest/checksum validation"):
        pilot.download_lodes_input("wi", "main", 2022, tmp_path, object())


def _fixture_block_flows():
    return pd.DataFrame(
        {
            "h_geocode": [
                "550010001001001",
                "550010001001002",
                "170010001001001",
                "260010001001001",
            ],
            "w_geocode": [
                "550010002001001",
                "550010001001001",
                "190010001001001",
                "480010001001001",
            ],
            "S000": [3, 1, 2, 4],
        }
    )


def _fixture_crosswalk():
    return pd.DataFrame(
        {
            "tabblk2020": [
                "550010001001001",
                "550010001001002",
                "550010002001001",
                "170010001001001",
                "190010001001001",
                "260010001001001",
            ],
            "cty": ["55001", "55001", "55001", "17001", "19001", "26001"],
            "trct": [
                "55001000100",
                "55001000100",
                "55001000200",
                "17001000100",
                "19001000100",
                "26001000100",
            ],
            "blklatdd": [43.00, 43.01, 43.10, 42.00, 41.00, 42.50],
            "blklondd": [-89.40, -89.39, -89.30, -88.00, -93.00, -83.00],
        }
    )


def _fixture_car_share():
    return pd.Series(
        {
            "55001000100": 0.80,
            "55001000200": 0.90,
            "17001000100": 0.75,
            "19001000100": 0.70,
            "26001000100": 0.65,
        }
    )


def test_build_pilot_tract_pairs_reports_external_and_missing_weights():
    pilot = _pilot_module()
    assert hasattr(pilot, "build_pilot_tract_pairs")

    pairs, diagnostics = pilot.build_pilot_tract_pairs(
        _fixture_block_flows(),
        _fixture_crosswalk(),
        _fixture_car_share(),
    )

    required_schema = {
        "route_id",
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
        "commuter_car_weight",
        "block_pair_straight_line_miles",
        "commuter_car_miles",
        "routing_eligible",
        "omitted_coordinate_worker_weight",
        "omitted_car_share_worker_weight",
        "same_tract",
        "home_state",
        "work_state",
    }
    assert required_schema.issubset(pairs.columns)
    assert pairs["route_id"].is_unique
    assert pairs["route_id"].tolist() == sorted(pairs["route_id"].tolist())
    assert pairs["commuter_car_weight"].equals(pairs["workers"] * pairs["home_car_share"])
    assert pairs["commuter_car_miles"].equals(
        pairs["commuter_car_weight"] * pairs["block_pair_straight_line_miles"]
    )
    assert diagnostics.loc[0, "input_worker_weight"] >= diagnostics.loc[0, "retained_worker_weight"]
    assert diagnostics.loc[0, "external_endpoint_worker_weight"] >= 0
    assert diagnostics.loc[0, "missing_coordinate_worker_weight"] >= 0
    assert diagnostics.loc[0, "missing_home_car_share_worker_weight"] >= 0


def test_build_pilot_tract_pairs_fails_closed_but_preserves_omitted_weights():
    pilot = _pilot_module()
    crosswalk = _fixture_crosswalk().copy()
    crosswalk.loc[crosswalk["tabblk2020"].eq("550010002001001"), "blklatdd"] = pd.NA
    car_share = _fixture_car_share().drop("17001000100")

    pairs, diagnostics = pilot.build_pilot_tract_pairs(
        _fixture_block_flows(), crosswalk, car_share
    )

    missing_coordinate = pairs.loc[pairs["missing_endpoint_workers"].gt(0)].iloc[0]
    missing_car = pairs.loc[pairs["home_car_share"].isna()].iloc[0]
    assert not bool(missing_coordinate["routing_eligible"])
    assert missing_coordinate["omitted_coordinate_worker_weight"] > 0
    assert not bool(missing_car["routing_eligible"])
    assert missing_car["omitted_car_share_worker_weight"] > 0
    assert pd.isna(missing_car["commuter_car_weight"])
    assert diagnostics.loc[0, "missing_coordinate_worker_weight"] > 0
    assert diagnostics.loc[0, "missing_home_car_share_worker_weight"] > 0


def test_build_pilot_tract_pairs_keeps_same_tract_rows():
    pilot = _pilot_module()

    pairs, diagnostics = pilot.build_pilot_tract_pairs(
        _fixture_block_flows(),
        _fixture_crosswalk(),
        _fixture_car_share(),
    )

    same_tract = pairs.loc[pairs["same_tract"]]
    assert not same_tract.empty
    assert same_tract.iloc[0]["home_tract"] == same_tract.iloc[0]["work_tract"]
    assert diagnostics.loc[0, "same_tract_worker_weight"] > 0


def test_build_pilot_tract_pairs_preserves_urban_rural_class_when_available():
    pilot = _pilot_module()
    crosswalk = _fixture_crosswalk().assign(
        urban_rural_class=["urban", "urban", "urban", "rural", "rural", "urban"]
    )

    pairs, _ = pilot.build_pilot_tract_pairs(
        _fixture_block_flows(), crosswalk, _fixture_car_share()
    )

    assert "urban_rural_class" in pairs
    assert pairs.set_index("home_tract").loc["17001000100", "urban_rural_class"] == "rural"


def test_main_four_state_mode_excludes_michigan_endpoint_rows(tmp_path, monkeypatch):
    pilot = _pilot_module()
    flows = pd.concat(
        [
            _fixture_block_flows(),
            pd.DataFrame(
                {
                    "h_geocode": ["260010001001001"],
                    "w_geocode": ["550010002001001"],
                    "S000": [5],
                }
            ),
        ],
        ignore_index=True,
    )

    monkeypatch.setattr(
        pilot,
        "download_lodes_input",
        lambda state, file_type, year, cache_dir, session: Path(cache_dir)
        / pilot._download_target_path(state, file_type, year),
    )
    monkeypatch.setattr(
        pilot,
        "manifest_record",
        lambda path, url, state, file_type, year: {
            "path": str(path),
            "url": url,
            "retrieved_at_utc": "2026-09-02T00:00:00+00:00",
            "sha256": "fixture",
            "bytes": 1,
            "state": state,
            "file_type": file_type,
            "year": year,
            "source_name": "fixture",
            "attribution": "fixture",
            "license": "fixture",
        },
    )
    monkeypatch.setattr(pilot, "load_lodes_block_flows", lambda paths: flows)
    monkeypatch.setattr(pilot, "_load_lodes_crosswalks", lambda paths: _fixture_crosswalk())
    monkeypatch.setattr(pilot, "_load_tract_car_share", lambda path: _fixture_car_share())

    cache_dir = tmp_path / "route-pilot"
    assert (
        pilot.main(
            [
                "--states",
                "wi",
                "il",
                "ia",
                "mn",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )

    pairs = pd.read_parquet(cache_dir / pilot.PAIR_OUTPUT_NAME)
    assert "26" not in set(pairs["home_state_fips"])
    assert "26" not in set(pairs["work_state_fips"])
    diagnostics = pd.read_csv(cache_dir / pilot.DIAGNOSTICS_OUTPUT_NAME)
    assert diagnostics.loc[0, "external_endpoint_worker_weight"] == pytest.approx(9.0)


def test_main_keeps_2022_and_2021_artifacts_separate(tmp_path, monkeypatch):
    pilot = _pilot_module()
    monkeypatch.setattr(
        pilot,
        "download_lodes_input",
        lambda state, file_type, year, cache_dir, session: Path(cache_dir)
        / pilot._download_target_path(state, file_type, year),
    )
    monkeypatch.setattr(
        pilot,
        "manifest_record",
        lambda path, url, state, file_type, year: {
            "path": str(path),
            "url": url,
            "retrieved_at_utc": "2026-09-02T00:00:00+00:00",
            "sha256": f"fixture-{year}",
            "bytes": 1,
            "state": state,
            "file_type": file_type,
            "year": year,
            "source_name": "fixture",
            "attribution": "fixture",
            "license": "fixture",
        },
    )

    def flows_for_year(paths):
        _, _, year = pilot._parse_flow_path(Path(paths[0]))
        return pd.DataFrame(
            {
                "h_geocode": ["550010001001001"],
                "w_geocode": ["550010002001001"],
                "S000": [year],
            }
        )

    monkeypatch.setattr(pilot, "load_lodes_block_flows", flows_for_year)
    monkeypatch.setattr(pilot, "_load_lodes_crosswalks", lambda paths: _fixture_crosswalk())
    monkeypatch.setattr(pilot, "_load_tract_car_share", lambda path: _fixture_car_share())

    cache_dir = tmp_path / "route-pilot"
    for year in (2022, 2021):
        assert (
            pilot.main(
                [
                    "--year",
                    str(year),
                    "--states",
                    "wi",
                    "--cache-dir",
                    str(cache_dir),
                ]
            )
            == 0
        )

    paths_2022 = pilot._output_paths(cache_dir, 2022)
    paths_2021 = pilot._output_paths(cache_dir, 2021)
    assert paths_2022["pairs"].name == pilot.PAIR_OUTPUT_NAME
    assert paths_2022["diagnostics"].name == pilot.DIAGNOSTICS_OUTPUT_NAME
    assert paths_2022["manifest"].name == pilot.MANIFEST_OUTPUT_NAME
    assert set(paths_2022.values()).isdisjoint(set(paths_2021.values()))
    assert all(path.is_file() for path in paths_2022.values())
    assert all(path.is_file() for path in paths_2021.values())
    assert pd.read_parquet(paths_2022["pairs"]).loc[0, "workers"] == 2022
    assert pd.read_parquet(paths_2021["pairs"]).loc[0, "workers"] == 2021


def test_load_tract_car_share_raises_actionable_preflight_error(monkeypatch, tmp_path):
    pilot = _pilot_module()
    monkeypatch.setattr(pilot, "TRACT_CAR_SHARE_PATH", tmp_path / "tract_car_share.parquet")

    def fail_read_parquet(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(pilot.pd, "read_parquet", fail_read_parquet)

    with pytest.raises(RuntimeError, match=r"pyarrow==25\.0\.1.*python code/build_acs_tract_car_share\.py"):
        pilot._load_tract_car_share()


def test_route_pilot_cache_is_ignored():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text()
    assert "data/processed/commuting/route_pilot/" in text
