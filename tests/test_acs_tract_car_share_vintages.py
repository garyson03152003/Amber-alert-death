import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_build_car_share_uses_car_total_over_workers():
    from build_acs_tract_car_share_vintages import build_car_share_frame

    result = build_car_share_frame(pd.DataFrame({
        "GEO_ID": ["1400000US55001000100"],
        "B08301_E001": [100], "B08301_E002": [80],
    }))

    assert result.loc[0, "tract"] == "55001000100"
    assert result.loc[0, "total_workers"] == 100
    assert result.loc[0, "car_total"] == 80
    assert result.loc[0, "car_share"] == 0.8


def test_missing_or_zero_worker_rows_are_omitted_and_counted():
    from build_acs_tract_car_share_vintages import build_car_share_frame

    result, diagnostics = build_car_share_frame(pd.DataFrame({
        "GEO_ID": ["1400000US55001000100", "1400000US55001000200", "not-a-tract"],
        "B08301_E001": [0, 10, "bad"], "B08301_E002": [0, 5, 3],
    }), return_diagnostics=True)

    assert result.to_dict("records") == [{
        "tract": "55001000200", "total_workers": 10.0, "car_total": 5.0,
        "car_share": 0.5,
    }]
    assert diagnostics["zero_worker_rows"] == 1
    assert diagnostics["malformed_rows"] == 1
    assert diagnostics["omitted_rows"] == 2


def test_out_of_range_car_shares_are_omitted_and_counted():
    from build_acs_tract_car_share_vintages import build_car_share_frame

    result, diagnostics = build_car_share_frame(pd.DataFrame({
        "GEO_ID": ["1400000US55001000100"],
        "B08301_E001": [10], "B08301_E002": [11],
    }), return_diagnostics=True)

    assert result.empty
    assert diagnostics["out_of_range_share_rows"] == 1


def test_loader_selects_containing_cached_window_without_filling_missing_shares(tmp_path):
    from build_acs_tract_car_share_vintages import load_car_share_for_analysis_year

    vintage_dir = tmp_path / "acs_2020"
    vintage_dir.mkdir()
    pd.DataFrame({
        "tract": ["55001000100"], "total_workers": [100],
        "car_total": [80], "car_share": [0.8],
    }).to_parquet(vintage_dir / "state=WI.parquet", index=False)
    (vintage_dir / "metadata.json").write_text(json.dumps({
        "acs_vintage": "2020", "window_start": 2016, "window_end": 2020,
        "sources": [{"url": "https://example.test/wi", "sha256": "abc"}],
    }))

    series, vintage, diagnostics = load_car_share_for_analysis_year(
        2018, tmp_path, expected_states=("WI",)
    )

    assert vintage == "2020"
    assert series.to_dict() == {"55001000100": 0.8}
    assert diagnostics["acs_window_start"] == 2016.0
    assert diagnostics["acs_window_end"] == 2020.0
    assert diagnostics["missing_share_fill_count"] == 0.0


def test_loader_fails_closed_when_national_state_coverage_is_incomplete(tmp_path):
    from build_acs_tract_car_share_vintages import load_car_share_for_analysis_year

    vintage_dir = tmp_path / "acs_2020"
    vintage_dir.mkdir()
    pd.DataFrame({"tract": ["55001000100"], "car_share": [0.8]}).to_parquet(
        vintage_dir / "state=WI.parquet", index=False
    )
    (vintage_dir / "metadata.json").write_text(json.dumps({
        "acs_vintage": "2020", "window_start": 2016, "window_end": 2020,
    }))

    import pytest
    with pytest.raises(FileNotFoundError, match="incomplete ACS tract car-share coverage"):
        load_car_share_for_analysis_year(2018, tmp_path, expected_states=("WI", "IL"))


def test_loader_accepts_explicitly_complete_state_coverage(tmp_path):
    from build_acs_tract_car_share_vintages import load_car_share_for_analysis_year

    vintage_dir = tmp_path / "acs_2020"
    vintage_dir.mkdir()
    for state, tract, share in (("WI", "55001000100", 0.8), ("IL", "17001000100", 0.7)):
        pd.DataFrame({"tract": [tract], "car_share": [share]}).to_parquet(
            vintage_dir / f"state={state}.parquet", index=False
        )
    (vintage_dir / "metadata.json").write_text(json.dumps({
        "acs_vintage": "2020", "window_start": 2016, "window_end": 2020,
    }))

    series, _, diagnostics = load_car_share_for_analysis_year(
        2018, tmp_path, expected_states=("WI", "IL")
    )

    assert series.to_dict() == {"17001000100": 0.7, "55001000100": 0.8}
    assert diagnostics["expected_state_partitions"] == 2.0
    assert diagnostics["present_state_partitions"] == 2.0


def test_failed_metadata_update_does_not_replace_existing_partition(monkeypatch, tmp_path):
    import build_acs_tract_car_share_vintages as builder

    raw = pd.DataFrame({
        "GEO_ID": ["1400000US55001000100"], "B08301_E001": [10], "B08301_E002": [8],
    })
    path, _ = builder.write_state_partition(
        raw, vintage=2020, state="WI", source_url="https://one.test", source_bytes=b"one",
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(builder, "_atomic_write_json", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no metadata")))
    replacement = raw.assign(B08301_E002=[5])

    import pytest
    with pytest.raises(OSError, match="no metadata"):
        builder.write_state_partition(
            replacement, vintage=2020, state="WI", source_url="https://two.test", source_bytes=b"two",
            cache_dir=tmp_path,
        )

    assert pd.read_parquet(path).loc[0, "car_share"] == 0.8


def test_partition_embeds_matching_source_provenance(tmp_path):
    import pyarrow.parquet as pq
    import build_acs_tract_car_share_vintages as builder

    raw = pd.DataFrame({
        "GEO_ID": ["1400000US55001000100"], "B08301_E001": [10], "B08301_E002": [8],
    })
    path, _ = builder.write_state_partition(
        raw, vintage=2020, state="WI", source_url="https://one.test", source_bytes=b"one",
        cache_dir=tmp_path,
    )

    provenance = json.loads(pq.read_metadata(path).metadata[b"route_national.provenance"])

    assert provenance["state"] == "WI"
    assert provenance["url"] == "https://one.test"
    assert provenance["sha256"] == "7692c3ad3540bb803c020b3aee66cd8887123234ea0c6e7143c0add73ff431ed"


def test_loader_uses_exact_window_identity_selected_by_resolver(tmp_path):
    from build_acs_tract_car_share_vintages import load_car_share_for_analysis_year

    for vintage, start, end, tract, share in (
        ("A", 2018, 2018, "55001000100", 0.1),
        ("B", 2017, 2019, "17001000100", 0.9),
    ):
        directory = tmp_path / f"acs_{vintage}"
        directory.mkdir()
        pd.DataFrame({"tract": [tract], "car_share": [share]}).to_parquet(directory / "state=WI.parquet", index=False)
        (directory / "metadata.json").write_text(json.dumps({
            "acs_vintage": vintage, "window_start": start, "window_end": end,
        }))

    series, vintage, _ = load_car_share_for_analysis_year(2017, tmp_path, expected_states=("WI",))

    assert vintage == "B"
    assert series.to_dict() == {"17001000100": 0.9}


def test_legacy_fetch_reuses_pilot_archive_parser(monkeypatch):
    import build_acs_tract_car_share_vintages as builder

    class Response:
        content = b"archive"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(builder.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(builder, "parse_legacy_tract_archive", lambda payload, sequence_number: pd.DataFrame({
        "tract": ["55001000100"], "total_workers": [10], "car_total": [8],
    }))

    raw, payload, url = builder.fetch_legacy_state(
        2020, "Wisconsin", sequence_number="0027", start_position=157
    )

    assert payload == b"archive"
    assert url.endswith("Wisconsin_Tracts_Block_Groups_Only.zip")
    assert raw.to_dict("records") == [{
        "GEO_ID": "1400000US55001000100", "B08301_E001": 10, "B08301_E002": 8,
    }]


def test_legacy_sequence_location_comes_from_vintage_lookup(monkeypatch):
    import build_acs_tract_car_share_vintages as builder

    class Response:
        content = (
            b"File ID,Table ID,Sequence Number,Line Number,Start Position,Total Cells in Table,Total Cells in Sequence,Table Title,Subject Area\n"
            b"ACSSF,B08301,0028,,157,21 CELLS,199 CELLS,MEANS OF TRANSPORTATION TO WORK,Employment\n"
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr(builder.requests, "get", lambda *args, **kwargs: Response())

    assert builder.resolve_legacy_table_location(2015) == ("0028", 157)


def test_legacy_fetch_passes_vintage_sequence_to_parser(monkeypatch):
    import build_acs_tract_car_share_vintages as builder

    class Response:
        content = b"archive"

        def raise_for_status(self):
            return None

    seen = {}
    monkeypatch.setattr(builder.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(builder, "resolve_legacy_table_location", lambda vintage: ("0028", 157))

    def fake_parse(payload, *, sequence_number, start_position=157):
        seen.update(sequence_number=sequence_number, start_position=start_position)
        return pd.DataFrame({
            "tract": ["55001000100"], "total_workers": [10], "car_total": [8],
        })

    monkeypatch.setattr(builder, "parse_legacy_tract_archive", fake_parse)
    raw, _, _ = builder.fetch_state_for_vintage(2015, "55")

    assert seen == {"sequence_number": "0028", "start_position": 157}
    assert raw["B08301_E001"].tolist() == [10]


def test_cli_selects_legacy_source_and_canonical_dc_archive_name(monkeypatch, tmp_path):
    import build_acs_tract_car_share_vintages as builder

    calls = []

    def fake_legacy(vintage, state_name):
        calls.append(("legacy", vintage, state_name))
        return pd.DataFrame({
            "GEO_ID": ["1400000US11001000100"],
            "B08301_E001": [10],
            "B08301_E002": [8],
        }), b"legacy", "https://example.test/legacy"

    def fake_write(raw, **kwargs):
        calls.append(("write", kwargs["vintage"], kwargs["state"]))
        return tmp_path / "partition.parquet", {"retained_rows": len(raw)}

    monkeypatch.setattr(builder, "fetch_legacy_state", fake_legacy)
    monkeypatch.setattr(builder, "write_state_partition", fake_write)

    assert builder.main(["2020", "11", "--source", "legacy", "--cache-dir", str(tmp_path)]) == 0

    assert calls == [
        ("legacy", 2020, "DistrictOfColumbia"),
        ("write", 2020, "11"),
    ]


def test_cli_auto_selects_table_source_for_2021(monkeypatch, tmp_path):
    import build_acs_tract_car_share_vintages as builder

    calls = []

    def fake_table(vintage, state_fips):
        calls.append(("table", vintage, state_fips))
        return pd.DataFrame({
            "GEO_ID": ["1400000US11001000100"],
            "B08301_E001": [10],
            "B08301_E002": [8],
        }), b"table", "https://example.test/table"

    def fake_write(raw, **kwargs):
        calls.append(("write", kwargs["vintage"], kwargs["state"]))
        return tmp_path / "partition.parquet", {"retained_rows": len(raw)}

    monkeypatch.setattr(builder, "fetch_table_based_state", fake_table)
    monkeypatch.setattr(builder, "write_state_partition", fake_write)

    assert builder.main(["2021", "11", "--cache-dir", str(tmp_path)]) == 0

    assert calls == [
        ("table", 2021, "11"),
        ("write", 2021, "11"),
    ]


def test_source_selection_covers_legacy_2009_through_2020_and_table_2021_through_2024():
    import build_acs_tract_car_share_vintages as builder

    assert builder.resolve_source(2009) == "legacy"
    assert builder.resolve_source(2020) == "legacy"
    assert builder.resolve_source(2021) == "table"
    assert builder.resolve_source(2024) == "table"


def test_legacy_state_name_uses_published_archive_names_for_dc_and_states():
    import build_acs_tract_car_share_vintages as builder

    assert builder.legacy_state_name("11") == "DistrictOfColumbia"
    assert builder.legacy_state_name("55") == "Wisconsin"
