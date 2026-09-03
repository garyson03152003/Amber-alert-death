import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_resolve_nearest_year_prefers_exact_then_earlier_tie():
    from route_vintages import resolve_nearest_year

    assert resolve_nearest_year(2022, [2018, 2022, 2023]).source_year == 2022
    assert resolve_nearest_year(2021, [2019, 2023]).source_year == 2019


def test_resolve_nearest_year_rejects_empty_candidates():
    from route_vintages import resolve_nearest_year

    with pytest.raises(ValueError, match="no available source years"):
        resolve_nearest_year(2022, [])


def test_resolve_nearest_year_validates_and_deduplicates_candidates():
    from route_vintages import resolve_nearest_year

    assert resolve_nearest_year(2021, [2019, 2019, 2023]).source_year == 2019
    with pytest.raises(TypeError, match="integer"):
        resolve_nearest_year("2021", [2020])
    with pytest.raises(TypeError, match="integer"):
        resolve_nearest_year(2021, [2020.0])


def test_resolve_acs_window_prefers_containing_window_and_midpoint_tie():
    from route_vintages import resolve_acs_window

    choice = resolve_acs_window(2021, [(2018, 2022, "2018-2022"), (2020, 2024, "2020-2024")])
    assert choice.source_year == 2020
    assert choice.status == "exact"
    assert "2018-2022" in choice.reason


def test_resolve_acs_window_handles_unavailable_and_noncontaining_midpoint_ties():
    from route_vintages import resolve_acs_window

    unavailable = resolve_acs_window(2022, [])
    assert unavailable.status == "unavailable"
    assert unavailable.source_year is None
    choice = resolve_acs_window(2022, [(2010, 2012, "2010-2012"), (2032, 2034, "2032-2034")])
    assert choice.source_year == 2011
    assert choice.gap == 11


def test_write_vintage_manifest_is_versioned_sorted_and_atomic(tmp_path):
    from route_vintages import write_vintage_manifest

    path = tmp_path / "vintages.json"
    write_vintage_manifest(
        [
            {"analysis_year": 2022, "state": "TX", "source_year": 2020},
            {"analysis_year": 2021, "state": "CA", "source_year": 2019},
            {"analysis_year": 2021, "state": "CA", "source_year": 2018},
        ],
        path,
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"]
    assert [(row["analysis_year"], row["state"], row["source_year"]) for row in payload["records"]] == [
        (2021, "CA", 2018),
        (2021, "CA", 2019),
        (2022, "TX", 2020),
    ]


def test_write_vintage_manifest_csv_has_schema_version_and_safe_mixed_sort(tmp_path):
    from route_vintages import write_vintage_manifest

    path = tmp_path / "vintages.csv"
    write_vintage_manifest(
        [
            {"analysis_year": 2021, "state": "CA", "lodes_source_year": 2019},
            {"analysis_year": 2021, "state": "CA", "source_year": None, "lodes_source_year": 2018},
            {"analysis_year": 2020, "state": None, "source_year": None},
        ],
        path,
    )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["analysis_year"] == "2020"
    assert rows[1]["state"] == "CA"
    assert rows[1]["lodes_source_year"] == "2018"
    assert rows[1]["schema_version"] == "route_vintages.v1"


def test_write_vintage_manifest_preserves_existing_file_on_failure(tmp_path):
    from route_vintages import write_vintage_manifest

    path = tmp_path / "vintages.json"
    path.write_text("original\n")
    with pytest.raises(TypeError, match="integer"):
        write_vintage_manifest([{"analysis_year": "bad", "state": "CA", "source_year": 2020}], path)
    assert path.read_text() == "original\n"
