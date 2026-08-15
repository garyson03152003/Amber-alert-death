import importlib
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _accidents():
    return pd.DataFrame(
        {
            "ST_CASE": [1, 2, 3, 4, 5, 5, 6],
            "STATE": [1, 1, 1, 72, 1, 1, 1],
            "COUNTY": [1, 0, 999, 1, 3, 3, 5],
            "YEAR": [2024] * 7,
            "MONTH": [1, 1, 1, 1, 2, 2, 2],
            "DAY": [2, 2, 2, 2, 30, 30, 29],
            "FATALS": [2, 3, 4, 5, 1, 1, 2],
            "WEATHER": [2, 1, 1, 1, 1, 1, 10],
        }
    )


def _vehicles():
    return pd.DataFrame({"ST_CASE": [1, 5, 6], "DR_DRINK": [1, 0, 1]})


def _builder(monkeypatch):
    module = importlib.import_module("build_fars_county_day")
    monkeypatch.setattr(module, "fetch_zip", lambda year, session: object())
    monkeypatch.setattr(
        module,
        "read_file",
        lambda _zip, keyword: _accidents() if keyword == "accident" else _vehicles(),
    )
    return module


def test_build_fars_year_rejects_invalid_rows_and_reconciles_2024(monkeypatch):
    builder = _builder(monkeypatch)

    events, coverage = builder.build_fars_year(2024, session=object())

    assert events.columns.tolist() == [
        "fips", "date", "fatal_crashes", "person_fatals", "drunk_fatals",
        "sober_fatals", "weather_adverse",
    ]
    assert events.to_dict("records") == [{
        "fips": "01001", "date": pd.Timestamp("2024-01-02"),
        "fatal_crashes": 1, "person_fatals": 2, "drunk_fatals": 2,
        "sober_fatals": 0, "weather_adverse": 1,
    }, {
        "fips": "01005", "date": pd.Timestamp("2024-02-29"),
        "fatal_crashes": 1, "person_fatals": 2, "drunk_fatals": 2,
        "sober_fatals": 0, "weather_adverse": 0,
    }]
    assert coverage.year == 2024
    assert coverage.fetched_records == 7
    assert coverage.duplicate_records == 1
    assert coverage.invalid_geography_count == 3  # county 000, 999, Puerto Rico
    assert coverage.invalid_date_count == 1  # duplicate ST_CASE is removed before cleaning
    assert coverage.retained_records == 2
    assert coverage.coverage_valid is False
    assert {"duplicate_records", "invalid_dates", "invalid_geography"}.issubset(
        coverage.failure_reasons
    )


def test_fars_county_universe_is_contiguous_us_only_and_excludes_connecticut():
    builder = importlib.import_module("build_fars_county_day")
    population = pd.DataFrame(
        {
            "fips": ["01001", "09001", "11001", "72001", "01001"],
            "year": [2024, 2024, 2024, 2024, 2023],
            "population": [10, 10, 10, 10, 10],
        }
    )

    universe = builder.fars_county_universe(population, years=[2023, 2024])

    assert universe.to_dict("records") == [
        {"fips": "01001", "year": 2023},
        {"fips": "01001", "year": 2024},
        {"fips": "11001", "year": 2024},
    ]
    assert builder.CONNECTICUT_LONGITUDINAL_POLICY == "exclude_until_crosswalk"
    assert builder.CONNECTICUT_MANIFEST_WARNING == "connecticut_excluded_from_longitudinal_panel"


def test_build_manifest_records_connecticut_policy_without_dropping_sparse_events(monkeypatch):
    builder = importlib.import_module("build_fars_county_day")
    monkeypatch.setattr(builder, "YEARS", (2024,))
    events = pd.DataFrame({
        "fips": ["09001"], "date": [pd.Timestamp("2024-01-01")],
        "fatal_crashes": [1], "person_fatals": [1], "drunk_fatals": [0],
        "sober_fatals": [1], "weather_adverse": [0],
    })
    coverage = builder.validate_reporting_unit(
        source=builder.FARS_SOURCE, state="US", year=2024,
        expected_records=1, fetched_records=1, retained_records=1,
        request_complete=True,
    )
    monkeypatch.setattr(builder, "build_fars_year", lambda year, session: (events, coverage))

    sparse, manifest = builder.build_fars(years=[2024])

    assert sparse["fips"].tolist() == ["09001"]
    warning = manifest.loc[manifest["source"].eq(f"{builder.FARS_SOURCE}_POLICY")].iloc[0]
    assert warning["state"] == "09"
    assert bool(warning["coverage_valid"]) is False
    assert warning["failure_reasons"] == (builder.CONNECTICUT_MANIFEST_WARNING,)
