import importlib
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest


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
    monkeypatch.setattr(module, "fetch_zip", lambda year, session: (object(), "a" * 64))
    monkeypatch.setattr(module, "permitted_fips_for_year", lambda year: {"01001", "01003", "01005"})
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
    # County 000 and 999 are FARS's own "unknown county" placeholders: they
    # are excluded from the panel but tracked separately and do not by
    # themselves fail coverage_valid. Puerto Rico (state 72) is a genuine
    # non-US-state row and remains a hard invalid_geography failure.
    assert coverage.invalid_geography_count == 1  # Puerto Rico only
    assert coverage.unresolvable_geography_count == 2  # county 000, 999
    assert coverage.invalid_date_count == 1  # duplicate ST_CASE is removed before cleaning
    assert coverage.retained_records == 2
    assert coverage.source_checksum == "a" * 64
    assert coverage.coverage_valid is False
    assert {"duplicate_records", "invalid_dates", "invalid_geography"}.issubset(
        coverage.failure_reasons
    )


def test_build_fars_year_excludes_unresolvable_fips_without_failing_coverage(monkeypatch):
    builder = _builder(monkeypatch)
    accidents = pd.concat([_accidents().iloc[[0]], pd.DataFrame({
        "ST_CASE": [99], "STATE": [1], "COUNTY": [997], "YEAR": [2024],
        "MONTH": [1], "DAY": [3], "FATALS": [1], "WEATHER": [1],
    })], ignore_index=True)
    monkeypatch.setattr(
        builder,
        "read_file",
        lambda _zip, keyword: accidents if keyword == "accident" else _vehicles(),
    )

    events, coverage = builder.build_fars_year(2024, session=object())

    # County 997 never matches any Census-recognized geography (the mocked
    # permitted set), so the row is excluded from the panel. On its own,
    # with nothing else wrong, this small bounded residual does not fail
    # the reporting unit.
    assert events["fips"].tolist() == ["01001"]
    assert coverage.invalid_geography_count == 0
    assert coverage.unresolvable_geography_count == 1
    assert coverage.coverage_valid is True


def test_build_fars_year_classifies_nonintegral_geography_as_invalid(monkeypatch):
    builder = _builder(monkeypatch)
    accidents = pd.DataFrame({
        "ST_CASE": [1], "STATE": [1.5], "COUNTY": [1], "YEAR": [2024],
        "MONTH": [1], "DAY": [3], "FATALS": [1], "WEATHER": [1],
    })
    monkeypatch.setattr(
        builder,
        "read_file",
        lambda _zip, keyword: accidents if keyword == "accident" else _vehicles(),
    )

    events, coverage = builder.build_fars_year(2024, session=object())

    assert events.empty
    assert coverage.invalid_geography_count == 1
    assert coverage.coverage_valid is False


def test_build_fars_year_rejects_rows_from_another_archive_year(monkeypatch):
    builder = _builder(monkeypatch)
    accidents = pd.DataFrame({
        "ST_CASE": [1], "STATE": [1], "COUNTY": [1], "YEAR": [2023],
        "MONTH": [1], "DAY": [3], "FATALS": [1], "WEATHER": [1],
    })
    monkeypatch.setattr(
        builder,
        "read_file",
        lambda _zip, keyword: accidents if keyword == "accident" else _vehicles(),
    )

    events, coverage = builder.build_fars_year(2024, session=object())

    assert events.empty
    assert coverage.invalid_date_count == 1
    assert coverage.coverage_valid is False
    assert "invalid_dates" in coverage.failure_reasons


def test_build_fars_year_rejects_nonfatal_accident_rows(monkeypatch):
    builder = _builder(monkeypatch)
    accidents = pd.DataFrame({
        "ST_CASE": [1], "STATE": [1], "COUNTY": [1], "YEAR": [2024],
        "MONTH": [1], "DAY": [3], "FATALS": [0], "WEATHER": [1],
    })
    monkeypatch.setattr(
        builder,
        "read_file",
        lambda _zip, keyword: accidents if keyword == "accident" else _vehicles(),
    )

    events, coverage = builder.build_fars_year(2024, session=object())

    assert events.empty
    assert coverage.invalid_date_count == 1
    assert coverage.coverage_valid is False


def test_fetch_zip_records_sha256(monkeypatch):
    builder = importlib.import_module("build_fars_county_day")
    payload_buffer = io.BytesIO()
    with zipfile.ZipFile(payload_buffer, "w") as archive:
        archive.writestr("accident.csv", "ST_CASE\n1\n")
    payload = payload_buffer.getvalue()

    class Response:
        content = payload

        def raise_for_status(self):
            return None

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    _archive, checksum = builder.fetch_zip(2024, Session())

    assert checksum == hashlib.sha256(payload).hexdigest()


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


def test_permitted_fips_retains_legacy_connecticut_events_through_transition():
    builder = importlib.import_module("build_fars_county_day")
    population = pd.DataFrame({
        "fips": ["01001", "09110", "09120"],
        "year": [2021, 2021, 2021],
        "population": [10, 10, 10],
    })

    permitted = builder.permitted_fips_for_year(2021, population=population)

    assert {"09001", "09003", "09005", "09007", "09009", "09011", "09013", "09015"}.issubset(permitted)
    assert {"09110", "09120"}.issubset(permitted)


def test_build_fars_year_retains_legacy_connecticut_event_with_planning_region_population(monkeypatch):
    builder = importlib.import_module("build_fars_county_day")
    population = pd.DataFrame({
        "fips": ["01001", "09110"], "year": [2022, 2022], "population": [10, 10],
    })
    permitted = builder.permitted_fips_for_year(2022, population=population)
    accidents = pd.DataFrame({
        "ST_CASE": [1], "STATE": [9], "COUNTY": [1], "YEAR": [2022],
        "MONTH": [1], "DAY": [3], "FATALS": [1], "WEATHER": [1],
    })
    monkeypatch.setattr(builder, "fetch_zip", lambda year, session: (object(), "b" * 64))
    monkeypatch.setattr(builder, "permitted_fips_for_year", lambda year: permitted)
    monkeypatch.setattr(
        builder,
        "read_file",
        lambda _zip, keyword: accidents if keyword == "accident" else _vehicles(),
    )

    events, coverage = builder.build_fars_year(2022, session=object())

    assert events["fips"].tolist() == ["09001"]
    assert coverage.coverage_valid is True


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


def test_build_fars_records_a_download_failure_in_its_manifest(monkeypatch):
    builder = importlib.import_module("build_fars_county_day")
    monkeypatch.setattr(builder, "YEARS", (2024,))

    def fail_download(year, session):
        raise RuntimeError("offline source")

    monkeypatch.setattr(builder, "build_fars_year", fail_download)

    with pytest.raises(builder.FarsValidationError) as exc_info:
        builder.build_fars(years=[2024])

    failed = exc_info.value.manifest.loc[lambda frame: frame["source"].eq(builder.FARS_SOURCE)].iloc[0]
    assert failed["year"] == 2024
    assert bool(failed["request_complete"]) is False
    assert bool(failed["coverage_valid"]) is False
    assert "terminal_page_error" in failed["failure_reasons"]
