import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from crash_coverage import (  # noqa: E402
    balance_validated_panel,
    validate_reporting_unit,
    write_manifest,
)


def _unit(**overrides):
    values = {
        "source": "NV_NDOT",
        "state": "NV",
        "year": 2024,
        "expected_records": 2,
        "fetched_records": 2,
        "retained_records": 2,
        "request_complete": True,
        "terminal_error": None,
        "invalid_date_count": 0,
        "invalid_geography_count": 0,
        "required_columns_ok": True,
        "observed_min_date": "2024-01-01",
        "observed_max_date": "2024-12-31",
    }
    values.update(overrides)
    return validate_reporting_unit(**values)


def test_exact_fetched_count_is_valid():
    result = _unit()
    assert result.coverage_valid is True
    assert result.failure_reasons == ()


def test_count_mismatch_invalidates_reporting_unit():
    result = validate_reporting_unit(
        source="NV_NDOT", state="NV", year=2024,
        expected_records=100, fetched_records=99, retained_records=99,
        request_complete=True, terminal_error=None,
        invalid_date_count=0, invalid_geography_count=0,
        required_columns_ok=True, observed_min_date="2024-01-01",
        observed_max_date="2024-12-31",
    )
    assert result.coverage_valid is False
    assert "fetch_count_mismatch" in result.failure_reasons


def test_terminal_page_failure_is_explicit():
    result = _unit(terminal_error="HTTP 503")
    assert result.coverage_valid is False
    assert "terminal_page_error" in result.failure_reasons


def test_invalid_dates_geography_and_columns_accumulate():
    result = _unit(
        invalid_date_count=1,
        invalid_geography_count=2,
        required_columns_ok=False,
    )
    assert result.coverage_valid is False
    assert {
        "invalid_dates",
        "invalid_geography",
        "missing_required_columns",
    }.issubset(result.failure_reasons)


def test_genuine_empty_wisconsin_county_year_is_valid():
    result = validate_reporting_unit(
        source="WI_WISDOT", state="WI", year=2024, county_fips="55001",
        expected_records=0, fetched_records=0, retained_records=0,
        request_complete=True, terminal_error=None,
        invalid_date_count=0, invalid_geography_count=0,
        required_columns_ok=True, observed_min_date=None,
        observed_max_date=None,
    )
    assert result.coverage_valid is True
    assert result.county_fips == "55001"


def test_manifest_is_serializable_and_deterministic(tmp_path):
    later = _unit(source="ZZ", state="CA", year=2024)
    earlier = _unit(source="AA", state="NV", year=2023)
    csv_path, parquet_path = write_manifest(
        [later, earlier], output_dir=tmp_path, filename="coverage"
    )
    assert csv_path == tmp_path / "coverage.csv"
    assert parquet_path == tmp_path / "coverage.parquet"
    frame = pd.read_csv(csv_path)
    assert frame["source"].tolist() == ["AA", "ZZ"]
    assert pd.read_parquet(parquet_path)["source"].tolist() == ["AA", "ZZ"]


def test_manifest_normalizes_missing_failure_reasons(tmp_path):
    manifest = pd.DataFrame([_unit().to_mapping(), _unit(source="ZZ").to_mapping()])
    manifest["failure_reasons"] = [None, float("nan")]
    csv_path, _ = write_manifest(manifest, output_dir=tmp_path, filename="missing")
    assert pd.read_csv(csv_path, keep_default_na=False)["failure_reasons"].tolist() == ["", ""]


def test_balance_fills_only_valid_units_and_available_outcomes():
    sparse = pd.DataFrame({
        "fips": ["01001"],
        "date": ["2024-01-01"],
        "crashes": [1],
        "person_fatals": [7],
    })
    manifest = pd.DataFrame([{
        **_unit(source="AL_DOT", state="AL").to_mapping(),
    }, {
        **_unit(source="AL_DOT", state="AL", year=2023).to_mapping(),
        "coverage_valid": False,
        "failure_reasons": ("terminal_page_error",),
    }])
    universe = pd.DataFrame({"fips": ["01001", "01003"], "state": ["AL", "AL"]})
    balanced = balance_validated_panel(
        sparse=sparse,
        manifest=manifest,
        county_universe=universe,
        outcome_availability={"crashes": True, "person_fatals": False},
        reporting_unit="state_year",
    )
    balanced["date"] = pd.to_datetime(balanced["date"])
    row = balanced.loc[
        (balanced.fips == "01003") &
        (balanced.date == pd.Timestamp("2024-01-02")),
    ].iloc[0]
    assert row["crashes"] == 0
    assert balanced["person_fatals"].isna().all()
    invalid_unit_dates = pd.date_range("2023-01-01", "2023-12-31")
    assert not invalid_unit_dates.isin(balanced["date"]).any()
    assert bool(row["coverage_valid"]) is True
    assert row["coverage_unit"] == "state_year"
    assert bool(row["structural_zero"]) is True
    assert balanced["source"].eq("AL_DOT").all()


def test_balance_excludes_events_from_invalid_source_sharing_county_date():
    sparse = pd.DataFrame({
        "source": ["VALID_DOT", "INVALID_DOT"],
        "fips": ["01001", "01001"],
        "date": ["2024-01-01", "2024-01-01"],
        "crashes": [1, 99],
    })
    valid = _unit(source="VALID_DOT", state="AL")
    invalid = _unit(source="INVALID_DOT", state="AL")
    manifest = pd.DataFrame([
        valid.to_mapping(),
        {**invalid.to_mapping(), "coverage_valid": False,
         "failure_reasons": ("terminal_page_error",)},
    ])
    balanced = balance_validated_panel(
        sparse=sparse,
        manifest=manifest,
        county_universe=pd.DataFrame({"fips": ["01001"], "state": ["AL"]}),
        outcome_availability={"crashes": True},
        reporting_unit="state_year",
    )
    row = balanced.loc[balanced["date"].eq(pd.Timestamp("2024-01-01"))].iloc[0]
    assert row["source"] == "VALID_DOT"
    assert row["crashes"] == 1
