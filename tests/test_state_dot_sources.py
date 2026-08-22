import ast
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from state_dot_sources import (  # noqa: E402
    STATE_SOURCE_SPECS,
    validate_source_frame,
    fetch_failure_diagnostics,
    validate_state_year,
    validate_wisconsin_county_year,
    filter_to_requested_years,
)


CODE_DIR = Path(__file__).resolve().parents[1] / "code"


def _validator_calls(path: str) -> list[ast.Call]:
    """Inspect production module call sites without running network builders."""
    tree = ast.parse((CODE_DIR / path).read_text())
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_source_frame"]


def _keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def _valid_diagnostics():
    return dict(
        expected_records=2,
        fetched_records=2,
        retained_records=2,
        request_complete=True,
        required_columns_ok=True,
        observed_min_date="2024-01-01",
        observed_max_date="2024-12-31",
    )


def test_all_twenty_state_specs_use_explicit_geography_and_outcome_concepts():
    assert set(STATE_SOURCE_SPECS) == {"CA", "FL", "IL", "IA", "MA", "NV", "NY", "OR", "TN", "TX", "VA", "WI", "DE", "NC", "UT", "CT", "MOCO", "HI", "INMPO", "IDCOMPASS"}
    for spec in STATE_SOURCE_SPECS.values():
        assert spec.expected_county_fips
        assert all(fips.startswith(spec.state_fips) and len(fips) == 5 for fips in spec.expected_county_fips)
        assert spec.native_outcomes
        assert spec.query_identifier


def test_florida_2019_is_explicitly_rejected_even_when_fetch_is_complete():
    result = validate_state_year("FL", 2019, **_valid_diagnostics())
    assert not result.coverage_valid
    assert "excluded_source_year" in result.failure_reasons


def test_tennessee_2025_is_not_a_requested_closed_year():
    result = validate_state_year("TN", 2025, **_valid_diagnostics())
    assert not result.coverage_valid
    assert "unrequested_year" in result.failure_reasons


def test_texas_years_are_validated_as_independent_reporting_units():
    first = validate_state_year("TX", 2020, **_valid_diagnostics())
    second = validate_state_year("TX", 2024, **_valid_diagnostics())
    assert first.coverage_valid and second.coverage_valid
    assert first.year != second.year
    assert first.source == second.source == "TX_TXDOT_CRIS"


def test_wisconsin_failed_request_is_not_a_genuine_empty_county_year():
    failed = validate_wisconsin_county_year(
        "55001", 2024, response_kind="failed", terminal_error="timeout"
    )
    empty = validate_wisconsin_county_year(
        "55001", 2024, response_kind="empty", request_complete=True
    )
    assert not failed.coverage_valid
    assert failed.fetched_records == 0
    assert "terminal_page_error" in failed.failure_reasons
    assert empty.coverage_valid
    assert empty.expected_records == empty.fetched_records == empty.retained_records == 0


def test_new_york_person_fatalities_and_serious_injuries_are_unavailable():
    spec = STATE_SOURCE_SPECS["NY"]
    assert spec.outcome_availability["person_fatals"] is False
    assert spec.outcome_availability["serious_injury_persons"] is False
    assert "person_fatals" not in spec.comparable_outcomes


@pytest.mark.parametrize("state", ["CA", "MA", "NV", "VA", "WI"])
def test_unverified_serious_injury_proxies_are_not_comparable(state):
    spec = STATE_SOURCE_SPECS[state]
    assert spec.outcome_availability["serious_injury_persons"] is False
    assert "serious_injury_persons" not in spec.comparable_outcomes
    assert "proxy" in spec.native_outcomes["serious_injury_persons"].lower()


def test_validator_uses_fetch_diagnostics_and_rejects_unknown_geography():
    result = validate_state_year(
        "TX", 2024, invalid_geography_count=1, **_valid_diagnostics()
    )
    assert not result.coverage_valid
    assert "invalid_geography" in result.failure_reasons


def test_raw_source_frame_failure_and_negative_native_outcome_are_manifest_invalid():
    failed = validate_source_frame(
        "CA", 2024, None, required_columns={"CRASH DATE TIME"},
        date_column="CRASH DATE TIME", outcome_columns={"NUMBERKILLED"},
        terminal_error="short_page",
    )
    negative = validate_source_frame(
        "CA", 2024,
        __import__("pandas").DataFrame({
            "CRASH DATE TIME": ["2024-01-02"], "NUMBERKILLED": [-1],
            "COUNTY CODE": [1],
        }),
        required_columns={"CRASH DATE TIME", "NUMBERKILLED", "COUNTY CODE"},
        date_column="CRASH DATE TIME", outcome_columns={"NUMBERKILLED"},
    )
    assert not failed.coverage_valid and "terminal_page_error" in failed.failure_reasons
    assert not negative.coverage_valid and "negative_outcomes" in negative.failure_reasons


def test_illinois_supported_raw_aliases_are_canonicalized_but_missing_native_fields_fail():
    raw = pd.DataFrame({
        "Crash Date": ["2024-01-02"], "County Code": [1],
        "Total Fatals": [0], "Incapacitating Injuries": [1],
    })
    valid = validate_source_frame("IL", 2024, raw,
        required_columns={"CRASH_DATE", "COUNTY_CODE", "TOTALFATALS", "AINJURIES"},
        date_column="CRASH_DATE", outcome_columns={"TOTALFATALS", "AINJURIES"},
        column_aliases={"CRASH_DATE": ("CrashDate", "Crash Date"),
                        "COUNTY_CODE": ("CountyCode", "County Code"),
                        "TOTALFATALS": ("TotalFatals", "Total Fatals"),
                        "AINJURIES": ("AInjuries", "Incapacitating Injuries")})
    missing = validate_source_frame("IL", 2024, raw.drop(columns="Total Fatals"),
        required_columns={"CRASH_DATE", "COUNTY_CODE", "TOTALFATALS", "AINJURIES"},
        date_column="CRASH_DATE", outcome_columns={"TOTALFATALS", "AINJURIES"},
        column_aliases={"CRASH_DATE": ("CrashDate", "Crash Date"),
                        "COUNTY_CODE": ("CountyCode", "County Code"),
                        "TOTALFATALS": ("TotalFatals", "Total Fatals"),
                        "AINJURIES": ("AInjuries", "Incapacitating Injuries")})
    assert valid.coverage_valid
    assert not missing.coverage_valid and "missing_required_columns" in missing.failure_reasons


def test_unmapped_raw_county_invalidates_coverage_before_balancing():
    raw = pd.DataFrame({"date": ["2024-01-02"], "county": ["UNKNOWN"], "death_cnt": [0]})
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert not result.coverage_valid
    assert "invalid_geography" in result.failure_reasons


def test_unmapped_raw_county_excluded_without_failing_when_declared_unresolvable():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"], "county": ["KNOWN", "UNKNOWN"],
        "death_cnt": [0, 0],
    })
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"},
        unresolvable_geography_values=frozenset({"UNKNOWN"}))
    assert result.coverage_valid
    assert result.invalid_geography_count == 0
    assert result.unresolvable_geography_count == 1
    assert result.retained_records == 1


def test_null_source_date_excluded_without_failing_coverage():
    raw = pd.DataFrame({
        "date": ["2024-01-02", None], "county": ["KNOWN", "KNOWN"], "death_cnt": [0, 0],
    })
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert result.coverage_valid
    assert result.invalid_date_count == 0
    assert result.unresolvable_date_count == 1
    assert result.retained_records == 1


def test_small_out_of_year_date_boundary_excluded_without_failing_coverage():
    # A valid, parseable date one calendar year off the requested year (the
    # boundary artifact seen with Delaware's Socrata `year` query field) is
    # excluded as a small bounded residual, not a hard failure.
    raw = pd.DataFrame({
        "date": ["2024-01-02"] * 199 + ["2023-12-31"],
        "county": ["KNOWN"] * 200, "death_cnt": [0] * 200,
    })
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert result.coverage_valid
    assert result.invalid_date_count == 0
    assert result.unresolvable_date_count == 1
    assert result.retained_records == 199


def test_large_out_of_year_date_share_remains_a_hard_failure():
    raw = pd.DataFrame({
        "date": ["2023-12-31"], "county": ["KNOWN"], "death_cnt": [0],
    })
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert not result.coverage_valid
    assert "retained_count_mismatch" in result.failure_reasons
    assert result.unresolvable_date_count == 0


def test_malformed_nonnull_date_remains_a_hard_failure():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "not-a-date"], "county": ["KNOWN", "KNOWN"], "death_cnt": [0, 0],
    })
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert not result.coverage_valid
    assert result.invalid_date_count == 1
    assert result.unresolvable_date_count == 0
    assert "invalid_dates" in result.failure_reasons


def test_small_negative_outcome_residual_excluded_without_failing_coverage():
    raw = pd.DataFrame({
        "date": ["2024-01-02"] * 200,
        "county": ["KNOWN"] * 200,
        "death_cnt": [0] * 199 + [-1],
    })
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert result.coverage_valid
    assert result.unresolvable_outcome_count == 1
    assert result.retained_records == 199


def test_large_negative_outcome_share_remains_a_hard_failure():
    raw = pd.DataFrame({"date": ["2024-01-02"], "county": ["KNOWN"], "death_cnt": [-1]})
    result = validate_source_frame("TX", 2024, raw,
        required_columns={"date", "county", "death_cnt"}, date_column="date",
        outcome_columns={"death_cnt"}, geography_column="county",
        geography_mapper={"KNOWN": "48001"})
    assert not result.coverage_valid
    assert "negative_outcomes" in result.failure_reasons
    assert result.unresolvable_outcome_count == 0


def test_strict_fetch_failure_diagnostics_preserve_count_and_terminal_error():
    from crash_download import IncompleteDownloadError
    error = IncompleteDownloadError("short page", expected_count=10, fetched_count=4,
                                   terminal_error="empty_page")
    assert fetch_failure_diagnostics(error) == {
        "expected_records": 10, "fetched_records": 4, "retained_records": 0,
        "request_complete": False, "terminal_error": "empty_page",
    }


def test_production_adapters_pass_raw_geography_mappers_before_aggregation():
    # These are the actual script call sites.  Static inspection avoids running
    # networked, top-level builders while protecting the integration boundary.
    adapters = {
        "build_california_ccrs.py": "CA",
        "build_florida_fdot.py": "FL",
        "build_illinois_idot.py": "IL",
        "build_iowa_dot.py": "IA",
        "build_massachusetts_massdot.py": "MA",
        "build_nevada_ndot.py": "NV",
        "build_newyork_dot.py": "NY",
        "build_oregon_odot.py": "OR",
        "build_tennessee_tdot.py": "TN",
        "build_texas_txdot.py": "TX",
        "build_virginia_vdot.py": "VA",
        "extend_florida_fdot.py": "FL",
        "extend_texas_txdot.py": "TX",
    }
    for filename, state in adapters.items():
        calls = [call for call in _validator_calls(filename)
                 if call.args and isinstance(call.args[0], ast.Constant)
                 and call.args[0].value == state]
        assert any({"geography_column", "geography_mapper"}.issubset(_keyword_names(call))
                   for call in calls), filename


def test_production_illinois_validation_uses_explicit_schema_aliases_and_mapper():
    calls = _validator_calls("build_illinois_idot.py")
    production = next(call for call in calls if call.args and isinstance(call.args[0], ast.Constant)
                      and call.args[0].value == "IL")
    assert {"column_aliases", "geography_column", "geography_mapper"}.issubset(
        _keyword_names(production)
    )
    source = (CODE_DIR / "build_illinois_idot.py").read_text()
    for supported_name in ("CrashDate", "Crash Date", "CrashDateTime", "CrashYr",
                           "Crash Year", "CrashMonth", "Crash Month", "CrashDay",
                           "Crash Day", "CountyCode", "County Code", "TotalFatals",
                           "Total Fatals", "AInjuries", "Incapacitating Injuries"):
        assert supported_name in source


def test_strict_fetch_main_builders_keep_terminal_error_at_manifest_callsite():
    for filename, state in {
        "build_florida_fdot.py": "FL", "build_illinois_idot.py": "IL",
        "build_iowa_dot.py": "IA", "build_massachusetts_massdot.py": "MA",
        "build_nevada_ndot.py": "NV", "build_newyork_dot.py": "NY",
        "build_oregon_odot.py": "OR", "build_tennessee_tdot.py": "TN",
        "build_texas_txdot.py": "TX", "build_virginia_vdot.py": "VA",
    }.items():
        source = (CODE_DIR / filename).read_text()
        assert "FETCH_FAILURES" in source, filename
        calls = [call for call in _validator_calls(filename)
                 if call.args and isinstance(call.args[0], ast.Constant)
                 and call.args[0].value == state]
        assert any("terminal_error" in _keyword_names(call) for call in calls), filename


def test_iowa_processing_contract_excludes_2013_and_2014_before_output():
    raw = pd.DataFrame({"crash_date": pd.to_datetime([
        "2013-01-01", "2014-12-31", "2015-01-01", "2024-12-31", "2025-01-01"
    ])})
    retained = filter_to_requested_years(raw, state="IA", date_column="crash_date")
    assert retained["crash_date"].dt.year.tolist() == [2015, 2024]
    source = (CODE_DIR / "build_iowa_dot.py").read_text()
    assert 'filter_to_requested_years(df, state="IA", date_column="crash_date")' in source
