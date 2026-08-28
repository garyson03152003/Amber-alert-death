import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _manifest(state: str, year: int, valid: bool, *, source: str) -> dict[str, object]:
    return {
        "source": source,
        "state": state,
        "year": year,
        "county_fips": None,
        "expected_records": 1,
        "fetched_records": 1,
        "retained_records": 1,
        "duplicate_records": 0,
        "invalid_date_count": 0,
        "invalid_geography_count": 0 if valid else 1,
        "observed_min_date": f"{year}-01-02",
        "observed_max_date": f"{year}-01-02",
        "request_complete": valid,
        "coverage_valid": valid,
        "failure_reasons": "" if valid else "invalid_geography",
        "source_url": "https://example.test/source",
        "source_checksum": "a" * 64,
    }


def test_validated_panel_balances_only_allowlisted_coverage_and_keeps_ny_unavailable_outcomes_missing(tmp_path):
    from build_validated_crash_panels import build_validated_state_panel
    from state_dot_sources import STATE_SOURCE_SPECS

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({
        "state": ["NY"], "year": [2024], "reason": ["fixture review"],
    }).to_csv(allowlist, index=False)
    sparse = pd.DataFrame({
        "fips": ["36001", "36001"],
        "date": ["2024-01-02", "2023-01-02"],
        "crashes": [1, 99],
        "person_fatals": [4, 99],
        "serious_injury_persons": [5, 99],
        "source": ["NY_DOT", "NY_DOT"],
    })
    manifest = pd.DataFrame([
        _manifest("NY", 2024, True, source="NY_DOT"),
        _manifest("NY", 2023, False, source="NY_DOT"),
    ])
    universe = pd.DataFrame({
        "fips": sorted(STATE_SOURCE_SPECS["NY"].expected_county_fips),
        "year": 2024,
    })

    panel, accepted = build_validated_state_panel(
        "NY", sparse, manifest, universe, years=[2024], allowlist_path=allowlist,
    )

    assert set(panel["year"]) == {2024}
    assert len(panel) == len(universe) * 366
    zero = panel.loc[(panel["fips"] == "36003") & (panel["date"] == pd.Timestamp("2024-01-02"))]
    assert zero["crashes"].item() == 0
    assert bool(zero["structural_zero"].item()) is True
    assert panel["person_fatals"].isna().all()
    assert panel["serious_injury_persons"].isna().all()
    assert accepted.to_dict("records") == [{
        "state": "NY", "year": 2024, "review_status": "accepted", "reason": "fixture review",
    }]


def test_validated_panel_rejects_fars_manifest_row_for_tx_even_when_allowlisted(tmp_path):
    from build_validated_crash_panels import InvalidCoverageManifestError, build_validated_state_panel

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["TX"], "year": [2024]}).to_csv(allowlist, index=False)

    with pytest.raises(InvalidCoverageManifestError, match="expected TX_TXDOT_CRIS"):
        build_validated_state_panel(
            "TX",
            pd.DataFrame({"fips": ["48001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([_manifest("TX", 2024, True, source="FARS_NHTSA")]),
            pd.DataFrame({"fips": ["48001"], "year": [2024]}),
            years=[2024], allowlist_path=allowlist,
        )


def test_validated_panel_requires_every_nonexcluded_contract_year_by_default(tmp_path):
    from build_validated_crash_panels import MissingCoverageManifestError, build_validated_state_panel

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["TX"], "year": [2024]}).to_csv(allowlist, index=False)

    with pytest.raises(MissingCoverageManifestError, match="TX 2020"):
        build_validated_state_panel(
            "TX",
            pd.DataFrame({"fips": ["48001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([_manifest("TX", 2024, True, source="TX_TXDOT_CRIS")]),
            pd.DataFrame({"fips": ["48001"], "year": [2024]}),
            allowlist_path=allowlist,
        )


def test_validated_panel_rejects_incomplete_tx_county_universe_before_balancing(tmp_path):
    from build_validated_crash_panels import InvalidCountyUniverseError, build_validated_state_panel

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["TX"], "year": [2024]}).to_csv(allowlist, index=False)

    with pytest.raises(InvalidCountyUniverseError, match="missing"):
        build_validated_state_panel(
            "TX",
            pd.DataFrame({"fips": ["48001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([_manifest("TX", 2024, True, source="TX_TXDOT_CRIS")]),
            pd.DataFrame({"fips": ["48001"], "year": [2024]}),
            years=[2024], allowlist_path=allowlist,
        )


def test_validated_panel_rejects_duplicate_ny_state_year_manifest_rows(tmp_path):
    from build_validated_crash_panels import DuplicateCoverageManifestError, build_validated_state_panel
    from state_dot_sources import STATE_SOURCE_SPECS

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["NY"], "year": [2024]}).to_csv(allowlist, index=False)
    duplicate = _manifest("NY", 2024, True, source="NY_DOT")

    with pytest.raises(DuplicateCoverageManifestError, match="NY 2024"):
        build_validated_state_panel(
            "NY",
            pd.DataFrame({"fips": ["36001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([duplicate, duplicate]),
            pd.DataFrame({"fips": sorted(STATE_SOURCE_SPECS["NY"].expected_county_fips), "year": 2024}),
            years=[2024], allowlist_path=allowlist,
        )


def test_validated_panel_rejects_duplicate_wisconsin_county_year_manifest_rows(tmp_path):
    from build_validated_crash_panels import DuplicateCoverageManifestError, build_validated_state_panel

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["WI"], "year": [2024]}).to_csv(allowlist, index=False)
    duplicate = _manifest("WI", 2024, True, source="WI_COMMUNITY_MAPS")
    duplicate["county_fips"] = "55001"

    with pytest.raises(DuplicateCoverageManifestError, match="55001"):
        build_validated_state_panel(
            "WI",
            pd.DataFrame({"fips": ["55001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([duplicate, duplicate]),
            pd.DataFrame({"fips": ["55001"], "year": [2024]}),
            years=[2024], allowlist_path=allowlist,
        )


def test_validated_panel_rejects_incomplete_wisconsin_county_manifest_before_balancing(tmp_path):
    from build_validated_crash_panels import InvalidCoverageManifestError, build_validated_state_panel
    from state_dot_sources import STATE_SOURCE_SPECS

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["WI"], "year": [2024]}).to_csv(allowlist, index=False)
    incomplete = _manifest("WI", 2024, True, source="WI_COMMUNITY_MAPS")
    incomplete["county_fips"] = "55001"

    with pytest.raises(InvalidCoverageManifestError, match="missing 71"):
        build_validated_state_panel(
            "WI",
            pd.DataFrame({"fips": ["55001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([incomplete]),
            pd.DataFrame({"fips": sorted(STATE_SOURCE_SPECS["WI"].expected_county_fips), "year": 2024}),
            years=[2024], allowlist_path=allowlist,
        )


def test_validated_panel_fails_closed_when_requested_manifest_is_absent(tmp_path):
    from build_validated_crash_panels import MissingCoverageManifestError, build_validated_state_panel

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["TX"], "year": [2024]}).to_csv(allowlist, index=False)

    with pytest.raises(MissingCoverageManifestError, match="TX 2024"):
        build_validated_state_panel(
            "TX",
            pd.DataFrame({"fips": ["48001"], "date": ["2024-01-02"], "crashes": [1]}),
            pd.DataFrame([_manifest("TX", 2023, True, source="TX_TXDOT_CRIS")]),
            pd.DataFrame({"fips": ["48001"], "year": [2024]}),
            years=[2024], allowlist_path=allowlist,
        )


def test_fatality_validation_computes_metrics_and_never_accepts_without_allowlist(tmp_path):
    from validate_state_fatalities import validate_state_fatalities

    state_events = pd.DataFrame({
        "fips": ["48001", "48003", "48001"],
        "date": ["2024-01-02", "2024-01-02", "2023-01-02"],
        "person_fatals": [1, 3, 8],
    })
    fars_events = pd.DataFrame({
        "fips": ["48001", "48003", "48001"],
        "date": ["2024-01-02", "2024-01-02", "2023-01-02"],
        "person_fatals": [1, 2, 8],
    })
    manifest = pd.DataFrame([
        _manifest("TX", 2024, True, source="TX_TXDOT_CRIS"),
        _manifest("TX", 2023, False, source="TX_TXDOT_CRIS"),
    ])

    report = validate_state_fatalities(
        state_events, fars_events, manifest, allowlist_path=tmp_path / "missing.csv",
    )

    accepted = report.loc[report["year"].eq(2024)].iloc[0]
    rejected = report.loc[report["year"].eq(2023)].iloc[0]
    assert accepted["dot_person_fatals"] == 4
    assert accepted["fars_person_fatals"] == 3
    assert accepted["dot_fars_ratio"] == pytest.approx(4 / 3)
    assert accepted["county_year_pearson"] == pytest.approx(1.0)
    assert accepted["county_date_agreement"] == pytest.approx(0.5)
    assert accepted["invalid_geography_count"] == 0
    assert accepted["review_status"] == "pending"
    assert rejected["review_status"] == "rejected_coverage"
    assert rejected["invalid_geography_count"] == 1


def test_fatality_validation_accepts_only_explicit_reviewed_state_year(tmp_path):
    from validate_state_fatalities import validate_state_fatalities

    allowlist = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["TX"], "year": [2024], "reason": ["reconciled"]}).to_csv(
        allowlist, index=False
    )
    events = pd.DataFrame({"fips": ["48001"], "date": ["2024-01-02"], "person_fatals": [1]})
    report = validate_state_fatalities(
        events, events, pd.DataFrame([_manifest("TX", 2024, True, source="TX_TXDOT_CRIS")]),
        allowlist_path=allowlist,
    )

    row = report.iloc[0]
    assert row["review_status"] == "accepted"
    assert row["review_reason"] == "reconciled"


def test_fatality_validation_ignores_national_fars_manifest_rows(tmp_path):
    from validate_state_fatalities import validate_state_fatalities

    events = pd.DataFrame({"fips": ["48001"], "date": ["2024-01-02"], "person_fatals": [1]})
    manifest = pd.DataFrame([
        _manifest("TX", 2024, True, source="TX_TXDOT_CRIS"),
        _manifest("US", 2024, True, source="FARS_NHTSA"),
    ])

    report = validate_state_fatalities(events, events, manifest, allowlist_path=tmp_path / "missing.csv")

    assert report[["state", "year"]].to_dict("records") == [{"state": "TX", "year": 2024}]
