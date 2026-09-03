import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _pilot_module():
    try:
        return importlib.import_module("run_route_exposure_pilot")
    except ModuleNotFoundError:
        pytest.fail("run_route_exposure_pilot module does not exist")


def _county_segments_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["r1", "r1", "r2"],
            "home_fips": ["55001", "55001", "55003"],
            "work_fips": ["55003", "55003", "55003"],
            "outcome_fips": ["55001", "55003", "55003"],
            "workers": [100.0, 100.0, 80.0],
            "home_car_share": [0.8, 0.8, 0.6],
            "route_miles_in_county": [10.0, 5.0, 12.0],
            "route_miles_total": [15.0, 15.0, 12.0],
            "unallocated_miles": [0.0, 0.0, 0.0],
            "segment_type": ["county", "county", "county"],
            "same_tract_imputed_miles": [0.0, 0.0, 0.0],
            "same_tract_mode": ["primary_calibrated", "primary_calibrated", "primary_calibrated"],
            "network_manifest_id": ["net-2022", "net-2022", "net-2022"],
        }
    )


def _alerts_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_fips": ["55001"],
            "alert_date": [pd.Timestamp("2022-01-02")],
            "geo_scope": ["county_same"],
        }
    )


def test_alert_date_exposure_splits_own_cross_and_passthrough():
    pilot = _pilot_module()
    assert hasattr(pilot, "build_alert_date_exposures")

    out = pilot.build_alert_date_exposures(_county_segments_fixture(), _alerts_fixture(), "primary_calibrated")
    row = out.query("outcome_fips == '55003' and alert_date == '2022-01-02'").iloc[0]

    assert row["cross_affected_car_miles"] > 0
    assert row["affected_route_share"] == pytest.approx(
        row["affected_commuter_car_miles"] / row["total_commuter_car_miles"]
    )
    assert row["same_tract_mode"] == "primary_calibrated"
    assert row["network_manifest_id"] == "net-2022"
    assert row["route_exposure_2022"] == 1


def test_zero_denominator_is_rejected():
    pilot = _pilot_module()
    with pytest.raises(ValueError, match="zero denominator"):
        pilot.build_alert_date_exposures(
            _county_segments_fixture().assign(route_miles_in_county=0.0),
            _alerts_fixture(),
            "zero",
        )


def test_unknown_alert_scope_is_rejected():
    pilot = _pilot_module()
    with pytest.raises(ValueError, match="unknown geo_scope"):
        pilot.build_alert_date_exposures(_county_segments_fixture(), _alerts_fixture().assign(geo_scope="mystery"), "zero")


def test_statewide_scope_affects_only_counties_expanded_by_reviewed_loader():
    pilot = _pilot_module()
    segments = _county_segments_fixture()
    alerts = pd.DataFrame(
        {
            "home_fips": ["55001"],
            "alert_date": [pd.Timestamp("2022-01-02")],
            "geo_scope": ["statewide_same"],
        }
    )

    out = pilot.build_alert_date_exposures(segments, alerts, "primary_calibrated")
    county = out.loc[out["outcome_fips"].eq("55003")].iloc[0]

    assert county["affected_commuter_car_miles"] == pytest.approx(100 * 0.8 * 5)
    assert county["affected_commuter_car_miles"] < county["total_commuter_car_miles"]


def test_failed_and_unallocated_rows_with_missing_outcome_are_auditable_but_not_dosage():
    pilot = _pilot_module()
    segments = pd.concat(
        [
            _county_segments_fixture(),
            pd.DataFrame(
                {
                    "route_id": ["r3", "r4"],
                    "home_fips": ["55001", "55001"],
                    "work_fips": ["55003", "55003"],
                    "outcome_fips": [pd.NA, pd.NA],
                    "workers": [3.0, 2.0],
                    "home_car_share": [1.0, 1.0],
                    "route_miles_in_county": [0.0, 0.0],
                    "route_miles_total": [0.0, 10.0],
                    "unallocated_miles": [0.0, 10.0],
                    "segment_type": ["failed_route", "unallocated"],
                    "same_tract_imputed_miles": [0.0, 0.0],
                    "same_tract_mode": ["primary_calibrated", "primary_calibrated"],
                    "network_manifest_id": ["net-2022", "net-2022"],
                }
            ),
        ],
        ignore_index=True,
    )

    out = pilot.build_alert_date_exposures(segments, _alerts_fixture(), "primary_calibrated")

    assert out["failed_route_commuter_car_weight"].eq(3.0).all()
    assert out["unallocated_commuter_car_weight"].eq(2.0).all()
    assert out["affected_commuter_car_miles"].sum() > 0


def test_missing_car_share_failure_preserves_explicit_omitted_worker_weight():
    pilot = _pilot_module()
    segments = _county_segments_fixture().copy()
    segments["omitted_car_share_worker_weight"] = 0.0
    segments["home_car_share"] = segments["home_car_share"].astype(object)
    segments.loc[len(segments)] = {
        "route_id": "missing-car",
        "home_fips": "55001",
        "work_fips": "55003",
        "outcome_fips": pd.NA,
        "workers": 50.0,
        "home_car_share": pd.NA,
        "route_miles_in_county": 0.0,
        "route_miles_total": 0.0,
        "unallocated_miles": 0.0,
        "segment_type": "failed_route",
        "same_tract_imputed_miles": 0.0,
        "same_tract_mode": "primary_calibrated",
        "network_manifest_id": "net-2022",
        "omitted_car_share_worker_weight": 50.0,
    }

    out = pilot.build_alert_date_exposures(
        segments,
        _alerts_fixture(),
        "primary_calibrated",
    )

    assert out["omitted_car_share_worker_weight"].eq(50.0).all()
    assert out["failed_route_commuter_car_weight"].eq(0.0).all()


def test_diagnostics_require_status_and_report_omitted_weight():
    pilot = _pilot_module()
    pairs = pd.DataFrame({"route_id": ["r1", "r2"], "workers": [100.0, 80.0], "home_car_share": [0.8, 0.6], "commuter_car_miles": [800.0, 576.0]})
    routes = pd.DataFrame({"route_id": ["r1", "r2"], "status": ["ok", "failed_route"], "workers": [100.0, 80.0], "home_car_share": [0.8, 0.6], "commuter_car_miles": [800.0, 576.0], "route_to_straight_ratio": [1.2, 1.1]})
    out = pilot.build_route_pilot_diagnostics(pairs, routes, _county_segments_fixture())
    coverage = out["coverage"].set_index("metric")["value"]
    route_diag = out["route"].set_index("metric")["value"]
    assert coverage["failed_route_worker_weight"] == 80
    assert coverage["omitted_external_endpoint_worker_weight"] == 0
    assert coverage["selected_commuter_car_weight"] == pytest.approx(128)
    assert coverage["successful_commuter_car_share"] == pytest.approx(80 / 128)
    assert route_diag["failed_route_count"] == 1


def test_compare_destination_and_route_exposure_returns_finite_summary():
    pilot = _pilot_module()
    assert hasattr(pilot, "compare_destination_and_route_exposure")

    route = pd.DataFrame(
        {
            "outcome_fips": ["55003", "55001"],
            "alert_date": [pd.Timestamp("2022-01-02"), pd.Timestamp("2022-01-02")],
            "affected_route_share": [0.25, 0.50],
            "affected_commuter_car_miles": [50.0, 100.0],
            "total_commuter_car_miles": [200.0, 200.0],
            "own_affected_car_miles": [20.0, 80.0],
            "cross_affected_car_miles": [30.0, 20.0],
            "pass_through_affected_car_miles": [0.0, 0.0],
            "commuter_car_miles": [60.0, 70.0],
            "destination_dosage": [1.0, 2.0],
            "simple_commuter_share": [0.10, 0.20],
            "straight_line_allocation": [0.12, 0.18],
            "current_county_denominator": [200.0, 200.0],
        }
    )
    existing = pd.DataFrame(
        {
            "outcome_fips": ["55003", "55001"],
            "alert_date": [pd.Timestamp("2022-01-02"), pd.Timestamp("2022-01-02")],
            "commuter_car_miles": [60.0, 70.0],
        }
    )

    out = pilot.compare_destination_and_route_exposure(route, existing)

    assert {"metric", "route_exposure_2022"}.issubset(out.columns)
    assert out["route_exposure_2022"].eq(1).all()
    assert out["metric"].notna().all()
    assert out.select_dtypes(include="number").apply(pd.Series.notna).all().all()


def test_write_pilot_report_creates_markdown_and_csvs(tmp_path):
    pilot = _pilot_module()
    assert hasattr(pilot, "write_pilot_report")

    diagnostics = {
        "coverage": pd.DataFrame({"metric": ["selected_worker_weight"], "value": [123.0]}),
        "comparisons": pd.DataFrame({"metric": ["affected_route_share"], "value": [0.25]}),
    }
    path = tmp_path / "route_exposure_pilot.md"

    pilot.write_pilot_report(diagnostics, path)

    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "route_exposure_2022" in text
    assert "same_tract_mode" in text
    assert "network_manifest_id" in text
    assert "source URLs" in text


def test_load_reviewed_route_alerts_preserves_expanded_scope_and_effective_date():
    pilot = _pilot_module()

    def fake_loader(**kwargs):
        assert kwargs == {"window": "night", "detail": True}
        return pd.DataFrame(
            {
                "fips": ["55001", "55003"],
                "effective_crash_date": ["2022-01-03", "2022-01-03"],
                "geo_scope": ["statewide_same", "statewide_same"],
                "original_fips": ["55000", "55000"],
            }
        )

    out = pilot.load_reviewed_route_alerts(loader=fake_loader)

    assert out[["home_fips", "geo_scope", "original_fips"]].to_dict("records") == [
        {"home_fips": "55001", "geo_scope": "statewide_same", "original_fips": "55000"},
        {"home_fips": "55003", "geo_scope": "statewide_same", "original_fips": "55000"},
    ]
    assert out["alert_date"].eq(pd.Timestamp("2022-01-03")).all()


def test_gate_requires_every_prespecified_criterion():
    pilot = _pilot_module()
    evidence = {
        "routing_coverage": 0.995,
        "row_conservation_pass": True,
        "aggregate_conservation_gap": 0.0005,
        "tract_aggregation_bias_acceptable": True,
        "same_tract_dominance_acceptable": True,
        "same_tract_sign_stable": True,
        "denominator_stable": True,
        "route_destination_material": True,
        "computationally_feasible": True,
    }

    accepted, table = pilot.evaluate_pilot_gate(evidence)
    rejected, rejected_table = pilot.evaluate_pilot_gate(
        {**evidence, "tract_aggregation_bias_acceptable": False}
    )

    assert accepted is True
    assert table["passed"].all()
    assert rejected is False
    assert not rejected_table.loc[
        rejected_table["criterion"].eq("tract_aggregation_bias"), "passed"
    ].iloc[0]

    false_string, _ = pilot.evaluate_pilot_gate(
        {**evidence, "same_tract_sign_stable": "false"}
    )
    assert false_string is False


def test_default_output_paths_use_tables_directory_and_mode_suffixes(tmp_path):
    pilot = _pilot_module()
    primary = pilot.default_output_paths(tmp_path, 2022, "primary_calibrated")
    zero = pilot.default_output_paths(tmp_path, 2022, "zero")

    assert all(path.parent == tmp_path / "output" / "tables" for path in primary["tables"].values())
    assert all(path.name.startswith("route_pilot_") for path in primary["tables"].values())
    assert primary["tables"]["input"].name == "route_pilot_input_diagnostics.csv"
    assert primary["tables"]["route"].name == "route_pilot_route_diagnostics.csv"
    assert primary["tables"]["county_exposure"].name == "route_pilot_county_exposure_summary.csv"
    assert primary["tables"]["comparison"].name == "route_pilot_exposure_comparison.csv"
    assert primary["report"].name == "ROUTE_EXPOSURE_PILOT_REPORT.md"
    assert "_zero" in zero["tables"]["input"].stem
    assert primary["report"] != zero["report"]
    assert set(primary["tables"].values()).isdisjoint(set(zero["tables"].values()))
