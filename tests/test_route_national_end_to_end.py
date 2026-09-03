import json
import hashlib
import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _complete_gate_metrics(**overrides):
    metrics = {
        "successful_commuter_car_share": 0.999,
        "maximum_route_conservation_error": 0.004,
        "aggregate_conservation_error": 0.0005,
        "positive_denominators": True,
        "all_partitions_available": True,
        "no_alerted_origin_bias": True,
        "same_tract_stable": True,
        "omissions_and_failures_explicit": True,
        "route_comparison_complete": True,
        "runtime_seconds": 12.5,
        "storage_bytes": 4096,
        "restart_reused_share": 1.0,
    }
    metrics.update(overrides)
    return metrics


def _production_fixture(tmp_path):
    import run_route_exposure_national as national

    fixture_root = tmp_path / "fixture"
    national.run_synthetic_national_dry_run(
        argparse.Namespace(
            output_dir=fixture_root,
            analysis_years=[2022],
            states=["wi", "il"],
            network_year=2022,
            same_tract_mode="primary_calibrated",
            chunk_rows=1,
            route_workers=1,
            checkpoint_every=2,
            geometry_sample_rate=0.0,
        )
    )
    model_panel = pd.read_parquet(
        fixture_root / "analysis" / "national_route_model_panel.parquet"
    )
    raw_panel_path = tmp_path / "raw_panel.parquet"
    model_panel[["fips", "date", "fatal_crashes"]].to_parquet(
        raw_panel_path, index=False
    )
    destination_path = tmp_path / "destination.parquet"
    model_panel[["fips", "date", "destination_dosage"]].to_parquet(
        destination_path, index=False
    )
    alerts_path = tmp_path / "alerts.csv"
    pd.DataFrame(
        {
            "home_fips": ["17001", "55001"],
            "alert_date": ["2022-06-15", "2022-06-15"],
            "geo_scope": ["county_same", "county_same"],
        }
    ).to_csv(alerts_path, index=False)
    run_metrics_path = tmp_path / "run_metrics.json"
    run_metrics_path.write_text(
        json.dumps({"runtime_seconds": 1.5, "restart_reused_share": 1.0}),
        encoding="utf-8",
    )
    manifest_path = fixture_root / "segments" / "segment_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    args = argparse.Namespace(
        segment_manifest=manifest_path,
        alerts=alerts_path,
        panel=raw_panel_path,
        destination_exposure=destination_path,
        output_dir=fixture_root / "analysis",
        analysis_years=[2022],
        states=["wi", "il"],
        network_year=2022,
        network_manifest=fixture_root / "manifests" / "network_manifest.json",
        run_metrics=run_metrics_path,
        same_tract_results=fixture_root / "analysis" / "same_tract_model_results.csv",
        same_tract_mode="primary_calibrated",
        chunk_rows=250_000,
        route_workers=8,
        checkpoint_every=10_000,
        geometry_sample_rate=0.001,
    )
    validation = national.validate_requested_partitions(
        manifest, analysis_years=[2022], states=["wi", "il"]
    )
    return national, fixture_root, manifest, args, validation


def test_national_gate_rejects_coverage_below_99_percent():
    from run_route_exposure_national import evaluate_national_gates

    result = evaluate_national_gates(
        _complete_gate_metrics(successful_commuter_car_share=0.989)
    )

    assert result["accepted"] is False
    assert "coverage" in result["failed_gates"]


def test_national_gate_accepts_only_complete_valid_metrics():
    from run_route_exposure_national import evaluate_national_gates

    accepted = evaluate_national_gates(_complete_gate_metrics())
    incomplete = evaluate_national_gates(
        {
            key: value
            for key, value in _complete_gate_metrics().items()
            if key != "storage_bytes"
        }
    )

    assert accepted["schema_version"] == "route_national.gates.v1"
    assert accepted["accepted"] is True
    assert {row["gate"] for row in accepted["gates"]} == {
        "coverage",
        "route_conservation",
        "aggregate_conservation",
        "denominators",
        "partition_availability",
        "alerted_origin_bias",
        "same_tract_stability",
        "explicit_accounting",
        "route_destination_comparison",
        "runtime_footprint",
        "storage_footprint",
        "restart_footprint",
    }
    assert incomplete["accepted"] is False
    assert "storage_footprint" in incomplete["failed_gates"]


def test_synthetic_dry_run_exercises_all_stages_and_writes_valid_v1_manifest(
    tmp_path, capsys
):
    from run_route_exposure_national import main

    exit_code = main(
        [
            "--analysis-years",
            "2022",
            "--states",
            "wi",
            "il",
            "--network-year",
            "2022",
            "--same-tract-mode",
            "primary_calibrated",
            "--chunk-rows",
            "1000",
            "--route-workers",
            "1",
            "--checkpoint-every",
            "2",
            "--geometry-sample-rate",
            "1.0",
            "--dry-run-fixture",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert "ACCEPTED" in capsys.readouterr().out
    vintage_manifest = tmp_path / "manifests" / "national_vintage_manifest.csv"
    network_manifest = tmp_path / "manifests" / "network_manifest.json"
    segment_manifest = tmp_path / "segments" / "segment_manifest.csv"
    gate_report = tmp_path / "gates" / "national_gate_report.json"
    gate_table = tmp_path / "gates" / "national_gate_table.csv"
    partition_gate_table = tmp_path / "gates" / "national_partition_gate_table.csv"
    model_panel = tmp_path / "analysis" / "national_route_model_panel.parquet"
    route_comparison = tmp_path / "analysis" / "route_vs_destination_comparison.csv"
    same_tract_summary = tmp_path / "analysis" / "same_tract_mode_summary.csv"
    same_tract_models = tmp_path / "analysis" / "same_tract_model_results.csv"
    for path in (
        vintage_manifest,
        network_manifest,
        segment_manifest,
        gate_report,
        gate_table,
        partition_gate_table,
        model_panel,
        route_comparison,
        same_tract_summary,
        same_tract_models,
    ):
        assert path.is_file(), path

    vintages = pd.read_csv(vintage_manifest)
    assert set(vintages["lodes_source_year"]) == {2021}
    assert set(vintages["lodes_year_gap"]) == {1}
    assert set(vintages["acs_car_share_vintage"]) == {"2018-2022"}
    assert set(vintages["selection_rule"]) == {
        "nearest source year/window; earlier year wins ties"
    }
    network = json.loads(network_manifest.read_text(encoding="utf-8"))
    assert network["network_year"] == 2022
    from build_route_national_network import NATIONAL_STATES
    assert network["scope"] == "national"
    assert network["states"] == sorted(NATIONAL_STATES)
    assert network["manifest_id"].startswith("sha256:")

    manifest = pd.read_csv(segment_manifest)
    assert manifest["schema_version"].eq("route_national.segments.v1").all()
    assert manifest[[
        "analysis_year",
        "lodes_source_year",
        "acs_car_share_vintage",
        "source_manifest_id",
        "network_manifest_id",
        "source_partition_id",
        "segment_sha256",
    ]].notna().all().all()
    for row in manifest.itertuples(index=False):
        path = Path(row.segment_path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.segment_sha256

    report = json.loads(gate_report.read_text(encoding="utf-8"))
    assert report["accepted"] is True
    assert report["failed_gates"] == []
    assert report["metrics"]["restart_reused_share"] == 1.0
    assert report["metrics"]["alerted_origin_comparable"] is True
    assert report["paths"]["segment_manifest"] == str(segment_manifest)
    assert len(report["partition_results"]) == 2
    assert all(result["accepted"] for result in report["partition_results"])
    assert set(pd.read_csv(gate_table)["gate"]) == {
        row["gate"] for row in report["gates"]
    }
    partition_gates = pd.read_csv(partition_gate_table)
    assert partition_gates["source_partition_id"].nunique() == 2
    assert partition_gates.groupby("source_partition_id")["passed"].all().all()
    panel = pd.read_parquet(model_panel)
    assert panel["route_coverage_status"].eq("included_positive_denominator").all()
    assert panel[[
        "own_affected_share",
        "cross_affected_share",
        "pass_through_affected_share",
    ]].notna().all().all()
    assert {
        "affected_route_share",
        "existing_destination_dosage",
        "correlation_affected_route_share_with_existing_destination_dosage",
    }.issubset(set(pd.read_csv(route_comparison)["metric"]))
    assert set(pd.read_csv(same_tract_summary)["same_tract_mode"]) == {
        "primary_calibrated",
        "zero",
        "exclude",
    }
    assert len(list(tmp_path.glob("flows/partitions/**/*.parquet"))) == 2
    assert len(list(tmp_path.glob("segments/**/route_audits.parquet"))) == 2
    assert len(list(tmp_path.glob("segments/**/county_segments.parquet"))) == 2
    assert len(list(tmp_path.glob("segments/**/qa_geometries/*.geojson"))) == 4


def test_synthetic_multi_year_partition_gates_use_only_partition_year_evidence(
    tmp_path,
):
    import run_route_exposure_national as national

    result = national.run_synthetic_national_dry_run(
        argparse.Namespace(
            output_dir=tmp_path,
            analysis_years=[2021, 2022],
            states=["wi"],
            network_year=2022,
            same_tract_mode="primary_calibrated",
            chunk_rows=1,
            route_workers=1,
            checkpoint_every=2,
            geometry_sample_rate=0.0,
        )
    )

    manifest = pd.read_csv(tmp_path / "segments" / "segment_manifest.csv")
    expected_year = dict(
        zip(
            manifest["source_partition_id"].astype(str),
            manifest["analysis_year"].astype(int),
            strict=True,
        )
    )
    assert len(result["partition_results"]) == 2
    for partition in result["partition_results"]:
        partition_id = partition["source_partition_id"]
        assert partition["metrics"]["model_panel_analysis_years"] == [
            expected_year[partition_id]
        ]

    sensitivities = pd.read_csv(
        tmp_path / "analysis" / "same_tract_model_results.csv"
    )
    partition_rows = sensitivities.loc[
        sensitivities["analysis_scope"].astype(str).eq("partition")
    ]
    assert set(partition_rows["source_partition_ids"].astype(str)) == set(
        expected_year
    )
    assert partition_rows.groupby("source_partition_ids")["analysis_years"].nunique().eq(1).all()
    assert {
        int(value)
        for value in partition_rows["analysis_years"].astype(str)
    } == {2021, 2022}


def test_requested_partition_validation_rejects_partial_state_year_grid():
    from run_route_exposure_national import validate_requested_partitions

    validation = validate_requested_partitions(
        pd.DataFrame(
            {
                "analysis_year": [2022],
                "state": ["wi"],
                "status": ["success"],
            }
        ),
        analysis_years=[2022],
        states=["wi", "il"],
    )

    assert validation["complete"] is False
    assert validation["missing_partitions"] == [
        {"analysis_year": 2022, "state": "il"}
    ]


def test_production_cli_rejects_partial_state_manifest_before_modeling(tmp_path):
    from run_route_exposure_national import main

    manifest = tmp_path / "partial.csv"
    pd.DataFrame(
        {
            "schema_version": ["route_national.segments.v1"],
            "analysis_year": [2022],
            "state": ["wi"],
            "lodes_source_year": [2021],
            "acs_car_share_vintage": ["2018-2022"],
            "source_manifest_id": ["source"],
            "network_manifest_id": ["network"],
            "source_partition_id": ["2022__2021__wi"],
            "segment_path": [str(tmp_path / "must-not-be-read.parquet")],
            "status": ["success"],
        }
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "--segment-manifest",
            str(manifest),
            "--panel",
            str(tmp_path / "must-not-be-read-panel.parquet"),
            "--analysis-years",
            "2022",
            "--states",
            "wi",
            "il",
            "--network-year",
            "2022",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    report = json.loads(
        (output_dir / "gates" / "national_gate_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["accepted"] is False
    assert report["partition_validation"]["missing_partitions"] == [
        {"analysis_year": 2022, "state": "il"}
    ]
    assert not (output_dir / "national_route_model_results.csv").exists()


def test_alerted_origin_bias_requires_positive_alerted_and_nonalerted_strata():
    from run_route_exposure_national import _national_gate_metrics

    audits = pd.DataFrame(
        {
            "route_signature": ["r1"],
            "commuter_car_weight": [8.0],
            "routing_eligible": [True],
            "status": ["Ok"],
            "route_miles_total": [2.0],
            "unallocated_miles": [0.0],
            "home_fips": ["55001"],
            "same_tract": [False],
            "omitted_coordinate_worker_weight": [0.0],
            "omitted_car_share_worker_weight": [0.0],
            "error_message": [None],
        }
    )
    segments = pd.DataFrame(
        {
            "route_signature": ["r1"],
            "county_fips": ["55001"],
            "route_miles_in_county": [2.0],
        }
    )
    model_panel = pd.DataFrame(
        {
            "route_coverage_status": ["included_positive_denominator"],
            "total_commuter_car_miles": [16.0],
            "destination_dosage": [0.2],
            "affected_route_share": [1.0],
        }
    )

    metrics = _national_gate_metrics(
        audits,
        segments,
        model_panel,
        all_partitions_available=True,
        restart_reused_share=1.0,
        runtime_seconds=1.0,
        storage_bytes=1,
        alerted_origin_fips={"55001"},
    )

    assert metrics["alerted_origin_comparable"] is False
    assert metrics["no_alerted_origin_bias"] is False


def test_national_gate_reconciles_failed_route_audit_weight_without_exposure():
    from run_route_exposure_national import _national_gate_metrics

    audits = pd.DataFrame(
        {
            "route_id": ["failed"],
            "route_signature": ["failed"],
            "commuter_car_weight": [4.0],
            "routing_eligible": [True],
            "status": ["RouteClientError"],
            "route_miles_total": [2.0],
            "unallocated_miles": [0.0],
            "home_fips": ["55001"],
            "same_tract": [False],
            "omitted_coordinate_worker_weight": [0.0],
            "omitted_car_share_worker_weight": [0.0],
            "error_message": ["routing failed"],
        }
    )
    segments = pd.DataFrame(
        {
            "county_fips": pd.Series(dtype="string"),
            "route_signature": pd.Series(dtype="string"),
            "route_miles_in_county": pd.Series(dtype="float64"),
        }
    )
    model_panel = pd.DataFrame(
        {
            "route_coverage_status": ["included_positive_denominator"],
            "total_commuter_car_miles": [10.0],
            "destination_dosage": [0.2],
            "affected_route_share": [0.5],
            "failed_route_commuter_car_weight": [4.0],
            "unallocated_commuter_car_weight": [0.0],
        }
    )

    metrics = _national_gate_metrics(
        audits,
        segments,
        model_panel,
        all_partitions_available=True,
        restart_reused_share=1.0,
        runtime_seconds=1.0,
        storage_bytes=1,
    )

    assert metrics["failed_route_commuter_car_weight"] == pytest.approx(4.0)
    assert metrics["route_audit_weights_reconciled"] is True
    assert metrics["omissions_and_failures_explicit"] is True

    mismatch = model_panel.assign(failed_route_commuter_car_weight=[0.0])
    mismatch_metrics = _national_gate_metrics(
        audits,
        segments,
        mismatch,
        all_partitions_available=True,
        restart_reused_share=1.0,
        runtime_seconds=1.0,
        storage_bytes=1,
    )
    assert mismatch_metrics["route_audit_weights_reconciled"] is False
    assert mismatch_metrics["omissions_and_failures_explicit"] is False


def test_national_gate_metrics_rejects_failed_audit_with_county_segments():
    from run_route_exposure_national import _national_gate_metrics

    audits = pd.DataFrame(
        {
            "route_signature": ["failed"],
            "commuter_car_weight": [4.0],
            "routing_eligible": [True],
            "status": ["RouteClientError"],
            "route_miles_total": [2.0],
            "unallocated_miles": [0.0],
            "home_fips": ["55001"],
            "same_tract": [False],
            "omitted_coordinate_worker_weight": [0.0],
            "omitted_car_share_worker_weight": [0.0],
            "error_message": ["routing failed"],
        }
    )
    segments = pd.DataFrame(
        {
            "route_signature": ["failed"],
            "county_fips": ["55001"],
            "route_miles_in_county": [2.0],
        }
    )
    model_panel = pd.DataFrame(
        {
            "route_coverage_status": ["included_positive_denominator"],
            "total_commuter_car_miles": [10.0],
            "destination_dosage": [0.2],
            "affected_route_share": [0.5],
            "failed_route_commuter_car_weight": [4.0],
            "unallocated_commuter_car_weight": [0.0],
        }
    )

    with pytest.raises(ValueError, match="non-Ok route audit has county segments"):
        _national_gate_metrics(
            audits,
            segments,
            model_panel,
            all_partitions_available=True,
            restart_reused_share=1.0,
            runtime_seconds=1.0,
            storage_bytes=1,
        )


def test_national_gate_requires_explicit_route_audit_weight_columns():
    from run_route_exposure_national import _national_gate_metrics

    audits = pd.DataFrame(
        {
            "route_signature": ["ok"],
            "commuter_car_weight": [4.0],
            "routing_eligible": [True],
            "status": ["Ok"],
            "route_miles_total": [2.0],
            "unallocated_miles": [0.0],
            "home_fips": ["55001"],
            "same_tract": [False],
            "omitted_coordinate_worker_weight": [0.0],
            "omitted_car_share_worker_weight": [0.0],
            "error_message": [None],
        }
    )
    segments = pd.DataFrame(
        {
            "route_signature": ["ok"],
            "county_fips": ["55001"],
            "route_miles_in_county": [2.0],
        }
    )
    model_panel = pd.DataFrame(
        {
            "route_coverage_status": ["included_positive_denominator"],
            "total_commuter_car_miles": [8.0],
            "destination_dosage": [0.2],
            "affected_route_share": [0.5],
        }
    )

    metrics = _national_gate_metrics(
        audits,
        segments,
        model_panel,
        all_partitions_available=True,
        restart_reused_share=1.0,
        runtime_seconds=1.0,
        storage_bytes=1,
    )

    assert metrics["route_audit_weights_reconciled"] is False
    assert metrics["omissions_and_failures_explicit"] is False


def test_partition_gate_panel_rejects_failed_audit_with_county_segments():
    import run_route_exposure_national as national

    rows = pd.DataFrame(
        {
            "analysis_year": [2022],
            "lodes_source_year": [2021],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": ["segments.parquet"],
            "source_partition_id": ["2022__2021__wi"],
            "source_manifest_id": ["flow-manifest"],
            "network_manifest_id": ["network-2022"],
        }
    )
    segments = pd.DataFrame(
        {
            "route_id": ["ok", "failed"],
            "route_signature": ["ok", "failed"],
            "home_fips": ["55001", "55001"],
            "work_fips": ["55001", "55001"],
            "outcome_fips": ["55001", "55001"],
            "workers": [10.0, 10.0],
            "home_car_share": [0.8, 0.5],
            "route_miles_in_county": [2.0, 5.0],
            "route_miles_total": [2.0, 5.0],
            "unallocated_miles": [0.0, 0.0],
            "segment_type": ["county", "county"],
            "source_manifest_id": ["flow-manifest", "flow-manifest"],
            "source_partition_id": ["2022__2021__wi", "2022__2021__wi"],
            "network_manifest_id": ["network-2022", "network-2022"],
        }
    )
    alerts = pd.DataFrame(
        {
            "home_fips": ["55001"],
            "alert_date": [pd.Timestamp("2022-06-15")],
            "geo_scope": ["county_same"],
        }
    )
    raw_panel = pd.DataFrame(
        {
            "fips": ["55001"],
            "date": [pd.Timestamp("2022-06-15")],
            "fatal_crashes": [0.0],
        }
    )
    route_audits = pd.DataFrame(
        {
            "route_id": ["ok", "failed", "unallocated"],
            "route_signature": ["ok", "failed", "unallocated"],
            "status": ["Ok", "RouteClientError", "Ok"],
            "routing_eligible": [True, True, True],
            "commuter_car_weight": [8.0, 4.0, 5.0],
            "unallocated_miles": [0.0, 0.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="non-Ok route audit has county segments"):
        national._partition_gate_panel(
            rows,
            segments,
            alerts,
            raw_panel,
            None,
            same_tract_mode="primary_calibrated",
            network_year=2022,
            route_audits=route_audits,
        )


def test_synthetic_and_production_partition_gates_receive_route_audits(
    tmp_path, monkeypatch
):
    import run_route_exposure_national as national

    seen = []
    real_partition_gate_panel = national._partition_gate_panel

    def capture_partition_gate_panel(*args, **kwargs):
        seen.append(kwargs.get("route_audits"))
        return real_partition_gate_panel(*args, **kwargs)

    monkeypatch.setattr(
        national, "_partition_gate_panel", capture_partition_gate_panel
    )

    _, fixture_root, _, production_args, validation = _production_fixture(tmp_path)
    national.build_production_gate_report(
        production_args, partition_validation=validation
    )
    assert seen
    assert all(isinstance(audits, pd.DataFrame) and not audits.empty for audits in seen)

    seen.clear()
    national.run_synthetic_national_dry_run(
        argparse.Namespace(
            output_dir=tmp_path / "synthetic-second",
            analysis_years=[2022],
            states=["wi", "il"],
            network_year=2022,
            same_tract_mode="primary_calibrated",
            chunk_rows=1,
            route_workers=1,
            checkpoint_every=2,
            geometry_sample_rate=0.0,
        )
    )
    assert seen
    assert all(isinstance(audits, pd.DataFrame) and not audits.empty for audits in seen)


def test_production_network_gate_rejects_scoped_manifest(tmp_path):
    import run_route_exposure_national as national
    from build_route_national_network import NATIONAL_STATES

    path = tmp_path / "network_manifest.json"
    path.write_text(
        json.dumps({
            "manifest_id": "network-2022",
            "network_year": 2022,
            "scope": "scoped",
            "states": ["wi", "il"],
        }),
        encoding="utf-8",
    )

    valid, details = national._network_manifest_valid(
        path,
        network_year=2022,
        segments=pd.DataFrame({"network_manifest_id": ["network-2022"]}),
    )

    assert valid is False
    assert details["scope"] == "scoped"
    assert details["missing_states"]
    assert len(NATIONAL_STATES) == 51


def test_production_network_gate_accepts_complete_national_manifest(tmp_path):
    import run_route_exposure_national as national
    from build_route_national_network import NATIONAL_STATES

    path = tmp_path / "network_manifest.json"
    path.write_text(
        json.dumps({
            "manifest_id": "network-2022",
            "network_year": 2022,
            "scope": "national",
            "states": list(NATIONAL_STATES),
        }),
        encoding="utf-8",
    )

    valid, details = national._network_manifest_valid(
        path,
        network_year=2022,
        segments=pd.DataFrame({"network_manifest_id": ["network-2022"]}),
    )

    assert valid is True
    assert details["state_count"] == 51


def test_same_tract_gate_rejects_treatment_sign_reversal():
    from run_route_exposure_national import evaluate_same_tract_results

    rows = []
    for term, estimates in {
        "own_affected_share": [-0.2, 0.1, -0.3],
        "cross_affected_share": [-0.1, -0.2, -0.3],
        "pass_through_affected_share": [-0.1, -0.2, -0.3],
    }.items():
        rows.extend(
            {
                "same_tract_mode": mode,
                "analysis_scope": "pooled",
                "term": term,
                "estimate": estimate,
            }
            for mode, estimate in zip(
                ["primary_calibrated", "zero", "exclude"], estimates, strict=True
            )
        )
    results = pd.DataFrame(rows)

    evaluation = evaluate_same_tract_results(
        results, same_tract_commuter_car_weight_share=0.10
    )

    assert evaluation["complete"] is True
    assert evaluation["stable"] is False
    assert evaluation["sign_reversal_terms"] == ["own_affected_share"]


def test_same_tract_gate_requires_all_route_treatment_estimates():
    from run_route_exposure_national import evaluate_same_tract_results

    results = pd.DataFrame(
        {
            "same_tract_mode": ["primary_calibrated", "zero", "exclude"],
            "analysis_scope": ["pooled"] * 3,
            "term": ["own_affected_share"] * 3,
            "estimate": [-0.2, -0.1, -0.3],
        }
    )

    evaluation = evaluate_same_tract_results(
        results, same_tract_commuter_car_weight_share=0.10
    )

    assert evaluation["complete"] is False
    assert evaluation["stable"] is False
    assert evaluation["missing_terms"] == [
        "cross_affected_share",
        "pass_through_affected_share",
    ]


def test_atomic_markdown_write_preserves_existing_report_when_replace_fails(
    tmp_path, monkeypatch
):
    import run_route_exposure_national as national

    report = tmp_path / "report.md"
    report.write_text("known-good\n", encoding="utf-8")
    real_replace = national.os.replace

    def interrupted_replace(source, destination):
        if Path(destination) == report:
            raise OSError("interrupted fixture")
        return real_replace(source, destination)

    monkeypatch.setattr(national.os, "replace", interrupted_replace)

    with pytest.raises(OSError, match="interrupted fixture"):
        national._atomic_write_text("new report\n", report)

    assert report.read_text(encoding="utf-8") == "known-good\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.md"]


def test_production_gate_reconciles_artifacts_and_records_invocation(tmp_path):
    import run_route_exposure_national as national

    fixture_root = tmp_path / "fixture"
    dry_args = argparse.Namespace(
        output_dir=fixture_root,
        analysis_years=[2022],
        states=["wi", "il"],
        network_year=2022,
        same_tract_mode="primary_calibrated",
        chunk_rows=1,
        route_workers=1,
        checkpoint_every=2,
        geometry_sample_rate=0.0,
    )
    national.run_synthetic_national_dry_run(dry_args)
    alerts_path = tmp_path / "alerts.csv"
    pd.DataFrame(
        {
            "home_fips": ["17001", "55001"],
            "alert_date": ["2022-06-15", "2022-06-15"],
            "geo_scope": ["county_same", "county_same"],
        }
    ).to_csv(alerts_path, index=False)
    run_metrics_path = tmp_path / "run_metrics.json"
    run_metrics_path.write_text(
        json.dumps({"runtime_seconds": 1.5, "restart_reused_share": 1.0}),
        encoding="utf-8",
    )
    manifest_path = fixture_root / "segments" / "segment_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    model_panel = pd.read_parquet(
        fixture_root / "analysis" / "national_route_model_panel.parquet"
    )
    raw_panel_path = tmp_path / "raw_panel.parquet"
    model_panel[["fips", "date", "fatal_crashes"]].to_parquet(
        raw_panel_path, index=False
    )
    destination_path = tmp_path / "destination.parquet"
    model_panel[["fips", "date", "destination_dosage"]].to_parquet(
        destination_path, index=False
    )
    validation = national.validate_requested_partitions(
        manifest, analysis_years=[2022], states=["wi", "il"]
    )
    args = argparse.Namespace(
        segment_manifest=manifest_path,
        alerts=alerts_path,
        panel=raw_panel_path,
        destination_exposure=destination_path,
        output_dir=fixture_root / "analysis",
        analysis_years=[2022],
        states=["wi", "il"],
        network_year=2022,
        network_manifest=fixture_root / "manifests" / "network_manifest.json",
        run_metrics=run_metrics_path,
        same_tract_results=fixture_root / "analysis" / "same_tract_model_results.csv",
        same_tract_mode="primary_calibrated",
        chunk_rows=250_000,
        route_workers=8,
        checkpoint_every=10_000,
        geometry_sample_rate=0.001,
    )

    result = national.build_production_gate_report(
        args, partition_validation=validation
    )

    assert result["accepted"] is True
    assert len(result["partition_results"]) == 2
    assert result["invocation"]["route_workers"] == 8
    assert result["network_manifest"]["network_year"] == 2022
    assert result["same_tract_evaluation"]["stable"] is True

    args.run_metrics = None
    missing_metrics = national.build_production_gate_report(
        args, partition_validation=validation
    )
    assert missing_metrics["accepted"] is False
    assert "runtime_footprint" in missing_metrics["failed_gates"]
    assert "restart_footprint" in missing_metrics["failed_gates"]


def test_model_fit_filters_manifest_to_requested_states(tmp_path):
    national, _, manifest, args, _ = _production_fixture(tmp_path)
    raw_panel = pd.read_parquet(args.panel)
    alerts = pd.read_csv(args.alerts)
    captured_fips = []
    captured_partitions = []

    def capture_runner(panel, **_kwargs):
        captured_fips.append(set(panel["fips"].astype(str)))
        captured_partitions.append(set(panel["source_partition_ids"].astype(str)))
        return pd.DataFrame({"status": ["captured"]})

    national.run_national_route_analysis(
        argparse.Namespace(
            segment_manifest=manifest,
            alerts=alerts,
            panel=raw_panel,
            destination_exposure=pd.read_parquet(args.destination_exposure),
            output_dir=None,
            analysis_years=[2022],
            states=["wi"],
            network_year=2022,
            same_tract_mode="primary_calibrated",
            bootstrap_reps=1,
            model_runner=capture_runner,
        )
    )

    assert captured_fips
    assert all(
        fips and all(county.startswith("55") for county in fips)
        for fips in captured_fips
    )
    wi_partition = manifest.loc[manifest["state"].eq("wi"), "source_partition_id"].iloc[0]
    assert all(partitions == {wi_partition} for partitions in captured_partitions)


def test_partition_gates_fail_when_partition_has_no_analysis_rows(tmp_path):
    national, _, manifest, args, validation = _production_fixture(tmp_path)
    panel = pd.read_parquet(args.panel)
    panel.loc[~panel["fips"].astype(str).str.startswith("55")].to_parquet(
        args.panel, index=False
    )

    result = national.build_production_gate_report(
        args, partition_validation=validation
    )

    wi_partition = manifest.loc[manifest["state"].eq("wi"), "source_partition_id"].iloc[0]
    wi_result = next(
        item
        for item in result["partition_results"]
        if item["source_partition_id"] == wi_partition
    )
    assert wi_result["accepted"] is False
    assert "denominators" in wi_result["failed_gates"]
    assert "route_destination_comparison" in wi_result["failed_gates"]


@pytest.mark.parametrize("tamper", ["missing", "mismatch"])
def test_same_tract_gate_rejects_absent_or_mismatched_provenance(tmp_path, tamper):
    national, _, _, args, validation = _production_fixture(tmp_path)
    results = pd.read_csv(args.same_tract_results)
    if tamper == "missing":
        results = results.drop(columns="network_manifest_ids")
    else:
        results["network_manifest_ids"] = "sha256:foreign-network"
    results.to_csv(args.same_tract_results, index=False)

    result = national.build_production_gate_report(
        args, partition_validation=validation
    )

    assert result["same_tract_evaluation"]["provenance"]["valid"] is False
    assert result["same_tract_evaluation"]["stable"] is False
    assert "same_tract_stability" in result["failed_gates"]


def test_partition_same_tract_gate_rejects_partition_sign_reversal(tmp_path):
    national, _, manifest, args, validation = _production_fixture(tmp_path)
    results = pd.read_csv(args.same_tract_results)
    wi_partition = manifest.loc[
        manifest["state"].eq("wi"), "source_partition_id"
    ].iloc[0]
    targeted = (
        results["analysis_scope"].astype(str).eq("partition")
        & results["source_partition_ids"].astype(str).eq(str(wi_partition))
        & results["same_tract_mode"].astype(str).eq("zero")
        & results["term"].astype(str).eq("own_affected_share")
    )
    assert targeted.any(), "fixture must contain partition-specific three-mode evidence"
    results.loc[targeted, "estimate"] = 0.01
    results.to_csv(args.same_tract_results, index=False)

    result = national.build_production_gate_report(
        args, partition_validation=validation
    )

    wi_result = next(
        item
        for item in result["partition_results"]
        if item["source_partition_id"] == wi_partition
    )
    assert result["same_tract_evaluation"]["stable"] is True
    assert wi_result["accepted"] is False
    assert "same_tract_stability" in wi_result["failed_gates"]
    assert wi_result["same_tract_evaluation"]["sign_reversal_terms"] == [
        "own_affected_share"
    ]


def test_alerted_origin_bias_uses_partition_analysis_year_only(tmp_path):
    national, _, _, args, validation = _production_fixture(tmp_path)
    alerts = pd.read_csv(args.alerts)
    stale = pd.DataFrame(
        {
            "home_fips": ["17003", "55003"],
            "alert_date": ["2024-06-15", "2024-06-15"],
            "geo_scope": ["county_same", "county_same"],
        }
    )
    pd.concat([alerts, stale], ignore_index=True).to_csv(args.alerts, index=False)

    result = national.build_production_gate_report(
        args, partition_validation=validation
    )

    assert result["accepted"] is True
    assert all(
        item["metrics"]["alerted_origin_comparable"]
        for item in result["partition_results"]
    )
