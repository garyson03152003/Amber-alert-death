import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _national_module():
    try:
        return importlib.import_module("run_route_exposure_national")
    except ModuleNotFoundError:
        pytest.fail("run_route_exposure_national module does not exist")


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["own", "cross", "pass"],
            "home_fips": ["55001", "55003", "55005"],
            "work_fips": ["55001", "55001", "55007"],
            "outcome_fips": ["55001", "55001", "55001"],
            "workers": [10.0, 10.0, 10.0],
            "home_car_share": [0.8, 0.5, 0.4],
            "route_miles_in_county": [2.0, 3.0, 4.0],
            "route_miles_total": [2.0, 3.0, 4.0],
            "unallocated_miles": [0.0, 0.0, 0.0],
            "segment_type": ["county", "county", "county"],
            "source_manifest_id": ["flow-manifest", "flow-manifest", "flow-manifest"],
            "network_manifest_id": ["network-2022", "network-2022", "network-2022"],
        }
    )


def _alerts(*dates: str) -> pd.DataFrame:
    rows = []
    for date in dates:
        rows.extend(
            {"home_fips": fips, "alert_date": pd.Timestamp(date), "geo_scope": "county_same"}
            for fips in ("55001", "55003", "55005")
        )
    return pd.DataFrame(rows)


def test_national_exposure_preserves_origin_components_and_vintages():
    national = _national_module()

    result = national.build_national_exposure(
        _segments(),
        _alerts("2022-01-02"),
        analysis_year=2022,
        flow_source_year=2021,
        car_share_vintage="2019-2023",
    )

    row = result.iloc[0]
    assert row["own_affected_car_miles"] == pytest.approx(16.0)
    assert row["cross_affected_car_miles"] == pytest.approx(15.0)
    assert row["pass_through_affected_car_miles"] == pytest.approx(16.0)
    assert row["lodes_source_year"] == 2021
    assert row["acs_car_share_vintage"] == "2019-2023"
    assert row["analysis_year"] == 2022
    assert row["route_exposure_national"] == 1


def test_same_tract_gate_accepts_established_coef_output_schema():
    national = _national_module()
    rows = []
    for term in national.ROUTE_TREATMENTS:
        rows.extend(
            {
                "same_tract_mode": mode,
                "analysis_scope": "pooled",
                "term": term,
                "coef": -0.1,
            }
            for mode in ("primary_calibrated", "zero", "exclude")
        )

    evaluation = national.evaluate_same_tract_results(
        pd.DataFrame(rows), same_tract_commuter_car_weight_share=0.10
    )

    assert evaluation["complete"] is True
    assert evaluation["stable"] is True


def test_same_tract_gate_rejects_missing_or_non_numeric_result_column():
    national = _national_module()
    missing = pd.DataFrame(
        {
            "same_tract_mode": ["primary_calibrated"],
            "term": ["own_affected_share"],
        }
    )
    invalid = missing.assign(coef=["not-a-number"])

    missing_evaluation = national.evaluate_same_tract_results(
        missing, same_tract_commuter_car_weight_share=0.10
    )
    invalid_evaluation = national.evaluate_same_tract_results(
        invalid, same_tract_commuter_car_weight_share=0.10
    )

    assert missing_evaluation["complete"] is False
    assert "estimate" in missing_evaluation["missing_columns"]
    assert invalid_evaluation["complete"] is False
    assert invalid_evaluation["missing_terms"] == [
        "cross_affected_share",
        "pass_through_affected_share",
    ]


def test_same_tract_gate_prefers_coef_and_rejects_invalid_coef_when_both_present():
    national = _national_module()
    rows = []
    for term in national.ROUTE_TREATMENTS:
        rows.extend(
            {
                "same_tract_mode": mode,
                "analysis_scope": "pooled",
                "term": term,
                "estimate": -0.1,
                "coef": "not-a-number",
            }
            for mode in ("primary_calibrated", "zero", "exclude")
        )

    evaluation = national.evaluate_same_tract_results(
        pd.DataFrame(rows), same_tract_commuter_car_weight_share=0.10
    )

    assert evaluation["complete"] is False
    assert evaluation["stable"] is False


def test_national_exposure_reconciles_audit_weights_without_failed_dosage():
    national = _national_module()
    audits = pd.DataFrame(
        {
            "route_id": ["failed", "unallocated"],
            "status": ["RouteClientError", "Ok"],
            "routing_eligible": [True, True],
            "commuter_car_weight": [4.0, 5.0],
            "unallocated_miles": [0.0, 2.0],
        }
    )

    result = national.build_national_exposure(
        _segments(),
        _alerts("2022-01-02"),
        analysis_year=2022,
        flow_source_year=2021,
        car_share_vintage="2019-2023",
        route_audits=audits,
    )

    row = result.iloc[0]
    assert row["failed_route_commuter_car_weight"] == pytest.approx(4.0)
    assert row["unallocated_commuter_car_weight"] == pytest.approx(5.0)
    assert row["affected_commuter_car_miles"] == pytest.approx(47.0)
    assert row["total_commuter_car_miles"] == pytest.approx(47.0)


def test_national_exposure_rejects_county_segments_for_failed_audits():
    national = _national_module()
    segments = pd.DataFrame(
        {
            "route_id": ["ok", "failed", "unallocated"],
            "route_signature": ["ok", "failed", "unallocated"],
            "home_fips": ["55001", "55001", "55001"],
            "work_fips": ["55001", "55001", "55001"],
            "outcome_fips": ["55001", "55001", "55001"],
            "workers": [10.0, 10.0, 10.0],
            "home_car_share": [0.8, 0.5, 0.5],
            "route_miles_in_county": [2.0, 5.0, 2.0],
            "route_miles_total": [2.0, 5.0, 4.0],
            "unallocated_miles": [0.0, 0.0, 2.0],
            "segment_type": ["county", "county", "county"],
            "source_manifest_id": ["flow-manifest"] * 3,
            "network_manifest_id": ["network-2022"] * 3,
        }
    )
    audits = pd.DataFrame(
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
        national.build_national_exposure(
            segments,
            _alerts("2022-01-02"),
            analysis_year=2022,
            flow_source_year=2021,
            car_share_vintage="2019-2023",
            route_audits=audits,
        )


def test_required_flow_partitions_reuses_nearest_source_year_and_rejects_missing_mapping():
    national = _national_module()

    assert national.required_flow_partitions(
        [2013, 2014, 2015], {2013: 2013, 2014: 2013, 2015: 2014}
    ) == [2013, 2014]
    with pytest.raises(ValueError, match="missing flow mapping"):
        national.required_flow_partitions([2013, 2014], {2013: 2013})


def test_zero_same_tract_mode_removes_same_tract_route_mileage():
    national = _national_module()
    segments = _segments().assign(same_tract=[True, False, False])

    result = national.build_national_exposure(
        segments,
        _alerts("2022-01-02"),
        analysis_year=2022,
        flow_source_year=2021,
        car_share_vintage="2019-2023",
        same_tract_mode="zero",
    )

    row = result.iloc[0]
    assert row["own_affected_car_miles"] == 0.0
    assert row["total_commuter_car_miles"] == pytest.approx(31.0)


def test_pilot_helper_accepts_custom_label_and_vintage_values_without_changing_default():
    pilot = importlib.import_module("run_route_exposure_pilot")

    custom = pilot.build_alert_date_exposures(
        _segments(),
        _alerts("2022-01-02"),
        "primary_calibrated",
        label="route_exposure_national",
        vintage_columns={"analysis_year": 2022, "lodes_source_year": 2021},
    )
    default = pilot.build_alert_date_exposures(
        _segments(), _alerts("2022-01-02"), "primary_calibrated"
    )

    assert custom["route_exposure_national"].eq(1).all()
    assert custom["analysis_year"].eq(2022).all()
    assert custom["lodes_source_year"].eq(2021).all()
    assert default["route_exposure_2022"].eq(1).all()


def test_national_runner_reuses_segments_and_keeps_destination_measure(tmp_path, monkeypatch):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    _segments().assign(
        lodes_source_year=2013,
        acs_car_share_vintage="2009-2013",
    ).to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.legacy.v0"] * 2,
            "analysis_year": [2013, 2014],
            "lodes_source_year": [2013, 2013],
            "acs_car_share_vintage": ["2009-2013", "2009-2013"],
            "segment_path": [str(segment_path), str(segment_path)],
            "status": ["success", "success"],
        }
    )
    panel = pd.DataFrame(
        {
            "fips": ["55001", "55001", "55001"],
            "date": pd.to_datetime(["2013-01-02", "2013-01-03", "2014-01-02"]),
            "fatal_crashes": [1.0, 0.0, 2.0],
        }
    )
    destination = panel[["fips", "date"]].assign(destination_dosage=[0.25, 0.0, 0.5])
    read_calls = []
    real_read = pd.read_parquet

    def counting_read(path, *args, **kwargs):
        if Path(path) == segment_path:
            read_calls.append(Path(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(national.pd, "read_parquet", counting_read)
    model_calls = []

    def model_runner(model_panel, *, scope, treatment_columns, **_):
        model_calls.append((scope, model_panel.copy(), tuple(treatment_columns)))
        return pd.DataFrame({"status": ["ok"], "nobs": [len(model_panel)]})

    args = argparse.Namespace(
        segment_manifest=manifest,
        alerts=_alerts("2013-01-02", "2014-01-02"),
        panel=panel,
        destination_exposure=destination,
        analysis_years=[2013, 2014],
        same_tract_mode="primary_calibrated",
        model_runner=model_runner,
        output_dir=None,
    )

    results = national.run_national_route_analysis(args)

    assert read_calls == [segment_path]
    assert set(results["analysis_scope"]) == {"year", "vintage", "pooled"}
    assert results["same_tract_mode"].eq("primary_calibrated").all()
    assert results["network_manifest_ids"].eq("network-2022").all()
    assert results["source_manifest_ids"].eq("flow-manifest").all()
    pooled_panel = next(frame for scope, frame, _ in model_calls if scope == "pooled")
    assert "destination_dosage" in pooled_panel
    assert pooled_panel["analysis_year"].tolist() == [2013, 2013, 2014]
    assert pooled_panel["lodes_source_year"].eq(2013).all()
    assert pooled_panel["acs_car_share_vintage"].eq("2009-2013").all()
    control = pooled_panel.loc[pooled_panel["date"].eq(pd.Timestamp("2013-01-03"))].iloc[0]
    assert control["affected_route_share"] == 0.0
    assert control["total_commuter_car_miles"] == pytest.approx(47.0)
    assert "cross_affected_share" in next(terms for scope, _, terms in model_calls if scope == "pooled")


def test_national_runner_preserves_state_specific_nearest_vintage_sets(tmp_path):
    national = _national_module()
    first_path = tmp_path / "wi.parquet"
    second_path = tmp_path / "mi.parquet"
    first = _segments().assign(
        analysis_year=2022,
        lodes_source_year=2021,
        acs_car_share_vintage="2017-2021",
    )
    second = _segments().assign(
        route_id=lambda frame: frame["route_id"] + "-mi",
        analysis_year=2022,
        lodes_source_year=2022,
        acs_car_share_vintage="2018-2022",
    )
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.legacy.v0"] * 2,
            "analysis_year": [2022, 2022],
            "lodes_source_year": [2021, 2022],
            "acs_car_share_vintage": ["2017-2021", "2018-2022"],
            "segment_path": [str(first_path), str(second_path)],
            "status": ["success", "success"],
        }
    )
    captured = []

    def model_runner(model_panel, **_):
        captured.append(model_panel.copy())
        return pd.DataFrame({"status": ["ok"]})

    national.run_national_route_analysis(
        argparse.Namespace(
            segment_manifest=manifest,
            alerts=_alerts("2022-01-02"),
            panel=pd.DataFrame(
                {"fips": ["55001"], "date": [pd.Timestamp("2022-01-02")], "fatal_crashes": [1.0]}
            ),
            destination_exposure=None,
            analysis_years=[2022],
            same_tract_mode="primary_calibrated",
            model_runner=model_runner,
            output_dir=None,
        )
    )

    assert captured[0]["lodes_source_years"].iloc[0] == "2021,2022"
    assert captured[0]["acs_car_share_vintages"].iloc[0] == "2017-2021,2018-2022"
    assert pd.isna(captured[0]["lodes_source_year"].iloc[0])
    assert captured[0]["acs_car_share_vintage"].iloc[0] == "mixed"


def test_loaded_segment_partition_rejects_manifest_vintage_relabel(tmp_path):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    _segments().assign(
        analysis_year=2021,
        lodes_source_year=2021,
        acs_car_share_vintage="2017-2021",
    ).to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.legacy.v0"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": [str(segment_path)],
            "status": ["success"],
        }
    )

    with pytest.raises(ValueError, match="partition provenance mismatch"):
        national.run_national_route_analysis(
            argparse.Namespace(
                segment_manifest=manifest,
                alerts=_alerts("2022-01-02"),
                panel=pd.DataFrame(
                    {"fips": ["55001"], "date": [pd.Timestamp("2022-01-02")], "fatal_crashes": [1.0]}
                ),
                destination_exposure=None,
                analysis_years=[2022],
                same_tract_mode="primary_calibrated",
                model_runner=lambda *_args, **_kwargs: pd.DataFrame({"status": ["ok"]}),
                output_dir=None,
            )
        )


def test_loaded_segment_partition_rejects_manifest_checksum_mismatch(tmp_path):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    _segments().assign(
        analysis_year=2022,
        lodes_source_year=2022,
        acs_car_share_vintage="2018-2022",
    ).to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.legacy.v0"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": [str(segment_path)],
            "segment_sha256": ["0" * 64],
            "status": ["success"],
        }
    )

    with pytest.raises(ValueError, match="checksum"):
        national.run_national_route_analysis(
            argparse.Namespace(
                segment_manifest=manifest,
                alerts=_alerts("2022-01-02"),
                panel=pd.DataFrame(
                    {"fips": ["55001"], "date": [pd.Timestamp("2022-01-02")], "fatal_crashes": [1.0]}
                ),
                destination_exposure=None,
                analysis_years=[2022],
                same_tract_mode="primary_calibrated",
                model_runner=lambda *_args, **_kwargs: pd.DataFrame({"status": ["ok"]}),
                output_dir=None,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="source_partition_id"), "source_partition_id"),
        (lambda frame: frame.assign(network_manifest_id="  "), "network_manifest_id"),
    ],
)
def test_v1_segment_manifest_requires_nonblank_task5_provenance_fields(
    mutation, message
):
    national = _national_module()
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.v1"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": ["segments.parquet"],
            "source_manifest_id": ["flow-manifest"],
            "network_manifest_id": ["network-2022"],
            "source_partition_id": ["2022__2022__wi"],
        }
    )

    with pytest.raises(ValueError, match=message):
        national._manifest_frame(mutation(manifest))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="source_manifest_id"), "source_manifest_id"),
        (
            lambda frame: frame.assign(
                network_manifest_id=["network-2022", "  ", "network-2022"]
            ),
            "network_manifest_id",
        ),
        (
            lambda frame: frame.assign(
                source_partition_id=["2022__2022__wi", "2022__2022__mi", "2022__2022__wi"]
            ),
            "source_partition_id",
        ),
        (lambda frame: frame.drop(columns="analysis_year"), "analysis_year"),
    ],
)
def test_v1_segment_rejects_missing_blank_or_mixed_task5_provenance(
    tmp_path, mutation, message
):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    segment = _segments().assign(
        schema_version="route_national.segments.v1",
        analysis_year=2022,
        lodes_source_year=2022,
        acs_car_share_vintage="2018-2022",
        source_partition_id="2022__2022__wi",
    )
    mutation(segment).to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.v1"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": [str(segment_path)],
            "source_manifest_id": ["flow-manifest"],
            "network_manifest_id": ["network-2022"],
            "source_partition_id": ["2022__2022__wi"],
            "status": ["success"],
        }
    )

    with pytest.raises(ValueError, match=message):
        national.run_national_route_analysis(
            argparse.Namespace(
                segment_manifest=manifest,
                alerts=_alerts("2022-01-02"),
                panel=pd.DataFrame(
                    {
                        "fips": ["55001"],
                        "date": [pd.Timestamp("2022-01-02")],
                        "fatal_crashes": [1.0],
                    }
                ),
                destination_exposure=None,
                analysis_years=[2022],
                same_tract_mode="primary_calibrated",
                model_runner=lambda *_args, **_kwargs: pd.DataFrame({"status": ["ok"]}),
                output_dir=None,
            )
        )


@pytest.mark.parametrize(
    "column",
    [
        "schema_version",
        "analysis_year",
        "lodes_source_year",
        "acs_car_share_vintage",
        "source_manifest_id",
        "network_manifest_id",
        "source_partition_id",
    ],
)
def test_v1_segment_rejects_partially_missing_provenance_rows(tmp_path, column):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    segment = _segments().assign(
        schema_version="route_national.segments.v1",
        analysis_year=2022,
        lodes_source_year=2022,
        acs_car_share_vintage="2018-2022",
        source_partition_id="2022__2022__wi",
    )
    segment.loc[1, column] = pd.NA
    segment.to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.v1"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": [str(segment_path)],
            "source_manifest_id": ["flow-manifest"],
            "network_manifest_id": ["network-2022"],
            "source_partition_id": ["2022__2022__wi"],
            "status": ["success"],
        }
    )

    with pytest.raises(ValueError, match=column):
        national.run_national_route_analysis(
            argparse.Namespace(
                segment_manifest=manifest,
                alerts=_alerts("2022-01-02"),
                panel=pd.DataFrame(
                    {
                        "fips": ["55001"],
                        "date": [pd.Timestamp("2022-01-02")],
                        "fatal_crashes": [1.0],
                    }
                ),
                destination_exposure=None,
                analysis_years=[2022],
                same_tract_mode="primary_calibrated",
                model_runner=lambda *_args, **_kwargs: pd.DataFrame({"status": ["ok"]}),
                output_dir=None,
            )
        )


def test_explicit_legacy_manifest_rejects_a_different_embedded_schema(tmp_path):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    _segments().assign(
        schema_version="unknown.segment.schema",
        lodes_source_year=2022,
        acs_car_share_vintage="2018-2022",
    ).to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.legacy.v0"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": [str(segment_path)],
        }
    )

    with pytest.raises(ValueError, match="schema_version"):
        national._combine_year_segments(
            national._manifest_frame(manifest), {}
        )


def test_model_sample_excludes_counties_without_route_denominator(tmp_path):
    national = _national_module()
    segment_path = tmp_path / "segments.parquet"
    _segments().assign(
        lodes_source_year=2022,
        acs_car_share_vintage="2018-2022",
    ).to_parquet(segment_path, index=False)
    manifest = pd.DataFrame(
        {
            "schema_version": ["route_national.segments.legacy.v0"],
            "analysis_year": [2022],
            "lodes_source_year": [2022],
            "acs_car_share_vintage": ["2018-2022"],
            "segment_path": [str(segment_path)],
            "status": ["success"],
        }
    )
    captured = []

    def model_runner(model_panel, **_):
        captured.append(model_panel.copy())
        return pd.DataFrame({"status": ["ok"]})

    national.run_national_route_analysis(
        argparse.Namespace(
            segment_manifest=manifest,
            alerts=_alerts("2022-01-02"),
            panel=pd.DataFrame(
                {
                    "fips": ["55001", "55099"],
                    "date": [pd.Timestamp("2022-01-02")] * 2,
                    "fatal_crashes": [1.0, 0.0],
                }
            ),
            destination_exposure=None,
            analysis_years=[2022],
            same_tract_mode="primary_calibrated",
            model_runner=model_runner,
            output_dir=None,
        )
    )

    assert all(set(frame["fips"]) == {"55001"} for frame in captured)
    assert all(frame["route_coverage_status"].eq("included_positive_denominator").all() for frame in captured)


def test_established_route_specs_retain_controls_fixed_effects_and_inference():
    national = _national_module()
    panel = pd.DataFrame(
        columns=[
            "is_holiday",
            "is_day_after_holiday",
            "prcp_mm",
            "tmax_c",
            "fatals_tm1",
            "day_alert",
            "other_wea_night_alert",
            "other_wea_night_count",
        ]
    )

    specs = national.established_route_model_specs(panel)

    assert {spec.fixed_effect_label for spec in specs} >= {
        "baseline_calendar",
        "county_year_weekday",
        "state_date",
    }
    assert {spec.inference for spec in specs} >= {"webb_state", "rademacher_state_month"}
    assert {spec.other_wea_control for spec in specs} >= {"binary", "dose"}
    for spec in specs:
        assert {
            "is_holiday",
            "is_day_after_holiday",
            "prcp_mm",
            "tmax_c",
            "fatals_tm1",
            "day_alert",
        }.issubset(spec.controls)


def test_established_model_runner_passes_controls_and_specification_ladder(monkeypatch):
    national = _national_module()
    robustness = importlib.import_module("run_symmetric_commuter_robustness")
    panel = pd.DataFrame(
        {
            "fips": ["55001", "55003"],
            "date": pd.to_datetime(["2022-01-02", "2022-01-03"]),
            "fatal_crashes": [1.0, 0.0],
            "own_affected_share": [0.2, 0.0],
            "cross_affected_share": [0.1, 0.0],
            "pass_through_affected_share": [0.05, 0.0],
            "is_holiday": [0, 0],
            "is_day_after_holiday": [0, 1],
            "prcp_mm": [0.0, 1.0],
            "tmax_c": [10.0, 11.0],
            "fatals_tm1": [0.0, 1.0],
            "day_alert": [0, 0],
            "other_wea_night_alert": [0, 1],
            "other_wea_night_count": [0, 2],
        }
    )
    calls = []

    def fake_fit(_panel, outcome, terms, **kwargs):
        calls.append((outcome, tuple(terms), kwargs))
        return [
            {"term": term, "status": "ok", "coef": 0.0}
            for term in terms
        ]

    monkeypatch.setattr(robustness, "_fit_analytic", fake_fit)
    result = national._established_model_runner(
        panel,
        scope="pooled",
        treatment_columns=national.ROUTE_TREATMENTS,
        bootstrap_reps=99,
    )

    assert calls
    assert {call[2]["fixed_effect_cols"] for call in calls} >= {
        ("fips_id", "year_id", "dow_id", "month_id"),
        ("fips_year_id", "fips_dow_id", "month_id"),
        ("fips_year_id", "fips_dow_id", "state_date_id"),
    }
    assert {call[2]["wild_kind"] for call in calls} == {"webb", "rademacher"}
    assert all("is_holiday" in terms and "fatals_tm1" in terms for _, terms, _ in calls)
    assert set(result["term"]) == set(national.ROUTE_TREATMENTS)
    assert set(result["other_wea_control"]) == {"binary", "dose"}
