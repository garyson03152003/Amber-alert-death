import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from state_dot_analysis_core import (
    add_spillover_classes,
    build_commuter_spillover,
    build_ppml_call_spec,
    extract_finite_coefficients,
    normalize_state_outcomes,
    prepare_ppml_sample,
    summarize_fit_statuses,
    validate_analysis_inputs,
)


def test_normalize_state_outcomes_preserves_unavailable_outcomes_as_nan():
    raw = pd.DataFrame({
        "fips": ["36001", "36001"],
        "date": ["2024-01-01", "2024-01-01"],
        "ny_crashes": [2, 3],
        "ny_fatal_crashes": [1, 0],
    })
    out = normalize_state_outcomes(
        raw,
        crashes_col="ny_crashes",
        fatals_col="ny_fatal_crashes",
        serious_col=None,
        fatals_comparable=False,
    )
    assert out.loc[0, "crashes"] == 5
    assert out["fatals"].isna().all()
    assert out["serious_inj"].isna().all()


def test_prepare_ppml_sample_keeps_zero_counts():
    df = pd.DataFrame({
        "fips": ["01001"] * 5,
        "date": pd.date_range("2024-01-01", periods=5),
        "year": [2024] * 5,
        "population": [1000, 1000, 1000, np.nan, 1000],
        "night_alert": [0, 1, 0, 1, 0],
        "crashes": [0, 2, np.nan, 1, -1],
    })
    out = prepare_ppml_sample(df, "crashes")
    assert out["crashes"].tolist() == [0.0, 2.0]
    assert np.isclose(out["_log_population"].iloc[0], np.log(1000))


def test_ppml_spec_uses_offset_when_supported():
    def fake_fepois(formula, data, offset=None, vcov=None):
        pass

    spec = build_ppml_call_spec(
        fake_fepois, count_col="crashes", treatment_cols=("night_alert",)
    )
    assert spec["offset"] == "_log_population"
    assert spec["formula"].startswith("crashes ~ night_alert")


def test_ppml_spec_is_explicit_when_offset_not_supported():
    def fake_fepois(formula, data, vcov=None):
        pass

    spec = build_ppml_call_spec(
        fake_fepois, count_col="crashes", treatment_cols=("night_alert",)
    )
    assert spec["offset"] is None
    assert "_log_population" in spec["formula"]


def test_build_commuter_spillover_uses_destination_commuter_share():
    alerts = pd.DataFrame({
        "fips": ["01001"],
        "effective_crash_date": [pd.Timestamp("2024-01-02")],
        "night_alert": [1],
    })
    flows = pd.DataFrame({
        "fips_home": [1001, 1001, 1003, 1005],
        "fips_work": [1001, 1003, 1003, 1003],
        "workers": [100.0, 40.0, 120.0, 40.0],
        "weight": [1.0, 0.20, 0.60, 0.20],
    })
    spill = build_commuter_spillover(alerts, flows)
    row = spill.loc[spill["fips"] == "01003"].iloc[0]
    assert row["spillover_commuters"] == 40.0
    assert np.isclose(row["spillover_share"], 0.20)


def test_multiple_alerted_origins_sum_commuter_shares():
    alerts = pd.DataFrame({
        "fips": ["01001", "01005"],
        "effective_crash_date": [pd.Timestamp("2024-01-02")] * 2,
        "night_alert": [1, 1],
    })
    flows = pd.DataFrame({
        "fips_home": [1001, 1003, 1005],
        "fips_work": [1003, 1003, 1003],
        "workers": [40.0, 120.0, 40.0],
        "weight": [0.20, 0.60, 0.20],
    })
    spill = build_commuter_spillover(alerts, flows)
    row = spill.loc[spill["fips"] == "01003"].iloc[0]
    assert row["spillover_commuters"] == 80.0
    assert np.isclose(row["spillover_share"], 0.40)


def test_cross_border_alert_origin_contributes_to_validated_destination_exposure():
    """Alert origins are nationwide even when crash outcomes are state-scoped."""
    alerts = pd.DataFrame({
        "fips": ["06001"],
        "effective_crash_date": [pd.Timestamp("2024-01-02")],
        "night_alert": [1],
    })
    flows = pd.DataFrame({
        "fips_home": ["06001", "01001"],
        "fips_work": ["01003", "01003"],
        "workers": [40.0, 60.0],
        "weight": [0.40, 0.60],
    })
    spill = build_commuter_spillover(alerts, flows)
    row = spill.loc[spill["fips"] == "01003"].iloc[0]
    assert row["spillover_commuters"] == 40.0
    assert np.isclose(row["spillover_share"], 0.40)


def test_validated_analysis_inputs_fail_closed_for_missing_review_and_weights():
    panel = pd.DataFrame({
        "fips": ["01001"], "date": ["2024-01-01"], "year": [2024],
        "state": ["AL"], "coverage_valid": [True], "structural_zero": [True],
        "source": ["AL_DOT"],
    })
    manifest = pd.DataFrame({
        "state": ["AL"], "year": [2024], "coverage_valid": [True],
        "source": ["AL_DOT"],
    })
    review = pd.DataFrame(columns=["state", "year", "review_status"])
    try:
        validate_analysis_inputs(panel, manifest, review, flows=None)
    except ValueError as exc:
        assert "commuting weights" in str(exc)
    else:
        raise AssertionError("missing weights must fail closed")

    flows = pd.DataFrame({"fips_home": ["01001"], "fips_work": ["01003"], "workers": [1], "weight": [1.0]})
    try:
        validate_analysis_inputs(panel, manifest, review, flows=flows)
    except ValueError as exc:
        assert "reviewed accepted" in str(exc)
    else:
        raise AssertionError("unreviewed state-year must fail closed")


def test_nonfinite_coefficients_are_rejected_and_status_counts_reconcile():
    class FakeFit:
        def tidy(self):
            return pd.DataFrame({
                "Estimate": [0.1, np.inf],
                "Std. Error": [0.2, 0.3],
                "Pr(>|t|)": [0.5, 0.1],
            }, index=["night_alert", "spillover_share_10pp"])

    rows, produced, errors = extract_finite_coefficients(
        FakeFit(), ("night_alert", "spillover_share_10pp")
    )
    assert [row["term"] for row in rows] == ["night_alert"]
    assert produced == ("night_alert",)
    assert errors == {"spillover_share_10pp": "nonfinite_coefficient"}
    counts = summarize_fit_statuses([
        {"status": "ok", "terms_requested": "night_alert|spillover_share_10pp", "terms_produced": "night_alert"},
        {"status": "skipped", "terms_requested": "night_alert", "terms_produced": ""},
    ])
    assert counts == {"expected_fits": 2, "produced_fits": 1, "expected_terms": 3, "produced_terms": 1}


def test_exposed_neighbor_is_not_clean_control():
    panel = pd.DataFrame({
        "night_alert": [1, 0, 0],
        "spillover_share": [0.0, 0.20, 0.0],
    })
    out = add_spillover_classes(panel)
    assert out["exposure_class"].tolist() == ["direct", "spillover", "clean_control"]
    assert out["clean_control"].tolist() == [0, 0, 1]
