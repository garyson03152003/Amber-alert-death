import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_state_dot_analysis_fixed as runner
import run_state_dot_analysis_share as share_runner
import run_validated_fars_share as fars_runner


def test_state_loader_logs_counts_without_formatter_errors(tmp_path, monkeypatch, caplog):
    data = pd.DataFrame({
        "fips": ["01001"],
        "date": ["2024-01-01"],
        "test_crashes": [2],
        "test_fatals": [0],
        "test_serious": [1],
    })
    data.to_parquet(tmp_path / "test_county_day.parquet", index=False)
    monkeypatch.setattr(runner, "DATA_PROC", tmp_path)
    monkeypatch.setattr(runner, "STATE_FILES", {
        "ZZ": {
            "file": "test_county_day.parquet",
            "crashes": "test_crashes",
            "fatals": "test_fatals",
            "serious": "test_serious",
        }
    })

    runner.load_state_crashes()

    messages = [record.getMessage() for record in caplog.records]
    assert any("ZZ: 1 county-days" in message for message in messages)


def test_validated_loader_rejects_sparse_legacy_substitute(tmp_path, monkeypatch):
    validated = tmp_path / "validated"
    validated.mkdir()
    pd.DataFrame({
        "fips": ["01001"], "date": ["2024-01-01"], "year": [2024],
        "crashes": [0], "person_fatals": [0], "serious_injury_persons": [0],
        "coverage_valid": [True], "structural_zero": [True], "source": ["AL_DOT"],
    }).to_parquet(validated / "al_county_day.parquet", index=False)
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    pd.DataFrame({"state": ["AL"], "year": [2024], "coverage_valid": [True], "source": ["AL_DOT"]}).to_parquet(
        coverage / "al_coverage.parquet", index=False
    )
    review = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["AL"], "year": [2024], "review_status": ["accepted"]}).to_csv(review, index=False)
    monkeypatch.setattr(runner, "DATA_PROC", tmp_path)
    monkeypatch.setattr(runner, "ACCEPTED_STATE_YEARS", review)

    out = runner.load_validated_state_crashes(direct_only=True)

    assert out.loc[0, "state"] == "AL"
    assert out.loc[0, "fatals"] == 0


def test_validated_state_loader_rejects_forged_panel_source(tmp_path, monkeypatch):
    validated = tmp_path / "validated"
    validated.mkdir()
    pd.DataFrame({
        "fips": ["01001"], "date": ["2024-01-01"], "year": [2024],
        "crashes": [0], "person_fatals": [0], "serious_injury_persons": [0],
        "coverage_valid": [True], "structural_zero": [True], "source": ["FORGED_SOURCE"],
    }).to_parquet(validated / "al_county_day.parquet", index=False)
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    pd.DataFrame({"state": ["AL"], "year": [2024], "coverage_valid": [True], "source": ["AL_DOT"]}).to_parquet(
        coverage / "al_coverage.parquet", index=False
    )
    review = tmp_path / "accepted_state_years.csv"
    pd.DataFrame({"state": ["AL"], "year": [2024], "review_status": ["accepted"]}).to_csv(review, index=False)
    monkeypatch.setattr(runner, "DATA_PROC", tmp_path)
    monkeypatch.setattr(runner, "ACCEPTED_STATE_YEARS", review)

    try:
        runner.load_validated_state_crashes(direct_only=True)
    except ValueError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("forged panel source must not pass state/year-only validation")


def test_validated_fars_loader_requires_balanced_provenance(tmp_path, monkeypatch):
    panel_path = tmp_path / "fars_balanced_county_day.parquet"
    pd.DataFrame({
        "fips": ["01001"], "date": ["2024-01-01"], "year": [2024],
        "person_fatals": [0], "fatal_crashes": [0], "coverage_valid": [True],
        "structural_zero": [True], "source": ["FARS_NHTSA"],
    }).to_parquet(panel_path, index=False)
    manifest_path = tmp_path / "fars_coverage.csv"
    pd.DataFrame({"state": ["US"], "year": [2024], "coverage_valid": [True], "source": ["FARS_NHTSA"]}).to_csv(
        manifest_path, index=False
    )
    monkeypatch.setattr(fars_runner, "FARS_BALANCED", panel_path)
    monkeypatch.setattr(fars_runner, "FARS_MANIFEST", manifest_path)

    out = fars_runner.load_validated_fars(direct_only=True)

    assert out.loc[0, "fatals"] == 0
    assert out.loc[0, "state"] == "01"


def test_validated_fars_loader_rejects_forged_source(tmp_path, monkeypatch):
    panel_path = tmp_path / "fars_balanced_county_day.parquet"
    pd.DataFrame({
        "fips": ["01001"], "date": ["2024-01-01"], "year": [2024],
        "person_fatals": [0], "fatal_crashes": [0], "coverage_valid": [True],
        "structural_zero": [True], "source": ["FORGED_SOURCE"],
    }).to_parquet(panel_path, index=False)
    manifest_path = tmp_path / "fars_coverage.csv"
    pd.DataFrame({"state": ["US"], "year": [2024], "coverage_valid": [True], "source": ["FARS_NHTSA"]}).to_csv(
        manifest_path, index=False
    )
    monkeypatch.setattr(fars_runner, "FARS_BALANCED", panel_path)
    monkeypatch.setattr(fars_runner, "FARS_MANIFEST", manifest_path)

    try:
        fars_runner.load_validated_fars(direct_only=True)
    except ValueError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("forged FARS panel source must fail")


def test_default_fars_joint_panel_scales_spillover_before_shared_estimators(tmp_path, monkeypatch):
    panel_path = tmp_path / "fars_balanced_county_day.parquet"
    days = pd.date_range("2024-01-01", periods=2)
    base_panel = pd.DataFrame({
        "fips": ["01001", "01003"] * len(days), "date": days.repeat(2),
        "year": [2024] * (2 * len(days)), "person_fatals": [0, 1] * len(days),
        "fatal_crashes": [0, 1] * len(days), "coverage_valid": [True] * (2 * len(days)),
        "structural_zero": [True] * (2 * len(days)), "source": ["FARS_NHTSA"] * (2 * len(days)),
    })
    base_panel.to_parquet(panel_path, index=False)
    manifest_path = tmp_path / "fars_coverage.csv"
    pd.DataFrame({"state": ["US"], "year": [2024], "coverage_valid": [True], "source": ["FARS_NHTSA"]}).to_csv(
        manifest_path, index=False
    )
    (tmp_path / "commuting").mkdir()
    pd.DataFrame({"fips_home": ["01001", "01003"], "fips_work": ["01003", "01001"], "workers": [10, 10], "weight": [1.0, 1.0]}).to_parquet(
        tmp_path / "commuting" / "county_commuting_weights.parquet", index=False
    )
    pd.DataFrame({"fips": ["01001", "01003"], "year": [2024, 2024], "population": [1000, 1000]}).to_parquet(
        tmp_path / "county_population.parquet", index=False
    )
    alerts = pd.DataFrame({"fips": ["01001"], "effective_crash_date": [days[0]], "night_alert": [1]})
    monkeypatch.setattr(fars_runner, "DATA_PROC", tmp_path)
    monkeypatch.setattr(fars_runner, "FARS_BALANCED", panel_path)
    monkeypatch.setattr(fars_runner, "FARS_MANIFEST", manifest_path)
    monkeypatch.setattr(fars_runner.base, "load_verified_night_alerts", lambda: alerts)

    panel = fars_runner.build_panel()
    assert "spillover_share_10pp" in panel.columns
    assert share_runner.run_wls(panel, "fatals_per_100k", "FARS_NATIONAL")[-1]["record_type"] == "fit_status"


def _estimable_panel():
    dates = pd.date_range("2024-01-01", periods=60).repeat(2)
    return pd.DataFrame({
        "fips": ["01001", "01003"] * 60,
        "date": dates,
        "year": [2024] * 120,
        "population": [10_000.0] * 120,
        "night_alert": [0, 1] * 60,
        "clean_control": [1, 0] * 60,
        "spillover_share": [0.0] * 120,
        "spillover_share_10pp": [0.0] * 120,
        "crashes": [0.0, 1.0] * 60,
        "crashes_per_100k": [0.0, 10.0] * 60,
    })


def test_wls_collinearity_is_logged_and_skipped(monkeypatch, caplog):
    def fail_collinear(*args, **kwargs):
        raise ValueError("All variables are collinear")

    monkeypatch.setattr(share_runner.pf, "feols", fail_collinear)
    rows = share_runner.run_wls(
        _estimable_panel(), "crashes_per_100k", "ZZ", clean_controls=True
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_reason"] == "All variables are collinear"
    assert "Skipping ZZ crashes_per_100k WLS_TWFE direct_vs_clean" in caplog.text


def test_ppml_estimation_failure_is_logged_and_skipped(monkeypatch, caplog):
    def fail_convergence(*args, **kwargs):
        raise ValueError("failed to converge")

    monkeypatch.setattr(share_runner.pf, "fepois", fail_convergence)
    rows = share_runner.run_ppml(
        _estimable_panel(), "crashes", "ZZ", clean_controls=True
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_reason"] == "failed to converge"
    assert rows[0]["input_n"] == 120
    assert rows[0]["terms_requested"] == "night_alert"
    assert "Skipping ZZ crashes PPML_raw_count direct_vs_clean" in caplog.text


def test_wls_nonfinite_estimate_is_recorded_not_emitted(monkeypatch):
    class FakeFit:
        _N = 120

        def tidy(self):
            return pd.DataFrame({
                "Estimate": [float("nan")], "Std. Error": [0.1], "Pr(>|t|)": [0.5],
            }, index=["night_alert"])

    monkeypatch.setattr(share_runner.pf, "feols", lambda *args, **kwargs: FakeFit())
    rows = share_runner.run_wls(
        _estimable_panel(), "crashes_per_100k", "ZZ", clean_controls=True
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_reason"] == "nonfinite_coefficient"


@pytest.mark.parametrize("analysis_module", [runner, share_runner])
def test_main_records_skipped_status_for_every_low_alert_and_unavailable_combination(
    tmp_path, monkeypatch, analysis_module
):
    panel = pd.DataFrame({
        "fips": ["01001", "01003"], "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "year": [2024, 2024], "state": ["AL", "AL"], "population": [1000, 1000],
        "night_alert": [0, 0], "clean_control": [1, 1], "spillover_share": [0.0, 0.0],
        "spillover_share_10pp": [0.0, 0.0], "crashes": [0.0, 0.0],
        "fatals": [float("nan"), float("nan")], "serious_inj": [float("nan"), float("nan")],
        "crashes_per_100k": [0.0, 0.0], "fatals_per_100k": [float("nan"), float("nan")],
        "serious_per_100k": [float("nan"), float("nan")], "exposure_class": ["clean_control", "clean_control"],
    })
    monkeypatch.setattr(analysis_module, "build_panel", lambda **kwargs: panel)
    monkeypatch.setattr(analysis_module, "OUTPUT_TABS", tmp_path)

    analysis_module.main([])

    suffix = "share" if analysis_module is share_runner else "fixed"
    statuses = pd.read_csv(tmp_path / f"state_dot_analysis_{suffix}_status.csv")
    fits = statuses.loc[statuses["record_type"].eq("fit_status")]
    # ALL plus AL; three outcomes; joint/direct-clean WLS and PPML.
    assert len(fits) == 2 * 3 * 4
    assert fits["status"].eq("skipped").all()
    assert fits["input_n"].notna().all()
    assert fits["error_reason"].eq("insufficient_estimable_sample").all()
