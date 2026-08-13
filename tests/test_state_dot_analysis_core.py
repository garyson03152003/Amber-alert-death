import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from state_dot_analysis_core import (
    add_spillover_classes,
    build_commuter_spillover,
    build_ppml_call_spec,
    normalize_state_outcomes,
    prepare_ppml_sample,
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


def test_build_commuter_spillover_marks_non_targeted_destination_exposure():
    alerts = pd.DataFrame({
        "fips": ["01001"],
        "effective_crash_date": [pd.Timestamp("2024-01-02")],
        "night_alert": [1],
    })
    flows = pd.DataFrame({
        "fips_home": [1001, 1001, 1003],
        "fips_work": [1001, 1003, 1001],
        "workers": [100.0, 40.0, 25.0],
    })
    spill = build_commuter_spillover(alerts, flows)
    row = spill.loc[spill["fips"] == "01003"].iloc[0]
    assert row["spillover_commuters"] == 40.0
    assert np.isclose(row["log_spillover_commuters"], np.log1p(40.0))


def test_exposed_neighbor_is_not_clean_control():
    panel = pd.DataFrame({
        "night_alert": [1, 0, 0],
        "spillover_commuters": [0.0, 40.0, 0.0],
    })
    out = add_spillover_classes(panel)
    assert out["exposure_class"].tolist() == ["direct", "spillover", "clean_control"]
    assert out["clean_control"].tolist() == [0, 0, 1]
