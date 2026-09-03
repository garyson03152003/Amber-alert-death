import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from build_lodes_county_pair_distance import YEAR_WEIGHTS as DISTANCE_YEAR_WEIGHTS
from build_lodes_tract_car_dosage import YEAR_WEIGHTS as CAR_DISTANCE_YEAR_WEIGHTS
from run_symmetric_commuter_fatigue import (
    build_linear_time_outcomes,
    build_pair_dosage,
    construct_symmetric_exposures,
    construct_year_matched_exposure_series,
    exposure_vintage_for_year,
    fit_within_ols,
    restrict_to_self_loop_counties,
    symmetric_exposure_series,
    wild_cluster_bootstrap,
)


D = pd.Timestamp("2024-03-05")


def _weights():
    return pd.DataFrame(
        {
            "fips_home": [1001, 1003, 1003, 1005, 1005],
            "fips_work": [1001, 1001, 1003, 1001, 1005],
            "weight": [0.60, 0.25, 0.40, 0.15, 0.70],
        }
    )


def _joint_distance():
    return pd.DataFrame(
        {
            "fips_home": ["01001", "01003", "01003", "01005", "01005"],
            "fips_work": ["01001", "01001", "01003", "01001", "01005"],
            "avg_car_x_dist": [4.0, 10.0, 6.0, 20.0, 5.0],
        }
    )


def test_pair_dosage_uses_identical_units_for_self_and_cross_edges():
    pairs = build_pair_dosage(_weights(), _joint_distance())
    got = pairs.set_index(["fips_home", "fips_work"])["commuter_car_miles"]
    assert got[("01001", "01001")] == pytest.approx(0.60 * 4.0)
    assert got[("01003", "01001")] == pytest.approx(0.25 * 10.0)
    assert got[("01005", "01005")] == pytest.approx(0.70 * 5.0)


def test_pair_dosage_fails_closed_when_joint_distance_is_missing():
    joint = _joint_distance().iloc[:-1]
    with pytest.raises(ValueError, match="missing tract-preserved"):
        build_pair_dosage(_weights(), joint)


def test_pair_dosage_uses_an_explicit_fallback_for_missing_joint_distance():
    joint = _joint_distance().iloc[:-1]
    fallback = pd.DataFrame(
        {
            "fips_home": ["01005"],
            "fips_work": ["01005"],
            "fallback_car_x_dist": [4.5],
        }
    )
    got = build_pair_dosage(_weights(), joint, fallback)
    row = got[(got["fips_home"] == "01005") & (got["fips_work"] == "01005")].iloc[0]
    assert row["commuter_car_miles"] == pytest.approx(0.70 * 4.5)
    assert row["dosage_source"] == "distance_driving_fallback"


def test_symmetric_exposure_separates_own_loop_and_alerted_origins():
    grid = pd.DataFrame(
        {
            "fips": ["01001", "01003", "01005"],
            "date": [D, D, D],
            "night_alert": [1, 1, 0],
        }
    )
    got = construct_symmetric_exposures(grid, build_pair_dosage(_weights(), _joint_distance()))
    got = got.set_index("fips")

    assert got.loc["01001", "own_driver_distance"] == pytest.approx(0.60 * 4.0)
    assert got.loc["01001", "cross_driver_distance"] == pytest.approx(0.25 * 10.0)
    assert got.loc["01003", "own_driver_distance"] == pytest.approx(0.40 * 6.0)
    assert got.loc["01005", "own_driver_distance"] == 0
    assert got.loc["01005", "cross_driver_distance"] == 0


def test_cross_exposure_adds_multiple_alerted_origins_but_never_self_loop():
    grid = pd.DataFrame(
        {
            "fips": ["01001", "01003", "01005"],
            "date": [D, D, D],
            "night_alert": [1, 1, 1],
        }
    )
    got = construct_symmetric_exposures(grid, build_pair_dosage(_weights(), _joint_distance()))
    row = got.set_index("fips").loc["01001"]
    assert row["cross_driver_distance"] == pytest.approx(0.25 * 10.0 + 0.15 * 20.0)
    assert row["own_driver_distance"] == pytest.approx(0.60 * 4.0)


def test_sparse_exposure_series_matches_dataframe_construction():
    grid = pd.DataFrame(
        {
            "fips": ["01001", "01003", "01005"],
            "date": [D, D, D],
            "night_alert": [1, 1, 0],
        }
    )
    pairs = build_pair_dosage(_weights(), _joint_distance())
    dense = construct_symmetric_exposures(grid, pairs)
    index = pd.MultiIndex.from_frame(grid[["fips", "date"]])
    own, cross = symmetric_exposure_series(
        index, grid.loc[grid["night_alert"].gt(0), ["fips", "date"]], pairs
    )
    np.testing.assert_allclose(own, dense["own_driver_distance"])
    np.testing.assert_allclose(cross, dense["cross_driver_distance"])


def test_analysis_universe_requires_a_real_commuting_self_loop():
    weights = pd.DataFrame(
        {
            "fips_home": ["01001", "01003", "09110"],
            "fips_work": ["01001", "01001", "09110"],
            "weight": [0.6, 0.2, 0.7],
        }
    )
    eligible, excluded = restrict_to_self_loop_counties(
        ["01001", "01003", "09001"], weights
    )
    assert eligible == ["01001"]
    assert excluded == ["01003", "09001"]


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2013, ("2015", 2013)),
        (2015, ("2015", 2013)),
        (2016, ("2015", 2018)),
        (2017, ("2015", 2018)),
        (2018, ("2020", 2018)),
        (2020, ("2020", 2018)),
        (2021, ("2020", 2022)),
        (2024, ("2020", 2022)),
    ],
)
def test_exposure_vintage_bins(year, expected):
    assert exposure_vintage_for_year(year) == expected


def test_lodes_pooling_weights_match_nearest_vintage_year_bins():
    expected = {2013: 3 / 12, 2018: 5 / 12, 2022: 4 / 12}
    assert DISTANCE_YEAR_WEIGHTS == expected
    assert CAR_DISTANCE_YEAR_WEIGHTS == expected


def test_lodes_download_allows_large_archives(monkeypatch, tmp_path):
    import build_lodes_tract_car_dosage as builder

    calls = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    builder._download("https://example.test/large.csv.gz", tmp_path / "large.csv.gz")

    assert len(calls) == 1
    command, _ = calls[0]
    max_time = int(command[command.index("--max-time") + 1])
    assert max_time >= 600
    assert "--retry-all-errors" in command


@pytest.mark.parametrize(
    ("state", "target_year", "expected_source_year"),
    [
        ("al", 2018, 2018),
        ("ak", 2018, 2016),
        ("ak", 2022, 2016),
        ("mi", 2022, 2021),
    ],
)
def test_lodes_source_year_uses_documented_state_fallbacks(
    state, target_year, expected_source_year
):
    from build_lodes_tract_car_dosage import source_year_for_state

    assert source_year_for_state(state, target_year) == expected_source_year


def test_lodes_source_failure_does_not_create_empty_checkpoint(monkeypatch, tmp_path):
    import build_lodes_tract_car_dosage as builder

    checkpoint = tmp_path / "failed.parquet"

    def fail_download(*args, **kwargs):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(builder, "_download", fail_download)
    with pytest.raises(RuntimeError, match="source unavailable"):
        builder.process_one_file(
            "https://example.test/missing.csv.gz",
            checkpoint,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            0.8,
        )

    assert not checkpoint.exists()


def test_lodes_fallback_label_records_state_source_years():
    from run_symmetric_commuter_fatigue import lodes_fallback_label

    assert lodes_fallback_label(2013) == ""
    assert lodes_fallback_label(2018) == "AK:2016"
    assert lodes_fallback_label(2022) == "AK:2016;MI:2021"


def test_year_matched_exposure_uses_each_periods_pair_dosage():
    dates = pd.to_datetime(["2015-06-01", "2016-06-01", "2018-06-01", "2021-06-01"])
    index = pd.MultiIndex.from_product([["01001"], dates], names=["fips", "date"])
    alerts = pd.DataFrame({"fips": ["01003"] * 4, "date": dates})

    def pairs(cross_value):
        return pd.DataFrame(
            {
                "fips_home": ["01001", "01003", "01003"],
                "fips_work": ["01001", "01003", "01001"],
                "commuter_car_miles": [2.0, 3.0, cross_value],
            }
        )

    pair_dosages = {
        ("2015", 2013): pairs(1.0),
        ("2015", 2018): pairs(2.0),
        ("2020", 2018): pairs(3.0),
        ("2020", 2022): pairs(4.0),
    }
    own, cross = construct_year_matched_exposure_series(index, alerts, pair_dosages)
    np.testing.assert_allclose(own, np.zeros(4))
    np.testing.assert_allclose(cross, [1.0, 2.0, 3.0, 4.0])


def test_linear_time_outcome_recovers_known_hourly_slope_and_total():
    rows = []
    for hour in range(6, 24):
        rows.append(
            {
                "fips": "01001",
                "date": D,
                "hour": hour,
                "person_fatals": 2.0 + 3.0 * (hour - 6),
            }
        )
    got = build_linear_time_outcomes(pd.DataFrame(rows)).iloc[0]
    assert got["fatals_0623"] == pytest.approx(sum(2.0 + 3.0 * h for h in range(18)))
    assert got["fatals_hours_awake_slope"] == pytest.approx(3.0)


def test_linear_time_outcome_inserts_missing_hour_zeros():
    hourly = pd.DataFrame(
        {
            "fips": ["01001", "01001"],
            "date": [D, D],
            "hour": [6, 23],
            "person_fatals": [1.0, 2.0],
        }
    )
    got = build_linear_time_outcomes(hourly).iloc[0]
    x = np.arange(18, dtype=float)
    weights = (x - x.mean()) / np.square(x - x.mean()).sum()
    assert got["fatals_0623"] == 3.0
    assert got["fatals_hours_awake_slope"] == pytest.approx(weights[0] + 2 * weights[-1])


def test_within_ols_matches_pyfixest_for_three_fixed_effects():
    rng = np.random.default_rng(12)
    n = 240
    data = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "fe1": np.repeat(np.arange(12), 20),
            "fe2": np.tile(np.repeat(np.arange(5), 4), 12),
            "fe3": np.tile(np.arange(4), 60),
        }
    )
    data["y"] = 1.5 * data["x1"] - 0.7 * data["x2"] + rng.normal(scale=0.3, size=n)
    expected = pf.feols("y ~ x1 + x2 | fe1 + fe2 + fe3", data=data).coef().to_numpy()
    got = fit_within_ols(
        data["y"].to_numpy(),
        data[["x1", "x2"]].to_numpy(),
        [data[c].to_numpy() for c in ("fe1", "fe2", "fe3")],
    )
    np.testing.assert_allclose(got["beta"], expected, atol=1e-8)


def test_fast_wild_cluster_bootstrap_is_seeded_and_bounded():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(12), 20)
    x = rng.normal(size=(len(clusters), 2))
    y = 0.8 * x[:, 0] + rng.normal(size=len(clusters))
    first = wild_cluster_bootstrap(y, x, clusters, np.array([1.0, 0.0]), reps=199, seed=9)
    second = wild_cluster_bootstrap(y, x, clusters, np.array([1.0, 0.0]), reps=199, seed=9)
    assert 0 <= first["pval"] <= 1
    assert first == second
