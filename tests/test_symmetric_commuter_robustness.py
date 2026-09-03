import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_state_date_ids_are_shared_within_state_date_and_unique_across_keys():
    from run_symmetric_commuter_robustness import build_state_date_ids

    fips = pd.Series(["01001", "01003", "02013", "01001"])
    dates = pd.Series(pd.to_datetime([
        "2020-06-01", "2020-06-01", "2020-06-01", "2020-06-02"
    ]))

    got = build_state_date_ids(fips, dates)

    assert got.tolist() == [0, 0, 2, 1]
    assert len(np.unique(got)) == 3


def test_analytic_fit_accepts_and_records_custom_fixed_effects(monkeypatch):
    import run_symmetric_commuter_robustness as robustness

    counties = np.repeat(np.arange(4), 8)
    dates = np.tile(np.arange(8), 4)
    states = np.repeat([0, 0, 1, 1], 8)
    x = ((counties * 5 + dates * 3) % 11).astype(float)
    data = pd.DataFrame(
        {
            "y": 0.75 * x + (counties % 2) - (dates % 3),
            "x": x,
            "county_id": counties,
            "state_date_id": states * 100 + dates,
            "state_cluster_id": states,
            "date_cluster_id": dates,
            "state_month_cluster_id": states,
        }
    )
    monkeypatch.setattr(robustness, "pf", None)

    rows = robustness._fit_analytic(
        data,
        "y",
        ["x"],
        spec="custom_fe",
        fixed_effect_cols=("county_id", "state_date_id"),
    )

    assert rows[0]["status"] == "ok"
    assert rows[0]["fixed_effects"] == "county_id + state_date_id"


def test_state_date_models_estimate_both_outcomes_with_joint_exposures(monkeypatch):
    import run_symmetric_commuter_robustness as robustness

    class UnexpectedHighLevelEstimator:
        @staticmethod
        def feols(*args, **kwargs):
            raise AssertionError("large state-date models must use the within-OLS path")

    county = np.repeat(np.arange(12), 10)
    date = np.tile(np.arange(10), 12)
    state = np.repeat(np.repeat(np.arange(4), 3), 10)
    rng = np.random.default_rng(31)
    own = rng.normal(size=len(county))
    cross = rng.normal(size=len(county))
    panel = pd.DataFrame(
        {
            "fatal_crashes": 0.2 * own + 0.4 * cross + (county % 3),
            "total_fatals": 0.3 * own + 0.6 * cross - (date % 2),
            "own_driver_distance": own,
            "cross_driver_distance": cross,
            "fips_year_id": county,
            "fips_dow_id": county * 7 + date % 7,
            "state_date_id": state * 100 + date,
            "state_cluster_id": state,
            "date_cluster_id": date,
            "state_month_cluster_id": state,
        }
    )
    monkeypatch.setattr(robustness, "pf", UnexpectedHighLevelEstimator())

    rows = robustness.run_state_date_models(panel, bootstrap_reps=0, seed=17)

    assert len(rows) == 4
    assert {row["outcome"] for row in rows} == {"fatal_crashes", "total_fatals"}
    assert {row["term"] for row in rows} == {
        "own_driver_distance", "cross_driver_distance"
    }
    assert {row["fixed_effects"] for row in rows} == {
        "fips_year_id + fips_dow_id + state_date_id"
    }


def test_network_permutation_preserves_degrees_self_loops_and_edge_dosages():
    from run_symmetric_commuter_robustness import permute_cross_destinations

    pair = pd.DataFrame(
        {
            "fips_home": [
                "01001", "01003", "01005", "01007",
                "01001", "01001", "01003", "01003",
                "01005", "01005", "01007", "01007",
            ],
            "fips_work": [
                "01001", "01003", "01005", "01007",
                "01003", "01005", "01005", "01007",
                "01007", "01001", "01001", "01003",
            ],
            "commuter_car_miles": np.arange(1.0, 13.0),
            "dosage_source": ["tract_preserved"] * 12,
        }
    )

    got = permute_cross_destinations(pair, np.random.default_rng(41), swaps_per_edge=8)
    repeated = permute_cross_destinations(pair, np.random.default_rng(41), swaps_per_edge=8)

    original_self = pair[pair["fips_home"] == pair["fips_work"]].reset_index(drop=True)
    got_self = got[got["fips_home"] == got["fips_work"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(got_self, original_self)
    pd.testing.assert_frame_equal(got, repeated)

    original_cross = pair[pair["fips_home"] != pair["fips_work"]]
    got_cross = got[got["fips_home"] != got["fips_work"]]
    assert len(got_cross) == len(original_cross)
    assert not got_cross.duplicated(["fips_home", "fips_work"]).any()
    assert set(zip(got_cross["fips_home"], got_cross["fips_work"])) != set(
        zip(original_cross["fips_home"], original_cross["fips_work"])
    )
    pd.testing.assert_series_equal(
        got_cross.groupby("fips_home").size().sort_index(),
        original_cross.groupby("fips_home").size().sort_index(),
    )
    pd.testing.assert_series_equal(
        got_cross.groupby("fips_work").size().sort_index(),
        original_cross.groupby("fips_work").size().sort_index(),
    )
    assert sorted(got_cross["commuter_car_miles"]) == sorted(
        original_cross["commuter_car_miles"]
    )


def test_randomization_pvalue_uses_finite_sample_two_sided_rank():
    from run_symmetric_commuter_robustness import randomization_pvalue

    assert randomization_pvalue(2.0, np.array([-3.0, 0.5, 1.0])) == 0.5


def test_randomization_tail_ranks_preserve_the_direction_of_the_falsification():
    from run_symmetric_commuter_robustness import randomization_tail_ranks

    got = randomization_tail_ranks(2.0, np.array([-3.0, 0.5, 1.0]))

    assert got == {"upper_pval": 0.25, "lower_pval": 1.0, "percentile": 1.0}


def test_holm_adjustment_is_monotone_and_restores_original_order():
    from run_symmetric_commuter_robustness import holm_adjust

    pvalues = pd.Series([0.04, 0.01, 0.03], index=["late", "morning", "evening"])

    got = holm_adjust(pvalues)

    assert got.to_dict() == {"late": 0.06, "morning": 0.03, "evening": 0.06}


def test_network_placebo_runner_returns_seeded_draws_and_finite_rank():
    import run_symmetric_commuter_fatigue as base
    import run_symmetric_commuter_robustness as robustness

    counties = ["01001", "01003", "01005", "01007"]
    dates = pd.date_range("2020-06-01", periods=14, freq="D")
    grid_index = pd.MultiIndex.from_product([counties, dates], names=["fips", "date"])
    pair = pd.DataFrame(
        {
            "fips_home": [
                "01001", "01003", "01005", "01007",
                "01001", "01001", "01003", "01003",
                "01005", "01005", "01007", "01007",
            ],
            "fips_work": [
                "01001", "01003", "01005", "01007",
                "01003", "01005", "01005", "01007",
                "01007", "01001", "01001", "01003",
            ],
            "commuter_car_miles": np.arange(1.0, 13.0),
        }
    )
    alerts = pd.DataFrame(
        {"fips": ["01001", "01003", "01005"], "date": dates[[1, 5, 10]]}
    )
    own, cross = base.construct_year_matched_exposure_series(
        grid_index, alerts, {("2020", 2018): pair}
    )
    county_id = np.repeat(np.arange(len(counties)), len(dates))
    date_id = np.tile(np.arange(len(dates)), len(counties))
    panel = pd.DataFrame(
        {
            "fatal_crashes": 0.1 * own + 0.3 * cross + (county_id % 2),
            "total_fatals": 0.2 * own + 0.5 * cross + (date_id % 3),
            "own_driver_distance": own,
            "cross_driver_distance": cross,
            "fips_year_id": county_id,
            "fips_dow_id": county_id * 7 + date_id % 7,
            "state_date_id": date_id,
        }
    )
    metadata = {"grid_index": grid_index, "pair_dosages": {("2020", 2018): pair}}

    distribution, summary = robustness.run_network_placebos(
        panel, alerts, metadata, draws=3, seed=23, swaps_per_edge=4
    )

    assert len(distribution) == 6
    assert distribution.groupby("outcome")["draw"].nunique().to_dict() == {
        "fatal_crashes": 3,
        "total_fatals": 3,
    }
    assert distribution["seed"].nunique() == 3
    assert {row["spec"] for row in summary} == {"observed_vs_placebo_network"}
    assert all(0 < row["pval_network"] <= 1 for row in summary)
    assert all(row["completed_draws"] == 3 for row in summary)


def test_time_block_models_label_fixed_effects_and_adjust_only_block_family(monkeypatch):
    import run_symmetric_commuter_robustness as robustness

    class UnexpectedHighLevelEstimator:
        @staticmethod
        def feols(*args, **kwargs):
            raise AssertionError("large time-block models must use the within-OLS path")

    county = np.repeat(np.arange(12), 10)
    date = np.tile(np.arange(10), 12)
    state = np.repeat(np.repeat(np.arange(4), 3), 10)
    rng = np.random.default_rng(44)
    own = rng.normal(size=len(county))
    cross = rng.normal(size=len(county))
    panel = pd.DataFrame(
        {
            "own_driver_distance": own,
            "cross_driver_distance": cross,
            "fips_year_id": county,
            "fips_dow_id": county * 7 + date % 7,
            "month_id": date // 5,
            "state_date_id": state * 100 + date,
            "state_cluster_id": state,
            "date_cluster_id": date,
            "state_month_cluster_id": state * 10 + date // 5,
        }
    )
    for index, outcome in enumerate(
        [
            "fatals_avg_0609",
            "fatals_avg_1014",
            "fatals_avg_1519",
            "fatals_avg_2023",
            "fatals_late_minus_morning",
        ]
    ):
        panel[outcome] = (index + 1) * 0.1 * own + 0.2 * cross + rng.normal(
            scale=0.2, size=len(panel)
        )
    monkeypatch.setattr(robustness, "pf", UnexpectedHighLevelEstimator())

    rows = robustness.run_time_block_models(panel, bootstrap_reps=0, seed=51)
    result = pd.DataFrame(rows)

    assert len(result) == 20
    assert set(result["spec"]) == {"baseline_time_blocks", "state_date_time_blocks"}
    contrast = result["outcome"].eq("fatals_late_minus_morning")
    assert result.loc[contrast, "pval_holm"].isna().all()
    finite_block = ~contrast & result["pval_state_date"].notna()
    assert result.loc[finite_block, "pval_holm"].notna().all()
    assert set(result.loc[~contrast, "multiplicity_family"]) == {
        "baseline_time_blocks:own_driver_distance",
        "baseline_time_blocks:cross_driver_distance",
        "state_date_time_blocks:own_driver_distance",
        "state_date_time_blocks:cross_driver_distance",
    }


def test_event_time_bin_labels_cover_leads_and_fatigue_windows():
    from run_symmetric_commuter_robustness import event_time_bin

    assert event_time_bin(-2) == "lead_2"
    assert event_time_bin(-1) == "lead_1"
    assert event_time_bin(0) == "post_0_2"
    assert event_time_bin(2) == "post_0_2"
    assert event_time_bin(3) == "post_3_5"
    assert event_time_bin(7) == "post_6_8"
    assert event_time_bin(11) == "post_9_12"
    assert event_time_bin(15) == "post_13_18"
    assert event_time_bin(19) is None


def test_daily_lagged_exposures_shift_alert_dates_and_preserve_own_cross_columns():
    from run_symmetric_commuter_robustness import build_daily_lagged_exposures

    dates = pd.date_range("2020-06-01", periods=4, freq="D")
    index = pd.MultiIndex.from_product([["01001"], dates], names=["fips", "date"])
    alerts = pd.DataFrame({"fips": ["01003"], "date": [dates[1]]})
    pair = pd.DataFrame(
        {
            "fips_home": ["01003", "01001", "01003"],
            "fips_work": ["01001", "01001", "01003"],
            "commuter_car_miles": [4.0, 2.0, 3.0],
        }
    )
    out = build_daily_lagged_exposures(index, alerts, {("2020", 2018): pair}, lags=(-1, 0, 1))

    assert list(out.columns) == [
        "own_lag_m1", "cross_lag_m1", "own_lag_0", "cross_lag_0", "own_lag_p1", "cross_lag_p1"
    ]
    # The alert on June 2 reaches the destination on June 2, with the
    # adjacent lead/lag exposures on June 1 and June 3.
    np.testing.assert_allclose(out["cross_lag_m1"].to_numpy(), [4.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out["cross_lag_0"].to_numpy(), [0.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(out["cross_lag_p1"].to_numpy(), [0.0, 0.0, 4.0, 0.0])
    assert (out.filter(like="own_") == 0).all().all()


def test_event_bin_exposures_sum_offsets_into_named_windows():
    from run_symmetric_commuter_robustness import build_daily_event_bin_exposures

    dates = pd.date_range("2020-06-01", periods=20, freq="D")
    index = pd.MultiIndex.from_product([["01001"], dates], names=["fips", "date"])
    alerts = pd.DataFrame({"fips": ["01003"], "date": [dates[0]]})
    pair = pd.DataFrame(
        {
            "fips_home": ["01003", "01003"],
            "fips_work": ["01001", "01003"],
            "commuter_car_miles": [4.0, 1.0],
        }
    )
    out = build_daily_event_bin_exposures(index, alerts, {("2020", 2018): pair})
    assert out.loc[("01001", dates[0]), "cross_post_0_2"] == 4.0
    assert out.loc[("01001", dates[2]), "cross_post_0_2"] == 4.0
    assert out.loc[("01001", dates[3]), "cross_post_3_5"] == 4.0
    assert out.loc[("01001", dates[13]), "cross_post_13_18"] == 4.0
    assert out.filter(like="own_").to_numpy().sum() == 0.0


def test_fatal_crash_outcome_aggregates_sparse_hourly_counts():
    from run_symmetric_commuter_robustness import build_daily_fatal_crash_outcomes

    hourly = pd.DataFrame(
        {
            "fips": ["01001", "01001", "01001"],
            "date": pd.to_datetime(["2020-06-01"] * 3),
            "hour": [6, 7, 8],
            "fatal_crashes": [1, 0, 2],
        }
    )
    out = build_daily_fatal_crash_outcomes(hourly)
    assert out.iloc[0]["fatal_crashes"] == 3


def test_fatal_crash_outcome_uses_the_existing_06_23_window():
    from run_symmetric_commuter_robustness import build_daily_fatal_crash_outcomes

    hourly = pd.DataFrame(
        {
            "fips": ["01001", "01001"],
            "date": pd.to_datetime(["2020-06-01"] * 2),
            "hour": [2, 6],
            "fatal_crashes": [9, 1],
        }
    )
    out = build_daily_fatal_crash_outcomes(hourly)
    assert out.iloc[0]["fatal_crashes"] == 1


def test_webb_weights_use_six_point_unit_variance_distribution():
    from run_symmetric_commuter_robustness import draw_wild_weights

    rng = np.random.default_rng(4)
    draws = draw_wild_weights("webb", 200_000, rng)
    expected = np.array([-np.sqrt(3 / 2), -1.0, -np.sqrt(1 / 2), np.sqrt(1 / 2), 1.0, np.sqrt(3 / 2)])
    assert set(np.unique(draws)).issubset(set(expected))
    assert abs(draws.mean()) < 0.01
    assert abs(draws.var() - 1.0) < 0.01


def test_exposure_bins_are_zero_safe_and_monotone_for_positive_values():
    from run_symmetric_commuter_robustness import add_exposure_bins

    data = pd.DataFrame({"own_driver_distance": [0.0, 1.0, 2.0, 10.0], "cross_driver_distance": [0.0, 3.0, 4.0, 20.0]})
    out, bins = add_exposure_bins(data)
    assert bins == ["own_bin_0", "own_bin_pos", "cross_bin_0", "cross_bin_pos"]
    assert out.loc[0, "own_bin_0"] == 1
    assert out.loc[0, "cross_bin_0"] == 1
    assert out.loc[3, "own_bin_pos"] == 1
    assert out.loc[3, "cross_bin_pos"] == 1


def test_positive_exposure_quantile_bins_keep_structural_zero_separate():
    from run_symmetric_commuter_robustness import add_positive_quantile_bins

    data = pd.DataFrame(
        {
            "own_driver_distance": [0.0, 1.0, 2.0, 3.0, 10.0],
            "cross_driver_distance": [0.0, 2.0, 4.0, 8.0, 20.0],
        }
    )
    out, terms = add_positive_quantile_bins(data, n_bins=2)
    assert terms == ["own_q1", "own_q2", "cross_q1", "cross_q2"]
    assert out.loc[0, terms].sum() == 0
    assert out.loc[4, ["own_q2", "cross_q2"]].tolist() == [1, 1]


def test_positive_tail_trim_keeps_zeros_and_trims_only_positive_extremes():
    from run_symmetric_commuter_robustness import keep_below_positive_tail

    values = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0])
    kept = keep_below_positive_tail(values, quantile=0.75)
    assert kept.tolist() == [True, True, True, True, False]
