"""Joint own- and cross-county commuter-fatigue analysis.

The two alert exposures use the same unit throughout:

    destination workforce share x E[car commute share x trip miles]

The own exposure is the self-loop (home county equals work county). The
cross exposure sums all alerted home counties other than the work county.
Both enter each regression jointly.

Two outcomes are estimated from FARS hours 06:00--23:59:

* the total number of fatalities in the window; and
* a pre-specified linear hours-since-06:00 contrast.  Its weights sum to
  zero and are normalized so a truly linear hourly profile ``a + b*h``
  yields exactly ``b``.  This gives one covariance-aware slope regression
  instead of a meta-regression over 18 correlated hourly regressions.

Analytic standard errors are two-way clustered by state and calendar date.
Because pyfixest's wild cluster bootstrap is one-way, the supplemental
null-imposed Rademacher bootstrap clusters by state only.

Outputs
-------
output/tables/reg_symmetric_commuter_fatigue_year_matched.csv
output/tables/symmetric_commuter_fatigue_year_matched_exposure_summary.csv
output/tables/symmetric_commuter_fatigue_vintage_summary.csv
output/tables/symmetric_commuter_fatigue_time_weights.csv
"""

from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pyfixest as pf
except ModuleNotFoundError:  # pure construction helpers remain importable
    pf = None
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from build_lodes_tract_car_dosage import STATE_YEAR_FALLBACKS
from build_nhts_car_share_by_distance import car_share_from_distance_by_county
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("symmetric_commuter_fatigue")

HOURS = tuple(range(6, 24))
WAKE_HOUR = 6
TIME_BLOCKS = {
    "fatals_avg_0609": tuple(range(6, 10)),
    "fatals_avg_1014": tuple(range(10, 15)),
    "fatals_avg_1519": tuple(range(15, 20)),
    "fatals_avg_2023": tuple(range(20, 24)),
}
FE = "fips_year_id + fips_dow_id + month_id"
OWN_EXPOSURE = "own_driver_distance"
CROSS_EXPOSURE = "cross_driver_distance"
PAIR_DOSAGE_YEAR_DIR = DATA_PROC / "commuting" / "_lodes_car_year_cache"
FLOW_VINTAGE_PATHS = {
    "2015": DATA_PROC / "commuting" / "county_commuting_weights_2015.parquet",
    "2020": DATA_PROC / "commuting" / "county_commuting_weights_2020.parquet",
}
FLOW_VINTAGE_FOR_YEAR = {year: ("2015" if year <= 2017 else "2020") for year in range(2013, 2025)}
LODES_VINTAGE_FOR_YEAR = {
    **{year: 2013 for year in range(2013, 2016)},
    **{year: 2018 for year in range(2016, 2021)},
    **{year: 2022 for year in range(2021, 2025)},
}
BOOTSTRAP_REPS = 9_999
BOOTSTRAP_SEED = 20260828


def lodes_fallback_label(target_year: int) -> str:
    """Return compact state:source-year provenance for a target vintage."""
    entries = sorted(
        (state.upper(), source_year)
        for (state, year), source_year in STATE_YEAR_FALLBACKS.items()
        if year == target_year
    )
    return ";".join(f"{state}:{source_year}" for state, source_year in entries)


def _fips(series: pd.Series) -> pd.Series:
    """Normalize numeric or string county identifiers to five characters."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def linear_time_weights() -> pd.DataFrame:
    """Return normalized weights for a linear hours-since-wake contrast."""
    hours_since_wake = np.arange(len(HOURS), dtype=float)
    centered = hours_since_wake - hours_since_wake.mean()
    weights = centered / np.square(centered).sum()
    return pd.DataFrame(
        {
            "hour": HOURS,
            "hours_since_wake": hours_since_wake.astype(int),
            "contrast_weight": weights,
        }
    )


def build_linear_time_outcomes(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse sparse hourly fatalities to a total and a linear contrast.

    Absent county-date-hour rows represent zero FARS fatalities. They need
    not be explicitly expanded because their contribution to both sums is
    zero. County-dates with no fatality row at any hour are filled with zero
    after this sparse result is merged onto the balanced analysis grid.
    """
    required = {"fips", "date", "hour", "person_fatals"}
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"hourly fatalities missing columns: {sorted(missing)}")

    data = hourly.loc[hourly["hour"].isin(HOURS), list(required)].copy()
    data["fips"] = _fips(data["fips"])
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["person_fatals"] = pd.to_numeric(data["person_fatals"], errors="raise")
    weights = linear_time_weights().set_index("hour")["contrast_weight"]
    data["weighted_fatals"] = data["person_fatals"] * data["hour"].map(weights)
    return (
        data.groupby(["fips", "date"], as_index=False)
        .agg(
            fatals_0623=("person_fatals", "sum"),
            fatals_hours_awake_slope=("weighted_fatals", "sum"),
        )
    )


def build_time_block_outcomes(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse sparse hourly fatalities into pre-specified average-hour blocks."""
    required = {"fips", "date", "hour", "person_fatals"}
    if missing := required - set(hourly.columns):
        raise ValueError(f"hourly fatalities missing columns: {sorted(missing)}")

    data = hourly.loc[hourly["hour"].isin(HOURS), list(required)].copy()
    data["fips"] = _fips(data["fips"])
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["person_fatals"] = pd.to_numeric(data["person_fatals"], errors="raise")
    index = data[["fips", "date"]].drop_duplicates().set_index(["fips", "date"])

    for outcome, hours in TIME_BLOCKS.items():
        totals = (
            data.loc[data["hour"].isin(hours)]
            .groupby(["fips", "date"])["person_fatals"]
            .sum()
        )
        index[outcome] = totals.reindex(index.index).fillna(0.0) / len(hours)
    index["fatals_late_minus_morning"] = (
        index["fatals_avg_2023"] - index["fatals_avg_0609"]
    )
    return index.reset_index()


def build_pair_dosage(
    weights: pd.DataFrame,
    joint_distance: pd.DataFrame,
    fallback_car_distance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach tract-preserved car-distance dosage to every commuting edge.

    Missing joint values fail closed unless an explicit pair-level fallback
    is supplied. The production analysis supplies the same documented
    distance-based driving fallback to own and cross edges and reports which
    source each edge used.
    """
    weight_cols = {"fips_home", "fips_work", "weight"}
    joint_cols = {"fips_home", "fips_work", "avg_car_x_dist"}
    if missing := weight_cols - set(weights.columns):
        raise ValueError(f"commuting weights missing columns: {sorted(missing)}")
    if missing := joint_cols - set(joint_distance.columns):
        raise ValueError(f"joint car-distance data missing columns: {sorted(missing)}")

    w = weights.loc[:, list(weight_cols)].copy()
    j = joint_distance.loc[:, list(joint_cols)].copy()
    for frame in (w, j):
        frame["fips_home"] = _fips(frame["fips_home"])
        frame["fips_work"] = _fips(frame["fips_work"])

    if w.duplicated(["fips_home", "fips_work"]).any():
        raise ValueError("commuting weights contain duplicate county pairs")
    if j.duplicated(["fips_home", "fips_work"]).any():
        raise ValueError("joint car-distance data contain duplicate county pairs")

    pairs = w.merge(
        j,
        on=["fips_home", "fips_work"],
        how="left",
        validate="one_to_one",
    )
    missing_joint = pairs["avg_car_x_dist"].isna()
    pairs["dosage_source"] = np.where(missing_joint, "missing", "tract_preserved")
    if missing_joint.any() and fallback_car_distance is not None:
        fallback_cols = {"fips_home", "fips_work", "fallback_car_x_dist"}
        if missing := fallback_cols - set(fallback_car_distance.columns):
            raise ValueError(f"fallback car-distance data missing columns: {sorted(missing)}")
        fallback = fallback_car_distance.loc[:, list(fallback_cols)].copy()
        fallback["fips_home"] = _fips(fallback["fips_home"])
        fallback["fips_work"] = _fips(fallback["fips_work"])
        if fallback.duplicated(["fips_home", "fips_work"]).any():
            raise ValueError("fallback car-distance data contain duplicate county pairs")
        pairs = pairs.merge(
            fallback,
            on=["fips_home", "fips_work"],
            how="left",
            validate="one_to_one",
        )
        filled = pairs["avg_car_x_dist"].isna() & pairs["fallback_car_x_dist"].notna()
        pairs.loc[filled, "avg_car_x_dist"] = pairs.loc[filled, "fallback_car_x_dist"]
        pairs.loc[filled, "dosage_source"] = "distance_driving_fallback"
        pairs = pairs.drop(columns="fallback_car_x_dist")
        missing_joint = pairs["avg_car_x_dist"].isna()
    if missing_joint.any():
        examples = pairs.loc[missing_joint, ["fips_home", "fips_work"]].head(5)
        raise ValueError(
            f"missing tract-preserved car-distance dosage for {int(missing_joint.sum())} "
            f"commuting pairs; examples={examples.to_dict('records')}"
        )

    pairs["weight"] = pd.to_numeric(pairs["weight"], errors="raise")
    pairs["avg_car_x_dist"] = pd.to_numeric(pairs["avg_car_x_dist"], errors="raise")
    if (pairs[["weight", "avg_car_x_dist"]] < 0).any().any():
        raise ValueError("commuting weights and car-distance dosage must be nonnegative")
    pairs["commuter_car_miles"] = pairs["weight"] * pairs["avg_car_x_dist"]
    return pairs


def build_distance_driving_fallback(
    weights: pd.DataFrame,
    lodes_distance: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Build the established NHTS driving-share x distance fallback.

    LODES distance is preferred. For uncovered cross-county edges, distance
    between population-weighted county centroids is used. The rare uncovered
    self-loop receives the median observed within-county LODES distance,
    because a county's distance to its own centroid is mechanically zero.
    """
    w = weights[["fips_home", "fips_work"]].copy()
    d = lodes_distance[["fips_home", "fips_work", "avg_dist_mi"]].copy()
    c = centroids[["fips", "lat", "lon"]].copy()
    for frame in (w, d):
        frame["fips_home"] = _fips(frame["fips_home"])
        frame["fips_work"] = _fips(frame["fips_work"])
    c["fips"] = _fips(c["fips"])
    out = w.merge(d, on=["fips_home", "fips_work"], how="left", validate="one_to_one")

    home = c.rename(columns={"fips": "fips_home", "lat": "lat_home", "lon": "lon_home"})
    work = c.rename(columns={"fips": "fips_work", "lat": "lat_work", "lon": "lon_work"})
    out = out.merge(home, on="fips_home", how="left", validate="many_to_one")
    out = out.merge(work, on="fips_work", how="left", validate="many_to_one")
    lat1 = np.radians(out["lat_home"])
    lat2 = np.radians(out["lat_work"])
    dlat = lat2 - lat1
    dlon = np.radians(out["lon_work"] - out["lon_home"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    out["centroid_dist_mi"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    out["dist_mi"] = out["avg_dist_mi"].fillna(out["centroid_dist_mi"])

    is_self = out["fips_home"] == out["fips_work"]
    observed_self_median = float(out.loc[is_self, "avg_dist_mi"].median())
    out.loc[is_self & out["avg_dist_mi"].isna(), "dist_mi"] = observed_self_median
    if out["dist_mi"].isna().any():
        n_missing = int(out["dist_mi"].isna().sum())
        raise ValueError(f"cannot construct fallback distance for {n_missing} commuting pairs")

    car_share = car_share_from_distance_by_county(out["dist_mi"], out["fips_home"])
    out["fallback_car_x_dist"] = car_share * out["dist_mi"]
    return out[["fips_home", "fips_work", "fallback_car_x_dist"]]


def restrict_to_self_loop_counties(
    candidate_counties: list[str],
    weights: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Coverage-match the outcome universe to counties with own-flow dosage."""
    normalized = weights[["fips_home", "fips_work"]].copy()
    normalized["fips_home"] = _fips(normalized["fips_home"])
    normalized["fips_work"] = _fips(normalized["fips_work"])
    self_fips = set(
        normalized.loc[
            normalized["fips_home"].eq(normalized["fips_work"]), "fips_home"
        ]
    )
    candidates = sorted(str(fips).zfill(5) for fips in candidate_counties)
    eligible = [fips for fips in candidates if fips in self_fips]
    excluded = [fips for fips in candidates if fips not in self_fips]
    return eligible, excluded


def construct_symmetric_exposures(
    grid: pd.DataFrame,
    pair_dosage: pd.DataFrame,
    *,
    alert_col: str = "night_alert",
) -> pd.DataFrame:
    """Add own and cross alert exposure in destination commuter-car-miles."""
    required = {"fips", "date", alert_col}
    if missing := required - set(grid.columns):
        raise ValueError(f"analysis grid missing columns: {sorted(missing)}")
    if grid.duplicated(["fips", "date"]).any():
        raise ValueError("analysis grid must have one row per county-date")

    out = grid.copy()
    out["fips"] = _fips(out["fips"])
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    own_values, cross_values = symmetric_exposure_series(
        pd.MultiIndex.from_frame(out[["fips", "date"]]),
        out.loc[out[alert_col].gt(0), ["fips", "date"]],
        pair_dosage,
    )
    out[OWN_EXPOSURE] = own_values
    out[CROSS_EXPOSURE] = cross_values
    return out


def symmetric_exposure_series(
    grid_index: pd.MultiIndex,
    alert_events: pd.DataFrame,
    pair_dosage: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct aligned own and cross exposure arrays without a dense merge."""
    if list(grid_index.names) != ["fips", "date"]:
        raise ValueError("grid_index levels must be named ['fips', 'date']")
    alerts = alert_events[["fips", "date"]].drop_duplicates().copy()
    alerts["fips"] = _fips(alerts["fips"])
    alerts["date"] = pd.to_datetime(alerts["date"]).dt.normalize()

    self_loops = pair_dosage[
        pair_dosage["fips_home"] == pair_dosage["fips_work"]
    ][["fips_work", "commuter_car_miles"]].rename(
        columns={"fips_work": "fips", "commuter_car_miles": "own_driver_distance_base"}
    )
    if self_loops.duplicated("fips").any():
        raise ValueError("pair dosage contains duplicate self-loops")
    own = alerts.merge(self_loops, on="fips", how="left", validate="many_to_one")
    missing_self = own.loc[own["own_driver_distance_base"].isna(), "fips"].unique()
    if len(missing_self):
        raise ValueError(
            "missing tract-preserved own-county dosage for alerted counties: "
            + ", ".join(missing_self[:10])
        )
    own_series = own.set_index(["fips", "date"])["own_driver_distance_base"]
    own_values = own_series.reindex(grid_index).fillna(0.0).to_numpy()

    alerts = alerts.rename(columns={"fips": "fips_home"})
    cross_pairs = pair_dosage[
        pair_dosage["fips_home"] != pair_dosage["fips_work"]
    ][["fips_home", "fips_work", "commuter_car_miles"]]
    lit = alerts.merge(cross_pairs, on="fips_home", how="inner", validate="many_to_many")
    cross = (
        lit.groupby(["fips_work", "date"], as_index=False)["commuter_car_miles"]
        .sum()
        .rename(columns={"fips_work": "fips", "commuter_car_miles": CROSS_EXPOSURE})
    )
    cross_values = (
        cross.set_index(["fips", "date"])[CROSS_EXPOSURE]
        .reindex(grid_index)
        .fillna(0.0)
        .to_numpy()
    )
    return own_values, cross_values


def exposure_vintage_for_year(year: int) -> tuple[str, int]:
    """Return the ACS-flow and nearest LODES vintage for a crash year."""
    try:
        return FLOW_VINTAGE_FOR_YEAR[int(year)], LODES_VINTAGE_FOR_YEAR[int(year)]
    except KeyError as exc:
        raise ValueError(f"no exposure vintage mapping for crash year {year}") from exc


def construct_year_matched_exposure_series(
    grid_index: pd.MultiIndex,
    alert_events: pd.DataFrame,
    pair_dosages: dict[tuple[str, int], pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Build exposure arrays using the vintage pair assigned to each date."""
    dates = pd.DatetimeIndex(grid_index.get_level_values("date"))
    years = dates.year.to_numpy()
    own = np.zeros(len(grid_index), dtype=np.float64)
    cross = np.zeros(len(grid_index), dtype=np.float64)
    alerts = alert_events.copy()
    alerts["date"] = pd.to_datetime(alerts["date"]).dt.normalize()

    regimes: dict[tuple[str, int], list[int]] = {}
    for year in sorted(set(years)):
        regimes.setdefault(exposure_vintage_for_year(int(year)), []).append(int(year))

    missing = sorted(set(regimes) - set(pair_dosages))
    if missing:
        raise ValueError(f"missing year-matched pair dosage regimes: {missing}")

    for regime, regime_years in regimes.items():
        mask = np.isin(years, regime_years)
        sub_index = grid_index[mask]
        sub_alerts = alerts[alerts["date"].dt.year.isin(regime_years)]
        own_values, cross_values = symmetric_exposure_series(
            sub_index, sub_alerts, pair_dosages[regime]
        )
        own[mask] = own_values
        cross[mask] = cross_values
    return own, cross


def _within_transform(
    values: np.ndarray,
    fixed_effects: list[np.ndarray],
    *,
    tol: float = 1e-10,
    max_iter: int = 1_000,
) -> np.ndarray:
    """Residualize columns by multiple categorical fixed effects via MAP."""
    out = np.asarray(values, dtype=np.float64).copy()
    was_vector = out.ndim == 1
    if was_vector:
        out = out[:, None]
    groups = []
    counts = []
    for raw in fixed_effects:
        codes, _ = pd.factorize(raw, sort=True)
        if (codes < 0).any():
            raise ValueError("fixed effects cannot contain missing values")
        codes = codes.astype(np.int32, copy=False)
        groups.append(codes)
        counts.append(np.bincount(codes).astype(np.float64))

    for _ in range(max_iter):
        largest_adjustment = 0.0
        for codes, count in zip(groups, counts):
            means = np.empty((len(count), out.shape[1]), dtype=np.float64)
            for col in range(out.shape[1]):
                means[:, col] = np.bincount(codes, weights=out[:, col], minlength=len(count)) / count
            largest_adjustment = max(largest_adjustment, float(np.max(np.abs(means))))
            out -= means[codes]
        if largest_adjustment < tol:
            return out[:, 0] if was_vector else out
    raise RuntimeError(f"fixed-effect residualization did not converge in {max_iter} iterations")


def fit_within_ols(
    y: np.ndarray,
    x: np.ndarray,
    fixed_effects: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Fit OLS after jointly residualizing outcome and regressors."""
    transformed = _within_transform(np.column_stack([y, x]), fixed_effects)
    y_tilde = transformed[:, 0]
    x_tilde = transformed[:, 1:]
    xtx = x_tilde.T @ x_tilde
    bread = np.linalg.inv(xtx)
    beta = bread @ (x_tilde.T @ y_tilde)
    residual = y_tilde - x_tilde @ beta
    return {
        "beta": beta,
        "bread": bread,
        "y_tilde": y_tilde,
        "x_tilde": x_tilde,
        "residual": residual,
    }


def _cluster_sums(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    codes, uniques = pd.factorize(clusters, sort=True)
    result = np.empty((len(uniques), values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        result[:, col] = np.bincount(codes, weights=values[:, col], minlength=len(uniques))
    return result


def wild_cluster_bootstrap(
    y_tilde: np.ndarray,
    x_tilde: np.ndarray,
    clusters: np.ndarray,
    restriction: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    """Fast null-imposed Rademacher WCR11 bootstrap for one contrast.

    The inputs are already residualized with respect to all fixed effects.
    The calculation uses state-level score and cross-product sufficient
    statistics, so memory depends on clusters and regressors rather than on
    bootstrap draws times observations.
    """
    y = np.asarray(y_tilde, dtype=np.float64)
    x = np.asarray(x_tilde, dtype=np.float64)
    r = np.asarray(restriction, dtype=np.float64)
    cluster_codes, cluster_levels = pd.factorize(clusters, sort=True)
    n, k = x.shape
    g = len(cluster_levels)
    if g < 2:
        raise ValueError("wild cluster bootstrap requires at least two clusters")
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ (x.T @ y)
    unrestricted_residual = y - x @ beta

    direction = bread @ r
    restriction_variance = float(r @ direction)
    beta_restricted = beta - direction * (float(r @ beta) / restriction_variance)
    restricted_residual = y - x @ beta_restricted

    score_unrestricted = _cluster_sums(x * unrestricted_residual[:, None], cluster_codes)
    contrast_score = score_unrestricted @ direction
    ssc = (g / (g - 1)) * ((n - 1) / (n - k))
    observed_se = float(np.sqrt(ssc * np.square(contrast_score).sum()))
    observed_t = float((r @ beta) / observed_se)

    score_restricted = _cluster_sums(x * restricted_residual[:, None], cluster_codes)
    group_crossproducts = np.empty((g, k, k), dtype=np.float64)
    for left in range(k):
        for right in range(k):
            group_crossproducts[:, left, right] = np.bincount(
                cluster_codes,
                weights=x[:, left] * x[:, right],
                minlength=g,
            )

    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    chunk_size = min(reps, 1_000)
    q_direction = score_restricted @ direction
    direction_h = np.einsum("k,gkl->gl", direction, group_crossproducts)
    while completed < reps:
        size = min(chunk_size, reps - completed)
        wild_weights = rng.choice(np.array([-1.0, 1.0]), size=(g, size))
        delta = bread @ (score_restricted.T @ wild_weights)
        bootstrap_cluster_scores = (
            q_direction[:, None] * wild_weights - direction_h @ delta
        )
        bootstrap_se = np.sqrt(ssc * np.square(bootstrap_cluster_scores).sum(axis=0))
        bootstrap_t = (r @ delta) / bootstrap_se
        exceedances += int((np.abs(bootstrap_t) >= abs(observed_t)).sum())
        completed += size

    return {
        "estimate": float(r @ beta),
        "cluster_se": observed_se,
        "t_stat": observed_t,
        "pval": float((exceedances + 1) / (reps + 1)),
    }


def _fit_model(data: pd.DataFrame, outcome: str, *, reps: int, seed: int) -> list[dict]:
    if pf is None:
        raise ImportError("pyfixest is required for regression estimation")
    cols = [
        outcome,
        OWN_EXPOSURE,
        CROSS_EXPOSURE,
        "fips_year_id",
        "fips_dow_id",
        "month_id",
        "state_cluster_id",
        "date_cluster_id",
    ]
    sub = data.loc[:, cols].dropna()
    formula = f"{outcome} ~ {OWN_EXPOSURE} + {CROSS_EXPOSURE} | {FE}"
    model = pf.feols(
        formula,
        data=sub,
        vcov={"CRV1": "state_cluster_id + date_cluster_id"},
        copy_data=False,
        store_data=False,
        lean=True,
    )
    tidy = model.tidy()
    coef_names = [OWN_EXPOSURE, CROSS_EXPOSURE]
    beta_analytic = model.coef().loc[coef_names].to_numpy()
    vcov_analytic = model._vcov
    df_t = float(model._df_t)

    within = fit_within_ols(
        sub[outcome].to_numpy(),
        sub[coef_names].to_numpy(),
        [sub[c].to_numpy() for c in ("fips_year_id", "fips_dow_id", "month_id")],
    )
    np.testing.assert_allclose(within["beta"], beta_analytic, rtol=1e-6, atol=1e-10)
    restrictions = {
        "own": np.array([1.0, 0.0]),
        "cross": np.array([0.0, 1.0]),
        "own_minus_cross": np.array([1.0, -1.0]),
    }
    bootstrap = {
        name: wild_cluster_bootstrap(
            within["y_tilde"],
            within["x_tilde"],
            sub["state_cluster_id"].to_numpy(),
            restriction,
            reps=reps,
            seed=seed + offset,
        )
        for offset, (name, restriction) in enumerate(restrictions.items())
    }
    rows = []
    for estimand, param in (
        (("own", OWN_EXPOSURE), ("cross", CROSS_EXPOSURE))
    ):
        row = tidy.loc[param]
        rows.append(
            {
                "outcome": outcome,
                "estimand": estimand,
                "treatment": param,
                "coef": float(row["Estimate"]),
                "se_state_date": float(row["Std. Error"]),
                "pval_state_date": float(row["Pr(>|t|)"]),
                "pval_wild_state": bootstrap[estimand]["pval"],
                "nobs": int(model._N),
                "bootstrap_reps": reps,
            }
        )

    difference_r = restrictions["own_minus_cross"]
    difference_coef = float(difference_r @ beta_analytic)
    difference_se = float(np.sqrt(difference_r @ vcov_analytic @ difference_r))
    difference_pval = float(2 * stats.t.sf(abs(difference_coef / difference_se), df_t))
    rows.append(
        {
            "outcome": outcome,
            "estimand": "own_minus_cross",
            "treatment": OWN_EXPOSURE,
            "coef": difference_coef,
            "se_state_date": difference_se,
            "pval_state_date": difference_pval,
            "pval_wild_state": bootstrap["own_minus_cross"]["pval"],
            "nobs": int(model._N),
            "bootstrap_reps": reps,
        }
    )
    del model, sub, within
    gc.collect()
    return rows


def main(*, bootstrap_reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"])
    fatals_col = "total_fatals" if "total_fatals" in fars.columns else "fatals"
    mean_annual = (
        fars.assign(year=fars["date"].dt.year)
        .groupby(["fips", "year"])[fatals_col]
        .sum()
        .groupby("fips")
        .mean()
    )
    active_candidates = sorted(
        _fips(mean_annual[mean_annual >= ntm.MIN_FATALS_PER_YEAR].index.to_series())
    )
    flow_tables = {}
    for vintage, path in FLOW_VINTAGE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing ACS flow vintage {vintage}: {path}")
        flow = pd.read_parquet(path)
        for col in ("fips_home", "fips_work"):
            flow[col] = _fips(flow[col])
        flow_tables[vintage] = flow

    active = active_candidates
    excluded_no_self_loop = set()
    for vintage, flow in flow_tables.items():
        eligible, excluded = restrict_to_self_loop_counties(active, flow)
        excluded_no_self_loop.update(excluded)
        active = eligible
        log.info("ACS %s self-loop coverage: %d eligible counties", vintage, len(active))
    excluded_no_self_loop = sorted(excluded_no_self_loop)
    log.info(
        "Self-loop coverage restriction: %d eligible counties; %d excluded (%s)",
        len(active),
        len(excluded_no_self_loop),
        ", ".join(excluded_no_self_loop[:12]) or "none",
    )
    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")
    grid_index = pd.MultiIndex.from_product([active, dates], names=["fips", "date"])
    log.info("Balanced sparse index: %d counties x %d dates = %d rows", len(active), len(dates), len(grid_index))

    hourly = pd.read_parquet(
        DATA_PROC / "fars_hourly_county_day.parquet",
        columns=["fips", "date", "hour", "person_fatals"],
    )
    outcomes = build_linear_time_outcomes(hourly).set_index(["fips", "date"])

    alerts = ntm.base.load_verified_alerts(window="night", detail=False).rename(
        columns={"effective_crash_date": "date"}
    )[["fips", "date"]]
    alerts["fips"] = _fips(alerts["fips"])
    alerts = alerts[alerts["fips"].isin(active)].drop_duplicates()

    sample_fips = set(active)
    # This matches the established spillover design: alert origins and crash
    # destinations must both be in the active FARS county sample. It also
    # excludes post-2022 Connecticut planning-region codes that have no
    # compatible FARS county outcome in this panel.
    centroids = pd.read_parquet(DATA_PROC / "county_pop_centroids.parquet")
    pair_dosages = {}
    vintage_rows = []
    regimes = sorted({exposure_vintage_for_year(year) for year in range(2013, 2025)})
    for flow_vintage, lodes_vintage in regimes:
        year_path = PAIR_DOSAGE_YEAR_DIR / f"county_pair_lodes_car_dosage_{lodes_vintage}.parquet"
        if not year_path.exists():
            raise FileNotFoundError(
                f"missing year-specific LODES car-distance dosage: {year_path}; "
                "run code/build_lodes_tract_car_dosage.py first"
            )
        weights = flow_tables[flow_vintage]
        weights = weights[
            weights["fips_home"].isin(sample_fips)
            & weights["fips_work"].isin(sample_fips)
        ].copy()
        joint = pd.read_parquet(year_path)
        fallback = build_distance_driving_fallback(
            weights,
            joint[["fips_home", "fips_work", "avg_dist_mi"]],
            centroids,
        )
        pairs = build_pair_dosage(weights, joint, fallback)
        pair_dosages[(flow_vintage, lodes_vintage)] = pairs
        fallback_edges = pairs["dosage_source"].eq("distance_driving_fallback")
        fallback_weight_share = float(
            pairs.loc[fallback_edges, "weight"].sum() / pairs["weight"].sum()
        )
        crash_years = [
            year for year in range(2013, 2025)
            if exposure_vintage_for_year(year) == (flow_vintage, lodes_vintage)
        ]
        vintage_rows.append(
            {
                "flow_vintage": flow_vintage,
                "lodes_vintage": lodes_vintage,
                "lodes_state_source_year_fallbacks": lodes_fallback_label(lodes_vintage),
                "crash_year_start": min(crash_years),
                "crash_year_end": max(crash_years),
                "n_edges": len(pairs),
                "fallback_edges": int(fallback_edges.sum()),
                "fallback_weight_share": fallback_weight_share,
            }
        )
        log.info(
            "ACS %s + LODES %d (%d-%d): %d edges; fallback %.3f%% of weight",
            flow_vintage,
            lodes_vintage,
            min(crash_years),
            max(crash_years),
            len(pairs),
            100 * fallback_weight_share,
        )
        del weights, joint, fallback
        gc.collect()

    vintage_summary = pd.DataFrame(vintage_rows)
    vintage_summary.to_csv(
        OUTPUT_TABS / "symmetric_commuter_fatigue_vintage_summary.csv", index=False
    )
    own_values, cross_values = construct_year_matched_exposure_series(
        grid_index, alerts, pair_dosages
    )

    n_fips, n_dates = len(active), len(dates)
    fips_id = np.repeat(np.arange(n_fips, dtype=np.int32), n_dates)
    date_id = np.tile(np.arange(n_dates, dtype=np.int32), n_fips)
    years = dates.year.to_numpy(dtype=np.int32)
    dows = dates.dayofweek.to_numpy(dtype=np.int32)
    months = dates.month.to_numpy(dtype=np.int32)
    state_by_fips = pd.factorize(pd.Index(active).str[:2], sort=True)[0].astype(np.int32)
    model_data = pd.DataFrame(
        {
            "fatals_0623": outcomes["fatals_0623"].reindex(grid_index).fillna(0.0).to_numpy(),
            "fatals_hours_awake_slope": outcomes["fatals_hours_awake_slope"].reindex(grid_index).fillna(0.0).to_numpy(),
            OWN_EXPOSURE: own_values,
            CROSS_EXPOSURE: cross_values,
            "fips_year_id": fips_id * len(np.unique(years)) + np.tile(years - years.min(), n_fips),
            "fips_dow_id": fips_id * 7 + np.tile(dows, n_fips),
            "month_id": np.tile(months, n_fips),
            "state_cluster_id": np.repeat(state_by_fips, n_dates),
            "date_cluster_id": date_id,
        }
    )

    diagnostics = (
        model_data[[OWN_EXPOSURE, CROSS_EXPOSURE]]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .T.reset_index(names="exposure")
    )
    diagnostics["nonzero"] = [
        int(model_data[OWN_EXPOSURE].gt(0).sum()),
        int(model_data[CROSS_EXPOSURE].gt(0).sum()),
    ]
    diagnostics["max_pair_fallback_weight_share"] = vintage_summary["fallback_weight_share"].max()
    diagnostics["analysis_counties"] = len(active)
    diagnostics["excluded_no_self_loop_counties"] = len(excluded_no_self_loop)
    diagnostics.to_csv(
        OUTPUT_TABS / "symmetric_commuter_fatigue_year_matched_exposure_summary.csv", index=False
    )
    linear_time_weights().to_csv(
        OUTPUT_TABS / "symmetric_commuter_fatigue_time_weights.csv", index=False
    )

    del fars, hourly, outcomes, flow_tables, centroids, pair_dosages, own_values, cross_values
    gc.collect()

    results = []
    for idx, outcome in enumerate(("fatals_0623", "fatals_hours_awake_slope")):
        log.info("Estimating joint symmetric model for %s", outcome)
        results.extend(
            _fit_model(
                model_data,
                outcome,
                reps=bootstrap_reps,
                seed=BOOTSTRAP_SEED + 10 * idx,
            )
        )

    result = pd.DataFrame(results)
    result["analytic_cluster"] = "state_code + date_str"
    result["wild_cluster"] = "state_code"
    result["wild_weights"] = "rademacher"
    result["wild_impose_null"] = True
    result["exposure_spec"] = "year_matched_acs_and_lodes"
    path = OUTPUT_TABS / "reg_symmetric_commuter_fatigue_year_matched.csv"
    result.to_csv(path, index=False)
    log.info("Saved -> %s", path)
    return result


if __name__ == "__main__":
    main()
