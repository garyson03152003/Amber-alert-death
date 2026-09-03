"""Robustness suite for the year-matched symmetric commuter-fatigue model.

The suite keeps the validated daily FARS fatal-crash outcome and common
ACS/LODES exposure construction, then adds six targeted checks:

1. weighted daily lead/lag exposure-response;
2. shifted-date and daytime-alert placebos;
3. statewide/county-only, trimmed, and leave-one-state-out sensitivity;
4. Webb six-point wild-bootstrap and state-month block randomization;
5. fatal-crash counts as the outcome (rather than person fatalities);
6. zero/positive and positive-quantile nonlinear exposure bins.
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
try:
    import pyfixest as pf
except ModuleNotFoundError:  # construction and diagnostics do not need it
    pf = None

sys.path.insert(0, str(Path(__file__).parent))
import run_symmetric_commuter_fatigue as base
import run_state_dot_analysis_fixed as state_base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("symmetric_commuter_robustness")

LAG_DAYS = (-2, -1, 0, 1, 2, 3)
EVENT_BIN_ORDER = ("lead_2", "lead_1", "post_0_2", "post_3_5", "post_6_8", "post_9_12", "post_13_18")
OUT_PATH = OUTPUT_TABS / "reg_symmetric_commuter_robustness.csv"
DEFAULT_FIXED_EFFECT_COLS = ("fips_year_id", "fips_dow_id", "month_id")
STATE_DATE_FIXED_EFFECT_COLS = ("fips_year_id", "fips_dow_id", "state_date_id")
TIME_BLOCK_OUTCOMES = tuple(base.TIME_BLOCKS) + ("fatals_late_minus_morning",)


def build_state_date_ids(fips: pd.Series, dates: pd.Series) -> np.ndarray:
    """Return stable integer identifiers for normalized state-date cells."""
    states = base._fips(pd.Series(fips, copy=False)).str[:2]
    normalized_dates = pd.to_datetime(pd.Series(dates, copy=False)).dt.normalize()
    keys = states + "|" + normalized_dates.dt.strftime("%Y-%m-%d")
    return pd.factorize(keys, sort=True)[0].astype(np.int32, copy=False)


def permute_cross_destinations(
    pair_dosage: pd.DataFrame,
    rng: np.random.Generator,
    *,
    swaps_per_edge: int = 2,
) -> pd.DataFrame:
    """Degree-preserving placebo rewiring of cross-county commuter edges.

    Valid double-edge swaps are performed separately inside each
    home-state/work-state block. Self loops and edge-level dosage values stay
    attached to their original rows, while cross-county destinations change.
    """
    required = {"fips_home", "fips_work", "commuter_car_miles"}
    if missing := required - set(pair_dosage.columns):
        raise ValueError(f"pair dosage missing columns: {sorted(missing)}")
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be nonnegative")

    out = pair_dosage.copy()
    out["fips_home"] = base._fips(out["fips_home"])
    out["fips_work"] = base._fips(out["fips_work"])
    if out.duplicated(["fips_home", "fips_work"]).any():
        raise ValueError("pair dosage contains duplicate county pairs")

    cross_mask = out["fips_home"].ne(out["fips_work"])
    cross = out.loc[cross_mask, ["fips_home", "fips_work"]].copy()
    cross["home_state"] = cross["fips_home"].str[:2]
    cross["work_state"] = cross["fips_work"].str[:2]
    successful_swaps = 0

    for _, block in cross.groupby(["home_state", "work_state"], sort=True):
        indices = block.index.to_numpy()
        if len(indices) < 2 or swaps_per_edge == 0:
            continue
        homes = out.loc[indices, "fips_home"].to_numpy(copy=True)
        works = out.loc[indices, "fips_work"].to_numpy(copy=True)
        edges = set(zip(homes, works))
        target = int(np.ceil(len(indices) * swaps_per_edge))
        attempts = 0
        completed = 0
        max_attempts = max(100, target * 100)
        while completed < target and attempts < max_attempts:
            attempts += 1
            left, right = rng.choice(len(indices), size=2, replace=False)
            home_left, home_right = homes[left], homes[right]
            work_left, work_right = works[left], works[right]
            if home_left == home_right or work_left == work_right:
                continue
            new_left = (home_left, work_right)
            new_right = (home_right, work_left)
            if new_left[0] == new_left[1] or new_right[0] == new_right[1]:
                continue
            old_left = (home_left, work_left)
            old_right = (home_right, work_right)
            remaining = edges - {old_left, old_right}
            if new_left in remaining or new_right in remaining or new_left == new_right:
                continue
            edges.remove(old_left)
            edges.remove(old_right)
            edges.add(new_left)
            edges.add(new_right)
            works[left], works[right] = work_right, work_left
            completed += 1
        out.loc[indices, "fips_work"] = works
        successful_swaps += completed

    out.attrs["successful_edge_swaps"] = successful_swaps
    return out


def randomization_pvalue(observed: float, placebo: np.ndarray) -> float:
    """Return a finite-sample, two-sided randomization p-value."""
    observed_value = float(observed)
    placebo_values = np.asarray(placebo, dtype=float)
    if not np.isfinite(observed_value):
        raise ValueError("observed estimate must be finite")
    if placebo_values.size == 0 or not np.isfinite(placebo_values).all():
        raise ValueError("placebo estimates must be a nonempty finite array")
    exceedances = int((np.abs(placebo_values) >= abs(observed_value)).sum())
    return float((exceedances + 1) / (placebo_values.size + 1))


def randomization_tail_ranks(observed: float, placebo: np.ndarray) -> dict[str, float]:
    """Return directional finite-sample p-values and the empirical percentile."""
    observed_value = float(observed)
    placebo_values = np.asarray(placebo, dtype=float)
    if not np.isfinite(observed_value):
        raise ValueError("observed estimate must be finite")
    if placebo_values.size == 0 or not np.isfinite(placebo_values).all():
        raise ValueError("placebo estimates must be a nonempty finite array")
    denominator = placebo_values.size + 1
    return {
        "upper_pval": float((1 + (placebo_values >= observed_value).sum()) / denominator),
        "lower_pval": float((1 + (placebo_values <= observed_value).sum()) / denominator),
        "percentile": float((placebo_values <= observed_value).mean()),
    }


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Apply Holm's step-down familywise correction to finite p-values."""
    numeric = pd.to_numeric(pvalues, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    if ((numeric.loc[finite] < 0) | (numeric.loc[finite] > 1)).any():
        raise ValueError("p-values must lie between zero and one")
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    values = numeric.loc[finite].to_numpy(dtype=float)
    if values.size == 0:
        return adjusted
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = (len(ranked) - np.arange(len(ranked))) * ranked
    monotone = np.minimum(1.0, np.maximum.accumulate(scaled))
    restored = np.empty_like(monotone)
    restored[order] = monotone
    adjusted.loc[finite] = restored
    return adjusted


def event_time_bin(offset: int) -> str | None:
    """Map an alert-relative day/hour offset into a prespecified bin."""
    offset = int(offset)
    if offset == -2:
        return "lead_2"
    if offset == -1:
        return "lead_1"
    if 0 <= offset <= 2:
        return "post_0_2"
    if 3 <= offset <= 5:
        return "post_3_5"
    if 6 <= offset <= 8:
        return "post_6_8"
    if 9 <= offset <= 12:
        return "post_9_12"
    if 13 <= offset <= 18:
        return "post_13_18"
    return None


def build_daily_lagged_exposures(
    grid_index: pd.MultiIndex,
    alert_events: pd.DataFrame,
    pair_dosages: dict[tuple[str, int], pd.DataFrame],
    *,
    lags: tuple[int, ...] = LAG_DAYS,
) -> pd.DataFrame:
    """Build own/cross exposure columns for each alert-relative day lag."""
    result: dict[str, np.ndarray] = {}
    for lag in lags:
        shifted = alert_events[["fips", "date"]].copy()
        shifted["date"] = pd.to_datetime(shifted["date"]).dt.normalize() + pd.Timedelta(days=int(lag))
        own, cross = base.construct_year_matched_exposure_series(
            grid_index, shifted, pair_dosages
        )
        suffix = f"m{abs(lag)}" if lag < 0 else ("0" if lag == 0 else f"p{lag}")
        result[f"own_lag_{suffix}"] = own
        result[f"cross_lag_{suffix}"] = cross
    return pd.DataFrame(result, index=grid_index)


def build_daily_event_bin_exposures(
    grid_index: pd.MultiIndex,
    alert_events: pd.DataFrame,
    pair_dosages: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate daily own/cross exposure into prespecified event-time bins.

    Each column is the sum of the same commuter-car-mile dosage over the
    alert-relative days in that bin.  The two negative bins are pre-alert
    placebo exposures; the remaining bins cover the first 18 post-alert days.
    Building the bins from the common dosage constructor preserves the
    year-specific ACS/LODES vintage assignment at every shifted date.
    """
    result = {
        f"own_{label}": np.zeros(len(grid_index), dtype=np.float64)
        for label in EVENT_BIN_ORDER
    }
    result.update({
        f"cross_{label}": np.zeros(len(grid_index), dtype=np.float64)
        for label in EVENT_BIN_ORDER
    })
    for offset in range(-2, 19):
        label = event_time_bin(offset)
        if label is None:
            continue
        shifted = alert_events[["fips", "date"]].copy()
        shifted["date"] = (
            pd.to_datetime(shifted["date"]).dt.normalize()
            + pd.Timedelta(days=offset)
        )
        own, cross = base.construct_year_matched_exposure_series(
            grid_index, shifted, pair_dosages
        )
        result[f"own_{label}"] += own
        result[f"cross_{label}"] += cross
    return pd.DataFrame(result, index=grid_index)


def build_daily_fatal_crash_outcomes(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sparse validated FARS 06:00--23:00 counts to county-days."""
    required = {"fips", "date", "fatal_crashes"}
    if missing := required - set(hourly.columns):
        raise ValueError(f"hourly crash data missing columns: {sorted(missing)}")
    data = hourly.loc[:, ["fips", "date", "fatal_crashes"]].copy()
    data["fips"] = data["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    if "hour" in hourly.columns:
        data = data.loc[hourly["hour"].between(min(base.HOURS), max(base.HOURS)).to_numpy()]
    data["fatal_crashes"] = pd.to_numeric(data["fatal_crashes"], errors="raise")
    return data.groupby(["fips", "date"], as_index=False)["fatal_crashes"].sum()


def draw_wild_weights(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw reproducible state-cluster weights, including Webb's six points."""
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=size)
    if kind == "webb":
        points = np.concatenate(
            [-np.sqrt(np.array([3.0, 2.0, 1.0]) / 2.0),
             np.sqrt(np.array([1.0, 2.0, 3.0]) / 2.0)]
        )
        return rng.choice(points, size=size)
    raise ValueError(f"unknown wild-weight distribution: {kind}")


def add_exposure_bins(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add zero/positive indicators for both exposures without dropping zeros."""
    out = data.copy()
    terms: list[str] = []
    for exposure in (base.OWN_EXPOSURE, base.CROSS_EXPOSURE):
        zero = f"{exposure.replace('_driver_distance', '')}_bin_0"
        positive = f"{exposure.replace('_driver_distance', '')}_bin_pos"
        out[zero] = out[exposure].eq(0).astype(int)
        out[positive] = out[exposure].gt(0).astype(int)
        terms.extend([zero, positive])
    return out, terms


def add_positive_quantile_bins(
    data: pd.DataFrame, *, n_bins: int = 4
) -> tuple[pd.DataFrame, list[str]]:
    """Add one-hot positive-exposure quantiles, leaving structural zeros out."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    out = data.copy()
    terms: list[str] = []
    for exposure, prefix in ((base.OWN_EXPOSURE, "own"), (base.CROSS_EXPOSURE, "cross")):
        positive = pd.to_numeric(out[exposure], errors="raise").gt(0)
        values = out.loc[positive, exposure]
        if values.empty:
            continue
        ranks = values.rank(method="first")
        labels = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop") + 1
        for quantile in sorted(pd.unique(labels)):
            name = f"{prefix}_q{int(quantile)}"
            out[name] = 0
            out.loc[labels.index[labels.eq(quantile)], name] = 1
            terms.append(name)
    return out, terms


def keep_below_positive_tail(values: pd.Series, quantile: float = 0.99) -> pd.Series:
    """Keep all structural zeros and trim only the upper positive tail."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    numeric = pd.to_numeric(values, errors="raise")
    positive = numeric.gt(0)
    if not positive.any():
        return pd.Series(True, index=values.index)
    cutoff = numeric.loc[positive].quantile(quantile)
    return numeric.le(cutoff) | ~positive


def _prepare_model_data(model_data: pd.DataFrame) -> pd.DataFrame:
    out = model_data.copy()
    out["state_cluster_id"] = out["state_cluster_id"].astype(int)
    out["date_cluster_id"] = out["date_cluster_id"].astype(int)
    return out


def _cluster_sums(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    codes, levels = pd.factorize(clusters, sort=True)
    result = np.empty((len(levels), values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        result[:, col] = np.bincount(codes, weights=values[:, col], minlength=len(levels))
    return result


def _cluster_covariance(
    x_tilde: np.ndarray,
    residual: np.ndarray,
    state_clusters: np.ndarray,
    date_clusters: np.ndarray,
    bread: np.ndarray,
) -> np.ndarray:
    """Two-way CRV1 covariance used by the local fallback estimator."""
    n, k = x_tilde.shape

    def meat(clusters: np.ndarray) -> np.ndarray:
        scores = _cluster_sums(x_tilde * residual[:, None], clusters)
        return scores.T @ scores

    state_codes = pd.factorize(state_clusters, sort=True)[0]
    date_codes = pd.factorize(date_clusters, sort=True)[0]
    joint_codes = state_codes.astype(np.int64) * (int(date_codes.max()) + 1) + date_codes
    g_state = max(int(state_codes.max()) + 1, 1)
    g_date = max(int(date_codes.max()) + 1, 1)
    g_joint = max(int(pd.unique(joint_codes).size), 1)
    meat_state = meat(state_clusters)
    meat_date = meat(date_clusters)
    meat_joint = meat(joint_codes)

    def correction(g: int) -> float:
        return (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))

    meat_tw = (
        correction(g_state) * meat_state
        + correction(g_date) * meat_date
        - correction(g_joint) * meat_joint
    )
    return bread @ meat_tw @ bread


def _wild_cluster_pvalue(y_tilde, x_tilde, clusters, restriction, *, reps, seed, weights_kind):
    """Fast null-imposed one-way WCR11 p-value for one contrast."""
    y = np.asarray(y_tilde, dtype=np.float64)
    x = np.asarray(x_tilde, dtype=np.float64)
    r = np.asarray(restriction, dtype=np.float64)
    codes, levels = pd.factorize(clusters, sort=True)
    n, k = x.shape
    g = len(levels)
    if g < 2:
        raise ValueError("wild cluster bootstrap requires at least two clusters")
    if reps < 1:
        raise ValueError("wild cluster bootstrap requires at least one draw")
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ (x.T @ y)
    direction = bread @ r
    variance = float(r @ direction)
    beta_restricted = beta - direction * (float(r @ beta) / variance)
    residual_restricted = y - x @ beta_restricted
    residual_unrestricted = y - x @ beta
    score_u = _cluster_sums(x * residual_unrestricted[:, None], codes)
    contrast_score = score_u @ direction
    ssc = (g / (g - 1)) * ((n - 1) / (n - k))
    observed_se = float(np.sqrt(ssc * np.square(contrast_score).sum()))
    observed_t = float((r @ beta) / observed_se)
    score_r = _cluster_sums(x * residual_restricted[:, None], codes)
    crossproducts = np.empty((g, k, k), dtype=np.float64)
    for left in range(k):
        for right in range(k):
            crossproducts[:, left, right] = np.bincount(
                codes, weights=x[:, left] * x[:, right], minlength=g
            )
    direction_h = np.einsum("k,gkl->gl", direction, crossproducts)
    q_direction = score_r @ direction
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    chunk_size = min(reps, 500)
    while completed < reps:
        size = min(chunk_size, reps - completed)
        wild = draw_wild_weights(weights_kind, g * size, rng).reshape(g, size)
        delta = bread @ (score_r.T @ wild)
        cluster_scores = q_direction[:, None] * wild - direction_h @ delta
        se = np.sqrt(ssc * np.square(cluster_scores).sum(axis=0))
        t_stats = (r @ delta) / se
        exceedances += int((np.abs(t_stats) >= abs(observed_t)).sum())
        completed += size
    return float((exceedances + 1) / (reps + 1))


def _fit_analytic(
    data: pd.DataFrame,
    outcome: str,
    terms: list[str],
    *,
    spec: str,
    wild_kind: str | None = None,
    bootstrap_reps: int = 0,
    bootstrap_seed: int = 0,
    randomization: bool = False,
    fixed_effect_cols: tuple[str, ...] = DEFAULT_FIXED_EFFECT_COLS,
    prefer_pyfixest: bool = True,
) -> list[dict]:
    """Fit a FE model and return tidy estimate rows for one robustness spec."""
    fixed_effect_label = " + ".join(fixed_effect_cols)
    if not terms:
        return [{
            "spec": spec,
            "outcome": outcome,
            "status": "skipped",
            "nobs": 0,
            "fixed_effects": fixed_effect_label,
        }]
    needed = list(dict.fromkeys([
        outcome, *terms, *fixed_effect_cols, "state_cluster_id",
        "date_cluster_id", "state_month_cluster_id",
    ]))
    sub = data.loc[:, needed].dropna()
    if sub.empty:
        return [{
            "spec": spec,
            "outcome": outcome,
            "status": "skipped",
            "nobs": 0,
            "fixed_effects": fixed_effect_label,
        }]
    within = base.fit_within_ols(
        sub[outcome].to_numpy(), sub[terms].to_numpy(),
        [sub[c].to_numpy() for c in fixed_effect_cols],
    )
    use_pyfixest = pf is not None and prefer_pyfixest
    if use_pyfixest:
        fit = pf.feols(
            f"{outcome} ~ {' + '.join(terms)} | {fixed_effect_label}",
            data=sub,
            vcov={"CRV1": "state_cluster_id + date_cluster_id"},
            copy_data=False,
            store_data=False,
            lean=True,
        )
        tidy = fit.tidy()
        nobs = int(fit._N)
        estimates = {
            term: {
                "coef": float(tidy.loc[term, "Estimate"]),
                "se": float(tidy.loc[term, "Std. Error"]),
                "pval": float(tidy.loc[term, "Pr(>|t|)"]),
            }
            for term in terms
        }
    else:
        covariance = _cluster_covariance(
            within["x_tilde"], within["residual"],
            sub["state_cluster_id"].to_numpy(), sub["date_cluster_id"].to_numpy(),
            within["bread"],
        )
        estimates = {}
        df_t = max(
            min(sub["state_cluster_id"].nunique(), sub["date_cluster_id"].nunique()) - 1,
            1,
        )
        for idx, term in enumerate(terms):
            se = float(np.sqrt(max(covariance[idx, idx], 0.0)))
            t_stat = float(within["beta"][idx] / se) if se > 0 else np.nan
            estimates[term] = {
                "coef": float(within["beta"][idx]),
                "se": se,
                "pval": float(2 * stats.t.sf(abs(t_stat), df=df_t))
                if np.isfinite(t_stat) else np.nan,
            }
        nobs = int(len(sub))
    rows: list[dict] = []
    for idx, term in enumerate(terms):
        estimate = estimates[term]
        record = {
            "spec": spec, "outcome": outcome, "term": term, "status": "ok",
            "coef": estimate["coef"], "se_state_date": estimate["se"],
            "pval_state_date": estimate["pval"], "nobs": nobs,
            "bootstrap_reps": int(bootstrap_reps) if wild_kind and bootstrap_reps else 0,
            "estimator": "pyfixest" if use_pyfixest else "numpy_within_ols",
            "fixed_effects": fixed_effect_label,
        }
        if wild_kind is not None and bootstrap_reps:
            restriction = np.zeros(len(terms))
            restriction[idx] = 1.0
            cluster = sub["state_month_cluster_id"].to_numpy() if randomization else sub["state_cluster_id"].to_numpy()
            record["pval_wild"] = _wild_cluster_pvalue(
                within["y_tilde"], within["x_tilde"], cluster, restriction,
                reps=bootstrap_reps, seed=bootstrap_seed + idx, weights_kind=wild_kind,
            )
            record["inference"] = "state_month_sign_randomization" if randomization else f"wild_{wild_kind}"
        rows.append(record)
    if use_pyfixest:
        del fit
    del sub, within
    gc.collect()
    return rows


def run_state_date_models(
    panel: pd.DataFrame,
    *,
    bootstrap_reps: int,
    seed: int,
) -> list[dict]:
    """Estimate joint own/cross effects using within-state/date variation."""
    terms = [base.OWN_EXPOSURE, base.CROSS_EXPOSURE]
    rows: list[dict] = []
    for outcome_index, outcome in enumerate(("fatal_crashes", "total_fatals")):
        rows.extend(
            _fit_analytic(
                panel,
                outcome,
                terms,
                spec="state_date_fixed_effects",
                wild_kind="webb",
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=seed + 10 * outcome_index,
                fixed_effect_cols=STATE_DATE_FIXED_EFFECT_COLS,
                prefer_pyfixest=False,
            )
        )
    return rows


def _joint_outcome_coefficients(
    panel: pd.DataFrame,
    cross_exposure: np.ndarray,
    *,
    outcomes: tuple[str, ...],
    fixed_effect_cols: tuple[str, ...],
    transformed_fixed: np.ndarray | None = None,
    fixed_effects: list[np.ndarray] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return own and cross coefficients for several outcomes at once."""
    if fixed_effects is None:
        fixed_effects = [panel[col].to_numpy() for col in fixed_effect_cols]
    if transformed_fixed is None:
        fixed_values = np.column_stack(
            [
                *(panel[outcome].to_numpy(dtype=float) for outcome in outcomes),
                panel[base.OWN_EXPOSURE].to_numpy(dtype=float),
            ]
        )
        transformed_fixed = base._within_transform(fixed_values, fixed_effects)
    cross_tilde = base._within_transform(
        np.asarray(cross_exposure, dtype=float), fixed_effects
    )
    outcome_values = transformed_fixed[:, : len(outcomes)]
    exposure_values = np.column_stack(
        [transformed_fixed[:, len(outcomes)], cross_tilde]
    )
    beta = np.linalg.solve(
        exposure_values.T @ exposure_values,
        exposure_values.T @ outcome_values,
    )
    return {
        outcome: (float(beta[0, index]), float(beta[1, index]))
        for index, outcome in enumerate(outcomes)
    }


def run_network_placebos(
    panel: pd.DataFrame,
    alerts: pd.DataFrame,
    metadata: dict,
    *,
    draws: int,
    seed: int,
    swaps_per_edge: int = 2,
) -> tuple[pd.DataFrame, list[dict]]:
    """Compare observed-network coefficients with degree-preserving placebos."""
    if draws < 1:
        raise ValueError("network placebo requires at least one draw")
    outcomes = ("fatal_crashes", "total_fatals")
    fixed_effects = [panel[col].to_numpy() for col in STATE_DATE_FIXED_EFFECT_COLS]
    fixed_values = np.column_stack(
        [
            *(panel[outcome].to_numpy(dtype=float) for outcome in outcomes),
            panel[base.OWN_EXPOSURE].to_numpy(dtype=float),
        ]
    )
    transformed_fixed = base._within_transform(fixed_values, fixed_effects)
    observed = _joint_outcome_coefficients(
        panel,
        panel[base.CROSS_EXPOSURE].to_numpy(),
        outcomes=outcomes,
        fixed_effect_cols=STATE_DATE_FIXED_EFFECT_COLS,
        transformed_fixed=transformed_fixed,
        fixed_effects=fixed_effects,
    )
    seed_rng = np.random.default_rng(seed)
    draw_seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=draws, dtype=np.uint32)
    records: list[dict] = []
    alert_events = alerts[["fips", "date"]]

    for draw_index, draw_seed in enumerate(draw_seeds):
        permuted: dict[tuple[str, int], pd.DataFrame] = {}
        successful_swaps = 0
        for regime_index, regime in enumerate(sorted(metadata["pair_dosages"])):
            regime_rng = np.random.default_rng(
                np.random.SeedSequence([int(draw_seed), regime_index])
            )
            pair = permute_cross_destinations(
                metadata["pair_dosages"][regime],
                regime_rng,
                swaps_per_edge=swaps_per_edge,
            )
            successful_swaps += int(pair.attrs.get("successful_edge_swaps", 0))
            permuted[regime] = pair
        placebo_own, placebo_cross = base.construct_year_matched_exposure_series(
            metadata["grid_index"], alert_events, permuted
        )
        if not np.allclose(
            placebo_own,
            panel[base.OWN_EXPOSURE].to_numpy(),
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("network placebo changed own-county exposure")
        estimates = _joint_outcome_coefficients(
            panel,
            placebo_cross,
            outcomes=outcomes,
            fixed_effect_cols=STATE_DATE_FIXED_EFFECT_COLS,
            transformed_fixed=transformed_fixed,
            fixed_effects=fixed_effects,
        )
        for outcome in outcomes:
            records.append(
                {
                    "draw": draw_index + 1,
                    "seed": int(draw_seed),
                    "outcome": outcome,
                    "own_coef": estimates[outcome][0],
                    "cross_coef": estimates[outcome][1],
                    "successful_edge_swaps": successful_swaps,
                    "status": "ok",
                }
            )
        if (draw_index + 1) % 10 == 0 or draw_index + 1 == draws:
            log.info("Network placebo draws completed: %d/%d", draw_index + 1, draws)

    distribution = pd.DataFrame(records)
    summary: list[dict] = []
    for outcome in outcomes:
        placebo = distribution.loc[
            distribution["outcome"].eq(outcome), "cross_coef"
        ].to_numpy()
        tail_ranks = randomization_tail_ranks(observed[outcome][1], placebo)
        summary.append(
            {
                "spec": "observed_vs_placebo_network",
                "outcome": outcome,
                "term": base.CROSS_EXPOSURE,
                "coef": observed[outcome][1],
                "own_coef": observed[outcome][0],
                "pval_network": randomization_pvalue(observed[outcome][1], placebo),
                "pval_network_upper": tail_ranks["upper_pval"],
                "pval_network_lower": tail_ranks["lower_pval"],
                "network_percentile": tail_ranks["percentile"],
                "placebo_mean": float(placebo.mean()),
                "placebo_sd": float(placebo.std(ddof=1)) if len(placebo) > 1 else np.nan,
                "requested_draws": draws,
                "completed_draws": len(placebo),
                "seed": seed,
                "fixed_effects": " + ".join(STATE_DATE_FIXED_EFFECT_COLS),
                "status": "ok",
            }
        )
    return distribution, summary


def run_time_block_models(
    panel: pd.DataFrame,
    *,
    bootstrap_reps: int,
    seed: int,
) -> list[dict]:
    """Estimate rich time blocks and apply Holm correction within each family."""
    terms = [base.OWN_EXPOSURE, base.CROSS_EXPOSURE]
    rows: list[dict] = []
    specifications = (
        ("baseline_time_blocks", DEFAULT_FIXED_EFFECT_COLS),
        ("state_date_time_blocks", STATE_DATE_FIXED_EFFECT_COLS),
    )
    for spec_index, (spec, fixed_effect_cols) in enumerate(specifications):
        for outcome_index, outcome in enumerate(TIME_BLOCK_OUTCOMES):
            rows.extend(
                _fit_analytic(
                    panel,
                    outcome,
                    terms,
                    spec=spec,
                    wild_kind="webb",
                    bootstrap_reps=bootstrap_reps,
                    bootstrap_seed=seed + spec_index * 100 + outcome_index * 10,
                    fixed_effect_cols=fixed_effect_cols,
                    prefer_pyfixest=False,
                )
            )

    result = pd.DataFrame(rows)
    result["pval_holm"] = np.nan
    result["multiplicity_family"] = None
    block_mask = result["outcome"].isin(base.TIME_BLOCKS)
    for (spec, term), index in result.loc[block_mask].groupby(["spec", "term"]).groups.items():
        result.loc[index, "pval_holm"] = holm_adjust(
            result.loc[index, "pval_state_date"]
        )
        result.loc[index, "multiplicity_family"] = f"{spec}:{term}"
    return result.to_dict("records")


def _load_common_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load validated outcomes, alerts, and year-matched pair dosages once."""
    fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
    fars["date"] = pd.to_datetime(fars["date"]).dt.normalize()
    mean_annual = (
        fars.assign(year=fars["date"].dt.year)
        .groupby(["fips", "year"])["total_fatals"].sum()
        .groupby("fips").mean()
    )
    candidates = sorted(base._fips(
        mean_annual[mean_annual >= base.ntm.MIN_FATALS_PER_YEAR].index.to_series()
    ))
    flow_tables: dict[str, pd.DataFrame] = {}
    for vintage, path in base.FLOW_VINTAGE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing ACS flow vintage {vintage}: {path}")
        flow = pd.read_parquet(path)
        flow["fips_home"] = base._fips(flow["fips_home"])
        flow["fips_work"] = base._fips(flow["fips_work"])
        flow_tables[vintage] = flow
    active = candidates
    excluded: set[str] = set()
    for flow in flow_tables.values():
        active, dropped = base.restrict_to_self_loop_counties(active, flow)
        excluded.update(dropped)
    dates = pd.date_range("2013-01-01", "2024-12-30", freq="D")
    grid_index = pd.MultiIndex.from_product([active, dates], names=["fips", "date"])
    active_set = set(active)

    alerts = state_base.load_verified_alerts(window="night", detail=True)
    alerts["date"] = pd.to_datetime(alerts["effective_crash_date"]).dt.normalize()
    alerts = alerts[alerts["fips"].isin(active_set)].drop_duplicates(
        subset=["fips", "date", "sent_local"]
    )
    centroids = pd.read_parquet(DATA_PROC / "county_pop_centroids.parquet")
    pair_dosages: dict[tuple[str, int], pd.DataFrame] = {}
    regimes = sorted({base.exposure_vintage_for_year(y) for y in range(2013, 2025)})
    for flow_vintage, lodes_vintage in regimes:
        path = base.PAIR_DOSAGE_YEAR_DIR / f"county_pair_lodes_car_dosage_{lodes_vintage}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing year-specific LODES dosage: {path}")
        weights = flow_tables[flow_vintage]
        weights = weights[weights["fips_home"].isin(active_set) & weights["fips_work"].isin(active_set)].copy()
        joint = pd.read_parquet(path)
        fallback = base.build_distance_driving_fallback(
            weights, joint[["fips_home", "fips_work", "avg_dist_mi"]], centroids
        )
        pair_dosages[(flow_vintage, lodes_vintage)] = base.build_pair_dosage(weights, joint, fallback)
        del weights, joint, fallback

    own, cross = base.construct_year_matched_exposure_series(
        grid_index, alerts[["fips", "date"]], pair_dosages
    )
    hourly = pd.read_parquet(DATA_PROC / "fars_hourly_county_day.parquet")
    crash = build_daily_fatal_crash_outcomes(hourly).set_index(["fips", "date"])["fatal_crashes"]
    day_window = hourly.loc[hourly["hour"].between(min(base.HOURS), max(base.HOURS))].copy()
    day_window["fips"] = base._fips(day_window["fips"])
    day_window["date"] = pd.to_datetime(day_window["date"]).dt.normalize()
    fatal = (
        day_window.groupby(["fips", "date"])["person_fatals"].sum()
        .rename("total_fatals")
    )
    time_blocks = base.build_time_block_outcomes(hourly).set_index(["fips", "date"])
    n_fips, n_dates = len(active), len(dates)
    fips_id = np.repeat(np.arange(n_fips, dtype=np.int32), n_dates)
    date_id = np.tile(np.arange(n_dates, dtype=np.int32), n_fips)
    years = dates.year.to_numpy(dtype=np.int32)
    state_ids = pd.factorize(pd.Index(active).str[:2], sort=True)[0].astype(np.int32)
    panel = pd.DataFrame({
        "fips": np.repeat(active, n_dates),
        "date": np.tile(dates, n_fips),
        "fatal_crashes": crash.reindex(grid_index).fillna(0.0).to_numpy(),
        "total_fatals": fatal.reindex(grid_index).fillna(0.0).to_numpy(),
        **{
            outcome: time_blocks[outcome].reindex(grid_index).fillna(0.0).to_numpy()
            for outcome in TIME_BLOCK_OUTCOMES
        },
        base.OWN_EXPOSURE: own,
        base.CROSS_EXPOSURE: cross,
        "fips_year_id": fips_id * len(np.unique(years)) + np.tile(years - years.min(), n_fips),
        "fips_dow_id": fips_id * 7 + np.tile(dates.dayofweek.to_numpy(), n_fips),
        "month_id": np.tile(dates.month.to_numpy(), n_fips),
        "state_cluster_id": np.repeat(state_ids, n_dates),
        "date_cluster_id": date_id,
        "state_date_id": np.repeat(state_ids, n_dates) * n_dates + date_id,
    })
    # Integer state-month blocks avoid a costly 7M-row datetime string
    # conversion and are stable across every sensitivity sample.
    month_keys = dates.year.to_numpy(dtype=np.int32) * 12 + dates.month.to_numpy(dtype=np.int32)
    panel["state_month_cluster_id"] = (
        np.repeat(state_ids, n_dates).astype(np.int32) * 100_000
        + np.tile(month_keys, n_fips)
    )
    metadata = {
        "active": active, "excluded": sorted(excluded), "alerts": alerts,
        "pair_dosages": pair_dosages, "grid_index": grid_index,
    }
    return panel, alerts, metadata


def _exposure_panel(panel: pd.DataFrame, alerts: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    out = panel.copy()
    own, cross = base.construct_year_matched_exposure_series(
        metadata["grid_index"], alerts[["fips", "date"]], metadata["pair_dosages"]
    )
    out[base.OWN_EXPOSURE] = own
    out[base.CROSS_EXPOSURE] = cross
    return out


def _lag_panel(panel: pd.DataFrame, alerts: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy()
    lagged = build_daily_lagged_exposures(
        metadata["grid_index"], alerts[["fips", "date"]], metadata["pair_dosages"]
    )
    out[lagged.columns] = lagged.to_numpy()
    return out, list(lagged.columns)


def main(*, bootstrap_reps: int = base.BOOTSTRAP_REPS, leave_one_state: bool = True) -> pd.DataFrame:
    """Run all six robustness families and save one labeled result table."""
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    panel, alerts, metadata = _load_common_panel()
    results: list[dict] = []
    baseline_terms = [base.OWN_EXPOSURE, base.CROSS_EXPOSURE]

    # (5) Fatal-crash counts, with (4) Webb and state-month sign inference.
    for outcome in ("fatal_crashes", "total_fatals"):
        results.extend(_fit_analytic(
            panel, outcome, baseline_terms, spec="baseline_year_matched",
            wild_kind="webb", bootstrap_reps=bootstrap_reps,
            bootstrap_seed=base.BOOTSTRAP_SEED,
        ))
        results.extend(_fit_analytic(
            panel, outcome, baseline_terms, spec="state_month_sign_randomization",
            wild_kind="rademacher", bootstrap_reps=min(199, bootstrap_reps),
            bootstrap_seed=base.BOOTSTRAP_SEED + 100, randomization=True,
        ))

    # (1) Flexible daily event-time response. Negative bins are the causal
    # placebo; positive bins cover the first 18 post-alert days.
    event_bins = build_daily_event_bin_exposures(
        metadata["grid_index"], alerts[["fips", "date"]], metadata["pair_dosages"]
    )
    lagged = panel.copy()
    lagged[event_bins.columns] = event_bins.to_numpy()
    lag_terms = list(event_bins.columns)
    for outcome in ("fatal_crashes", "total_fatals"):
        results.extend(_fit_analytic(lagged, outcome, lag_terms, spec="daily_lead_lag"))
    del event_bins, lagged
    gc.collect()

    # (2) Backward shifts and daytime-alert negative control.
    for shift in (-1, -2, -7):
        shifted = alerts[["fips", "date"]].copy()
        shifted["date"] += pd.Timedelta(days=shift)
        placebo_panel = _exposure_panel(panel, shifted, metadata)
        results.extend(_fit_analytic(
            placebo_panel, "fatal_crashes", baseline_terms,
            spec=f"placebo_shift_{shift:+d}d",
        ))
        del placebo_panel
    daytime = state_base.load_verified_alerts(window="day", detail=True)
    daytime["date"] = pd.to_datetime(daytime["effective_crash_date"]).dt.normalize()
    daytime = daytime[daytime["fips"].isin(set(metadata["active"]))].drop_duplicates(
        subset=["fips", "date", "sent_local"]
    )
    daytime_panel = _exposure_panel(panel, daytime, metadata)
    results.extend(_fit_analytic(
        daytime_panel, "fatal_crashes", baseline_terms,
        spec="placebo_daytime_alerts",
    ))
    del daytime_panel, daytime

    # (3) Alert-scope and top-tail sensitivity checks.
    for scope in ("county_same", "statewide_same"):
        scoped_panel = _exposure_panel(panel, alerts[alerts["geo_scope"] == scope], metadata)
        results.extend(_fit_analytic(
            scoped_panel, "fatal_crashes", baseline_terms,
            spec=f"sensitivity_{scope}",
        ))
        del scoped_panel
    trim = panel[
        keep_below_positive_tail(panel[base.OWN_EXPOSURE])
        & keep_below_positive_tail(panel[base.CROSS_EXPOSURE])
    ]
    results.extend(_fit_analytic(
        trim, "fatal_crashes", baseline_terms,
        spec="sensitivity_drop_top_1pct_exposure",
    ))
    del trim

    if leave_one_state:
        state_names = (
            panel[["state_cluster_id", "fips"]]
            .drop_duplicates("state_cluster_id")
            .set_index("state_cluster_id")["fips"]
            .str[:2]
            .to_dict()
        )
        for state_id in sorted(panel["state_cluster_id"].unique()):
            log.info("Leave-one-state-out: excluding %s", state_names[int(state_id)])
            sub = panel[panel["state_cluster_id"] != state_id]
            rows = _fit_analytic(
                sub, "fatal_crashes", baseline_terms,
                spec=f"leave_state_out_{state_names[int(state_id)]}",
            )
            for row in rows:
                row["inference"] = "analytic_only"
            results.extend(rows)
            del sub

    # (6) Nonlinear zero-safe positive exposure response.
    zero_safe, zero_terms = add_exposure_bins(panel)
    # The zero indicator is the reference category; estimating both indicators
    # would duplicate the absorbed intercept.  Keep both columns in the data
    # for auditability but fit the positive-vs-zero contrasts only.
    positive_terms = [term for term in zero_terms if term.endswith("_bin_pos")]
    results.extend(_fit_analytic(
        zero_safe, "fatal_crashes", positive_terms, spec="zero_vs_positive_exposure",
    ))
    del zero_safe
    nonlinear, nonlinear_terms = add_positive_quantile_bins(panel, n_bins=4)
    results.extend(_fit_analytic(
        nonlinear, "fatal_crashes", nonlinear_terms,
        spec="positive_exposure_quantiles",
    ))
    del nonlinear

    out = pd.DataFrame(results)
    out["analysis_counties"] = len(metadata["active"])
    out["excluded_no_self_loop_counties"] = len(metadata["excluded"])
    out["exposure_spec"] = "year_matched_acs_and_lodes"
    out.to_csv(OUT_PATH, index=False)
    log.info("Saved robustness results -> %s (%d rows)", OUT_PATH, len(out))
    return out


if __name__ == "__main__":
    main()
