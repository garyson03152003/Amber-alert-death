"""
run_commuting_distance_robustness.py
=============================================================
Does the commuting-spillover result depend on genuinely long commutes, or
is it really just an adjacent-county proximity effect (shared weather,
shared local news covering the same abduction, a shared media market)
riding on the fact that most commuting flows happen to be short?

The network-permutation placebo (run_commuting_network_placebo.py) shows
the effect needs the REAL commuting network, not a random one -- but a
random relabeling scrambles adjacency too, so it can't separate "real
commuting flows matter" from "real geographic proximity matters", since
the two are highly correlated (most commuters live nearby).

This script instead splits/weights the SAME real network by commuting
distance (great-circle distance between county POPULATION-WEIGHTED
centroids, data/processed/county_pop_centroids.parquet -- the official
2020 Census mean center of population, built by
build_county_pop_centroids.py). An earlier version of this script used
a geometric centroid file (county_centroids.parquet) that only covered
1,646/3,144 counties and, for large unevenly-populated counties, sat
60-100+ miles from where people actually live (e.g. Nye County NV was
108.6 miles off) -- exactly the kind of measurement error that would
bias a distance-based test. The population-weighted centroid has full
national coverage and is materially more accurate for this purpose.

  1. dist-weighted:  cross_spillover_dist = sum_j w_jc * dist_jc * alert_jt
     -- the same spillover measure, but scaled by how far that commuting
     flow travels (more distance ~ more time exposed / more total driving).
  2. short-only:     network restricted to edges below the exposure-
     weighted median commuting distance (~28 miles) -- the "could just be
     next door" subset.
  3. long-only:      edges at or above that median -- these pairs are
     less plausibly just sharing a local weather/news shock, since 28+
     miles is already beyond a single local media market or weather cell
     in most of the country.
  4. coverage-matched baseline: the real network, restricted to ONLY the
     edges with known distances (both endpoints have a centroid) -- so
     (2) and (3) are compared against the same denominator, not against
     the full run_commuting_network_placebo.py number (which used every
     edge, including the ~19% of exposure weight whose centroid is
     missing here).

If the effect survives (or strengthens) in the long-only subset, that is
real evidence against a pure local-shock/adjacency confound. If it only
shows up in the short-only subset, the commuting story looks more like a
proximity artifact.

Centroid coverage: county_pop_centroids.parquet covers 3,221 of 3,144
commuting-network counties (i.e. essentially all of them; the extra rows
are county-equivalents outside the commuting network). Coverage of the
"coverage-matched baseline" below is therefore near-total rather than
the 62%/44% edges/weight the geometric-centroid version had.

Output: output/tables/reg_commuting_distance_robustness.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("commuting_distance")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FE = "fips_year + fips_dow + month_str"
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def build_weights_with_distance():
    weights = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    weights = weights[weights["fips_home"] != weights["fips_work"]].copy()
    weights["fips_home_s"] = weights["fips_home"].astype(str).str.zfill(5)
    weights["fips_work_s"] = weights["fips_work"].astype(str).str.zfill(5)

    cent = pd.read_parquet(DATA_PROC / "county_pop_centroids.parquet")
    cent["fips"] = cent["fips"].astype(str).str.zfill(5)
    cm = cent.set_index("fips")

    lat1 = weights["fips_home_s"].map(cm["lat"]); lon1 = weights["fips_home_s"].map(cm["lon"])
    lat2 = weights["fips_work_s"].map(cm["lat"]); lon2 = weights["fips_work_s"].map(cm["lon"])
    weights["dist_mi"] = haversine_miles(lat1, lon1, lat2, lon2)

    n_total, w_total = len(weights), weights["weight"].sum()
    covered = weights.dropna(subset=["dist_mi"]).copy()
    log.info("Centroid coverage: %d/%d edges (%.1f%%), %.1f%% of exposure weight",
             len(covered), n_total, 100 * len(covered) / n_total,
             100 * covered["weight"].sum() / w_total)

    covered = covered.sort_values("dist_mi")
    cum = covered["weight"].cumsum() / covered["weight"].sum()
    median_dist = covered.loc[(cum - 0.5).abs().idxmin(), "dist_mi"]
    log.info("Exposure-weighted median commuting distance: %.1f miles", median_dist)
    return covered, median_dist


def spillover_series(grid_index, alert_events, weights, fips_in_sample, value_col="weight"):
    spill_pairs = alert_events.merge(weights, on="fips_home", how="inner")
    spill_pairs["fips_work_str"] = spill_pairs["fips_work"].astype(str).str.zfill(5)
    spill_pairs = spill_pairs[spill_pairs["fips_work_str"].isin(fips_in_sample)]

    agg = (spill_pairs.groupby(["fips_work_str", "date"])[value_col]
           .sum().rename("value"))
    agg.index = agg.index.set_names(["fips", "date"])
    aligned = agg.reindex(grid_index).fillna(0.0)
    return aligned.to_numpy(), len(spill_pairs), spill_pairs["weight"].sum() if len(spill_pairs) else 0.0


def fit_spillover(grid: pd.DataFrame, col: str):
    cluster_cols = ntm.CLUSTER_VARS.split(" + ")
    sub = grid.loc[:, [col, "night_alert", "fatals_0623", "fips_year", "fips_dow",
                        "month_str"] + cluster_cols].dropna()
    formula = f"fatals_0623 ~ {col} + night_alert | {FE}"
    fit = pf.feols(formula, data=sub, vcov={"CRV1": ntm.CLUSTER_VARS}, lean=True)
    row = fit.tidy().loc[col]
    coef, se, pval = float(row["Estimate"]), float(row["Std. Error"]), float(row["Pr(>|t|)"])
    del fit, sub
    return coef, se, pval


def main():
    grid = ntm.build_outcome_grid()
    grid = ntm.attach_night_alert(grid)
    grid["year_str"] = grid["date"].dt.year.astype(str)
    grid["dow"] = grid["date"].dt.dayofweek.astype(str)
    grid["month_str"] = grid["date"].dt.month.astype(str)
    grid["fips_dow"] = grid["fips"] + "_" + grid["dow"]
    grid["fips_year"] = grid["fips"] + "_" + grid["year_str"]
    grid["state_code"] = grid["fips"].str[:2]
    grid["date_str"] = grid["date"].dt.strftime("%Y-%m-%d")

    fips_in_sample = set(grid["fips"].unique())
    grid_index = pd.MultiIndex.from_frame(grid[["fips", "date"]])

    alert_events = grid.loc[grid["night_alert"] > 0, ["fips", "date"]].copy()
    alert_events["fips_home"] = alert_events["fips"].astype(int)

    weights, median_dist = build_weights_with_distance()
    weights["weight_x_dist"] = weights["weight"] * weights["dist_mi"]

    variants = {
        "coverage_matched (all edges w/ known distance)": (weights, "weight"),
        f"short_only (<{median_dist:.0f} mi, exposure-weighted median)":
            (weights[weights["dist_mi"] < median_dist], "weight"),
        f"long_only (>={median_dist:.0f} mi)":
            (weights[weights["dist_mi"] >= median_dist], "weight"),
        "distance_weighted (weight x miles, continuous)": (weights, "weight_x_dist"),
    }

    results = []
    for label, (w_subset, value_col) in variants.items():
        values, n_pairs, w_sum = spillover_series(grid_index, alert_events, w_subset,
                                                   fips_in_sample, value_col=value_col)
        grid["_spill"] = values
        coef, se, pval = fit_spillover(grid, "_spill")
        sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
        log.info("[%s] beta=%+.6f se=%.6f p=%.4f %s (%d edges lit up, weight sum=%.1f)",
                 label, coef, se, pval, sig, n_pairs, w_sum)
        results.append({"variant": label, "coef": coef, "se": se, "pval": pval,
                        "n_edges_lit": n_pairs, "weight_sum": w_sum})
        gc.collect()

    grid.drop(columns=["_spill"], inplace=True, errors="ignore")

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_commuting_distance_robustness.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
