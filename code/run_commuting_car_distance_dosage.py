"""
run_commuting_car_distance_dosage.py
=============================================================
Combines the two exposure refinements tried separately so far -- car-mode
weighting and commuting-distance weighting -- into one "pair dosage":

    cross_spillover_car_dist_ct = sum_j w_jc * car_share_j * dist_jc * alert_jt

Motivation: the plain cross_spillover measure treats every commuter from
an alerted home county the same, regardless of whether they actually
drive to work (a transit/walk commuter can't have a drowsy CAR crash) or
how far they drive (more miles ~ more exposure). We don't have true
per-county-PAIR driving-mode shares (Census's county-to-county commuting
flow product only reports total worker counts, no mode breakdown -- see
build_commuting_weights.py) -- so car_share_j is a proxy: the HOME
county's overall car-commute share (ACS B08301, county_car_commuters.parquet,
covers 3,143/3,144 commuting-network counties), on the reasoning that mode
choice is more a property of where a commuter lives (car ownership, local
transit access) than of their destination.

Because car_share has near-complete coverage but distance requires county
centroids (only 1,646/3,144 counties -- see run_commuting_distance_robustness.py),
this script reports BOTH:
  - full-network variants (100% coverage) isolating the car-share effect
    alone, comparable to the original cross_spillover headline number
  - coverage-matched variants (same distance-covered subset as
    run_commuting_distance_robustness.py) isolating what adding distance
    contributes ON TOP of car-weighting, apples-to-apples

Output: output/tables/reg_commuting_car_distance_dosage.csv
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
import run_commuting_distance_robustness as dist_mod
from build_nhts_car_share_by_distance import car_share_from_distance, car_share_from_distance_by_county
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("commuting_car_dist")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)


def load_car_share():
    car = pd.read_parquet(DATA_PROC / "county_car_commuters.parquet",
                          columns=["fips", "car_total", "total_workers"])
    car["fips"] = car["fips"].astype(str).str.zfill(5)
    car["car_share"] = car["car_total"] / car["total_workers"].clip(lower=1)
    log.info("Car-share coverage: %d counties, mean=%.1f%%, median=%.1f%%",
             len(car), car["car_share"].mean() * 100, car["car_share"].median() * 100)
    return car


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

    car = load_car_share()
    weights_full = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    weights_full = weights_full[weights_full["fips_home"] != weights_full["fips_work"]].copy()
    weights_full["fips_home_s"] = weights_full["fips_home"].astype(str).str.zfill(5)

    car_map = car.set_index("fips")["car_share"]
    weights_full["car_share_home"] = weights_full["fips_home_s"].map(car_map)
    n_missing_car = weights_full["car_share_home"].isna().sum()
    log.info("Full network: %d edges, %d missing car_share_home (%.1f%%)",
             len(weights_full), n_missing_car, 100 * n_missing_car / len(weights_full))
    weights_full["weight_x_car"] = weights_full["weight"] * weights_full["car_share_home"].fillna(
        weights_full["car_share_home"].mean())

    weights_dist, median_dist = dist_mod.build_weights_with_distance()
    weights_dist = weights_dist.merge(
        car[["fips", "car_share"]].rename(columns={"fips": "fips_home_s", "car_share": "car_share_home"}),
        on="fips_home_s", how="left")
    weights_dist["car_share_home"] = weights_dist["car_share_home"].fillna(weights_dist["car_share_home"].mean())
    weights_dist["weight_x_car"] = weights_dist["weight"] * weights_dist["car_share_home"]
    weights_dist["weight_x_dist"] = weights_dist["weight"] * weights_dist["dist_mi"]
    weights_dist["weight_x_car_x_dist"] = weights_dist["weight_x_car"] * weights_dist["dist_mi"]

    # NHTS-derived, DISTANCE-ADJUSTED car share (car_share_from_distance.py):
    # car mode share is not flat across trip distance -- it dips for very
    # short (walkable) trips, peaks around 20-30mi, and dips again for
    # long-haul trips that substitute to bus/rail/air. Using this instead
    # of the flat county-level ACS average corrects for exactly the bias
    # the flat number introduces: understating car use on short intra-
    # county hops relative to longer cross-county commutes.
    weights_dist["car_share_dist_adj"] = car_share_from_distance(weights_dist["dist_mi"])
    weights_dist["weight_x_car_dist_adj"] = weights_dist["weight"] * weights_dist["car_share_dist_adj"]
    weights_dist["weight_x_car_dist_adj_x_dist"] = weights_dist["weight_x_car_dist_adj"] * weights_dist["dist_mi"]

    # Same distance adjustment, but stratified by the HOME county's own
    # metro type (NYC's short trips are far more transit-substitutable
    # than a car-dependent Sunbelt metro's) -- see
    # build_nhts_car_share_by_distance.py's MSASIZE bucketing.
    weights_dist["car_share_msa_adj"] = car_share_from_distance_by_county(
        weights_dist["dist_mi"], weights_dist["fips_home_s"])
    weights_dist["weight_x_car_msa_adj"] = weights_dist["weight"] * weights_dist["car_share_msa_adj"]
    weights_dist["weight_x_car_msa_adj_x_dist"] = weights_dist["weight_x_car_msa_adj"] * weights_dist["dist_mi"]

    variants = {
        "full_network: weight only (headline reference)": (weights_full, "weight"),
        "full_network: weight x car_share_home": (weights_full, "weight_x_car"),
        "coverage_matched: weight only": (weights_dist, "weight"),
        "coverage_matched: weight x car_share_home": (weights_dist, "weight_x_car"),
        "coverage_matched: weight x dist (no car)": (weights_dist, "weight_x_dist"),
        "coverage_matched: weight x car_share_home x dist (full pair dosage)": (weights_dist, "weight_x_car_x_dist"),
        "coverage_matched: weight x car_share_DIST-ADJUSTED (NHTS)": (weights_dist, "weight_x_car_dist_adj"),
        "coverage_matched: weight x car_share_DIST-ADJUSTED x dist": (weights_dist, "weight_x_car_dist_adj_x_dist"),
        "coverage_matched: weight x car_share_METRO+DIST-ADJUSTED (NYC!=LA)": (weights_dist, "weight_x_car_msa_adj"),
        "coverage_matched: weight x car_share_METRO+DIST-ADJUSTED x dist": (weights_dist, "weight_x_car_msa_adj_x_dist"),
    }

    results = []
    for label, (w_subset, value_col) in variants.items():
        values, n_pairs, w_sum = dist_mod.spillover_series(
            grid_index, alert_events, w_subset, fips_in_sample, value_col=value_col)
        grid["_spill"] = values
        coef, se, pval = dist_mod.fit_spillover(grid, "_spill")
        sig = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else "n.s."
        log.info("[%s] beta=%+.6g se=%.6g p=%.4f %s (%d edges, weight sum=%.2f)",
                 label, coef, se, pval, sig, n_pairs, w_sum)
        results.append({"variant": label, "coef": coef, "se": se, "pval": pval,
                        "n_edges_lit": n_pairs, "weight_sum": w_sum})
        gc.collect()

    grid.drop(columns=["_spill"], inplace=True, errors="ignore")

    out = pd.DataFrame(results)
    out_path = OUTPUT_TABS / "reg_commuting_car_distance_dosage.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
