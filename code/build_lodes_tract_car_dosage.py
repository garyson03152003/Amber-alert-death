"""
build_lodes_tract_car_dosage.py
=============================================================
Extends build_lodes_county_pair_distance.py to carry TRACT-level car
share through the aggregation before collapsing to county pairs,
instead of applying a single flat number (whether ACS county average,
NHTS distance-curve, or MSASIZE-bucket) uniformly across a whole county
pair.

Why this is different from (and better than) every car-share variant
tried so far: a county is not homogeneous -- it can contain both a
dense, low-car-share urban tract and high-car-share suburban/rural
tracts. Applying one number to the whole county (or inferring car share
from distance/metro-type alone) can't capture that within-county
texture. Real per-tract ACS car share (build_acs_tract_car_share.py)
CAN, but only if it's joined at the TRACT-PAIR level, before any
aggregation -- joining it after the fact (e.g. onto an already-collapsed
county-pair table) is mathematically just re-deriving a flat county
average, which loses exactly the information this is meant to add.

Concretely, for each block-level LODES row this computes THREE
quantities before ever collapsing to county level:
    S000                              (the worker count, same as before)
    S000 * dist_mi                    (as in the distance pipeline)
    S000 * car_share_home_tract * dist_mi   (the new joint quantity)
so the per-county-pair average of (car_share x distance) reflects the
TRUE joint distribution across the tracts that make up the pair, not
the product of two separately-averaged marginals (which would silently
assume car share and distance are uncorrelated within a county pair --
almost certainly false, since a home tract close to the county's work-
county border is both shorter-distance AND, if that border area happens
to be less dense, more car-heavy than a home tract on the far side).

Same three-vintage time-weighting as build_lodes_county_pair_distance.py
(2013/2018/2022, weights 3/12, 4/12, 5/12). Raw per-state LODES files
were not retained from that run, so this re-downloads and re-processes
from scratch -- there's no way to reuse those checkpoints since they
only stored county-level sums, not the tract-level detail this needs.

Output:
  data/processed/commuting/county_pair_lodes_car_dosage.parquet
    Columns: fips_home, fips_work, total_workers, avg_dist_mi,
    avg_car_share, avg_car_x_dist
    (avg_car_x_dist is the joint quantity to use directly as the
    distance-and-car-weighted dosage; avg_car_share x avg_dist_mi would
    be the -- generally wrong -- independence-assuming approximation)
"""
import gc
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC
from utils import get_logger

log = get_logger("lodes_tract_car")

LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"
TRACT_CENTROID_PATH = CROSSWALK_RAW / "CenPop2020_Mean_TR.txt"
TRACT_CAR_SHARE_PATH = DATA_PROC / "tract_car_share.parquet"
CHECKPOINT_DIR = DATA_PROC / "commuting" / "_lodes_car_checkpoints"
YEAR_CACHE_DIR = DATA_PROC / "commuting" / "_lodes_car_year_cache"
OUT_PATH = DATA_PROC / "commuting" / "county_pair_lodes_car_dosage.parquet"
EARTH_RADIUS_MI = 3958.8

YEAR_WEIGHTS = {2013: 3 / 12, 2018: 4 / 12, 2022: 5 / 12}

STATES = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi",
    "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn",
    "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh",
    "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa",
    "wv", "wi", "wy",
]


def haversine_miles(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def load_tract_centroids() -> pd.DataFrame:
    df = pd.read_csv(TRACT_CENTROID_PATH, encoding="utf-8-sig",
                     dtype={"STATEFP": str, "COUNTYFP": str, "TRACTCE": str})
    df["tract"] = df["STATEFP"].str.zfill(2) + df["COUNTYFP"].str.zfill(3) + df["TRACTCE"].str.zfill(6)
    df = df.drop_duplicates(subset="tract").set_index("tract")
    return df[["LATITUDE", "LONGITUDE"]].rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})


def load_tract_car_share() -> pd.Series:
    df = pd.read_parquet(TRACT_CAR_SHARE_PATH)
    national_mean = df["car_total"].sum() / df["total_workers"].sum()
    s = df.set_index("tract")["car_share"].fillna(national_mean)
    log.info("Tract car share: %d tracts, national worker-weighted mean=%.3f (fallback for missing)",
             len(s), national_mean)
    return s, national_mean


def _download(url: str, tmp_path: Path, attempts: int = 3):
    last_exc = None
    for i in range(attempts):
        result = subprocess.run(
            ["curl", "-sS", "--fail", "--max-time", "180",
             "--retry", "2", "--retry-delay", "2", "-o", str(tmp_path), url],
            capture_output=True, text=True)
        if result.returncode == 0:
            return
        last_exc = RuntimeError(f"curl exit {result.returncode}: {result.stderr[:200]}")
    raise last_exc


def process_one_file(url: str, checkpoint_path: Path, tract_lat, tract_lon,
                     tract_car, car_fallback: float):
    if checkpoint_path.exists():
        log.info("  [cached] %s", checkpoint_path.name)
        return

    with tempfile.NamedTemporaryFile(suffix=".csv.gz") as tmp:
        t0 = time.time()
        try:
            _download(url, Path(tmp.name))
        except Exception as exc:
            log.warning("  download failed (%s), skipping: %s", url, exc)
            pd.DataFrame(columns=["h_county", "w_county", "weight_sum", "weighted_dist_sum",
                                  "weighted_car_sum", "weighted_car_dist_sum"]).to_parquet(checkpoint_path)
            return

        try:
            df = pd.read_csv(tmp.name, compression="gzip",
                             usecols=["w_geocode", "h_geocode", "S000"],
                             dtype={"w_geocode": str, "h_geocode": str, "S000": "int32"})
        except Exception as exc:
            log.warning("  read failed (%s), skipping: %s", url, exc)
            pd.DataFrame(columns=["h_county", "w_county", "weight_sum", "weighted_dist_sum",
                                  "weighted_car_sum", "weighted_car_dist_sum"]).to_parquet(checkpoint_path)
            return

    df["h_tract"] = df["h_geocode"].str[:11]
    df["w_tract"] = df["w_geocode"].str[:11]
    df["h_county"] = df["h_geocode"].str[:5]
    df["w_county"] = df["w_geocode"].str[:5]

    lat1 = df["h_tract"].map(tract_lat); lon1 = df["h_tract"].map(tract_lon)
    lat2 = df["w_tract"].map(tract_lat); lon2 = df["w_tract"].map(tract_lon)
    df["dist_mi"] = haversine_miles(lat1, lon1, lat2, lon2)
    df["car_share"] = df["h_tract"].map(tract_car).fillna(car_fallback)
    n_missing = df["dist_mi"].isna().sum()

    df["w_x_dist"] = df["S000"] * df["dist_mi"]
    df["w_x_car"] = df["S000"] * df["car_share"]
    df["w_x_car_x_dist"] = df["w_x_car"] * df["dist_mi"]

    agg = (df.dropna(subset=["dist_mi"])
           .groupby(["h_county", "w_county"], as_index=False)
           .agg(weight_sum=("S000", "sum"), weighted_dist_sum=("w_x_dist", "sum"),
                weighted_car_sum=("w_x_car", "sum"), weighted_car_dist_sum=("w_x_car_x_dist", "sum")))
    agg.to_parquet(checkpoint_path, index=False)

    elapsed = time.time() - t0
    log.info("  %s: %d rows -> %d county pairs (%.1f%% missing tract centroid) [%.0fs]",
             checkpoint_path.name, len(df), len(agg), 100 * n_missing / max(len(df), 1), elapsed)
    del df, agg
    gc.collect()


def build_year(year: int, tract_lat, tract_lon, tract_car, car_fallback: float) -> Path:
    YEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    year_out = YEAR_CACHE_DIR / f"county_pair_lodes_car_dosage_{year}.parquet"
    if year_out.exists():
        log.info("[year %d] already combined -> %s", year, year_out)
        return year_out

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for i, state in enumerate(STATES, 1):
        log.info("[year %d] [%d/%d] %s", year, i, len(STATES), state.upper())
        for filetype in ("main", "aux"):
            url = f"{LODES_BASE}/{state}/od/{state}_od_{filetype}_JT00_{year}.csv.gz"
            ckpt = CHECKPOINT_DIR / f"{state}_{filetype}_{year}.parquet"
            process_one_file(url, ckpt, tract_lat, tract_lon, tract_car, car_fallback)

    log.info("[year %d] all states processed, combining...", year)
    parts = [pd.read_parquet(p) for p in sorted(CHECKPOINT_DIR.glob(f"*_{year}.parquet"))]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(["h_county", "w_county"], as_index=False).agg(
        weight_sum=("weight_sum", "sum"), weighted_dist_sum=("weighted_dist_sum", "sum"),
        weighted_car_sum=("weighted_car_sum", "sum"), weighted_car_dist_sum=("weighted_car_dist_sum", "sum"))
    combined["avg_dist_mi"] = combined["weighted_dist_sum"] / combined["weight_sum"].clip(lower=1)
    combined["avg_car_share"] = combined["weighted_car_sum"] / combined["weight_sum"].clip(lower=1)
    combined["avg_car_x_dist"] = combined["weighted_car_dist_sum"] / combined["weight_sum"].clip(lower=1)
    combined = combined.rename(columns={"h_county": "fips_home", "w_county": "fips_work",
                                        "weight_sum": "total_workers"})
    keep = ["fips_home", "fips_work", "total_workers", "avg_dist_mi", "avg_car_share", "avg_car_x_dist"]
    combined[keep].to_parquet(year_out, index=False)
    log.info("[year %d] saved %d county pairs -> %s", year, len(combined), year_out)
    return year_out


def combine_years(year_paths: dict) -> pd.DataFrame:
    frames = []
    for year, path in year_paths.items():
        df = pd.read_parquet(path)
        w = YEAR_WEIGHTS[year] * df["total_workers"]
        frames.append(pd.DataFrame({
            "fips_home": df["fips_home"], "fips_work": df["fips_work"],
            "total_workers": df["total_workers"], "w": w,
            "w_x_dist": w * df["avg_dist_mi"],
            "w_x_car": w * df["avg_car_share"],
            "w_x_car_x_dist": w * df["avg_car_x_dist"],
        }))

    stacked = pd.concat(frames, ignore_index=True)
    out = stacked.groupby(["fips_home", "fips_work"], as_index=False).agg(
        total_workers=("total_workers", "sum"), w=("w", "sum"),
        w_x_dist=("w_x_dist", "sum"), w_x_car=("w_x_car", "sum"), w_x_car_x_dist=("w_x_car_x_dist", "sum"))
    out["avg_dist_mi"] = out["w_x_dist"] / out["w"].clip(lower=1e-9)
    out["avg_car_share"] = out["w_x_car"] / out["w"].clip(lower=1e-9)
    out["avg_car_x_dist"] = out["w_x_car_x_dist"] / out["w"].clip(lower=1e-9)
    return out[["fips_home", "fips_work", "total_workers", "avg_dist_mi", "avg_car_share", "avg_car_x_dist"]]


def main():
    tract_cent = load_tract_centroids()
    tract_lat, tract_lon = tract_cent["lat"], tract_cent["lon"]
    tract_car, car_fallback = load_tract_car_share()

    year_paths = {}
    for year in YEAR_WEIGHTS:
        year_paths[year] = build_year(year, tract_lat, tract_lon, tract_car, car_fallback)

    log.info("Combining %d vintages with time weights %s...", len(year_paths), YEAR_WEIGHTS)
    out = combine_years(year_paths)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d county pairs -> %s", len(out), OUT_PATH)


if __name__ == "__main__":
    main()
