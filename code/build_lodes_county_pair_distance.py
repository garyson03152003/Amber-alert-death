"""
build_lodes_county_pair_distance.py
=============================================================
Replaces the single centroid-to-centroid distance per county pair (used
in run_commuting_distance_robustness.py) with the TRUE worker-weighted
average commuting distance for that pair, computed from LEHD LODES
Origin-Destination (OD) data -- home/work flows at the Census BLOCK
level (finer than tract), the most granular commuting flow data the
Census Bureau publishes.

Why this matters: a single county-centroid-to-county-centroid distance
assumes every commuter in a county pair travels the same distance. In a
large county, that's a poor approximation -- e.g. a home block near the
county's work-county border commutes a fraction of the distance a block
on the far side of the same county would. LODES lets us instead average
over every real home-block/work-block pair, weighted by actual worker
counts, which is the actual quantity "average commuting distance for
this county pair" is supposed to mean.

Multiple vintages, time-weighted
---------------------------------
The AMBER alert study spans 2013-2024, but commuting patterns aren't
static over 12 years (least of all across the 2020-2021 remote-work
shock). Rather than pick one arbitrary LODES year, this pulls THREE
vintages spanning the study period (2013, 2018, 2022 -- the most recent
year with data at time of writing; LODES has none for 2024, and 2023
was skipped in favor of already-completed 2022 downloads since the two
are one year apart and commuting DISTANCE, unlike commute volume, is
not expected to move much year to year) and combines them with weights
equal to each vintage's share of study years it is closest to:
    2013 -> nearest for study years 2013-2015 (3/12 = 0.250)
    2018 -> nearest for study years 2016-2020 (5/12 = 0.417)
    2022 -> nearest for study years 2021-2024 (4/12 = 0.333)
For each county pair, the combined average distance further weights by
that vintage's own worker count (so a pair with much more 2022 volume
than 2013 volume leans toward its 2022 distance, on top of the study-year
weight) and only uses vintages where the pair actually has data --
see combine_years().

Method (block-level data is too large to hold nationally at once, so
this aggregates in two stages per state to keep memory bounded):
  1. For each state, download its LODES8 OD "main" file (jobs with home
     AND work in that state) and "aux" file (jobs with work in that
     state, home in a DIFFERENT state) -- together these cover every job
     whose WORKPLACE is in that state, regardless of home state, so
     looping over all 50 states + DC covers every job nationally exactly
     once (a home/work pair is only ever present in the work-state's
     main or aux file, never both, never in another state's files).
  2. Derive home/work TRACT (first 11 of the 15-digit block geocode) and
     COUNTY (first 5) directly by string slicing -- no separate
     block/tract crosswalk needed.
  3. Join tract population-weighted centroids (CenPop2020_Mean_TR.txt,
     same Census "center of population" product used for counties in
     build_county_pop_centroids.py) and compute haversine distance per
     block-pair row.
  4. Aggregate to (h_county, w_county): sum(S000) and sum(S000 * dist),
     checkpointed to disk per state/filetype/YEAR so a killed/restarted
     run doesn't re-download or re-process completed (state, year)
     combinations.

Output:
  data/processed/commuting/county_pair_lodes_distance.parquet
    Columns: fips_home, fips_work, total_workers, avg_dist_mi
    (total_workers/avg_dist_mi are combined across the 3 vintages --
    see combine_years())
"""
import gc
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC
from utils import get_logger

log = get_logger("lodes_distance")

LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"
TRACT_CENTROID_PATH = CROSSWALK_RAW / "CenPop2020_Mean_TR.txt"
CHECKPOINT_DIR = DATA_PROC / "commuting" / "_lodes_checkpoints"
RAW_CACHE_DIR = DATA_PROC / "commuting" / "_lodes_raw_cache"
YEAR_CACHE_DIR = DATA_PROC / "commuting" / "_lodes_year_cache"
OUT_PATH = DATA_PROC / "commuting" / "county_pair_lodes_distance.parquet"
EARTH_RADIUS_MI = 3958.8

# year -> share of the 2013-2024 study period this vintage represents
# (each study year assigned to its nearest available LODES vintage)
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
    log.info("Loaded %d tract centroids", len(df))
    return df[["LATITUDE", "LONGITUDE"]].rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})


def process_one_file(url: str, checkpoint_path: Path, tract_lat: pd.Series, tract_lon: pd.Series):
    if checkpoint_path.exists():
        log.info("  [cached] %s", checkpoint_path.name)
        return

    tmp_path = RAW_CACHE_DIR / (checkpoint_path.stem + ".csv.gz")
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except Exception as exc:
        log.warning("  download failed (%s), skipping: %s", url, exc)
        # write an empty checkpoint so we don't retry a genuinely-missing file forever
        pd.DataFrame(columns=["h_county", "w_county", "weight_sum", "weighted_dist_sum"]).to_parquet(checkpoint_path)
        return

    try:
        df = pd.read_csv(tmp_path, compression="gzip",
                         usecols=["w_geocode", "h_geocode", "S000"],
                         dtype={"w_geocode": str, "h_geocode": str, "S000": "int32"})
    except Exception as exc:
        log.warning("  read failed (%s), skipping: %s", url, exc)
        tmp_path.unlink(missing_ok=True)
        pd.DataFrame(columns=["h_county", "w_county", "weight_sum", "weighted_dist_sum"]).to_parquet(checkpoint_path)
        return

    df["h_tract"] = df["h_geocode"].str[:11]
    df["w_tract"] = df["w_geocode"].str[:11]
    df["h_county"] = df["h_geocode"].str[:5]
    df["w_county"] = df["w_geocode"].str[:5]

    lat1 = df["h_tract"].map(tract_lat); lon1 = df["h_tract"].map(tract_lon)
    lat2 = df["w_tract"].map(tract_lat); lon2 = df["w_tract"].map(tract_lon)
    df["dist_mi"] = haversine_miles(lat1, lon1, lat2, lon2)
    n_missing = df["dist_mi"].isna().sum()

    df["w_x_dist"] = df["S000"] * df["dist_mi"]
    agg = (df.dropna(subset=["dist_mi"])
           .groupby(["h_county", "w_county"], as_index=False)
           .agg(weight_sum=("S000", "sum"), weighted_dist_sum=("w_x_dist", "sum")))
    agg.to_parquet(checkpoint_path, index=False)

    elapsed = time.time() - t0
    log.info("  %s: %d rows -> %d county pairs (%.1f%% missing tract centroid) [%.0fs]",
             checkpoint_path.name, len(df), len(agg), 100 * n_missing / max(len(df), 1), elapsed)

    tmp_path.unlink(missing_ok=True)
    del df, agg
    gc.collect()


def build_year(year: int, tract_lat: pd.Series, tract_lon: pd.Series) -> Path:
    """Runs the 51-state main+aux pull for one LODES vintage; returns the
    path to that year's combined (fips_home, fips_work, total_workers,
    avg_dist_mi) parquet, itself cached so re-running is a no-op."""
    YEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    year_out = YEAR_CACHE_DIR / f"county_pair_lodes_distance_{year}.parquet"
    if year_out.exists():
        log.info("[year %d] already combined -> %s", year, year_out)
        return year_out

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for i, state in enumerate(STATES, 1):
        log.info("[year %d] [%d/%d] %s", year, i, len(STATES), state.upper())
        for filetype in ("main", "aux"):
            url = f"{LODES_BASE}/{state}/od/{state}_od_{filetype}_JT00_{year}.csv.gz"
            ckpt = CHECKPOINT_DIR / f"{state}_{filetype}_{year}.parquet"
            process_one_file(url, ckpt, tract_lat, tract_lon)

    log.info("[year %d] all states processed, combining...", year)
    parts = [pd.read_parquet(p) for p in sorted(CHECKPOINT_DIR.glob(f"*_{year}.parquet"))]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(["h_county", "w_county"], as_index=False).agg(
        weight_sum=("weight_sum", "sum"), weighted_dist_sum=("weighted_dist_sum", "sum"))
    combined["avg_dist_mi"] = combined["weighted_dist_sum"] / combined["weight_sum"].clip(lower=1)
    combined = combined.rename(columns={"h_county": "fips_home", "w_county": "fips_work",
                                        "weight_sum": "total_workers"})
    combined[["fips_home", "fips_work", "total_workers", "avg_dist_mi"]].to_parquet(year_out, index=False)
    log.info("[year %d] saved %d county pairs -> %s", year, len(combined), year_out)
    return year_out


def combine_years(year_paths: dict) -> pd.DataFrame:
    """Combines multiple vintages' (total_workers, avg_dist_mi) into one
    time-weighted average per county pair:
        combined_dist = sum_y(study_weight_y * workers_y * dist_y)
                       / sum_y(study_weight_y * workers_y)
    using only the vintages where a given pair actually has data (a pair
    absent from one vintage just drops out of that year's term, rather
    than being treated as zero)."""
    frames = []
    for year, path in year_paths.items():
        df = pd.read_parquet(path)
        df["study_weight"] = YEAR_WEIGHTS[year]
        df["w"] = df["study_weight"] * df["total_workers"]
        df["w_x_dist"] = df["w"] * df["avg_dist_mi"]
        frames.append(df[["fips_home", "fips_work", "total_workers", "w", "w_x_dist"]])

    stacked = pd.concat(frames, ignore_index=True)
    out = stacked.groupby(["fips_home", "fips_work"], as_index=False).agg(
        total_workers=("total_workers", "sum"), w=("w", "sum"), w_x_dist=("w_x_dist", "sum"))
    out["avg_dist_mi"] = out["w_x_dist"] / out["w"].clip(lower=1e-9)
    return out[["fips_home", "fips_work", "total_workers", "avg_dist_mi"]]


def main():
    cent = load_tract_centroids()
    tract_lat, tract_lon = cent["lat"], cent["lon"]

    year_paths = {}
    for year in YEAR_WEIGHTS:
        year_paths[year] = build_year(year, tract_lat, tract_lon)

    log.info("Combining %d vintages with time weights %s...", len(year_paths), YEAR_WEIGHTS)
    out = combine_years(year_paths)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d county pairs -> %s", len(out), OUT_PATH)
    log.info("Total (study-weighted) worker mass covered: %.0f", out["total_workers"].sum())


if __name__ == "__main__":
    main()
