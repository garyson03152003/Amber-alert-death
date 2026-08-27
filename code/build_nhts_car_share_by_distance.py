"""
build_nhts_car_share_by_distance.py
=============================================================
Replaces the flat, county-level ACS car_share (used identically for both
the intra-county "own" dosage and the cross-county "spillover" dosage)
with a car-share-BY-DISTANCE curve estimated from the National Household
Travel Survey (2017) trip microdata.

Stratified by metro type (NYC != LA)
-------------------------------------
A single national distance-car-share curve still treats a short trip in
transit-rich Manhattan the same as a short trip in car-dependent
suburban LA. NHTS's MSASIZE variable lets us split the curve by metro
size AND (for the two largest-metro categories) whether the metro has
rail transit at all:
    1 = not in an MSA
    2 = MSA < 250,000
    3 = MSA 250,000-499,999
    4 = MSA 500,000-999,999
    5 = MSA 1,000,000+, WITH rail transit  (e.g. NYC, Chicago, DC, Boston)
    6 = MSA 1,000,000+, WITHOUT rail transit (e.g. many Sunbelt metros)
(confirmed empirically: NHTS's household-level RAIL variable is 0% "has
rail" for every MSASIZE=6 respondent and a large, non-trivial share for
MSASIZE=5, consistent with 5/6 being the with/without-rail split of the
same size bucket.)

Each of our counties is assigned to one of these 6 buckets via
build_county_msasize_bucket() -- CBSA membership and population from the
Census CBSA delineation file (cbsa_delineation_2023.xlsx) rolled up from
county population (co-est2023-alldata.csv), with the 1M+ split done by
matching against a hand-compiled list of CBSAs with a meaningful rail
transit system (RAIL_TRANSIT_METRO_KEYWORDS below -- inevitably a
judgment call at the margin, e.g. minimal streetcar-only systems are
excluded, but unambiguous for the cases that matter most: NYC, LA,
Chicago, the Bay Area, DC, Boston, etc.).

Motivation: applying one flat car_share number to both regimes assumes
mode choice doesn't depend on trip distance, which is false in an
important direction -- very short trips substitute toward walking/biking
(lower car share), while very long trips substitute toward bus/rail/air
(also lower car share), with a broad middle range (roughly 5-50 miles,
which covers essentially all of our commuting pairs) where car share is
consistently high (~90-93%). Confirmed directly from NHTS commute-trip
(WHYTRP1S == 10, "go to work") microdata, weighted by WTTRDFIN:

    distance bin   car share
    0-2 mi         65.3%   <- short walkable/bikeable trips
    2-5 mi         89.4%
    5-10 mi        90.9%
    10-15 mi       92.2%
    15-20 mi       91.7%
    20-30 mi       93.3%   <- peak
    30-50 mi       90.8%
    50-75 mi       86.6%
    75-100 mi      87.3%
    100-150 mi     65.9%   <- long-haul substitution to bus/rail/air
    150-250 mi     64.1%
    250-500 mi     63.2%
    500-1000 mi    60.1%
    1000+ mi       75.2%   (n=17, noisy -- essentially irrelevant weight
                            in our commuting data anyway)

"Car" = TRPTRANS in {3,4,5,6,7,8,9,10} (car/SUV/van/pickup/other truck/RV/
motorcycle/golf-cart-type private vehicle codes; empirically confirmed by
each code's short-to-medium trip-distance profile). Excludes walk(1),
bike(2), all transit/rail/bus/taxi/rental codes, and airplane(19, median
trip distance ~785mi in the raw data -- unambiguously long-haul flights,
dropped entirely as irrelevant to road commuting).

Output:
  data/processed/nhts_car_share_by_distance.parquet
    Columns: dist_bin, mean_dist, weighted_n, car_share
  Also exposes car_share_from_distance(dist_mi) for other scripts to
  import directly (log-distance interpolation against bin midpoints,
  clipped to the empirical range).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC, DATA_RAW
from utils import get_logger

log = get_logger("nhts_car_share")

NHTS_URL = "https://nhts.ornl.gov/assets/2016/download/Csv.zip"
NHTS_RAW_ZIP = DATA_RAW / "nhts" / "Csv.zip"
OUT_PATH = DATA_PROC / "nhts_car_share_by_distance.parquet"
COUNTY_BUCKET_PATH = DATA_PROC / "county_msasize_bucket.parquet"
CBSA_DELINEATION_PATH = CROSSWALK_RAW / "cbsa_delineation_2023.xlsx"
CBSA_DELINEATION_URL = ("https://www2.census.gov/programs-surveys/metro-micro/geographies/"
                        "reference-files/2023/delineation-files/list1_2023.xlsx")

CAR_CODES = {3, 4, 5, 6, 7, 8, 9, 10}
WORK_PURPOSE = 10
AIRPLANE_CODE = 19

BIN_EDGES = [0, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 500, 1000, 10000]
BIN_LABELS = ["0-2", "2-5", "5-10", "10-15", "15-20", "20-30", "30-50",
             "50-75", "75-100", "100-150", "150-250", "250-500",
             "500-1000", "1000+"]

# NHTS MSASIZE buckets 1-6 (see module docstring). Coarser distance bins
# for the smaller buckets since NHTS sample sizes shrink fast once split
# 6 ways -- a bucket with too few trips per fine bin would just be noise.
MSASIZE_BIN_EDGES = [0, 5, 10, 20, 50, 10000]
MSASIZE_BIN_LABELS = ["0-5", "5-10", "10-20", "20-50", "50+"]

# Metro areas with a meaningful heavy/light/commuter rail transit system
# (subway, light rail, or commuter rail carrying a non-trivial ridership
# share) -- matched as a substring against the CBSA delineation file's
# "CBSA Title" column. Deliberately excludes minimal streetcar-only
# systems (e.g. a single downtown loop) as not materially changing mode
# choice. Judgment call at the margin; unambiguous for the metros that
# matter most to the "NYC != LA" distinction this stratification exists for.

# Deliberately just the primary/first city name, not the full official
# multi-city CBSA title -- matching on the FULL title is brittle to the
# Census Bureau periodically re-ordering/renaming CBSA titles (verified:
# an earlier version of this list used "Houston-The Woodlands" and
# "Cleveland-Elyria", which no longer match the current 2023 delineation
# file's "Houston-Pasadena-The Woodlands, TX" / "Cleveland, OH" -- both
# genuinely have light/heavy rail and were silently misclassified until
# this was switched to first-city-name matching).
RAIL_TRANSIT_METRO_KEYWORDS = [
    "New York-Newark", "Chicago-Naperville", "Washington-Arlington",
    "Boston-Cambridge", "Philadelphia-Camden", "San Francisco-Oakland",
    "Atlanta-Sandy Springs", "Los Angeles-Long Beach", "Miami-Fort Lauderdale",
    "Baltimore-Columbia", "Seattle-Tacoma", "Portland-Vancouver-Hillsboro",
    "Denver-Aurora", "Dallas-Fort Worth", "Houston-",
    "San Diego-Chula Vista", "Minneapolis-St. Paul", "St. Louis",
    "Cleveland,", "Sacramento-Roseville", "Salt Lake City",
    "Charlotte-Concord", "Phoenix-Mesa-Chandler", "Buffalo-Cheektowaga",
    "Honolulu", "Virginia Beach-Chesapeake-Norfolk", "San Juan",
]


def download_if_missing():
    NHTS_RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if NHTS_RAW_ZIP.exists():
        log.info("Already downloaded: %s", NHTS_RAW_ZIP)
        return
    log.info("Downloading NHTS 2017 trip data (~84MB)...")
    urllib.request.urlretrieve(NHTS_URL, NHTS_RAW_ZIP)
    log.info("Saved to %s", NHTS_RAW_ZIP)


def download_cbsa_delineation_if_missing():
    CBSA_DELINEATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CBSA_DELINEATION_PATH.exists():
        return
    log.info("Downloading Census CBSA delineation file...")
    urllib.request.urlretrieve(CBSA_DELINEATION_URL, CBSA_DELINEATION_PATH)


def build_county_msasize_bucket() -> pd.DataFrame:
    """Assigns every US county to an NHTS-MSASIZE-equivalent bucket
    (1=not in MSA ... 6=1M+ metro without rail), using the Census CBSA
    delineation file (county -> CBSA) and county population estimates
    (CBSA population = sum of its member counties' population)."""
    if COUNTY_BUCKET_PATH.exists():
        return pd.read_parquet(COUNTY_BUCKET_PATH)

    download_cbsa_delineation_if_missing()
    cbsa = pd.read_excel(CBSA_DELINEATION_PATH, skiprows=2)
    cbsa = cbsa.dropna(subset=["CBSA Code", "FIPS State Code", "FIPS County Code"]).copy()
    cbsa["fips"] = (cbsa["FIPS State Code"].astype(int).astype(str).str.zfill(2) +
                    cbsa["FIPS County Code"].astype(int).astype(str).str.zfill(3))
    cbsa = cbsa[["fips", "CBSA Code", "CBSA Title",
                "Metropolitan/Micropolitan Statistical Area"]].rename(
        columns={"CBSA Code": "cbsa_code", "CBSA Title": "cbsa_title",
                "Metropolitan/Micropolitan Statistical Area": "cbsa_type"})

    pop = pd.read_csv(CROSSWALK_RAW / "co-est2023-alldata.csv", encoding="latin-1",
                      dtype={"STATE": str, "COUNTY": str})
    pop = pop[pop["COUNTY"] != "000"].copy()
    pop["fips"] = pop["STATE"].str.zfill(2) + pop["COUNTY"].str.zfill(3)
    pop = pop[["fips", "POPESTIMATE2023"]].rename(columns={"POPESTIMATE2023": "population"})

    merged = cbsa.merge(pop, on="fips", how="left")
    cbsa_pop = merged.groupby("cbsa_code")["population"].transform("sum")
    merged["cbsa_population"] = cbsa_pop

    def bucket_row(row):
        if row["cbsa_type"] != "Metropolitan Statistical Area":
            return 1  # micropolitan or unclassified -> treat as "not in MSA"
        p = row["cbsa_population"]
        if pd.isna(p) or p < 250_000:
            return 2
        if p < 500_000:
            return 3
        if p < 1_000_000:
            return 4
        has_rail = any(kw in str(row["cbsa_title"]) for kw in RAIL_TRANSIT_METRO_KEYWORDS)
        return 5 if has_rail else 6

    merged["msasize"] = merged.apply(bucket_row, axis=1)
    all_counties = pop[["fips"]].copy()
    out = all_counties.merge(merged[["fips", "msasize", "cbsa_title"]], on="fips", how="left")
    out["msasize"] = out["msasize"].fillna(1).astype(int)  # no CBSA row at all -> not in MSA

    log.info("County MSASIZE bucket assignment: %s", out["msasize"].value_counts().sort_index().to_dict())
    COUNTY_BUCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(COUNTY_BUCKET_PATH, index=False)
    return out


def build() -> pd.DataFrame:
    download_if_missing()
    with zipfile.ZipFile(NHTS_RAW_ZIP) as z, z.open("trippub.csv") as f:
        df = pd.read_csv(f, usecols=["TRPMILES", "TRPTRANS", "WHYTRP1S", "WTTRDFIN", "MSASIZE"])

    commute = df[(df["WHYTRP1S"] == WORK_PURPOSE) & (df["TRPTRANS"] > 0) &
                (df["TRPTRANS"] != AIRPLANE_CODE) & (df["TRPMILES"] > 0)].copy()
    commute["is_car"] = commute["TRPTRANS"].isin(CAR_CODES).astype(int)
    log.info("Commute trips (valid, non-airplane): %d", len(commute))
    log.info("Overall weighted car share: %.1f%%",
             100 * np.average(commute["is_car"], weights=commute["WTTRDFIN"]))

    def wavg(g):
        return pd.Series({
            "n": len(g),
            "weighted_n": g["WTTRDFIN"].sum(),
            "car_share": np.average(g["is_car"], weights=g["WTTRDFIN"]),
            "mean_dist": g["TRPMILES"].mean(),
        })

    # National (pooled) curve -- fine bins, used as the fallback when no
    # MSASIZE bucket is available.
    commute["dist_bin"] = pd.cut(commute["TRPMILES"], bins=BIN_EDGES, labels=BIN_LABELS, right=False)
    national = commute.groupby("dist_bin", observed=True).apply(wavg).reset_index()
    national["msasize"] = 0  # 0 = pooled/national
    national = national.sort_values("mean_dist").reset_index(drop=True)
    log.info("National car share by distance bin:\n%s", national.to_string())

    # Per-MSASIZE-bucket curves -- coarser bins (see MSASIZE_BIN_EDGES).
    commute["dist_bin_coarse"] = pd.cut(commute["TRPMILES"], bins=MSASIZE_BIN_EDGES,
                                        labels=MSASIZE_BIN_LABELS, right=False)
    by_msa = (commute.groupby(["MSASIZE", "dist_bin_coarse"], observed=True)
             .apply(wavg).reset_index().rename(columns={"MSASIZE": "msasize", "dist_bin_coarse": "dist_bin"}))
    by_msa = by_msa.sort_values(["msasize", "mean_dist"]).reset_index(drop=True)
    log.info("Per-MSASIZE car share by distance bin:\n%s", by_msa.to_string())

    result = pd.concat([national[["msasize", "dist_bin", "n", "weighted_n", "car_share", "mean_dist"]],
                       by_msa[["msasize", "dist_bin", "n", "weighted_n", "car_share", "mean_dist"]]],
                      ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)
    log.info("Saved -> %s", OUT_PATH)
    return result


_lookup_cache = None


def car_share_from_distance(dist_mi, msasize=None):
    """Vectorized: given an array/Series of commute distances in miles,
    returns the NHTS-derived car share at that distance via log-distance
    interpolation against the empirical bin midpoints, clipped to the
    observed range (no extrapolation beyond the shortest/longest bin).

    msasize: None/0 for the pooled national curve; 1-6 for a specific
    MSASIZE bucket (falls back to the national curve if that bucket has
    fewer than 2 usable distance bins, e.g. an under-sampled combination)."""
    global _lookup_cache
    if _lookup_cache is None:
        if not OUT_PATH.exists():
            build()
        _lookup_cache = pd.read_parquet(OUT_PATH)

    key = 0 if msasize is None else msasize
    sub = _lookup_cache[_lookup_cache["msasize"] == key].sort_values("mean_dist")
    if len(sub) < 2:
        sub = _lookup_cache[_lookup_cache["msasize"] == 0].sort_values("mean_dist")

    x = np.log(sub["mean_dist"].to_numpy())
    y = sub["car_share"].to_numpy()
    dist_arr = np.asarray(dist_mi, dtype=float)
    log_dist = np.log(np.clip(dist_arr, 0.01, None))
    return np.interp(log_dist, x, y)


def car_share_from_distance_by_county(dist_mi, fips):
    """Same as car_share_from_distance, but looks up each row's own
    county's MSASIZE bucket (via build_county_msasize_bucket()) instead
    of using one pooled curve for everyone -- so a short trip in NYC and
    a short trip in a small non-metro county get different car shares."""
    buckets = build_county_msasize_bucket().set_index("fips")["msasize"]
    fips_arr = pd.Series(fips).astype(str).str.zfill(5)
    msasize_arr = fips_arr.map(buckets).fillna(1).astype(int).to_numpy()
    dist_arr = np.asarray(dist_mi, dtype=float)

    out = np.empty(len(dist_arr), dtype=float)
    for bucket in np.unique(msasize_arr):
        mask = msasize_arr == bucket
        out[mask] = car_share_from_distance(dist_arr[mask], msasize=int(bucket))
    return out


if __name__ == "__main__":
    build()
    build_county_msasize_bucket()
