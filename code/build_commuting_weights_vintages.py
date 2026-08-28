"""Build county-to-county commuting weights for MULTIPLE ACS vintages.

Why more than one vintage
-------------------------
The crash panel runs 2013-2024 but the spillover treatment has been built
from a single ACS 2016-2020 flow matrix, so the earliest years are matched
against commuting patterns measured up to seven years later. Census
publishes a 2011-2015 5-year table as well, which is temporally much closer
to the start of the panel:

    2011-2015 flows  ->  crash years 2013-2017
    2016-2020 flows  ->  crash years 2018-2024

Header detection
----------------
The two workbooks are NOT laid out identically: the 2020 table needs
``skiprows=7`` and the 2015 table needs ``skiprows=6``. Rather than hardcode
a number per vintage, the header row is located by searching for the
expected column names -- a layout change in a future release then fails
loudly instead of silently parsing the wrong row as the header.

Output: data/processed/commuting/county_commuting_weights_{vintage}.parquet
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

log = get_logger("commuting_vintages")

DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "commuting"

VINTAGES = {
    "2015": ("https://www2.census.gov/programs-surveys/demo/tables/metro-micro/"
             "2015/commuting-flows-2015/table1.xlsx"),
    "2020": ("https://www2.census.gov/programs-surveys/demo/tables/metro-micro/"
             "2020/commuting-flows-2020/table1.xlsx"),
}

# crash year -> flow vintage, choosing the temporally nearest measurement
VINTAGE_FOR_YEAR = {y: ("2015" if y <= 2017 else "2020") for y in range(2013, 2026)}

REQUIRED = ["State FIPS Code", "County FIPS Code",
            "State FIPS Code.1", "County FIPS Code.1",
            "Workers in Commuting Flow"]


def _find_header_row(path: Path, max_skip: int = 12) -> int:
    """Locate the header row by content rather than assuming a fixed offset."""
    for skip in range(max_skip):
        try:
            probe = pd.read_excel(path, skiprows=skip, header=0, nrows=1)
        except Exception:                                    # noqa: BLE001
            continue
        if all(any(str(c) == want for c in probe.columns) for want in REQUIRED):
            return skip
    raise ValueError(
        f"could not locate the expected header row in {path.name}; "
        f"the Census layout may have changed (looked for {REQUIRED})"
    )


def build_vintage(vintage: str) -> pd.DataFrame:
    url = VINTAGES[vintage]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    xlsx = DATA_DIR / f"acs_commuting_flows_{vintage}.xlsx"
    if not xlsx.exists():
        log.info("[%s] downloading %s", vintage, url)
        urllib.request.urlretrieve(url, xlsx)

    skip = _find_header_row(xlsx)
    log.info("[%s] header row detected at skiprows=%d", vintage, skip)
    raw = pd.read_excel(xlsx, skiprows=skip, header=0)
    raw = raw.dropna(subset=["Workers in Commuting Flow"])
    raw = raw.dropna(subset=["State FIPS Code.1", "County FIPS Code.1"])

    raw["fips_home"] = (raw["State FIPS Code"].astype(float).astype(int) * 1000
                        + raw["County FIPS Code"].astype(float).astype(int))
    raw["fips_work"] = (raw["State FIPS Code.1"].astype(float).astype(int) * 1000
                        + raw["County FIPS Code.1"].astype(float).astype(int))
    raw["workers"] = raw["Workers in Commuting Flow"].astype(float)

    # 50 states + DC only; territories have state FIPS > 56.
    raw = raw[(raw["fips_home"] // 1000 <= 56) & (raw["fips_work"] // 1000 <= 56)].copy()

    totals = raw.groupby("fips_work")["workers"].sum().rename("total_to_work")
    raw = raw.merge(totals, on="fips_work")
    raw["weight"] = raw["workers"] / raw["total_to_work"]

    out = raw[["fips_home", "fips_work", "workers", "weight"]].copy()
    path = DATA_DIR / f"county_commuting_weights_{vintage}.parquet"
    out.to_parquet(path, index=False)
    log.info("[%s] %s OD pairs | %s work counties | %s home counties -> %s",
             vintage, f"{len(out):,}", out["fips_work"].nunique(),
             out["fips_home"].nunique(), path.name)
    return out


def load_flows_for_year(year: int) -> pd.DataFrame:
    """Return the flow matrix whose ACS window is nearest to ``year``."""
    vintage = VINTAGE_FOR_YEAR.get(year, "2020")
    path = DATA_DIR / f"county_commuting_weights_{vintage}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"flow vintage {vintage} not built: {path}")
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vintages", nargs="*", default=list(VINTAGES),
                    choices=list(VINTAGES))
    args = ap.parse_args()
    built = {v: build_vintage(v) for v in args.vintages}

    if len(built) > 1:
        a, b = built["2015"], built["2020"]
        merged = a.merge(b, on=["fips_home", "fips_work"], suffixes=("_15", "_20"))
        log.info("overlap: %s shared OD pairs | corr(weight) = %.4f",
                 f"{len(merged):,}", merged["weight_15"].corr(merged["weight_20"]))
        log.info("pairs only in 2015: %s | only in 2020: %s",
                 f"{len(a) - len(merged):,}", f"{len(b) - len(merged):,}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
