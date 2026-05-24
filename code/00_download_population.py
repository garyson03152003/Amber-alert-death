"""
00_download_population.py — Download Census county population estimates (2013–2022).

Used in 04_build_panel.py for population-density heterogeneity analysis.

Source: Census Bureau Population Estimates Program (PEP)
  https://www.census.gov/data/datasets/time-series/demo/popest/2020s-counties-total.html

Output: data/processed/county_population.parquet
    Columns: fips, year, population

Run: python code/00_download_population.py
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC, STUDY_YEARS
from utils import get_logger, download_file

log = get_logger("00_population")

# 2020s-vintage estimates (covers 2020–2023)
PEP_2020S_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/counties/totals/co-est2023-alldata.csv"
)
# 2010s-vintage estimates (covers 2010–2019)
PEP_2010S_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2010-2020/counties/totals/co-est2020-alldata.csv"
)


def fetch_pep(url: str, session: requests.Session) -> pd.DataFrame:
    """Download and parse a Census PEP alldata CSV."""
    fname = url.split("/")[-1]
    dest = CROSSWALK_RAW / fname
    download_file(url, dest, session=session)

    df = pd.read_csv(dest, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    # STATE and COUNTY are already numeric FIPS components
    df = df[df["COUNTY"] != 0]   # drop state-level summary rows
    df["fips"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(str).str.zfill(3)

    # Population estimate columns are named POPESTIMATE{YEAR}
    pop_cols = [c for c in df.columns if c.startswith("POPESTIMATE")]
    df_long = df[["fips"] + pop_cols].melt(
        id_vars="fips", var_name="year_col", value_name="population"
    )
    df_long["year"] = df_long["year_col"].str.replace("POPESTIMATE", "").astype(int)
    df_long = df_long.drop(columns=["year_col"])
    return df_long[["fips", "year", "population"]]


def main() -> None:
    CROSSWALK_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})

    frames = []
    for url in [PEP_2010S_URL, PEP_2020S_URL]:
        try:
            df = fetch_pep(url, session)
            frames.append(df)
            log.info("Loaded %d rows from %s", len(df), url.split("/")[-1])
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)

    if not frames:
        log.error("No population data obtained.")
        sys.exit(1)

    pop = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["fips", "year"])
        .query("year in @STUDY_YEARS")
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )

    out = DATA_PROC / "county_population.parquet"
    pop.to_parquet(out, index=False)
    log.info("Saved %s — %d rows, %d counties", out, len(pop), pop["fips"].nunique())


if __name__ == "__main__":
    main()
