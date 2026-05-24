"""
01_download_fars.py — Download and clean NHTSA FARS accident data (2013–2022).

Output: data/processed/fars_county_day.parquet
    Columns: fips, date, fatals

Run: python code/01_download_fars.py
"""

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import FARS_RAW, DATA_PROC, FARS_URL_TEMPLATE, STUDY_YEARS
from utils import get_logger, download_file, fips5

log = get_logger("01_fars")


# ---------------------------------------------------------------------------
# Key columns from FARS accident.csv
# ---------------------------------------------------------------------------
# NHTSA uses slightly different column names across years; we normalise below.
STATE_COLS   = ["STATE", "STATENAME"]
COUNTY_COLS  = ["COUNTY", "COUNTYNAME"]
DATE_COLS    = ["YEAR", "MONTH", "DAY", "HOUR", "MINUTE"]
FATALS_COL   = "FATALS"

# Codes that indicate "unknown" in FARS
UNKNOWN_COUNTY = [0, 999]


def download_year(year: int, session: requests.Session) -> Path:
    url = FARS_URL_TEMPLATE.format(year=year)
    dest = FARS_RAW / f"FARS{year}NationalCSV.zip"
    return download_file(url, dest, session=session)


def extract_accident_csv(zip_path: Path) -> pd.DataFrame:
    """Pull accident.csv from the year's zip and return a DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # The file is variously named Accident.csv / accident.CSV / ACCIDENT.CSV
        match = [n for n in names if "accident" in n.lower() and n.lower().endswith(".csv")]
        if not match:
            raise FileNotFoundError(f"No accident CSV found in {zip_path}. Contents: {names}")
        with zf.open(match[0]) as f:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    log.info("  Loaded %d rows from %s", len(df), match[0])
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case all column names for consistent access, stripping any BOM."""
    df.columns = [c.strip().lstrip("﻿\xef\xbb\xbf").strip().upper() for c in df.columns]
    return df


def build_county_day(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Aggregate from accident-level to county-day fatality counts.

    Returns DataFrame with columns: fips, date, fatals
    """
    df = normalise_columns(df)

    # ------------------------------------------------------------------
    # 1. Keep only needed columns, handling minor year-to-year name changes
    # ------------------------------------------------------------------
    needed = {"STATE", "COUNTY", "YEAR", "MONTH", "DAY", FATALS_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in FARS {year}: {missing}")

    df = df[list(needed)].copy()

    # ------------------------------------------------------------------
    # 2. Drop accidents with unknown state or county
    # ------------------------------------------------------------------
    df = df[~df["COUNTY"].isin(UNKNOWN_COUNTY)]
    df = df[df["STATE"].between(1, 56)]   # valid FIPS state range

    # ------------------------------------------------------------------
    # 3. Build 5-digit FIPS, parse date
    # ------------------------------------------------------------------
    df["fips"] = df.apply(lambda r: fips5(r["STATE"], r["COUNTY"]), axis=1)

    # FARS DAY/MONTH/YEAR are integers; rare invalid dates (e.g., month=99) exist
    df = df[df["MONTH"].between(1, 12) & df["DAY"].between(1, 31)]
    df["date"] = pd.to_datetime(
        {"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY"]},
        errors="coerce",
    )
    df = df.dropna(subset=["date"])

    # ------------------------------------------------------------------
    # 4. Aggregate to county-day
    # ------------------------------------------------------------------
    out = (
        df.groupby(["fips", "date"])[FATALS_COL]
        .sum()
        .reset_index()
        .rename(columns={FATALS_COL: "fatals"})
    )
    return out


def main() -> None:
    FARS_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})

    frames = []
    for year in tqdm(STUDY_YEARS, desc="FARS years"):
        log.info("Processing FARS %d", year)
        try:
            zip_path = download_year(year, session)
            df = extract_accident_csv(zip_path)
            county_day = build_county_day(df, year)
            frames.append(county_day)
            log.info("  → %d county-day rows for %d", len(county_day), year)
        except Exception as exc:
            log.error("Failed on year %d: %s", year, exc)
            raise

    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["fips", "date"])
        .reset_index(drop=True)
    )

    out_path = DATA_PROC / "fars_county_day.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(
        "Saved %s — %d county-day observations across %d counties and %d years",
        out_path,
        len(panel),
        panel["fips"].nunique(),
        panel["date"].dt.year.nunique(),
    )


if __name__ == "__main__":
    main()
