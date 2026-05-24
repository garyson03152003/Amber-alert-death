"""
01b_extract_serious_injuries.py — Extract serious injury counts from FARS person files.

FARS records every person involved in a fatal crash with an injury severity code:
  4 = Fatal Injury (K)
  3 = Suspected Serious Injury (A)  ← what we extract
  2 = Suspected Minor Injury (B)
  1 = Possible Injury (C)
  0 = No Apparent Injury (O)

Note: FARS covers only crashes with at least one fatality. Serious injuries here
are people seriously hurt in the same crash that killed someone else — not a
census of all serious injury crashes. Use alongside fatality counts, not instead.

Output: data/processed/fars_serious_injuries.parquet
  Columns: fips, date, serious_injuries

Run: python code/01b_extract_serious_injuries.py
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import FARS_RAW, DATA_PROC, STUDY_YEARS
from utils import get_logger, fips5

log = get_logger("01b_injuries")

UNKNOWN_COUNTY = {0, 999}
SERIOUS_INJ_CODE = 3


def extract_serious_injuries(zip_path: Path, year: int) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist()
                   if "person" in n.lower() and n.lower().endswith(".csv")
                   and "rf" not in n.lower()]
        if not matches:
            log.warning("No person.csv in %s", zip_path.name)
            return pd.DataFrame(columns=["fips", "date", "serious_injuries"])

        with zf.open(matches[0]) as f:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)

    # Normalise column names
    df.columns = [c.strip().lstrip("\xef\xbb\xbf").strip().upper() for c in df.columns]

    needed = {"STATE", "COUNTY", "MONTH", "DAY", "INJ_SEV"}
    missing = needed - set(df.columns)
    if missing:
        log.warning("Missing columns in %d person file: %s", year, missing)
        return pd.DataFrame(columns=["fips", "date", "serious_injuries"])

    # Filter to serious injuries only
    serious = df[df["INJ_SEV"] == SERIOUS_INJ_CODE].copy()
    serious = serious[~serious["COUNTY"].isin(UNKNOWN_COUNTY)]
    serious = serious[serious["STATE"].between(1, 56)]
    serious = serious[serious["MONTH"].between(1, 12) & serious["DAY"].between(1, 31)]

    serious["fips"] = serious.apply(lambda r: fips5(r["STATE"], r["COUNTY"]), axis=1)
    serious["date"] = pd.to_datetime(
        {"year": year, "month": serious["MONTH"], "day": serious["DAY"]},
        errors="coerce",
    )
    serious = serious.dropna(subset=["date"])

    out = (
        serious.groupby(["fips", "date"])
        .size()
        .reset_index(name="serious_injuries")
    )
    log.info("  %d: %d serious injury person-records → %d county-day rows",
             year, len(serious), len(out))
    return out


def main() -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in tqdm(STUDY_YEARS, desc="Injury years"):
        zip_path = FARS_RAW / f"FARS{year}NationalCSV.zip"
        if not zip_path.exists():
            log.warning("Missing zip for %d — skipping", year)
            continue
        frames.append(extract_serious_injuries(zip_path, year))

    if not frames:
        log.error("No data extracted.")
        return

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["fips", "date"])
        .reset_index(drop=True)
    )
    out_path = DATA_PROC / "fars_serious_injuries.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s — %d county-day rows, %d counties, %d years",
             out_path, len(out), out["fips"].nunique(),
             out["date"].dt.year.nunique())


if __name__ == "__main__":
    main()
