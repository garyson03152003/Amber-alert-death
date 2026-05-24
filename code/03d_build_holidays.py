"""
03d_build_holidays.py — Build a county-day holiday indicator.

Sources:
  1. Federal holidays  — US Office of Personnel Management (OPM) official list.
     Covers New Year's, MLK Day, Presidents' Day, Memorial Day, Juneteenth,
     Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving,
     Christmas.  Exact observed dates (with Monday substitution) are computed
     via the `holidays` Python library.
  2. State public holidays — same library's US state-specific holidays.
     Includes events like Texas Emancipation Day, California César Chávez Day,
     Mardi Gras in Louisiana, etc.

The output is a county-day indicator `is_holiday` (1 = federal or state holiday)
plus a string label `holiday_name` for the primary holiday on that date.

Output: data/processed/holidays_county_day.parquet
Run:    python code/03d_build_holidays.py

Requires: pip install holidays
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, STUDY_YEARS
from utils import get_logger

log = get_logger("03d_build_holidays")

try:
    import holidays as hol_lib
except ImportError:
    raise SystemExit("Install the holidays library:  pip install holidays")


# State FIPS → 2-letter abbreviation map
FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


def build_federal_holidays(years: range) -> dict:
    """Return {date: name} for all US federal observed holidays."""
    us_hols = hol_lib.country_holidays("US", years=list(years))
    return dict(us_hols)


def build_state_holidays(state_abbr: str, years: range) -> dict:
    """Return {date: name} for state-specific public holidays."""
    try:
        s_hols = hol_lib.country_holidays("US", subdiv=state_abbr,
                                           years=list(years))
        return dict(s_hols)
    except Exception:
        return {}


def main() -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # Load county list from panel
    panel_path = DATA_PROC / "panel_county_day.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Run 04_build_panel.py first.")

    panel = pd.read_parquet(panel_path, columns=["fips", "date"])
    panel["date"] = pd.to_datetime(panel["date"])
    counties = panel[["fips"]].drop_duplicates().copy()
    counties["state_fips"] = counties["fips"].str[:2]
    log.info("Building holidays for %d counties, years %d–%d",
             len(counties), min(STUDY_YEARS), max(STUDY_YEARS))

    years = range(min(STUDY_YEARS), max(STUDY_YEARS) + 1)

    # Federal holidays (apply to all counties)
    fed = build_federal_holidays(years)
    fed_df = pd.DataFrame([
        {"date": pd.Timestamp(d), "holiday_name": n, "is_federal": True}
        for d, n in fed.items()
    ])
    log.info("Federal holidays: %d date-entries across study period", len(fed_df))

    # State holidays — keyed by state abbr
    state_hol_cache: dict[str, dict] = {}
    for sf, abbr in FIPS_TO_ABBR.items():
        sh = build_state_holidays(abbr, years)
        if sh:
            state_hol_cache[sf] = sh

    log.info("State holiday sets loaded for %d states", len(state_hol_cache))

    # Build per-county holiday rows
    rows = []
    for _, row in counties.iterrows():
        sf = row["state_fips"]
        abbr = FIPS_TO_ABBR.get(sf, "")

        # Federal
        for d, name in fed.items():
            rows.append({
                "fips": row["fips"],
                "date": pd.Timestamp(d),
                "holiday_name": name,
                "is_federal": True,
                "is_state_only": False,
            })

        # State-specific (not already in federal)
        if sf in state_hol_cache:
            for d, name in state_hol_cache[sf].items():
                if d not in fed:
                    rows.append({
                        "fips": row["fips"],
                        "date": pd.Timestamp(d),
                        "holiday_name": name,
                        "is_federal": False,
                        "is_state_only": True,
                    })

    hdf = pd.DataFrame(rows)
    hdf = hdf.drop_duplicates(subset=["fips", "date"]).copy()
    hdf["is_holiday"] = 1

    # Filter to study period dates only
    hdf = hdf[hdf["date"].dt.year.isin(STUDY_YEARS)]

    out = DATA_PROC / "holidays_county_day.parquet"
    hdf.to_parquet(out, index=False)
    log.info("Saved %s — %d county-holiday rows (%d unique dates)",
             out, len(hdf), hdf["date"].nunique())

    # Summary
    fed_dates  = hdf[hdf["is_federal"]]["date"].nunique()
    state_dates = hdf[hdf["is_state_only"]]["date"].nunique()
    log.info("  Federal holiday-dates: %d", fed_dates)
    log.info("  State-only holiday-dates per county: ~%d", state_dates // max(len(counties), 1))

    top = (hdf[hdf["is_federal"]]
           .drop_duplicates("date")
           .sort_values("date")["holiday_name"]
           .value_counts().head(15))
    log.info("Top federal holiday names:\n%s", top.to_string())


if __name__ == "__main__":
    main()
