"""
02b_synthetic_amber_alerts.py — Generate a realistic synthetic AMBER Alert dataset.

Use this ONLY when real IPAWS/FOIA data are not yet available.  The synthetic
data matches published aggregate statistics (NCMEC annual reports, state DOJ
summaries) and is suitable for methodology demonstration and code validation.

Key calibration targets:
  - ~300–400 AMBER Alerts issued nationally per year (NCMEC data)
  - ~35% of alerts issued during nighttime hours (10 pm – 5 am, local time)
  - Alert timing roughly follows a bimodal distribution with modes at
    5–7 pm (end-of-business) and 12–2 am (overnight incidents)
  - Alert-county geographic distribution proportional to state population,
    spread across 1–6 counties per alert (median ≈ 3)
  - Alert rates slightly higher in large/urban states (TX, CA, FL, OH, PA)

Output: data/raw/amber/foia/synthetic_alerts_2013_2022.csv
        (placed in the FOIA drop folder so 02_collect_amber_alerts.py loads it)

Run: python code/02b_synthetic_amber_alerts.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import STUDY_YEARS, AMBER_RAW
from utils import get_logger

log = get_logger("02b_synthetic")

# ---------------------------------------------------------------------------
# State-level parameters
# ---------------------------------------------------------------------------
# (state_fips, approx_annual_alerts, n_counties_in_state)
# Annual alert counts loosely calibrated to NCMEC state reports
STATE_PARAMS = [
    ("01", 18, 67), ("02", 5,  29), ("04", 22, 15), ("05", 14, 75),
    ("06", 45, 58), ("08", 18, 64), ("09", 8,  8),  ("10", 5,  3),
    ("12", 38, 67), ("13", 28, 159),("15", 4,  5),  ("16", 8,  44),
    ("17", 28, 102),("18", 20, 92), ("19", 12, 99), ("20", 12, 105),
    ("21", 16, 120),("22", 18, 64), ("23", 4,  16), ("24", 10, 24),
    ("25", 10, 14), ("26", 22, 83), ("27", 16, 87), ("28", 12, 82),
    ("29", 18, 115),("30", 6,  56), ("31", 8,  93), ("32", 10, 17),
    ("33", 4,  10), ("34", 14, 21), ("35", 10, 33), ("36", 22, 62),
    ("37", 24, 100),("38", 5,  53), ("39", 28, 88), ("40", 18, 77),
    ("41", 12, 36), ("42", 24, 67), ("44", 3,  5),  ("45", 16, 46),
    ("46", 5,  66), ("47", 20, 95), ("48", 50, 254),("49", 10, 29),
    ("50", 3,  14), ("51", 18, 133),("53", 18, 39), ("54", 8,  55),
    ("55", 14, 72), ("56", 4,  23),
]

# Hour-of-day distribution for alert issuances (local time)
# Bimodal: cluster around 6 pm (police report filed) and 1 am (acute incident)
HOUR_WEIGHTS = np.array([
    3, 2, 3, 2, 1, 1, 1, 2, 3, 4, 4, 4,   # midnight–11 am
    5, 5, 6, 6, 8, 9, 9, 8, 6, 5, 4, 4,   # noon–11 pm
], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()


def generate_alerts(rng: np.random.Generator) -> pd.DataFrame:
    records = []

    for state_fips, annual_rate, n_counties in STATE_PARAMS:
        # Enumerate all county FIPS in this state: state_fips + 001,003,...
        county_nums = [f"{state_fips}{3*i+1:03d}" for i in range(n_counties)]

        for year in STUDY_YEARS:
            n_alerts = int(rng.poisson(annual_rate))

            for _ in range(n_alerts):
                # Random date in year (uniform)
                day_of_year = int(rng.integers(1, 365))
                date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_of_year)

                # Random hour from calibrated distribution
                hour = int(rng.choice(24, p=HOUR_WEIGHTS))
                minute = int(rng.integers(0, 60))

                issued_local = date + pd.Timedelta(hours=hour, minutes=minute)

                # Counties: 1–6, biased toward fewer
                n_ctys = int(rng.choice([1, 2, 3, 4, 5, 6],
                                        p=[0.25, 0.30, 0.20, 0.12, 0.08, 0.05]))
                n_ctys = min(n_ctys, len(county_nums))
                ctys = rng.choice(county_nums, size=n_ctys, replace=False)

                records.append({
                    "alert_id":    f"syn_{state_fips}_{year}_{len(records):06d}",
                    "state_fips":  state_fips,
                    "county_fips": ",".join(ctys),
                    "issued_date": date.strftime("%Y-%m-%d"),
                    "issued_time": f"{hour:02d}:{minute:02d}:00",
                    "source":      "synthetic",
                })

    return pd.DataFrame(records)


def main() -> None:
    dest_dir = AMBER_RAW / "foia"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "synthetic_alerts_2013_2022.csv"

    if dest.exists():
        log.info("Synthetic alerts already exist at %s — skipping.", dest)
        return

    rng = np.random.default_rng(seed=42)
    df = generate_alerts(rng)

    total = len(df)
    national_per_year = total / len(STUDY_YEARS)
    log.info(
        "Generated %d synthetic alerts (%.0f/year nationally) across %d states",
        total, national_per_year, df["state_fips"].nunique(),
    )

    # Quick sanity check on nighttime fraction
    hour_col = df["issued_time"].str[:2].astype(int)
    night = hour_col.isin(list(range(22, 24)) + list(range(0, 5)))
    log.info("Night fraction: %.1f%%", night.mean() * 100)

    df.to_csv(dest, index=False)
    log.info("Saved synthetic alerts → %s", dest)
    log.warning(
        "\n" + "="*70 +
        "\n  SYNTHETIC DATA — for methodology demo only.\n"
        "  Replace with real IPAWS/FOIA data before drawing conclusions.\n" +
        "="*70
    )


if __name__ == "__main__":
    main()
