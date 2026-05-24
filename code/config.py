"""
Central configuration for the AMBER Alert → Traffic Fatalities project.

Edit the constants here rather than hunting through individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUT_FIGS = ROOT / "output" / "figures"
OUTPUT_TABS = ROOT / "output" / "tables"

FARS_RAW = DATA_RAW / "fars"
AMBER_RAW = DATA_RAW / "amber"
WEATHER_RAW = DATA_RAW / "weather"
CROSSWALK_RAW = DATA_RAW / "crosswalks"

# ---------------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------------
# WEA (Wireless Emergency Alerts) became mandatory for carriers ~2012.
# Use 2013–2022 to avoid pre-WEA noise.
STUDY_YEARS = list(range(2013, 2025))   # inclusive on both ends  (2013–2024)

# ---------------------------------------------------------------------------
# Treatment definition
# ---------------------------------------------------------------------------
# AMBER Alerts issued between these hours (local time) count as "nighttime"
NIGHT_START_HOUR = 22   # 10 pm
NIGHT_END_HOUR = 6      # 6 am  (wraps past midnight)

# Sub-bands for heterogeneity analysis
NIGHT_BANDS = {
    "early_night":  (22, 24),   # 10 pm – midnight
    "deep_night":   (0,  3),    # midnight – 3 am  (deepest sleep)
    "late_night":   (3,  6),    # 3 am – 6 am
}

# ---------------------------------------------------------------------------
# NOAA CDO API
# ---------------------------------------------------------------------------
# Get a free token at https://www.ncdc.noaa.gov/cdo-web/token
# Set in a .env file as NOAA_CDO_TOKEN=<your_token>
NOAA_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# ---------------------------------------------------------------------------
# FEMA IPAWS public alert feed
# ---------------------------------------------------------------------------
IPAWS_BASE = "https://apps.fema.gov/IPAWSOPEN_EAS_SERVICE/rest/public"

# ---------------------------------------------------------------------------
# GDELT fallback (used when IPAWS/FOIA data unavailable)
# ---------------------------------------------------------------------------
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# ---------------------------------------------------------------------------
# FARS download URL pattern (NHTSA FTP mirror)
# ---------------------------------------------------------------------------
FARS_URL_TEMPLATE = (
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/"
    "{year}/National/FARS{year}NationalCSV.zip"
)

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
# Cluster SE at county level
CLUSTER_VAR = "fips"

# Days around alert to use in event study
EVENT_WINDOW = (-3, 3)

# Minimum mean annual fatalities to keep county in sample.
# Counties below this are essentially zero-fatality noise; ≥5 halves the
# county count but retains ~95% of treated county-days (alerts cluster in
# populated areas).
MIN_FATALS_PER_YEAR = 5
