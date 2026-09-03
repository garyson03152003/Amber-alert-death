"""Complete county-level IANA timezone construction for alert timestamps."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
# timezonefinder 8.x enables numba caching at import time.  On some Python
# 3.13 installations the package is present but numba cannot locate its
# installed source file, which makes the import fail before a map is built.
# The map is tiny (county centroids only), so disabling JIT avoids that
# environment-specific failure without changing the timezone lookup result.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    from timezonefinder import TimezoneFinder
except Exception:  # only needed when constructing the production map
    TimezoneFinder = None


# Historical county-equivalent SAME codes that appear in the 2013--2024
# archive but no longer exist in the current centroid file. These are explicit
# code-level mappings, not state-wide timezone fallbacks.
HISTORICAL_COUNTY_TIMEZONES = {
    "02261": "America/Anchorage",  # former Valdez-Cordova Census Area, AK
    "51560": "America/New_York",   # former Clifton Forge independent city, VA
}


@lru_cache(maxsize=4)
def county_timezone_map(centroid_path: str | Path) -> dict[str, str]:
    """Map every county population centroid to a DST-aware IANA timezone.

    A county that crosses a timezone boundary receives the timezone at its
    population-weighted centroid. Missing coordinates or failed timezone
    lookups are errors: callers must not silently substitute a state-wide
    timezone for an unknown county.
    """
    if TimezoneFinder is None:
        raise ImportError("timezonefinder is required to build county timezone maps")
    path = Path(centroid_path)
    centroids = pd.read_parquet(path, columns=["fips", "lat", "lon"])
    centroids["fips"] = centroids["fips"].astype(str).str.zfill(5)
    if centroids["fips"].duplicated().any():
        raise ValueError(f"duplicate county FIPS in timezone centroids: {path}")
    if centroids[["lat", "lon"]].isna().any().any():
        raise ValueError(f"missing county centroid coordinates: {path}")

    finder = TimezoneFinder(in_memory=True)
    mapping = {
        row.fips: finder.timezone_at(lng=float(row.lon), lat=float(row.lat))
        for row in centroids.itertuples(index=False)
    }
    missing = sorted(fips for fips, timezone in mapping.items() if timezone is None)
    if missing:
        raise ValueError(f"IANA timezone lookup failed for county FIPS: {missing[:10]}")
    mapping.update(HISTORICAL_COUNTY_TIMEZONES)
    return mapping
