"""Pool the state county-hour crash panels into one multi-state panel.

Why pool
--------
Every hour-level result so far has been power-limited. The California
matched-week event study rested on 328 alert events and returned intervals
of roughly +/-8pp; the FARS hourly windows are worse still -- 2 to 70 treated
county-days carry each estimate, because FARS is a census of a rare event
(415,595 fatal crashes spread over ~330 million county-hour cells, 99.9%
true zeros). Pooling the state ALL-crash panels is the only way to get
enough events: California alone holds roughly seven times more crashes than
all of national FARS.

Reporting-threshold differences
-------------------------------
States do not share a reportable-crash definition -- property-damage
minimums, injury coding and private-property inclusion all differ, and even
on fatalities (the most standardised outcome) the state-vs-FARS ratio ranges
from 0.96 in Florida to 1.08 in North Carolina.

That does not preclude pooling here, because the estimand is proportional.
Every specification carries county-level fixed effects and is fitted with
PPML, so a state's threshold scales its own baseline and is absorbed: each
county is compared against itself and the coefficient is a percentage change
in that county's own crash rate. What pooling must NOT do is compare raw
counts across states, so no specification here uses cross-state levels.

The pooled outcome is deliberately named ``crashes`` rather than any state's
native column, to make it explicit that this is a harmonised construct and
not a single agency's measure.

Output: data/processed/pooled_county_hour.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

log = get_logger("pooled_hourly")

OUT_PATH = DATA_PROC / "pooled_county_hour.parquet"

# panel file -> (source key, its native crash column)
SOURCES = {
    "california_ccrs_county_hour.parquet": ("CA", "ca_crashes"),
    "de_county_hour.parquet":              ("DE", "de_crashes"),
    "ut_county_hour.parquet":              ("UT", "ut_crashes"),
    "ma_county_hour.parquet":              ("MA", "ma_crashes"),
    "ct_county_hour.parquet":              ("CT", "ct_crashes"),
    "ia_county_hour.parquet":              ("IA", "ia_crashes"),
}


def load_pooled() -> pd.DataFrame:
    parts = []
    for fname, (key, col) in SOURCES.items():
        path = DATA_PROC / fname
        if not path.is_file():
            log.warning("[%s] not built yet, skipping (%s)", key, fname)
            continue
        d = pd.read_parquet(path)
        if col not in d.columns:
            col = next((c for c in d.columns
                        if pd.api.types.is_numeric_dtype(d[c]) and c != "hour"), None)
            if col is None:
                log.warning("[%s] no crash column found, skipping", key)
                continue
        out = d[["fips", "date", "hour", col]].rename(columns={col: "crashes"}).copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["source"] = key
        parts.append(out)
        log.info("[%s] %s county-hours, %s crashes, %s-%s",
                 key, f"{len(out):,}", f"{int(out['crashes'].sum()):,}",
                 out["date"].dt.year.min(), out["date"].dt.year.max())

    if not parts:
        raise FileNotFoundError("no state county-hour panels available to pool")
    pooled = pd.concat(parts, ignore_index=True)
    # A county should come from exactly one source; guard against a county
    # being double-counted if two panels ever overlap geographically.
    dupes = pooled.duplicated(subset=["fips", "date", "hour"], keep=False)
    if dupes.any():
        log.warning("%s duplicated county-hours across sources -- summing",
                    f"{int(dupes.sum()):,}")
        pooled = pooled.groupby(["fips", "date", "hour"], as_index=False).agg(
            crashes=("crashes", "sum"), source=("source", "first"))
    return pooled


def main() -> None:
    pooled = load_pooled()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(OUT_PATH, index=False)
    log.info("POOLED: %s county-hours | %s crashes | %s counties | %s sources",
             f"{len(pooled):,}", f"{int(pooled['crashes'].sum()):,}",
             pooled["fips"].nunique(), pooled["source"].nunique())
    log.info("Wrote -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
