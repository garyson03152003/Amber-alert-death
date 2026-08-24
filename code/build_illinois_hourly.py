"""
build_illinois_hourly.py
========================================================
Build an Illinois county-hour crash panel from the IDOT ArcGIS crash feeds.

Unlike every other hourly source built this session, Illinois needs NO
timezone decoding at all: the raw feed carries a native ``CrashHour`` field
(00-23, zero-padded string) alongside ``CrashDate``, confirmed present and
consistently named across every year 2016-2024 via a live schema probe. This
makes Illinois the cleanest hourly source in the pool -- a direct read, not
an inference from an epoch/UTC field.

Reuses the fetch machinery (download-API + FeatureServer fallback, county
code -> FIPS mapping) from build_illinois_idot.py rather than re-implementing
it, since Illinois's split item-id/FeatureServer-url wiring per year is
already solved there.

Output: data/processed/il_county_hour.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_illinois_idot import (
    IDOT_ITEMS, idot_county_to_fips, fetch_via_download_api, fetch_via_featureserver,
)
from config import DATA_PROC
from utils import get_logger

log = get_logger("illinois_hourly")

OUT_PATH = DATA_PROC / "il_county_hour.parquet"
DAY_PANEL_PATH = DATA_PROC / "illinois_idot_county_day.parquet"
YEARS = list(range(2016, 2025))


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]

    fatal_col = next((c for c in df.columns
                      if c in ("TotalFatals", "Total Fatals")), None)
    serious_col = next((c for c in df.columns
                        if c in ("AInjuries", "Incapacitating Injuries")), None)

    date_col = next((c for c in df.columns
                     if c in ("CrashDate", "Crash Date", "CRASH_DATE")), None)
    yr_col = next((c for c in df.columns if c in ("CrashYr", "Crash Year")), None)
    mo_col = next((c for c in df.columns if c in ("CrashMonth", "Crash Month")), None)
    day_col = next((c for c in df.columns if c in ("CrashDay", "Crash Day")), None)
    hour_col = next((c for c in df.columns
                     if c in ("CrashHour", "Crash_Hour", "Crash Hour")), None)
    county_col = next((c for c in df.columns
                       if c in ("CountyCode", "County Code", "COUNTY", "County")), None)
    if hour_col is None or county_col is None:
        log.warning("  %d: missing hour/county column; cols=%s",
                    year, list(df.columns)[:30])
        return None

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    elif yr_col and mo_col and day_col:
        yr_vals = pd.to_numeric(df[yr_col], errors="coerce")
        yr_str = yr_vals.apply(lambda y: f"20{int(y):02d}" if pd.notna(y) and y < 100
                               else str(int(y)) if pd.notna(y) else "")
        df["date"] = pd.to_datetime(
            yr_str + "-" + df[mo_col].astype(str).str.zfill(2) + "-"
            + df[day_col].astype(str).str.zfill(2), errors="coerce")
    else:
        log.warning("  %d: missing date info; cols=%s", year, list(df.columns)[:30])
        return None
    df["hour"] = pd.to_numeric(df[hour_col], errors="coerce")
    df = df.dropna(subset=["date", "hour"])
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df["hour"] = df["hour"].astype(int)

    df["_county_num"] = pd.to_numeric(df[county_col], errors="coerce")
    df = df.dropna(subset=["_county_num"])
    df["fips"] = df["_county_num"].apply(idot_county_to_fips)
    df = df.dropna(subset=["fips"])

    if fatal_col and serious_col:
        df["_fatals"] = pd.to_numeric(df[fatal_col], errors="coerce").fillna(0)
        df["_serious"] = pd.to_numeric(df[serious_col], errors="coerce").fillna(0)
        agg = (df.groupby(["fips", "date", "hour"], as_index=False)
                 .agg(il_crashes=("_fatals", "size"),
                      il_fatals=("_fatals", "sum"),
                      il_serious_inj=("_serious", "sum")))
    else:
        agg = (df.groupby(["fips", "date", "hour"])
                  .size().rename("il_crashes").reset_index())
    log.info("  %d: %s county-hours, %s crashes", year, f"{len(agg):,}",
             f"{int(agg['il_crashes'].sum()):,}")
    return agg


def reconcile(hourly: pd.DataFrame) -> None:
    if not DAY_PANEL_PATH.is_file():
        log.warning("no county-day panel to reconcile against")
        return
    day = pd.read_parquet(DAY_PANEL_PATH)
    lhs = hourly.groupby(["fips", "date"])["il_crashes"].sum()
    rhs = day.set_index(["fips", pd.to_datetime(day["date"]).dt.normalize()])["il_crashes"]
    rhs.index.names = ["fips", "date"]
    joined = pd.concat([lhs.rename("hourly"), rhs.rename("daily")], axis=1).dropna()
    if joined.empty:
        log.warning("no overlapping county-days to reconcile")
        return
    diff = (joined["hourly"] - joined["daily"]).abs()
    agree = float((diff < 1e-6).mean())
    log.info("county-day reconciliation: %.4f of %s overlapping county-days match",
              agree, f"{len(joined):,}")


def main() -> None:
    parts = []
    for yr in YEARS:
        log.info("Year %d …", yr)
        info = IDOT_ITEMS.get(yr, {})
        item_id = info.get("item_id")
        fs_url = info.get("fs_url")

        raw = fetch_via_download_api(item_id, yr) if item_id else None
        if (raw is None or raw.empty) and fs_url:
            log.info("  Falling back to FeatureServer pagination …")
            raw = fetch_via_featureserver(fs_url, yr)

        agg = process_year(raw, yr)
        if agg is not None:
            parts.append(agg)
        else:
            log.warning("  Year %d: no data obtained", yr)
        time.sleep(2.0)

    if not parts:
        log.error("No Illinois hourly data obtained.")
        sys.exit(1)

    panel = pd.concat(parts, ignore_index=True)
    sum_cols = [c for c in panel.columns if c not in ("fips", "date", "hour")]
    panel = panel.groupby(["fips", "date", "hour"], as_index=False)[sum_cols].sum()
    log.info("[IL] wrote %s rows -> %s", f"{len(panel):,}", OUT_PATH)

    hour_weights = panel.groupby("hour")["il_crashes"].sum()
    counts = hour_weights.reindex(range(24), fill_value=0).sort_index()
    share = counts / counts.sum()
    peak_hour, trough_hour = int(share.idxmax()), int(share.idxmin())
    ok = (14 <= peak_hour <= 19) and (1 <= trough_hour <= 5)
    log.info("[IL] diurnal profile %s (peak %02d:00, trough %02d:00)",
              "OK" if ok else "IMPLAUSIBLE", peak_hour, trough_hour)
    reconcile(panel)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
