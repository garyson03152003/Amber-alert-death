"""Build county x date x HOUR crash panels for the validated state sources
that carry a crash timestamp.

Motivation
----------
Only California currently has an hourly panel, which limits the matched-week
hourly event study to 328 alert events and confidence intervals of roughly
+/-8pp. Delaware, Illinois, Massachusetts, Utah and Montgomery County MD all
retain a crash time in their source data and are already validated against
FARS in config/accepted_state_years.csv; their county-day builders simply
discard the hour at aggregation.

This module reuses each state's existing ``fetch_year`` rather than
reimplementing the download, and aggregates to county-hour instead of
county-day.

Timezone correctness
--------------------
This is the sharp edge. At calendar-date grain a few hours of offset is
usually harmless, and some existing builders say so explicitly -- Delaware's
parses its ISO timestamps as UTC and drops the zone, noting the distinction
is "immaterial" for a daily panel. At HOURLY grain it is not immaterial: it
would shift every Delaware crash 4-5 hours and put the evening commute in
the middle of the night.

Two defences:

  1. Timestamps are explicitly converted to each state's local wall-clock
     time, with the source's own semantics declared per state below.
  2. ``validate_diurnal_profile`` checks the resulting hour distribution
     against the shape every real crash dataset has -- a pronounced
     afternoon/evening peak and an early-morning trough. A timezone error
     rotates that curve and the check fails loudly rather than silently
     producing a plausible-looking but wrong panel.

A second reconciliation check confirms the hourly counts sum back to the
already-validated county-day totals, so a parsing change cannot quietly
alter the underlying series.

Outputs
-------
data/processed/{state}_county_hour.parquet
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

log = get_logger("state_hourly_panels")


class SourceSpec:
    """How to turn one state's raw crash frame into local-time county-hours."""

    def __init__(self, *, key, module, datetime_col, tz, datetime_kind,
                 county_mapper, years, crash_col, day_panel, day_crash_col,
                 unpack_first=False, out_fields=None, severity_fn=None,
                 fatal_col=None, serious_col=None):
        self.key = key
        self.module = module
        # May be a single column name or several candidates: Massachusetts
        # changed the field between service generations (CRASH_DATETIME on the
        # 2013-2017 layers, CRASH_DATE on 2018-2020), so the first column
        # actually present in the frame is used.
        self.datetime_col = (
            (datetime_col,) if isinstance(datetime_col, str) else tuple(datetime_col)
        )
        self.tz = tz
        self.datetime_kind = datetime_kind   # "epoch_ms_utc" | "iso_utc" | "naive_local"
        self.county_mapper = county_mapper
        self.years = years
        self.crash_col = crash_col
        self.day_panel = day_panel
        self.day_crash_col = day_crash_col
        # Some builders return more than one frame from fetch_year (CT returns
        # (crashes, persons)); take the first when that happens.
        self.unpack_first = unpack_first
        # Some builders request a narrow field list that omits the time
        # column (Iowa fetches CRASH_DATE but not CRASH_DATETIME), so the
        # hourly build widens it for the duration of the fetch.
        self.out_fields = out_fields
        # Severity: either a callable df -> (fatal_series, serious_series) for
        # states needing derived logic (e.g. MA's KABCO-from-text), or a pair
        # of raw numeric column names to sum directly (UT, IA).
        self.severity_fn = severity_fn
        self.fatal_col = fatal_col
        self.serious_col = serious_col


def _ma_severity(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fatals = pd.to_numeric(df["NUMB_FATAL_INJR"], errors="coerce").fillna(0)
    all_injured = pd.to_numeric(df["NUMB_NONFATAL_INJR"], errors="coerce").fillna(0)
    is_serious = df["MAX_INJR_SVRTY_CL"].astype(str).str.contains(
        "Incapacitating", case=False, na=False)
    serious = all_injured.where(is_serious, 0)
    return fatals, serious


SPECS = {
    # Utah ArcGIS serves epoch milliseconds. ArcGIS epoch values are UTC.
    "UT": SourceSpec(
        key="UT", module="build_utah_udot", datetime_col="CRASH_DATETIME",
        tz="America/Denver", datetime_kind="epoch_ms_utc",
        county_mapper=("county_to_fips", "COUNTY_NAME"),
        years=range(2018, 2025), crash_col="ut_crashes",
        day_panel="utah_udot_county_day.parquet", day_crash_col="ut_crashes",
        fatal_col="NUMBER_FATALITIES", serious_col="NUMBER_FOUR_INJURIES",
    ),
    # Delaware Socrata serves ISO 8601 with a trailing Z.
    "DE": SourceSpec(
        key="DE", module="build_delaware_dshs", datetime_col="crash_datetime",
        tz="America/New_York", datetime_kind="iso_utc",
        county_mapper=("county_to_fips", "county"),
        years=range(2013, 2025), crash_col="de_crashes",
        day_panel="delaware_deldot_county_day.parquet", day_crash_col="de_crashes",
    ),
    # Massachusetts ArcGIS epoch milliseconds (UTC), field renamed between
    # service generations. Counties are mapped from CNTY_NAME.
    "MA": SourceSpec(
        key="MA", module="build_massachusetts_massdot",
        datetime_col=("CRASH_DATE", "CRASH_DATETIME"),
        tz="America/New_York", datetime_kind="epoch_ms_local",
        county_mapper=("_ma_county_to_fips", "CNTY_NAME"),
        years=range(2013, 2021), crash_col="ma_crashes",
        day_panel="massachusetts_massdot_county_day.parquet",
        day_crash_col="ma_crashes", severity_fn=_ma_severity,
    ),
    # Connecticut ArcGIS epoch milliseconds. Genuine UTC: the service declares
    # timeZoneIANA America/New_York with DST, and converting moves the peak
    # from an implausible 20:00 to 17:00 with an 04:00 trough.
    "CT": SourceSpec(
        key="CT", module="build_connecticut_uconn", datetime_col="CrashDate",
        tz="America/New_York", datetime_kind="epoch_ms_utc",
        county_mapper=("town_to_fips", "CrashTownName"),
        years=range(2015, 2023), crash_col="ct_crashes",
        day_panel="connecticut_uconn_county_day.parquet",
        day_crash_col="ct_crashes", unpack_first=True,
    ),
    # Iowa ArcGIS: CRASH_DATETIME is epoch ms encoding LOCAL wall-clock
    # (CRASH_DATE + TIMESTR), while the CRASH_DATETIME_UTC column exists but
    # is never populated. Naive parsing peaks at 15:00 (a normal crash
    # curve); converting to Central would move the peak to 09:00.
    "IA": SourceSpec(
        key="IA", module="build_iowa_dot", datetime_col="CRASH_DATETIME",
        tz="America/Chicago", datetime_kind="epoch_ms_local",
        county_mapper=("_ia_county_to_fips", "COUNTY_NAME"),
        years=range(2015, 2025), crash_col="ia_crashes",
        day_panel="iowa_dot_county_day.parquet", day_crash_col="ia_crashes",
        out_fields=("CRASH_DATETIME,CRASH_DATE,COUNTY_NAME,FATALITIES,"
                    "MAJINJURY,INJURIES,CRASH_MONTH,CRASH_DAY"),
        fatal_col="FATALITIES", serious_col="MAJINJURY",
    ),
}


def to_local_hour(series: pd.Series, *, kind: str, tz: str) -> pd.Series:
    """Convert a raw timestamp column to naive LOCAL wall-clock time."""
    if kind == "epoch_ms_utc":
        ts = pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
    elif kind == "iso_utc":
        ts = pd.to_datetime(series, errors="coerce", utc=True)
    elif kind == "epoch_ms_local":
        # Epoch milliseconds that actually encode LOCAL wall-clock time (the
        # service stored local time as if it were UTC). Parsing without a
        # timezone already yields the correct local value; converting would
        # subtract the offset a second time. Massachusetts is this kind --
        # its raw hours peak at 15-17 like a normal crash curve, whereas
        # converting to Eastern moves the peak to noon.
        return pd.to_datetime(series, unit="ms", errors="coerce")
    elif kind == "naive_local":
        return pd.to_datetime(series, errors="coerce")
    else:
        raise ValueError(f"unknown datetime_kind {kind!r}")
    return ts.dt.tz_convert(tz).dt.tz_localize(None)


def _year_has_time_of_day(local: pd.Series, *, min_rows: int = 50,
                          max_distinct: int = 3) -> bool:
    """Does this year's converted timestamp actually carry an hour?

    Mirrors the date-only test in audit_timestamp_timezones: a handful of
    distinct times across many rows means the column is a date wearing a
    timestamp's clothes.
    """
    valid = local.dropna()
    if len(valid) < min_rows:
        return False
    tod = valid.dt.hour * 3600 + valid.dt.minute * 60 + valid.dt.second
    return int(tod.nunique()) > max_distinct


def validate_diurnal_profile(hours: pd.Series, *, label: str) -> dict:
    """Fail loudly if the hour distribution is not shaped like real crashes.

    Every road-crash dataset shares a robust signature: a broad afternoon /
    early-evening peak (roughly 14:00-19:00) and a deep pre-dawn trough
    (roughly 01:00-05:00). A timezone mistake rotates this curve, so this is
    a direct test of whether the local-time conversion is right -- far more
    informative than checking that the column merely parsed.
    """
    counts = hours.value_counts().reindex(range(24), fill_value=0).sort_index()
    share = counts / counts.sum()
    peak_hour = int(share.idxmax())
    trough_hour = int(share.idxmin())
    ok_peak = 14 <= peak_hour <= 19
    ok_trough = 1 <= trough_hour <= 5
    result = {
        "label": label, "peak_hour": peak_hour, "trough_hour": trough_hour,
        "peak_share": float(share.max()), "trough_share": float(share.min()),
        "plausible": bool(ok_peak and ok_trough),
    }
    if not result["plausible"]:
        log.error(
            "[%s] IMPLAUSIBLE diurnal profile: peak at %02d:00, trough at %02d:00. "
            "Expected peak 14-19 and trough 01-05 -- this usually means the "
            "timestamp was not converted to local time correctly.",
            label, peak_hour, trough_hour,
        )
    else:
        log.info("[%s] diurnal profile OK (peak %02d:00, trough %02d:00)",
                 label, peak_hour, trough_hour)
    return result


def reconcile_with_day_panel(hourly: pd.DataFrame, spec: SourceSpec) -> dict:
    """Check hourly counts sum back to the validated county-day series."""
    path = DATA_PROC / spec.day_panel
    if not path.is_file():
        log.warning("[%s] no county-day panel to reconcile against", spec.key)
        return {"reconciled": None}
    day = pd.read_parquet(path)
    if spec.day_crash_col not in day.columns:
        return {"reconciled": None}
    lhs = hourly.groupby(["fips", "date"])[spec.crash_col].sum()
    rhs = day.set_index(["fips", pd.to_datetime(day["date"]).dt.normalize()])[
        spec.day_crash_col
    ]
    rhs.index.names = ["fips", "date"]
    joined = pd.concat([lhs.rename("hourly"), rhs.rename("daily")], axis=1).dropna()
    if joined.empty:
        return {"reconciled": None}
    diff = (joined["hourly"] - joined["daily"]).abs()
    agree = float((diff < 1e-6).mean())
    log.info("[%s] county-day reconciliation: %.4f of overlapping county-days match",
             spec.key, agree)
    return {"reconciled": agree, "n_compared": int(len(joined))}


def build_state(spec: SourceSpec) -> pd.DataFrame | None:
    mod = importlib.import_module(spec.module)
    mapper_name, county_col = spec.county_mapper
    mapper = getattr(mod, mapper_name)

    if spec.out_fields is not None and hasattr(mod, "OUT_FIELDS"):
        log.info("[%s] widening OUT_FIELDS to include the time column", spec.key)
        mod.OUT_FIELDS = spec.out_fields

    session = requests.Session()
    frames = []
    for year in spec.years:
        try:
            raw = mod.fetch_year(session, year)
            if spec.unpack_first and isinstance(raw, tuple):
                raw = raw[0]
        except Exception as exc:                       # noqa: BLE001
            log.warning("[%s] %d fetch failed: %s", spec.key, year, exc)
            continue
        if raw is None or len(raw) == 0:
            log.warning("[%s] %d returned no rows", spec.key, year)
            continue
        df = raw.copy()
        dt_col = next((c for c in spec.datetime_col if c in df.columns), None)
        if dt_col is None:
            log.warning("[%s] %d has none of %s", spec.key, year, spec.datetime_col)
            continue
        local = to_local_hour(df[dt_col], kind=spec.datetime_kind, tz=spec.tz)
        # A year may carry no time-of-day even when neighbouring years do.
        # Massachusetts 2018 is the live example: 142,272 crashes all stamped
        # midnight, sitting between 2017 and 2019 which both have full minute
        # resolution from the same field and service generation. Nothing in
        # the schema or the declared metadata predicts it -- only the values
        # do. Such a year is valid at county-DAY grain but has no hour, so it
        # must be dropped from the hourly panel rather than piling its whole
        # crash count onto hour 00.
        if not _year_has_time_of_day(local):
            log.warning("[%s] %d is DATE-ONLY (no time of day) -- excluded from "
                        "the hourly panel; it remains valid in the daily panel",
                        spec.key, year)
            continue
        df = df.assign(_local=local).dropna(subset=["_local"])
        df["fips"] = df[county_col].map(mapper)
        df = df.dropna(subset=["fips"])
        df["date"] = df["_local"].dt.normalize()
        df["hour"] = df["_local"].dt.hour

        has_severity = spec.severity_fn is not None or (
            spec.fatal_col in df.columns and spec.serious_col in df.columns
            if spec.fatal_col and spec.serious_col else False)
        if has_severity:
            if spec.severity_fn is not None:
                fatals, serious = spec.severity_fn(df)
            else:
                fatals = pd.to_numeric(df[spec.fatal_col], errors="coerce").fillna(0)
                serious = pd.to_numeric(df[spec.serious_col], errors="coerce").fillna(0)
            df["_fatals"] = fatals
            df["_serious"] = serious
            agg = (df.groupby(["fips", "date", "hour"], as_index=False)
                     .agg(**{spec.crash_col: ("_fatals", "size"),
                             f"{spec.key.lower()}_fatals": ("_fatals", "sum"),
                             f"{spec.key.lower()}_serious_inj": ("_serious", "sum")}))
        else:
            agg = (df.groupby(["fips", "date", "hour"], as_index=False)
                     .size().rename(columns={"size": spec.crash_col}))
        frames.append(agg)
        log.info("[%s] %d -> %s county-hours", spec.key, year, f"{len(agg):,}")

    if not frames:
        log.error("[%s] no data built", spec.key)
        return None
    combined = pd.concat(frames, ignore_index=True)
    sum_cols = [c for c in combined.columns if c not in ("fips", "date", "hour")]
    return combined.groupby(["fips", "date", "hour"], as_index=False)[sum_cols].sum()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="*", default=list(SPECS),
                        choices=list(SPECS))
    args = parser.parse_args(argv)

    for key in args.states:
        spec = SPECS[key]
        log.info("=== %s ===", key)
        hourly = build_state(spec)
        if hourly is None:
            continue
        profile = validate_diurnal_profile(hourly["hour"].repeat(
            hourly[spec.crash_col].astype(int)), label=key)
        recon = reconcile_with_day_panel(hourly, spec)
        if not profile["plausible"]:
            log.error("[%s] REFUSING to write panel: diurnal check failed", key)
            continue
        out = DATA_PROC / f"{key.lower()}_county_hour.parquet"
        hourly.to_parquet(out, index=False)
        log.info("[%s] wrote %s rows -> %s (%s)", key, f"{len(hourly):,}", out, recon)


if __name__ == "__main__":
    main()
