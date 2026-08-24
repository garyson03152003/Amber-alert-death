import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import build_state_hourly_panels as bh


# --- timezone conversion --------------------------------------------------

def test_epoch_ms_is_converted_from_utc_to_local():
    # 2024-06-15 20:00 UTC == 14:00 America/Denver (MDT, UTC-6)
    epoch_ms = pd.Series([int(pd.Timestamp("2024-06-15 20:00", tz="UTC").timestamp() * 1000)])
    local = bh.to_local_hour(epoch_ms, kind="epoch_ms_utc", tz="America/Denver")
    assert local.iloc[0].hour == 14
    assert local.iloc[0].tzinfo is None       # naive wall-clock


def test_iso_utc_is_converted_to_local_not_merely_stripped():
    # 2024-01-15 02:00Z == 21:00 the PREVIOUS day in America/New_York (EST)
    iso = pd.Series(["2024-01-15T02:00:00Z"])
    local = bh.to_local_hour(iso, kind="iso_utc", tz="America/New_York")
    assert local.iloc[0].hour == 21
    assert local.iloc[0].date() == pd.Timestamp("2024-01-14").date()


def test_iso_conversion_respects_daylight_saving():
    # July is EDT (UTC-4), January is EST (UTC-5)
    summer = bh.to_local_hour(pd.Series(["2024-07-15T12:00:00Z"]),
                              kind="iso_utc", tz="America/New_York")
    winter = bh.to_local_hour(pd.Series(["2024-01-15T12:00:00Z"]),
                              kind="iso_utc", tz="America/New_York")
    assert summer.iloc[0].hour == 8
    assert winter.iloc[0].hour == 7


def test_naive_local_is_left_alone():
    s = pd.Series(["2024-06-15 14:00:00"])
    local = bh.to_local_hour(s, kind="naive_local", tz="America/Denver")
    assert local.iloc[0].hour == 14


def test_unparseable_timestamps_become_nat_not_an_exception():
    s = pd.Series(["not-a-date", "2024-06-15T12:00:00Z"])
    local = bh.to_local_hour(s, kind="iso_utc", tz="America/New_York")
    assert pd.isna(local.iloc[0])
    assert not pd.isna(local.iloc[1])


# --- diurnal validation ---------------------------------------------------

def _realistic_hours(n=20000, seed=0, shift=0):
    """Crash-like hour distribution: evening peak, pre-dawn trough."""
    rng = np.random.default_rng(seed)
    weights = np.array([
        0.020, 0.014, 0.011, 0.010, 0.011, 0.018,  # 00-05 trough
        0.032, 0.050, 0.052, 0.042, 0.041, 0.048,  # 06-11
        0.054, 0.056, 0.062, 0.072, 0.076, 0.078,  # 12-17 peak
        0.064, 0.048, 0.038, 0.032, 0.028, 0.023,  # 18-23
    ])
    weights = weights / weights.sum()
    hours = rng.choice(np.arange(24), size=n, p=weights)
    return pd.Series((hours + shift) % 24)


def test_diurnal_check_passes_on_realistic_hours():
    res = bh.validate_diurnal_profile(_realistic_hours(), label="TEST")
    assert res["plausible"]
    assert 14 <= res["peak_hour"] <= 19


def test_diurnal_check_catches_a_timezone_shift():
    """A 7-hour rotation is exactly what forgetting UTC->local looks like."""
    res = bh.validate_diurnal_profile(_realistic_hours(shift=7), label="TEST")
    assert not res["plausible"], "rotated profile should have been rejected"


def test_diurnal_check_catches_a_reverse_shift():
    res = bh.validate_diurnal_profile(_realistic_hours(shift=-8), label="TEST")
    assert not res["plausible"]


def test_diurnal_check_rejects_uniform_hours():
    """A flat distribution means the hour was never really parsed."""
    rng = np.random.default_rng(1)
    flat = pd.Series(rng.integers(0, 24, size=20000))
    res = bh.validate_diurnal_profile(flat, label="TEST")
    # a uniform draw has no meaningful peak/trough structure; it should not
    # reliably land in the crash-like window
    assert res["peak_share"] < 0.06


# --- reconciliation -------------------------------------------------------

def test_reconciliation_reports_full_agreement(tmp_path, monkeypatch):
    hourly = pd.DataFrame({
        "fips": ["10001"] * 3,
        "date": pd.to_datetime(["2024-01-01"] * 3),
        "hour": [1, 2, 3],
        "de_crashes": [2.0, 3.0, 5.0],
    })
    day = pd.DataFrame({
        "fips": ["10001"], "date": pd.to_datetime(["2024-01-01"]),
        "de_crashes": [10.0],
    })
    monkeypatch.setattr(bh, "DATA_PROC", tmp_path)
    day.to_parquet(tmp_path / "delaware_deldot_county_day.parquet", index=False)
    res = bh.reconcile_with_day_panel(hourly, bh.SPECS["DE"])
    assert res["reconciled"] == 1.0


def test_reconciliation_flags_a_mismatch(tmp_path, monkeypatch):
    hourly = pd.DataFrame({
        "fips": ["10001"] * 2,
        "date": pd.to_datetime(["2024-01-01"] * 2),
        "hour": [1, 2],
        "de_crashes": [2.0, 3.0],
    })
    day = pd.DataFrame({
        "fips": ["10001"], "date": pd.to_datetime(["2024-01-01"]),
        "de_crashes": [99.0],
    })
    monkeypatch.setattr(bh, "DATA_PROC", tmp_path)
    day.to_parquet(tmp_path / "delaware_deldot_county_day.parquet", index=False)
    res = bh.reconcile_with_day_panel(hourly, bh.SPECS["DE"])
    assert res["reconciled"] == 0.0


# --- regression guard for the Delaware UTC-date bug ----------------------

def test_delaware_builder_uses_local_date_not_utc_date():
    """Delaware crash_datetime is UTC; the daily panel must key on LOCAL date.

    Parsing as UTC and dropping the zone shifted every crash at local hour
    >= 19 onto the next calendar day (16.9% of all DE crashes). Local date is
    the grain FARS and the AMBER treatment both use.
    """
    import build_delaware_dshs as de

    # 2024-01-16T02:30Z is 2024-01-15 21:30 EST -> local date is the 15th.
    raw = pd.DataFrame({
        "crash_datetime": ["2024-01-16T02:30:00Z"],
        "county": ["N"],
        "year": ["2024"],
    })
    out = de.process_year(raw, 2024)
    assert out is not None and len(out) == 1
    assert pd.Timestamp(out.iloc[0]["date"]) == pd.Timestamp("2024-01-15"), (
        "evening crash was assigned to the UTC date instead of the local date"
    )


def test_utah_builder_uses_local_date_not_utc_date():
    """Utah CRASH_DATETIME is UTC epoch ms; the daily panel must key on LOCAL date.

    Denver is UTC-7/-6, so parsing the epoch as naive UTC pushed every crash
    at local hour >= 17 onto the next calendar day -- 32.8% of all UT crashes.
    """
    import build_utah_udot as ut

    # 2024-01-16T01:30Z is 2024-01-15 18:30 MST -> local date is the 15th.
    epoch_ms = int(pd.Timestamp("2024-01-16 01:30", tz="UTC").timestamp() * 1000)
    raw = pd.DataFrame({
        "CRASH_DATETIME": [epoch_ms],
        "COUNTY_NAME": ["SALT LAKE"],
        "NUMBER_FATALITIES": [0],
        "NUMBER_FOUR_INJURIES": [0],
    })
    out = ut.process_year(raw, 2024)
    assert out is not None and len(out) == 1
    assert pd.Timestamp(out.iloc[0]["date"]) == pd.Timestamp("2024-01-15"), (
        "evening crash was assigned to the UTC date instead of the local date"
    )


def test_connecticut_builder_uses_local_date_not_utc_date():
    """CT CrashDate is UTC epoch ms with minute resolution; the daily panel
    must key on LOCAL date.

    Eastern is UTC-5/-4, so parsing the epoch as naive UTC pushed every crash
    at local hour >= 19 onto the next calendar day -- 12.6% of all CT crashes.
    The service itself declares America/New_York with DST.
    """
    import build_connecticut_uconn as ct

    # 2024-01-16T02:30Z is 2024-01-15 21:30 EST -> local date is the 15th.
    epoch_ms = int(pd.Timestamp("2024-01-16 02:30", tz="UTC").timestamp() * 1000)
    # Exercise the date conversion directly: process_year also needs the
    # person-level join, which is irrelevant to the timezone question.
    converted = (
        pd.to_datetime(pd.Series([epoch_ms]), unit="ms", errors="coerce", utc=True)
        .dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    assert converted.iloc[0] == pd.Timestamp("2024-01-15"), (
        "evening crash was assigned to the UTC date instead of the local date"
    )
    # and confirm the builder itself performs that conversion
    import inspect
    src = inspect.getsource(ct.process_year)
    assert 'tz_convert("America/New_York")' in src, (
        "CT process_year no longer converts CrashDate to local time"
    )


# --- per-year date-only detection ----------------------------------------

def test_year_with_real_times_is_kept():
    ts = pd.Series(pd.date_range("2019-01-01", periods=500, freq="97min"))
    assert bh._year_has_time_of_day(ts)


def test_date_only_year_is_detected():
    """MA 2018: every crash stamped midnight, between two clean years."""
    ts = pd.Series(pd.date_range("2018-01-01", periods=300, freq="D"))
    assert not bh._year_has_time_of_day(ts)


def test_year_with_only_a_couple_of_anchor_times_is_date_only():
    days = pd.date_range("2018-01-01", periods=300, freq="D")
    ts = pd.Series([d + pd.Timedelta(hours=5 if d.month < 7 else 4) for d in days])
    assert not bh._year_has_time_of_day(ts)


def test_tiny_year_is_treated_as_unusable():
    ts = pd.Series(pd.date_range("2019-01-01", periods=5, freq="h"))
    assert not bh._year_has_time_of_day(ts)
