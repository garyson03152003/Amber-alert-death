import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import audit_timestamp_timezones as au


def _epoch_ms(ts):
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def test_date_only_epochs_are_classified_safe():
    """Values pinned to UTC midnight carry no time -> parsing as UTC is fine."""
    days = pd.date_range("2019-01-01", periods=200, freq="D")
    vals = pd.Series([_epoch_ms(d) for d in days])
    assert au.classify_epoch_ms(vals)["verdict"] == "date_only_SAFE"


def test_too_few_rows_falls_back_to_instant_rather_than_guessing():
    """With a handful of rows the anchor pattern is not established, so the
    audit must flag for review rather than declare the source safe."""
    vals = pd.Series([_epoch_ms("2019-03-04"), _epoch_ms("2019-07-19")])
    assert au.classify_epoch_ms(vals)["verdict"] == "instant_AFFECTED"


def test_epochs_with_a_time_component_are_flagged_affected():
    vals = pd.Series([_epoch_ms("2019-03-04 21:30"), _epoch_ms("2019-07-19 08:15")])
    assert au.classify_epoch_ms(vals)["verdict"] == "instant_AFFECTED"


def test_a_single_timed_row_among_midnights_still_flags():
    """A mostly-midnight column with real times is still an instant column."""
    vals = pd.Series([_epoch_ms("2019-03-04")] * 100 + [_epoch_ms("2019-03-05 18:00")])
    assert au.classify_epoch_ms(vals)["verdict"] == "instant_AFFECTED"


def test_date_shift_share_matches_the_evening_share():
    """Only evening crashes cross the UTC boundary; morning ones do not."""
    evening = [_epoch_ms("2019-01-15 21:00")] * 3      # 16:00 EST -> no shift
    late = [_epoch_ms("2019-01-16 02:00")] * 1         # 21:00 EST prev day -> shift
    vals = pd.Series(evening + late)
    share = au.date_shift_share(vals, "America/New_York")
    assert abs(share - 0.25) < 1e-9


def test_date_shift_respects_dst_boundary_hour():
    """Eastern rolls over at 19:00 EST but 20:00 EDT -- a fixed offset is wrong."""
    # 19:30 local on a winter date -> already next UTC day
    winter = pd.Series([_epoch_ms("2019-01-16 00:30")])   # 19:30 EST on the 15th
    # 19:30 local on a summer date -> still same UTC day (EDT is UTC-4)
    summer = pd.Series([_epoch_ms("2019-07-15 23:30")])   # 19:30 EDT on the 15th
    assert au.date_shift_share(winter, "America/New_York") == 1.0
    assert au.date_shift_share(summer, "America/New_York") == 0.0


def test_honolulu_has_no_dst_but_still_shifts():
    """Hawaii is UTC-10 year round -- the largest offset of any source here."""
    vals = pd.Series([_epoch_ms("2019-01-16 05:00")])     # 19:00 HST on the 15th
    assert au.date_shift_share(vals, "Pacific/Honolulu") == 1.0


def test_empty_column_is_handled():
    assert au.classify_epoch_ms(pd.Series([], dtype=float))["verdict"] == "no_data"


# --- three-way instant classification -------------------------------------

def _crashlike_hours(n=6000, seed=0):
    """Hours drawn from a realistic crash curve: evening peak, pre-dawn trough."""
    import numpy as np
    rng = np.random.default_rng(seed)
    w = np.array([.020,.014,.011,.010,.011,.018,.032,.050,.052,.042,.041,.048,
                  .054,.056,.062,.072,.076,.078,.064,.048,.038,.032,.028,.023])
    return rng.choice(np.arange(24), size=n, p=w/w.sum())


def _epochs_at_local_hours(hours, tz, *, stored_as):
    """Build epoch-ms for crashes at given LOCAL hours.

    stored_as="utc"   -> the service publishes the true instant (needs convert)
    stored_as="local" -> the service publishes local wall-clock as if UTC
    """
    base = pd.Timestamp("2019-06-15")
    out = []
    for h in hours:
        local = pd.Timestamp(base + pd.Timedelta(hours=int(h)))
        if stored_as == "utc":
            inst = local.tz_localize(tz).tz_convert("UTC")
        else:
            inst = local.tz_localize("UTC")
        out.append(int(inst.timestamp() * 1000))
    return pd.Series(out)


def test_genuine_utc_source_is_flagged_affected():
    hours = _crashlike_hours()
    vals = _epochs_at_local_hours(hours, "America/New_York", stored_as="utc")
    res = au.classify_instant_timezone(vals, "America/New_York")
    assert res["verdict"] == "genuine_utc_AFFECTED"
    assert 14 <= res["local_peak"] <= 19


def test_local_stored_as_utc_source_is_flagged_safe():
    """The Massachusetts case: converting would CREATE the bug."""
    hours = _crashlike_hours()
    vals = _epochs_at_local_hours(hours, "America/New_York", stored_as="local")
    res = au.classify_instant_timezone(vals, "America/New_York")
    assert res["verdict"] == "local_stored_as_utc_SAFE"
    assert 14 <= res["raw_utc_peak"] <= 19


def test_the_two_kinds_are_distinguished_despite_identical_dtypes():
    hours = _crashlike_hours()
    utc_kind = _epochs_at_local_hours(hours, "America/Denver", stored_as="utc")
    loc_kind = _epochs_at_local_hours(hours, "America/Denver", stored_as="local")
    assert utc_kind.dtype == loc_kind.dtype          # byte-identical types
    a = au.classify_instant_timezone(utc_kind, "America/Denver")["verdict"]
    b = au.classify_instant_timezone(loc_kind, "America/Denver")["verdict"]
    assert a == "genuine_utc_AFFECTED"
    assert b == "local_stored_as_utc_SAFE"


# --- date-only columns anchored somewhere other than UTC midnight ---------

def test_date_only_anchored_at_est_midnight_is_safe():
    """Florida's real encoding: every crash pinned to 05:00 UTC = EST midnight."""
    days = pd.date_range("2019-01-01", periods=200, freq="D")
    vals = pd.Series([int((d + pd.Timedelta(hours=5)).timestamp() * 1000) for d in days])
    res = au.classify_epoch_ms(vals)
    assert res["verdict"] == "date_only_SAFE"
    assert res["n_distinct_times_of_day"] == 1
    assert res["anchor_hours_utc"] == [5.0]


def test_date_only_anchored_at_utc_noon_is_safe():
    """Oregon's real encoding: every crash pinned to 12:00 UTC."""
    days = pd.date_range("2019-01-01", periods=200, freq="D")
    vals = pd.Series([int((d + pd.Timedelta(hours=12)).timestamp() * 1000) for d in days])
    res = au.classify_epoch_ms(vals)
    assert res["verdict"] == "date_only_SAFE"
    assert res["anchor_hours_utc"] == [12.0]


def test_date_only_with_a_dst_pair_of_anchors_is_still_safe():
    """A source anchored on LOCAL midnight alternates two UTC offsets."""
    days = pd.date_range("2019-01-01", periods=200, freq="D")
    vals = pd.Series([
        int((d + pd.Timedelta(hours=5 if d.month in (1, 2, 11, 12) else 4)).timestamp() * 1000)
        for d in days
    ])
    res = au.classify_epoch_ms(vals)
    assert res["verdict"] == "date_only_SAFE"
    assert res["n_distinct_times_of_day"] == 2


def test_a_real_instant_column_is_still_flagged():
    """Minute-resolution times must not be mistaken for a date-only anchor."""
    rng = pd.date_range("2019-01-01", periods=500, freq="97min")
    vals = pd.Series([int(t.timestamp() * 1000) for t in rng])
    assert au.classify_epoch_ms(vals)["verdict"] == "instant_AFFECTED"
