import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_state_dot_analysis_fixed as fixed_runner
import run_day_alert_crash_analysis as day_runner


# --- alert-window construction -------------------------------------------

def _alert_csv(tmp_path, rows):
    path = tmp_path / "openfema_ipaws_alerts_2013_2024.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _patch_alert_path(monkeypatch, tmp_path, rows):
    raw = tmp_path / "amber" / "foia"
    raw.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(raw / "openfema_ipaws_alerts_2013_2024.csv", index=False)
    monkeypatch.setattr(fixed_runner, "DATA_RAW", tmp_path)


def test_day_window_keeps_daytime_alert_on_same_date(monkeypatch, tmp_path):
    # 18:00 UTC = 13:00 America/Chicago -> a daytime alert on the same date.
    _patch_alert_path(monkeypatch, tmp_path, [{
        "alert_id": "a1", "sent_utc": "2024-06-03T18:00:00Z",
        "fips": "01001", "state_fips": "01", "msg_type": "Alert",
    }])
    out = fixed_runner.load_verified_alerts(window="day")
    assert len(out) == 1
    assert out.iloc[0]["day_alert"] == 1
    # Same-day assignment: no next-day shift for daytime alerts.
    assert out.iloc[0]["effective_crash_date"] == pd.Timestamp("2024-06-03")


def test_day_window_excludes_night_alert_and_vice_versa(monkeypatch, tmp_path):
    # 04:00 UTC = 23:00 America/Chicago the previous day -> night window.
    _patch_alert_path(monkeypatch, tmp_path, [{
        "alert_id": "a1", "sent_utc": "2024-06-04T04:00:00Z",
        "fips": "01001", "state_fips": "01", "msg_type": "Alert",
    }])
    assert fixed_runner.load_verified_alerts(window="day").empty
    night = fixed_runner.load_verified_alerts(window="night")
    assert len(night) == 1
    # 23:00 local is carried forward to the next day's driving.
    assert night.iloc[0]["effective_crash_date"] == pd.Timestamp("2024-06-04")


def test_windows_partition_alerts_without_overlap(monkeypatch, tmp_path):
    rows = [
        {"alert_id": f"a{h}", "sent_utc": f"2024-06-03T{h:02d}:00:00Z",
         "fips": "01001", "state_fips": "01", "msg_type": "Alert"}
        for h in range(24)
    ]
    _patch_alert_path(monkeypatch, tmp_path, rows)
    day = fixed_runner.load_verified_alerts(window="day", detail=True)
    night = fixed_runner.load_verified_alerts(window="night", detail=True)
    assert len(day) + len(night) == 24
    assert set(day["alert_id"]) & set(night["alert_id"]) == set()
    assert day["hour_local"].between(6, 21).all()
    assert (~night["hour_local"].between(6, 21)).all()


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        fixed_runner.load_verified_alerts(window="evening")


def test_night_wrapper_matches_night_window(monkeypatch, tmp_path):
    _patch_alert_path(monkeypatch, tmp_path, [{
        "alert_id": "a1", "sent_utc": "2024-06-04T04:00:00Z",
        "fips": "01001", "state_fips": "01", "msg_type": "Alert",
    }])
    pd.testing.assert_frame_equal(
        fixed_runner.load_verified_night_alerts(),
        fixed_runner.load_verified_alerts(window="night"),
    )


# --- estimation -----------------------------------------------------------

def _crash_panel(day_effect=0.0, seed=0, n_counties=60, n_days=90):
    """Synthetic county-day crash panel with a planted day-alert effect.

    Treatment is scattered randomly across county-days rather than on a fixed
    modular schedule: two-way county+date clustering needs treated units
    spread over many date clusters, or the clustered vcov is not positive
    definite and the standard error comes back non-finite.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_counties):
        fips = f"{c:05d}"
        base_rate = rng.uniform(1.0, 4.0)
        for d in range(n_days):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            day_alert = int(rng.random() < 0.04)
            night_alert = int(rng.random() < 0.03)
            mu = base_rate * np.exp(day_effect * day_alert)
            rows.append({
                "fips": fips, "date": date, "year": 2024, "state": "XX",
                "population": 100_000.0,
                "crashes": float(rng.poisson(mu)),
                "fatals": float(rng.poisson(mu * 0.05)),
                "serious_inj": float(rng.poisson(mu * 0.1)),
                "day_alert": day_alert, "night_alert": night_alert,
            })
    panel = pd.DataFrame(rows)
    for count, rate in [("crashes", "crashes_per_100k"),
                        ("fatals", "fatals_per_100k"),
                        ("serious_inj", "serious_per_100k")]:
        panel[rate] = 100_000 * panel[count] / panel["population"]
    return panel


def test_ppml_recovers_planted_positive_day_alert_effect():
    panel = _crash_panel(day_effect=0.30, seed=1)
    rows = day_runner.run_ppml(panel, "crashes", "TEST")
    est = next(r for r in rows
               if r["record_type"] == "estimate" and r["term"] == "day_alert")
    assert est["beta"] > 0
    assert est["pvalue"] < 0.05
    # exp(0.30) - 1 = ~35% ; allow generous tolerance for Poisson noise
    assert 15 < est["pct_change"] < 60


def test_ppml_null_effect_is_not_significant():
    panel = _crash_panel(day_effect=0.0, seed=2)
    rows = day_runner.run_ppml(panel, "crashes", "TEST")
    est = next(r for r in rows
               if r["record_type"] == "estimate" and r["term"] == "day_alert")
    assert est["pvalue"] > 0.05
    assert est["ci_low_pct"] < 0 < est["ci_high_pct"]


def test_both_treatments_estimated_jointly():
    panel = _crash_panel(day_effect=0.2, seed=3)
    rows = day_runner.run_ppml(panel, "crashes", "TEST")
    terms = {r["term"] for r in rows if r["record_type"] == "estimate"}
    assert terms == {"day_alert", "night_alert"}


def test_treatment_without_variation_is_dropped_not_fitted():
    panel = _crash_panel(day_effect=0.2, seed=4)
    panel["night_alert"] = 0  # no variation
    rows = day_runner.run_ppml(panel, "crashes", "TEST")
    terms = {r["term"] for r in rows if r["record_type"] == "estimate"}
    assert terms == {"day_alert"}


def test_single_county_sample_is_skipped_not_fitted():
    panel = _crash_panel(day_effect=0.2, seed=5)
    panel = panel[panel["fips"] == panel["fips"].iloc[0]]
    rows = day_runner.run_ppml(panel, "crashes", "TEST")
    assert rows[0]["status"] == "skipped"
    assert rows[0]["error_reason"] == "insufficient_estimable_sample"


def test_wls_agrees_in_sign_with_ppml():
    panel = _crash_panel(day_effect=0.30, seed=6)
    wls = next(r for r in day_runner.run_wls(panel, "crashes_per_100k", "TEST")
               if r["record_type"] == "estimate" and r["term"] == "day_alert")
    assert wls["beta"] > 0


def test_nonfinite_standard_error_is_dropped_not_reported():
    """A treatment too sparse for two-way clustering yields a non-finite SE.

    That must be filtered out rather than surfaced as an apparently valid
    estimate -- reporting a coefficient whose SE is NaN would be worse than
    reporting nothing.
    """
    panel = _crash_panel(day_effect=0.2, seed=7)
    # Concentrate night alerts on a single date so the clustered vcov for
    # that term cannot be computed.
    panel["night_alert"] = 0
    one_date = panel["date"].iloc[0]
    panel.loc[panel["date"] == one_date, "night_alert"] = 1
    rows = day_runner.run_ppml(panel, "crashes", "TEST")
    estimates = [r for r in rows if r["record_type"] == "estimate"]
    for r in estimates:
        assert np.isfinite(r["se"]), f"non-finite SE surfaced for {r['term']}"
        assert np.isfinite(r["pvalue"])


# --- daylight-saving handling in the alert treatment ---------------------

def test_alert_local_hour_is_dst_correct_across_both_transitions(monkeypatch, tmp_path):
    """UTC->local must apply the right offset on either side of a transition.

    Converting *from* UTC is always unambiguous (the ambiguity of a repeated
    or skipped local hour only arises going local->UTC), so tz_convert is the
    correct tool and a fixed offset is not.
    """
    rows = [
        # 2024 spring forward: 02:00 EST -> 03:00 EDT on Mar 10
        {"alert_id": "pre",  "sent_utc": "2024-03-10T06:30:00Z",   # 01:30 EST
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
        {"alert_id": "post", "sent_utc": "2024-03-10T07:30:00Z",   # 03:30 EDT
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
        # 2024 fall back: 01:00 occurs twice on Nov 3
        {"alert_id": "fall1", "sent_utc": "2024-11-03T05:30:00Z",  # 01:30 EDT
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
        {"alert_id": "fall2", "sent_utc": "2024-11-03T06:30:00Z",  # 01:30 EST
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
    ]
    _patch_alert_path(monkeypatch, tmp_path, rows)
    detail = fixed_runner.load_verified_alerts(window="night", detail=True)
    by_id = detail.set_index("alert_id")["hour_local"].to_dict()

    assert by_id["pre"] == 1     # 01:30 EST
    assert by_id["post"] == 3    # 03:30 EDT -- 02:00 never exists
    # both repetitions of the fall-back hour read as 01:00 local...
    assert by_id["fall1"] == 1
    assert by_id["fall2"] == 1


def test_both_fallback_repetitions_stay_inside_the_night_window(monkeypatch, tmp_path):
    """The repeated 01:00 hour is night either way, so no alert is
    misclassified by the fall-back transition."""
    rows = [
        {"alert_id": "fall1", "sent_utc": "2024-11-03T05:30:00Z",
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
        {"alert_id": "fall2", "sent_utc": "2024-11-03T06:30:00Z",
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
    ]
    _patch_alert_path(monkeypatch, tmp_path, rows)
    assert len(fixed_runner.load_verified_alerts(window="night", detail=True)) == 2
    assert fixed_runner.load_verified_alerts(window="day", detail=True).empty


def test_next_day_shift_is_calendar_arithmetic_not_24_hours(monkeypatch, tmp_path):
    """A 22:00-23:59 alert rolls to the next CALENDAR date.

    The shift is applied to a naive normalised date, so it is immune to the
    23- and 25-hour days a DST transition creates. Adding 24h to a tz-aware
    timestamp would not be.
    """
    rows = [
        # 2024-03-09 23:30 EST -> effective date should be 2024-03-10,
        # the 23-hour spring-forward day.
        {"alert_id": "a", "sent_utc": "2024-03-10T04:30:00Z",
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"},
    ]
    _patch_alert_path(monkeypatch, tmp_path, rows)
    out = fixed_runner.load_verified_alerts(window="night")
    assert out.iloc[0]["effective_crash_date"] == pd.Timestamp("2024-03-10")


# --- night-window boundary ------------------------------------------------

def test_night_start_20_captures_evening_alerts_the_2200_cutoff_drops(monkeypatch, tmp_path):
    """An overnight alert window runs night_start -> 06:00 across a date
    boundary. With the legacy 22:00 cutoff a 20:30 alert is classed as a DAY
    alert on its own date; with night_start=20 it becomes a night alert
    mapped to the NEXT day, which is the day whose driving it affects."""
    rows = [{
        "alert_id": "evening", "sent_utc": "2024-06-04T01:30:00Z",  # 20:30 CDT Jun 3
        "fips": "01001", "state_fips": "01", "msg_type": "Alert",
    }]
    _patch_alert_path(monkeypatch, tmp_path, rows)

    legacy = fixed_runner.load_verified_alerts(window="night", night_start=22)
    assert legacy.empty, "20:30 alert should not be a night alert under the 22:00 cutoff"

    corrected = fixed_runner.load_verified_alerts(window="night", night_start=20)
    assert len(corrected) == 1
    assert corrected.iloc[0]["effective_crash_date"] == pd.Timestamp("2024-06-04")


def test_day_window_is_the_complement_of_the_night_window(monkeypatch, tmp_path):
    rows = [
        {"alert_id": f"h{h}", "sent_utc": f"2024-06-03T{h:02d}:00:00Z",
         "fips": "36001", "state_fips": "36", "msg_type": "Alert"}
        for h in range(24)
    ]
    _patch_alert_path(monkeypatch, tmp_path, rows)
    for ns in (20, 22):
        night = fixed_runner.load_verified_alerts(window="night", detail=True, night_start=ns)
        day = fixed_runner.load_verified_alerts(window="day", detail=True, night_start=ns)
        assert len(night) + len(day) == 24, f"windows must partition at night_start={ns}"
        assert set(night["alert_id"]) & set(day["alert_id"]) == set()
        assert day["hour_local"].between(6, ns - 1).all()


def test_invalid_night_start_rejected():
    for bad in (6, 3, 24, 0):
        try:
            fixed_runner.load_verified_alerts(window="night", night_start=bad)
            assert False, f"night_start={bad} should be rejected"
        except ValueError:
            pass
