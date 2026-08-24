import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_hourly_alert_event_study as ev


def _sparse(rows):
    return pd.DataFrame(rows, columns=["fips", "date", "hour", "ca_crashes"]).astype(
        {"date": "datetime64[ns]"}
    )


# --- balancing ------------------------------------------------------------

def test_balanced_panel_materialises_absent_hours_as_zero():
    sparse = _sparse([
        ("06001", "2016-01-01", 3, 2),
        ("06003", "2016-01-02", 7, 5),
    ])
    panel = ev.build_balanced_hourly_panel(sparse, years=(2016, 2016))
    # 2 counties x 366 days (2016 is a leap year) x 24 hours
    assert len(panel) == 2 * 366 * 24
    assert panel["ca_crashes"].notna().all()
    # the two observed cells keep their values
    hit = panel[(panel["fips"] == "06001") & (panel["date"] == "2016-01-01")
                & (panel["hour"] == 3)]
    assert hit["ca_crashes"].iloc[0] == 2
    # an unobserved cell is a real zero, not NaN
    miss = panel[(panel["fips"] == "06001") & (panel["date"] == "2016-01-01")
                 & (panel["hour"] == 4)]
    assert miss["ca_crashes"].iloc[0] == 0


def test_balanced_panel_covers_every_hour_of_every_day():
    sparse = _sparse([("06001", "2016-06-15", 12, 1)])
    panel = ev.build_balanced_hourly_panel(sparse, years=(2016, 2016))
    per_day = panel.groupby("date")["hour"].nunique()
    assert (per_day == 24).all()


# --- event-hour alignment -------------------------------------------------

def _panel_for(fips, day, n_days=3):
    dates = pd.date_range(day, periods=n_days, freq="D")
    grid = pd.MultiIndex.from_product(
        [[fips], dates, range(24)], names=["fips", "date", "hour"]
    ).to_frame(index=False)
    grid["ca_crashes"] = 0.0
    return grid


def test_event_hour_zero_is_the_alert_clock_hour():
    panel = _panel_for("06001", "2016-03-01")
    alerts = pd.DataFrame({
        "fips": ["06001"],
        "sent_local": [pd.Timestamp("2016-03-01 14:37:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    at14 = out[(out["date"] == "2016-03-01") & (out["hour"] == 14)]
    assert at14["event_hour"].iloc[0] == 0
    at15 = out[(out["date"] == "2016-03-01") & (out["hour"] == 15)]
    assert at15["event_hour"].iloc[0] == 1
    at13 = out[(out["date"] == "2016-03-01") & (out["hour"] == 13)]
    assert at13["event_hour"].iloc[0] == -1


def test_event_window_crosses_midnight_correctly():
    panel = _panel_for("06001", "2016-03-01")
    alerts = pd.DataFrame({
        "fips": ["06001"],
        "sent_local": [pd.Timestamp("2016-03-01 22:10:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    # +3 hours from 22:00 is 01:00 the NEXT day
    nxt = out[(out["date"] == "2016-03-02") & (out["hour"] == 1)]
    assert nxt["event_hour"].iloc[0] == 3


def test_hours_outside_window_are_unlabelled():
    panel = _panel_for("06001", "2016-03-01")
    alerts = pd.DataFrame({
        "fips": ["06001"], "sent_local": [pd.Timestamp("2016-03-01 12:00:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    far = out[(out["date"] == "2016-03-01") & (out["hour"] == 23)]
    assert pd.isna(far["event_hour"].iloc[0])


def test_overlapping_alert_windows_take_the_closest_alert():
    panel = _panel_for("06001", "2016-03-01")
    alerts = pd.DataFrame({
        "fips": ["06001", "06001"],
        "sent_local": [pd.Timestamp("2016-03-01 10:00:00"),
                       pd.Timestamp("2016-03-01 14:00:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    # 13:00 is +3 from the 10:00 alert but -1 from the 14:00 alert -> closest wins
    at13 = out[(out["date"] == "2016-03-01") & (out["hour"] == 13)]
    assert at13["event_hour"].iloc[0] == -1


def test_alerts_do_not_leak_across_counties():
    panel = pd.concat([_panel_for("06001", "2016-03-01"),
                       _panel_for("06003", "2016-03-01")], ignore_index=True)
    alerts = pd.DataFrame({
        "fips": ["06001"], "sent_local": [pd.Timestamp("2016-03-01 14:00:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    other = out[(out["fips"] == "06003")]
    assert other["event_hour"].isna().all()


def test_no_alerts_yields_all_missing_event_hours():
    panel = _panel_for("06001", "2016-03-01")
    out = ev.attach_event_hours(panel, pd.DataFrame(columns=["fips", "sent_local"]))
    assert out["event_hour"].isna().all()


# --- dummies --------------------------------------------------------------

def test_reference_hour_has_no_dummy_and_others_do():
    panel = _panel_for("06001", "2016-03-01")
    alerts = pd.DataFrame({
        "fips": ["06001"], "sent_local": [pd.Timestamp("2016-03-01 12:00:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    out, terms = ev.add_event_dummies(out)
    assert "ev_m1" not in terms          # omitted reference
    assert "ev_p0" in terms and "ev_p1" in terms and "ev_m2" in terms
    assert len(terms) == (ev.EVENT_MAX - ev.EVENT_MIN + 1) - 1
    # the reference hour carries no dummy at all
    ref = out[out["event_hour"] == ev.REFERENCE_EVENT_HOUR]
    assert ref[terms].to_numpy().sum() == 0


def test_dummies_are_mutually_exclusive():
    panel = _panel_for("06001", "2016-03-01")
    alerts = pd.DataFrame({
        "fips": ["06001"], "sent_local": [pd.Timestamp("2016-03-01 12:00:00")],
    })
    out = ev.attach_event_hours(panel, alerts)
    out, terms = ev.add_event_dummies(out)
    assert out[terms].sum(axis=1).max() <= 1


# --- estimation recovers a planted hourly spike ---------------------------

def _panel_with_planted_spike(effect_at_zero=0.0, seed=0, n_counties=20, n_days=200):
    """Balanced county-hour panel where the alert hour has a known multiplier."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    counties = [f"060{c:02d}" for c in range(n_counties)]
    grid = pd.MultiIndex.from_product(
        [counties, dates, range(24)], names=["fips", "date", "hour"]
    ).to_frame(index=False)

    # one alert every 5th day per county, at a rotating hour
    alerts = []
    for i, c in enumerate(counties):
        for d in range(2, n_days, 4):
            hr = 8 + ((i + d) % 10)
            alerts.append({"fips": c,
                           "sent_local": dates[d] + pd.Timedelta(hours=int(hr))})
    alerts = pd.DataFrame(alerts)

    grid = ev.attach_event_hours(grid, alerts)
    # realistic diurnal profile + county scale
    county_scale = {c: rng.uniform(1.0, 4.0) for c in counties}
    diurnal = 1.0 + 0.8 * np.sin((grid["hour"] - 3) / 24 * 2 * np.pi)
    mu = grid["fips"].map(county_scale).to_numpy() * diurnal.to_numpy()
    mu = mu * np.where(grid["event_hour"].to_numpy() == 0, np.exp(effect_at_zero), 1.0)
    grid["ca_crashes"] = rng.poisson(mu).astype(float)
    return grid, alerts


def test_event_study_recovers_planted_alert_hour_spike():
    panel, _ = _panel_with_planted_spike(effect_at_zero=0.40, seed=1)
    panel, terms = ev.add_event_dummies(panel)
    rows = ev.run_event_study(panel, "ca_crashes", terms)
    est = {r["event_hour"]: r for r in rows if r["record_type"] == "estimate"}
    spike = est[0]
    assert spike["beta"] > 0
    assert spike["pvalue"] < 0.05
    # exp(0.40)-1 = ~49%
    assert 25 < spike["pct_change"] < 80


def test_event_study_pre_period_is_flat_when_no_true_effect():
    panel, _ = _panel_with_planted_spike(effect_at_zero=0.40, seed=2)
    panel, terms = ev.add_event_dummies(panel)
    rows = ev.run_event_study(panel, "ca_crashes", terms)
    est = {r["event_hour"]: r for r in rows if r["record_type"] == "estimate"}
    # A term whose clustered SE is non-finite is deliberately dropped rather
    # than reported, so assert over the pre-periods that actually estimated.
    leads = [k for k in (-6, -5, -4, -3, -2) if k in est]
    assert len(leads) >= 3, f"too few estimable pre-periods: {sorted(est)}"
    # alerts cannot affect hours before issuance -> placebo leads near zero
    for k in leads:
        assert est[k]["pvalue"] > 0.01, f"pre-period {k} spuriously significant"


def test_event_study_null_when_nothing_planted():
    panel, _ = _panel_with_planted_spike(effect_at_zero=0.0, seed=3)
    panel, terms = ev.add_event_dummies(panel)
    rows = ev.run_event_study(panel, "ca_crashes", terms)
    est = {r["event_hour"]: r for r in rows if r["record_type"] == "estimate"}
    assert est[0]["pvalue"] > 0.05
    assert est[0]["ci_low_pct"] < 0 < est[0]["ci_high_pct"]


# --- joint placebo test ---------------------------------------------------

def test_joint_pre_period_test_passes_when_only_alert_hour_is_affected():
    panel, _ = _panel_with_planted_spike(effect_at_zero=0.40, seed=11)
    panel, terms = ev.add_event_dummies(panel)
    rows = ev.run_event_study(panel, "ca_crashes", terms)
    pre = next(r for r in rows if r.get("test") == "pre_period_placebo")
    post = next(r for r in rows if r.get("test") == "post_period_joint")
    # leads jointly flat; post-period jointly nonzero because of the spike
    assert pre["pvalue"] > 0.05
    assert post["pvalue"] < 0.05


def test_joint_pre_period_test_detects_a_planted_pre_trend():
    """If crashes rise *before* issuance, the placebo must fail loudly."""
    rng = np.random.default_rng(21)
    panel, _ = _panel_with_planted_spike(effect_at_zero=0.0, seed=21)
    # inflate the hours preceding each alert -> a genuine pre-trend
    pre_mask = panel["event_hour"].isin([-4, -3, -2])
    panel.loc[pre_mask, "ca_crashes"] = panel.loc[pre_mask, "ca_crashes"] + rng.poisson(
        3.0, size=int(pre_mask.sum())
    )
    panel, terms = ev.add_event_dummies(panel)
    rows = ev.run_event_study(panel, "ca_crashes", terms)
    pre = next(r for r in rows if r.get("test") == "pre_period_placebo")
    assert pre["pvalue"] < 0.05, "planted pre-trend was not detected"
