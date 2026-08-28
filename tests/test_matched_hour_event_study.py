import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_matched_hour_event_study as mh


def _panel(counties=("06001",), start="2016-01-01", n_days=120, value=1.0):
    dates = pd.date_range(start, periods=n_days, freq="D")
    grid = pd.MultiIndex.from_product(
        [list(counties), dates, range(24)], names=["fips", "date", "hour"]
    ).to_frame(index=False)
    grid["ts"] = grid["date"] + pd.to_timedelta(grid["hour"], unit="h")
    grid["ca_crashes"] = value
    return grid


def _alert(fips="06001", when="2016-02-10 14:00"):
    return pd.DataFrame({"fips": [fips], "sent_local": [pd.Timestamp(when)]})


def test_controls_are_same_weekday_and_same_clock_hour():
    panel = _panel()
    alerts = _alert(when="2016-02-10 14:00")   # a Wednesday
    s = mh.build_matched_sample(panel, alerts, outcome="ca_crashes", control_weeks=2)
    at0 = s[s["offset"] == 0]
    assert set(at0["ts"].dt.hour) == {14}
    assert set(at0["ts"].dt.dayofweek) == {2}          # all Wednesdays
    assert set(at0["week"]) == {-2, -1, 0, 1, 2}


def test_exactly_one_treated_week_per_stratum():
    panel = _panel()
    s = mh.build_matched_sample(panel, _alert(), outcome="ca_crashes", control_weeks=3)
    per = s.groupby("stratum")["treated"].sum()
    assert (per == 1).all()


def test_control_weeks_offset_by_whole_weeks():
    panel = _panel()
    alerts = _alert(when="2016-02-10 14:00")
    s = mh.build_matched_sample(panel, alerts, outcome="ca_crashes", control_weeks=2)
    at0 = s[s["offset"] == 0].sort_values("week")
    deltas = at0["ts"].diff().dropna().dt.days.unique()
    assert set(deltas) == {7}


def test_contaminated_control_weeks_are_dropped():
    """A control slot that falls inside another alert's window is not a control."""
    panel = _panel()
    # two alerts exactly 7 days apart -> each is in the other's control week
    alerts = pd.DataFrame({
        "fips": ["06001", "06001"],
        "sent_local": [pd.Timestamp("2016-02-10 14:00"),
                       pd.Timestamp("2016-02-17 14:00")],
    })
    s = mh.build_matched_sample(panel, alerts, outcome="ca_crashes", control_weeks=2)
    # no surviving control row may coincide with either alert's window
    controls = s[s["treated"] == 0]
    bad = controls["ts"].isin([
        pd.Timestamp("2016-02-10 14:00"), pd.Timestamp("2016-02-17 14:00"),
    ])
    assert not bad.any()


def test_alerts_do_not_borrow_controls_from_other_counties():
    panel = _panel(counties=("06001", "06003"))
    s = mh.build_matched_sample(panel, _alert("06001"), outcome="ca_crashes",
                                control_weeks=2)
    assert set(s["fips"]) == {"06001"}


def test_empty_alerts_yields_empty_sample():
    panel = _panel()
    s = mh.build_matched_sample(panel, pd.DataFrame(columns=["fips", "sent_local"]),
                                outcome="ca_crashes")
    assert s.empty


def test_offset_interactions_are_treated_only():
    panel = _panel()
    s = mh.build_matched_sample(panel, _alert(), outcome="ca_crashes", control_weeks=2)
    s, terms = mh.add_offset_interactions(s)
    assert terms, "expected at least one estimable interaction"
    # every interaction dummy implies treated == 1
    for t in terms:
        assert (s.loc[s[t] == 1, "treated"] == 1).all()
    # controls carry no interaction dummies at all
    assert s.loc[s["treated"] == 0, terms].to_numpy().sum() == 0


# --- recovery of a planted effect ----------------------------------------

def _panel_with_alerts(effect_at_zero=0.0, pre_effect_at_m4=0.0, seed=0,
                       n_counties=6, n_weeks=40):
    """Balanced panel with alerts, a planted alert-hour effect, and optionally
    a planted PRE-alert bump to mimic incident contamination."""
    rng = np.random.default_rng(seed)
    n_days = n_weeks * 7
    counties = [f"060{c:02d}" for c in range(n_counties)]
    panel = _panel(counties=counties, n_days=n_days, value=0.0)

    alerts = []
    for i, c in enumerate(counties):
        for wk in range(5, n_weeks - 5, 3):
            day = wk * 7 + (i % 5)
            hr = 10 + (i + wk) % 8
            alerts.append({"fips": c,
                           "sent_local": pd.Timestamp("2016-01-01")
                           + pd.Timedelta(days=int(day), hours=int(hr))})
    alerts = pd.DataFrame(alerts)

    alert_ts = {(r.fips, r.sent_local.floor("h")) for r in alerts.itertuples()}
    mu = np.full(len(panel), 2.0)
    key = list(zip(panel["fips"], panel["ts"]))
    for k, (f, ts) in enumerate(key):
        for dk, eff in ((0, effect_at_zero), (-4, pre_effect_at_m4)):
            if eff and (f, ts - pd.Timedelta(hours=dk)) in alert_ts:
                mu[k] *= np.exp(eff)
    panel["ca_crashes"] = rng.poisson(mu).astype(float)
    return panel, alerts


def test_matched_model_recovers_planted_alert_hour_effect():
    panel, alerts = _panel_with_alerts(effect_at_zero=0.45, seed=1)
    s = mh.build_matched_sample(panel, alerts, outcome="ca_crashes", control_weeks=3)
    s, terms = mh.add_offset_interactions(s)
    rows = mh.run_matched_model(s, "ca_crashes", terms)
    est = {r["event_hour"]: r for r in rows if r["record_type"] == "estimate"}
    assert est[0]["beta"] > 0
    assert est[0]["pvalue"] < 0.05
    assert 25 < est[0]["pct_change"] < 90


def test_matched_model_is_null_when_nothing_planted():
    panel, alerts = _panel_with_alerts(effect_at_zero=0.0, seed=2)
    s = mh.build_matched_sample(panel, alerts, outcome="ca_crashes", control_weeks=3)
    s, terms = mh.add_offset_interactions(s)
    rows = mh.run_matched_model(s, "ca_crashes", terms)
    est = {r["event_hour"]: r for r in rows if r["record_type"] == "estimate"}
    assert est[0]["pvalue"] > 0.05
    assert est[0]["ci_low_pct"] < 0 < est[0]["ci_high_pct"]


def test_matched_model_separates_pre_alert_contamination_from_alert_effect():
    """The whole point of this design: a pre-alert bump must show up at -4 and
    must NOT be differenced into the alert-hour estimate."""
    panel, alerts = _panel_with_alerts(effect_at_zero=0.0, pre_effect_at_m4=0.50,
                                       seed=3)
    s = mh.build_matched_sample(panel, alerts, outcome="ca_crashes", control_weeks=3)
    s, terms = mh.add_offset_interactions(s)
    rows = mh.run_matched_model(s, "ca_crashes", terms)
    est = {r["event_hour"]: r for r in rows if r["record_type"] == "estimate"}
    assert est[-4]["beta"] > 0 and est[-4]["pvalue"] < 0.05, "pre-bump not detected"
    # alert hour has no planted effect and must stay null despite the -4 bump
    assert est[0]["pvalue"] > 0.05
