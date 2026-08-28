import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_pooled_matched_event_study as pm


def _src(fips, src, start, end):
    dates = pd.date_range(start, end, freq="D")
    g = pd.MultiIndex.from_product([[fips], dates, [3]],
                                   names=["fips", "date", "hour"]).to_frame(index=False)
    g["crashes"] = 1.0
    g["source"] = src
    return g


def test_each_source_is_balanced_only_over_its_own_span():
    """A source must not be zero-filled outside the years it actually covers."""
    pooled = pd.concat([
        _src("06001", "CA", "2016-01-01", "2016-01-05"),
        _src("10001", "DE", "2013-01-01", "2013-01-03"),
    ], ignore_index=True)
    out = pm.balance_per_source(pooled)
    ca = out[out["source"] == "CA"]
    de = out[out["source"] == "DE"]
    assert ca["date"].min() == pd.Timestamp("2016-01-01")
    assert ca["date"].max() == pd.Timestamp("2016-01-05")
    # DE is never given CA's 2016 dates
    assert de["date"].max() == pd.Timestamp("2013-01-03")
    assert de["date"].dt.year.unique().tolist() == [2013]


def test_balancing_materialises_all_24_hours():
    pooled = _src("06001", "CA", "2016-01-01", "2016-01-02")
    out = pm.balance_per_source(pooled)
    assert sorted(out["hour"].unique()) == list(range(24))
    assert len(out) == 2 * 24


def test_absent_hours_inside_coverage_become_true_zeros():
    pooled = _src("06001", "CA", "2016-01-01", "2016-01-01")   # only hour 3 present
    out = pm.balance_per_source(pooled)
    assert out["crashes"].notna().all()
    assert out.loc[out["hour"] == 3, "crashes"].iloc[0] == 1.0
    assert out.loc[out["hour"] == 4, "crashes"].iloc[0] == 0.0


def test_crash_totals_are_preserved_by_balancing():
    pooled = pd.concat([
        _src("06001", "CA", "2016-01-01", "2016-01-04"),
        _src("25017", "MA", "2015-06-01", "2015-06-03"),
    ], ignore_index=True)
    out = pm.balance_per_source(pooled)
    assert out["crashes"].sum() == pooled["crashes"].sum()
