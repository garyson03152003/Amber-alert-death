import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import build_pooled_hourly_panel as bp


def _panel(fips, col, n=3, val=2.0):
    return pd.DataFrame({
        "fips": [fips] * n,
        "date": pd.to_datetime(["2019-01-01"] * n),
        "hour": list(range(n)),
        col: [val] * n,
    })


def test_native_columns_are_harmonised_to_one_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "DATA_PROC", tmp_path)
    _panel("06001", "ca_crashes").to_parquet(tmp_path / "california_ccrs_county_hour.parquet")
    _panel("10001", "de_crashes").to_parquet(tmp_path / "de_county_hour.parquet")
    out = bp.load_pooled()
    assert "crashes" in out.columns
    assert set(out["source"]) == {"CA", "DE"}
    assert len(out) == 6


def test_missing_panels_are_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "DATA_PROC", tmp_path)
    _panel("06001", "ca_crashes").to_parquet(tmp_path / "california_ccrs_county_hour.parquet")
    out = bp.load_pooled()
    assert set(out["source"]) == {"CA"}


def test_no_panels_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "DATA_PROC", tmp_path)
    try:
        bp.load_pooled()
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_overlapping_county_hours_are_summed_not_duplicated(tmp_path, monkeypatch):
    """Two sources covering the same county must not double the panel rows."""
    monkeypatch.setattr(bp, "DATA_PROC", tmp_path)
    _panel("06001", "ca_crashes", val=2.0).to_parquet(tmp_path / "california_ccrs_county_hour.parquet")
    _panel("06001", "de_crashes", val=5.0).to_parquet(tmp_path / "de_county_hour.parquet")
    out = bp.load_pooled()
    assert len(out) == 3                      # not 6
    assert (out["crashes"] == 7.0).all()      # 2 + 5


def test_crash_totals_are_preserved_across_pooling(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "DATA_PROC", tmp_path)
    a = _panel("06001", "ca_crashes", n=4, val=3.0)
    b = _panel("49035", "ut_crashes", n=5, val=7.0)
    a.to_parquet(tmp_path / "california_ccrs_county_hour.parquet")
    b.to_parquet(tmp_path / "ut_county_hour.parquet")
    out = bp.load_pooled()
    assert out["crashes"].sum() == a["ca_crashes"].sum() + b["ut_crashes"].sum()
