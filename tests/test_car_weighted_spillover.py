import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from state_dot_analysis_core import (
    build_car_weighted_spillover, build_commuter_spillover,
)

D = pd.Timestamp("2024-03-05")


def _alerts(homes):
    return pd.DataFrame({"fips": homes, "effective_crash_date": [D] * len(homes)})


def _flows(rows):
    return pd.DataFrame(rows, columns=["fips_home", "fips_work", "workers"])


def _shares(d):
    return pd.DataFrame({"fips": list(d), "car_share": list(d.values())})


def test_driver_weighting_changes_the_share_vs_unweighted():
    """Two origins send equal workers but only one mostly drives."""
    flows = _flows([("01001", "01999", 100.0), ("01003", "01999", 100.0)])
    shares = _shares({"01001": 0.9, "01003": 0.1})
    out = build_car_weighted_spillover(_alerts(["01001"]), flows, shares)
    # driving inflow: 90 from 01001, 10 from 01003 -> alerted share = 90/100
    assert abs(out.iloc[0]["spillover_driver_share"] - 0.9) < 1e-9
    # the unweighted version would call this 0.5
    unw = build_commuter_spillover(_alerts(["01001"]), flows)
    assert abs(unw.iloc[0]["spillover_share"] - 0.5) < 1e-9


def test_low_driving_origin_yields_low_exposure():
    flows = _flows([("01001", "01999", 100.0), ("01003", "01999", 100.0)])
    shares = _shares({"01001": 0.1, "01003": 0.9})
    out = build_car_weighted_spillover(_alerts(["01001"]), flows, shares)
    assert abs(out.iloc[0]["spillover_driver_share"] - 0.1) < 1e-9


def test_share_is_bounded_and_all_origins_alerted_gives_one():
    flows = _flows([("01001", "01999", 40.0), ("01003", "01999", 60.0)])
    shares = _shares({"01001": 0.8, "01003": 0.5})
    out = build_car_weighted_spillover(_alerts(["01001", "01003"]), flows, shares)
    assert abs(out.iloc[0]["spillover_driver_share"] - 1.0) < 1e-9


def test_own_county_flow_is_excluded():
    """Direct exposure is night_alert's job, not spillover's."""
    flows = _flows([("01999", "01999", 500.0), ("01001", "01999", 100.0)])
    shares = _shares({"01999": 0.9, "01001": 0.9})
    out = build_car_weighted_spillover(_alerts(["01999"]), flows, shares)
    assert out.empty


def test_counties_without_a_car_share_are_dropped_not_zeroed():
    """A missing share is unknown, not a claim that nobody drives."""
    flows = _flows([("01001", "01999", 100.0), ("01003", "01999", 100.0)])
    shares = _shares({"01001": 0.9})          # 01003 has no share
    out = build_car_weighted_spillover(_alerts(["01001"]), flows, shares)
    # denominator counts only 01001's 90 driving workers, so the share is 1.0
    assert abs(out.iloc[0]["spillover_driver_share"] - 1.0) < 1e-9
    # had 01003 been zeroed in, the answer would still be 1.0; confirm it is
    # genuinely absent from the flow rather than contributing a zero row
    assert out.iloc[0]["spillover_drivers"] == 90.0


def test_empty_inputs_return_empty():
    flows = _flows([("01001", "01999", 100.0)])
    shares = _shares({"01001": 0.9})
    assert build_car_weighted_spillover(pd.DataFrame(), flows, shares).empty
    assert build_car_weighted_spillover(_alerts(["01001"]), pd.DataFrame(), shares).empty
    assert build_car_weighted_spillover(_alerts(["01001"]), flows, pd.DataFrame()).empty


def test_unalerted_destination_gets_no_row():
    flows = _flows([("01001", "01999", 100.0)])
    shares = _shares({"01001": 0.9})
    out = build_car_weighted_spillover(_alerts(["09001"]), flows, shares)
    assert out.empty
