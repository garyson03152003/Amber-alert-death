"""FARS national: unweighted vs driver-weighted commuter spillover.

The spillover term is the only significant coefficient the national FARS
analysis produced (+7.2%, p=0.025), and it is the least well measured: it
counts commuters, not drivers, so a Manhattan rail commuter carries the same
weight as an Iowa driver. Manhattan's ACS car share is 7.7% against a
national county median of 90.3%, so the mismatch is concentrated in exactly
the dense counties that dominate commuting flows.

This runs the same PPML specification twice -- once with the existing
worker-share treatment, once with the ACS-B08301 driver-weighted share --
so the difference is attributable to the weighting alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
import run_state_dot_analysis_share as share
import run_validated_fars_share as fars
from config import DATA_PROC, OUTPUT_TABS
from state_dot_analysis_core import build_car_weighted_spillover

log = base.log
CAR = DATA_PROC / "county_car_commuters.parquet"
FLOWS = DATA_PROC / "commuting" / "county_commuting_weights.parquet"


def main() -> None:
    panel = fars.build_panel(direct_only=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    alerts = base.load_verified_alerts(window="night")
    flows = pd.read_parquet(FLOWS)
    car = pd.read_parquet(CAR)

    missing = {"11001"} - set(car["fips"].astype(str))
    if missing:
        log.warning("no ACS car share for %s -- those origins are DROPPED from the "
                    "driver weighting, not treated as zero drivers", sorted(missing))

    dw = build_car_weighted_spillover(alerts, flows, car)
    log.info("driver-weighted spillover rows: %s", f"{len(dw):,}")
    panel = panel.merge(
        dw.rename(columns={"effective_crash_date": "date"}),
        on=["fips", "date"], how="left",
    )
    panel["spillover_driver_share"] = panel["spillover_driver_share"].fillna(0.0)
    panel["spillover_driver_share_10pp"] = panel["spillover_driver_share"] / 0.10

    both = panel[(panel["spillover_share"] > 0) | (panel["spillover_driver_share"] > 0)]
    log.info("county-days with any spillover: %s | corr(unweighted, driver) = %.3f",
             f"{len(both):,}",
             both["spillover_share"].corr(both["spillover_driver_share"]))

    rows = []
    for label, term in (("unweighted", "spillover_share_10pp"),
                        ("driver_weighted", "spillover_driver_share_10pp")):
        est = share.run_ppml(panel, "fatals", f"FARS_SPILL_{label.upper()}",
                             treatment_override=("night_alert", term)) \
            if "treatment_override" in share.run_ppml.__code__.co_varnames else None
        if est is None:
            # runner picks its own treatments; emulate by temporarily renaming
            tmp = panel.copy()
            tmp["spillover_share_10pp"] = tmp[term]
            est = share.run_ppml(tmp, "fatals", f"FARS_SPILL_{label.upper()}")
        for r in est:
            if r.get("record_type") == "estimate":
                r["weighting"] = label
                rows.append(r)

    out = pd.DataFrame(rows)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_TABS / "fars_driver_weighted_spillover.csv", index=False)
    keep = [c for c in ["weighting", "term", "pct_change", "beta", "se", "pvalue", "n_obs"]
            if c in out.columns]
    print("\n" + out[keep].to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
