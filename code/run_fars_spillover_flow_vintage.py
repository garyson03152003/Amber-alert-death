"""FARS national: unweighted (workforce) spillover, single vs year-matched FLOWS.

The car-share vintage turned out not to matter because the spillover measure
is a ratio and the car-share drift is near-proportional across counties. The
commuting FLOWS are a different case: only 64,731 OD pairs appear in both the
2011-2015 and 2016-2020 tables, out of ~136k and ~119k respectively, so over
half the pairs are present in one vintage and absent from the other. That is
a compositional change, not a level shift, and it does not obviously cancel.

Caveat on the late years: Census's most recent county-to-county flow table is
the 2016-2020 window, so 2021-2024 must reuse it. Given remote work reshaped
commuting after 2020, those years are the least well measured in the panel
and no available vintage fixes that.

Output: output/tables/fars_spillover_flow_vintage.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
import run_state_dot_analysis_share as share
import run_validated_fars_share as fars
from build_commuting_weights_vintages import load_flows_for_year, VINTAGE_FOR_YEAR
from config import DATA_PROC, OUTPUT_TABS
from state_dot_analysis_core import build_commuter_spillover

log = base.log


def year_matched_flow_spillover(alerts: pd.DataFrame) -> pd.DataFrame:
    alerts = alerts.copy()
    alerts["effective_crash_date"] = pd.to_datetime(alerts["effective_crash_date"])
    parts = []
    for year, chunk in alerts.groupby(alerts["effective_crash_date"].dt.year):
        fl = load_flows_for_year(int(year))
        out = build_commuter_spillover(chunk, fl)
        if not out.empty:
            parts.append(out)
        log.info("  %s -> flow vintage %s | %s rows", year,
                 VINTAGE_FOR_YEAR.get(int(year), "2020"), f"{len(out):,}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    panel = fars.build_panel(direct_only=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    alerts = base.load_verified_alerts(window="night")

    ym = year_matched_flow_spillover(alerts)
    panel = panel.merge(
        ym[["fips", "effective_crash_date", "spillover_share"]]
          .rename(columns={"effective_crash_date": "date",
                           "spillover_share": "spill_ym"}),
        on=["fips", "date"], how="left")
    panel["spill_ym"] = panel["spill_ym"].fillna(0.0)

    exposed = panel[(panel["spillover_share"] > 0) | (panel["spill_ym"] > 0)]
    d = (exposed["spill_ym"] - exposed["spillover_share"]).abs()
    log.info("exposed county-days=%s | corr=%.6f | max|diff|=%.4f | share>0.001=%.4f",
             f"{len(exposed):,}",
             exposed["spillover_share"].corr(exposed["spill_ym"]),
             d.max(), (d > 0.001).mean())

    rows = []
    for label, col in (("flows_2020_only", "spillover_share"),
                       ("flows_year_matched", "spill_ym")):
        tmp = panel.copy()
        tmp["spillover_share_10pp"] = tmp[col].fillna(0.0).clip(0, 1) / 0.10
        for r in share.run_ppml(tmp, "fatals", f"FARS_{label.upper()}"):
            if r.get("record_type") == "estimate":
                r["spec"] = label
                rows.append(r)

    out = pd.DataFrame(rows)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_TABS / "fars_spillover_flow_vintage.csv", index=False)
    keep = [c for c in ["spec", "term", "pct_change", "beta", "se", "pvalue", "n_obs"]
            if c in out.columns]
    print("\n" + out[keep].to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
