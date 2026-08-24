"""FARS national spillover under three exposure measures.

    1. unweighted            share of the inbound WORKFORCE that was alerted
    2. driver_2020           workforce reweighted by ACS 2020 car share
    3. driver_year_matched   reweighted by the car-share vintage nearest each
                             crash year (2015 / 2017 / 2020 / 2023)

Why (3) is not a formality. The ACS car-commute share is not stable across
the panel: worker-weighted it runs 0.859 (2015) -> 0.856 (2017) -> 0.838
(2020) -> 0.788 (2023), a 7.1pp fall driven by remote work rather than by
mode switching. Against 2015, large-county correlation decays 0.9987 ->
0.9898 -> 0.9455. A uniform shift would cancel in the spillover ratio, but
that decaying correlation is *differential* movement across counties, which
does not cancel -- so a single vintage applied to 2021-2024 misstates
exposure where the drift is largest.

Output: output/tables/fars_spillover_vintage_matched.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
import run_state_dot_analysis_share as share
import run_validated_fars_share as fars
from build_car_shares_vintages import load_car_share_for_year, VINTAGE_FOR_YEAR
from config import DATA_PROC, OUTPUT_TABS
from state_dot_analysis_core import build_car_weighted_spillover

log = base.log
FLOWS = DATA_PROC / "commuting" / "county_commuting_weights.parquet"


def year_matched_spillover(alerts: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Compute driver-weighted spillover year by year, each with its own vintage."""
    alerts = alerts.copy()
    alerts["effective_crash_date"] = pd.to_datetime(alerts["effective_crash_date"])
    parts = []
    for year, chunk in alerts.groupby(alerts["effective_crash_date"].dt.year):
        try:
            cs = load_car_share_for_year(int(year))
        except FileNotFoundError as exc:
            log.warning("year %s: %s -- falling back to 2020", year, exc)
            cs = pd.read_parquet(DATA_PROC / "county_car_commuters.parquet")
        out = build_car_weighted_spillover(chunk, flows, cs)
        if not out.empty:
            out["vintage"] = VINTAGE_FOR_YEAR.get(int(year), "2020")
            parts.append(out)
        log.info("  %s -> vintage %s | %s spillover rows",
                 year, VINTAGE_FOR_YEAR.get(int(year), "2020"), f"{len(out):,}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    panel = fars.build_panel(direct_only=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    alerts = base.load_verified_alerts(window="night")
    flows = pd.read_parquet(FLOWS)

    # (2) single 2020 vintage
    cs2020 = pd.read_parquet(DATA_PROC / "county_car_commuters.parquet")
    dw2020 = build_car_weighted_spillover(alerts, flows, cs2020)
    # (3) year-matched vintages
    dwym = year_matched_spillover(alerts, flows)

    for tag, dw in (("d20", dw2020), ("dym", dwym)):
        if dw.empty:
            log.warning("%s spillover empty", tag); continue
        cols = ["fips", "effective_crash_date", "spillover_driver_share"]
        panel = panel.merge(
            dw[cols].rename(columns={"effective_crash_date": "date",
                                     "spillover_driver_share": f"sds_{tag}"}),
            on=["fips", "date"], how="left")
        panel[f"sds_{tag}"] = panel[f"sds_{tag}"].fillna(0.0)

    both = panel[(panel["sds_d20"] > 0) | (panel["sds_dym"] > 0)]
    log.info("corr(2020-vintage, year-matched) on exposed county-days = %.4f",
             both["sds_d20"].corr(both["sds_dym"]))

    specs = {
        "unweighted": "spillover_share",
        "driver_2020": "sds_d20",
        "driver_year_matched": "sds_dym",
    }
    rows = []
    for label, col in specs.items():
        tmp = panel.copy()
        tmp["spillover_share_10pp"] = tmp[col].fillna(0.0).clip(0, 1) / 0.10
        for r in share.run_ppml(tmp, "fatals", f"FARS_{label.upper()}"):
            if r.get("record_type") == "estimate":
                r["spec"] = label
                rows.append(r)

    out = pd.DataFrame(rows)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_TABS / "fars_spillover_vintage_matched.csv", index=False)
    keep = [c for c in ["spec", "term", "pct_change", "beta", "se", "pvalue", "n_obs"]
            if c in out.columns]
    print("\n" + out[keep].to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
