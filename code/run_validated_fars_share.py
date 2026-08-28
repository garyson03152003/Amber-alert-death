"""Run the national FARS fatality model from the validated balanced panel.

This runner is intentionally separate from state-DOT estimates: FARS measures
fatal crashes and person fatalities nationwide, not all police-reported crashes
or serious injuries.  It accepts only the Task 3 balanced FARS output and its
coverage manifest, never the legacy sparse FARS file.
"""
from __future__ import annotations

import argparse

import pandas as pd

import run_state_dot_analysis_fixed as base
import run_state_dot_analysis_share as share
from config import DATA_PROC, OUTPUT_TABS
from state_dot_analysis_core import (
    add_spillover_classes,
    build_commuter_spillover,
    summarize_fit_statuses,
    validate_analysis_inputs,
)


FARS_BALANCED = DATA_PROC / "fars_balanced_county_day.parquet"
FARS_MANIFEST = DATA_PROC / "coverage" / "fars_coverage.csv"


def load_validated_fars(*, direct_only: bool = False, flows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load canonical FARS only after its balancing provenance is verified."""
    if not FARS_BALANCED.is_file():
        raise FileNotFoundError(f"validated balanced FARS panel not found: {FARS_BALANCED}")
    if not FARS_MANIFEST.is_file():
        raise FileNotFoundError(f"FARS coverage manifest not found: {FARS_MANIFEST}")
    panel = pd.read_parquet(FARS_BALANCED).copy()
    manifest = pd.read_csv(FARS_MANIFEST)
    # The manifest deliberately includes invalid Connecticut policy rows. They
    # document a geography exclusion and are not FARS download reporting units;
    # only canonical national FARS coverage may authorize this panel.
    if not {"source", "state"}.issubset(manifest.columns):
        raise ValueError("FARS coverage manifest missing source or state")
    manifest = manifest.loc[
        manifest["source"].astype(str).eq("FARS_NHTSA") & manifest["state"].astype(str).eq("US")
    ].copy()
    panel_years = set(pd.to_datetime(panel["date"], errors="coerce").dt.year.dropna().astype(int))
    manifest_years = set(pd.to_numeric(manifest["year"], errors="coerce").dropna().astype(int))
    if panel_years - manifest_years:
        raise ValueError(f"FARS coverage manifest missing panel years: {sorted(panel_years - manifest_years)}")
    validate_analysis_inputs(panel, manifest, review=None, flows=flows,
                             direct_only=direct_only, require_review=False)
    required = {"person_fatals", "fatal_crashes"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"validated FARS panel missing canonical outcome columns: {sorted(missing)}")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["year"] = panel["date"].dt.year
    panel["state"] = panel["fips"].str[:2]
    return panel.rename(columns={"person_fatals": "fatals"})


def build_panel(*, direct_only: bool = False) -> pd.DataFrame:
    """Merge population and nationwide origin alerts into FARS destinations."""
    flows_path = DATA_PROC / "commuting" / "county_commuting_weights.parquet"
    flows = pd.read_parquet(flows_path) if flows_path.is_file() else None
    panel = load_validated_fars(direct_only=direct_only, flows=flows)
    pop_path = DATA_PROC / "county_population.parquet"
    if not pop_path.is_file():
        raise FileNotFoundError(f"county population panel not found: {pop_path}")
    population = pd.read_parquet(pop_path).loc[:, ["fips", "year", "population"]].copy()
    population["fips"] = population["fips"].astype(str).str.zfill(5)
    panel = panel.merge(population, on=["fips", "year"], how="inner")
    if panel.empty:
        raise ValueError("validated FARS panel has no rows with positive population")
    panel = panel.loc[panel["population"].gt(0)].copy()
    panel["fatals_per_100k"] = 100_000 * panel["fatals"] / panel["population"]

    alerts = base.load_verified_night_alerts()
    panel = panel.merge(
        alerts[["fips", "effective_crash_date", "night_alert"]],
        left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
    ).drop(columns=["effective_crash_date"], errors="ignore")
    panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)
    if direct_only:
        panel["spillover_commuters"] = 0.0
        panel["spillover_share"] = 0.0
    else:
        assert flows is not None
        spill = build_commuter_spillover(alerts, flows)
        panel = panel.merge(
            spill, left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
        ).drop(columns=["effective_crash_date"], errors="ignore")
    panel = add_spillover_classes(panel)
    # The shared share-based estimators require this explicit 10 pp scale.
    # Keep it in the national runner rather than assuming the state runner's
    # wrapper has already been applied.
    panel["spillover_share_10pp"] = panel["spillover_share"].fillna(0.0).clip(0.0, 1.0) / 0.10
    return panel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args(argv)
    panel = build_panel(direct_only=args.direct_only)
    rows: list[dict] = []
    rows.extend(share.run_wls(panel, "fatals_per_100k", "FARS_NATIONAL", direct_only=args.direct_only))
    rows.extend(share.run_ppml(panel, "fatals", "FARS_NATIONAL", direct_only=args.direct_only))
    rows.extend(share.run_wls(panel, "fatals_per_100k", "FARS_NATIONAL", clean_controls=True))
    rows.extend(share.run_ppml(panel, "fatals", "FARS_NATIONAL", clean_controls=True))
    all_rows = pd.DataFrame(rows)
    estimates = all_rows.loc[all_rows["record_type"].eq("estimate")].copy()
    statuses = all_rows.loc[all_rows["record_type"].eq("fit_status")].copy()
    statuses = pd.concat([statuses, pd.DataFrame([{
        "record_type": "model_count_summary", **summarize_fit_statuses(statuses.to_dict("records")),
    }])], ignore_index=True)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUTPUT_TABS / "fars_validated_analysis_share.csv", index=False)
    statuses.to_csv(OUTPUT_TABS / "fars_validated_analysis_share_status.csv", index=False)


if __name__ == "__main__":
    main()
