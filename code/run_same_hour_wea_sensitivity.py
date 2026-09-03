"""Exact-hour WEA-control sensitivity for the matched alert-hour study.

The overnight WEA control is intentionally not reused here: exact-hour
identification requires controls at the same local county/date/hour.  This
runner reports the rich fixed-effect same-hour estimates after adding both a
direct non-AMBER WEA indicator and its commuter-weighted spillover.  It runs
the all-WEA and CAP-``Met``-excluded versions, plus an overlap-excluded check
for county-hours containing both an AMBER and another WEA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
import run_same_hour_event_study as base
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

log = get_logger("same_hour_wea_sensitivity")

CONTROL_PATH = DATA_PROC / "other_wea_hour_controls.parquet"
OUTPUT_PATH = OUTPUT_TABS / "reg_same_hour_wea_controls.csv"
CONTROL_VARIANTS = {
    "all_non_amber_wea": "all_wea_hour",
    "non_weather_wea": "non_weather_wea_hour",
}


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["hour"] = pd.to_numeric(out["hour"], errors="raise").astype(int)
    return out


def build_hour_spillover(
    controls: pd.DataFrame,
    alert_col: str,
    *,
    weights: pd.DataFrame,
    out_col: str,
) -> pd.DataFrame:
    """Construct same-hour commuter spillover from a control alert column."""
    controls = _normalise_keys(controls)
    weights = weights.copy()
    weights["fips_home"] = pd.to_numeric(weights["fips_home"], errors="raise").astype(int)
    weights["fips_work"] = pd.to_numeric(weights["fips_work"], errors="raise").astype(int)
    events = controls.loc[controls[alert_col].gt(0), ["fips", "date", "hour"]].copy()
    if events.empty:
        return pd.DataFrame(columns=["fips", "date", "hour", out_col])
    events["fips_home"] = events["fips"].astype(int)
    pairs = events.merge(weights, on="fips_home", how="inner")
    pairs = pairs[pairs["fips_home"] != pairs["fips_work"]].copy()
    if pairs.empty:
        return pd.DataFrame(columns=["fips", "date", "hour", out_col])
    pairs["fips"] = pairs["fips_work"].astype(str).str.zfill(5)
    return (
        pairs.groupby(["fips", "date", "hour"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": out_col})
    )


def attach_hour_wea_controls(
    panel: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Merge direct same-hour controls and commuter spillovers onto a panel."""
    out = _normalise_keys(panel)
    controls = _normalise_keys(controls)
    direct_cols = [
        "fips", "date", "hour",
        "all_wea_hour_alert", "all_wea_hour_count",
        "non_weather_wea_hour_alert", "non_weather_wea_hour_count",
    ]
    missing = set(direct_cols) - set(controls.columns)
    if missing:
        raise ValueError(f"hour WEA control missing columns: {sorted(missing)}")
    controls = controls[direct_cols].drop_duplicates(["fips", "date", "hour"], keep="last")
    out = out.merge(controls, on=["fips", "date", "hour"], how="left", validate="one_to_one")
    for col in direct_cols[3:]:
        out[col] = out[col].fillna(0)
        if col.endswith("_alert") or col.endswith("_count"):
            out[col] = out[col].astype(int)
    for variant, prefix in CONTROL_VARIANTS.items():
        alert_col = f"{prefix}_alert"
        spill_col = f"{prefix}_spillover"
        spill = build_hour_spillover(controls, alert_col, weights=weights, out_col=spill_col)
        out = out.merge(spill, on=["fips", "date", "hour"], how="left", validate="one_to_one")
        out[spill_col] = out[spill_col].fillna(0.0)
    return out


def _add_fe_columns(grid: pd.DataFrame, day_match: str) -> pd.DataFrame:
    out = grid.copy()
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["hour"] = out["hour"].astype(int)
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["year_month"] = out["year"].astype(str) + "_" + out["month"].astype(str)
    out["fips_year"] = out["fips"] + "_" + out["year"].astype(str)
    out["state_code"] = out["fips"].str[:2]
    out["date_str"] = out["date"].dt.strftime("%Y-%m-%d")
    if day_match == "dow":
        out["fips_hour_dow"] = (
            out["fips"] + "_" + out["hour"].astype(str) + "_" + out["dow"].astype(str)
        )
        out["fe"] = "fips_hour_dow + fips_year + year_month"
    else:
        out["weekend"] = (out["dow"] >= 5).astype(int)
        out["fips_hour_weekend"] = (
            out["fips"] + "_" + out["hour"].astype(str) + "_" + out["weekend"].astype(str)
        )
        out["fe"] = "fips_hour_weekend + fips_year + year_month"
    return out


def _run_variant(
    grid: pd.DataFrame,
    *,
    source: str,
    overlap_policy: str,
    results: list[dict],
) -> None:
    prefix = CONTROL_VARIANTS[source]
    control_terms = [f"{prefix}_alert", f"{prefix}_spillover"]
    direct = control_terms[0]
    sample = grid
    if overlap_policy == "overlap_excluded":
        sample = grid.loc[~((grid["is_alert_hour"] == 1) & (grid[direct] == 1))].copy()
    fe = str(grid["fe"].iloc[0])
    specs = [
        ("fatals", "person_fatals", "is_alert_hour", "any alert hour", []),
        ("serious", "serious_inj", "is_alert_hour", "any alert hour", []),
        ("fatals", "person_fatals", "is_first_alert_hour", "first-alert hour", []),
        ("serious", "serious_inj", "is_first_alert_hour", "first-alert hour", []),
        ("fatals", "person_fatals", "is_alert_hour_tomorrow", "tomorrow placebo",
         ["is_alert_hour"]),
    ]
    for outcome_label, outcome, treatment, label, extras in specs:
        before = len(results)
        base.run(
            sample,
            f"{outcome_label}: {label}; {source}; {overlap_policy}",
            outcome,
            treatment,
            fe,
            "ols",
            results,
            extra_controls=[*extras, *control_terms],
        )
        for row in results[before:]:
            row["control_source"] = source
            row["overlap_policy"] = overlap_policy
            row["control_spec"] = "direct+spillover"
            row["day_match"] = "weekend" if "weekend" in fe else "dow"


def run_sensitivity(control_path: Path = CONTROL_PATH) -> pd.DataFrame:
    """Run the two exact-hour control variants and overlap check."""
    controls = pd.read_parquet(control_path)
    weights = pd.read_parquet(ntm.COMMUTING_WEIGHTS_PATH)
    active = base.active_counties()
    events = base.load_any_time_alert_hours(active)
    results: list[dict] = []
    for day_match in ("dow", "weekend"):
        grid = base.build_matched_referent_grid(events, day_match=day_match)
        grid = attach_hour_wea_controls(grid, controls, weights=weights)
        grid = _add_fe_columns(grid, day_match)
        for source in CONTROL_VARIANTS:
            _run_variant(grid, source=source, overlap_policy="all_hours", results=results)
            _run_variant(grid, source=source, overlap_policy="overlap_excluded", results=results)
    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-path", type=Path, default=CONTROL_PATH)
    args = parser.parse_args(argv)
    if not args.control_path.is_file():
        raise FileNotFoundError(
            f"hour WEA control not found at {args.control_path}; "
            "run code/build_other_wea_hour_controls.py first"
        )
    out = run_sensitivity(args.control_path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved exact-hour WEA sensitivity -> %s (%d rows)", OUTPUT_PATH, len(out))
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
