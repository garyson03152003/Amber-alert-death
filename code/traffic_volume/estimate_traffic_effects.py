"""Station-day TWFE estimates of the AMBER-alert effect on traffic volume
(Task 4, station-day model, of TRAFFIC_VOLUME_INSTRUCTIONS.md).

log(volume) ~ night_alert_ct [+ spillover_share_10pp] | station_id + date

Direct county exposure (night_alert_ct) is the main treatment; commuter
spillover exposure is estimated jointly but reported as a separate
coefficient -- never folded into a single "exposed" indicator, per the
instructions.

Fixed effects: station + calendar date (mirrors the crash-panel's
county + calendar-date design, with station standing in for county).
Clustering: two-way county + calendar-date CRV1, matching the crash panel's
inference approach and reflecting that one alert treats many stations/counties.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger
from state_dot_analysis_core import extract_finite_coefficients, fit_status_row

log = get_logger("estimate_traffic_effects")

ROOT = Path(__file__).resolve().parent.parent.parent
DAY_OUTCOMES_PATH = ROOT / "data" / "processed" / "traffic" / "tmas_station_day_outcomes.parquet"
OUT_PATH = ROOT / "output" / "tables" / "traffic_volume_main.csv"

OUTCOME_COLS = ["total_volume", "vol_05_10", "vol_07_10", "vol_10_16", "vol_16_19"]


def _prep(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    # county_fips is required for the CRV1 cluster variable; a handful of
    # stations lack a station->county match in some years' station-metadata
    # extract and must be dropped rather than fed to the clustered fit as NaN.
    sub = panel.dropna(subset=[outcome, "night_alert_ct", "county_fips"]).copy()
    sub = sub[sub[outcome] > 0].copy()  # log(volume) requires positive volume
    sub["log_outcome"] = np.log(sub[outcome].astype(float))
    sub["spillover_share_10pp"] = sub["spillover_share_ct"].fillna(0.0).clip(0.0, 1.0) / 0.10
    sub["_station_str"] = sub["state_fips"].astype(str) + "_" + sub["station_id"].astype(str)
    sub["_county_str"] = sub["county_fips"].astype(str)
    sub["_date_str"] = pd.to_datetime(sub["date"]).astype(str)
    return sub


def run_station_day_model(panel: pd.DataFrame, outcome: str) -> list[dict]:
    sub = _prep(panel, outcome)
    treatments = ("night_alert_ct", "spillover_share_10pp") if sub["spillover_share_10pp"].gt(0).any() else ("night_alert_ct",)
    zero_share = float((panel[outcome].fillna(0) <= 0).mean()) if len(panel) else None

    if len(sub) < 100 or sub["night_alert_ct"].nunique() < 2 or sub["_station_str"].nunique() < 2:
        return [fit_status_row(
            status="skipped", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason="insufficient_estimable_sample",
        ) | {"outcome": outcome, "model": "station_day_TWFE"}]

    formula = f"log_outcome ~ {' + '.join(treatments)} | _station_str + _date_str"
    try:
        fit = pf.feols(formula, data=sub, vcov={"CRV1": "_county_str + _date_str"})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("Skipping %s station_day_TWFE: %s", outcome, exc)
        return [fit_status_row(
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason=str(exc),
        ) | {"outcome": outcome, "model": "station_day_TWFE"}]

    coefficients, produced, errors = extract_finite_coefficients(fit, treatments)
    rows = []
    for coefficient in coefficients:
        b, se, p = coefficient["beta"], coefficient["se"], coefficient["pvalue"]
        rows.append({
            "record_type": "estimate", "status": "ok",
            "model": "station_day_TWFE", "outcome": outcome,
            "term": coefficient["term"], "beta": b, "se": se, "pvalue": p,
            "pct_change": float(100 * (np.exp(b) - 1)),
            "n_obs": int(fit._N), "cluster": "county+date",
        })
    status = "ok" if not errors else ("partial" if produced else "failed")
    error_reason = next(iter(set(errors.values()))) if len(set(errors.values())) == 1 else "; ".join(
        f"{term}:{reason}" for term, reason in errors.items()
    )
    rows.append(fit_status_row(
        status=status, input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=treatments, terms_produced=produced, error_reason=error_reason,
    ) | {"outcome": outcome, "model": "station_day_TWFE"})
    return rows


def main() -> None:
    if not DAY_OUTCOMES_PATH.is_file():
        raise FileNotFoundError(f"station-day outcomes not found at {DAY_OUTCOMES_PATH}")
    panel = pd.read_parquet(DAY_OUTCOMES_PATH)
    log.info("Loaded %s station-day rows", f"{len(panel):,}")
    log.info("Exposure classes: %s", panel["exposure_class"].value_counts().to_dict())

    rows = []
    for outcome in OUTCOME_COLS:
        rows.extend(run_station_day_model(panel, outcome))

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    log.info("Wrote %s -> %s", f"{len(out):,}", OUT_PATH)

    estimates = out[out["record_type"] == "estimate"]
    if len(estimates):
        log.info("\n%s", estimates[["outcome", "term", "beta", "se", "pvalue", "pct_change", "n_obs"]].to_string(index=False))
    else:
        log.warning("No estimable coefficients produced.")


if __name__ == "__main__":
    main()
