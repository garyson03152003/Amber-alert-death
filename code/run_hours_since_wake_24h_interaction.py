"""Estimate a 24-hour wake-to-wake alert interaction.

For an alert-day anchor ``t``, this specification uses the complete waking
cycle from 06:00 on ``t`` through 06:00 on ``t+1``: hours 06--23 on ``t``
and hours 00--05 on ``t+1``.  The alert exposure remains attached to the
anchor date, so the second calendar date does not silently become a new
treatment day.

``own_alert`` follows the project's effective-date rule: an alert in the
same county at 22:00--23:59 on ``t-1`` or 00:00--05:59 on ``t`` sets exposure
for anchor date ``t``.

The sample is a compact time-stratified case-crossover panel.  For each
active county-date with an own alert, it keeps that date and same-weekday
referents at +/-7, +/-14, +/-21, and +/-28 days, then stacks 24 hourly
outcomes.  The interaction coefficient is the change in the alert contrast
per additional hour since the nominal 06:00 wake time.

Output: ``output/tables/reg_hours_since_wake_24h_interaction.csv``
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_night_to_morning_window as ntm
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger

log = get_logger("hours_since_wake_24h_interaction")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WAKE_HOUR = 6
N_HOURS = 24
OUT_PATH = OUTPUT_TABS / "reg_hours_since_wake_24h_interaction.csv"
REFERENT_OFFSETS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)
CONTROL_SPECS = ntm.OTHER_WEA_CONTROL_SPECS
INTERACTION_TERMS = (
    "own_alert",
    "own_alert_x_hours_since_wake",
    "cross_spillover",
    "cross_spillover_x_hours_since_wake",
)
CONTROL_TERMS = CONTROL_SPECS[0][1]


def relative_hour_window() -> pd.DataFrame:
    """Return the 24 elapsed-hour bins from 06:00 ``t`` to 06:00 ``t+1``."""
    elapsed = np.arange(N_HOURS, dtype=int)
    clock = (WAKE_HOUR + elapsed) % 24
    day_offset = (WAKE_HOUR + elapsed) // 24
    return pd.DataFrame(
        {
            "hours_since_wake": elapsed,
            "outcome_day_offset": day_offset,
            "hour": clock,
        }
    )


def add_interaction_terms(panel: pd.DataFrame) -> pd.DataFrame:
    """Add own- and cross-exposure interactions with elapsed wake hours."""
    out = panel.copy()
    if "hours_since_wake" not in out.columns:
        if "hour" not in out.columns:
            raise ValueError("panel must contain hour or hours_since_wake")
        out["hours_since_wake"] = (
            pd.to_numeric(out["hour"], errors="raise").astype(int) - WAKE_HOUR
        )
    missing = [c for c in ("own_alert", "cross_spillover") if c not in out.columns]
    if missing:
        raise ValueError(f"panel missing exposure columns: {missing}")
    out["own_alert"] = pd.to_numeric(out["own_alert"], errors="raise").astype(float)
    out["cross_spillover"] = pd.to_numeric(
        out["cross_spillover"], errors="raise"
    ).astype(float)
    out["own_alert_x_hours_since_wake"] = (
        out["own_alert"] * out["hours_since_wake"]
    )
    out["cross_spillover_x_hours_since_wake"] = (
        out["cross_spillover"] * out["hours_since_wake"]
    )
    return out


def _referent_days(
    events: pd.DataFrame, lower: pd.Timestamp, upper: pd.Timestamp
) -> pd.DataFrame:
    frames = []
    for offset in REFERENT_OFFSETS:
        part = events[["fips", "date"]].copy()
        part["date"] = part["date"] + pd.Timedelta(days=offset)
        frames.append(part)
    referents = pd.concat(frames, ignore_index=True)
    referents = referents[referents["date"].between(lower, upper)]
    return referents.drop_duplicates(["fips", "date"]).reset_index(drop=True)


def build_hourly_interaction_panel() -> pd.DataFrame:
    """Build the compact county-anchor-date x 24-hour interaction sample."""
    daily = ntm.build_outcome_grid()
    daily = ntm.attach_night_alert(daily)
    daily = ntm.attach_cross_spillover(daily)
    daily = ntm.attach_other_wea_control(daily)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["fips"] = daily["fips"].astype(str).str.zfill(5)

    # The final anchor date is dropped because its t+1 early-morning hours
    # fall outside the available balanced daily grid.
    lower = daily["date"].min()
    anchor_upper = daily["date"].max() - pd.Timedelta(days=1)
    events = daily.loc[
        daily["night_alert"].eq(1)
        & daily["date"].between(lower, anchor_upper),
        ["fips", "date"],
    ].drop_duplicates()
    if events.empty:
        raise ValueError("no own-alert county-dates available for interaction sample")

    sample_days = _referent_days(events, lower, anchor_upper)
    sample_days = sample_days.merge(
        daily[["fips", "date", "night_alert", "cross_spillover",
               "other_wea_night_alert", "other_wea_night_count"]],
        on=["fips", "date"],
        how="left",
        validate="one_to_one",
    )
    sample_days["night_alert"] = sample_days["night_alert"].fillna(0).astype(int)
    sample_days["cross_spillover"] = sample_days["cross_spillover"].fillna(0.0)
    sample_days["other_wea_night_alert"] = sample_days["other_wea_night_alert"].fillna(0).astype(int)
    sample_days["other_wea_night_count"] = sample_days["other_wea_night_count"].fillna(0).astype(int)
    sample_days = sample_days.rename(columns={"night_alert": "own_alert"})

    window = relative_hour_window()
    sample_days["_key"] = 1
    window["_key"] = 1
    panel = sample_days.merge(window, on="_key", how="inner").drop(columns="_key")
    panel["outcome_date"] = panel["date"] + pd.to_timedelta(
        panel["outcome_day_offset"], unit="D"
    )

    hourly = pd.read_parquet(
        DATA_PROC / "fars_hourly_county_day.parquet",
        columns=["fips", "date", "hour", "person_fatals", "serious_inj"],
    )
    hourly["fips"] = hourly["fips"].astype(str).str.zfill(5)
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.normalize()
    hourly = hourly[hourly["fips"].isin(set(sample_days["fips"]))]
    hourly = hourly.rename(columns={"date": "outcome_date"})
    panel = panel.merge(
        hourly,
        on=["fips", "outcome_date", "hour"],
        how="left",
        validate="one_to_one",
    )
    for outcome in ("person_fatals", "serious_inj"):
        panel[outcome] = panel[outcome].fillna(0.0).astype(float)

    panel["outcome_dow"] = panel["outcome_date"].dt.dayofweek.astype(int)
    panel["outcome_year"] = panel["outcome_date"].dt.year.astype(int)
    panel["outcome_month"] = panel["outcome_date"].dt.month.astype(int)
    panel["fips_hour_dow"] = (
        panel["fips"]
        + "_"
        + panel["hour"].astype(str)
        + "_"
        + panel["outcome_dow"].astype(str)
    )
    panel["fips_year"] = panel["fips"] + "_" + panel["outcome_year"].astype(str)
    panel["year_month"] = (
        panel["outcome_year"].astype(str) + "_" + panel["outcome_month"].astype(str)
    )
    panel["state_code"] = panel["fips"].str[:2]
    panel["outcome_date_str"] = panel["outcome_date"].dt.strftime("%Y-%m-%d")
    panel = add_interaction_terms(panel)
    log.info(
        "24-hour interaction panel: %s rows, %s anchor county-days, %s own-alert anchors",
        f"{len(panel):,}",
        f"{len(sample_days):,}",
        f"{len(events):,}",
    )
    del daily, hourly, sample_days, events, window
    gc.collect()
    return panel


def run_ols(panel: pd.DataFrame, fe: str, label: str,
            control_terms: tuple[str, ...] = CONTROL_TERMS,
            control_label: str = "binary") -> list[dict[str, object]]:
    """Fit the joint own/spillover interaction model with two-way clustering."""
    import pyfixest as pf

    rhs = " + ".join((*INTERACTION_TERMS, *control_terms))
    formula = f"person_fatals ~ {rhs} | {fe}"
    sub = panel.dropna(subset=["person_fatals", *INTERACTION_TERMS, *control_terms]).copy()
    fit = pf.feols(
        formula,
        data=sub,
        vcov={"CRV1": "state_code + outcome_date_str"},
        lean=True,
    )
    tidy = fit.tidy()
    rows = []
    for term in INTERACTION_TERMS:
        if term not in tidy.index:
            continue
        row = tidy.loc[term]
        rows.append(
            {
                "record_type": "estimate",
                "status": "ok",
                "model": "OLS_TWFE_hourly_24h_interaction",
                "sample": "night_alert_case_crossover_24h",
                "specification": label,
                "control": control_label,
                "outcome": "person_fatals",
                "term": term,
                "beta": float(row["Estimate"]),
                "se": float(row["Std. Error"]),
                "pvalue": float(row["Pr(>|t|)"]),
                "n_obs": int(fit._N),
                "n_counties": int(sub["fips"].nunique()),
                "n_anchor_days": int(
                    sub[["fips", "date"]].drop_duplicates().shape[0]
                ),
                "fe": fe,
                "cluster": "state+outcome_date",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    panel = build_hourly_interaction_panel()
    rows = []
    for control_label, control_terms in CONTROL_SPECS:
        rows.extend(run_ols(panel, "fips_hour_dow + year_month", "baseline",
                            control_terms, control_label))
        rows.extend(run_ols(panel, "fips_hour_dow + fips_year + year_month", "robust",
                            control_terms, control_label))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    log.info("Saved -> %s", OUT_PATH)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
