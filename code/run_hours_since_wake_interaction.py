"""Estimate a pooled time-since-waking interaction for alert exposure.

The existing dose-response runner estimates one county-day regression per
outcome hour and then meta-regresses the 18 coefficients.  This runner makes
the interaction explicit in one stacked hourly panel.  ``own_alert`` follows
the effective-date rule used throughout the project: an alert at 22:00--23:59
on ``t-1`` or 00:00--05:59 on ``t`` sets exposure for date ``t``.

The sample is a compact time-stratified case-crossover panel.  For each
active county-date with an own alert, it keeps that date and the same-weekday
referents at +/-7, +/-14, +/-21, and +/-28 days, then stacks outcome hours
06:00--24:00 (the hourly bins 06:00--07:00 through 23:00--24:00).  This
avoids materialising the full 18-hour national panel.

The primary specification is:

    person_fatals ~ own_alert + own_alert:hours_since_wake
                    + cross_spillover + cross_spillover:hours_since_wake
                    + other_wea_night_alert
                    | fips_hour_dow + fips_year + year_month

The interaction coefficient is the change in the alert contrast per
additional hour since the nominal 06:00 wake time.  The spillover interaction
is included so the own-alert interaction is not forced to absorb the separate
fatigue mechanism.

Output: ``output/tables/reg_hours_since_wake_interaction.csv``
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

log = get_logger("hours_since_wake_interaction")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WAKE_HOUR = 6
END_HOUR = 24  # 24:00 is the exclusive end; hour=23 covers 23:00--24:00.
OUT_PATH = OUTPUT_TABS / "reg_hours_since_wake_interaction.csv"
REFERENT_OFFSETS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)
HOURS = tuple(range(WAKE_HOUR, END_HOUR))
CONTROL_SPECS = ntm.OTHER_WEA_CONTROL_SPECS
INTERACTION_TERMS = (
    "own_alert",
    "own_alert_x_hours_since_wake",
    "cross_spillover",
    "cross_spillover_x_hours_since_wake",
)
CONTROL_TERMS = CONTROL_SPECS[0][1]


def add_interaction_terms(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the two treatment-by-hours-since-wake interaction columns."""
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


def _referent_days(events: pd.DataFrame, lower: pd.Timestamp,
                   upper: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for offset in REFERENT_OFFSETS:
        part = events[["fips", "date"]].copy()
        part["date"] = part["date"] + pd.Timedelta(days=offset)
        frames.append(part)
    referents = pd.concat(frames, ignore_index=True)
    referents = referents[referents["date"].between(lower, upper)]
    # Offsets are multiples of seven, so every referent has the same weekday
    # as its event. Deduplication prevents dates serving many events from
    # receiving artificial multiplicity in the pooled regression.
    return referents.drop_duplicates(["fips", "date"]).reset_index(drop=True)


def build_hourly_interaction_panel() -> pd.DataFrame:
    """Build the compact stacked county-date-hour interaction sample."""
    daily = ntm.build_outcome_grid()
    daily = ntm.attach_night_alert(daily)
    daily = ntm.attach_cross_spillover(daily)
    daily = ntm.attach_other_wea_control(daily)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["fips"] = daily["fips"].astype(str).str.zfill(5)

    events = daily.loc[daily["night_alert"].eq(1), ["fips", "date"]].drop_duplicates()
    if events.empty:
        raise ValueError("no own-alert county-dates available for interaction sample")
    lower, upper = daily["date"].min(), daily["date"].max()
    sample_days = _referent_days(events, lower, upper)
    sample_days = sample_days.merge(
        daily[["fips", "date", "night_alert", "cross_spillover",
               "other_wea_night_alert", "other_wea_night_count"]],
        on=["fips", "date"], how="left", validate="one_to_one",
    )
    sample_days["night_alert"] = sample_days["night_alert"].fillna(0).astype(int)
    sample_days["cross_spillover"] = sample_days["cross_spillover"].fillna(0.0)
    sample_days["other_wea_night_alert"] = sample_days["other_wea_night_alert"].fillna(0).astype(int)
    sample_days["other_wea_night_count"] = sample_days["other_wea_night_count"].fillna(0).astype(int)
    sample_days = sample_days.rename(columns={"night_alert": "own_alert"})

    hours = pd.DataFrame({"hour": HOURS})
    sample_days["_key"] = 1
    hours["_key"] = 1
    panel = sample_days.merge(hours, on="_key", how="inner").drop(columns="_key")

    hourly_path = DATA_PROC / "fars_hourly_county_day.parquet"
    hourly = pd.read_parquet(
        hourly_path,
        columns=["fips", "date", "hour", "person_fatals", "serious_inj"],
    )
    hourly["fips"] = hourly["fips"].astype(str).str.zfill(5)
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.normalize()
    hourly = hourly[hourly["hour"].isin(HOURS)]
    hourly = hourly[hourly["fips"].isin(set(sample_days["fips"]))]
    panel = panel.merge(
        hourly, on=["fips", "date", "hour"], how="left", validate="one_to_one",
    )
    for outcome in ("person_fatals", "serious_inj"):
        panel[outcome] = panel[outcome].fillna(0.0).astype(float)

    panel["dow"] = panel["date"].dt.dayofweek.astype(int)
    panel["year"] = panel["date"].dt.year.astype(int)
    panel["month"] = panel["date"].dt.month.astype(int)
    panel["fips_hour_dow"] = (
        panel["fips"] + "_" + panel["hour"].astype(str) + "_" + panel["dow"].astype(str)
    )
    panel["fips_year"] = panel["fips"] + "_" + panel["year"].astype(str)
    panel["year_month"] = panel["year"].astype(str) + "_" + panel["month"].astype(str)
    panel["state_code"] = panel["fips"].str[:2]
    panel["date_str"] = panel["date"].dt.strftime("%Y-%m-%d")
    panel = add_interaction_terms(panel)
    log.info(
        "Interaction panel: %s rows, %s county-days, %s own-alert days, %s hours",
        f"{len(panel):,}", f"{len(sample_days):,}", f"{len(events):,}", len(HOURS),
    )
    del daily, hourly, sample_days, events
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
    fit = pf.feols(formula, data=sub, vcov={"CRV1": "state_code + date_str"}, lean=True)
    tidy = fit.tidy()
    rows = []
    for term in INTERACTION_TERMS:
        if term not in tidy.index:
            continue
        row = tidy.loc[term]
        rows.append({
            "record_type": "estimate",
            "status": "ok",
            "model": "OLS_TWFE_hourly_interaction",
            "sample": "night_alert_case_crossover",
            "specification": label,
            "control": control_label,
            "outcome": "person_fatals",
            "term": term,
            "beta": float(row["Estimate"]),
            "se": float(row["Std. Error"]),
            "pvalue": float(row["Pr(>|t|)"]),
            "n_obs": int(fit._N),
            "n_counties": int(sub["fips"].nunique()),
            "n_county_days": int(sub[["fips", "date"]].drop_duplicates().shape[0]),
            "fe": fe,
            "cluster": "state+date",
        })
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
