"""Test sensitivity to shifting the crash-outcome window.

The alert exposure is held fixed at anchor date ``t`` using the established
night-alert effective-date rule.  This runner reports an explicit 5 x 5
table of window endpoints: starts 04:00, 05:00, 06:00, 07:00, and 08:00
crossed with ends 22:00, 23:00, 24:00, 01:00, and 02:00.  This includes
the requested 04:00--23:00 and 04:00--24:00 cells.

Hours since wake is still measured from 06:00, rather than from the shifted
window start.  Thus the interaction coefficient remains a common
time-awake contrast while the outcome window moves.  The 00:00--06:00 next
morning portion is allowed to cross the calendar date, but exposure remains
the anchor-date exposure rather than being re-labeled using ``t+1`` alerts.

Output: ``output/tables/reg_window_shift_sensitivity.csv``
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

log = get_logger("window_shift_sensitivity")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WAKE_HOUR = 6
WINDOW_STARTS = (4, 5, 6, 7, 8)
WINDOW_ENDS = (22, 23, 24, 1, 2)
WINDOW_SPECS = tuple((start, end) for start in WINDOW_STARTS for end in WINDOW_ENDS)
OUT_PATH = OUTPUT_TABS / "reg_window_shift_sensitivity.csv"
REFERENT_OFFSETS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)
INTERACTION_TERMS = (
    "own_alert",
    "own_alert_x_hours_since_wake",
    "cross_spillover",
    "cross_spillover_x_hours_since_wake",
)


def window_duration(start_hour: int, end_hour: int) -> int:
    """Return half-open duration for a clock endpoint, allowing midnight wrap."""
    if not 0 <= start_hour <= 23:
        raise ValueError("start_hour must be between 0 and 23")
    if not 0 <= end_hour <= 24:
        raise ValueError("end_hour must be between 0 and 24")
    end_absolute = end_hour
    if end_absolute <= start_hour:
        end_absolute += 24
    duration = end_absolute - start_hour
    if not 1 <= duration <= 24:
        raise ValueError("window duration must be between 1 and 24 hours")
    return duration


def relative_window(start_hour: int, end_hour: int) -> pd.DataFrame:
    """Map elapsed hours to clock hour/date offset from anchor date ``t``."""
    duration_hours = window_duration(start_hour, end_hour)
    elapsed = np.arange(duration_hours, dtype=int)
    absolute = start_hour + elapsed
    return pd.DataFrame(
        {
            "elapsed_hour": elapsed,
            "hours_since_wake": absolute - WAKE_HOUR,
            "outcome_day_offset": absolute // 24,
            "hour": absolute % 24,
        }
    )


def window_label(start_hour: int, end_hour: int) -> str:
    """Return a human-readable half-open clock window label."""
    end_absolute = start_hour + window_duration(start_hour, end_hour)
    end_text = "24:00" if end_hour == 24 else f"{end_hour:02d}:00"
    day_suffix = " (+1 day)" if end_absolute > 24 else ""
    return f"{start_hour:02d}:00-{end_text}{day_suffix}"


def add_interaction_terms(panel: pd.DataFrame) -> pd.DataFrame:
    """Add own- and cross-exposure interactions with hours since wake."""
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


def build_sample_days() -> tuple[pd.DataFrame, int]:
    """Build anchor county-days and fixed own/spillover exposure labels."""
    daily = ntm.build_outcome_grid()
    daily = ntm.attach_night_alert(daily)
    daily = ntm.attach_cross_spillover(daily)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["fips"] = daily["fips"].astype(str).str.zfill(5)

    lower = daily["date"].min()
    # Every shifted window in this sensitivity grid can reach t+1, so use a
    # common anchor boundary for all windows and keep the sample comparable.
    anchor_upper = daily["date"].max() - pd.Timedelta(days=1)
    events = daily.loc[
        daily["night_alert"].eq(1)
        & daily["date"].between(lower, anchor_upper),
        ["fips", "date"],
    ].drop_duplicates()
    if events.empty:
        raise ValueError("no own-alert county-dates available for sensitivity sample")

    sample_days = _referent_days(events, lower, anchor_upper)
    sample_days = sample_days.merge(
        daily[["fips", "date", "night_alert", "cross_spillover"]],
        on=["fips", "date"],
        how="left",
        validate="one_to_one",
    )
    sample_days["night_alert"] = sample_days["night_alert"].fillna(0).astype(int)
    sample_days["cross_spillover"] = sample_days["cross_spillover"].fillna(0.0)
    sample_days = sample_days.rename(columns={"night_alert": "own_alert"})
    log.info(
        "Sensitivity anchors: %s county-days, %s own-alert anchors",
        f"{len(sample_days):,}",
        f"{len(events):,}",
    )
    del daily, events
    gc.collect()
    return sample_days, int(sample_days["fips"].nunique())


def load_hourly(sample_days: pd.DataFrame) -> pd.DataFrame:
    """Load sparse hourly outcomes for counties represented in the sample."""
    hourly = pd.read_parquet(
        DATA_PROC / "fars_hourly_county_day.parquet",
        columns=["fips", "date", "hour", "person_fatals"],
    )
    hourly["fips"] = hourly["fips"].astype(str).str.zfill(5)
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.normalize()
    hourly = hourly[hourly["fips"].isin(set(sample_days["fips"]))]
    return hourly.rename(columns={"date": "outcome_date"})


def build_panel(
    sample_days: pd.DataFrame,
    hourly: pd.DataFrame,
    start_hour: int,
    end_hour: int,
) -> pd.DataFrame:
    """Build one shifted window's county-anchor x elapsed-hour panel."""
    window = relative_window(start_hour, end_hour)
    anchors = sample_days.copy()
    anchors["_key"] = 1
    window["_key"] = 1
    panel = anchors.merge(window, on="_key", how="inner").drop(columns="_key")
    panel["outcome_date"] = panel["date"] + pd.to_timedelta(
        panel["outcome_day_offset"], unit="D"
    )
    panel = panel.merge(
        hourly,
        on=["fips", "outcome_date", "hour"],
        how="left",
        validate="one_to_one",
    )
    panel["person_fatals"] = panel["person_fatals"].fillna(0.0).astype(float)
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
    return add_interaction_terms(panel)


def run_ols(
    panel: pd.DataFrame,
    fe: str,
    specification: str,
    start_hour: int,
    end_hour: int,
    n_counties: int,
) -> list[dict[str, object]]:
    """Fit the interaction model for one shifted outcome window."""
    import pyfixest as pf

    rhs = " + ".join(INTERACTION_TERMS)
    formula = f"person_fatals ~ {rhs} | {fe}"
    sub = panel.dropna(subset=["person_fatals", *INTERACTION_TERMS]).copy()
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
                "model": "OLS_TWFE_hourly_window_shift",
                "sample": "night_alert_case_crossover_shifted_window",
                "window": window_label(start_hour, end_hour),
                "window_start_hour": start_hour,
                "window_end_hour": end_hour,
                "window_duration_hours": window_duration(start_hour, end_hour),
                "specification": specification,
                "outcome": "person_fatals",
                "term": term,
                "beta": float(row["Estimate"]),
                "se": float(row["Std. Error"]),
                "pvalue": float(row["Pr(>|t|)"]),
                "n_obs": int(fit._N),
                "n_counties": n_counties,
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
    sample_days, n_counties = build_sample_days()
    hourly = load_hourly(sample_days)
    rows: list[dict[str, object]] = []
    for start_hour, end_hour in WINDOW_SPECS:
        panel = build_panel(sample_days, hourly, start_hour, end_hour)
        rows.extend(
            run_ols(
                panel,
                "fips_hour_dow + year_month",
                "baseline",
                start_hour,
                end_hour,
                n_counties,
            )
        )
        rows.extend(
            run_ols(
                panel,
                "fips_hour_dow + fips_year + year_month",
                "robust",
                start_hour,
                end_hour,
                n_counties,
            )
        )
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        log.info(
            "Saved checkpoint for %s -> %s",
            window_label(start_hour, end_hour),
            OUT_PATH,
        )
        del panel
        gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    log.info("Saved -> %s", OUT_PATH)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
