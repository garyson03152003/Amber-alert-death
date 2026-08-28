"""Matched-week hourly event study: compare each alert county-hour against
the SAME county, hour, and weekday in neighbouring non-alert weeks.

Why not the within-day design
-----------------------------
``run_hourly_alert_event_study.py`` uses adjacent hours of the same day as
the reference. Its pre-period placebo fails (leads jointly nonzero,
p=0.021 even after controlling hour-of-day x day-of-week), and the violation
sits at event hour -4 -- *before* issuance.

That is not obviously a bug. An AMBER alert is issued only after an
abduction is reported and verified, typically hours later, and many
qualifying abductions are vehicle-involved. Hours shortly before issuance
may therefore carry crashes belonging to the incident itself. If so, the
adjacent-hour reference is contaminated by the very thing being measured,
and no amount of extra fixed effects repairs it -- the control group is
wrong.

Design
------
For each alert event e (county c, local date d, alert hour h0) and each
offset k in the event window, the treated observation is

    (c, d, h0 + k)

and its controls are the identical clock slot on the same weekday in
neighbouring weeks:

    (c, d + 7w, h0 + k)   for w in +/-1..W, w != 0

Estimation is PPML with an event-by-offset fixed effect:

    crashes ~ sum_k treated * 1{offset = k}  |  event x offset

so each coefficient is a difference-in-differences: at offset k, is the
alert week elevated relative to the same slot in adjacent weeks? Crucially
the reference is *other weeks*, never other hours of the contaminated day,
so a pre-alert incident effect is measured rather than differenced into the
baseline.

Control weeks in which the same county had any alert inside the window are
dropped, so controls are genuinely untreated.

Interpretation
--------------
  * offset 0 = the alert hour itself
  * offset >0 = hours after issuance (the distraction hypothesis)
  * offset <0 = hours before issuance. These are no longer a placebo that
    must be flat: under the incident-contamination story they may be
    genuinely positive. What identifies a *distraction* effect is the
    post-issuance coefficients exceeding the pre-issuance ones.

Outputs
-------
output/tables/matched_hour_event_study.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from state_dot_analysis_core import extract_finite_coefficients, fit_status_row
from run_hourly_alert_event_study import (
    build_balanced_hourly_panel, CA_HOURLY, CA_YEARS, EVENT_MIN, EVENT_MAX,
)

log = base.log

CONTROL_WEEKS = 4        # +/- weeks used as matched controls
OUT_PATH = OUTPUT_TABS / "matched_hour_event_study.csv"


def build_matched_sample(
    panel: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    outcome: str,
    control_weeks: int = CONTROL_WEEKS,
    event_min: int = EVENT_MIN,
    event_max: int = EVENT_MAX,
) -> pd.DataFrame:
    """Stack each alert's event window with its matched neighbouring-week slots.

    Returns one row per (event, offset, week), with ``treated`` marking the
    alert week (week offset 0).
    """
    if alerts.empty:
        return pd.DataFrame()

    lookup = panel.set_index(["fips", "ts"])[outcome]

    events = alerts[["fips", "sent_local"]].copy()
    events["alert_ts"] = pd.to_datetime(events["sent_local"]).dt.floor("h")
    events = events[["fips", "alert_ts"]].drop_duplicates().reset_index(drop=True)
    events["event_id"] = events.index.astype(str)

    # Any alert hour in a county, used to disqualify contaminated control weeks.
    alert_hours_by_county: dict[str, set] = {
        f: set(g["alert_ts"]) for f, g in events.groupby("fips")
    }

    offsets = np.arange(event_min, event_max + 1)
    weeks = [w for w in range(-control_weeks, control_weeks + 1)]

    frames = []
    for w in weeks:
        blk = events.loc[events.index.repeat(len(offsets))].copy()
        blk["offset"] = np.tile(offsets, len(events))
        blk["week"] = w
        blk["ts"] = (
            blk["alert_ts"]
            + pd.to_timedelta(blk["offset"], unit="h")
            + pd.to_timedelta(7 * w, unit="D")
        )
        frames.append(blk)
    stacked = pd.concat(frames, ignore_index=True)

    stacked["treated"] = (stacked["week"] == 0).astype(int)

    # Drop control rows whose slot is itself inside another alert's window in
    # the same county -- those are treated, not controls.
    def _contaminated(row) -> bool:
        if row["week"] == 0:
            return False
        hours = alert_hours_by_county.get(row["fips"])
        if not hours:
            return False
        ts = row["ts"]
        return any(
            abs((ts - a) / np.timedelta64(1, "h")) <= max(abs(event_min), event_max)
            for a in hours
        )

    stacked["_bad"] = stacked.apply(_contaminated, axis=1)
    stacked = stacked[~stacked["_bad"]].drop(columns=["_bad"])

    stacked[outcome] = lookup.reindex(
        pd.MultiIndex.from_arrays([stacked["fips"], stacked["ts"]])
    ).to_numpy()
    stacked = stacked.dropna(subset=[outcome])

    # Keep only events that still have both a treated and a control week at
    # a given offset; otherwise the event x offset FE absorbs everything.
    stacked["stratum"] = stacked["event_id"] + "_" + stacked["offset"].astype(str)
    usable = (
        stacked.groupby("stratum")["treated"].agg(["nunique", "size"])
        .query("nunique == 2 and size >= 3").index
    )
    stacked = stacked[stacked["stratum"].isin(usable)].copy()

    stacked["date"] = stacked["ts"].dt.normalize()
    return stacked.reset_index(drop=True)


def add_offset_interactions(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """treated x offset dummies -- one DiD coefficient per event hour."""
    out = sample.copy()
    terms: list[str] = []
    for k in sorted(out["offset"].unique()):
        name = f"tr_{'m' if k < 0 else 'p'}{abs(int(k))}"
        out[name] = ((out["offset"] == k) & (out["treated"] == 1)).astype(int)
        if out[name].nunique() >= 2:
            terms.append(name)
    return out, terms


def run_matched_model(sample: pd.DataFrame, outcome: str, terms: list[str]) -> list[dict]:
    if sample.empty or not terms:
        return [fit_status_row(
            status="skipped", input_n=len(sample), fitted_n=0, zero_share=None,
            terms_requested=tuple(terms), error_reason="empty_matched_sample",
        ) | {"outcome": outcome, "model": "PPML_matched_week"}]

    sub = sample.copy()
    sub["_county_str"] = sub["fips"].astype(str)
    sub["_date_str"] = sub["date"].dt.strftime("%Y-%m-%d")
    zero_share = float((sub[outcome] == 0).mean())

    formula = f"{outcome} ~ {' + '.join(terms)} | stratum"
    log.info("[%s] matched sample %s rows; %.1f%% zero hours",
             outcome, f"{len(sub):,}", 100 * zero_share)
    try:
        fit = pf.fepois(formula, data=sub, vcov={"CRV1": "_county_str + _date_str"})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        log.warning("[%s] matched model failed: %s", outcome, exc)
        return [fit_status_row(
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=tuple(terms), error_reason=str(exc),
        ) | {"outcome": outcome, "model": "PPML_matched_week"}]

    coefficients, produced, errors = extract_finite_coefficients(fit, tuple(terms))
    rows = []
    for c in coefficients:
        raw = c["term"].split("_")[1]
        k = -int(raw[1:]) if raw.startswith("m") else int(raw[1:])
        rows.append({
            "record_type": "estimate", "status": "ok",
            "model": "PPML_matched_week", "outcome": outcome,
            "term": c["term"], "event_hour": k,
            "beta": c["beta"], "se": c["se"], "pvalue": c["pvalue"],
            "pct_change": float(100 * (np.exp(c["beta"]) - 1)),
            "ci_low_pct": float(100 * (np.exp(c["beta"] - 1.96 * c["se"]) - 1)),
            "ci_high_pct": float(100 * (np.exp(c["beta"] + 1.96 * c["se"]) - 1)),
            "n_obs": int(fit._N), "cluster": "county+date",
            "reference": "same county/hour/weekday, neighbouring weeks",
        })
    status = "ok" if not errors else ("partial" if produced else "failed")
    rows.append(fit_status_row(
        status=status, input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=tuple(terms), terms_produced=produced,
        error_reason="; ".join(f"{t}:{r}" for t, r in errors.items()) or None,
    ) | {"outcome": outcome, "model": "PPML_matched_week"})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="day", choices=["day", "night"])
    parser.add_argument("--outcome", default="ca_crashes")
    parser.add_argument("--control-weeks", type=int, default=CONTROL_WEEKS)
    args = parser.parse_args(argv)

    sparse = pd.read_parquet(CA_HOURLY)
    panel = build_balanced_hourly_panel(sparse, years=CA_YEARS)
    panel["ts"] = panel["date"] + pd.to_timedelta(panel["hour"], unit="h")
    log.info("Balanced CA panel: %s rows", f"{len(panel):,}")

    alerts = base.load_verified_alerts(window=args.window, detail=True)
    alerts = alerts[alerts["state_fips"] == "06"].copy()
    log.info("CA %s alerts: %s", args.window, f"{len(alerts):,}")

    sample = build_matched_sample(
        panel, alerts, outcome=args.outcome, control_weeks=args.control_weeks
    )
    if sample.empty:
        log.error("matched sample is empty -- aborting")
        sys.exit(1)
    log.info("Matched sample: %s rows; %s events; treated rows %s",
             f"{len(sample):,}", sample["event_id"].nunique(),
             f"{int(sample['treated'].sum()):,}")

    sample, terms = add_offset_interactions(sample)
    rows = run_matched_model(sample, args.outcome, terms)

    out = pd.DataFrame(rows)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    log.info("Saved -> %s", OUT_PATH)

    est = out[out["record_type"] == "estimate"]
    if len(est):
        log.info("\n%s", est.sort_values("event_hour")[
            ["event_hour", "pct_change", "ci_low_pct", "ci_high_pct", "pvalue"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
