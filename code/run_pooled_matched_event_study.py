"""Matched-week hourly event study on the POOLED multi-state county-hour panel.

Design is unchanged from run_matched_hour_event_study.py: each alert
county-hour is compared against the SAME county, hour and weekday in
neighbouring non-alert weeks, with an event-by-offset fixed effect. Only the
data is larger -- 5.8M crashes across five states rather than 3.2M in
California alone, and ~14x all of national FARS.

Pooling buys power, not identification. Two guards are therefore reported
next to every coefficient:

  * the pre-period (offset < 0) coefficients, which alerts cannot cause and
    which must be jointly flat for the post-period to mean anything;
  * the state composition of the treated events, because California supplies
    roughly half the pooled crashes and a "pooled" result that is really
    California with noise attached should be visible as such.

Balancing
---------
Each source covers a different year range (CA 2016-2022, DE 2013-2024,
UT 2018-2024, MA 2013-2020 excluding date-only 2018, CT 2015-2022). The
panel is therefore balanced per source over its own observed span, so an
absent county-hour inside coverage becomes a true zero while a county-hour
outside a source's coverage is never invented.

Output: output/tables/pooled_matched_event_study.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
from config import DATA_PROC, OUTPUT_TABS
from run_matched_hour_event_study import (
    build_matched_sample, add_offset_interactions, run_matched_model, CONTROL_WEEKS,
)

log = base.log

POOLED = DATA_PROC / "pooled_county_hour.parquet"
OUT_PATH = OUTPUT_TABS / "pooled_matched_event_study.csv"


def balance_per_source(pooled: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill each source over its own coverage span only."""
    parts = []
    for src, g in pooled.groupby("source"):
        counties = np.sort(g["fips"].unique())
        dates = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        grid = pd.MultiIndex.from_product(
            [counties, dates, range(24)], names=["fips", "date", "hour"]
        ).to_frame(index=False)
        merged = grid.merge(g[["fips", "date", "hour", "crashes"]],
                            on=["fips", "date", "hour"], how="left")
        merged["crashes"] = merged["crashes"].fillna(0.0)
        merged["source"] = src
        parts.append(merged)
        log.info("[%s] balanced to %s county-hours (%s..%s)", src, f"{len(merged):,}",
                 dates.min().date(), dates.max().date())
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="day", choices=["day", "night"])
    parser.add_argument("--control-weeks", type=int, default=CONTROL_WEEKS)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="source keys to drop (e.g. CA for a sensitivity run)")
    args = parser.parse_args(argv)

    pooled = pd.read_parquet(POOLED)
    pooled["date"] = pd.to_datetime(pooled["date"]).dt.normalize()
    if args.exclude:
        pooled = pooled[~pooled["source"].isin(args.exclude)]
        log.info("excluded sources: %s", args.exclude)
    panel = balance_per_source(pooled)
    panel["ts"] = panel["date"] + pd.to_timedelta(panel["hour"], unit="h")
    log.info("Balanced pooled panel: %s county-hours | %s crashes",
             f"{len(panel):,}", f"{int(panel['crashes'].sum()):,}")

    alerts = base.load_verified_alerts(window=args.window, detail=True)
    alerts = alerts[alerts["fips"].isin(set(panel["fips"]))].copy()
    log.info("%s alerts inside pooled coverage: %s", args.window, f"{len(alerts):,}")

    sample = build_matched_sample(panel, alerts, outcome="crashes",
                                  control_weeks=args.control_weeks)
    if sample.empty:
        log.error("matched sample empty"); sys.exit(1)

    src_by_fips = panel.drop_duplicates("fips").set_index("fips")["source"]
    sample["source"] = sample["fips"].map(src_by_fips)
    comp = (sample[sample["treated"] == 1]
            .drop_duplicates(["event_id"])["source"].value_counts())
    log.info("Matched sample: %s rows | %s events | treated rows %s",
             f"{len(sample):,}", sample["event_id"].nunique(),
             f"{int(sample['treated'].sum()):,}")
    log.info("Treated events by source: %s", comp.to_dict())

    sample, terms = add_offset_interactions(sample)
    rows = run_matched_model(sample, "crashes", terms)

    out = pd.DataFrame(rows)
    out["excluded_sources"] = ",".join(args.exclude) if args.exclude else ""
    for src, n in comp.items():
        out[f"events_{src}"] = n
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    suffix = "_no" + "".join(args.exclude) if args.exclude else ""
    path = OUT_PATH.with_name(OUT_PATH.stem + suffix + OUT_PATH.suffix)
    out.to_csv(path, index=False)
    log.info("Saved -> %s", path)

    est = out[out["record_type"] == "estimate"]
    if len(est):
        log.info("\n%s", est.sort_values("event_hour")[
            ["event_hour", "pct_change", "ci_low_pct", "ci_high_pct", "pvalue"]
        ].to_string(index=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
