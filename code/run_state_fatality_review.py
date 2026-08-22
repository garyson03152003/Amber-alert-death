"""Generate the FARS-vs-state-DOT fatality comparison report for review.

This produces evidence only. It never writes to the reviewed allowlist
(``config/accepted_state_years.csv``) itself -- that decision is made
separately after inspecting this report's metrics.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_validated_crash_panels import _STATE_SPARSE_FILES, canonicalize_state_sparse
from state_dot_sources import STATE_SOURCE_SPECS
from validate_state_fatalities import validate_state_fatalities, write_validation_report

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
COVERAGE = DATA_PROC / "coverage"

# Only states whose full requested-year range passed strict coverage
# validation in this session. TX (only 2020/5 years passed before repeated
# server failures) and NV (source server outage) are excluded until their
# coverage is complete.
READY_STATES = ["CA", "FL", "IL", "IA", "MA", "NY", "OR", "TN", "VA", "WI", "NC", "TX", "UT", "CT", "MOCO", "HI", "INMPO", "IDCOMPASS"]

MANIFEST_FILES = {
    "CA": "ca_coverage.csv", "FL": "fl_coverage.csv", "IL": "il_coverage.csv",
    "IA": "ia_coverage.csv", "MA": "ma_coverage.csv", "NY": "ny_coverage.csv",
    "OR": "or_coverage.csv", "TN": "tn_coverage.csv", "VA": "va_coverage.csv",
    "WI": "wisconsin_coverage.csv", "NC": "nc_coverage.csv", "TX": "tx_coverage.csv",
    "UT": "ut_coverage.csv", "CT": "ct_coverage.csv", "MOCO": "moco_coverage.csv", "HI": "hi_coverage.csv",
    "INMPO": "inmpo_coverage.csv", "IDCOMPASS": "idcompass_coverage.csv",
}


def main() -> None:
    state_events = []
    manifests = [pd.read_csv(COVERAGE / "fars_coverage.csv")]
    for state in READY_STATES:
        sparse = pd.read_parquet(DATA_PROC / _STATE_SPARSE_FILES[state])
        canonical = canonicalize_state_sparse(state, sparse)
        state_events.append(canonical)
        state_manifest = pd.read_csv(COVERAGE / MANIFEST_FILES[state])
        # Coverage manifests key ``state`` by numeric FIPS; the review helper
        # expects the same 2-letter abbreviation used by STATE_SOURCE_SPECS.
        state_manifest["state"] = state
        manifests.append(state_manifest)

    state_events_df = pd.concat(state_events, ignore_index=True)
    fars_events = pd.read_parquet(DATA_PROC / "fars_events_county_day.parquet")
    manifest = pd.concat(manifests, ignore_index=True)

    report = validate_state_fatalities(
        state_events_df, fars_events, manifest,
        allowlist_path=ROOT / "config" / "accepted_state_years.csv",
        states=READY_STATES,
    )
    out = write_validation_report(report, ROOT / "output" / "tables" / "state_fars_fatality_validation.csv")
    print(f"wrote {len(report)} state-year rows to {out}")
    pd.set_option("display.width", 200)
    print(report.sort_values(["state", "year"]).to_string(index=False))


if __name__ == "__main__":
    main()
