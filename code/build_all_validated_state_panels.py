"""Build validated state-DOT panels for every state with an accepted allowlist entry.

Thin driver around ``build_validated_state_panel`` -- constructs the county
universe from each state's own source contract (``expected_county_fips``)
since there is no separate observed-data county-universe file, then writes
one validated panel per state under ``data/processed/validated/``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_validated_crash_panels import (
    _STATE_SPARSE_FILES,
    ACCEPTED_STATE_YEARS_PATH,
    VALIDATED_DIR,
    build_validated_state_panel,
    write_accepted_state_years,
    write_validated_state_panel,
)
from state_dot_sources import STATE_SOURCE_SPECS, get_spec
from validate_state_fatalities import load_review_allowlist

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
COVERAGE = DATA_PROC / "coverage"
# The reviewed input allowlist lives in config/; ACCEPTED_STATE_YEARS_PATH
# (imported above) is a *different*, output-only path that
# write_accepted_state_years() writes to -- do not read decisions from it.
REVIEW_ALLOWLIST_PATH = ROOT / "config" / "accepted_state_years.csv"

MANIFEST_FILES = {
    "CA": "ca_coverage.csv", "FL": "fl_coverage.csv", "IL": "il_coverage.csv",
    "IA": "ia_coverage.csv", "MA": "ma_coverage.csv", "NY": "ny_coverage.csv",
    "OR": "or_coverage.csv", "TN": "tn_coverage.csv", "VA": "va_coverage.csv",
    "WI": "wisconsin_coverage.csv",
    "DE": "de_coverage.csv",
    "NC": "nc_coverage.csv",
    "TX": "tx_coverage.csv",
    "UT": "ut_coverage.csv",
    "CT": "ct_coverage.csv",
    "MOCO": "moco_coverage.csv",
    "HI": "hi_coverage.csv",
    "INMPO": "inmpo_coverage.csv",
    "IDCOMPASS": "idcompass_coverage.csv",
}


def _county_universe_for(state: str, years: list[int]) -> pd.DataFrame:
    spec = get_spec(state)
    fips = sorted(spec.expected_county_fips)
    # An explicit "state" column (rather than relying on balance_validated_panel's
    # fallback of taking fips[:2] as the state code) is required for any source
    # whose "state" identifier isn't a real 2-letter/2-digit code -- e.g. a
    # sub-state (single-county) source keyed by its own 5-digit county FIPS.
    rows = [{"fips": f, "year": year, "state": spec.state} for year in years for f in fips]
    return pd.DataFrame(rows)


def main() -> None:
    allowlist = load_review_allowlist(REVIEW_ALLOWLIST_PATH)
    accepted_all: list[pd.DataFrame] = []
    for state in MANIFEST_FILES:
        spec = get_spec(state)
        requested = sorted(spec.requested_years - spec.excluded_years)
        sparse = pd.read_parquet(DATA_PROC / _STATE_SPARSE_FILES[state])
        manifest = pd.read_csv(COVERAGE / MANIFEST_FILES[state])
        manifest["state"] = state
        county_universe = _county_universe_for(state, requested)
        panel, accepted = build_validated_state_panel(
            state, sparse, manifest, county_universe,
            years=requested, allowlist_path=REVIEW_ALLOWLIST_PATH,
        )
        accepted_all.append(accepted)
        if panel.empty:
            print(f"{state}: no accepted units, skipped")
            continue
        out = write_validated_state_panel(panel, state, output_dir=VALIDATED_DIR)
        print(f"{state}: wrote {len(panel):,} rows to {out}")

    combined_accepted = pd.concat(accepted_all, ignore_index=True) if accepted_all else pd.DataFrame()
    if not combined_accepted.empty:
        out = write_accepted_state_years(combined_accepted, path=ACCEPTED_STATE_YEARS_PATH)
        print(f"wrote {len(combined_accepted)} accepted rows to {out}")


if __name__ == "__main__":
    main()
