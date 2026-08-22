"""Build fail-closed balanced state-DOT county-day panels.

Legacy state files are sparse and may have incomplete retrieval.  This module
reads their coverage manifests first, keeps a state-year only after an
explicit review decision, and writes a separate validated output tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from crash_coverage import balance_validated_panel
from state_dot_sources import STATE_SOURCE_SPECS, get_spec
from validate_state_fatalities import load_review_allowlist


ROOT = Path(__file__).resolve().parent.parent
VALIDATED_DIR = ROOT / "data" / "processed" / "validated"
ACCEPTED_STATE_YEARS_PATH = ROOT / "output" / "tables" / "accepted_state_years.csv"


class MissingCoverageManifestError(RuntimeError):
    """Raised when an explicitly requested state-year lacks diagnostics."""


class InvalidCoverageManifestError(RuntimeError):
    """Raised when a state manifest row belongs to a different source."""


class InvalidCountyUniverseError(RuntimeError):
    """Raised when a state-year county universe is not its source contract."""


class DuplicateCoverageManifestError(RuntimeError):
    """Raised when a coverage reporting unit appears more than once."""


_STATE_SPARSE_FILES = {
    "CA": "california_ccrs_county_day.parquet", "FL": "florida_fdot_county_day.parquet",
    "IL": "illinois_idot_county_day.parquet", "IA": "iowa_dot_county_day.parquet",
    "MA": "massachusetts_massdot_county_day.parquet", "NV": "nevada_ndot_county_day.parquet",
    "NY": "newyork_dot_county_day.parquet", "OR": "oregon_odot_county_day.parquet",
    "TN": "tennessee_tdot_county_day.parquet", "TX": "texas_txdot_county_day.parquet",
    "VA": "virginia_vdot_county_day.parquet", "WI": "wisconsin_dot_county_day.parquet",
    "DE": "delaware_deldot_county_day.parquet",
    "NC": "northcarolina_ncdot_county_day.parquet",
    "UT": "utah_udot_county_day.parquet",
    "CT": "connecticut_uconn_county_day.parquet",
    "MOCO": "montgomery_moco_county_day.parquet",
    "HI": "hawaii_dot_county_day.parquet",
    "INMPO": "indianapolis_mpo_county_day.parquet",
    "IDCOMPASS": "idaho_compass_county_day.parquet",
}
_STATE_COLUMNS = {
    "CA": {"crashes": "ca_crashes", "person_fatals": "ca_fatals", "serious_injury_persons": "ca_serious_inj"},
    "FL": {"crashes": "fl_crashes", "person_fatals": "fl_fatals", "serious_injury_persons": "fl_serious_inj"},
    "IL": {"crashes": "il_crashes", "person_fatals": "il_fatals", "serious_injury_persons": "il_serious_inj"},
    "IA": {"crashes": "ia_crashes", "person_fatals": "ia_fatals", "serious_injury_persons": "ia_serious_inj"},
    "MA": {"crashes": "ma_crashes", "person_fatals": "ma_fatals", "serious_injury_persons": "ma_serious_inj"},
    "NV": {"crashes": "nv_crashes", "person_fatals": "nv_fatals", "serious_injury_persons": "nv_serious_inj"},
    "NY": {"crashes": "ny_crashes"},
    "OR": {"crashes": "or_crashes", "person_fatals": "or_fatals", "serious_injury_persons": "or_serious_inj"},
    "TN": {"crashes": "tn_crashes", "person_fatals": "tn_fatals", "serious_injury_persons": "tn_serious_inj"},
    "TX": {"crashes": "tx_crashes", "person_fatals": "tx_fatals", "serious_injury_persons": "tx_serious_inj"},
    "VA": {"crashes": "va_crashes", "person_fatals": "va_fatals", "serious_injury_persons": "va_serious_inj"},
    "WI": {"crashes": "wi_crashes", "person_fatals": "wi_fatals", "serious_injury_persons": "wi_serious_inj"},
    "DE": {"crashes": "de_crashes"},
    "NC": {"crashes": "nc_crashes", "person_fatals": "nc_fatals", "serious_injury_persons": "nc_serious_inj"},
    "UT": {"crashes": "ut_crashes", "person_fatals": "ut_fatals", "serious_injury_persons": "ut_serious_inj"},
    "CT": {"crashes": "ct_crashes", "person_fatals": "ct_fatals", "serious_injury_persons": "ct_serious_inj"},
    "MOCO": {"crashes": "moco_crashes", "person_fatals": "moco_fatals", "serious_injury_persons": "moco_serious_inj"},
    "HI": {"person_fatals": "hi_fatals"},
    "INMPO": {"person_fatals": "inmpo_fatals", "serious_injury_persons": "inmpo_serious_inj"},
    "IDCOMPASS": {"crashes": "idc_crashes", "person_fatals": "idc_fatals", "serious_injury_persons": "idc_serious_inj"},
}


def canonicalize_state_sparse(state: str, sparse: pd.DataFrame) -> pd.DataFrame:
    """Map a legacy state sparse extract to stable outcome names.

    Fixture callers may already supply canonical columns.  Missing native
    fields are retained as missing, never fabricated as zero.
    """
    state = state.upper()
    spec = get_spec(state)
    required = {"fips", "date"}
    missing = required - set(sparse.columns)
    if missing:
        raise ValueError(f"{state} sparse events missing columns: {sorted(missing)}")
    output = sparse.loc[:, ["fips", "date"]].copy()
    for canonical in spec.outcome_availability:
        native = _STATE_COLUMNS[state].get(canonical)
        if canonical in sparse.columns:
            output[canonical] = sparse[canonical]
        elif native is not None and native in sparse.columns:
            output[canonical] = sparse[native]
        else:
            output[canonical] = float("nan")
    output["source"] = spec.source
    return output


def _state_manifest(manifest: pd.DataFrame, state: str) -> pd.DataFrame:
    if not {"state", "year", "coverage_valid"}.issubset(manifest.columns):
        raise ValueError("manifest must contain state, year, and coverage_valid")
    out = manifest.copy()
    out["state"] = out["state"].astype(str).str.strip().str.upper()
    return out.loc[out["state"].eq(state)].copy()


def _accepted_units(state: str, manifest: pd.DataFrame, years: Iterable[int], allowlist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = sorted({int(year) for year in years})
    state_manifest = _state_manifest(manifest, state)
    observed = set(pd.to_numeric(state_manifest["year"], errors="coerce").dropna().astype(int))
    absent = sorted(set(requested) - observed)
    if absent:
        rendered = ", ".join(f"{state} {year}" for year in absent)
        raise MissingCoverageManifestError(f"missing coverage manifest for {rendered}")

    expected_source = get_spec(state).source
    selected = state_manifest.loc[
        pd.to_numeric(state_manifest["year"], errors="coerce").isin(requested)
    ].copy()
    sources = set(selected.get("source", pd.Series(index=selected.index, dtype=str)).fillna("").astype(str))
    if sources != {expected_source}:
        rendered = ", ".join(sorted(sources)) or "<missing>"
        raise InvalidCoverageManifestError(
            f"{state} selected manifest source {rendered!r}; expected {expected_source}"
        )
    key = ["source", "state", "year"]
    if get_spec(state).reporting_unit == "county_year":
        if "county_fips" not in selected.columns or selected["county_fips"].isna().any():
            raise InvalidCoverageManifestError(
                f"{state} county-year manifest rows require county_fips"
            )
        selected["county_fips"] = _normalize_fips(selected["county_fips"])
        key.append("county_fips")
    duplicate = selected.duplicated(key, keep=False)
    if duplicate.any():
        repeated = selected.loc[duplicate, key].drop_duplicates()
        units = []
        for row in repeated.itertuples(index=False):
            values = row._asdict()
            label = f"{values['state']} {values['year']}"
            if "county_fips" in values:
                label = f"{label} {values['county_fips']}"
            units.append(label)
        raise DuplicateCoverageManifestError(
            f"duplicate coverage manifest reporting unit(s): {', '.join(units)}"
        )
    if get_spec(state).reporting_unit == "county_year":
        expected_counties = set(get_spec(state).expected_county_fips)
        for year in requested:
            observed_counties = set(selected.loc[
                pd.to_numeric(selected["year"], errors="coerce").eq(year), "county_fips"
            ])
            missing, extra = expected_counties - observed_counties, observed_counties - expected_counties
            if missing or extra:
                diagnostics: list[str] = []
                if missing:
                    diagnostics.append(f"missing {len(missing)} county manifest rows")
                if extra:
                    diagnostics.append(f"unexpected {len(extra)} county manifest rows")
                raise InvalidCoverageManifestError(
                    f"{state} {year} county-year coverage manifest is incomplete: {', '.join(diagnostics)}"
                )

    accepted_rows: list[dict[str, object]] = []
    allowed = {(row.state, int(row.year)): row.reason for row in allowlist.itertuples(index=False)}
    for year in requested:
        unit = state_manifest.loc[pd.to_numeric(state_manifest["year"], errors="coerce").eq(year)]
        if unit.empty or not unit["coverage_valid"].map(_as_bool).all():
            continue
        reason = allowed.get((state, year))
        if reason is not None:
            accepted_rows.append({"state": state, "year": year, "review_status": "accepted", "reason": str(reason)})
    accepted = pd.DataFrame(accepted_rows, columns=["state", "year", "review_status", "reason"])
    if accepted.empty:
        return state_manifest.iloc[0:0], accepted
    allowed_years = set(accepted["year"])
    return state_manifest.loc[pd.to_numeric(state_manifest["year"], errors="coerce").isin(allowed_years)], accepted


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if value is not None and not pd.isna(value) else False


def _normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def validate_county_universe(
    state: str, county_universe: pd.DataFrame, years: Iterable[int]
) -> None:
    """Require the target state's contract counties exactly once for each year.

    National county-universe files may contain other states.  For the selected
    state, however, an observed-data-derived subset or a noncounty FIPS would
    create false structural zeros, so it is rejected before any panel write.
    """
    if not {"fips", "year"}.issubset(county_universe.columns):
        raise InvalidCountyUniverseError("county universe must contain fips and year")
    spec = get_spec(state)
    universe = county_universe.loc[:, ["fips", "year"]].copy()
    universe["fips"] = _normalize_fips(universe["fips"])
    universe["year"] = pd.to_numeric(universe["year"], errors="coerce")
    if universe["year"].isna().any():
        raise InvalidCountyUniverseError("county universe contains nonnumeric years")
    universe["year"] = universe["year"].astype(int)
    expected = set(spec.expected_county_fips)
    for year in sorted({int(year) for year in years}):
        observed = set(universe.loc[
            universe["year"].eq(year) & universe["fips"].str.startswith(spec.state_fips), "fips"
        ])
        missing, extra = expected - observed, observed - expected
        if missing or extra:
            diagnostics: list[str] = []
            if missing:
                diagnostics.append(f"missing {len(missing)} county FIPS")
            if extra:
                diagnostics.append(f"unexpected {len(extra)} county FIPS")
            raise InvalidCountyUniverseError(
                f"{state} {year} county universe is invalid: {', '.join(diagnostics)}"
            )


def build_validated_state_panel(
    state: str,
    sparse: pd.DataFrame,
    manifest: pd.DataFrame,
    county_universe: pd.DataFrame,
    *,
    years: Iterable[int] | None = None,
    allowlist_path: str | Path | None = ROOT / "config" / "accepted_state_years.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balance only valid, explicitly reviewed reporting units for one state."""
    state = state.upper()
    spec = get_spec(state)
    state_manifest = _state_manifest(manifest, state)
    requested = (sorted({int(year) for year in years}) if years is not None else
                 sorted(spec.requested_years - spec.excluded_years))
    if not requested:
        raise MissingCoverageManifestError(f"missing coverage manifest for {state}")
    eligible_manifest, accepted = _accepted_units(state, manifest, requested, load_review_allowlist(allowlist_path))
    validate_county_universe(state, county_universe, requested)
    canonical_sparse = canonicalize_state_sparse(state, sparse)
    if eligible_manifest.empty:
        columns = ["fips", "date", "year", *spec.outcome_availability,
                   "coverage_valid", "coverage_unit", "structural_zero", "source"]
        return pd.DataFrame(columns=columns), accepted
    panel = balance_validated_panel(
        canonical_sparse, eligible_manifest, county_universe, spec.outcome_availability,
        reporting_unit=spec.reporting_unit,
    )
    return panel, accepted


def write_validated_state_panel(panel: pd.DataFrame, state: str, *, output_dir: str | Path = VALIDATED_DIR) -> Path:
    """Persist a validated panel separately from legacy sparse source files."""
    target = Path(output_dir) / f"{state.lower()}_county_day.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    return target


def write_accepted_state_years(accepted: pd.DataFrame, *, path: str | Path = ACCEPTED_STATE_YEARS_PATH) -> Path:
    """Write accepted decisions in deterministic order for downstream runners."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    accepted.sort_values(["state", "year"], kind="mergesort").to_csv(target, index=False, lineterminator="\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, choices=sorted(STATE_SOURCE_SPECS))
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--county-universe", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, default=ROOT / "config" / "accepted_state_years.csv")
    parser.add_argument("--year", type=int, action="append", dest="years")
    parser.add_argument("--output-dir", type=Path, default=VALIDATED_DIR)
    parser.add_argument("--accepted-output", type=Path, default=ACCEPTED_STATE_YEARS_PATH)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise MissingCoverageManifestError(f"missing coverage manifest file: {args.manifest}")
    panel, accepted = build_validated_state_panel(
        args.state, pd.read_parquet(args.sparse), pd.read_parquet(args.manifest),
        pd.read_parquet(args.county_universe), years=args.years, allowlist_path=args.allowlist,
    )
    if panel.empty:
        raise RuntimeError(f"no accepted coverage units for {args.state}; inspect review allowlist")
    output = write_validated_state_panel(panel, args.state, output_dir=args.output_dir)
    write_accepted_state_years(accepted, path=args.accepted_output)
    print(f"wrote {len(panel):,} rows to {output}")


if __name__ == "__main__":
    main()
