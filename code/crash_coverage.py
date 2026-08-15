"""Coverage manifests and validated zero-balanced crash panels.

The functions in this module intentionally separate two concerns: validating
whether a source reporting unit is complete, and expanding sparse events only
within units that passed that validation.  A missing row therefore cannot be
silently interpreted as a zero for a failed download.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

import pandas as pd


@dataclass(frozen=True)
class CoverageResult:
    """Machine-readable validation result for one reporting unit."""

    source: str
    state: str
    year: int
    county_fips: str | None
    expected_records: int | None
    fetched_records: int
    retained_records: int
    duplicate_records: int
    invalid_date_count: int
    invalid_geography_count: int
    observed_min_date: str | None
    observed_max_date: str | None
    request_complete: bool
    coverage_valid: bool
    failure_reasons: tuple[str, ...]
    source_url: str
    source_checksum: str | None

    def to_mapping(self) -> dict:
        """Return a deterministic, serialization-friendly mapping."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "CoverageResult":
        """Construct a result from a manifest row or mapping.

        This accepts CSV/parquet rows where ``failure_reasons`` may have been
        serialized as a delimiter-separated string.
        """
        if isinstance(mapping, cls):
            return mapping
        values = dict(mapping)
        reasons = values.get("failure_reasons", ())
        if isinstance(reasons, str):
            reasons = tuple(x for x in reasons.split("|") if x)
        elif reasons is None or (isinstance(reasons, float) and pd.isna(reasons)):
            reasons = ()
        else:
            reasons = tuple(reasons)
        values["failure_reasons"] = reasons
        if values.get("county_fips") is not None and not pd.isna(values["county_fips"]):
            values["county_fips"] = str(values["county_fips"]).zfill(5)
        else:
            values["county_fips"] = None
        for field in ("expected_records", "fetched_records", "retained_records",
                      "duplicate_records", "invalid_date_count",
                      "invalid_geography_count"):
            value = values.get(field)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                values[field] = None if field == "expected_records" else 0
            else:
                values[field] = int(value)
        values["year"] = int(values["year"])
        values["request_complete"] = _as_bool(values["request_complete"])
        values["coverage_valid"] = _as_bool(values["coverage_valid"])
        values.setdefault("source_url", "")
        values.setdefault("source_checksum", None)
        values.setdefault("state", "")
        values.setdefault("source", "")
        return cls(**values)


def validate_reporting_unit(
    *,
    source: str,
    state: str,
    year: int,
    expected_records: int | None = None,
    fetched_records: int = 0,
    retained_records: int = 0,
    duplicate_records: int = 0,
    invalid_date_count: int = 0,
    invalid_geography_count: int = 0,
    request_complete: bool = False,
    terminal_error: object | None = None,
    required_columns_ok: bool = True,
    observed_min_date: str | None = None,
    observed_max_date: str | None = None,
    county_fips: str | None = None,
    source_url: str = "",
    source_checksum: str | None = None,
) -> CoverageResult:
    """Validate one source/state/year (or county/year) reporting unit.

    All applicable failures are collected so a manifest row explains the full
    reason a unit was rejected.  An expected count of zero is a valid genuine
    empty response when the request itself completed without diagnostics.
    """
    failures: list[str] = []

    if terminal_error is not None:
        failures.append("terminal_page_error")
    if not request_complete:
        failures.append("request_incomplete")
    if expected_records is not None and fetched_records != expected_records:
        failures.append("fetch_count_mismatch")
    if fetched_records < 0 or retained_records < 0:
        failures.append("invalid_record_count")
    if retained_records > fetched_records:
        failures.append("retained_exceeds_fetched")
    if retained_records != fetched_records:
        failures.append("retained_count_mismatch")
    if duplicate_records > 0:
        failures.append("duplicate_records")
    if invalid_date_count > 0:
        failures.append("invalid_dates")
    if invalid_geography_count > 0:
        failures.append("invalid_geography")
    if not required_columns_ok:
        failures.append("missing_required_columns")
    if expected_records is not None and expected_records < 0:
        failures.append("invalid_expected_count")

    # Date bounds describe observed events, not retrieval completeness.  Empty
    # and low-frequency units therefore legitimately have null bounds.  If
    # bounds are provided, malformed or reversed values remain diagnostics.
    parsed_bounds = []
    for label, value in (("observed_min_date", observed_min_date),
                         ("observed_max_date", observed_max_date)):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            parsed_bounds.append(None)
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            failures.append("invalid_observed_dates")
            parsed_bounds.append(None)
        else:
            parsed_bounds.append(pd.Timestamp(parsed).date().isoformat())
    if all(bound is not None for bound in parsed_bounds) and parsed_bounds[0] > parsed_bounds[1]:
        failures.append("reversed_observed_dates")

    # Preserve insertion order while preventing duplicate diagnostics when a
    # caller supplied overlapping error conditions.
    failures = list(dict.fromkeys(failures))
    normalized_county = None if county_fips is None else str(county_fips).zfill(5)
    normalized_min = parsed_bounds[0]
    normalized_max = parsed_bounds[1]
    return CoverageResult(
        source=str(source),
        state=str(state),
        year=int(year),
        county_fips=normalized_county,
        expected_records=None if expected_records is None else int(expected_records),
        fetched_records=int(fetched_records),
        retained_records=int(retained_records),
        duplicate_records=int(duplicate_records),
        invalid_date_count=int(invalid_date_count),
        invalid_geography_count=int(invalid_geography_count),
        observed_min_date=normalized_min,
        observed_max_date=normalized_max,
        request_complete=bool(request_complete),
        coverage_valid=not failures,
        failure_reasons=tuple(failures),
        source_url=str(source_url),
        source_checksum=source_checksum,
    )


_STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}


def _state_code(value: object) -> str:
    text = str(value).strip().upper()
    if text.isdigit():
        return text.zfill(2)
    return _STATE_FIPS.get(text, text)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "", "nan", "none"}:
            return False
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value)


def _normalize_fips(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def _manifest_frame(manifest: pd.DataFrame | Iterable[CoverageResult]) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        frame = manifest.copy()
    else:
        rows = [item.to_mapping() if isinstance(item, CoverageResult) else dict(item)
                for item in manifest]
        frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    required = {"source", "state", "year", "coverage_valid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    frame["state"] = frame["state"].map(_state_code)
    frame["year"] = frame["year"].astype(int)
    frame["coverage_valid"] = frame["coverage_valid"].map(_as_bool)
    if "county_fips" not in frame.columns:
        frame["county_fips"] = None
    frame["county_fips"] = frame["county_fips"].where(frame["county_fips"].notna(), None)
    frame.loc[frame["county_fips"].notna(), "county_fips"] = _normalize_fips(
        frame.loc[frame["county_fips"].notna(), "county_fips"]
    )
    return frame


def _universe_for_unit(
    universe: pd.DataFrame,
    unit: pd.Series,
    reporting_unit: Literal["state_year", "county_year"],
) -> pd.DataFrame:
    if "fips" not in universe.columns:
        raise ValueError("county_universe must contain a fips column")
    out = universe.copy()
    out["fips"] = _normalize_fips(out["fips"])
    if "state" in out.columns:
        states = out["state"].map(_state_code)
    else:
        states = out["fips"].str[:2]
    mask = states.eq(_state_code(unit["state"]))
    if "year" in out.columns:
        mask &= pd.to_numeric(out["year"], errors="coerce").eq(int(unit["year"]))
    if reporting_unit == "county_year":
        county = unit.get("county_fips")
        if county is None or pd.isna(county):
            return out.iloc[0:0]
        mask &= out["fips"].eq(str(county).zfill(5))
    return out.loc[mask, ["fips"]].drop_duplicates()


def balance_validated_panel(
    sparse: pd.DataFrame,
    manifest: pd.DataFrame,
    county_universe: pd.DataFrame,
    outcome_availability: Mapping[str, bool],
    *,
    reporting_unit: Literal["state_year", "county_year"],
) -> pd.DataFrame:
    """Expand sparse events over county-day grids for valid units only."""
    if reporting_unit not in {"state_year", "county_year"}:
        raise ValueError("reporting_unit must be 'state_year' or 'county_year'")
    if "fips" not in sparse.columns or "date" not in sparse.columns:
        raise ValueError("sparse must contain fips and date columns")
    if not outcome_availability:
        raise ValueError("outcome_availability cannot be empty")

    sparse_frame = sparse.copy()
    sparse_frame["fips"] = _normalize_fips(sparse_frame["fips"])
    sparse_frame["date"] = pd.to_datetime(sparse_frame["date"], errors="coerce").dt.normalize()
    sparse_frame = sparse_frame.dropna(subset=["date"])
    sparse_has_source = "source" in sparse_frame.columns
    if sparse_has_source:
        sparse_frame["source"] = sparse_frame["source"].where(
            sparse_frame["source"].notna(), ""
        ).astype(str)
    outcomes = list(outcome_availability)
    for column in outcomes:
        if column not in sparse_frame.columns:
            sparse_frame[column] = pd.NA
        sparse_frame[column] = pd.to_numeric(sparse_frame[column], errors="coerce")
    sparse_group_keys = (["source"] if sparse_has_source else []) + ["fips", "date"]
    sparse_frame = sparse_frame.groupby(
        sparse_group_keys, as_index=False, dropna=False
    )[outcomes].sum(min_count=1)
    universe = county_universe.copy()
    valid_manifest = _manifest_frame(manifest)
    if valid_manifest.empty:
        return pd.DataFrame(columns=["fips", "date", "year", *outcomes,
                                     "coverage_valid", "coverage_unit",
                                     "structural_zero", "source"])

    # A source-less sparse frame is safe only when the manifest names one
    # source. Otherwise there is no identity with which to exclude events from
    # failed sources, so fail closed rather than silently leaking them.
    manifest_sources = valid_manifest["source"].dropna().astype(str).unique()
    if not sparse_has_source and len(manifest_sources) > 1:
        raise ValueError(
            "sparse must contain source when manifest has multiple sources"
        )

    chunks: list[pd.DataFrame] = []
    for _, unit in valid_manifest.loc[valid_manifest["coverage_valid"]].iterrows():
        counties = _universe_for_unit(universe, unit, reporting_unit)
        if counties.empty:
            continue
        dates = pd.date_range(f"{int(unit['year'])}-01-01",
                              f"{int(unit['year'])}-12-31", freq="D")
        grid = pd.MultiIndex.from_product(
            [counties["fips"].tolist(), dates], names=["fips", "date"]
        ).to_frame(index=False)
        grid["year"] = int(unit["year"])
        source = str(unit["source"])
        if sparse_has_source:
            unit_sparse = sparse_frame.loc[sparse_frame["source"].eq(source)]
            grid["source"] = source
            merge_keys = ["source", "fips", "date"]
        else:
            unit_sparse = sparse_frame
            merge_keys = ["fips", "date"]
        grid = grid.merge(unit_sparse, on=merge_keys, how="left", indicator=True)
        grid["structural_zero"] = ~grid["_merge"].eq("both")
        grid = grid.drop(columns="_merge")
        for column, available in outcome_availability.items():
            if available:
                grid[column] = grid[column].fillna(0)
            else:
                # Use IEEE NaN (rather than nullable ``pd.NA``) so downstream
                # numeric estimators see an ordinary missing numeric outcome.
                grid[column] = float("nan")
        grid["coverage_valid"] = True
        grid["coverage_unit"] = reporting_unit
        grid["source"] = source
        chunks.append(grid)

    if not chunks:
        return pd.DataFrame(columns=["fips", "date", "year", *outcomes,
                                     "coverage_valid", "coverage_unit",
                                     "structural_zero", "source"])
    result = pd.concat(chunks, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    return result[["fips", "date", "year", *outcomes, "coverage_valid",
                   "coverage_unit", "structural_zero", "source"]]


def _serialize_failure_reasons(value: object) -> str:
    """Serialize tuple/list diagnostics while normalizing missing values."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return "|".join(str(item) for item in value if item is not None)
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def write_manifest(
    results: pd.DataFrame | Iterable[CoverageResult],
    output_dir: str | Path = "data/processed/coverage",
    *,
    filename: str = "coverage",
) -> tuple[Path, Path]:
    """Write deterministic CSV and parquet manifest files.

    Rows are sorted by source, state, year, and county FIPS.  Failure reasons
    are pipe-delimited in both formats so the output remains scalar and stable.
    """
    frame = _manifest_frame(results)
    if frame.empty:
        frame = pd.DataFrame(columns=[field.name for field in CoverageResult.__dataclass_fields__.values()])
    else:
        if "failure_reasons" in frame.columns:
            frame["failure_reasons"] = frame["failure_reasons"].map(
                _serialize_failure_reasons
            )
        columns = [field.name for field in CoverageResult.__dataclass_fields__.values()]
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[columns]
        frame = frame.sort_values(
            ["source", "state", "year", "county_fips"],
            kind="mergesort", na_position="first",
        ).reset_index(drop=True)
    # Stable scalar dtypes avoid pandas emitting different parquet schemas for
    # empty/nonempty manifests.
    if "failure_reasons" in frame.columns:
        frame["failure_reasons"] = frame["failure_reasons"].fillna("").astype(str)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / f"{filename}.csv"
    parquet_path = target / f"{filename}.parquet"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    frame.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path
