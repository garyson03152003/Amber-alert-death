"""Build a validated national FARS county-day fatal-crash panel.

FARS is a national census of *fatal* motor-vehicle crashes.  It must not be
used as an all-crash or serious-injury source.  This module is deliberately
import-safe: downloads happen only through :func:`main` or an explicit build
function, never at import time.
"""
from __future__ import annotations

import io
import hashlib
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from crash_coverage import CoverageResult, validate_reporting_unit, write_manifest


ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
EVENTS_OUT = DATA_PROC / "fars_events_county_day.parquet"
BALANCED_OUT = DATA_PROC / "fars_balanced_county_day.parquet"
POPULATION_PATH = DATA_PROC / "county_population.parquet"
FIPS_CROSSWALK_PATH = DATA_PROC / "county_fips_crosswalk.parquet"
YEARS = tuple(range(2013, 2025))
FARS_SOURCE = "FARS_NHTSA"
FARS_URL_TEMPLATE = (
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/"
    "{year}/National/FARS{year}NationalCSV.zip"
)
ADVERSE_WEATHER = {2, 3, 4, 5, 6, 11, 12}

# Connecticut retired its county-equivalent FIPS geography in 2022.  Do not
# silently combine old counties with the new planning regions.  Events remain
# available in the sparse FARS output; only the longitudinal balanced panel
# excludes them until a tested crosswalk is added.
CONNECTICUT_LONGITUDINAL_POLICY = "exclude_until_crosswalk"
CONNECTICUT_FIPS = "09"
CONNECTICUT_MANIFEST_WARNING = "connecticut_excluded_from_longitudinal_panel"
# Connecticut's former counties remain valid sparse-event geography during the
# Census transition to planning regions.  This is event validation only: the
# balanced longitudinal panel still excludes every Connecticut FIPS.
LEGACY_CONNECTICUT_COUNTY_FIPS = frozenset({
    "09001", "09003", "09005", "09007", "09009", "09011", "09013", "09015",
})

CONNECTICUT_TRANSITION_YEARS = frozenset(range(2021, 2025))

# FARS's own accident-level source files do not always adopt a Census county
# rename or boundary change the same year Census does -- for example South
# Dakota's Shannon County (46113, renamed Oglala Lakota County 46102 in 2015)
# and a handful of Alaska borough codes appear in FARS archives for several
# years after Census retired them. ``permitted_fips_for_year`` therefore
# treats a FIPS code as permitted in every year if the Census Gazetteer
# recognized it in *any* year of the 2013-2024 crosswalk window, rather than
# only the single processing year.
UNKNOWN_COUNTY_CODES = frozenset({0, 997, 998, 999})

# 50 states plus the District of Columbia.  This intentionally excludes PR and
# other territories even though some use numeric state-like FIPS codes.
US_STATE_FIPS = frozenset({
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
})
CANONICAL_COLUMNS = [
    "fips", "date", "fatal_crashes", "person_fatals", "drunk_fatals",
    "sober_fatals", "weather_adverse",
]


def fetch_zip(year: int, session: requests.Session) -> tuple[zipfile.ZipFile, str]:
    """Fetch one FARS national archive with bounded retries."""
    url = FARS_URL_TEMPLATE.format(year=year)
    last_error: Exception | None = None
    for delay in (0, 4, 8, 16):
        if delay:
            time.sleep(delay)
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            payload = response.content
            return zipfile.ZipFile(io.BytesIO(payload)), hashlib.sha256(payload).hexdigest()
        except Exception as exc:  # downloaded archive errors are retryable
            last_error = exc
    raise RuntimeError(f"Could not download FARS {year}: {last_error}")


def clean_cols(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize FARS CSV headers without mutating the caller's frame."""
    out = frame.copy()
    out.columns = [str(col).encode("ascii", "ignore").decode().strip().upper()
                   for col in out.columns]
    return out


def read_file(zf: zipfile.ZipFile, keyword: str) -> pd.DataFrame:
    """Read the primary FARS CSV whose basename contains ``keyword``."""
    candidates = [
        name for name in zf.namelist()
        if keyword.lower() in name.lower()
        and name.lower().endswith(".csv")
        and "sf" not in name.rsplit("/", 1)[-1].lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No FARS CSV matching {keyword!r}")
    with zf.open(candidates[0]) as handle:
        return clean_cols(pd.read_csv(handle, encoding="latin1", low_memory=False))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _required_columns(frame: pd.DataFrame, names: set[str], year: int, file_name: str) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise ValueError(f"FARS {year} {file_name} missing columns: {sorted(missing)}")


def _drunk_cases(vehicles: pd.DataFrame, year: int) -> set[object]:
    vehicles = clean_cols(vehicles)
    _required_columns(vehicles, {"ST_CASE"}, year, "vehicle")
    drink_column = "DR_DRINK"
    if drink_column not in vehicles.columns:
        alternatives = [col for col in vehicles.columns if "DRINK" in col]
        if not alternatives:
            return set()
        drink_column = alternatives[0]
    drink = pd.to_numeric(vehicles[drink_column], errors="coerce").fillna(0)
    return set(vehicles.loc[drink.eq(1), "ST_CASE"].tolist())


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame({
        "fips": pd.Series(dtype="string"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "fatal_crashes": pd.Series(dtype="int64"),
        "person_fatals": pd.Series(dtype="int64"),
        "drunk_fatals": pd.Series(dtype="int64"),
        "sober_fatals": pd.Series(dtype="int64"),
        "weather_adverse": pd.Series(dtype="int64"),
    })


def _population_counties(population: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Census-derived county universe without hiding rows."""
    required = {"fips", "year", "population"}
    missing = required - set(population.columns)
    if missing:
        raise ValueError(f"population missing columns: {sorted(missing)}")
    pop = population.loc[:, ["fips", "year", "population"]].copy()
    pop["fips"] = pop["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop = pop.loc[
        pop["year"].notna() & pop["population"].notna() & pop["population"].ge(0)
        & pop["fips"].str[:2].isin(US_STATE_FIPS)
        & ~pop["fips"].str.endswith(("000", "999"))
    ].copy()
    if pop.empty:
        raise ValueError("population has no valid 50-state-plus-DC counties")
    pop["year"] = pop["year"].astype(int)
    return pop[["fips", "year"]].drop_duplicates()


def permitted_fips_for_year(year: int, population: pd.DataFrame | None = None) -> set[str]:
    """Return the Census county-equivalent FIPS set FARS may use for one year.

    The Population Estimates Program backcasts historical rows onto *current*
    county boundaries, so it cannot distinguish a genuinely-in-effect legacy
    code from a source-quality error. When no explicit ``population`` frame is
    supplied, prefer the year-accurate Census Gazetteer crosswalk built by
    ``build_county_fips_crosswalk.py``, unioned across every crosswalk year
    (2013-2024) rather than restricted to ``year`` alone, because FARS's own
    source files lag Census renames and boundary changes by several years.
    Callers that pass ``population`` directly (unit tests, callers without
    network access) keep the deterministic population-derived behavior
    unchanged. Connecticut is retained here because sparse FARS events are
    still published for it; only the balanced longitudinal panel excludes
    Connecticut.
    """
    if population is None and FIPS_CROSSWALK_PATH.is_file():
        crosswalk = pd.read_parquet(FIPS_CROSSWALK_PATH)
        crosswalk["fips"] = crosswalk["fips"].astype(str).str.zfill(5)
        if not crosswalk.empty:
            permitted = set(crosswalk["fips"])
            if int(year) in CONNECTICUT_TRANSITION_YEARS:
                permitted.update(LEGACY_CONNECTICUT_COUNTY_FIPS)
            return permitted

    pop = _population_counties(
        pd.read_parquet(POPULATION_PATH) if population is None else population
    )
    candidates = pop.loc[pop["year"].le(int(year)), "year"]
    source_year = int(candidates.max()) if not candidates.empty else int(pop["year"].min())
    permitted = set(pop.loc[pop["year"].eq(source_year), "fips"])
    if int(year) in CONNECTICUT_TRANSITION_YEARS:
        # Keep current Census planning-region codes from ``permitted`` and
        # union the legacy county codes so transition-era FARS rows are not
        # discarded merely because the population reference changed geography.
        permitted.update(LEGACY_CONNECTICUT_COUNTY_FIPS)
    return permitted


def build_fars_year(year: int, session: requests.Session) -> tuple[pd.DataFrame, CoverageResult]:
    """Download, validate, and aggregate one complete FARS national archive."""
    if int(year) not in YEARS:
        raise ValueError(f"FARS canonical build requires a year in {YEARS[0]}-{YEARS[-1]}")
    archive, source_checksum = fetch_zip(int(year), session)
    accidents = clean_cols(read_file(archive, "accident"))
    _required_columns(
        accidents,
        {"ST_CASE", "STATE", "COUNTY", "YEAR", "MONTH", "DAY", "FATALS"},
        int(year),
        "accident",
    )
    raw_count = len(accidents)
    duplicate_count = int(accidents.duplicated(subset=["ST_CASE"], keep="first").sum())
    # A FARS ST_CASE identifies one crash.  Keep first deterministically so
    # diagnostics describe raw duplication without double-counting deaths.
    accidents = accidents.drop_duplicates(subset=["ST_CASE"], keep="first").copy()

    state = _numeric(accidents, "STATE")
    county = _numeric(accidents, "COUNTY")
    state_integral = state.notna() & state.eq(state.round())
    county_integral = county.notna() & county.eq(county.round())
    state_code = state.where(state_integral).astype("Int64").astype(str).str.zfill(2)
    county_code = county.where(county_integral).astype("Int64").astype(str).str.zfill(3)
    fips = state_code + county_code
    # Validate against actual Census geography, not merely a numeric range:
    # syntactically plausible but nonexistent codes (for example 01997) are
    # source-quality failures, not counties with zero crashes.
    geography_valid = (
        state_integral & county_integral
        & state_code.isin(US_STATE_FIPS)
        & county.between(1, 998)
        & fips.isin(permitted_fips_for_year(int(year)))
    )
    # A row with a well-formed, real-US-state FIPS that still cannot be
    # resolved to any Census county in the 2013-2024 crosswalk (explicit
    # "unknown/not applicable" placeholders such as county 0/997/998/999, or
    # a code Census never recognized in any covered year) is a genuinely
    # unresolvable source record, not a structurally malformed one. It is
    # still excluded from the retained panel below, but a small, bounded
    # residual of these does not by itself invalidate the reporting unit.
    # Malformed rows (non-integral geography, non-US-state territories such
    # as Puerto Rico) remain a hard invalid_geography failure.
    unresolvable_mask = (
        ~geography_valid & state_integral & county_integral & state_code.isin(US_STATE_FIPS)
    )
    unresolvable_geography_count = int(unresolvable_mask.sum())
    invalid_geography_count = int((~geography_valid & ~unresolvable_mask).sum())
    archive_year = _numeric(accidents, "YEAR")
    date = pd.to_datetime(
        {"year": archive_year, "month": _numeric(accidents, "MONTH"),
         "day": _numeric(accidents, "DAY")},
        errors="coerce",
    )
    date_valid = date.notna() & archive_year.eq(int(year)) & date.dt.year.eq(int(year))
    invalid_date_count = int((~date_valid).sum())
    fatalities = _numeric(accidents, "FATALS")
    fatality_valid = fatalities.notna() & fatalities.ge(1)
    invalid_date_count += int((~fatality_valid & date_valid).sum())
    retained = accidents.loc[geography_valid & date_valid & fatality_valid].copy()
    retained["date"] = date.loc[retained.index].dt.normalize()
    retained["person_fatals"] = fatalities.loc[retained.index].astype(int)
    retained["fips"] = fips.loc[retained.index]
    retained["weather_adverse"] = (
        _numeric(accidents, "WEATHER").loc[retained.index].isin(ADVERSE_WEATHER).astype(int)
        if "WEATHER" in accidents.columns else 0
    )
    drunk_cases = _drunk_cases(read_file(archive, "vehicle"), int(year))
    retained["is_drunk"] = retained["ST_CASE"].isin(drunk_cases)
    if retained.empty:
        events = _empty_events()
    else:
        retained["drunk_fatals"] = retained["person_fatals"].where(retained["is_drunk"], 0)
        events = (
            retained.groupby(["fips", "date"], as_index=False)
            .agg(fatal_crashes=("ST_CASE", "nunique"), person_fatals=("person_fatals", "sum"),
                 drunk_fatals=("drunk_fatals", "sum"), weather_adverse=("weather_adverse", "max"))
        )
        events["sober_fatals"] = events["person_fatals"] - events["drunk_fatals"]
        events = events[CANONICAL_COLUMNS].sort_values(["fips", "date"]).reset_index(drop=True)

    observed = events["date"] if not events.empty else pd.Series(dtype="datetime64[ns]")
    coverage = validate_reporting_unit(
        source=FARS_SOURCE, state="US", year=int(year), expected_records=raw_count,
        fetched_records=raw_count, retained_records=len(retained), duplicate_records=duplicate_count,
        invalid_date_count=invalid_date_count, invalid_geography_count=invalid_geography_count,
        unresolvable_geography_count=unresolvable_geography_count,
        request_complete=True, required_columns_ok=True,
        observed_min_date=observed.min() if not observed.empty else None,
        observed_max_date=observed.max() if not observed.empty else None,
        source_url=FARS_URL_TEMPLATE.format(year=int(year)),
        source_checksum=source_checksum,
    )
    return events, coverage


def fars_county_universe(population: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    """Return 50-state-plus-DC, non-Connecticut counties for each requested year.

    The population extract ends at 2023.  A later requested year carries its
    latest available county list forward as geography only, not a population
    estimate; this permits a complete 2024 FARS balancing universe.
    """
    wanted = sorted({int(year) for year in years})
    if not wanted:
        return pd.DataFrame(columns=["fips", "year"])
    pop = _population_counties(population)
    pop = pop.loc[~pop["fips"].str.startswith(CONNECTICUT_FIPS)].copy()
    rows: list[pd.DataFrame] = []
    for target_year in wanted:
        candidates = pop.loc[pop["year"].le(target_year), "year"]
        source_year = int(candidates.max()) if not candidates.empty else int(pop["year"].min())
        counties = pop.loc[pop["year"].eq(source_year), ["fips"]].drop_duplicates()
        if counties.empty:
            raise ValueError(f"no county universe available for {target_year}")
        counties["year"] = target_year
        rows.append(counties)
    return pd.concat(rows, ignore_index=True).sort_values(["fips", "year"]).reset_index(drop=True)


def _balance_fars_events(events: pd.DataFrame, universe: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Zero-balance valid national FARS years over the geographic universe."""
    valid_years = set(manifest.loc[
        manifest["coverage_valid"] & manifest["source"].eq(FARS_SOURCE), "year"
    ].astype(int))
    pieces: list[pd.DataFrame] = []
    event_frame = events.copy()
    event_frame["date"] = pd.to_datetime(event_frame["date"]).dt.normalize()
    for year in sorted(valid_years):
        counties = universe.loc[universe["year"].eq(year), "fips"].drop_duplicates().tolist()
        if not counties:
            continue
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        grid = pd.MultiIndex.from_product([counties, dates], names=["fips", "date"]).to_frame(index=False)
        merged = grid.merge(event_frame, on=["fips", "date"], how="left", indicator=True)
        merged["structural_zero"] = merged.pop("_merge").eq("left_only")
        for column in CANONICAL_COLUMNS[2:]:
            merged[column] = merged[column].fillna(0).astype(int)
        merged["year"] = year
        merged["coverage_valid"] = True
        merged["coverage_unit"] = "national_year"
        merged["source"] = FARS_SOURCE
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame(columns=[*CANONICAL_COLUMNS, "year", "coverage_valid", "coverage_unit", "structural_zero", "source"])
    return pd.concat(pieces, ignore_index=True)


class FarsValidationError(RuntimeError):
    """A complete FARS download failed validation, carrying its diagnostics."""

    def __init__(self, manifest: pd.DataFrame, failed_years: list[int]) -> None:
        self.manifest = manifest
        self.failed_years = failed_years
        super().__init__(f"FARS validation failed for years: {failed_years}")


def build_fars(years: Iterable[int] = YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build canonical FARS 2013-2024 and fail closed on an invalid year."""
    requested = tuple(int(year) for year in years)
    if requested != YEARS:
        raise ValueError(f"canonical FARS build requires every year {YEARS[0]}-{YEARS[-1]}")
    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0"})
    frames: list[pd.DataFrame] = []
    results: list[CoverageResult] = []
    for year in requested:
        try:
            events, result = build_fars_year(year, session)
        except Exception as exc:
            # Retrieval failures are coverage failures too.  Preserve them as
            # a reporting-unit manifest row instead of letting a network
            # exception make the requested year disappear from diagnostics.
            result = validate_reporting_unit(
                source=FARS_SOURCE,
                state="US",
                year=year,
                request_complete=False,
                terminal_error=exc,
                source_url=FARS_URL_TEMPLATE.format(year=year),
            )
        else:
            frames.append(events)
        results.append(result)
    national_manifest = pd.DataFrame([result.to_mapping() for result in results])
    # This is intentionally a separate invalid coverage row: it records a
    # geographic comparability policy, not a failed FARS download.  Sparse
    # Connecticut events remain in EVENTS_OUT, while its panel counties are
    # omitted by fars_county_universe until a tested crosswalk exists.
    policy_manifest = pd.DataFrame([
        CoverageResult(
            source=f"{FARS_SOURCE}_POLICY",
            state=CONNECTICUT_FIPS,
            year=year,
            county_fips=None,
            expected_records=0,
            fetched_records=0,
            retained_records=0,
            duplicate_records=0,
            invalid_date_count=0,
            invalid_geography_count=0,
            observed_min_date=None,
            observed_max_date=None,
            request_complete=True,
            coverage_valid=False,
            failure_reasons=(CONNECTICUT_MANIFEST_WARNING,),
            source_url="https://www.census.gov/programs-surveys/geography/guidance/geo-areas/connecticut.html",
            source_checksum=None,
        ).to_mapping()
        for year in requested
    ])
    manifest = pd.concat([national_manifest, policy_manifest], ignore_index=True)
    if not national_manifest["coverage_valid"].all():
        failed = national_manifest.loc[~national_manifest["coverage_valid"], "year"].tolist()
        raise FarsValidationError(manifest, failed)
    events = pd.concat(frames, ignore_index=True).sort_values(["fips", "date"]).reset_index(drop=True)
    return events, manifest


def main() -> None:
    """Write canonical sparse events, exact coverage, and balanced panel."""
    try:
        events, manifest = build_fars()
    except FarsValidationError as exc:
        # Preserve exact diagnostics even though no canonical panel may be
        # written.  A strict validation failure must remain inspectable.
        write_manifest(exc.manifest, DATA_PROC / "coverage", filename="fars_coverage")
        raise
    universe = fars_county_universe(pd.read_parquet(POPULATION_PATH), YEARS)
    balanced = _balance_fars_events(events, universe, manifest)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_OUT, index=False)
    balanced.to_parquet(BALANCED_OUT, index=False)
    write_manifest(manifest, DATA_PROC / "coverage", filename="fars_coverage")
    print(f"Saved {EVENTS_OUT} ({len(events):,} fatal-event county-days)")
    print(f"Saved {BALANCED_OUT} ({len(balanced):,} validated county-days)")
    print("Connecticut sparse events retained; longitudinal panel policy: "
          f"{CONNECTICUT_LONGITUDINAL_POLICY}", file=sys.stderr)


if __name__ == "__main__":
    main()
