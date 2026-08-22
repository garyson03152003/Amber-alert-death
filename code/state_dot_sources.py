"""Declarative validation contracts for the state crash-source adapters.

This module deliberately keeps coverage decisions separate from sparse event
tables.  Builders supply request diagnostics (counts, schema audit, parsed
dates and geography failures); these helpers then create the manifest rows
consumed by the balancing and analysis stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import pandas as pd

from crash_coverage import CoverageResult, validate_reporting_unit, write_manifest
from crash_download import IncompleteDownloadError, fetch_arcgis_pages, fetch_socrata_pages


OutcomeAvailability = Mapping[str, bool]


@dataclass(frozen=True)
class StateSourceSpec:
    """Source-specific, non-observed-data coverage contract."""

    source: str
    state: str
    state_fips: str
    reporting_unit: Literal["state_year", "county_year"]
    requested_years: frozenset[int]
    expected_county_fips: frozenset[str]
    native_outcomes: Mapping[str, str]
    comparable_outcomes: frozenset[str]
    query_identifier: str
    excluded_years: frozenset[int] = frozenset()

    @property
    def outcome_availability(self) -> dict[str, bool]:
        """Whether each canonical count can be structurally zero-filled."""
        return {
            name: name in self.comparable_outcomes
            for name in self.native_outcomes
        }


def _odd_fips(state: str, count: int) -> frozenset[str]:
    """Census county codes that are exactly the documented odd-code sequence."""
    return frozenset(f"{state}{code:03d}" for code in range(1, count * 2, 2))


# These are source geography contracts, not counties observed in any crash
# extract.  The small exception lists cover states whose Census county codes do
# not follow the ordinary odd-code pattern (FL, NV, VA, WI).
_FL = (_odd_fips("12", 68) - {"12025", "12135"}) | {"12086"}
_NV = (_odd_fips("32", 17) - {"32025"}) | {"32510"}
_WI = _odd_fips("55", 71) | {"55078"}
_VA_SUFFIXES = """001 003 005 007 009 011 013 015 017 019 021 023 025 027 029 031 033 035 036 037 041 043 045 047 049 051 053 057 059 061 063 065 067 069 071 073 075 077 079 081 083 085 087 089 091 093 095 097 099 101 103 105 107 109 111 113 115 117 119 121 125 127 131 133 135 137 139 141 143 145 147 149 153 155 157 159 161 163 165 167 169 171 173 175 177 179 181 183 185 187 191 193 195 197 199 510 520 530 540 550 570 580 590 595 600 610 620 630 640 650 660 670 678 680 683 685 690 700 710 720 730 735 740 750 760 770 775 790 800 810 820 830 840""".split()
_VA = frozenset(f"51{suffix}" for suffix in _VA_SUFFIXES)


def _outcomes(*, crashes: str, person_fatals: str, serious: str) -> dict[str, str]:
    return {
        "crashes": crashes,
        "person_fatals": person_fatals,
        "serious_injury_persons": serious,
    }


STATE_SOURCE_SPECS: dict[str, StateSourceSpec] = {
    "CA": StateSourceSpec("CA_CCRS", "CA", "06", "state_year", frozenset(range(2016, 2025)), _odd_fips("06", 58), _outcomes(crashes="one crash record", person_fatals="NUMBERKILLED (person fatalities)", serious="NUMBERINJURED (all-injury proxy; no verified KABCO-A field)"), frozenset({"crashes", "person_fatals"}), "data.ca.gov CCRS annual crashes CSV"),
    "FL": StateSourceSpec("FL_FDOT", "FL", "12", "state_year", frozenset(range(2013, 2020)), frozenset(_FL), _outcomes(crashes="one crash record", person_fatals="NUMBER_OF_KILLED", serious="NUMBER_OF_SERIOUS_INJURIES"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "FDOT Crashes_All FeatureServer", frozenset({2019})),
    "IL": StateSourceSpec("IL_IDOT", "IL", "17", "state_year", frozenset(range(2016, 2025)), _odd_fips("17", 102), _outcomes(crashes="one crash record", person_fatals="TotalFatals", serious="AInjuries"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "IDOT annual Open Data item"),
    "IA": StateSourceSpec("IA_DOT", "IA", "19", "state_year", frozenset(range(2015, 2025)), _odd_fips("19", 99), _outcomes(crashes="one crash record", person_fatals="FATALITIES", serious="MAJINJURY"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "Iowa DOT SOR crash download"),
    "MA": StateSourceSpec("MA_MASSDOT", "MA", "25", "state_year", frozenset(range(2013, 2021)), _odd_fips("25", 14), _outcomes(crashes="one crash record", person_fatals="fatal-person count", serious="nonfatal injuries on incapacitating crash (proxy; no person-level severity count)"), frozenset({"crashes", "person_fatals"}), "MassDOT annual ArcGIS services"),
    "NV": StateSourceSpec("NV_NDOT", "NV", "32", "state_year", frozenset(range(2016, 2025)), frozenset(_NV), _outcomes(crashes="one crash record", person_fatals="Fatalities", serious="all injured on Injury_Type=A crash (proxy; no person-level severity count)"), frozenset({"crashes", "person_fatals"}), "Nevada NDOT CrashData_OpenData FeatureServer"),
    "NY": StateSourceSpec("NY_DOT", "NY", "36", "state_year", frozenset(range(2021, 2025)), _odd_fips("36", 62), _outcomes(crashes="one accident record", person_fatals="unavailable: fatal accident is not person count", serious="unavailable: injury accident is not serious-injury person count"), frozenset({"crashes"}), "NY Open Data e8ky-4vqe Socrata"),
    "OR": StateSourceSpec("OR_ODOT", "OR", "41", "state_year", frozenset(range(2019, 2025)), _odd_fips("41", 36), _outcomes(crashes="one crash record", person_fatals="TOT_FATAL_CNT", serious="TOT_INJ_LVL_A_CNT"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "ODOT OTSDE_Crash MapServer"),
    "TN": StateSourceSpec("TN_TDOT", "TN", "47", "state_year", frozenset(range(2021, 2025)), _odd_fips("47", 95), _outcomes(crashes="one crash record", person_fatals="TOTALKILLE", serious="TOTAL_INCA"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "TDOT crash FeatureServer"),
    "TX": StateSourceSpec("TX_TXDOT_CRIS", "TX", "48", "state_year", frozenset(range(2020, 2025)), _odd_fips("48", 254), _outcomes(crashes="crash_id", person_fatals="death_cnt", serious="sus_serious_injry_cnt"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "TxDOT CRIS FeatureServer"),
    "VA": StateSourceSpec("VA_VDOT", "VA", "51", "state_year", frozenset(range(2017, 2025)), _VA, _outcomes(crashes="one crash record", person_fatals="K_PEOPLE", serious="PERSONS_INJURED on A-severity crash (proxy; no person-level severity count)"), frozenset({"crashes", "person_fatals"}), "VDOT CrashData FeatureServer"),
    "WI": StateSourceSpec("WI_COMMUNITY_MAPS", "WI", "55", "county_year", frozenset(range(2013, 2025)), frozenset(_WI), _outcomes(crashes="one crash record", person_fatals="totfatl", serious="totinj on A-severity crashes (all-injury proxy; native serious-person count unverified)"), frozenset({"crashes", "person_fatals"}), "Wisconsin Community Maps county-year API"),
    # DSHS's Socrata "Public Crash Data" (827n-m6xc) is the current,
    # actively-maintained statewide source (consistent 31-37k crashes/year
    # 2013-2024, including a plausible COVID-era dip in 2020). This
    # supersedes an older DelDOT ArcGIS layer (DE_ODP_CRASH_DATA) that was
    # evidently abandoned after August 2017 -- confirmed both by a >98%
    # crash-volume cliff and by fatal-crash counts collapsing to single
    # digits, which rules out a reporting-threshold policy change.
    "DE": StateSourceSpec("DE_DSHS", "DE", "10", "state_year", frozenset(range(2013, 2025)), frozenset({"10001", "10003", "10005"}), _outcomes(crashes="one crash record", person_fatals="unavailable: crash_class is a crash-level flag, not a person-fatality count", serious="unavailable: no person-level serious-injury count"), frozenset({"crashes"}), "Delaware DSHS Public Crash Data (data.delaware.gov Socrata)"),
    # Source table only has Year>=2021; earlier years are structurally absent,
    # not excluded for a data-quality reason.
    "NC": StateSourceSpec("NC_NCDOT", "NC", "37", "state_year", frozenset(range(2021, 2025)), _odd_fips("37", 100), _outcomes(crashes="one crash record", person_fatals="NumFatalities", serious="NumAInjuries"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "NCDOT StatewideCrashTable FeatureServer"),
    # Source has a separate MapServer layer per year; earlier years are
    # structurally absent (not excluded for a data-quality reason).
    "UT": StateSourceSpec("UT_UDOT", "UT", "49", "state_year", frozenset(range(2018, 2025)), _odd_fips("49", 29), _outcomes(crashes="one crash record", person_fatals="NUMBER_FATALITIES", serious="NUMBER_FOUR_INJURIES"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "UDOT Crash_Locations MapServer"),
    # Restricted to 2015-2021: Connecticut retired its 8 counties for 9
    # planning regions in January 2022, and this project's per-state contract
    # assumes one fixed county universe across all requested years (the same
    # simplification already used for every other state). Extending past
    # 2021 would need bespoke dual-geography-regime handling, out of scope
    # for this addition.
    "CT": StateSourceSpec("CT_UCONN", "CT", "09", "state_year", frozenset(range(2015, 2022)), frozenset({
        "09001", "09003", "09005", "09007", "09009", "09011", "09013", "09015",
    }), _outcomes(crashes="one crash record", person_fatals="InjuryStatus = 'Fatal Injury (K)' on Person layer", serious="InjuryStatus = 'Suspected Serious Injury (A)' on Person layer"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "UConn CTDOT ConnecticutCrash FeatureServer"),
    # Sub-state (single-county) addition: no Maryland statewide feed was
    # found (MDOT SHA's public ArcGIS layers are fatal-crash-only; the old
    # data.maryland.gov statewide Socrata dataset is retired), but Montgomery
    # County publishes a live, actively-updated, crash-level Socrata source.
    # "state" is this county's own FIPS (24031), not a real 2-letter postal
    # code; see the _STATE_FIPS comment in crash_coverage.py.
    #
    # person_fatals/serious_injury_persons are structurally present (summed
    # from Drivers/Non-Motorists injury_severity, joined by report_number)
    # but NOT comparable: the FARS fatality-ratio review showed a consistent
    # 15-33% *undercount* every single year 2015-2024 (ratio 0.67-0.85, never
    # above 1.0) -- the signature of a police on-scene severity snapshot that
    # is not retroactively updated when a hospitalized victim dies days
    # later, unlike FARS's 30-day-death standard. This is a systematic
    # definitional gap, not noise, so (like NY and DE) only crashes is
    # treated as comparable; person_fatals/serious_injury_persons are
    # reported as NaN throughout rather than a biased count.
    "MOCO": StateSourceSpec("MOCO_MCPD", "MOCO", "24031", "state_year", frozenset(range(2015, 2026)), frozenset({"24031"}), _outcomes(crashes="one crash record (acrs_report_type)", person_fatals="unavailable: injury_severity='Fatal Injury' undercounts FARS by 15-33% every year (on-scene snapshot, no retroactive update for later hospital deaths)", serious="unavailable: same on-scene-snapshot limitation as person_fatals"), frozenset({"crashes"}), "Montgomery County MD Crash Reporting (data.montgomerycountymd.gov Socrata)"),
    # Statewide, but fatal-crash-only: the mirror image of NY/DE's
    # crashes-only contract (no all-crash denominator, no serious-injury
    # field). Confirmed live (recently edited per ArcGIS metadata) and
    # continuously active 2012-2024 (~80-115 fatal crashes/year, no volume
    # cliff anywhere in the range, checked year-by-year against the live
    # source, not just a page's claims).
    # 2012-2015, 2017, 2018 excluded: for each of these years, essentially
    # every record shares the exact same Crash_Date (confirmed directly
    # against the live source -- e.g. all 93 of 2015's records parse to
    # just 2 distinct dates, 12/30-12/31/2015; 2012/2013/2014/2017/2018 each
    # collapse to a *single* date for the entire year's 90-115 records).
    # This is not a per-crash date at all -- almost certainly a bulk-load
    # timestamp stamped across a whole year's backfilled records -- so it
    # cannot support a calendar county-day panel for those years, even
    # though the annual Crash_Year-level fatality totals themselves are
    # accurate (confirmed against FARS). 2020 separately excluded: a small
    # (1/81) but above-the-standard-1%-threshold share of rows whose parsed
    # Crash_Date falls in a different calendar year than Crash_Year (e.g. a
    # 2020-tagged crash with Crash_Date=2021-01-01), a source-side
    # year-tagging inconsistency. Only 2019 and 2021-2024 have genuine
    # per-crash date variance (record count roughly equals unique-date
    # count) and are kept.
    "HI": StateSourceSpec("HI_FATALCRASH", "HI", "15", "state_year", frozenset(range(2012, 2025)), frozenset({
        "15001", "15003", "15005", "15007", "15009",
    }), _outcomes(crashes="unavailable: source is fatal-crash-only, no all-crash denominator", person_fatals="Total_Fatalities (fatal crashes only, statewide)", serious="unavailable: no serious-injury field (fatal-crash-only source)"), frozenset({"person_fatals"}), "Hawaii statewide FatalCrash FeatureServer", frozenset({2012, 2013, 2014, 2015, 2016, 2017, 2018, 2020})),
    # Multi-county (8-county Indianapolis MPO region) addition: no Indiana
    # statewide crash-level feed was found. This MPO-published dataset
    # covers all 8 of its member counties -- verified genuinely
    # county-wide, not just the principal city, by checking that Marion
    # County's own records include Lawrence, Speedway, Beech Grove, and
    # Southport, not only Indianapolis proper (the same check that
    # disqualified a same-session Seattle/Kansas City candidate, whose
    # single-city-PD jurisdiction covered only a fraction of their nominal
    # county). Fatal/SSI-only (a crash-level severity flag, not a person
    # count): `person_fatals` as "count of Fatal-flagged crashes" tracked
    # FARS's true person-fatality count within 2-8% every year 2018-2024
    # (ratio 0.92-1.01, checked directly against the live source before
    # building), an acceptable proxy given no per-crash fatality count
    # exists; `serious` (SSI-flagged crash count) is an analogous proxy,
    # unvalidated against an independent benchmark, same caveat as VA/WI's
    # injury-count proxies.
    "INMPO": StateSourceSpec("INMPO_ARCGIS", "INMPO", "18", "state_year", frozenset(range(2018, 2025)), frozenset({
        "18011", "18057", "18059", "18063", "18081", "18097", "18109", "18145",
    }), _outcomes(crashes="unavailable: source only includes Fatal/SSI-severity crashes, no all-crash denominator", person_fatals="count of Incapacitated_Fatal='Fatal' crashes (crash-level flag; verified within 2-8% of FARS every year)", serious="count of Incapacitated_Fatal='SSI' crashes (crash-level flag, all-injury proxy; unvalidated against an independent benchmark)"), frozenset({"person_fatals", "serious_injury_persons"}), "Indianapolis MPO 8-county Fatal/SSI Crash Data FeatureServer"),
    # Multi-county (2-county: Ada, Canyon) COMPASS/ITD region addition: no
    # Idaho statewide crash-level feed was found. Verified genuinely
    # county-wide (not a single-city-PD jurisdiction) by checking that both
    # counties' records span all their member cities and multiple reporting
    # agencies (Ada: Boise, Meridian, Eagle, Garden City, Kuna, Star, via
    # Boise PD/Ada Co Sheriff/Meridian PD/Idaho State Police/Garden City PD;
    # Canyon: Nampa, Caldwell, Middleton, Parma, and 7 more). Has a genuine
    # per-crash fatality COUNT (unlike Indianapolis's severity flag) and a
    # KABCO severity classification enabling a serious-injury proxy;
    # `person_fatals` matched FARS almost exactly (0.95-1.00 ratio, checked
    # directly against the live source for 5 sample years before building).
    "IDCOMPASS": StateSourceSpec("ID_COMPASS", "IDCOMPASS", "16", "state_year", frozenset(range(2013, 2025)), frozenset({
        "16001", "16027",
    }), _outcomes(crashes="one crash record", person_fatals="fatalities", serious="injuries on severity='A Injury Accident' crashes (all-injury proxy; no verified person-level KABCO-A count)"), frozenset({"crashes", "person_fatals", "serious_injury_persons"}), "COMPASS/ITD 2-county (Ada, Canyon) Idaho CrashData FeatureServer"),
}


def get_spec(state_or_spec: str | StateSourceSpec) -> StateSourceSpec:
    if isinstance(state_or_spec, StateSourceSpec):
        return state_or_spec
    try:
        return STATE_SOURCE_SPECS[str(state_or_spec).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown state source {state_or_spec!r}") from exc


def filter_to_requested_years(frame: pd.DataFrame, *, state: str, date_column: str) -> pd.DataFrame:
    """Return only source-contract years after a builder has parsed its dates.

    This is intentionally separate from validation: a bulk extract can contain
    older rows, but they must never reach the sparse event output.
    """
    spec = get_spec(state)
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    return frame.loc[dates.dt.year.isin(spec.requested_years)].copy()


def _diagnostic_failure(result: CoverageResult, *reasons: str) -> CoverageResult:
    failures = tuple(dict.fromkeys((*result.failure_reasons, *reasons)))
    return CoverageResult(**{**result.to_mapping(), "coverage_valid": not failures, "failure_reasons": failures})


def validate_state_year(state_or_spec: str | StateSourceSpec, year: int, /, **diagnostics: object) -> CoverageResult:
    """Create a coverage row from request diagnostics for one state-year.

    It intentionally does not read a final parquet; a sparse parquet cannot
    reveal whether a page failed upstream.
    """
    spec = get_spec(state_or_spec)
    result = validate_reporting_unit(
        source=spec.source, state=spec.state, year=year, county_fips=None,
        expected_records=diagnostics.get("expected_records"),
        fetched_records=int(diagnostics.get("fetched_records", 0)),
        retained_records=int(diagnostics.get("retained_records", 0)),
        duplicate_records=int(diagnostics.get("duplicate_records", 0)),
        invalid_date_count=int(diagnostics.get("invalid_date_count", 0)),
        invalid_geography_count=int(diagnostics.get("invalid_geography_count", 0)),
        unresolvable_geography_count=int(diagnostics.get("unresolvable_geography_count", 0)),
        unresolvable_date_count=int(diagnostics.get("unresolvable_date_count", 0)),
        unresolvable_outcome_count=int(diagnostics.get("unresolvable_outcome_count", 0)),
        request_complete=bool(diagnostics.get("request_complete", False)),
        terminal_error=diagnostics.get("terminal_error"),
        required_columns_ok=bool(diagnostics.get("required_columns_ok", False)),
        observed_min_date=diagnostics.get("observed_min_date"),
        observed_max_date=diagnostics.get("observed_max_date"),
        source_url=str(diagnostics.get("source_url", spec.query_identifier)),
        source_checksum=diagnostics.get("source_checksum"),
    )
    reasons = []
    if year not in spec.requested_years:
        reasons.append("unrequested_year")
    if year in spec.excluded_years:
        reasons.append("excluded_source_year")
    if int(diagnostics.get("negative_outcome_count", 0)):
        reasons.append("negative_outcomes")
    return _diagnostic_failure(result, *reasons)


def validate_wisconsin_county_year(county_fips: str, year: int, /, *, response_kind: Literal["success", "empty", "failed"], **diagnostics: object) -> CoverageResult:
    """Validate one of the required 72 x year Wisconsin API requests."""
    spec = STATE_SOURCE_SPECS["WI"]
    county_fips = str(county_fips).zfill(5)
    if response_kind == "empty":
        diagnostics = {**diagnostics, "expected_records": 0, "fetched_records": 0, "retained_records": 0, "request_complete": True, "required_columns_ok": True}
    elif response_kind == "failed":
        diagnostics = {**diagnostics, "expected_records": diagnostics.get("expected_records"), "fetched_records": diagnostics.get("fetched_records", 0), "retained_records": diagnostics.get("retained_records", 0), "request_complete": False, "required_columns_ok": bool(diagnostics.get("required_columns_ok", True)), "terminal_error": diagnostics.get("terminal_error", "request_failed")}
    elif response_kind != "success":
        raise ValueError("response_kind must be success, empty, or failed")
    result = validate_reporting_unit(
        source=spec.source, state="WI", year=year, county_fips=county_fips,
        expected_records=diagnostics.get("expected_records"), fetched_records=int(diagnostics.get("fetched_records", 0)), retained_records=int(diagnostics.get("retained_records", 0)), duplicate_records=int(diagnostics.get("duplicate_records", 0)), invalid_date_count=int(diagnostics.get("invalid_date_count", 0)), invalid_geography_count=int(diagnostics.get("invalid_geography_count", 0)), request_complete=bool(diagnostics.get("request_complete", False)), terminal_error=diagnostics.get("terminal_error"), required_columns_ok=bool(diagnostics.get("required_columns_ok", False)), observed_min_date=diagnostics.get("observed_min_date"), observed_max_date=diagnostics.get("observed_max_date"), source_url=str(diagnostics.get("source_url", spec.query_identifier)), source_checksum=diagnostics.get("source_checksum"),
    )
    reasons = []
    if county_fips not in spec.expected_county_fips:
        reasons.append("unexpected_county_fips")
    if year not in spec.requested_years:
        reasons.append("unrequested_year")
    return _diagnostic_failure(result, *reasons)


def strict_arcgis_dataframe(session: object, *, url: str, where: str, expected_count: int, id_field: str, out_fields: str, page_size: int = 2_000, order_by_field: str | None = None) -> pd.DataFrame:
    """Use the common strict pager and return a raw DataFrame or raise."""
    requested = [field.strip().lower() for field in out_fields.split(",")]
    if id_field.lower() not in requested and out_fields != "*":
        out_fields = f"{out_fields},{id_field}"
    return pd.DataFrame(fetch_arcgis_pages(session, url=url, where=where, expected_count=expected_count, id_field=id_field, order_by_field=order_by_field, out_fields=out_fields, page_size=page_size))


def strict_socrata_dataframe(session: object, *, url: str, where: str, id_field: str, page_size: int = 50_000) -> pd.DataFrame:
    """Use Socrata count-first strict pagination and return a raw DataFrame."""
    return pd.DataFrame(fetch_socrata_pages(session, url=url, where=where, id_field=id_field, page_size=page_size))


def validate_bulk_extract(frame: pd.DataFrame, *, year: int, required_columns: set[str], date_column: str, source_checksum: str | None) -> dict[str, object]:
    """Return builder diagnostics for a downloaded annual bulk extract.

    A nonempty file that lacks a required native field or only contains dates
    outside its claimed year is invalid; callers pass this mapping directly to
    ``validate_state_year`` and write the resulting manifest even on failure.
    """
    columns = {str(column).strip().upper() for column in frame.columns}
    required = {column.upper() for column in required_columns}
    dates = pd.to_datetime(frame.get(date_column), errors="coerce") if date_column in frame else pd.Series(dtype="datetime64[ns]")
    in_year = dates.dt.year.eq(year) if not dates.empty else pd.Series(dtype=bool)
    return {
        "expected_records": len(frame), "fetched_records": len(frame),
        "retained_records": int(in_year.sum()) if len(in_year) else 0,
        "request_complete": source_checksum is not None,
        "required_columns_ok": required.issubset(columns),
        "invalid_date_count": int(dates.isna().sum()) if len(dates) else len(frame),
        "source_checksum": source_checksum,
        "observed_min_date": dates.loc[in_year].min() if len(in_year) and in_year.any() else None,
        "observed_max_date": dates.loc[in_year].max() if len(in_year) and in_year.any() else None,
    }


def validate_source_frame(
    state: str,
    year: int,
    frame: pd.DataFrame | None,
    *,
    required_columns: set[str],
    date_column: str,
    outcome_columns: set[str],
    source_checksum: str | None = None,
    terminal_error: object | None = None,
    date_unit: str | None = None,
    column_aliases: Mapping[str, tuple[str, ...]] | None = None,
    geography_column: str | None = None,
    geography_mapper: Mapping[object, str] | Callable[[object], str | None] | None = None,
    unresolvable_geography_values: frozenset[str] | None = None,
) -> CoverageResult:
    """Validate a downloaded state-year before its sparse aggregate is used.

    This is deliberately called on the raw response, so a failed download is
    represented as an invalid manifest row rather than silently disappearing
    when no county-day rows are produced.
    """
    if frame is None:
        failure = (fetch_failure_diagnostics(terminal_error)
                   if isinstance(terminal_error, BaseException)
                   else {"expected_records": None, "fetched_records": 0,
                         "retained_records": 0, "request_complete": False})
        if "terminal_error" not in failure:
            failure["terminal_error"] = terminal_error or "fetch_failed"
        return validate_state_year(
            state, year, required_columns_ok=False, source_checksum=source_checksum,
            **failure,
        )
    normalized = frame.copy()
    # Canonicalize only explicitly declared, supported raw aliases.  This makes
    # the source schema contract auditable instead of accepting fuzzy matches.
    for canonical, aliases in (column_aliases or {}).items():
        if canonical in normalized.columns:
            continue
        actual = next((name for name in aliases if name in normalized.columns), None)
        if actual is not None:
            normalized[canonical] = normalized[actual]
    columns = {str(column).strip().upper() for column in normalized.columns}
    required = {column.upper() for column in required_columns}
    date_actual = next((column for column in normalized.columns if str(column).strip().upper() == date_column.upper()), None)
    dates = (pd.to_datetime(normalized[date_actual], unit=date_unit, errors="coerce")
             if date_actual else pd.Series(pd.NaT, index=normalized.index))
    in_year = dates.dt.year.eq(year)
    negative_row = pd.Series(False, index=normalized.index)
    for expected in outcome_columns:
        actual = next((column for column in normalized.columns if str(column).strip().upper() == expected.upper()), None)
        if actual is not None:
            negative_row |= pd.to_numeric(normalized[actual], errors="coerce") < 0
    negative_count = int(negative_row.sum())
    # A negative crash/fatality/injury count can never be a legitimate
    # observation -- it is always corrupt data, unlike an unmapped geography
    # or missing date, which can be a genuine source limitation. A single
    # stray row is excluded from the panel without failing the reporting
    # unit; a large share instead signals a real pipeline bug (wrong column,
    # sign flip) and must still fail loudly.
    unresolvable_outcome = 0
    negative = negative_count
    if negative_count and len(normalized) and negative_count / len(normalized) <= 0.01:
        unresolvable_outcome = negative_count
        negative = 0
    invalid_geography = 0
    unresolvable_geography = 0
    geo_excluded = pd.Series(False, index=normalized.index)
    if geography_column and geography_mapper is not None:
        geo_actual = next((column for column in normalized.columns if str(column).strip().upper() == geography_column.upper()), None)
        if geo_actual is None:
            invalid_geography = len(normalized)
            geo_excluded = pd.Series(True, index=normalized.index)
        else:
            mapped = normalized[geo_actual].map(geography_mapper)
            bad = mapped.isna() | ~mapped.isin(get_spec(state).expected_county_fips)
            geo_excluded = bad
            # A raw geography field that is genuinely null (the crash simply
            # has no county recorded), or an explicit source-documented
            # placeholder token (for example NY's literal county_name of
            # "UNKNOWN", opted in via ``unresolvable_geography_values``), is a
            # genuinely unresolvable record, not a mapping-table gap: it is
            # still dropped from the panel, but a small bounded residual does
            # not by itself fail the reporting unit. Any other unmapped,
            # non-null value remains a hard failure, since it may be a real
            # county our mapping table is missing.
            raw_geo = normalized[geo_actual]
            is_null_source = raw_geo.isna()
            if unresolvable_geography_values:
                raw_upper = raw_geo.astype(str).str.strip().str.upper()
                is_null_source = is_null_source | raw_upper.isin(unresolvable_geography_values)
            is_known_unresolvable = bad & is_null_source
            unresolvable_geography = int(is_known_unresolvable.sum())
            invalid_geography = int((bad & ~is_known_unresolvable).sum())
    # ``retained_records`` must reflect rows that would actually reach the
    # balanced panel -- in the requested year *and* geography-resolvable --
    # so it reconciles the same way for every source, including the small
    # bounded unresolvable-geography exclusion.
    retained = in_year & ~geo_excluded & ~negative_row
    unresolvable_date = 0
    invalid_date = int(dates.isna().sum())
    if date_actual is not None:
        # A source date field that is genuinely null/empty (the crash simply
        # was never dated in the source system) is a distinct, evidence-bound
        # case from a non-null value that fails to parse (which suggests a
        # real schema or corruption problem and stays a hard failure).
        raw_date = normalized[date_actual]
        date_is_null_source = raw_date.isna() | (raw_date.astype(str).str.strip().isin(("", "None", "nan", "NaT")))
        unresolvable_date = int((dates.isna() & date_is_null_source).sum())
        invalid_date = int((dates.isna() & ~date_is_null_source).sum())
        # A validly-parsed date just outside the requested year is not a
        # missing or corrupt date -- it is a boundary artifact of using an
        # approximate query-partitioning field (a source-provided ``year``
        # tag, rather than a precise date range) to fetch one year at a time.
        # Observed: Delaware's Socrata `year` field disagrees with the parsed
        # calendar year of `crash_datetime` for a handful of records per year
        # (likely a UTC/local timezone boundary effect); a small bounded
        # residual is excluded rather than failing the whole reporting unit.
        out_of_year = dates.notna() & ~in_year
        out_of_year_count = int(out_of_year.sum())
        if out_of_year_count and len(normalized) and out_of_year_count / len(normalized) <= 0.01:
            unresolvable_date += out_of_year_count
    return validate_state_year(
        state, year, expected_records=len(frame), fetched_records=len(frame),
        retained_records=int(retained.sum()), request_complete=True,
        required_columns_ok=required.issubset(columns),
        invalid_date_count=invalid_date,
        unresolvable_date_count=unresolvable_date,
        invalid_geography_count=invalid_geography,
        unresolvable_geography_count=unresolvable_geography,
        negative_outcome_count=negative, unresolvable_outcome_count=unresolvable_outcome,
        source_checksum=source_checksum,
        observed_min_date=dates.loc[in_year].min() if in_year.any() else None,
        observed_max_date=dates.loc[in_year].max() if in_year.any() else None,
    )


def fetch_failure_diagnostics(error: BaseException) -> dict[str, object]:
    """Convert a strict-pager exception into lossless manifest diagnostics."""
    if isinstance(error, IncompleteDownloadError):
        return {"expected_records": error.expected_count, "fetched_records": error.fetched_count,
                "retained_records": 0, "request_complete": False,
                "terminal_error": error.terminal_error}
    return {"expected_records": None, "fetched_records": 0, "retained_records": 0,
            "request_complete": False, "terminal_error": str(error)}


def write_state_manifest_or_raise(state: str, rows: list[CoverageResult], *, output_dir: str = "data/processed/coverage") -> None:
    """Persist every requested unit's diagnostics, then fail closed if needed."""
    spec = get_spec(state)
    write_manifest(rows, output_dir, filename=f"{spec.state.lower()}_coverage")
    failed = [row.year for row in rows if not row.coverage_valid]
    if failed:
        raise RuntimeError(f"{spec.source} coverage validation failed for years {failed}")
