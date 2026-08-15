# Validated Zero-Balanced Crash Panels Design

## Goal

Produce defensible zero-balanced county-day crash panels by distinguishing complete event reporting from missing or partial source coverage. Use national FARS as the primary fatality source and validated state DOT records for all-crash and serious-injury outcomes.

## Scope

This work covers:

- a national 2013–2024 FARS fatality panel for the 50 states and District of Columbia;
- strict completeness validation for the 12 state DOT builders used by the commuter-share analysis;
- balanced county-date expansion only inside validated reporting units;
- cross-source state fatality validation against FARS;
- regenerated commuter-share regression outputs and documentation.

This work does not use CRSS or NASS/GES as county-day outcomes. They are probability samples rather than censuses and cannot identify public county-level event counts. It also does not claim that FARS measures all crashes or all serious injuries.

## Source roles

### FARS

FARS is the national census source for qualifying fatal traffic crashes. Annual national accident files provide crash date, county, and fatalities. The pipeline will use FARS for the primary national fatality panel and for validating comparable person-fatality fields in state DOT sources.

The implementation will:

- download annual national files for 2013–2024;
- fail the build if any requested year is unavailable or incomplete;
- retain a single canonical FARS output schema;
- reject unknown or invalid county codes, including county code `000` and `999`;
- use a complete Census county/equivalent universe rather than counties observed in fatal crashes;
- exclude Puerto Rico from the primary 50-state-plus-DC analysis while retaining a documented option to include it;
- address Connecticut's transition from legacy counties to planning-region county equivalents with a documented longitudinal geography policy.

### State DOT sources

State DOT sources remain necessary for all police-reported crashes and serious injuries. Each source consists of crash-level event records that currently aggregate only observed crash-active county-days. A missing county-date may be converted to zero only after the associated reporting unit has passed completeness validation.

The reporting unit is state-year for CA, FL, IL, IA, MA, NV, NY, OR, TN, TX, and VA. Wisconsin uses county-year because its API is queried separately for each county and year.

## Architecture

### 1. Coverage manifest

Every builder will emit a machine-readable coverage manifest alongside its processed parquet. The manifest will contain one row per reporting unit with:

- source and state;
- requested year and, for Wisconsin, requested county FIPS;
- source URL or dataset identifier;
- request completion status;
- source-reported expected record count when available;
- number of records fetched;
- number of records retained after parsing;
- duplicate record count;
- invalid-date count;
- unmapped or invalid-geography count;
- observed minimum and maximum event dates;
- number of distinct calendar dates represented statewide;
- number of distinct valid county/equivalent FIPS represented;
- source-specific schema checks;
- validation status and explicit failure reasons;
- source file checksum when a bulk file is downloaded.

Manifests will be written under `data/processed/coverage/`. They are required inputs to panel balancing and regression runners.

### 2. Download completeness rules

A reporting unit is valid only if all applicable checks pass:

- the request completed without an unhandled or terminal page error;
- fetched records equal the source count query when the source exposes a count;
- pagination reaches the expected count and never terminates merely because of an empty or short page before that count;
- annual bulk files download successfully and have a recorded checksum;
- event dates parse successfully or excluded date errors are enumerated;
- records lie inside the requested year;
- geography codes map to the expected universe or every exclusion is enumerated;
- required outcome columns exist and have valid nonnegative values;
- the unit spans the expected closed-year boundaries, subject to source-specific documented exceptions;
- implausible discontinuities in annual record totals trigger review rather than automatic acceptance.

A failed unit remains missing. Builders must not silently save partial accumulated pages as complete data.

### 3. County universes and geography

The national FARS panel will use an explicit Census county/equivalent universe for each year. State panels will use explicit state-specific universes checked against Census geography and the source's jurisdiction definitions.

Independent cities and county equivalents will be handled explicitly. Virginia towns rolled into parent counties will retain the builder's documented mapping. Connecticut requires one harmonized policy across the study window: the preferred policy is a stable legacy-county geography with planning-region events crosswalked back to legacy counties when a defensible geographic crosswalk is available. If no lossless or population-weighted crosswalk is accepted, Connecticut will be excluded from the national longitudinal analysis and documented as such.

Never derive the county universe from counties that happen to contain an observed crash.

### 4. Zero-balanced panel construction

Panel construction will be a side-effect-free shared function receiving:

- sparse event aggregates;
- a validated coverage manifest;
- an explicit county universe;
- outcome availability metadata.

For each valid reporting unit, it will construct every county-date combination and left-join observed outcomes. Absent event rows become structural zeros only for outcomes the source actually measures. Examples:

- FARS: zero-fill fatal crashes and person fatalities; do not synthesize all crashes or all serious injuries.
- New York: zero-fill all crashes and fatal-crash/injury-crash measures; keep person fatalities and serious injuries missing.
- States with valid person-fatality and serious-injury fields: zero-fill those outcomes within validated units.
- Failed or unvalidated reporting units: omit the unit or retain outcomes as missing; never convert it to zero.

The balanced output will contain provenance columns such as `coverage_valid`, `coverage_unit`, `structural_zero`, and `source`.

### 5. Outcome-definition audit

Before rebuilding results, every native state outcome will be mapped to a documented comparable concept:

- crashes;
- person fatalities;
- serious-injury persons under a stated severity definition;
- noncomparable proxy or unavailable.

California's current serious-injury construction and fatality mapping require correction against the native schema. Wisconsin's serious-injury construction must distinguish seriously injured persons from all injured persons in A-severity crashes. New York fatal crashes remain separate from person fatalities.

Noncomparable measures will remain missing in pooled models and may be reported separately under explicit labels.

### 6. FARS/state cross-validation

For each state-year with a comparable person-fatality measure, compare state DOT fatalities with FARS using:

- annual totals and ratio;
- county-year totals and correlation;
- county-date agreement on fatal-event days;
- unmatched geography and date counts.

The first implementation will not hard-code a universal acceptance ratio without evidence. It will produce a validation report and require an explicit reviewed allowlist of state-years. Obvious failures, such as the current California construction and materially incomplete Florida or Illinois years, will be excluded until corrected.

Validation outputs will be written under `output/tables/` and will state why each state-year is accepted or rejected.

### 7. Analysis integration

The commuter-share analysis will consume only balanced panels whose manifest units are valid.

- National FARS is the primary fatality analysis.
- State DOT data provide all-crash and serious-injury analyses for accepted state-years.
- National and state-source estimates remain separate.
- PPML uses raw nonnegative counts with `log(population)` as an exposure offset and retains structural zeros.
- WLS uses rates per 100,000 with population weights.
- County and calendar-date fixed effects remain.
- Direct treatment and commuter-share spillover remain jointly estimated.
- Regression outputs include expected model counts, fit status, input N, fitted N, zero share, and failure reason.

The preferred runner will fail fast if required manifests, commuting weights, or validated panels are absent.

## Outputs

The rebuild will produce:

- canonical national FARS sparse events and balanced county-day panel;
- per-source coverage manifests;
- validated balanced state DOT panels;
- state/FARS fatality comparison table;
- accepted and rejected state-year coverage table;
- regenerated commuter-share regression and descriptive tables;
- structured model-status diagnostics;
- an updated Markdown results summary recording data versions, checksums, code commit, exclusions, and warnings.

Existing outputs will not be overwritten until the validated rebuild and its tests pass. Corrected outputs will use distinct filenames during development.

## Error handling

- Network or pagination failure invalidates only the affected reporting unit but causes the strict aggregate build command to exit nonzero.
- Partial data may be retained in a staging location for diagnosis but cannot enter validated panels.
- Embedded API error responses are errors, not empty datasets.
- Empty successful responses require source-specific evidence before being classified as genuine zero-event units.
- Unexpected schemas, negative counts, invalid dates, or unknown geography codes fail validation.
- Nonfinite regression estimates are recorded as failed fits and excluded from interpreted results.

## Testing strategy

Unit tests will cover:

- complete and incomplete pagination;
- fetched-versus-expected reconciliation;
- invalid dates and geographies;
- genuine empty units versus failed requests;
- county-universe expansion;
- zero filling only inside validated units;
- preservation of unavailable outcomes as missing;
- FARS invalid county-code removal;
- New York noncomparability;
- Connecticut geography policy;
- nonfinite estimator inference and structured fit failures.

Integration tests will use miniature fixture downloads to build manifests, sparse events, balanced panels, and regression inputs end to end. A final audit will assert that:

- valid count outcomes contain structural zeros;
- no unvalidated reporting unit enters regression samples;
- FARS annual fatality totals reconcile with the downloaded files;
- expected and produced model counts reconcile;
- all delivered tables and logs are nonempty and reproducible.

## Migration and reproducibility

The two existing FARS writers that target the same parquet will be consolidated or given distinct outputs so incompatible schemas cannot overwrite one another. Dependency versions will be pinned. Downloaded bulk files and generated manifests will record checksums. The final result note will identify the exact clean commit used; analyses from a dirty working tree will be labeled as such and not promoted as canonical.

## Primary references

- NHTSA FARS: https://www.nhtsa.gov/crash-data-systems/fatality-analysis-reporting-system
- NHTSA CRSS: https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system
- Census TIGER/Line county geography: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2024.html
- Census Connecticut county-equivalent change note: https://www.census.gov/programs-surveys/acs/technical-documentation/user-notes/2023-01.html
