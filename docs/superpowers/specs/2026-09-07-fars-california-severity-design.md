# FARS-plus-California Severity Design

## Goal

Add a defensible severity-separated California crash panel while retaining
the canonical nationwide FARS fatality panel as the primary national outcome.

## Scope and contracts

- Nationwide FARS remains the 50-state-plus-DC fatal-crash census. Its canonical
  outcomes remain `fatal_crashes`, `person_fatals`, `drunk_fatals`,
  `sober_fatals`, and `weather_adverse`; the pipeline must not infer all-crash
  or all-serious-injury counts from FARS.
- California CCRS is restricted to 2016--2022 for the hourly alert analysis,
  matching the existing validated alert window. The source is the annual
  `Crashes_YEAR.csv` file joined to `InjuredWitnessPassengers_YEAR.csv` by
  collision ID.
- A collision is assigned exactly one severity class, in priority order:
  `fatal` if `NumberKilled > 0` or any non-witness injury row has
  `ExtentOfInjuryCode == Fatal`; otherwise `serious` if any non-witness injury
  row has `ExtentOfInjuryCode` in the reviewed serious set
  (`SevereInactive`, `SuspectSerious`); otherwise `nonserious`.
- Witness-only injury rows do not classify the collision. Unknown or missing
  injury codes remain nonserious only when no fatal/serious evidence exists and
  are counted in diagnostics; they are not silently treated as serious.
- Every retained collision contributes exactly one to `ca_crashes` and exactly
  one to one of `ca_fatal_crashes`, `ca_serious_crashes`, or
  `ca_nonserious_crashes`. `ca_fatals` remains the person-fatality count from
  the crash file for validation and comparability.

## Outputs and provenance

- `data/processed/california_ccrs_severity_county_day.parquet`
- `data/processed/california_ccrs_severity_county_hour.parquet`
- `data/processed/coverage/ca_severity_coverage.parquet`
- A raw cache under `data/raw/california_ccrs/<year>/` with SHA-256 sidecars;
  the cache is untracked and never replaces the legacy crash-only panel.
- The severity runner reads the new hourly panel and estimates the three
  mutually exclusive crash outcomes. FARS hourly analyses continue to use the
  validated FARS panel and are reported separately.

## Failure handling

- Missing crash or injury files, missing required columns, invalid dates, or
  unmapped counties produce a failed coverage row and stop the build rather
  than silently dropping a year.
- Injury rows whose collision IDs do not match a retained crash are recorded
  as diagnostics and excluded from classification.
- Duplicate crash IDs are resolved deterministically by the greatest report
  version when available; the count is recorded in the coverage diagnostics.

## Testing

- Unit tests cover severity priority, witness exclusion, unknown-code
  diagnostics, duplicate resolution, county/hour aggregation, and required
  schema checks using in-memory fixtures.
- Existing FARS tests must remain green. The California severity runner must
  fail clearly when the new hourly panel is absent and must not fall back to
  the legacy all-injury proxy.
