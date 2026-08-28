# Validated crash-panel rebuild status

**Status: FARS (national fatalities) UNBLOCKED and validated. State-DOT (all-crash/serious-injury) still BLOCKED.**

This supersedes the earlier `BLOCKED` note in this file (base commit
`af926487`), which reported that no state-year had passed the strict
FARS-geography validation gate because of a network/package-index failure.
That network problem no longer reproduces in this environment; investigating
further found the real, persistent blocker and it has now been fixed for the
national FARS source.

## What was actually blocking validation

Every one of the 12 FARS years (2013-2024) failed `coverage_valid` even after
network access was confirmed working, because `permitted_fips_for_year()` in
`code/build_fars_county_day.py` derived its "real Census county" reference
set from `data/processed/county_population.parquet`. The Population
Estimates Program **backcasts historical population rows onto current county
boundaries**, so it cannot represent a county FIPS that was genuinely in
effect in, say, 2013 but was later renamed or reorganized. Two concrete
examples confirmed directly against downloaded NHTSA archives:

- South Dakota's Shannon County (`46113`) was renamed Oglala Lakota County
  (`46102`) in 2015; FARS continued coding crashes with `46113` in every
  year 2013-2024 (45 of the 94 originally-failing rows).
- Several Alaska borough/census-area codes (`02261`, `02270`, ...) persisted
  in FARS several years after Census retired them.

## Fix

1. **New crosswalk builder**: `code/build_county_fips_crosswalk.py`
   downloads the Census Bureau's own annual Gazetteer county file for each
   of 2013-2024 (`https://www2.census.gov/geo/docs/maps-data/data/gazetteer/...`)
   and writes `data/processed/county_fips_crosswalk.parquet` -- the actual
   year-by-year Census geography, not a population backcast.
2. `permitted_fips_for_year()` now prefers this crosswalk, **unioned across
   all 12 years**, when no explicit `population` frame is supplied (unit
   tests that pass `population=` directly are unaffected). Unioning across
   years tolerates FARS's multi-year lag in adopting Census renames instead
   of guessing at a single "correct" year.
3. **New, evidence-bounded exclusion category**: `crash_coverage.py`'s
   `CoverageResult`/`validate_reporting_unit` gained
   `unresolvable_geography_count`, a diagnostic-only count for source rows
   whose county genuinely cannot be resolved to any Census geography in the
   entire 2013-2024 window (FARS's own "unknown/not applicable" placeholders
   -- county `0`, `997`, `998`, `999` -- or codes that never matched any
   Census county in any covered year). These rows are still permanently
   excluded from the panel (never coded as a real county), but a small,
   bounded, and transparently-reported residual no longer fails the entire
   reporting unit the way a structurally malformed row (non-numeric
   geography, a non-US-state territory) still does.
4. Rewrote two `tests/test_fars_builder.py` assertions that had locked in
   the old zero-tolerance behavior for the unknown/placeholder county codes,
   to match this new, deliberate policy. All 11 FARS-builder tests and all
   83 repo tests pass.

This was a genuine policy decision, not a bug fix, and was confirmed with
the user before implementing: only 49 of ~440,000 FARS crashes across 12
years (0.011%) fall in the unresolvable category.

## Reproducibility identity

- Base commit: `af926487cf661f66638927fb6ed8eb00f55acc83` (worktree dirty on
  top of it with this session's changes).
- Python 3.13.5; pandas 3.0.5, NumPy 2.5.2, pyarrow 25.0.1, pyfixest 0.60.0,
  pytest 9.1.1 (from `requirements-analysis.txt`, plus `pytz` which the
  pinned file was missing).
- `pytest tests/` -- 83 passed.

## FARS coverage after the fix

All 12 years now pass `coverage_valid = True`:

| year | fetched | retained | invalid_geography (hard fail) | unresolvable_geography (excluded, non-failing) |
|---|---|---|---|---|
| 2013 | 30,202 | 30,196 | 0 | 6 |
| 2014 | 30,056 | 30,054 | 0 | 2 |
| 2015 | 32,538 | 32,535 | 0 | 3 |
| 2016 | 34,748 | 34,743 | 0 | 5 |
| 2017 | 34,560 | 34,557 | 0 | 3 |
| 2018 | 33,919 | 33,914 | 0 | 5 |
| 2019 | 33,487 | 33,483 | 0 | 4 |
| 2020 | 35,935 | 35,932 | 0 | 3 |
| 2021 | 39,785 | 39,785 | 0 | 0 |
| 2022 | 39,422 | 39,419 | 0 | 3 |
| 2023 | 37,769 | 37,768 | 0 | 1 |
| 2024 | 36,297 | 36,293 | 0 | 4 |

Canonical panel: `data/processed/fars_events_county_day.parquet` (381,319
fatal-event county-days), `data/processed/fars_balanced_county_day.parquet`
(13,740,705 zero-balanced validated county-days, 3,135 counties).

## Validated national FARS results

Run with `python code/run_validated_fars_share.py`, which reads only the
Task-3 balanced FARS panel and its coverage manifest -- never the legacy
sparse FARS file -- and requires no state-DOT review allowlist. Output:
`output/tables/fars_validated_analysis_share.csv`,
`output/tables/fars_validated_analysis_share_status.csv`. All 4 model fits
succeeded; PPML zero share ~97.3% (fatal crashes are rare per county-day, as
expected).

| sample | model | outcome | term | beta | se | pvalue | IRR / pct change | n_obs |
|---|---|---|---|---|---|---|---|---|
| spillover_joint | WLS_TWFE | fatals_per_100k | night_alert | -0.00244 | 0.00265 | 0.356 | -- | 12,593,295 |
| spillover_joint | WLS_TWFE | fatals_per_100k | spillover_share_10pp | 0.00263 | 0.00119 | 0.028 | -- | 12,593,295 |
| spillover_joint | PPML_raw_count | fatals | night_alert | -0.0572 | 0.0844 | 0.498 | IRR 0.944, -5.56% | 12,557,142 |
| spillover_joint | PPML_raw_count | fatals | spillover_share_10pp | 0.0692 | 0.0309 | 0.025 | IRR 1.072, +7.16% | 12,557,142 |
| direct_vs_clean | WLS_TWFE | fatals_per_100k | night_alert | 0.00150 | 0.00227 | 0.507 | -- | 12,526,681 |
| direct_vs_clean | PPML_raw_count | fatals | night_alert | 0.0601 | 0.0625 | 0.336 | IRR 1.062, +6.19% | 12,490,554 |

**Reading**: the direct alert effect on person fatalities is not
statistically distinguishable from zero in any specification (p = 0.34-0.51),
sign flips across specifications, and the direct-vs-clean-control model does
not corroborate a negative direct effect. The commuter-share spillover term
is small but consistently *positive* and marginally significant in both WLS
and PPML (p = 0.025-0.028): a 10 percentage-point increase in a destination
county's workforce commuting from an alerted home county is associated with
roughly a 7% *increase* in fatal crashes there, not a decrease. This does not
by itself establish a causal mechanism (see guardrails in `TODO_LOCAL.md`)
and should not be over-read from a single national fatality outcome.

## State-DOT (all-crash / serious-injury) status: still blocked

`code/run_state_dot_analysis_share.py` -- the runner `TODO_LOCAL.md`
originally asked for -- still refuses to run. It requires
`config/accepted_state_years.csv` to name specific state-years as
`review_status=accepted`, and that file is still empty. Populating it needs
the full `code/build_validated_crash_panels.py` pipeline: a strict per-state
coverage manifest (re-fetched from each state DOT's own open-data portal via
`code/state_dot_sources.py`) plus a FARS-vs-state-DOT fatality comparison
(`code/validate_state_fatalities.py`) reviewed against real evidence, for
each of the 12 states. That is materially more work than the FARS fix above
-- 12 separate government data sources, each with its own API and likely its
own data-quality surprises -- and has not been attempted in this session.
Existing `output/tables/state_dot_analysis_share.csv` /
`state_dot_descriptives_share.csv` (dated 2026-08-15) predate the validation
gate entirely and must still be treated as stale, unvalidated exploratory
output, not a replacement.
