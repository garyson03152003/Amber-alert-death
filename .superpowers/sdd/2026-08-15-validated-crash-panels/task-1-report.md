# Task 1 report: coverage manifest and zero-balancing core

## Result

Implemented the shared coverage-validation and zero-balanced-panel foundation
in `code/crash_coverage.py`, with tests in
`tests/test_crash_coverage.py`.

## RED evidence

Before creating the implementation module, ran:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_coverage.py -v
```

Pytest collected the new test module and failed during import with
`ModuleNotFoundError: No module named 'crash_coverage'`, confirming the tests
were exercising missing behavior.

## GREEN evidence

After implementation and cleanup:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_coverage.py -v
7 passed in 0.78s

/tmp/amber-alert-analysis-venv/bin/pytest \
  tests/test_crash_coverage.py tests/test_state_dot_analysis_core.py -q
14 passed in 1.16s

/tmp/amber-alert-analysis-venv/bin/python -m py_compile \
  code/crash_coverage.py tests/test_crash_coverage.py
```

The tests cover exact count reconciliation, terminal errors, accumulated
validation failures, valid genuine-empty Wisconsin county-years, deterministic
CSV/parquet manifests, valid-unit-only date expansion, structural zero filling,
and preservation of unavailable outcomes as numeric missing values.

## Files

- `code/crash_coverage.py`: frozen `CoverageResult`, reporting-unit validator,
  validated county-day balancer, and deterministic manifest writer.
- `tests/test_crash_coverage.py`: focused unit tests for the manifest and panel
  contract.

## Self-review

- Validation accumulates applicable failure reasons and allows a completed,
  explicitly counted zero-record unit.
- Failed manifest units are omitted before county-date expansion.
- Available outcomes are filled only for missing rows inside valid grids;
  unavailable outcomes are forced to `NaN`.
- State abbreviations and numeric FIPS are normalized when matching a county
  universe, and year-specific universes are honored when supplied.
- Manifest rows are sorted deterministically and failure tuples are encoded as
  scalar pipe-delimited strings for CSV and parquet serialization.
- Existing dirty files outside Task 1 were not modified.

## Concerns

- The balancer expects sparse event data to be pre-aggregated by `fips` and
  `date`; source-specific duplicate diagnostics belong in their builders.
- Parquet output requires the repository's existing pyarrow-capable runtime.
- Source-specific retrieval-boundary exceptions and outcome semantics remain
  the responsibility of later source adapters.

## Fix round 1

### RED evidence

Added the source-isolation and missing-diagnostic round-trip tests before
changing production code, then ran:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_coverage.py -v
```

The new tests failed as intended:

- `test_manifest_normalizes_missing_failure_reasons` raised
  `TypeError: can only join an iterable` for a `NaN` diagnostic.
- `test_balance_excludes_events_from_invalid_source_sharing_county_date`
  observed `100.0` instead of `1`, proving that the invalid source's event
  leaked into the valid panel.

### Fix and GREEN evidence

`balance_validated_panel` now groups and joins sparse data by `source` when
present, filters each valid manifest unit to its source, and fails closed when
a source-less sparse frame is paired with multiple manifest sources. This
preserves backward compatibility for a genuinely single-source input while
preventing failed-source events from contributing to valid units.

`write_manifest` now serializes tuple/list diagnostics and normalizes
`None`/`NaN` to an empty scalar string. The test verifies an empty value after
CSV round-trip with `keep_default_na=False` (the pandas reader otherwise
interprets an intentionally empty CSV field as missing `NaN`).

After the fixes:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_coverage.py -v
9 passed in 0.86s

/tmp/amber-alert-analysis-venv/bin/pytest \
  tests/test_crash_coverage.py tests/test_state_dot_analysis_core.py -q
16 passed in 1.04s

/tmp/amber-alert-analysis-venv/bin/python -m py_compile \
  code/crash_coverage.py tests/test_crash_coverage.py
```

### Fix-round files

- `code/crash_coverage.py`: source-aware sparse grouping/joining and robust
  failure-reason serialization.
- `tests/test_crash_coverage.py`: source leakage regression and missing
  failure-reason CSV round-trip tests.
- This report: appended fix-round RED/GREEN evidence and review notes.

### Fix-round self-review and concerns

- Invalid source rows sharing a valid source's county-date are excluded when
  sparse records carry `source`.
- Multiple manifest sources with source-less sparse records now raise rather
  than infer provenance; callers must supply source identity in that case.
- Single-source sparse callers remain compatible with the original interface.
- Existing dirty analysis files and generated outputs remain unstaged.
- Empty CSV fields require `keep_default_na=False` for pandas to preserve the
  empty string on read; the written field itself is deterministic and empty.
