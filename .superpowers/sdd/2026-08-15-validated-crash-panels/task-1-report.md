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

