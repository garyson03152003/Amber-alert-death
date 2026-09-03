# Task 1 report: deterministic nearest-vintage resolution

## Scope

Added `code/route_vintages.py` with immutable `VintageChoice` records,
deterministic LODES year resolution, ACS window resolution, and an atomic,
schema-versioned manifest writer. Added focused tests in
`tests/test_route_vintages.py`.

## TDD evidence

### RED

Command:

```text
pytest tests/test_route_vintages.py -q
```

Result: 4 failed. Each failure was the expected
`ModuleNotFoundError: No module named 'route_vintages'` before the production
module existed.

### GREEN

Focused command:

```text
pytest tests/test_route_vintages.py -q
```

Result: `4 passed in 0.01s`.

Full suite command:

```text
pytest -q
```

Result: `366 passed, 9 warnings in 54.75s`.

## Behavior and implementation notes

- Candidate years are integer-validated and deduplicated.
- LODES selection prefers an exact year, then minimum absolute gap, with the
  earlier year selected on ties; an empty LODES candidate list raises the
  specified `ValueError`.
- ACS selection prefers a containing window, then nearest midpoint with an
  earlier-midpoint tie break; an empty window list produces an unavailable
  choice.
- Manifest rows are sorted by analysis year, state, and source year (or
  `lodes_source_year`) and atomically replaced into place. JSON is canonical;
  CSV output is supported with a schema-version column.
- The module performs no work at import time.

## Review fix round

Added targeted coverage for integer validation, candidate deduplication,
unavailable ACS windows, non-containing midpoint ties, CSV schema output, and
preservation of an existing manifest when validation fails. Manifest sorting
now normalizes fields, validates integer years, places missing values last, and
falls back from a null `source_year` to `lodes_source_year`.

Focused verification:

```text
pytest tests/test_route_vintages.py -q
```

Result: `8 passed in 0.01s`.

Full verification:

```text
pytest -q
```

Result: `370 passed, 9 warnings in 52.91s`.
