# Task 2 report: strict pagination and bulk-download primitives

## Result

Implemented `code/crash_download.py` with strict ArcGIS and Socrata paging,
machine-readable `IncompleteDownloadError` diagnostics, atomic streamed bulk
downloads, and SHA-256 helpers. Focused tests are in
`tests/test_crash_download.py`.

## RED evidence

After writing the focused tests and before creating the implementation module,
ran:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_download.py -v
```

Pytest collected the new test module and failed during import with
`ModuleNotFoundError: No module named 'crash_download'`, confirming the tests
were exercising missing behavior.

## GREEN evidence

Focused tests:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_download.py -v
7 passed in 0.05s
```

Compilation and the focused regression set:

```text
/tmp/amber-alert-analysis-venv/bin/python -m py_compile \
  code/crash_download.py tests/test_crash_download.py
/tmp/amber-alert-analysis-venv/bin/pytest \
  tests/test_crash_download.py tests/test_crash_coverage.py \
  tests/test_state_dot_analysis_core.py tests/test_state_dot_analysis_runner.py -q
26 passed in 32.98s
```

The tests cover ordered ArcGIS feature pagination, early empty pages,
duplicate IDs, embedded API errors, Socrata count-first reconciliation and
ordered pages, duplicate IDs, early empty pages, bulk streaming and checksum
calculation, atomic replacement, and preservation of an existing destination
after a failed download.

## Self-review

- ArcGIS requests include `orderByFields=<id> ASC`, explicit offsets and page
  sizes, and extract feature attributes for source adapters.
- Socrata always performs `$select=count(*)` first, applies `$where` to both
  count and data requests, uses `$order=<id> ASC` when a stable ID is supplied,
  and reconciles the final row count.
- Both pagers reject embedded API errors, malformed responses, missing IDs,
  duplicate IDs, premature empty pages, and final count mismatches.
- `IncompleteDownloadError` records expected count, fetched count, and a
  terminal diagnostic for manifest consumers.
- Bulk files are streamed to a same-directory temporary file, fsynced, and
  atomically replaced only after a successful stream; failures clean up the
  temporary file and leave an existing destination unchanged.
- Existing dirty files outside Task 2 were not modified or staged.

## Concerns

- Strict Socrata duplicate/order validation requires callers to supply a
  stable `id_field` (or `stable_id_field`) when the dataset exposes one; a
  conventional ID is detected for validation when omitted, but no speculative
  `$order` expression is sent.
- Bulk-download callers receive the SHA-256 string; the destination path is
  the path they supplied and can be recorded alongside the digest.

## Fix round 1

### RED evidence

Added regressions for premature non-empty short pages from both APIs, missing
Socrata stable-ID configuration, and rows missing the configured Socrata ID.
Before changing production code, ran:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_download.py -v
```

The new tests failed as intended: both short-page tests exhausted the fake
responses and reported a terminal request failure, while the missing stable-ID
test attempted unordered pagination instead of rejecting the request. The
existing explicit missing-row-ID test already remained green.

### Fix and GREEN evidence

ArcGIS and Socrata now require each non-empty page to contain exactly
`min(page_size, expected_count - fetched_count)` records before any records
are extended or the offset advances. Socrata now requires `id_field` or
`stable_id_field` for non-empty extracts, always sends `$order=<id> ASC`, and
always validates every returned row against that ID field.

After the fixes:

```text
/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_download.py -v
11 passed in 0.06s

/tmp/amber-alert-analysis-venv/bin/python -m py_compile \
  code/crash_download.py tests/test_crash_download.py
/tmp/amber-alert-analysis-venv/bin/pytest \
  tests/test_crash_download.py tests/test_crash_coverage.py \
  tests/test_state_dot_analysis_core.py tests/test_state_dot_analysis_runner.py -q
30 passed in 32.51s
```

### Fix-round concerns

- A non-empty page shorter or longer than the exact requested remainder now
  invalidates the download immediately; only an exactly sized final page may
  be shorter than `page_size`.
- A Socrata count of zero returns an empty extract without a data-page request,
  so no stable identifier is needed when there is no pagination.
