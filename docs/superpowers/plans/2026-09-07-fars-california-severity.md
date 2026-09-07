# FARS-plus-California Severity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a collision-level California severity panel and wire the hourly California analysis to fatal, serious, and non-serious crash outcomes while preserving nationwide FARS fatality outputs.

**Architecture:** A new import-safe California adapter downloads and caches paired CCRS collision/injury files, classifies one severity per collision, and writes county-day/hour aggregates plus a coverage manifest. The existing FARS builders remain the authoritative national source; only the California severity runner changes its input contract.

**Tech Stack:** Python 3.13, pandas, requests, PyArrow, pytest, existing project coverage utilities, and pyfixest for the unchanged analysis runners.

**Spec:** `docs/superpowers/specs/2026-09-07-fars-california-severity-design.md`

## Global Constraints

- Do not overwrite `california_ccrs_county_day.parquet` or `california_ccrs_county_hour.parquet`.
- Treat FARS as fatal-crash/person-fatality data, not an all-crash census.
- California severity classes are mutually exclusive and collision-level.
- Fail closed on missing source files, schema, dates, or geography.
- Keep raw downloads and generated panels out of version control.

---

### Task 1: Add failing classifier and aggregation tests

**Files:**
- Create: `tests/test_ca_ccrs_severity.py`
- Test: `code/ca_ccrs_severity.py`

**Interfaces:**
- `classify_collisions(crashes, injuries) -> (classified, diagnostics)`
- `aggregate_severity(classified) -> (county_day, county_hour)`

- [ ] **Step 1: Write the failing tests** for fatal priority, serious injury, witness exclusion, unknown-code diagnostics, duplicate report-version resolution, and aggregate conservation.
- [ ] **Step 2: Run `pytest tests/test_ca_ccrs_severity.py -q` and verify the import/function failures are expected.**

### Task 2: Implement the pure California classifier

**Files:**
- Create: `code/ca_ccrs_severity.py`
- Modify: `tests/test_ca_ccrs_severity.py`

**Interfaces:**
- `SERIOUS_INJURY_CODES`, `FATAL_INJURY_CODES`
- `normalize_crashes(frame) -> DataFrame`
- `normalize_injuries(frame) -> DataFrame`
- `classify_collisions(crashes, injuries) -> tuple[DataFrame, dict[str, int]]`
- `aggregate_severity(classified) -> tuple[DataFrame, DataFrame]`

- [ ] **Step 1: Implement only schema normalization and collision classification.** Normalize whitespace/case, resolve duplicate collision IDs by report version, exclude witness-only injury rows, and apply fatal > serious > nonserious priority.
- [ ] **Step 2: Run the focused tests and confirm they pass.**
- [ ] **Step 3: Add county/date/hour aggregation and conservation assertions.**
- [ ] **Step 4: Run the focused tests again.**

### Task 3: Add cached source downloads and the production build

**Files:**
- Modify: `code/ca_ccrs_severity.py`
- Create: `tests/test_ca_ccrs_severity_download.py`

**Interfaces:**
- `build_year(year, session, raw_dir) -> tuple[county_day, county_hour, coverage]`
- `build(years, raw_dir, output_dir) -> dict[str, Path]`
- CLI: `python code/ca_ccrs_severity.py --years 2016 2017 ... 2022`

- [ ] **Step 1: Write failing tests** for URL/resource selection, checksum sidecars, and fail-closed missing-column behavior using mocked HTTP responses.
- [ ] **Step 2: Run the focused download tests and verify the expected failures.**
- [ ] **Step 3: Implement resource discovery from the CA data catalog, streamed downloads with atomic writes, checksum sidecars, chunked CSV parsing, and coverage diagnostics.**
- [ ] **Step 4: Run the focused download tests and confirm they pass.**
- [ ] **Step 5: Run the 2016--2022 production build, then inspect counts, conservation, coverage, and output schema.**

### Task 4: Wire the California severity analysis

**Files:**
- Modify: `code/run_same_hour_ca_severity.py`
- Modify: `code/run_hourly_alert_event_study.py`
- Create: `tests/test_ca_severity_runner_contract.py`

**Interfaces:**
- New hourly input: `data/processed/california_ccrs_severity_county_hour.parquet`
- Outcomes: `ca_fatal_crashes`, `ca_serious_crashes`, `ca_nonserious_crashes`

- [ ] **Step 1: Write failing contract tests** showing the runner requires the new panel and never substitutes `ca_serious_inj`.
- [ ] **Step 2: Run the contract tests and verify failure.**
- [ ] **Step 3: Update the runners, labels, and output metadata to use the new mutually exclusive outcomes.**
- [ ] **Step 4: Run contract tests and the existing hourly tests.**

### Task 5: Verify FARS nationwide and report limitations

**Files:**
- Modify: `docs/superpowers/specs/2026-09-07-fars-california-severity-design.md`
- Create: `output/tables/ca_severity_build_report.csv` (generated, untracked)

- [ ] **Step 1: Run the full FARS builder tests and verify the canonical nationwide columns and annual totals remain unchanged.**
- [ ] **Step 2: Run the California severity analysis for all three outcomes.**
- [ ] **Step 3: Run the relevant model tests and write a compact coverage/diagnostic report.**
- [ ] **Step 4: Report FARS nationwide results separately from California severity results, including the 2016--2022 California window and any failed/partial source years.**
