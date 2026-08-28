# Validated Crash Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build validated, zero-balanced national FARS and state-DOT county-day panels, then regenerate commuter-share estimates only from coverage units proven complete.

**Architecture:** A side-effect-free coverage core defines manifest rows, validates reporting units, expands validated sparse events over explicit county universes, and preserves unavailable outcomes as missing. Source-specific builders produce sparse events plus manifests; a validation pipeline cross-checks state person fatalities against canonical FARS before the analysis runners accept balanced panels.

**Tech Stack:** Python 3.13, pandas, numpy, pyarrow, requests, pyfixest, pytest; annual NHTSA FARS CSV ZIP files; existing state DOT bulk/API sources.

## Global Constraints

- Never convert an absent event row to zero unless its state-year or county-year coverage unit is valid.
- FARS is the primary national person-fatality source; it is not an all-crash or all-serious-injury source.
- CRSS and NASS/GES must not enter county-day count models.
- Keep person fatalities, fatal crashes, injury crashes, serious-injury persons, and injury proxies as distinct outcome concepts.
- Exclude Puerto Rico from the primary national panel; include the 50 states and District of Columbia.
- Reject FARS county codes `000` and `999` and all non-Census county/equivalent codes.
- Preserve New York person fatalities and serious injuries as missing.
- Use raw nonnegative counts with `log(population)` PPML exposure offsets; retain structural zeros.
- Existing result files must not be overwritten until validated replacements and tests pass.
- Preserve the user's pre-existing uncommitted analysis changes and generated outputs.

---

### Task 1: Coverage manifest and zero-balancing core

**Files:**
- Create: `code/crash_coverage.py`
- Create: `tests/test_crash_coverage.py`

**Interfaces:**
- Produces: `CoverageResult`, `validate_reporting_unit(...)`, `balance_validated_panel(...)`, `write_manifest(...)`.
- Consumed by: all later source builders and analysis loaders.

- [ ] **Step 1: Write failing manifest-validation tests**

Test literal reporting units for: exact fetched/count match; terminal page failure; invalid dates; invalid geography; missing outcome columns; and a valid genuine-empty Wisconsin county-year. Assert explicit failure codes such as `fetch_count_mismatch`, `terminal_page_error`, and `invalid_geography`.

```python
def test_count_mismatch_invalidates_reporting_unit():
    result = validate_reporting_unit(
        source="NV_NDOT", state="NV", year=2024,
        expected_records=100, fetched_records=99, retained_records=99,
        request_complete=True, terminal_error=None,
        invalid_date_count=0, invalid_geography_count=0,
        required_columns_ok=True, observed_min_date="2024-01-01",
        observed_max_date="2024-12-31",
    )
    assert result.coverage_valid is False
    assert "fetch_count_mismatch" in result.failure_reasons
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_coverage.py -v`

Expected: import failure because `crash_coverage` does not exist.

- [ ] **Step 3: Implement the manifest contract**

Use a frozen dataclass with serializable fields:

```python
@dataclass(frozen=True)
class CoverageResult:
    source: str
    state: str
    year: int
    county_fips: str | None
    expected_records: int | None
    fetched_records: int
    retained_records: int
    duplicate_records: int
    invalid_date_count: int
    invalid_geography_count: int
    observed_min_date: str | None
    observed_max_date: str | None
    request_complete: bool
    coverage_valid: bool
    failure_reasons: tuple[str, ...]
    source_url: str
    source_checksum: str | None
```

`validate_reporting_unit` must accumulate all applicable failures rather than stopping at the first. `write_manifest` must write deterministic CSV and parquet outputs under `data/processed/coverage/`.

- [ ] **Step 4: Write failing balancing tests**

Cover a two-county, three-day valid unit with one observed crash row, an invalid unit, and an unavailable outcome:

```python
assert balanced.loc[(balanced.fips == "01003") &
                    (balanced.date == "2024-01-02"), "crashes"].item() == 0
assert balanced["person_fatals"].isna().all()
assert not invalid_unit_dates.isin(balanced["date"]).any()
```

- [ ] **Step 5: Implement `balance_validated_panel`**

Signature:

```python
def balance_validated_panel(
    sparse: pd.DataFrame,
    manifest: pd.DataFrame,
    county_universe: pd.DataFrame,
    outcome_availability: Mapping[str, bool],
    *, reporting_unit: Literal["state_year", "county_year"],
) -> pd.DataFrame:
```

It must build only valid county-date grids, fill only available outcomes with zero, preserve unavailable outcomes as `NaN`, and add `coverage_valid`, `coverage_unit`, `structural_zero`, and `source`.

- [ ] **Step 6: Verify GREEN and commit**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_coverage.py -v`

Commit: `feat: add validated crash coverage core`

---

### Task 2: Strict pagination and bulk-download primitives

**Files:**
- Create: `code/crash_download.py`
- Create: `tests/test_crash_download.py`

**Interfaces:**
- Produces: `fetch_arcgis_pages(...)`, `fetch_socrata_pages(...)`, `download_bulk_file(...)`, and `sha256_file(...)`.
- Consumed by: Tasks 3 and 4 source adapters.

- [ ] **Step 1: Write failing pagination tests**

Use deterministic fake response sequences to verify that the fetcher returns exactly the expected records, rejects an empty page before the expected count, rejects duplicate page records, and records terminal failures.

```python
with pytest.raises(IncompleteDownloadError, match="expected 3, fetched 2"):
    fetch_arcgis_pages(fake_session, url="https://example.test/query",
                       where="YEAR=2024", expected_count=3,
                       page_size=2, id_field="OBJECTID")
```

- [ ] **Step 2: Run and verify RED**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_download.py -v`

- [ ] **Step 3: Implement strict fetchers**

Every page request must specify stable ordering by the unique ID, reject embedded API errors, verify unique IDs, and require `len(records) == expected_count`. Socrata pagination must first query `$select=count(*)`, order by a stable source identifier when available, and reconcile the final count. Bulk downloads must stream to a temporary file, atomically rename on success, and record SHA-256.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_crash_download.py -v`

Commit: `feat: add strict crash download primitives`

---

### Task 3: Canonical national FARS 2013–2024 builder

**Files:**
- Modify: `code/build_fars_county_day.py`
- Modify: `code/01_download_fars.py`
- Create: `tests/test_fars_builder.py`
- Create: `data/processed/coverage/.gitkeep`

**Interfaces:**
- Produces: `data/processed/fars_events_county_day.parquet`, `data/processed/coverage/fars_coverage.csv`, and `data/processed/fars_balanced_county_day.parquet`.
- Consumes: Task 1 coverage core and a year-specific county universe derived from `county_population.parquet`, with explicit 50-state-plus-DC FIPS validation.

- [ ] **Step 1: Write failing FARS cleaning tests**

Fixtures must include valid counties, `000`, `999`, Puerto Rico, duplicate `ST_CASE`, invalid dates, and 2024 records. Assert that invalid/PR rows are excluded with manifest counts and that valid fatality totals reconcile.

- [ ] **Step 2: Run and verify RED**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_fars_builder.py -v`

- [ ] **Step 3: Refactor the builder into import-safe functions**

Create:

```python
def build_fars_year(year: int, session: requests.Session) -> tuple[pd.DataFrame, CoverageResult]
def build_fars(years: Iterable[int] = range(2013, 2025)) -> tuple[pd.DataFrame, pd.DataFrame]
def fars_county_universe(population: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame
```

Require all 12 years. Count raw accident rows, reconcile unique `ST_CASE`, reject invalid county/date records, and write a canonical schema containing `fips`, `date`, `fatal_crashes`, `person_fatals`, `drunk_fatals`, `sober_fatals`, and `weather_adverse`.

- [ ] **Step 4: Remove the FARS output collision**

Change `code/01_download_fars.py` to write `data/processed/fars_legacy_fatalities_county_day.parquet`, print a deprecation notice, and never overwrite canonical FARS outputs.

- [ ] **Step 5: Balance FARS and test geography**

Use the population county universe for each year, restrict state FIPS to 01–56 plus DC and exclude territories. Add an explicit Connecticut policy constant; until a tested crosswalk exists, exclude Connecticut from the canonical longitudinal regression panel while retaining its sparse events and a manifest warning.

- [ ] **Step 6: Verify GREEN and commit**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_fars_builder.py tests/test_crash_coverage.py -v`

Commit: `feat: build canonical validated FARS panel`

---

### Task 4: State-source validation adapters

**Files:**
- Create: `code/state_dot_sources.py`
- Create: `tests/test_state_dot_sources.py`
- Modify: `code/build_california_ccrs.py`
- Modify: `code/build_florida_fdot.py`
- Modify: `code/extend_florida_fdot.py`
- Modify: `code/build_illinois_idot.py`
- Modify: `code/build_iowa_dot.py`
- Modify: `code/build_massachusetts_massdot.py`
- Modify: `code/build_nevada_ndot.py`
- Modify: `code/build_newyork_dot.py`
- Modify: `code/build_oregon_odot.py`
- Modify: `code/build_tennessee_tdot.py`
- Modify: `code/build_texas_txdot.py`
- Modify: `code/extend_texas_txdot.py`
- Modify: `code/build_virginia_vdot.py`
- Modify: `code/build_wisconsin_dot.py`

**Interfaces:**
- Produces: `STATE_SOURCE_SPECS`, `validate_state_year(...)`, `validate_wisconsin_county_year(...)`, per-source sparse events, and coverage manifests.
- Consumes: Tasks 1–2.

- [ ] **Step 1: Write source-spec and failure tests**

Define literal expected universes and outcome concepts for all 12 states. Tests must assert: FL 2019 is rejected; TN 2025 is rejected; TX years validate independently; Wisconsin failed and empty responses differ; NY person fatalities/serious injuries are unavailable; California and Wisconsin serious-injury proxies are not labeled comparable until native fields pass audit.

- [ ] **Step 2: Run and verify RED**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_state_dot_sources.py -v`

- [ ] **Step 3: Implement declarative source specs**

Each `StateSourceSpec` records source, reporting unit, requested years, expected county FIPS, native outcome mapping, comparable outcomes, and query identifier. The validator consumes fetch diagnostics rather than inferring completeness from the final sparse parquet.

- [ ] **Step 4: Replace permissive paging paths**

Patch ArcGIS/Socrata builders to use Task 2 strict fetchers. A failed page must invalidate that unit, and the strict aggregate command must exit nonzero after writing diagnostic manifests. Patch CA/IL/IA bulk paths to record checksums and reject nonempty-but-incomplete schema/year extracts. Wisconsin must manifest every one of 72 county-year requests separately.

- [ ] **Step 5: Correct outcome mappings**

Audit native columns in builder fixtures and map only verified concepts. California person fatalities must reconcile with native fatal-person fields; `NUMBERINJURED` remains an explicitly named injury proxy unless a serious-injury field is verified. Wisconsin must count serious-injury persons rather than all injured persons on A-severity crashes or mark the field noncomparable.

- [ ] **Step 6: Verify adapters and commit**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_state_dot_sources.py tests/test_crash_download.py -v`

Commit: `feat: validate state crash reporting units`

---

### Task 5: Build validated state panels and FARS comparison

**Files:**
- Create: `code/build_validated_crash_panels.py`
- Create: `code/validate_state_fatalities.py`
- Create: `tests/test_validated_crash_pipeline.py`

**Interfaces:**
- Produces: `data/processed/validated/{state}_county_day.parquet`, `output/tables/state_fars_fatality_validation.csv`, and `output/tables/accepted_state_years.csv`.
- Consumes: manifests, sparse events, explicit county universes, and canonical FARS.

- [ ] **Step 1: Write failing end-to-end fixture test**

Use one accepted and one rejected state-year. Assert that only the accepted unit is balanced, contains zeros, preserves unavailable measures, and appears in the acceptance report with its reviewed decision.

- [ ] **Step 2: Run and verify RED**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_validated_crash_pipeline.py -v`

- [ ] **Step 3: Implement validated panel orchestration**

Fail if a requested manifest is absent. Balance each valid unit using Task 1. Write outputs under `data/processed/validated/` without replacing legacy sparse files.

- [ ] **Step 4: Implement state/FARS comparison**

Calculate state-year totals, DOT/FARS ratio, county-year Pearson correlation, county-date agreement, invalid geography counts, and a reviewed status. Generate an initial candidate report with `review_status="pending"`; accept only state-years listed in a version-controlled allowlist created after inspecting the report.

- [ ] **Step 5: Verify GREEN and commit**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_validated_crash_pipeline.py -v`

Commit: `feat: build validated balanced state panels`

---

### Task 6: Integrate nationwide alerts and validated panels into analysis

**Files:**
- Modify: `code/run_state_dot_analysis_fixed.py`
- Modify: `code/run_state_dot_analysis_share.py`
- Create: `code/run_validated_fars_share.py`
- Modify: `code/state_dot_analysis_core.py`
- Modify: `tests/test_state_dot_analysis_core.py`
- Modify: `tests/test_state_dot_analysis_runner.py`

**Interfaces:**
- Produces: validated state-DOT result tables, national FARS result tables, and structured model-status diagnostics.
- Consumes: Task 5 accepted panels and nationwide alert origins.

- [ ] **Step 1: Write failing integration tests**

Cover cross-border commuter exposure, fail-fast behavior for missing manifests/weights, structural-zero retention, nonfinite coefficient rejection, and expected-versus-produced model counts.

- [ ] **Step 2: Run and verify RED**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_state_dot_analysis_core.py tests/test_state_dot_analysis_runner.py -v`

- [ ] **Step 3: Load nationwide alert origins**

Separate alert-origin geography from outcome-state filtering. Compute spillovers from all valid US alert home counties represented in commuting flows while restricting outcome destinations to the validated analysis panels.

- [ ] **Step 4: Require validated inputs**

Preferred runners must reject missing coverage manifests, missing commuting weights, unreviewed state-years, and sparse legacy state files. Add explicit `--direct-only` only if invoked by the user; never silently relabel it `spillover_joint`.

- [ ] **Step 5: Add structured fit diagnostics**

For every expected fit, emit status, input N, fitted N, zero share, terms requested, terms produced, and error reason. Reject rows with nonfinite beta, SE, or p-value.

- [ ] **Step 6: Verify GREEN and commit**

Run: `/tmp/amber-alert-analysis-venv/bin/pytest tests/test_state_dot_analysis_core.py tests/test_state_dot_analysis_runner.py -v`

Commit: `feat: run validated national and state crash models`

---

### Task 7: Strict rebuild, review allowlist, and regenerate results

**Files:**
- Create: `config/accepted_state_years.csv`
- Create: `output/VALIDATED_CRASH_RESULTS.md`
- Generate: validated data, manifest, validation, result, descriptive, status, and log files specified above.

**Interfaces:**
- Consumes all preceding tasks.
- Produces the final reproducible analysis deliverables.

- [ ] **Step 1: Install pinned dependencies in a temporary environment**

Create `requirements-analysis.txt` with exact versions verified in the current environment, then install into `/tmp/amber-validated-venv`.

- [ ] **Step 2: Rebuild FARS and all state sources**

Run every strict builder. Any invalid reporting unit must appear in a manifest and make the aggregate strict build exit nonzero; fix source logic rather than overriding validation.

- [ ] **Step 3: Inspect fatality validation and create allowlist**

Review all state-years using the generated comparison metrics. Record each accepted state-year and a concise evidence reason in `config/accepted_state_years.csv`; rejected units remain excluded.

- [ ] **Step 4: Build balanced panels and run analyses**

Run validated FARS and state-DOT share analyses with complete logs. Confirm substantial zero shares for crash, fatality, and serious-injury outcomes where those outcomes are available.

- [ ] **Step 5: Write results summary**

Record commit, dependency versions, source checksums, coverage decisions, panel dimensions, pooled estimates, comparison with stale outputs, fit failures, and interpretation guardrails in `output/VALIDATED_CRASH_RESULTS.md`.

- [ ] **Step 6: Commit generated code/config/docs, excluding raw downloads**

Commit: `analysis: regenerate validated crash estimates`

---

### Task 8: Final verification and branch handoff

**Files:**
- Verify all changed files and final outputs.

**Interfaces:**
- Produces a clean verification report and integration choice.

- [ ] **Step 1: Run the full test suite**

Run: `/tmp/amber-validated-venv/bin/pytest -v`

Expected: all tests pass with no warnings indicating formatter, schema, or incomplete-download errors.

- [ ] **Step 2: Run output invariants**

Assert every required file is nonempty, no unvalidated unit appears in regression inputs, structural zeros exist, FARS totals reconcile, expected/produced model counts reconcile, no interpreted estimate has nonfinite inference, and logs contain no traceback.

- [ ] **Step 3: Inspect repository state**

Run: `git diff --check`, `git status --short`, and review commits to ensure the user's pre-existing changes were preserved.

- [ ] **Step 4: Request code review and address findings**

Use `superpowers:requesting-code-review`, fix Critical and Important issues with TDD, and rerun Steps 1–3.

- [ ] **Step 5: Finish the branch**

Use `superpowers:finishing-a-development-branch` and present the standard integration choices without pushing or merging automatically.
