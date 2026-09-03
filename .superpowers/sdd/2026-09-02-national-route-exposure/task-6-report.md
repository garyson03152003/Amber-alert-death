# Task 6 implementation report

Status: DONE_WITH_CONCERNS

## Changes

- Added `code/run_route_exposure_national.py` with `build_national_exposure`, `required_flow_partitions`, and `run_national_route_analysis`.
- Extended `build_alert_date_exposures` with keyword-only `label` and `vintage_columns` inputs while preserving the pilot default label and call signature.
- Built national own/cross/pass-through commuter-car-mile exposure rows from county segments, including route shares, denominators, audit weights, same-tract mode, network/source IDs, analysis year, LODES year set, and ACS vintage set.
- Added state-specific mixed-vintage support without coercing multiple source years into one false scalar value.
- Added cached partition reads, so a segment artifact mapped to multiple analysis years is read once per run.
- Added analysis-panel integration that keeps route and destination measures side by side and carries route denominators onto covered non-alert control dates while setting affected dosage to zero.
- Added per-year, per-vintage, and pooled model invocations and atomic combined/scope-specific output tables.
- The default estimator delegates to the established symmetric commuter robustness implementation across baseline-calendar, county-year/weekday, and state-date fixed effects; available holiday, day-after-holiday, weather, lagged-outcome, daytime-alert, and non-AMBER WEA controls; and Webb state-cluster plus Rademacher state-month inference variants.
- Segment partitions now retain their analysis year, LODES source year, ACS vintage, source partition, source manifest, and network manifest. The national loader validates these embedded fields against every manifest row and verifies a supplied segment SHA-256 before using cached content.
- Every segment-manifest row must now declare either `route_national.segments.v1` or the explicit `route_national.segments.legacy.v0` compatibility contract. V1 manifests require nonblank Task 5 source/network/partition IDs, and v1 partitions require singular matching schema, analysis year, source year, ACS vintage, and source/network/partition IDs.
- Counties without a finite positive commuter-car-mile denominator are explicitly marked `excluded_missing_or_nonpositive_denominator`, retain missing route treatments in the audit panel, and are excluded from every estimation scope.

## TDD evidence

### RED

Initial focused run:

`pytest tests/test_route_exposure_national.py -q`

Result: 4 failed. Three failures were the expected missing `run_route_exposure_national` module and one was the expected unsupported `label` keyword in the pilot helper.

Same-tract sensitivity RED:

`pytest tests/test_route_exposure_national.py::test_zero_same_tract_mode_removes_same_tract_route_mileage -q`

Result: failed with observed own affected miles 16.0 instead of 0.0.

Mixed-vintage RED:

`pytest tests/test_route_exposure_national.py::test_national_runner_preserves_state_specific_nearest_vintage_sets -q`

Result: failed because a mixed state-specific LODES year was incorrectly treated as a conflicting scalar vintage.

Control-day denominator RED:

`pytest tests/test_route_exposure_national.py::test_national_runner_reuses_segments_and_keeps_destination_measure -q`

Result: failed because the non-alert county-date had a missing denominator instead of the year/county route denominator.

Result-provenance RED:

The same focused runner test failed because model rows did not yet include `same_tract_mode`, source manifest IDs, and network manifest IDs.

Review-fix RED:

`pytest tests/test_route_exposure_national.py::test_loaded_segment_partition_rejects_manifest_vintage_relabel tests/test_route_exposure_national.py::test_loaded_segment_partition_rejects_manifest_checksum_mismatch tests/test_route_exposure_national.py::test_model_sample_excludes_counties_without_route_denominator tests/test_route_exposure_national.py::test_established_route_specs_retain_controls_fixed_effects_and_inference -q`

Result: the two provenance tests did not raise, the uncovered county reached the model runner as a zero-exposure control, and the established route specification interface was absent.

Segment-provenance RED:

`pytest tests/test_route_national_segments.py::test_streaming_segments_retain_flow_vintage_provenance -q`

Result: failed because streamed county segments did not retain the flow/vintage provenance needed for downstream validation.

Strict-v1 provenance RED:

`pytest tests/test_route_exposure_national.py::test_v1_segment_manifest_requires_nonblank_task5_provenance_fields -q`

Result: 2 failed because a v1 manifest accepted a missing source-partition ID and a blank network-manifest ID.

`pytest tests/test_route_exposure_national.py::test_v1_segment_rejects_missing_blank_or_mixed_task5_provenance -q`

Result before the strict contract: the missing embedded analysis-year case did not raise; missing, blank, and mixed source identity happened to fail only when the manifest supplied the corresponding optional field.

`pytest tests/test_route_exposure_national.py::test_explicit_legacy_manifest_rejects_a_different_embedded_schema -q`

Result: failed because the legacy path accepted an arbitrary embedded segment schema.

Partial-row provenance RED:

`pytest tests/test_route_exposure_national.py::test_v1_segment_rejects_partially_missing_provenance_rows -q`

Result before the completeness fix: the v1 loader accepted a partition when one row had a missing provenance value because `_unique_partition_value` dropped nulls before checking uniqueness. The regression covers every required v1 field; the existing strict-v1 test also covers a partially blank field.

### GREEN

Focused national, pilot, and segment regression tests after the review fixes:

`pytest tests/test_route_exposure_national.py tests/test_route_exposure_pilot.py tests/test_route_national_segments.py -q`

Result after the partial-row completeness fix: 54 passed, 1 pre-existing pandas FutureWarning in 1.52s.

Default estimator smoke test:

The established FE/Webb adapter fit a 200-row, five-state synthetic panel and returned three successful treatment rows.

Full suite:

`pytest -q`

Result after the partial-row completeness fix: 447 passed, 9 warnings in 56.40s.

Static checks:

`git diff --check`

Result: clean.

`python -m py_compile code/run_route_exposure_national.py code/run_route_exposure_pilot.py`

Result: clean.

## Concerns / handoff

- Task 6 consumes a successful, explicitly schema-labelled segment manifest. V1 rows additionally require `source_manifest_id`, `network_manifest_id`, and `source_partition_id`; Task 7 still owns end-to-end manifest production, gate evaluation, broader CLI build flags, and the synthetic dry run.
- The default model adapter intentionally reuses the current symmetric commuter robustness estimator and its two-way state/date covariance plus wild-bootstrap implementation rather than defining a new estimator. The route treatments are estimated jointly in each established control/FE/inference specification; control coefficient rows are not emitted as route-effect results.
- V1 partitions always require and validate embedded `analysis_year`. Cross-analysis-year reuse of older partitions remains available only through the explicit `route_national.segments.legacy.v0` manifest contract; that path still validates LODES source year and ACS vintage, checks embedded analysis year when present, and rejects conflicting embedded schemas or any supplied manifest identity/checksum mismatch.
- Route coverage exclusions remain visible in `national_route_model_panel.parquet`; they are not silently rewritten to zero. Task 7 still owns run-level coverage gates and acceptance thresholds.
- No national downloads, routing, or real-data analysis were started.
