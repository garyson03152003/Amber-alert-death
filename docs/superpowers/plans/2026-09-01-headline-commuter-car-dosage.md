# Headline Commuter-Car Dosage Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a year-matched, tract-preserved commuter-car-mile exposure to the headline sleep/next-morning analysis and report it alongside the existing binary and county-share specifications.

**Architecture:** Reuse the validated LODES tract-car dosage and year-matching logic already used by `run_symmetric_commuter_fatigue.py`. The main night-to-morning runner will attach `own_driver_distance` and `cross_driver_distance` for each county-day, preserving the existing direct-alert and simple-share variables for comparability and for downstream modules. New regression rows will estimate the rich own/cross exposures jointly under the same fixed effects and other-WEA controls.

**Tech Stack:** Python, pandas, numpy, pyfixest, pytest, Parquet/CSV outputs.

**Spec:** User request to use the tract-level commute distance and car-share dosage in the analysis.

## Global Constraints

- Preserve existing `night_alert`, `cross_spillover`, and `alert_last2nights_*` columns because other analyses import `attach_cross_spillover` and rely on their units.
- Do not change the exact-hour WEA treatment or its simple commuter-share control in this task.
- Use the existing year mapping: ACS 2015 flows through 2017, ACS 2020 flows from 2018 onward; LODES 2013/2018/2022 nearest-vintage mapping.
- Use the validated `avg_car_x_dist` joint dosage; never substitute `avg_car_share * avg_dist_mi`.
- Keep all pre-existing user changes in the dirty worktree.

---

### Task 1: Lock the rich exposure interface with tests

**Files:**
- Create: `tests/test_night_to_morning_commuter_exposure.py`
- Modify: `code/run_night_to_morning_window.py` only after the test fails

**Interfaces:**
- `attach_year_matched_commuter_exposure(grid, alert_col="night_alert", ...)` returns the grid with `own_driver_distance` and `cross_driver_distance`.
- The own value is the alerted county's self-loop `commuter_car_miles`; the cross value sums alerted origin counties' non-self-loop `commuter_car_miles` into the destination county.

- [x] **Step 1: Write the failing tests** using tiny in-memory grids, pair dosages, and monkeypatched year-vintage loaders. Assert that a self-loop contributes only to `own_driver_distance`, an alerted origin contributes to `cross_driver_distance`, and the rich units are `weight * avg_car_x_dist`.
- [x] **Step 2: Run `pytest tests/test_night_to_morning_commuter_exposure.py -q` and confirm failure because the production helper does not yet exist.

### Task 2: Implement the year-matched exposure attachment

**Files:**
- Modify: `code/run_night_to_morning_window.py`
- Modify: `code/run_symmetric_commuter_fatigue.py` only if a small import-safe helper is needed to avoid duplicating pair-dosage loading

**Interfaces:**
- The helper accepts a county-day grid and returns two numeric exposure columns while preserving all existing columns.
- It loads existing `_lodes_car_year_cache` pair tables and ACS flow vintages, applies the established fallback for uncovered pairs, and uses `construct_year_matched_exposure_series`.

- [x] **Step 1: Add the minimal loading/attachment code** with explicit file-not-found errors telling the user to build the LODES dosage first.
- [ ] **Step 2:** Run the focused tests and confirm they pass.
- [x] **Step 3: Add input normalization and diagnostics for active counties, number of pair edges, fallback share, and nonzero exposure rows.

### Task 3: Use the richer exposure in the headline regressions

**Files:**
- Modify: `code/run_night_to_morning_window.py`
- Modify: `docs/symmetric_commuter_robustness.md` or add a short analysis note documenting which rows are tract-preserved

- [x] **Step 1:** Attach the rich exposures in `main()` after the alert grid is built.
- [x] **Step 2:** Add joint own/cross fatality models for naive and robust fixed effects, with binary other-WEA control and count-dose sensitivity.
- [x] **Step 3:** Label the rows explicitly as `year_matched_tract_car_distance` and retain the legacy binary/share rows for comparison.
- [x] **Step 4:** Add a summary output describing the dosage source and exposure units.

### Task 4: Verify and materialize results

**Files:**
- Test: `tests/test_night_to_morning_commuter_exposure.py`
- Outputs: `output/tables/reg_night_to_morning_window.csv` and a rich-exposure summary CSV

- [x] **Step 1:** Run the focused exposure tests.
- [x] **Step 2:** Run the complete test suite with `pytest -q`.
- [x] **Step 3:** Run the headline analysis and inspect the new rich rows, sample size, nonzero counts, and fallback diagnostics.
- [x] **Step 4:** Report the coefficient comparison and the remaining interpretation caveats (centroid distance and employment flows do not establish actual alert-time driving).
