# Commuter Identification Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add within-state/date identification, an observed-network falsification, and a pre-specified rich time-of-day mechanism analysis without changing existing headline outputs.

**Architecture:** Extend the existing symmetric commuter robustness runner with small, testable construction helpers and a generalized fixed-effects estimator interface. Keep the common 7.3-million-row panel and year-matched pair dosages loaded once, then write three new labelled output tables. Pure functions own ID construction, network shuffling, time-block aggregation, and multiplicity correction.

**Tech Stack:** Python, pandas, numpy, scipy, pyfixest, pytest, Parquet/CSV outputs.

**Spec:** `docs/superpowers/specs/2026-09-01-commuter-identification-improvements-design.md`

## Global Constraints

- Do not overwrite existing headline or robustness CSV files.
- Preserve the ACS/LODES year mapping and `avg_car_x_dist` dosage definition.
- Keep structural-zero county-dates and the established active-county universe.
- Preserve untracked WEA downloads and processed controls.
- Use seeded placebo draws and record draw counts and fixed-effect labels.

---

### Task 1: Generalize fixed effects and construct state-date IDs

**Files:**
- Modify: `tests/test_symmetric_commuter_robustness.py`
- Modify: `code/run_symmetric_commuter_robustness.py`

**Interfaces:**
- Produces: `build_state_date_ids(fips: pd.Series, dates: pd.Series) -> np.ndarray`.
- Extends: `_fit_analytic(..., fixed_effect_cols: tuple[str, ...] = ("fips_year_id", "fips_dow_id", "month_id"))`.

- [ ] **Step 1: Write a failing state-date ID test.** Create two counties in the same state/date and one county in another state/date; assert the first two IDs match and every distinct state/date receives a distinct integer.
- [ ] **Step 2: Run `pytest tests/test_symmetric_commuter_robustness.py -q` and confirm failure because `build_state_date_ids` does not exist.**
- [ ] **Step 3: Implement the minimal ID helper and add `state_date_id` to `_load_common_panel()`.** Normalize FIPS to five digits and dates to midnight before factorizing the state/date key.
- [ ] **Step 4: Write a failing estimator-interface test.** Fit a tiny synthetic model with `fixed_effect_cols=("county_id", "state_date_id")` and assert the returned row records `fixed_effects="county_id + state_date_id"`.
- [ ] **Step 5: Generalize `_fit_analytic` to residualize and formulate against the requested fixed effects, then run the focused test to green.**
- [ ] **Step 6: Commit the tested estimator infrastructure.**

### Task 2: Add the within-state/date robustness specification

**Files:**
- Modify: `code/run_symmetric_commuter_robustness.py`
- Modify: `tests/test_symmetric_commuter_robustness.py`

**Interfaces:**
- Produces labelled rows with `spec="state_date_fixed_effects"` in `reg_symmetric_commuter_identification.csv`.
- Uses fixed effects `fips_year_id + fips_dow_id + state_date_id`.

- [ ] **Step 1: Write a failing orchestration test around a small panel.** Assert both outcomes are requested, own and cross terms enter jointly, and the state-date fixed-effect tuple is passed to the estimator.
- [ ] **Step 2: Run the focused test and confirm the new orchestration helper is missing.**
- [ ] **Step 3: Implement `run_state_date_models(panel, bootstrap_reps, seed)` and label analytic and Webb inference.**
- [ ] **Step 4: Run the focused robustness tests to green.**
- [ ] **Step 5: Commit the state-date specification.**

### Task 3: Add the commuting-network falsification

**Files:**
- Modify: `tests/test_symmetric_commuter_robustness.py`
- Modify: `code/run_symmetric_commuter_robustness.py`

**Interfaces:**
- Produces: `permute_cross_destinations(pair_dosage: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame`.
- Produces: `randomization_pvalue(observed: float, placebo: np.ndarray) -> float`.
- Produces: `run_network_placebos(panel, alerts, metadata, draws, seed) -> tuple[pd.DataFrame, list[dict]]`.

- [ ] **Step 1: Write failing permutation tests.** Assert self loops are unchanged, cross-edge counts and dosages are preserved within each home-state/work-state block, no placebo cross edge becomes a self loop, and equal seeds reproduce equal tables.
- [ ] **Step 2: Run the permutation tests and confirm failure because the helper is absent.**
- [ ] **Step 3: Implement seeded blockwise destination shuffling with collision repair and duplicate-edge aggregation.** Raise a clear error when a block cannot be permuted without creating a self loop.
- [ ] **Step 4: Write and run a failing finite-sample p-value test.** For observed `2.0` and placebo `[-3.0, 0.5, 1.0]`, assert a two-sided p-value of `0.5`.
- [ ] **Step 5: Implement `randomization_pvalue` and run the pure-function tests to green.**
- [ ] **Step 6: Add the placebo runner.** Rebuild only the cross-exposure vector for every seeded draw, estimate it jointly with the observed own exposure, and save one row per draw plus an observed-network summary row.
- [ ] **Step 7: Commit the network falsification.**

### Task 4: Add pre-specified rich time-of-day outcomes

**Files:**
- Modify: `tests/test_symmetric_commuter_fatigue.py`
- Modify: `code/run_symmetric_commuter_fatigue.py`
- Modify: `code/run_symmetric_commuter_robustness.py`

**Interfaces:**
- Produces: `build_time_block_outcomes(hourly: pd.DataFrame) -> pd.DataFrame`.
- Produces: `holm_adjust(pvalues: pd.Series) -> pd.Series`.
- Produces four average-hour outcomes plus `fatals_late_minus_morning`.

- [ ] **Step 1: Write failing block-outcome tests.** Use known constant hourly values to verify block membership, division by block duration, sparse zero handling, and the late-minus-morning contrast.
- [ ] **Step 2: Run the focused fatigue test and confirm the new helper is missing.**
- [ ] **Step 3: Implement the block aggregation using the fixed 06--09, 10--14, 15--19, and 20--23 definitions.**
- [ ] **Step 4: Write a failing Holm-adjustment test with known ordered p-values and implement the monotone step-down adjustment.**
- [ ] **Step 5: Add the block outcomes to the common panel and estimate each with both the state-date and established fixed effects.** Apply Holm adjustment to the four own and four cross block families, leaving the contrast separate.
- [ ] **Step 6: Run the focused tests to green and commit the mechanism analysis.**

### Task 5: Materialize and verify results

**Files:**
- Modify: `docs/symmetric_commuter_robustness.md`
- Output: `output/tables/reg_symmetric_commuter_identification.csv`
- Output: `output/tables/symmetric_commuter_network_placebo.csv`
- Output: `output/tables/reg_symmetric_commuter_time_blocks.csv`

**Interfaces:**
- The runner accepts reduced placebo/bootstrap counts for tests and defaults to 199 placebo draws and 9,999 Webb draws for production.

- [ ] **Step 1: Run all focused symmetric commuter tests.**
- [ ] **Step 2: Run the complete test suite in the pinned analysis environment.**
- [ ] **Step 3: Execute the production runner and inspect row counts, seeds, completed draws, nonfinite estimates, fixed-effect labels, and output paths.**
- [ ] **Step 4: Compare the state-date coefficient, observed-network rank, and time-block estimates with the existing headline result.**
- [ ] **Step 5: Update the results note with coefficients, uncertainty, multiplicity-adjusted p-values, and limitations.**
- [ ] **Step 6: Re-run focused and full verification, inspect `git diff --check`, and commit only code, tests, documentation, and the three new result tables.**

