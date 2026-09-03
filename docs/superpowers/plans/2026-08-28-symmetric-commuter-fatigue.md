# Symmetric Commuter-Fatigue Analysis Implementation Plan

**Goal:** Compare own-county and cross-county AMBER-alert exposure on identical commuter-share, driving, distance, and hours-awake scales, with two-way clustered analytic inference and state wild-cluster-bootstrap robustness.

**Architecture:** Add one import-safe analysis module. Pure helpers construct a tract-preserved car-distance dosage for every home/work county pair, split it into self-loop and cross-county exposure, and collapse the hourly outcome into both a 06:00–23:00 total and a pre-specified linear hours-since-wake contrast. A joint fixed-effects model estimates own and cross exposure together; one-way state wild-cluster bootstrap supplements the analytic state-plus-date clustered standard errors.

**Tech stack:** Python, pandas, numpy, scipy, pyfixest, pytest.

## Task 1: Lock the construction with tests

- Add tests proving that the self-loop uses `weight * avg_car_x_dist`.
- Add tests proving that cross exposure excludes the self-loop and adds alerted origins.
- Add tests proving that own and cross exposures use the same units.
- Add tests for the zero-sum linear hours-awake contrast and known linear hourly profiles.

## Task 2: Implement the import-safe analysis helpers

- Build and validate tract-preserved pair dosage.
- Construct symmetric own and cross county-day exposures.
- Build total and linear-contrast fatality outcomes from hours 06–23.
- Preserve zeros and fail closed on missing dosage inputs.

## Task 3: Estimate and bootstrap the joint models

- Fit own and cross dosage jointly with county-year, county-weekday, and month fixed effects.
- Report analytic CRV1 standard errors clustered by state and date.
- Run 9,999 null-imposed Rademacher wild-cluster-bootstrap draws by state.
- Test equality of own and cross coefficients with the same bootstrap procedure.

## Task 4: Run and review

- Run focused tests, then the full test suite.
- Materialize the tracked analysis data required by the sparse checkout.
- Run the new analysis and inspect both CSV outputs and exposure diagnostics.
- Record limitations clearly, especially that wild bootstrap is one-way by state while analytic inference is two-way by state and date.
