# Commuter Identification Improvements Design

## Goal

Strengthen the causal interpretation of the year-matched commuter-car-mile
spillover estimate without changing or overwriting the existing headline
specification.

## Scope

This addition produces three separately labelled robustness families:

1. a within-state, within-date specification that absorbs every state-day
   shock before estimating own- and cross-county exposure effects;
2. a degree-preserving commuting-network falsification that keeps alert dates,
   alert origins, self loops, edge dosages, and state-pair edge counts fixed but
   breaks the observed origin-to-destination links; and
3. a pre-specified time-of-day analysis using the same year-matched tract-level
   car-share-by-distance dosage as the total-outcome model.

The existing headline and robustness CSV files remain unchanged. New results
use distinct output paths and explicitly identify observed-network estimates,
placebo draws, fixed effects, inference, and outcome windows.

## Identification specifications

### State-by-date fixed effects

Estimate fatal-crash counts and person fatalities from 06:00 through 23:59
with own and cross commuter-car-mile exposures entered jointly. Absorb
county-by-year, county-by-weekday, and state-by-date fixed effects. Continue to
report two-way state/date clustered analytic standard errors and a state-level
Webb wild-cluster p-value. Calendar month is not included because it is nested
in state-by-date.

This specification is identified by differences across destination counties
within a state on the same date. It therefore absorbs holidays, statewide
weather or enforcement shocks, statewide alerts, and any other common
state-day disturbance.

### Commuting-network falsification

For each ACS/LODES vintage and each home-state/work-state block, preserve every
cross-edge row and its commuter-car-mile dosage while shuffling destination
county labels. Preserve self loops exactly. Repair any shuffled same-county
assignment by swapping destination labels, and aggregate duplicate placebo
edges if a shuffle produces them.

Each placebo draw uses the same alerts, dates, county sample, outcomes, and
fixed-effects specification as the observed-network model. The two-sided
randomization p-value is `(1 + number(|beta_placebo| >= |beta_observed|)) /
(1 + draws)`. The primary falsification outcome is the 06:00--23:59 fatal-crash
count; person fatalities are secondary. The production default is 199 seeded
draws, while tests use small in-memory examples.

### Rich time-of-day mechanism

Aggregate sparse FARS hourly fatalities into four pre-specified average-hour
outcomes so coefficients are comparable despite unequal block lengths:

- morning commute: 06:00--09:59;
- midday: 10:00--14:59;
- evening commute: 15:00--19:59; and
- late evening: 20:00--23:59.

Also construct the zero-sum late-evening-minus-morning contrast. Estimate each
outcome with the same joint own/cross year-matched commuter-car-mile exposure.
Use the state-by-date fixed-effects specification as the primary version and
the established county-year/county-weekday/month specification as a labelled
sensitivity. Apply Holm adjustment separately to the four own coefficients and
the four cross coefficients; the pre-specified contrast is reported separately
and is not included in that family.

## Reproducibility and safeguards

- Use the existing ACS 2015/2020 and LODES 2013/2018/2022 year mapping.
- Use `avg_car_x_dist`; do not replace it with a product of separate averages.
- Keep structural-zero county-dates.
- Use deterministic random seeds and record the requested and completed draw
  counts.
- Preserve the user's untracked WEA source and control files.
- Never overwrite the current headline result tables.
- Unit-test each new pure transformation before wiring it into the production
  runner.

## Outputs

- `output/tables/reg_symmetric_commuter_identification.csv`
- `output/tables/symmetric_commuter_network_placebo.csv`
- `output/tables/reg_symmetric_commuter_time_blocks.csv`
- an updated interpretation section in `docs/symmetric_commuter_robustness.md`

