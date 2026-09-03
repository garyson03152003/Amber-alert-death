# Symmetric commuter-fatigue robustness checks

The six requested checks are implemented in
code/run_symmetric_commuter_robustness.py. All specifications use the
year-matched ACS/LODES commuter-car-mile dosage, county-year, county-weekday,
and calendar-month fixed effects, and the validated FARS 06:00--23:00 outcome
window.

The final run used 9,999 Webb six-point wild draws, 199 state-month
Rademacher sign draws, and all 50 active-state leave-one-out regressions. It
produced 160 estimate rows in
output/tables/reg_symmetric_commuter_robustness.csv.

## Headline sleep-window integration

The headline `run_night_to_morning_window.py` now retains the original
binary-alert and ACS county-share specifications but also reports a
year-matched tract-preserved commuter-car-mile specification. For each
observed LODES home-tract/work-tract flow, the dosage carries the home
tract's ACS car share and the tract-centroid distance through aggregation;
the resulting `avg_car_x_dist` is multiplied by the destination county's ACS
commuter share. The rich specification is identified in
`output/tables/reg_night_to_morning_window.csv` by
`exposure_spec=year_matched_tract_car_distance`, with construction diagnostics
in `output/tables/night_to_morning_commuter_car_exposure_summary.csv`.

The direct same-hour alert analysis is intentionally unchanged. Commuter-car
miles are a mechanism/dosage measure for the next-morning sleep window, not a
claim that every LODES worker was physically driving at the alert hour.

## Results

| Check | Cross-county result | Interpretation |
| --- | --- | --- |
| Baseline fatal-crash count | β=0.000322, two-way p=0.0247, Webb p=0.0255 | Positive cross-county association; own β=0.000110, p=0.853 |
| Baseline person fatalities | β=0.000560, two-way p=0.0112, Webb p=0.0181 | Same sign and stronger than crash counts; own β=-0.000196, p=0.761 |
| State-month sign randomization | crash p=0.055; fatality p=0.030 | More conservative block-level inference; crash-count result is borderline |
| Daily event-time bins | Leads are null; fatality cross 0--2 days β=0.000288, p=0.094 | No pre-trend signal, but no monotone post-alert profile |
| Backward-date placebos | cross p=0.907, 0.673, 0.496 for −1, −2, −7 days | Supports temporal direction |
| Daytime-alert placebo | cross β=0.000002, p=0.984 | No cross-county daytime placebo effect |
| Scope sensitivity | County-only β=0.000797, p=0.0125; statewide-only β=0.000192, p=0.278 | Statewide alerts attenuate the estimate |
| Positive-tail trim | β=0.000408, p=0.172 | Sign survives, precision does not |
| Leave-one-state-out | All 50 coefficients positive; β range 0.000266--0.000537 | Sign is not driven by one state, though the largest p-value is 0.126 |
| Nonlinear bins | Positive cross quantiles are not monotone; q2 p=0.066, q3 p=0.051 | Evidence does not support a clean dose-response |

## Criticism and limitations

* The event study is a **daily** distributed-lag check. It is not the
  earlier hourly event study, so it should not be described as identifying
  hour-by-hour fatigue dynamics.
* Webb and Rademacher bootstrap p-values are one-way score resampling
  procedures. The headline analytic standard errors are two-way clustered by
  state and date; the state-month sign test is a separate, conservative
  block-level check rather than an exact permutation of alert timing.
* The crash-count outcome is the number of validated fatal crashes, not an
  all-crash or injury count. The person-fatality comparison uses the same
  06:00--23:00 window.
* Structural zeros are retained. The zero-vs-positive model uses zero as the
  reference category; positive quantile bins are formed only among positive
  exposure observations.
* The local verification environment lacked pyfixest, so the saved run
  records estimator=numpy_within_ols and uses an equivalent within-OLS
  two-way CRV1 fallback. The pinned analysis environment in
  requirements-analysis.txt will use pyfixest automatically.

## Within-state/date and network-identification checks

The additional identification runner is
`code/run_symmetric_commuter_identification.py`. It writes separate tables and
does not replace the earlier headline or robustness outputs. The state-date
and time-block results below use 9,999 Webb wild-cluster draws. The network
falsification currently uses 19 seeded degree-preserving rewires, so its
finite-sample tail probability has coarse 0.05 resolution and should be
described as an informative falsification rather than a final randomization
test.

### State-by-date fixed effects

Adding state-by-date fixed effects removes the positive aggregate cross-county
estimate:

| Outcome | Own coefficient | Cross coefficient | Cross analytic p | Cross Webb p |
| --- | ---: | ---: | ---: | ---: |
| Fatal crashes, 06:00--23:59 | -0.002488 | -0.000047 | 0.743 | 0.796 |
| Person fatalities, 06:00--23:59 | -0.003079 | 0.000194 | 0.355 | 0.598 |

These models absorb county-year, county-weekday, and state-date fixed effects.
The earlier positive result is therefore not recovered from differences among
destination counties in the same state on the same date. This does not prove
the baseline association is spurious: state-date effects also remove all
statewide-alert variation and may discard part of the relevant treatment.
It does mean the stronger within-state/date causal comparison does not support
the headline positive spillover.

### Degree-preserving commuting-network falsification

Each placebo keeps alert origins and dates, self loops, edge dosages, and the
origin and destination degree sequences fixed within home-state/work-state
blocks. Only the cross-county links are rewired.

| Outcome | Observed cross coefficient | Placebo mean | Placebo SD | Observed percentile | Upper-tail p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fatal crashes | -0.000047 | 0.000682 | 0.000263 | 0/19 | 1.00 |
| Person fatalities | 0.000194 | 0.000939 | 0.000320 | 0/19 | 1.00 |

All 19 rewired-network coefficients exceed the observed-network coefficient
for both outcomes. Thus the actual commuting links do not generate a stronger
effect than generic networks with the same degree structure. This is evidence
against interpreting the rich weight result as a commuting-path mechanism;
the weighting construction can generate positive estimates even after the
origin-destination links are broken.

### Pre-specified rich time blocks

The four average-hour blocks are 06:00--09:59, 10:00--14:59,
15:00--19:59, and 20:00--23:59. Holm correction is applied within the four
own and four cross families. Under state-date fixed effects, no cross-county
block survives the correction or the wild bootstrap. The late-evening-minus-
morning cross contrast is 0.000032 (analytic p=0.058; Webb p=0.218).

The baseline fixed-effects model has a positive own-county late-evening block
(0.000216; Holm p=0.0097; Webb p=0.0209), but its late-minus-morning contrast
does not survive the wild bootstrap (p=0.108), and the late-evening own effect
is null after adding state-date fixed effects. The state-date model instead
has a negative own-county midday coefficient. Taken together, these patterns
do not provide a coherent increasing-hours-awake or commuting-time mechanism.

The new outputs are:

- `output/tables/reg_symmetric_commuter_identification.csv`;
- `output/tables/symmetric_commuter_network_placebo.csv`; and
- `output/tables/reg_symmetric_commuter_time_blocks.csv`.
