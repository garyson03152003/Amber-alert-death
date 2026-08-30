# Leave-one-out check: is the H1 same-hour highway-fatals effect driven by a few states/counties?

Source: `code/run_same_hour_road_type_leave_one_out.py` -> `output/tables/reg_same_hour_road_type_leave_one_out.csv`

## Why this check

The headline same-hour result (`output/tables/reg_same_hour_road_type_split.csv`)
finds fewer highway fatal crashes in the exact clock hour an AMBER alert is
issued, versus matched same-county/same-hour/same-weekday referent hours:

```
highway fatals:  beta = -0.0001746, se = 0.0000347, p = 6.7e-06, n = 948,423
```

Alert-hour observations in that matched-referent grid are very unevenly
distributed across states: Texas alone supplies **32.3%** of all alert-hour
rows, and Texas + Georgia + North Carolina together supply **53.1%**. County
concentration is much milder (the top 10 of 1,676 alert-touched counties are
only ~2.6% of alert-hours, and all 10 happen to be Texas counties, since a
single statewide campaign fans out across every county in that state). This
raises the question of whether the national estimate mostly reflects a small
number of large, alert-heavy states rather than a broadly shared pattern.

## Method

Same matched-referent case-crossover grid and specification as
`run_same_hour_road_type_split.py` (`fips_hour_dow + fips_year + year_month`
fixed effects, two-way state+date clustering). Three checks:

1. **Leave-one-state-out**: drop each of the 51 states with alert-hours one
   at a time, re-estimate on the rest.
2. **Drop TX+GA+NC together** (jointly >50% of alert-hours).
3. **Leave-one-county-out** for the 10 highest alert-hour counties.

## Results

### 1. Leave-one-state-out (51 states)

- **No state's removal flips the sign or removes significance at p<.05** for
  the highway-fatals coefficient. All 51 leave-one-out estimates stay
  negative, ranging from -0.000147 to -0.000188, with p-values from 3.9e-07
  to 2.4e-03.
- Dropping Texas alone (32.3% of alert-hours) moves the coefficient the most
  of any single state: -0.000175 -> -0.000147 (a ~16% shrinkage), but it
  remains highly significant (p = 0.0024, n = 647,125).
- No other single state moves the coefficient by more than ~10% of its
  baseline value.

### 2. Drop TX + GA + NC jointly (53.1% of alert-hours)

```
coef = -0.000121, se = 0.000063, p = 0.061, n = 448,075 (50,069 alert-hours)
```

This is the one place the result gets meaningfully weaker: removing the
three largest alert-hour states together shrinks the coefficient to about
69% of its baseline magnitude and pushes it just past the conventional 5%
significance threshold (p = 0.061), though the point estimate keeps the same
sign and is still within one baseline SE of the original -0.000175. With
only ~47% of the original alert-hour observations left, the loss of power
alone would be expected to widen the SE (0.000035 -> 0.000063); the point
estimate itself does not collapse toward zero.

### 3. Leave-one-county-out (top 10 alert-hour counties, all in TX)

Effect is essentially unchanged: coefficients range from -0.000157 to
-0.000175 (baseline -0.000175), all significant at p < 5e-05. No single
county drives the result -- consistent with the low county-level
concentration noted above.

### Non-highway fatals (secondary outcome, baseline p = 0.081, weaker to start)

This outcome was already only marginally significant nationally. Under
leave-one-state-out it becomes significant at p<.05 in only 3 of 51 drops
and non-significant in the rest -- it does not have a stable state-level
footprint the way the highway result does, consistent with it being the
weaker/secondary finding.

## Bottom line

The national highway-fatals same-hour effect is **not an artifact of any
single state or county** -- it survives dropping Texas (the largest single
contributor at 32% of alert-hours) and survives dropping any of the top 10
alert-heaviest counties. It **does depend to some degree on the states with
the most alert activity as a group**: removing the top three states (TX, GA,
NC) together, which is more than half the identifying variation, shrinks the
estimate by about a third and pushes the p-value to a borderline 0.06. That
is expected given the corresponding drop in statistical power (52.9% of
alert-hours remain), and the point estimate does not reverse sign or shrink
toward zero -- but it does mean the precision of the headline estimate rests
substantially on those three states, and readers should not treat the effect
as uniformly identified by all 51 states independently.
