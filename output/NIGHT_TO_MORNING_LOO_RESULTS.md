# Leave-one-out check: is the H2 sleep/commuting-spillover effect driven by a few states/counties?

Source: `code/run_night_to_morning_leave_one_out.py` -> `output/tables/reg_night_to_morning_leave_one_out.csv`

## Why this check

The headline H2 ("sleep disruption travels with commuters") result
(`output/tables/reg_night_to_morning_window.csv`, robust spec) finds more
fatal crashes 06:00-23:59 in a work county as the commuting-weighted share
of its workforce living in a county with a nighttime AMBER alert in the
last two nights rises:

```
CROSS_SPILLOVER (own-controlled) -> fatals:
  beta = +0.030464, se = 0.011635, p = 0.0117, n = 7,348,614
```

Both sides of this exposure measure are concentrated in the same handful
of states that drove the H1 same-hour highway-fatals result: Georgia,
Texas, and North Carolina supply ~51% of night-alert county-days (the
spillover source) and, largely through within-state commuting, ~51% of
total cross_spillover exposure mass received (the work-county side).
Georgia alone is the single largest work-state recipient at 18.4% of all
spillover mass.

## Method

Same national county-day panel and specification as
`run_night_to_morning_window.py`'s robust spec (`night_alert` +
`cross_spillover` jointly, `fips_year + fips_dow + month_str` FE, two-way
state+date clustering). Same three checks as the H1 script:

1. **Leave-one-state-out** for all 51 states with nonzero spillover mass.
2. **Drop Georgia+Texas+North Carolina jointly** (the top 3 by spillover
   mass, ~51% combined).
3. **Leave-one-county-out** for the 10 highest spillover-mass work-counties.

(The original national grid construction OOM-killed a 15GB container when
re-estimated 60+ times with its default object-dtype fixed-effect columns;
the leave-one-out script casts them to pandas categoricals first, which
reproduces the exact baseline estimate above at ~140MB grid size / ~4GB
peak fit memory instead of >13.9GB.)

## Results

### 1. Leave-one-state-out (51 states)

- **50 of 51 state drops keep the coefficient positive and significant at
  p<.05** (range 0.024-0.036, p from 0.003 to 0.038).
- **Dropping Georgia alone flips it to non-significant**: coef=0.023228,
  se=0.011745, **p=0.0536** -- the coefficient shrinks by about 24% and
  crosses the conventional 5% threshold. Georgia is the single largest
  work-state recipient of spillover mass (18.4%).
- No other single state drop comes close to that: the next-largest shifts
  (dropping Tennessee, Texas, Alabama, Michigan, North Carolina, Nevada,
  California) move the coefficient by at most ~17% and all stay well under
  p=0.04.

### 2. Drop Georgia + Texas + North Carolina jointly (51% of spillover mass)

```
coef = +0.031989, se = 0.015811, p = 0.0488, n = 5,985,812 (160,214 nonzero-spillover rows, ~82% of baseline)
```

Landing almost exactly at the 5% boundary. This is a milder shrinkage than
dropping Georgia alone (the point estimate actually holds up slightly
better than Georgia-alone once Texas and North Carolina -- whose individual
removal each nudges the coefficient *up* slightly -- are removed alongside
it), but with only 82% of the identifying spillover observations left, the
wider standard error is enough to cross the threshold either way.

### 3. Leave-one-county-out (top 10 spillover-mass work-counties, 7 in GA, 2 in NC, 1 in TX)

Coefficients stay in a narrow 0.027-0.033 band and every one keeps p<.05
(range 0.006-0.014). No single county drives the result -- consistent with
the diffuse county-level distribution, even though the state-level picture
is fragile.

## Bottom line

The H2 sleep/commuting-spillover effect is **noticeably more fragile to
state exclusion than the H1 same-hour highway-fatals effect**. Where H1
survived dropping Texas alone (32% of that grid's alert-hours, by far the
single largest contributor) with p still under 0.003, H2's headline p=0.012
result **crosses the 5% significance threshold when Georgia alone (18.4%
of spillover mass, not even the largest single-state share seen in H1) is
excluded**, and sits right at the boundary (p=0.049) when the top three
states are dropped together. County-level robustness is fine -- no single
county matters -- but the state-level fragility here is a real and
distinguishing caveat: readers should treat the H2 commuting-spillover
finding as resting more heavily on a small number of states' data than the
H1 highway-fatals finding does, and as closer to the edge of conventional
significance under reasonable sample perturbations.
