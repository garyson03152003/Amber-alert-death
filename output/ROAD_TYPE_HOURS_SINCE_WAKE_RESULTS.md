# Does the hours-since-wake dose-response live specifically in highway crashes?

Source: `code/run_road_type_hours_since_wake.py` ->
`output/tables/reg_road_type_hours_since_wake.csv`

## The question

The pooled H2 dose-response test (`run_hours_since_wake_dose_response.py`)
finds a rising CROSS_SPILLOVER effect across the day -- an inverse-
variance-weighted meta-regression of 18 single-hour point estimates on
hours-since-wake:

```
Pooled (all roads): slope = +0.000461, se = 0.000175, p = 0.0180
```

A prior road-type check in this repo (`reg_road_type_split.csv`, commit
`86a2bef`) used only 4 coarse time-of-day windows and read the pattern as
confirming a highway-specific mechanism, but that read rested on a single
nominally-significant cell (p=0.035) that fell in a non-commute window
(19:00-24:00), not either actual commute window -- morning_commute was
p=0.105 and evening_commute was p=0.069, both non-significant. This script
redoes the test at the SAME single-hour resolution as the pooled
dose-response, separately for highway_fatals and nonhighway_fatals, to see
whether a genuine highway-specific dose-response emerges once road type
and fine time resolution are both held.

## Result

```
HIGHWAY CROSS_SPILLOVER:     slope = +0.000071, se = 0.000075, p = 0.360  (n.s.)
NON-HIGHWAY CROSS_SPILLOVER: slope = +0.000290, se = 0.000149, p = 0.070  (marginal)
```

**Neither road type reproduces the pooled dose-response on its own, and
non-highway is the closer of the two to conventional significance** --
the opposite of what a highway-driving-fatigue or cross-county-highway-
commuting mechanism would predict.

Per-hour, each road type has exactly 1 of 18 hours significant at p<.05,
and neither forms a coherent story:

- **Highway**'s only hit is hours-since-wake=15 (21:00 / 9pm):
  beta=+0.0062, p=0.024 -- not a commute-relevant hour.
- **Non-highway**'s only hit is hours-since-wake=4 (10:00 / 10am):
  beta=-0.0056, p=0.0003 -- highly significant but the WRONG sign (a
  *drop* in non-highway crashes mid-morning, not a rise).

Both look like isolated draws in a noisy 18-point series rather than
evidence of a genuine, monotonic dose-response within either road type.

## Bottom line

This is the third independent test in this repo to come back unfavorable
for a genuine highway-driving-fatigue channel behind the H2 result (after
the short/long commuting-distance split, where the effect required <21mi
pairs and vanished at >=21mi, and the coarse road-type x time-window
split, where the one significant cell fell outside both commute windows).
Splitting the pooled dose-response's rising slope by road type at full
hourly resolution does not isolate it to highway crashes -- if anything
the marginal signal that remains leans toward local roads. Combined with
`NIGHT_TO_MORNING_SPILLOVER_NONALERT_ONLY_RESULTS.md` (the effect is
undetectable once the recipient county's own same-night alert status is
held at zero), the weight of evidence across these checks points away
from "sleep-disrupted commuters carry elevated highway-driving risk into
their work county" as this data's explanation, and toward a shared local/
regional or statewide-campaign-night confound instead.
