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

Neither slope individually clears p<.05, and non-highway is nominally
closer to significance than highway. **On its own this comparison is NOT
sufficient evidence that the dose-response is a highway phenomenon** --
but a first pass at this repo mis-read that same fact as evidence the
pattern doesn't hold in either slice, which does not follow: "neither
half individually reaches p<.05" is not the same claim as "the effect is
concentrated in neither half."

An exact check makes this precise. Because every crash in
fars_road_type_county_day.parquet falls into exactly one of highway/
non-highway, and both regressions share the identical design matrix
(same regressors, same FE, same sample), OLS linearity means the pooled
per-hour coefficient MUST equal the highway coefficient plus the
non-highway coefficient at that hour -- and it does, exactly, for all 18
hours (max discrepancy ~3.6e-10, floating-point noise):

```
hours_since_wake=9:  highway +0.001374  +  non-highway -0.002433  =  pooled -0.000744  (all match run_hours_since_wake_dose_response.py)
...(all 18 hours match to floating-point precision)...
```

So the pooled dose-response is not merely "compatible with" both road
types -- it is their literal sum, hour by hour. Nothing about the split
makes the pooled pattern go away; both components are present and
contributing throughout.

**A direct test of whether the two slopes actually differ is the right
question, not whether each clears significance alone.** A weighted
regression of all 36 (road_type x hour) point estimates on
`hours_since_wake * is_highway`, weighted by inverse variance, gives:

```
hours_since_wake (non-highway slope):  coef=+0.000290, se=0.000139, p=0.044
is_highway (level shift):              coef=+0.002334, se=0.001374, p=0.099
hsw x is_highway (SLOPE DIFFERENCE):   coef=-0.000219, se=0.000161, p=0.183
```

The highway-vs-non-highway slope difference is **not statistically
significant (p=0.183)** -- the two road types' dose-response slopes are
not distinguishable from each other. And directionally, both rise from
the first half of the day to the second:

```
                       hours-since-wake 0-8 avg     9-17 avg
Highway coef:          +0.0007                       +0.0020   (rising)
Non-highway coef:      -0.0005                       +0.0012   (rising)
```

Per-hour, each road type also has exactly 1 of 18 hours significant at
p<.05 on its own (highway: hour 21:00/9pm, beta=+0.0062, p=0.024 --
not commute-relevant; non-highway: hour 10:00/10am, beta=-0.0056,
p=0.0003 -- wrong sign), neither of which should be read as more than a
single noisy draw in an 18-point series either way.

## Bottom line (corrected)

**The rising-with-hours-since-wake pattern looks like a real, broadly
shared feature of the data, present in both highway and non-highway
crashes and not statistically distinguishable between them** -- not
something road-type splitting makes vanish, once the comparison is done
properly (a joint interaction test, not "does each half individually
clear p<.05"). That is NOT evidence *for* the specific highway-commuting
mechanism proposed for H2, since the whole point of that mechanism is
that the effect should concentrate in highway driving specifically, and
this test cannot confirm that it does -- but it is also not evidence
*against* there being some genuine time-of-day pattern in the data, which
an earlier version of this document overstated by treating "not
individually significant in an underpowered half-sample" as if it were
"absent." The distance-split result (effect present at <21mi commuting
pairs, null at >=21mi, see `NIGHT_TO_MORNING_SPILLOVER_NONALERT_ONLY_
RESULTS.md`'s sibling `reg_commuting_distance_robustness.csv`) and the
non-alert-affected-counties restriction remain the stronger pieces of
evidence bearing on whether H2 reflects a genuine commuting-mediated
mechanism; this road-type x hours-since-wake check is better read as
inconclusive on road-type attribution specifically, rather than as
additional evidence against H2 overall.
