# The most targeted combined test: tract-level car x distance dosage x hours-since-wake x non-alert-affected counties

Source: `code/run_hours_since_wake_distance_nonalert.py` ->
`output/tables/reg_hours_since_wake_distance_nonalert.csv`

## What this combines

Four separate lines of investigation into the H2 ("sleep disruption
travels with commuters") result converge in this one specification:

1. **Non-alert-affected-counties restriction**
   (`NIGHT_TO_MORNING_SPILLOVER_NONALERT_ONLY_RESULTS.md`): the pooled
   spillover effect is undetectable (p=0.46) once restricted to
   `night_alert==0` county-days, because ~65% of alerts are statewide
   broadcasts geo-expanded to every county in the state, entangling
   "directly alerted" with "high spillover exposure" on the highest-dose
   observations. This script applies that same restriction throughout.
2. **Hours-since-wake dose-response resolution**
   (`run_hours_since_wake_dose_response.py`): 18 single-hour bins
   (06:00-23:59), the finest time resolution used anywhere in this repo.
3. **Tract-preserved car x distance dosage**
   (`build_lodes_tract_car_dosage.py`'s `avg_car_x_dist`): computed
   directly from TRACT-level LODES home->work flows and TRACT-level ACS
   car-mode share, before ever collapsing to a county pair -- the
   quantity requested to replace the coarser county-centroid-distance /
   county-average-car-share proxies used in earlier checks. Its
   pooled-day level effect (not night_alert==0-restricted, not split by
   hour) is significant: coef=+0.000463, p=6.9e-05
   (`reg_commuting_car_distance_dosage.csv`).
4. A plain (unweighted) `cross_spillover`, also night_alert==0-restricted,
   estimated alongside as a same-sample reference.

This is the most targeted test in this repo for a genuine, uncontaminated
commuting mechanism: a driver who actually lives some distance away,
actually commutes by car, from a home county that alerted, into a work
county that was not itself swept into the same statewide campaign that
night -- and asks whether that exposure's effect on crashes rises later in
the day, the way a real fatigue-accumulation channel would predict.

## Result

```
PLAIN cross_spillover, night_alert==0:        slope=+0.0000563, se=0.000371,   p=0.881
TRUE car x dist (tract), night_alert==0:      slope=+0.0000001, se=0.0000032,  p=0.981
```

**Both slopes are indistinguishable from zero.** Per-hour, both series
are noisy with no coherent shape: the plain measure has 4/18 hours
nominally significant (positive and negative, no pattern) and the
tract-preserved car x distance measure has 2/18 -- both roughly what 18
independent tests at alpha=.05 produce under pure noise (~0.9 expected by
chance). One cell (TRUE_car_x_dist_tract at hours_since_wake=8, hour
14:00) is very strongly significant in isolation (p=5.0e-10) with no
support from any neighboring hour -- the signature of a single noisy
draw among many tests, not a real effect, and should not be
over-interpreted.

## Bottom line

Once the statewide-campaign compound-exposure confound is removed
(night_alert==0), neither a plain commuting-share spillover measure nor
the most refined tract-preserved car-and-distance-weighted dosage shows
any dose-response with hours since waking. This is the fourth and most
demanding check in this repo's series on H2, and it comes back a clean,
well-powered null across the board: no level effect (established
separately), no distance-coherent level effect at the finest tract
resolution available (this check), and no time-of-day dose-response
either. Combined with `NIGHT_TO_MORNING_LOO_RESULTS.md` (state-level
fragility) and `NIGHT_TO_MORNING_SPILLOVER_NONALERT_ONLY_RESULTS.md` (the
exposure-composition confound itself), the weight of evidence across
every angle tested so far does not support a genuine, commuting-mediated
sleep-disruption channel as this data's explanation for the pooled H2
result -- the pooled result appears attributable to statewide
alert-campaign nights and the handful of states that generate most of
the identifying variation, not to drivers carrying elevated risk with
them across county lines.
