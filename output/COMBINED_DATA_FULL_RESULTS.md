# Combined AMBER + missing-person data: full results across H1 and H2

Sources:
- `code/run_same_hour_road_type_split_combined.py` -> `reg_same_hour_road_type_split_combined.csv` (H1, road-type split, reported separately in `SAME_HOUR_COMBINED_DATA_RESULTS.md`)
- `code/run_same_hour_event_study_combined.py` -> `reg_same_hour_event_study_combined.csv` (H1, pooled all-road-types)
- `code/run_night_to_morning_window_combined.py` -> `reg_night_to_morning_window_combined.csv` (H2, sleep/commuting-spillover)

All three add the Silver-Alert-type missing-person data
(`02f_geocode_missing_person_alerts.py`, 1,418 unique alerts, 2014-2024,
found only via free-text screening since no dedicated IPAWS event code
existed for this population before September 2025) to AMBER as a single
combined "any WEA missing-person alert" treatment -- both of 02f's
population labels (`missing_person`, the elderly/adult population, and
`child_amber_adjacent`, missing minors caught by generic event codes)
are unioned in, per instruction to treat the latter as AMBER-equivalent
exposure.

## H1 same-hour "immediate distraction" test

### Road-type split (the sharper, headline spec)

```
                    AMBER-only baseline                Combined (+5,220 alert-hours, +2.8%)
highway fatals:     coef=-0.000175, p=7.0e-06           coef=-0.000166, p=5.0e-06
non-highway:        coef=+0.000346, p=0.081 (n.s.)      coef=+0.000323, p=0.104 (n.s.)
placebo (both):     n.s.                                n.s.
```

Replicates almost exactly, slightly *stronger* significance despite the
added treated hours being independently sourced. Full detail already in
`SAME_HOUR_COMBINED_DATA_RESULTS.md`.

### Pooled (all road types) -- for completeness

```
                    AMBER-only baseline                Combined
fatals:             coef=+0.000171, p=0.377 (n.s.)      coef=+0.000157, p=0.412 (n.s.)
serious injuries:   coef=-0.000154, p=0.081 (marginal)  coef=-0.000137, p=0.075 (marginal)
placebo (both):     n.s.                                n.s.
```

The pooled number was already null in the AMBER-only baseline and stays
essentially unchanged with combined data -- expected, since (verified
earlier in this repo as an exact OLS identity) the pooled all-road
coefficient is mathematically the SUM of the highway and non-highway
road-type coefficients, which partly offset (a real negative highway
effect plus an insignificant positive non-highway one). The road-type
split above is the informative test; the pooled number is included here
only for completeness against the outcome the road-type file can't
provide (serious injuries).

## H2 sleep/commuting-spillover test

```
                              AMBER-only baseline (robust spec)   Combined
OWN night_alert -> fatals:    coef=-0.003395, p=0.240 (n.s.)      coef=-0.003302, p=0.252 (n.s.)
CROSS_SPILLOVER -> fatals:    coef=+0.030464, p=0.0117            coef=+0.030040, p=0.0123
```

Essentially unchanged. The combined treatment only adds 497 additional
night-alert county-dates to AMBER's existing 34,807 (a 1.4% increase --
much smaller than the same-hour test's 2.8% boost in alert-*hours*,
since the missing-person data is smaller in volume and concentrated in
fewer of the night-window hours specifically), so this isn't a
meaningful new test of the H2 mechanism -- it mainly confirms the
combined night_alert/cross_spillover construction works correctly and
doesn't introduce any instability. All of the caveats already
established for the AMBER-only H2 result in this repo (state-level
fragility to dropping Georgia, the exposure-composition confound with
statewide alert campaigns, the requirement for implausibly short
commuting distances, and the clean null on the most targeted tract-
level car x distance x hours-since-wake x non-alert-affected-counties
test) apply equally to this combined number, since the added data
barely moves the estimate.

## Bottom line

Adding the independently-sourced missing-person data leaves the overall
picture unchanged in the way that matters: H1's highway-specific
same-hour effect replicates cleanly (a genuine strengthening of that
finding, given the new data was found through a completely different
mechanism), while H2's sleep/spillover result remains at essentially the
same fragile, confound-prone magnitude and significance it already had
with AMBER alone -- the missing-person data doesn't materially change
either conclusion, because its contribution to the night-time exposure
measure specifically (as opposed to the all-hours same-hour measure) is
small.
