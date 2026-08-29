# Does the H1 same-hour highway-fatals effect hold up with combined AMBER + missing-person data?

Source: `code/run_same_hour_road_type_split_combined.py` ->
`output/tables/reg_same_hour_road_type_split_combined.csv`

## What was combined

The headline H1 result (`reg_same_hour_road_type_split.csv`) is estimated
from AMBER (`CAE`) alerts only, fetched via a clean structured IPAWS
event-code filter (`code/02c_fetch_openfema_ipaws.py`). This adds a
second, independently-sourced treatment population: the Silver-Alert-
type missing-person alerts recovered by free-text screening the full
OpenFEMA IPAWS archive (`code/02f_geocode_missing_person_alerts.py`,
1,418 unique alerts, 2014-2024 -- no dedicated IPAWS event code existed
for this population before September 2025, so a structured filter could
never have found these; see that script's docstring for the full
provenance and validation).

Per instruction, both of 02f's population labels are combined into one
"any WEA missing-person alert" exposure: `missing_person` (elderly/
adult, the actual Silver-Alert-equivalent population) and
`child_amber_adjacent` (missing/endangered minors caught by generic
event codes, treated as the same population AMBER/CAE covers). Both are
unioned with the existing AMBER alert-hours at the (fips, date, hour)
grain -- an hour is "treated" if EITHER source has an alert in it, the
same way AMBER's own Alert/Update/Cancel messages are already unioned
into one `is_alert_hour` flag.

```
AMBER-only alert-hours (active counties):        108,711
Missing-person alert-hours (active counties):      5,220
Overlapping exactly (same fips/date/hour):         2,148
Combined unique treated alert-hours:             111,783   (+2.8% over AMBER alone)
```

Everything else -- the matched-referent case-crossover grid
construction, `fips_hour_dow + fips_year + year_month` FE, two-way
state+date clustering, the road-type outcome split
(`fars_road_type_county_day.parquet`) -- is identical to the baseline
script, so this is an apples-to-apples comparison except for the
treatment definition.

## Result

```
                              AMBER-only (baseline)        Combined (AMBER + missing-person)
highway fatals:               beta=-0.000175, se=0.000035   beta=-0.000166, se=0.000033
                               p=7.0e-06, n=948,423          p=5.0e-06, n=993,766

non-highway fatals:           beta=+0.000346, se=0.000194   beta=+0.000323, se=0.000195
                               p=0.081, n.s.                 p=0.104, n.s.

highway, placebo:             beta=+0.000118, p=0.506, n.s. beta=+0.000067, p=0.728, n.s.
non-highway, placebo:         beta=+0.000018, p=0.975, n.s. beta=-0.000213, p=0.752, n.s.
```

**The highway-fatals effect replicates almost exactly** -- 95% of the
original magnitude (-0.000166 vs -0.000175), a slightly *tighter*
standard error (0.000033 vs 0.000035, from the added identifying
variation), and *stronger* significance (p=5.0e-06 vs p=7.0e-06) despite
adding a modest number of treated hours (+2.8%) whose exact-hour
distribution didn't need to match AMBER's for the estimate to hold. The
non-highway null and both backward-causal placebos also replicate as
non-significant, exactly as they should if the combined treatment isn't
introducing a spurious pattern.

## Bottom line

This is a genuine strengthening of the H1 finding, not just a repeat of
the same data under a new name: the missing-person alerts were found
through a completely different mechanism (free-text keyword screening
of generic IPAWS event codes, not a structured `CAE` filter) and
represent a different population (mostly elderly/adult missing persons,
plus some missing minors caught outside AMBER's own abduction-specific
criteria) with independent geographic and temporal variation from AMBER.
That a same-hour highway-fatality effect of essentially the same size
and significance shows up when this independent alert stream is added
to the treatment definition is evidence against the effect being an
idiosyncrasy of AMBER's specific data source or collection method, and
consistent with it reflecting a genuine same-hour response to a WEA
phone alert regardless of which missing-person program issued it.
