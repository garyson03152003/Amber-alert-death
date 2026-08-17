# Traffic-Volume Mechanism Analysis Instructions

> **Purpose:** Test whether nighttime AMBER alerts alter next-day driving exposure enough to explain any estimated change in crashes.

This is a mechanism/interpretation exercise for the existing AMBER-alert crash analysis. The key question is whether traffic volume changes after a nighttime alert. A tightly estimated near-zero traffic-volume effect would make a simple driving-exposure explanation less plausible; it would not by itself prove a sleep channel.

## Hard data-resolution requirement

**Do not use annual AADT, annual VMT, yearly traffic totals, monthly Traffic Volume Trends aggregates, or any other temporally aggregated traffic measure as the main treatment outcome.** Those data are too coarse for a nighttime-alert design.

The required raw data are:

```text
station x calendar-date x hour traffic counts
```

Daily outcomes may be constructed **only by summing valid hourly observations for that station-day**. If hourly/daily raw counts are unavailable for a location/year, mark that observation unavailable rather than substituting annual or monthly traffic data.

The preferred national backbone is FHWA TMAS continuous-count data. FHWA may distribute/archive these files by month or year, but the underlying observation must still contain station-date-hour traffic volume (or a station-day record with 24 hourly count fields). Verify this before using any file.

---

## Research logic

Existing work on hazard-related Wireless Emergency Alerts shows that alerts can alter mobility and traffic exposure. For AMBER alerts, the underlying emergency is generally not itself a road hazard, so traffic volume is a useful mechanism test.

The pattern most informative against a pure mobility/exposure explanation is approximately:

```text
nighttime AMBER alert
    -> next-morning traffic volume approximately unchanged
    -> next-morning crashes change materially
```

If traffic volume changes materially, raw crash counts cannot be interpreted as changes in driving risk without accounting for that exposure change.

Do not force a sleep interpretation. A traffic-volume response is itself an important behavioral result.

---

## Preferred data source: FHWA TMAS continuous counters

Start with Federal Highway Administration Traffic Monitoring Analysis System (TMAS) continuous-count data.

Source page:

```text
https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/
```

Preserve the finest available temporal resolution.

Minimum fields to retain where available:

```text
state
station_id
county_fips
latitude
longitude
date
hour
traffic_volume
road / functional class
direction if separately reported
source
```

If county FIPS is absent or unreliable, recover county assignment from validated station coordinates rather than fuzzy place-name matching.

Where TMAS coverage is poor, state DOT permanent/continuous counter data may supplement it. Keep TMAS as the common national baseline and clearly label state-specific extensions.

Do not silently combine incompatible definitions such as directional counts, combined directions, lane counts, station totals, estimates, or observed counts.

---

## Task 1: Build a station-hour traffic panel

Suggested files:

```text
code/traffic_volume/
    download_tmas.py
    parse_tmas.py
    build_station_hour_panel.py
```

Preferred output:

```text
data/processed/traffic/tmas_station_hour.parquet
```

Minimum columns:

```text
state
station_id
county_fips
date
hour
traffic_volume
latitude
longitude
source
```

Preserve valid zero counts. Distinguish valid zeros from missing, suppressed, counter-offline, or corrupted observations.

### Quality checks

- [ ] Identify duplicate station-date-hour rows.
- [ ] Flag impossible/corrupted counts.
- [ ] Check long missing-data gaps and station entry/exit.
- [ ] Check whether station location/coding changes over time.
- [ ] Do not interpolate missing hourly traffic for the main specification.
- [ ] Record active stations by state and year.
- [ ] Record the share of treated county-days with at least one usable continuous counter.

Save:

```text
output/tables/traffic_counter_coverage.csv
```

---

## Task 2: Merge nighttime AMBER treatment

Use the same verified nighttime AMBER-alert treatment as the preferred crash analysis.

For each station in county `c`, attach:

```text
night_alert_ct
alert_time_local
alert_hour_local
spillover_share_ct
exposure_class
```

Align alerts and counter observations in **local time**. Preserve daylight-saving transitions correctly.

For an alert in the project's nighttime window (currently 10 pm-5 am), define the following morning/day consistently with the crash analysis.

The main traffic-volume treatment is direct county exposure. Analyze commuter spillover separately rather than silently mixing the two exposures.

---

## Task 3: Construct traffic outcomes

At minimum construct:

```text
1. total next-day station traffic volume
2. 05:00-10:00 next-morning volume
3. 07:00-10:00 morning-commute volume
4. 10:00-16:00 midday volume
5. 16:00-19:00 evening volume
6. hourly volume in an event-study window around alert issuance
```

Also consider normalized station outcomes relative to typical traffic for the same hour-of-week, but do not use normalization to replace the raw-count specification.

For county-level comparisons, aggregate available counters carefully. Unless a defensible expansion procedure exists, call the result a **monitored traffic-volume index**, not county VMT.

---

## Task 4: Preferred estimation

### Station-hour model

A starting specification is:

```text
log(volume_scth)
    = beta * night_alert_ct
    + station-by-hour-of-day fixed effects
    + calendar-date-by-hour fixed effects
    + error_scth
```

Use an alternative count model if valid zero volumes are common; do not mechanically use `log(1 + volume)` without justification.

Inference must reflect treatment assignment. County clustering is a minimum. Consider alert-event clustering or multiway clustering because one alert may treat many counties/stations.

### Station-day model

For total next-day and morning-window volume, estimate station-day models with station fixed effects and rich calendar fixed effects, matching the crash design as closely as possible.

### Hourly event study

Construct approximately `-12` through `+36` hours around issuance.

Show whether traffic changes:

```text
- immediately after issuance
- during the remaining nighttime hours
- during the next morning commute
- later the next day
```

Plot confidence intervals and report contributing stations/events by relative hour.

---

## Task 5: Equivalence / bounds test

**Do not conclude that AMBER alerts do not affect traffic because `p > 0.05`.**

The mechanism test must quantify how large a traffic response can be ruled out.

Before interpretation, choose a substantively meaningful equivalence margin. Possible approaches:

```text
A. pre-specify a narrow bound such as +/-2%
B. benchmark against traffic responses found for hazard WEAs
C. derive the traffic-volume change required to mechanically explain the estimated crash effect
```

Report 90% and 95% confidence intervals. When practical, run a formal two-one-sided-tests (TOST) equivalence test and report the chosen bound explicitly.

Example interpretation:

```text
next-morning traffic effect = -0.2%
95% CI = [-0.9%, +0.5%]

If crashes increase by 5%, the interval rules out a traffic-volume change large enough
to explain most of the crash increase through exposure alone.
```

---

## Task 6: Compare crashes with traffic exposure

Where coverage permits, construct exploratory exposure-adjusted measures such as:

```text
crashes per monitored traffic volume
fatal crashes per monitored traffic volume
```

Do not call these literal crash rates per VMT unless traffic counts have been validly expanded to county VMT.

Preferred hierarchy:

```text
1. show raw crash effect
2. show traffic-volume effect
3. show whether plausible volume changes can explain the crash effect
4. then show exposure-adjusted crash measures as supporting evidence
```

---

## Task 7: Time-of-night heterogeneity

Use the same pre-specified nighttime bands as the crash analysis where possible.

A sleep-disruption interpretation becomes more plausible if:

```text
deeper/later-night alerts -> larger next-morning crash effect
AND
deeper/later-night alerts -> little or no corresponding decline in traffic volume
```

If later alerts instead substantially reduce morning traffic, treat that as evidence of mobility/behavioral adjustment.

Avoid specification mining across many arbitrary hour bins.

---

## Task 8: Commuter spillovers

For non-alerted destination/work counties, examine whether commuter exposure predicts both traffic and crashes:

```text
spillover_share_ct -> destination traffic volume
spillover_share_ct -> destination crashes
```

Potentially informative pattern:

```text
higher exposed commuter share
    -> destination traffic volume approximately unchanged
    -> destination crashes increase
```

This would be consistent with an effect traveling with exposed drivers rather than being generated solely by local conditions in the destination county.

Do not call the commuter-spillover design causal without further identification work.

---

## Required outputs

Create at least:

```text
output/tables/traffic_counter_coverage.csv
output/tables/traffic_volume_main.csv
output/tables/traffic_volume_equivalence.csv
output/figures/traffic_volume_event_study.pdf
output/figures/traffic_volume_event_study.png
output/TRAFFIC_VOLUME_RESULTS.md
```

Recommended columns for the main results table:

```text
sample
outcome
window
term
beta
se
pvalue
ci_low_95
ci_high_95
pct_change
n_obs
n_stations
n_counties
n_alert_events
```

The equivalence table should also include:

```text
equivalence_bound
TOST_pvalue_lower
TOST_pvalue_upper
equivalent_at_bound
```

---

## Interpretation relative to prior WEA work

Ferris & Newburn (2017), *Journal of Environmental Economics and Management*, studies flash-flood Wireless Emergency Alerts and automobile accidents and finds evidence that alerts reduced traffic exposure. Use that paper to motivate measuring traffic here, while keeping the conceptual distinction clear:

```text
hazard WEA:
alert contains information directly relevant to road/travel risk
-> protective mobility response can reduce accidents

AMBER WEA:
underlying child-abduction event is generally not a county-wide road hazard
-> next-day mobility response is not mechanically required
```

The contribution is not that wireless alerts have never been linked to traffic outcomes. The relevant question is whether **nighttime, non-road-hazard alerts generate next-day safety effects that cannot be explained by changes in driving exposure**.

---

## Guardrails

- **Never substitute annual AADT/VMT or monthly aggregates for hourly/daily continuous-counter data.**
- Do not describe an insignificant volume coefficient as proof of no effect.
- Use confidence intervals and equivalence bounds.
- Do not call sparse counter data county VMT.
- Do not drop treated county-days because no counter exists; report coverage/selection instead.
- Do not interpolate missing counter data in the main analysis.
- Do not interpret a negative crash coefficient as safer driving without checking traffic exposure.
- Do not interpret a positive crash coefficient as sleep impairment if traffic exposure also rises.
- Preserve local-time alignment and DST handling.
- Keep direct-alert and commuter-spillover mechanisms conceptually separate.
- Traffic-volume evidence is a mechanism test; it does not solve endogenous AMBER-alert targeting.

## Result-note requirements

`output/TRAFFIC_VOLUME_RESULTS.md` should report:

1. source(s) and years;
2. number of stations, counties, states, and alert events;
3. treated-county coverage rate;
4. data-quality exclusions;
5. next-day total traffic-volume estimate;
6. 05:00-10:00 and 07:00-10:00 traffic estimates;
7. hourly event-study pattern;
8. confidence intervals and equivalence-test results;
9. whether plausible traffic changes could explain the crash estimate;
10. commuter-spillover traffic results, if estimable;
11. limitations of station counts as a proxy for county VMT.

The most important comparison is:

```text
estimated % change in crashes after nighttime AMBER alert
versus
estimated % change in next-morning traffic volume after nighttime AMBER alert
```

If traffic volume is tightly estimated near zero while crashes change materially, describe that as evidence against a simple driving-exposure explanation, not definitive proof of sleep disruption.
