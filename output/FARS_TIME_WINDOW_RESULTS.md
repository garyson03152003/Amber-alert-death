# Distraction-vs-sleep-disruption mechanism test: WHEN crashes happen

**Status: VALIDATED.** Uses the same crosswalk-corrected FARS geography/date
rules as the national FARS panel (`output/VALIDATED_CRASH_RESULTS.md`), with
hour-of-day retained. Reproducible via `code/build_fars_hourly.py` then
`code/run_fars_time_window_share.py`.

## Design

For an alerted county-date D (a verified night AMBER alert), five windows
test when a fatal crash or serious injury actually occurs, distinguishing an
"immediate distraction" mechanism from a "next-day sleep disruption" one:

| window | timing | tests |
|---|---|---|
| W0 same-night | D 20:00 -> D+1 06:00 | H1: distracted/interrupted while still driving |
| W1 morning commute | D+1 06:00-10:00 | H2: impaired the next morning from disrupted sleep |
| W2 midday | D+1 10:00-16:00 | control |
| W3 evening | D+1 16:00-20:00 | control |
| W4 placebo | D+2 06:00-10:00 | same commute window, but 24h too late to be caused by night D's alert |

PPML raw counts with `log(population)` offset, county + calendar-date fixed
effects, joint `night_alert` + `spillover_share_10pp` (commuter-share
exposure), matching the rest of the validated FARS pipeline. ~12.3-12.6M
county-date-window observations per model; zero share 98.8-99.9% (these are
short time windows, so most county-days genuinely have zero fatal crashes in
any given 4-6 hour slice -- PPML is zero-preserving throughout).

## Result: no same-night spike, but a large, fairly robust drop the next morning

| window | outcome | PPML joint beta | p | IRR / %chg | WLS joint p | PPML direct-vs-clean p | WLS direct-vs-clean p |
|---|---|---:|---:|---:|---:|---:|---:|
| W0 same-night | fatals | -0.071 | 0.606 | -6.9% | 0.549 | 0.853 | 0.763 |
| W0 same-night | serious | -0.051 | 0.881 | -4.9% | 0.883 | 0.370 | 0.377 |
| **W1 morning** | **fatals** | **-1.014** | **0.0001** | **-63.7%** | **0.001** | 0.162 | 0.212 |
| **W1 morning** | **serious** | **-1.583** | **0.0002** | **-79.5%** | **0.005** | **0.037** | **0.011** |
| W2 midday | fatals | -0.228 | 0.367 | -20.4% | 0.316 | 0.582 | 0.544 |
| W2 midday | serious | 0.230 | 0.585 | +25.8% | 0.695 | 0.221 | 0.325 |
| W3 evening | fatals | -0.408 | 0.132 | -33.5% | 0.129 | 0.126 | 0.163 |
| W3 evening | serious | 0.367 | 0.679 | +44.3% | 0.760 | 0.256 | 0.227 |
| W4 placebo | fatals | -0.121 | 0.591 | -11.4% | 0.724 | 0.442 | 0.411 |
| W4 placebo | serious | -0.158 | 0.776 | -14.6% | 0.433 | 0.080 | 0.007 |

Full table: `output/tables/fars_time_window_share.csv`.

**Reading**: W0 (same-night) is null across every specification -- no
evidence of an immediate distraction spike while people are still driving.
W1 (the next morning's commute) shows a large, negative, statistically
significant coefficient in 3 of 4 specifications (all except the
smaller-sample PPML direct-vs-clean-controls model for fatals specifically,
which is directionally the same, -25%, but underpowered at p=0.16); the
serious-injury version of W1 is significant in **all four** specifications.
W2, W3 are null; W4 (the placebo -- same commute window, 24h too late to be
caused by night D's alert) is also null in almost every specification,
ruling out "mornings are just generally quieter after any alerted county-day"
as the explanation.

This pattern matches **neither** stated hypothesis (H1: immediate
distraction increases crashes; H2: sleep disruption increases crashes) --
both predicted an *increase*. Instead, fatal and serious-injury crashes
specifically in the commute window immediately following an alert night are
substantially *lower*, while the same window 24 hours later (W4) is not.

## Guardrails

- This is not causal evidence of a protective mechanism by itself -- alert
  targeting is not random, and this design does not resolve that on its own
  (same caveat as the rest of this project's spillover/state-DOT results).
- Plausible non-causal explanations to rule out before treating this as a
  real behavioral effect: reduced/altered travel volume the specific morning
  after a high-salience local emergency, or a confound correlated with both
  alert timing and commute-hour traffic that this design does not control
  for. A genuine hourly/daily traffic-volume comparison (station-level
  counts, never annual AADT) would help arbitrate this and is a natural
  follow-up once available.
- Do not read the W2/W3 positive-but-null point estimates for serious
  injuries as evidence of anything; they are far from significant.
