# Is the H1 same-hour highway-fatals effect a DUI-deterrence artifact?

Sources:
- `code/run_same_hour_dui_split.py` -> `output/tables/reg_same_hour_dui_split.csv` (plain drunk/sober, no road-type control)
- `code/build_fars_road_type_dui.py` + `code/run_same_hour_road_type_dui_split.py` -> `output/tables/reg_same_hour_road_type_dui_split.csv` (road type x alcohol involvement, crossed)

## The question

The headline H1 result (`reg_same_hour_road_type_split.csv`): fewer
highway fatal crashes in the exact clock hour an AMBER alert is issued,
versus matched same-county/same-hour/same-weekday referent hours:

```
highway fatals: beta = -0.0001746, se = 0.0000347, p = 6.7e-06, n = 948,423
```

An active AMBER alert search plausibly brings heightened police
presence and traffic stops to an area -- exactly the conditions that
would suppress DUI-related crashes specifically via enforcement/
deterrence, independent of any driver-attention mechanism. If the
same-hour effect is really a DUI-deterrence artifact, it should
concentrate in `drunk_fatals` and be weak/absent in `sober_fatals`.

## First pass: plain drunk/sober split (no road-type control)

```
drunk fatals: beta=+0.000107, se=0.000066, p=0.114  (n.s.)
sober fatals: beta=+0.000064, se=0.000169, p=0.705  (n.s.)
```

Neither is significant, and -- importantly -- **both have the wrong
sign** relative to the highway-only headline effect (both positive, vs.
the headline's negative). This is expected, not a contradiction: pooling
highway and non-highway crashes together dilutes/offsets the negative
highway-specific effect with the positive (non-significant) non-highway
coefficient already documented in `reg_same_hour_road_type_split.csv`
(`+0.000346`, p=0.081). A DUI split that doesn't also isolate road type
doesn't actually test the mechanism behind the highway-specific result.

## Second pass: road type x alcohol involvement, crossed

```
highway,    drunk:  beta=-0.000023, se=0.000027, p=0.393  (n.s.)
highway,    sober:  beta=-0.000152, se=0.000033, p=0.0003 (***)
non-highway,drunk:  beta=+0.000130, se=0.000075, p=0.090  (*)
non-highway,sober:  beta=+0.000216, se=0.000163, p=0.191  (n.s.)
```

(Exact-additivity check: highway_drunk + highway_sober = -0.000023 +
-0.000152 = -0.000175, matching the pooled highway headline coefficient
to the fourth decimal -- the same OLS-linearity identity verified for
the H2 road-type split, confirming this decomposition is exact, not
approximate.)

**The entire highway effect lives in sober driving.** `highway_sober` is
large, precisely estimated, and highly significant (p=0.0003) -- nearly
identical in magnitude to the pooled highway headline number on its own.
`highway_drunk`, meanwhile, is small and not remotely significant
(p=0.393). This is the **opposite** of what a DUI-deterrence/enforcement
story predicts: if heightened policing during an active search were
driving the highway result, the effect should concentrate in
alcohol-involved crashes, not sober ones.

Backward-causal placebo check (does tomorrow's alert "predict" today's
crashes, controlling for today's real one): `highway_sober, placebo` is
null (p=0.415), consistent with the real `highway_sober` effect being
genuine rather than a pre-existing trend artifact.

## Bottom line

The H1 same-hour highway-fatals effect is **not a DUI-deterrence
artifact** -- it holds specifically for sober driving and is essentially
absent in drunk driving, the reverse of what heightened
enforcement/traffic-stop activity during an active AMBER alert search
would predict. This strengthens the case for the effect reflecting a
genuine driver-attention/vigilance mechanism among ordinary (sober)
drivers on highways during an active search, rather than a confound tied
to policing intensity. Combined with the earlier road-type finding
(concentrated in highway, not local-road, driving) and the leave-one-out
robustness results, H1 continues to hold up well under every
mechanism-oriented check applied so far -- in contrast to H2, which did
not survive any of the analogous checks.
