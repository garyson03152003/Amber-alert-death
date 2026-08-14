# Local Analysis Handoff

> **For agentic workers:** Use the repository as-is on branch `claude/review-handoff-docs-OxJVr`. Do not revert the recent estimator/treatment fixes. Work through the checklist in order and record the resulting estimates clearly.

**Goal:** Run and validate the corrected AMBER-alert state-DOT analysis locally, with zero-preserving PPML and commuter-share spillover exposure, then compare the new results with the stale results currently described in PR #2.

**Current branch:** `claude/review-handoff-docs-OxJVr`

**Current preferred runner:** `code/run_state_dot_analysis_share.py`

**Important recent files:**
- `code/state_dot_analysis_core.py`
- `code/run_state_dot_analysis_fixed.py`
- `code/run_state_dot_analysis_share.py`
- `tests/test_state_dot_analysis_core.py`
- `code/build_commuting_weights.py`

## Global constraints

- Do **not** use `log(1 + commuters)` as the main spillover regressor.
- Preferred spillover intensity is the share of workers in destination county `i` whose home counties received the nighttime AMBER alert.
- `spillover_share_10pp = spillover_share / 0.10`, so its coefficient is the effect of a 10 percentage-point increase in exposed commuter share.
- Direct treatment remains `night_alert` for counties explicitly targeted by a verified county-level nighttime alert.
- Non-alerted counties with `spillover_share > 0` are **not** clean controls.
- Missing/non-comparable outcomes must remain missing, not be coded as zero.
- New York `ny_fatal_crashes` is fatal crashes, not fatalities; it must not enter the pooled fatalities outcome.
- PPML must keep valid zero outcomes.
- Preferred PPML specification is raw crash count with `log(population)` as an exposure offset.
- Keep county and calendar-date fixed effects.
- Treat the existing PR-body headline estimates as stale until the corrected runner is executed.

---

## Task 1: Confirm branch and environment

- [ ] Check out the current analysis branch.

```bash
git fetch origin
git checkout claude/review-handoff-docs-OxJVr
git pull --ff-only
```

- [ ] Confirm the recent commuter-share commits are present.

```bash
git log --oneline -8
```

Look for commits including:

```text
Add commuter-share spillover analysis
Use commuter-share spillover exposure
Test commuter-share spillover exposure
```

- [ ] Inspect Python version and install project dependencies if needed.

At minimum the corrected runner requires:

```bash
python --version
python -m pip install pandas numpy pyarrow pytz pyfixest pytest
```

If the repo already has an environment/requirements file, prefer that and only install missing packages afterward.

---

## Task 2: Run the regression tests before the full analysis

- [ ] Run the focused tests.

```bash
pytest tests/test_state_dot_analysis_core.py -v
```

Expected behaviors that must pass:

1. unavailable outcomes remain `NaN`;
2. PPML sample retains zero outcomes;
3. population exposure setup is explicit;
4. a non-targeted destination county can receive spillover exposure from alerted home counties;
5. commuter spillover is measured as a destination commuter share;
6. multiple alerted origins add their shares;
7. spillover-exposed counties are not labeled clean controls.

If any test fails, fix the failure before running the regressions and preserve the intended behaviors above.

---

## Task 3: Verify required data exist locally

- [ ] Confirm state DOT processed files are present.

```bash
find data/processed -maxdepth 2 -type f | sort
```

The preferred runner can use these state files when present:

```text
california_ccrs_county_day.parquet
florida_fdot_county_day.parquet
illinois_idot_county_day.parquet
iowa_dot_county_day.parquet
massachusetts_massdot_county_day.parquet
nevada_ndot_county_day.parquet
newyork_dot_county_day.parquet
oregon_odot_county_day.parquet
tennessee_tdot_county_day.parquet
texas_txdot_county_day.parquet
virginia_vdot_county_day.parquet
wisconsin_dot_county_day.parquet
county_population.parquet
commuting/county_commuting_weights.parquet
```

- [ ] Confirm the AMBER alert data exist.

```bash
ls -lh data/raw/amber/foia/openfema_ipaws_alerts_2013_2024.csv \
       data/raw/amber/foia/openfema_ipaws_alerts_2013_2022.csv 2>/dev/null
```

The runner will use the 2013-2024 file if available and otherwise the 2013-2022 fallback.

- [ ] If commuting weights are missing, build them.

```bash
python code/build_commuting_weights.py
```

Then confirm:

```bash
ls -lh data/processed/commuting/county_commuting_weights.parquet
```

---

## Task 4: Inspect the commuter-share construction before estimating

- [ ] Confirm `code/state_dot_analysis_core.py` constructs `spillover_share` from destination-work-county commuting weights.

The intended quantity is:

```text
spillover_share_it
    = sum of commuting weights into destination county i
      from home counties that received a nighttime alert for date t
```

Own-county flows are excluded from spillover because direct exposure is represented by `night_alert`.

- [ ] Confirm `code/run_state_dot_analysis_share.py` uses:

```text
spillover_share_10pp = spillover_share / 0.10
```

and jointly estimates:

```text
night_alert + spillover_share_10pp
```

for the spillover-aware specification.

- [ ] Check the distribution after building the panel.

A convenient diagnostic snippet is:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'code')
import run_state_dot_analysis_share as m
p = m.build_panel()
print(p['exposure_class'].value_counts(dropna=False))
print(p.loc[p.spillover_share > 0, 'spillover_share'].describe(percentiles=[.5,.75,.9,.95,.99]))
print('Direct alert county-days:', int(p.night_alert.sum()))
print('Spillover-only county-days:', int((p.exposure_class == 'spillover').sum()))
print('Clean controls:', int((p.exposure_class == 'clean_control').sum()))
PY
```

Inspect whether the positive spillover-share distribution is plausible. Flag obvious values at 1.0 caused by data errors rather than genuine commuting structure.

---

## Task 5: Run the corrected preferred analysis

- [ ] Run the share-based analysis.

```bash
python code/run_state_dot_analysis_share.py 2>&1 | tee output/state_dot_analysis_share.log
```

The preferred runner estimates four variants for each available outcome/state:

```text
1. WLS TWFE, direct + spillover share jointly
2. PPML raw counts, direct + spillover share jointly
3. WLS TWFE, directly alerted counties vs clean controls only
4. PPML raw counts, directly alerted counties vs clean controls only
```

Expected output files:

```text
output/tables/state_dot_analysis_share.csv
output/tables/state_dot_descriptives_share.csv
```

- [ ] Verify the files were created and are nonempty.

```bash
ls -lh output/tables/state_dot_analysis_share.csv \
       output/tables/state_dot_descriptives_share.csv
wc -l output/tables/state_dot_analysis_share.csv
```

---

## Task 6: Verify PPML is actually retaining zero outcomes

This is the key estimator correction.

- [ ] Inspect log lines reporting PPML input size and zero share.

```bash
grep 'PPML' output/state_dot_analysis_share.log | head -30
```

For fatalities and serious injuries, zero shares should be substantial. A PPML sample that collapses to only positive-outcome county-days indicates the old bug has returned.

- [ ] Compare pooled PPML `n_obs` with the corresponding available raw-count sample size.

```bash
python - <<'PY'
import pandas as pd
r = pd.read_csv('output/tables/state_dot_analysis_share.csv')
print(r[(r.state == 'ALL') & (r.model == 'PPML_raw_count')][
    ['sample','outcome','term','beta','se','pvalue','n_obs','zero_share_input','pct_change']
].to_string(index=False))
PY
```

Do not interpret a PPML model if zeros were inadvertently dropped.

---

## Task 7: Produce the main pooled comparison

- [ ] Print pooled all-crash estimates.

```bash
python - <<'PY'
import pandas as pd
r = pd.read_csv('output/tables/state_dot_analysis_share.csv')
x = r[(r.state == 'ALL') & r.outcome.isin(['crashes','crashes_per_100k'])]
cols = ['sample','model','outcome','term','beta','se','pvalue','irr','pct_change','n_obs']
print(x[[c for c in cols if c in x.columns]].to_string(index=False))
PY
```

Report separately:

```text
A. Direct alert effect from joint direct + spillover model
B. Spillover effect per +10 percentage points exposed commuter share
C. Direct alert effect when spillover-exposed non-targeted counties are removed
```

For PPML, report beta, IRR, percent change, SE/p-value, and N.

For WLS, report beta in crashes per 100k, SE/p-value, and N.

---

## Task 8: Compare corrected results with the old/stale estimates

The currently committed older table `output/tables/state_dot_analysis.csv` was generated with code that dropped zero outcomes in PPML. Do not treat it as the preferred result.

- [ ] Print old pooled rows for comparison only.

```bash
python - <<'PY'
import pandas as pd
old = pd.read_csv('output/tables/state_dot_analysis.csv')
print(old[old.state == 'ALL'].to_string(index=False))
PY
```

- [ ] Compare signs and magnitudes, focusing on whether correcting zero retention changes the previous negative crash estimate.

Record explicitly whether:

```text
- sign stays negative / becomes positive;
- magnitude shrinks / grows materially;
- significance disappears / remains;
- direct-vs-clean-control result differs from the joint spillover model.
```

Do not frame a changed p-value alone as the main substantive finding.

---

## Task 9: Inspect spillover evidence as a mechanism test

- [ ] Extract pooled `spillover_share_10pp` coefficients for all outcomes.

```bash
python - <<'PY'
import pandas as pd
r = pd.read_csv('output/tables/state_dot_analysis_share.csv')
x = r[(r.state == 'ALL') & (r.term == 'spillover_share_10pp')]
cols = ['model','outcome','beta','se','pvalue','irr','pct_change','n_obs']
print(x[[c for c in cols if c in x.columns]].to_string(index=False))
PY
```

Interpretation:

```text
A coefficient on spillover_share_10pp corresponds to a 10 percentage-point
increase in the share of a destination county's workforce commuting from
alerted home counties.
```

Evidence that effects propagate into non-alerted work counties in proportion to exposed commuter share is potentially informative about a driver/mobility channel. Null spillover estimates are also informative and should be reported directly.

Do not call the spillover design causal without further identification work.

---

## Task 10: Check state heterogeneity without specification mining

- [ ] Produce a compact state-level table for the all-crash direct effect from the same preferred specification.

```bash
python - <<'PY'
import pandas as pd
r = pd.read_csv('output/tables/state_dot_analysis_share.csv')
x = r[(r.outcome == 'crashes') &
      (r.model == 'PPML_raw_count') &
      (r.sample == 'spillover_joint') &
      (r.term == 'night_alert')]
print(x[['state','beta','se','pvalue','irr','pct_change','n_obs']].sort_values('state').to_string(index=False))
PY
```

Treat these as heterogeneity diagnostics, not a search for significant states. Note states with very few direct alert county-days.

---

## Task 11: Sanity-check missing/non-comparable outcomes

- [ ] Confirm New York is excluded from pooled person-fatality estimation rather than coded as zero fatalities.

- [ ] Confirm states without serious-injury measures contribute `NaN`, not zero-filled rows.

Use:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'code')
import run_state_dot_analysis_share as m
p = m.build_panel()
print(p.groupby('state')[['crashes','fatals','serious_inj']].agg(lambda s: s.notna().sum()))
PY
```

If any unavailable outcome appears fully populated with zeros, stop and fix that issue before interpreting its pooled model.

---

## Task 12: Report the results back in a reproducible summary

Create or update a short Markdown result note, for example:

```text
output/STATE_DOT_SHARE_RESULTS.md
```

It should contain:

1. branch and commit SHA used;
2. Python and PyFixest versions;
3. test result (`pytest tests/test_state_dot_analysis_core.py -v`);
4. number of panel rows, counties, direct alert county-days, spillover-only county-days, and clean controls;
5. pooled all-crash WLS and PPML direct effects;
6. pooled spillover-share effect per +10pp;
7. direct-vs-clean-control sensitivity;
8. fatality and serious-injury pooled estimates;
9. whether the corrected zero-preserving PPML changes the old headline result;
10. any warnings/errors/estimator convergence failures encountered.

Also save the exact commands used or leave the shell log in `output/state_dot_analysis_share.log`.

---

## Interpretation guardrails

The research question is broader than the original sleep hypothesis. Do not force a sleep-deprivation interpretation onto a negative crash coefficient. Plausible channels include changes in mobility, avoidance of driving, public-safety salience, phone interruption/distraction, sleep disruption, and other behavioral responses.

The commuter-share specification is mainly useful for handling interference and probing whether effects travel with exposed drivers. It does not by itself solve endogenous alert targeting.

A future stronger design should consider alert-event-level identification, richer actual broadcast geography, and traffic/VMT exposure. For this local session, however, the priority is to establish the corrected baseline results before adding further specifications.

## Deliverable to send back

Return the contents of these outputs or commit them to the branch:

```text
output/tables/state_dot_analysis_share.csv
output/tables/state_dot_descriptives_share.csv
output/state_dot_analysis_share.log
output/STATE_DOT_SHARE_RESULTS.md
```

The most important numbers to report immediately are:

```text
Pooled all-crash PPML:
- joint model night_alert effect
- joint model spillover_share_10pp effect
- direct-vs-clean-control night_alert effect

For each: beta, IRR/% change, SE, p-value, N.
```
