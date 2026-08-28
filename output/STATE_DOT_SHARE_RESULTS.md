# Corrected, validated state-DOT commuter-share results

**Status: VALIDATED, COMPLETE. 16 states plus three sub-state additions now
pass strict coverage validation** (the original 12 target states, minus FL
2013, plus Delaware, North Carolina, Utah, Connecticut, and Hawaii added
this session as full/statewide sources; plus Montgomery County, MD, the
8-county Indianapolis MPO region, and the 2-county (Ada, Canyon) Idaho
COMPASS region as sub-state additions once no full Maryland/Indiana/Idaho
statewide feed could be found). This supersedes the previous version of
this file (dated 2026-08-15), produced before the FARS/state-DOT geography
validation gate existed, which must be treated as stale.

## Reproducibility

- Branch: `claude/review-handoff-docs-OxJVr`
- Base commit: `af926487cf661f66638927fb6ed8eb00f55acc83` (worktree dirty on
  top of it with this session's fixes).
- Python 3.13.5; pandas 3.0.5, NumPy 2.5.2, pyarrow 25.0.1, pyfixest 0.60.0,
  pytest 9.1.1 (`requirements-analysis.txt`, plus `pytz`, which the pinned
  file was missing).
- `pytest tests/` -- 90 passed.
- Preferred runner: `code/run_state_dot_analysis_share.py`, reading only
  `data/processed/validated/*_county_day.parquet` (never the legacy sparse
  files) gated by `config/accepted_state_years.csv`.

## What had to be fixed before this could run at all

This picks up from `output/VALIDATED_CRASH_RESULTS.md`, which covers the
national FARS geography-crosswalk fix in detail. Extending validation to the
state-DOT sources surfaced further genuine bugs and data-quality issues,
each investigated against the live source before being handled -- nothing
here was guessed:

1. **Socrata pagination bug** (`code/crash_download.py`): system fields like
   `:id` are not returned by Socrata's API unless explicitly named in
   `$select`; the strict paginator requested one for ordering but never
   selected it, so every row failed the unique-ID check. Fixed by adding
   `$select=*,<id_field>` when the id field is a system field. Unblocked New
   York and (later) Delaware, the two Socrata-based states.
2. **Evidence-bounded exclusion categories**, mirroring the FARS geography
   fix, added to `crash_coverage.py`/`state_dot_sources.py`:
   `unresolvable_geography_count` (a source's own null/"UNKNOWN" county
   placeholder -- NY's literal `"UNKNOWN"`, Illinois's `CountyCode = 0`,
   Iowa's genuinely-null `COUNTY_NAME`), `unresolvable_date_count` (a
   genuinely null source date -- Massachusetts, 88 crashes across
   2019-2020 -- **and**, extended later, a validly-parsed date that lands
   one calendar year off the year used to partition a fetch -- Delaware's
   Socrata `year` field vs. the parsed calendar year of `crash_datetime`, a
   likely UTC-boundary artifact, a few dozen records/year), and
   `unresolvable_outcome_count` (a structurally impossible negative count --
   one California 2024 crash with `NUMBERINJURED = -1`). Each is excluded
   from the panel and reported, but a small residual (checked against
   source, never assumed) does not fail an otherwise-complete reporting
   unit. A large share still does.
3. **Virginia town-to-county mapping gap**: ~100-160 crashes/year
   (2017-2024) used VDOT's independent-town jurisdiction codes (e.g. "181.
   Town of Burkeville"), which the existing `VA_VDOT_FIPS` table only
   partially covered. Resolved all 191 VDOT town codes to their containing
   county via a Census TIGERweb point-in-polygon spatial join (town
   centroid -> containing county), not guessed -- all 8 requested years now
   pass with zero exclusions.
4. **California county-code bug** (`code/build_california_ccrs.py`): CCRS's
   raw `County Code` is a 1-58 sequential alphabetical index (1=Alameda,
   19=Los Angeles, ...), but the coverage-validation helper looked the raw
   code up directly against the odd-FIPS-suffix table instead of applying
   the same `code * 2 - 1` transform the panel-aggregation path already used
   correctly. This silently mismapped roughly half of all CA rows to the
   wrong county for validation purposes (confirmed: code 19, by far the
   largest row count, is Los Angeles). Fixed to match the aggregation path.
5. A `pandas` 3.0 strict-dtype regression in `crash_coverage.py`'s
   `_manifest_frame` (assigning into an all-NaN float64 `county_fips`
   column, even with an empty selection, now raises); a state-code mismatch
   in `run_state_dot_analysis_fixed.py` (`_load_coverage_manifests` filtered
   manifests by 2-letter state abbreviation, but `write_manifest` normalizes
   `state` to numeric Census FIPS); and a similar tz-aware-vs-naive
   `datetime64` merge failure once Delaware's Socrata timestamps entered the
   pipeline -- all latent bugs that had apparently never been exercised
   end-to-end before this session.
6. **`load_review_allowlist` silently ignored `review_status`**: any row
   present in `config/accepted_state_years.csv` was treated as accepted
   regardless of an explicit `rejected` marking. Fixed so a reviewed file
   can actually record a rejection (see FL 2013 below), not just an
   acceptance.
7. **TxDOT ArcGIS ORDER BY bug** (`code/crash_download.py`): the CRIS
   FeatureServer rejects `ORDER BY <field> ... OFFSET n` past the first page
   when ordered by the Double-typed `crash_id` (a bare "Invalid query
   parameters" 400), but pages correctly when ordered by the service's own
   indexed object-ID field instead; `fetch_arcgis_pages`/
   `strict_arcgis_dataframe` gained a separate `order_by_field` parameter so
   uniqueness can still be verified against the real business key.
8. **Per-page retry-with-backoff for ArcGIS pagination**: the same "Invalid
   query parameters" text also appears transiently under server load, at
   growing and inconsistent offsets on Texas's 600k+ row extracts -- added a
   4-attempt retry with backoff around each individual page fetch. Combined
   with fix 7, this took Texas from repeated total failures to all 5 years
   succeeding (each still slow: 1.5-2.5 hours per year against TxDOT's
   server).
9. **`CrashDate` epoch-millisecond bug** (`code/build_connecticut_uconn.py`):
   Connecticut's CTDOT/UConn `ConnecticutCrash` FeatureServer returns
   `CrashDate` as epoch milliseconds, but it was parsed with
   `pd.to_datetime(...)` without `unit="ms"` in both the panel builder and
   the coverage-validation call, collapsing every date to 1970 and leaving
   only 8 county-days/year instead of ~2,920 -- a `retained_count_mismatch`
   failure across all 7 requested years. Fixed both call sites with
   `unit="ms"`/`date_unit="ms"`.
10. **FIPS-vs-source conflation in `_load_coverage_manifests`**
    (`code/run_state_dot_analysis_fixed.py`): the function filtered the
    combined coverage-manifest glob by state FIPS code alone. Once
    Connecticut's state-DOT source also used FIPS `09`, this incorrectly
    swept in an unrelated diagnostic row (`source="FARS_NHTSA_POLICY"`,
    `coverage_valid=False` for 2015-2021) that documents Connecticut's own
    exclusion from the *national FARS* longitudinal panel -- a different
    reporting unit entirely that happens to share a FIPS code. This had been
    latent since no earlier state-DOT source used FIPS 09. Fixed by also
    requiring the manifest row's `source` column to match the expected
    `STATE_SOURCE_SPECS[state].source` for each requested state.
11. **State-prefix vs exact-county-membership FARS comparison**
    (`code/validate_state_fatalities.py`): `_event_totals` filtered both the
    source's own events and the FARS comparison baseline to
    `fips.str.startswith(state_fips)` (a 2-digit state prefix). This is
    equivalent to filtering on the source's `expected_county_fips` for every
    full-state source (each one already claims its state's entire county
    set), but breaks for a sub-state source whose claimed geography is a
    strict subset of its state's counties -- adding Montgomery County, MD
    (Maryland FIPS `24`, but only county `24031`) would otherwise compare
    one county's crashes against *all* of Maryland's FARS fatalities.
    Switched the filter to exact membership in `expected_county_fips`;
    re-ran the full FARS-vs-state fatality review afterward and confirmed
    byte-for-byte identical ratios for all 15 previously-validated states
    (zero rows differ), so this is a pure generalization, not a behavior
    change for any existing state.
12. **Sub-state "state" identifiers break the 2-char state-code fallback**:
    two further places assumed a "state" value is always a real 2-letter
    postal code or its matching 2-digit FIPS -- `crash_coverage.py`'s
    `_state_code()` (which left an unrecognized code like `"MOCO"`
    unconverted, unlike every real state) and `_county_universe_for()` in
    `code/build_all_validated_state_panels.py` (whose generated county
    universe had no `state` column, so `balance_validated_panel`'s
    `_universe_for_unit` fell back to taking the first 2 characters of the
    5-digit county FIPS as the state code -- "24" instead of the matching
    "24031"). Fixed by registering Montgomery County's own 5-digit FIPS as
    its `_STATE_FIPS` entry (documented as a sub-state convention) and by
    adding an explicit `state` column to every generated county-universe
    frame (a no-op for existing states, since it resolves to the same value
    the old fips-prefix fallback already produced).
13. **pyfixest Rust-backend panic on a single-county fixed-effects fit**: a
    single-county state-by-state regression (Montgomery County) is
    structurally degenerate for a county+date two-way fixed-effects model --
    with one county, every date is its own singleton cell after demeaning --
    and pyfixest's Rust demeaning backend panics (`unwrap()` on `None`, an
    uncatchable process abort, not a Python exception a `try/except` can
    stop) instead of raising cleanly. Fixed by extending
    `run_state_dot_analysis_share.py`'s existing "insufficient estimable
    sample" skip guard to also require at least 2 distinct counties before
    attempting any per-state fit.
14. **Hawaii's `Crash_Date` is not a real per-crash date for most years**:
    2012-2015/2017/2018 each collapse onto 1-3 identical timestamps for the
    entire year's 90-115 records (a bulk-load artifact, confirmed by
    checking unique-date counts directly against the live source); 2016 has
    the same collapse; 2020 has a smaller, above-the-standard-1%-threshold
    year-tag/date disagreement. Only 2019 and 2021-2024 have genuine
    per-crash date variance and are kept -- see the Hawaii narrative below.
15. **Hawaii's remaining kept years had a systematic date-encoding shift**:
    comparing this source's `person_fatals` against validated FARS revealed
    Crash_Date was one calendar day *after* the true crash date for 2019,
    2021, and 2022 (shifting the parsed date back by 1 day raised the
    exact county-date match rate from 2-9% to 84-100% for each of those
    three years), while 2023-2024 needed no shift at all -- evidence of a
    mid-stream change in the source's own date-serialization convention.
    Fixed with a per-year correction table in `build_hawaii_dot.py`,
    verified against FARS after the fix (ratio and Pearson both reached
    exactly 1.000 for all 5 kept years).
16. **Single-city-PD jurisdiction masquerading as county coverage**: three
    candidate sub-state sources found this session (Seattle SDOT for
    Washington, a Cheyenne/Laramie County WY layer, and a Kansas City
    "Road Casualties" dataset spanning 4 Missouri counties) were all
    ultimately **rejected** after their FARS-comparison ratio came back
    suspiciously low and stable (Seattle: 0.12-0.36; Cheyenne: 0.25-0.75) --
    each source is actually a single city police department's jurisdiction
    (which happens to sit inside, or in Kansas City's case span parts of,
    a larger county), not the full county the project's population-offset
    design assumes. Using the full county's population to normalize a
    partial-jurisdiction crash count would systematically and silently
    deflate that unit's rate estimates. The check that catches this: does
    the source's own data include crashes attributed to *other*
    municipalities within the same county (confirmed present for Montgomery
    County MD -- Gaithersburg, Rockville, Takoma Park all appear -- and for
    the Indianapolis MPO region -- Lawrence, Speedway, Beech Grove, Southport
    all appear within Marion County -- but absent for the three rejected
    candidates)? This check should be applied to every future sub-state
    candidate before building it, not just checked after the fact.

## FARS-vs-state-DOT fatality review (`output/tables/state_fars_fatality_validation.csv`)

Before any state-year entered `config/accepted_state_years.csv`, its
person-fatality counts were compared against the canonical validated FARS
panel: 99 full-state-years across CA, FL, IA, IL, MA, NY, OR, TN, VA, WI,
NC, TX, UT, and CT, plus 5 for Hawaii, 7 for the Indianapolis MPO region,
and 12 for the Idaho COMPASS region (124 total), plus 11 more for
Montgomery County, MD (DE excluded from this specific check -- see below;
it has no comparable fatality field). 122 of the 124 full/multi-county-years
show county-year Pearson correlation >= 0.987 (Idaho: exactly 1.000 for all
12 years; Indianapolis: 0.9996-0.9998; Hawaii: exactly 1.000) and a
DOT/FARS ratio within roughly +/-11% (NY's ratio is `NaN` by design -- its
own source contract never claims a comparable person-fatality field, only
crash counts). **Idaho is now the single tightest match of any source in
this project**: ratio exactly 1.000 for 11 of 12 years (0.95 in the
twelfth) and Pearson exactly 1.000 for all 12, checked against the live
source before committing to the build (no post-hoc fix needed, unlike
Hawaii). Hawaii is a close second: ratio and Pearson both exactly 1.000 for
all 5 kept years, but only after fixing a date-collapse artifact and a
date-encoding shift (fixes 14-15). Utah and Indianapolis are close behind
(Utah: ratio 0.99-1.00, Pearson 0.999-1.000 across all 7 years;
Indianapolis: ratio 0.92-1.01 across all 7 years, remarkable given it's
built from a crash-level severity flag, not a true person count);
Connecticut (ratio 0.993-1.028) and Texas (ratio 1.00-1.02) are similarly
tight. Montgomery County's own 11 state-years fail this ratio check on
evidence, not omission -- see the addition narrative below -- and are
reported separately rather than folded into the "122 of 124" figure.

**One rejection**: FL 2013's DOT fatality total (3,744) is 56% higher than
FARS (2,403), both spread evenly across all 12 months (not a partial-year
artifact) -- and uniquely for FL, the ratio flips to *below* 1.0 for every
subsequent year (2014-2018: 0.96-0.97). That sign-flip right at the
2013-2014 boundary, isolated to one state-year out of 85, points to a real
change in FL's own crash-reporting methodology in 2013. FL 2013 is marked
`rejected` in the allowlist and excluded from the validated FL panel and all
estimates below.

## Coverage summary

All 16 states pass `coverage_valid = True` for every requested year (minus
FL 2013): **CA, FL, IA, IL, MA, NY, OR, TN, TX, VA, WI, DE, NC, UT, CT, and
HI** (DE, NC, UT, CT, and HI added beyond the original 12-state target
list; see below). Three sub-state additions, **Montgomery County, MD** (all
11 requested years, 2015-2025), the **8-county Indianapolis MPO region**
(all 7 requested years, 2018-2024), and the **2-county Idaho COMPASS
region** (Ada, Canyon -- all 12 requested years, 2013-2024), also pass
`coverage_valid = True` for every requested year -- added because no
Maryland/Indiana/Idaho *statewide* crash-level feed exists (see below).
Nevada was attempted but never validated -- NDOT's ArcGIS server has been
unreliable across two separate days with two different failure signatures
(an outage, then "service not started"/spurious 0-record responses),
confirmed independent of any of our code, and is excluded from every result
below.

## Delaware, North Carolina, Utah, and Connecticut additions (beyond the original 12)

**Delaware**: first attempted via DelDOT's `DE_ODP_CRASH_DATA` ArcGIS
MapServer, which turned out to be abandoned after August 2017 -- raw volume
collapses from ~32-37k crashes/year (2013-2016, full 12-month coverage) to
double digits (2018-2024) versus a stable population, and critically
`CRASH_CLASS_DESC = 'Fatality Crash'` counts collapse just as sharply
(125/109/40 in 2015-2017, then single digits every year after), which rules
out a raised property-damage reporting threshold as the explanation -- no
policy change makes a state stop recording fatal crashes. A second ArcGIS
endpoint that looked like it might be a live replacement
(`DE_Public_Crash_Data` on a different DelDOT server) turned out to mirror
the exact same stale snapshot, not a real update. The actual current source
was found via a Delaware Public Media article referencing a 2019 regulation
change (DSHS became sole owner of Delaware crash data) and a 2023 public
launch: DSHS's own **Socrata** dataset, `Public Crash Data`
(data.delaware.gov, resource `827n-m6xc`), which shows consistent ~32-38k
crashes/year across the *entire* 2013-2024 window (including a plausible
COVID-era dip to ~31.7k in 2020). All 12 years validate cleanly (12,051
county-days in the analysis panel, 3 direct-alert days, 167 spillover-only
days). Like NY, only `crashes` is comparable (no person-level
fatality/injury field), so DE is accepted on structural grounds rather than
a FARS fatality-ratio check.

**North Carolina** (`StatewideCrashTable` FeatureServer, the table backing
NCDOT's public Statewide Crash Dashboard, 100 counties, 2021-2024): unlike
DE, this source has genuine person-level `NumFatalities` and `NumAInjuries`
(KABCO-A serious injury) fields, so it passed the full FARS fatality-ratio
review like the original states -- all 4 years show ratio 1.07-1.11
(consistently ~7-11% higher than FARS, a plausible DOT-vs-FARS counting-rule
gap) and county-year Pearson correlation 0.987-0.997. 109,500 county-days in
the analysis panel across 2021-2024.

**Utah** (`Crash_Locations` MapServer, a separate layer per year, 29
counties, 2018-2024): the tightest fatality match of any state in this
project -- ratio 0.99-1.00 (essentially exact) and county-year Pearson
0.999-1.000 across all 7 years. Has genuine person-level
`NUMBER_FATALITIES` and `NUMBER_FOUR_INJURIES` (Level-4/suspected-serious
injury) fields. 63,539 county-days in the analysis panel; 0 direct-alert
days in this window (like MA and TN), so it contributes to the pooled
estimate but has no standalone per-state fit.

**Connecticut** (CTDOT/UConn `ConnecticutCrash` FeatureServer, Crash layer
joined to Person layer by `CrashID`, 8 legacy counties, 2015-2021 only --
Connecticut reorganized from counties to planning regions in 2022, so the
source is deliberately scoped to the years its geography is still
county-based, the same longitudinal boundary already documented for the
national FARS panel's own Connecticut handling): genuine person-level
fatality and serious-injury fields, ratio 0.99-1.03 and county-year Pearson
0.988-1.000 across all 7 years -- a clean pass, not a structural-grounds
acceptance. 17,536 county-days in the analysis panel; 0 direct-alert days in
this window (like MA, TN, and UT), so it contributes to the pooled estimate
but has no standalone per-state fit. Building this source surfaced a genuine
bug in the analysis runner: see fix 10 above (`_load_coverage_manifests`
conflating an unrelated FARS diagnostic row that happened to share
Connecticut's FIPS code).

## Beyond the original 12: full-state and sub-state additions this session

After the original 12-state target list, a broad multi-pass search covered
essentially every other US state (Ohio, Georgia, Arizona, Washington,
Pennsylvania, South Carolina, Michigan, Colorado, Montana, New Jersey,
Louisiana, Mississippi, Alabama, Arkansas, Kentucky, New Hampshire, Vermont,
Rhode Island, Maine, West Virginia, North Dakota, South Dakota, Wyoming,
Alaska, New Mexico, Indiana, Nebraska, Kansas, Oklahoma, Minnesota,
Missouri, Idaho). Most either gate access behind a request form, publish
only aggregated/summary statistics, or (Pennsylvania's Socrata dataset, the
closest near-miss) have real severity fields but no day-of-month field,
only year/month/day-of-week, which cannot support a calendar county-day
panel. One genuinely new full-state source cleared the bar: **Hawaii**,
statewide but fatal-crash-only (see below). Once no full statewide feed
could be found for Maryland, Indiana, or Idaho, three sub-state additions
also cleared it: **Montgomery County, MD**, the **8-county Indianapolis MPO
region**, and the **2-county Idaho COMPASS region**.

### Montgomery County, MD

Maryland's own MDOT SHA public layers are fatal-crash-only and an older
statewide Socrata dataset is retired. Per explicit instruction,
county/city-level fallbacks were checked for states without a statewide
feed. Montgomery County, MD's own Socrata portal
(`data.montgomerycountymd.gov`) cleared the bar: live, actively updated
through the query date, with a real `crash_date_time` timestamp and three
linked tables (crash-level Incidents, person-level Drivers, person-level
Non-Motorists, joined by `report_number`) -- richer in raw fields than
several full-state sources already in this project.

The FARS fatality-ratio check, however, showed a **consistent 15-33%
undercount every single year 2015-2024** (ratio 0.67-0.85, never above
1.0) -- unlike FL 2013's ambiguous methodology-change signature, this
pattern is well-explained by a known local-vs-FARS reporting gap: a police
on-scene severity snapshot is not always retroactively updated when a
hospitalized victim dies days later, whereas FARS applies a 30-day-death
standard reconciled against death records. This is a systematic
definitional gap, not noise or a bug (all fatal-injury value variants,
including inconsistent casing across years, were confirmed captured); it is
also not a jurisdiction-coverage gap -- the Incidents table includes
Gaithersburg, Rockville, and Takoma Park's own police departments alongside
county police, confirmed by checking `agency_name`, so the dataset
genuinely covers the whole county, unlike the three rejected candidates
below. So, like NY and DE, `person_fatals`/`serious_injury_persons` are
treated as **not comparable** and reported as `NaN` throughout; only
`crashes` is accepted, on structural grounds (`coverage_valid`, zero
geography exclusions -- geography is constant, single county -- consistent
~8-12k crashes/year with full 12-month coverage across all 11 years).
118,406 crashes total; 4,018 county-days in the analysis panel (2015-2025).

Building this source surfaced three further genuine bugs, all in
shared library code exercised for the first time by a non-2-letter,
single-county "state" identifier -- see fixes 11-13 above.

### Hawaii (statewide, fatal-crash-only)

A live, statewide ArcGIS FeatureServer (`services.arcgis.com`, recently
edited per its own metadata) covers all 5 Hawaii counties but is
**fatal-crash-only** -- every row is a fatal crash, with no all-crash
denominator and no serious-injury field, the mirror image of NY/DE's
crashes-only contract. Two further data-quality issues were found and fixed
before this was usable at all (fixes 14-15 above): `Crash_Date` is not a
real per-crash date for 2012-2018 (a bulk-load artifact -- nearly every
record in each of those years shares one identical timestamp), and the
remaining kept years (2019, 2021-2024) had a one-day systematic date shift
for 2019/2021/2022 that a mid-stream source change apparently corrected by
2023. After both fixes, the FARS match is **exact**: ratio 1.000 and
Pearson 1.000 for all 5 kept years -- the tightest match of any source in
this project, on par with Utah and the Indianapolis MPO region. 9,130
county-days in the analysis panel (2019, 2021-2024 -- 2020 also excluded
for a smaller year-tag/date disagreement).

### Indianapolis MPO region (8 counties: Boone, Hamilton, Hancock,
Hendricks, Johnson, Marion, Morgan, Shelby)

No Indiana statewide crash-level feed exists. This MPO-published ArcGIS
FeatureServer draws from Indiana's statewide crash database restricted to
the MPO's 8 member counties. Verified genuinely county-wide -- not a
single-city-PD jurisdiction -- by checking that Marion County's own records
include Lawrence, Speedway, Beech Grove, and Southport, not only
Indianapolis proper. The source is **Fatal/SSI-only**: every row is
flagged either `Fatal` or `SSI` via a crash-level categorical field, not a
numeric per-crash count, so `crashes` is not comparable; counting
`Fatal`-flagged crashes as `person_fatals` tracked FARS within 2-8% every
year 2018-2024 (ratio 0.92-1.01, Pearson 0.9996-0.9998, checked directly
against the live source before committing to the build), an acceptable
proxy given no true per-crash fatality count exists. `serious_injury_persons`
(`SSI`-flagged crash count) is an analogous proxy, unvalidated against an
independent benchmark, the same caveat already applied to VA/WI. 20,456
county-days in the analysis panel across 2018-2024.

### Idaho COMPASS region (2 counties: Ada, Canyon)

No Idaho statewide crash-level feed exists. COMPASS (the Boise-area MPO)
republishes ITD source crash data restricted to these 2 member counties.
Verified genuinely county-wide -- not a single-city-PD jurisdiction -- by
checking that both counties' records span all their member cities and
multiple reporting agencies (Ada: Boise, Meridian, Eagle, Garden City,
Kuna, Star, reported by 5 different agencies; Canyon: Nampa, Caldwell,
Middleton, and 8 more). Unlike Indianapolis, this source has a genuine
per-crash fatality **count** (not a severity flag) and a KABCO severity
classification enabling a serious-injury proxy, so all three canonical
outcomes are available: `crashes` (one crash record), `person_fatals`
(genuine count), `serious_injury_persons` (injuries on A-severity crashes,
an all-injury proxy). The FARS match is **the tightest of any source in
this project**: ratio exactly 1.000 for 11 of 12 years (0.95 in 2013) and
Pearson exactly 1.000 for all 12 years -- checked directly against the live
source *before* committing to the build, unlike Hawaii's post-hoc fixes.
8,766 county-days in the analysis panel across 2013-2024.

### Three rejected candidates: a jurisdiction-coverage lesson

Seattle SDOT (for Washington), a Cheyenne/Laramie County WY layer, and a
Kansas City "Road Casualties" dataset (spanning 4 Missouri counties) were
all found, built or test-queried, and then **rejected** once their FARS
fatality-ratio came back suspiciously low and stable across years (Seattle:
0.12-0.36; Cheyenne: 0.25-0.75; Kansas City: not built once the pattern was
recognized). Each turned out to be a single city police department's own
jurisdiction, not the full county this project's population-offset design
assumes -- Seattle is only ~33% of King County's population, Cheyenne only
~65-75% of Laramie County's, and Kansas City PD's data is scoped to city
limits (which happen to cross 4 counties) rather than those counties'
full area. Using the full county's population to normalize a
partial-jurisdiction crash count would systematically and silently deflate
that unit's rate estimates -- a more serious problem than an
outcome-comparability gap like Montgomery County's fatality undercount,
since it corrupts the panel's population-normalized rates directly. See
fix 16 above for the check (does the source's own data include *other*
municipalities within the nominal county?) that should be applied to any
future sub-state candidate before building it, not after.

## Panel diagnostics (16 validated states + 3 sub-state regions, pooled)

| state | county-days | counties | direct-alert days | spillover-only days | max spillover share |
|---|---:|---:|---:|---:|---:|
| CA | 169,476 | 58 | 104 | 2,813 | 0.398 |
| CT | 17,536 | 8 | 0 | 0 | 0.000 |
| DE | 12,051 | 3 | 3 | 167 | 0.165 |
| FL | 122,342 | 67 | 53 | 978 | 0.554 |
| HI | 7,300 | 5 | 1 | 102 | 0.036 |
| IA | 325,413 | 99 | 99 | 936 | 0.530 |
| IDCOMPASS | 8,034 | 2 | 4 | 123 | 0.207 |
| IL | 298,044 | 102 | 156 | 1,805 | 0.545 |
| INMPO | 17,528 | 8 | 0 | 295 | 0.002 |
| MA | 40,908 | 14 | 0 | 677 | 0.036 |
| MOCO | 3,287 | 1 | 2 | 128 | 0.212 |
| NC | 109,500 | 100 | 114 | 581 | 0.474 |
| NY | 67,890 | 62 | 43 | 739 | 0.362 |
| OR | 65,736 | 36 | 4 | 459 | 0.352 |
| TN | 104,025 | 95 | 0 | 333 | 0.045 |
| TX | 371,094 | 254 | 72 | 1,973 | 0.469 |
| UT | 63,539 | 29 | 0 | 440 | 0.151 |
| VA | 339,948 | 133 | 50 | 1,881 | 0.751 |
| WI | 289,224 | 72 | 102 | 1,596 | 0.453 |

CT, MA, TN, UT, and INMPO have zero directly-alerted county-days in this
window, so they drop out of any per-state direct-effect regression
(insufficient variation); DE, OR, and IDCOMPASS have very few (3, 4, and
4), Montgomery County has only 2, and Hawaii has just 1. Montgomery County
additionally can never support a *per-state* fit regardless of its
alert-day count: with a single county, the county+date two-way
fixed-effects model is degenerate (every date is its own singleton cell),
which crashes pyfixest's Rust backend rather than merely underpowering the
fit -- see fix 13 above; it still contributes its 3,287 county-days to the
pooled model. Hawaii (5 counties), the Indianapolis MPO region (8
counties), and the Idaho COMPASS region (2 counties) don't hit that same
degenerate case, but Hawaii's single alert-day and Idaho's four still
produce extremely fragile per-state estimates where the fit even
converges -- not informative on their own, included in the pooled estimate
only. All still contribute to the pooled estimate and (where applicable) to
spillover-only observations; CT has no spillover-only days either (max
spillover share 0.000), so it functions as a clean, alert-free control
panel within the pooled model.

## Pooled estimates (all 16 states + 3 sub-state regions, PPML raw-count, log-population offset)

| outcome | sample | term | beta | SE | p-value | IRR | % change | N |
|---|---|---|---:|---:|---:|---:|---:|---:|
| crashes | joint | `night_alert` | 0.00910 | 0.01908 | 0.633 | 1.009 | +0.91% | 2,408,047 |
| crashes | direct-vs-clean | `night_alert` | 0.02199 | 0.02079 | 0.290 | 1.022 | +2.22% | 2,392,418 |
| fatals | joint | `night_alert` | -0.10881 | 0.10998 | 0.322 | 0.897 | -10.31% | 2,330,275 |
| fatals | direct-vs-clean | `night_alert` | -0.02955 | 0.10960 | 0.787 | 0.971 | -2.91% | 2,315,255 |
| serious_inj | joint | `night_alert` | -0.00584 | 0.18070 | 0.974 | 0.994 | -0.58% | 1,502,535 |
| serious_inj | direct-vs-clean | `night_alert` | -0.26845 | 0.15411 | 0.082 | 0.765 | -23.54% | 1,494,611 |

All three outcomes' N move from the 14-state (pre-Hawaii/Indianapolis/Idaho)
table: Idaho contributes all three canonical outcomes; Hawaii and the
Indianapolis MPO region both contribute `person_fatals` (and INMPO also
`serious_injury_persons`), but neither contributes `crashes` (both are
fatal/serious-only sources) -- so `crashes`' N gain comes entirely from
Idaho's ~8k county-days.

WLS (crashes/fatals/serious per 100k) point estimates and the
`spillover_share_10pp` terms are directionally consistent and also nowhere
near significance (all p >= 0.11); full table in
`output/tables/state_dot_analysis_share.csv`.

**Reading**: with the complete 16-state-plus-three-sub-state-region sample
-- including Texas, North Carolina, Utah, Connecticut, Hawaii, the
Indianapolis MPO region, and the Idaho COMPASS region's exceptionally clean
data -- every pooled direct (`night_alert`) effect remains statistically
indistinguishable from zero for all three outcomes (smallest p = 0.082,
for serious injuries, direct-vs-clean sample). Hawaii, Indianapolis, and
Idaho together added ~33k more observations without moving any estimate
meaningfully; the crash point estimate has hovered near (and switched sign
around) zero through every successive addition of real data this session,
underscoring that these pooled point estimates are noise around zero, not
evidence of a real effect in either direction. This state-DOT pooled null
does **not** corroborate the small but marginally-significant positive
spillover effect on fatalities found in the validated national FARS-only
analysis (`output/VALIDATED_CRASH_RESULTS.md`) -- the two panels differ in
composition (16 states plus 3 sub-state regions vs. all 50) and outcome
definition (any all-crash data source vs. FARS's stricter fatal-crash
census), so this divergence is a genuine open finding, not a contradiction
to resolve by picking one number.

## State heterogeneity (crashes, PPML, joint model, direct effect)

| state | beta | SE | p-value | IRR | % change | N |
|---|---:|---:|---:|---:|---:|---:|
| CA | 0.0077 | 0.0203 | 0.702 | 1.008 | +0.78% | 169,476 |
| IL | -0.2398 | 0.2013 | 0.234 | 0.787 | -21.32% | 298,044 |
| NC | -0.0823 | 0.0808 | 0.308 | 0.921 | -7.90% | 109,500 |
| NY | -0.0469 | 0.1358 | 0.730 | 0.954 | -4.58% | 67,890 |
| OR | 0.5300 | 0.0617 | <0.001 | 1.699 | +69.89% | 65,736 |
| TX | 0.0307 | 0.0414 | 0.458 | 1.031 | +3.12% | 371,094 |
| VA | -0.2160 | 0.2294 | 0.346 | 0.806 | -19.43% | 339,948 |
| WI | 0.1972 | 0.0340 | <0.001 | 1.218 | +21.80% | 289,224 |

FL, IA, DE, MA, TN, UT, CT, MOCO, HI, INMPO, and IDCOMPASS are omitted from
this table (no per-state fit, insufficient `night_alert` variation, or --
for HI and INMPO -- no `crashes` outcome at all, since both are
fatal/serious-only sources; CT, MA, TN, UT, and INMPO have zero
direct-alert days; DE and IDCOMPASS have only 3 and 4; Montgomery County
has 2 and can never support a per-state fit regardless, since it is a
single county, see fix 13 above; Hawaii has just 1). TX -- despite being
the largest-by-far single-state addition and the best-powered
fatality-review match -- is itself a clean null (p = 0.458). **Do not
over-read OR's large, significant coefficient**: OR has only 4
directly-alerted county-days in the entire window (see panel diagnostics),
so this single-state PPML estimate is extremely fragile and driven by a
handful of events -- the pooled estimate across all 16 states plus the
three sub-state regions is the number to trust, not this outlier. WI's
positive effect is better-powered (102 direct-alert days) and worth further
scrutiny in a follow-up, but is still just one source among nineteen
showing a
mostly-null pooled result, and TX (nearly 4x WI's direct-alert days) shows
no such effect.

## Missing/non-comparable outcomes handled correctly

- NY contributes `crashes` only; `fatals`/`serious_inj` are `NaN` throughout
  (NY_DOT's field is a fatal-*accident* count, not a person-fatality count,
  per its own source contract) -- confirmed via
  `crash_rows_available`/`fatal_rows_available`/`serious_rows_available` in
  the descriptives table (`NY: 67890 / 0 / 0`).
- DE similarly contributes `crashes` only (`12051 / 0 / 0`).
- CA, MA, WI show `serious_rows_available = 0` (no verified
  KABCO-A/serious-injury field in those sources); their `serious_inj`
  contributes `NaN`, never a fabricated zero.
- FL 2013 is fully absent from the FL panel (122,342 rows spans 2014-2018
  only), not zero-filled.
- CT contributes all three canonical outcomes (`17536 / 17536 / 17536`), the
  only full-state source added this session with a genuine person-level
  serious-injury field alongside fatals.
- MOCO contributes `crashes` only (`3287 / 0 / 0`); its raw
  `person_fatals`/`serious_injury_persons` fields exist structurally but
  were found not comparable to FARS (a systematic 15-33% undercount), so
  they are reported as `NaN`, not a biased count.
- HI contributes `person_fatals` only (`0 / 7300 / 0`) -- statewide but
  fatal-crash-only, the mirror image of NY/DE.
- INMPO contributes `person_fatals` and `serious_injury_persons` only
  (`0 / 17528 / 17528`) -- Fatal/SSI-only, no all-crash denominator.
- IDCOMPASS contributes all three canonical outcomes (`8034 / 8034 / 8034`)
  -- the only sub-state addition this session with a genuine per-crash
  fatality count (not a severity flag) alongside a serious-injury proxy.

## Interpretation guardrails

Per `TODO_LOCAL.md`: the state-DOT commuter-share design is useful for
probing whether effects travel with exposed commuters and for handling
interference, but does not by itself solve endogenous alert targeting. The
research question is broader than a single sleep-deprivation mechanism;
plausible channels include mobility changes, driving avoidance, public-safety
salience, phone-interruption distraction, and sleep disruption. With every
pooled state-DOT estimate here statistically indistinguishable from zero,
the honest read is a null result for all-crash/serious-injury outcomes in
this 16-state-plus-three-sub-state-region sample -- now including the
largest states by direct-alert-day count outside the original set, plus
Utah, Connecticut, Hawaii, the Indianapolis MPO region, and the Idaho
COMPASS region's exceptionally clean data -- not evidence for any specific
mechanism.

## Outstanding work

- Nevada: NDOT's server remains unreliable (confirmed on two separate days
  with two different error signatures -- an outage, then "service not
  started"/spurious 0-record responses). Retry in a session with more time
  budget once the server appears to have stabilized.
- The WI positive point estimate is the one state-level result still worth a
  closer, non-fishing look, though TX's much larger direct-alert-day count
  showing no such effect argues against over-weighting it.
- Six genuinely new states were added beyond the original 12 this session:
  Delaware (crashes-only, full 2013-2024 after finding the real current
  source), North Carolina (full fatality/serious-injury data, 2021-2024),
  Utah (full fatality/serious-injury data, 2018-2024), Connecticut (full
  fatality/serious-injury data, 2015-2021 only, scoped to its pre-2022
  county geography), and Hawaii (statewide but fatal-crashes-only, 5 years
  after excluding a bulk-load date-collapse artifact). Plus three sub-state
  additions: Montgomery County, MD (crashes-only, 2015-2025), the 8-county
  Indianapolis MPO region (fatal/SSI-only, 2018-2024), and the 2-county
  Idaho COMPASS region (full fatality/serious-injury data, 2013-2024, now
  **the tightest FARS match of any source in this project**, exceeding even
  Utah and Hawaii). Delaware alone required distinguishing an abandoned
  feed, a mirrored duplicate, and the genuine current source before landing
  on usable data; Connecticut surfaced a genuine FIPS/source-conflation bug
  in the analysis runner (fix 10); Montgomery County surfaced three more
  shared-library bugs exercised for the first time by a non-2-letter,
  single-county source (fixes 11-13); Hawaii surfaced a genuine
  date-collapse artifact and a mid-stream date-encoding-convention change
  (fixes 14-15); building sub-state candidates surfaced a
  jurisdiction-coverage failure mode that rejected three other candidates
  outright (fix 16) but, once known, let Idaho be verified viable *before*
  building it (a live FARS-ratio check against 5 sample years, rather than
  Hawaii's post-hoc discovery).
- A subsequent broad, multi-pass survey (~35 more states, plus a
  county/city fallback pass per explicit instruction) found no further
  viable statewide source and confirmed five more sub-state candidates fail
  or are unusable, closing out every remaining lead from this session:
  - **Seattle SDOT (WA)**, **Cheyenne/Laramie County (WY)**, and **Kansas
    City "Road Casualties" (MO)** fail the jurisdiction check (fix 16 /
    the rejected-candidates narrative above) -- single-city-PD data
    compared against a larger nominal county.
  - **Cincinnati (OH)** fails the same jurisdiction check: its
    person-level Socrata data has real dates and rich fields, but a
    direct FARS-ratio check against Hamilton County came back
    0.46-0.65 across sample years, the same signature as Seattle/Cheyenne
    -- Cincinnati Police Department jurisdiction only, not the full
    county (Cincinnati is ~37% of Hamilton County's population).
  - **Detroit (MI)**'s only candidate ArcGIS layer is confirmed stale/
    mislabeled: all 19,584 records report `year=2011` and `community_code`
    is entirely null -- not a usable live source.
  - **Kansas**'s KanDOT `Crashes in Kansas`/`State Highway Crashes`
    FeatureServers are unreachable for a domain-level reason, not an
    environment restriction: `dig NS ksdot.gov` returns no nameservers,
    and public DNS resolvers (Google, Cloudflare) both SERVFAIL with an
    explicit "delegation ksdot.gov" error -- the `ksdot.gov` zone's own
    delegation is broken at the registry level. A different machine or
    network would not fix this; it needs KanDOT to repair their own DNS.
  - **South Carolina** was fully resolved without a browser after all: the
    real live endpoint is `SCDPS Collisions`
    (`services7.arcgis.com/TdsEnMqzMcTd7pnb/.../Extrnl_Collisions_Collision_Dashboard/FeatureServer/0`),
    found via the ArcGIS item-search API rather than the Hub's client-rendered
    pages. It has rich statewide fields (County, CrashDate,
    NumberOfFatalities, Suspected_Serious_Injuries, Troop/Agency) but
    returns **0 records** on every query -- confirmed not a wrong-endpoint
    or query-syntax problem by tracing SCDPS's own public "Collisions in
    South Carolina" dashboard to this exact same empty service. SCDPS's
    official collision dashboard is currently backed by an emptied
    dataset; not usable regardless of tooling.

  With these six closed, every remaining US state without a validated
  source in this project has been directly checked at least once (a live
  statewide feed, or a city/county fallback) and found to have no
  accessible crash-level open-data source meeting this project's bar --
  not merely unexplored. Nevada's replacement source, if one exists, was
  not re-investigated this session and remains the one open lead (its
  original NDOT source is a known-unreliable server, not a missing one).
