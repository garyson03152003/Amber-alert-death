# Wisconsin commuter-route exposure pilot

## Objective

Test whether LODES home-to-work flows can be routed over a road network and
allocated to every county traversed, producing a county-specific measure of
alert-affected commuter car-miles. If the pilot satisfies the pre-specified
quality and scalability criteria below, extend the same pipeline nationally
for the 2013, 2018, and 2022 commuting vintages.

This is a measurement pilot first. It will not replace the existing
destination-based commuter specifications until the route construction has
passed validation and its coverage is understood.

## Pilot scope

Use 2022 Wisconsin and its bordering states: Wisconsin, Illinois, Iowa,
Minnesota, and Michigan. Wisconsin is the focal state because the reviewed
alert data contain 111 locally scoped alerts across 14 Wisconsin counties,
providing substantially more county-level variation than other compact pilot
states.

The pilot will use LODES JT00 total-job origin-destination flows whose home and
work blocks are both in the five-state routing region. This includes:

- Wisconsin local, inbound, and outbound commutes within the routing region;
- commutes between other states in the routing region that pass through
  Wisconsin; and
- commutes entirely within the neighboring states, which are necessary to
  construct meaningful denominators for counties reached by Wisconsin-origin
  commuters.

Flows with an endpoint outside the five-state region are outside the pilot.
Their worker and current commuter-car-mile shares will be measured from the
existing national county-pair tables and reported as omitted coverage. The
pilot output must never be described as a complete Wisconsin traffic measure.

## Data sources and vintages

- LODES8 2022 JT00 main and auxiliary OD files for the five pilot states.
- LODES8 state geography crosswalks for block internal-point coordinates and
  county/tract identifiers. Census documents `blklatdd` and `blklondd` as
  block internal points, not geometric centroids.
- Existing ACS 2020 tract car shares in
  `data/processed/tract_car_share.parquet`, consistent with the current rich
  commuter dosage.
- Historical 2022 OpenStreetMap state extracts from Geofabrik, merged into one
  routing region before OSRM preparation.
- 2022 Census TIGER/Line county boundaries for route/county intersection.
- The reviewed AMBER, missing-person, and silver-alert panel already used by
  the combined analysis.

Every downloaded input will have its source URL, retrieval time, size, and
checksum recorded in a manifest. OpenStreetMap attribution and ODbL source
metadata will be retained with the derived route outputs.

## Approaches considered

### Selected: local OSRM routes with county intersections

Prepare a local OSRM car-routing graph from the merged five-state
OpenStreetMap extract. Request the full GeoJSON geometry for each unique
representative tract pair, intersect each route with county polygons in an
equal-distance projection, and retain the miles assigned to every traversed
county.

This directly represents intermediate and pass-through counties and does not
depend on a rate-limited public routing service. It is the preferred approach
because the route geometry, not only origin-to-destination distance, is needed.

### Rejected as primary: straight-line county intersection

Intersecting the home-to-work line with counties is inexpensive and scalable,
but it can assign exposure to counties that the road route never enters and
miss highway detours. It will be retained only as a diagnostic and a possible
residual-flow approximation in the national build.

### Rejected for the pilot: route every block pair

LODES is block-to-block and its crosswalk supplies block internal points, so
block routing is technically possible. The number of queries and route
geometries would make a national build much more expensive. The pilot will
route tract-pair representatives and use block-level information to improve
their endpoints and handle same-tract flows. A weighted sample of block-pair
routes will be used to quantify aggregation error.

## Tract-pair construction

Read each OD file at block-pair level and join both block geocodes to the LODES
crosswalk. Preserve the LODES worker count `S000`, home and work counties,
home and work tracts, and both block internal-point coordinates.

For every home-tract/work-tract pair:

1. sum workers;
2. calculate a worker-weighted home coordinate from the home-block internal
   points;
3. calculate a worker-weighted work coordinate from the work-block internal
   points;
4. attach the existing home-tract car share; and
5. retain worker-weighted block-level straight-line distance diagnostics.

The routed quantity is one representative morning commute from the weighted
home point to the weighted work point. Direction is home to work. Each route is
cached by routing-data version, profile, and endpoint coordinates so reruns do
not repeat completed queries.

### Same-tract flows

A generic tract-centroid route would make all same-tract commutes zero. The
pilot instead uses the distinct worker-weighted home and work block internal
points. If OSRM still returns a zero or negligible route for a same-tract pair,
the primary construction will impute its distance using the worker-weighted
block-pair straight-line distance multiplied by the median road-to-straight
distance ratio among successfully routed short trips in the same urban/rural
class. These miles are assigned to the tract's county.

Results will be reported under three same-tract treatments:

- primary calibrated imputation;
- zero same-tract miles; and
- exclusion of same-tract flows.

The imputed share of each county denominator must be reported so the analysis
does not conceal dependence on this choice.

## Route-to-county allocation

OSRM returns the full fastest-driving-route geometry. Reproject route and
county geometries to an appropriate equal-distance coordinate system before
measuring intersections. For each tract-pair route `r` and traversed county
`c`, save:

- `route_miles_total_r`;
- `route_miles_in_county_rc`;
- `workers_r`;
- `home_car_share_r`;
- `commuter_car_miles_rc = workers_r * home_car_share_r *
  route_miles_in_county_rc`;
- home/work tract and county identifiers; and
- routing and source-vintage metadata.

Route portions outside the United States or outside available county polygons
will remain explicit unallocated miles rather than being silently dropped.

## County exposure measures

For county `c` and alert date `t`, define the pilot denominator:

```
total_commuter_car_miles_c =
    sum_r workers_r * home_car_share_r * route_miles_in_county_rc
```

Define affected miles using the alert status of the route's home county:

```
affected_commuter_car_miles_ct =
    sum_r alert_home(r),t * workers_r * home_car_share_r
          * route_miles_in_county_rc
```

The primary normalized treatment is:

```
affected_route_share_ct =
    affected_commuter_car_miles_ct / total_commuter_car_miles_c
```

Also retain affected miles in absolute units and per 10,000 baseline commuter
car-miles. Split affected miles into:

- own-origin: home county equals crash county;
- cross-origin: home county differs from crash county; and
- pass-through: both home and work counties differ from crash county.

Own and cross coefficients will therefore use the same county denominator.
For statewide alerts, only routes whose home county was actually included in
the alert geography enter the numerator; the denominator remains unchanged.

## Pipeline structure

Keep data acquisition, routing, geometry allocation, and analysis attachment
separate so each stage can resume and be tested independently:

1. `build_route_pilot_flows.py` downloads/validates LODES inputs and builds
   representative tract pairs.
2. `build_route_pilot_network.py` validates the regional OpenStreetMap input
   and prepares or checks the local OSRM graph.
3. `route_exposure_core.py` contains import-safe endpoint, response,
   allocation, conservation, and exposure helpers.
4. `build_route_pilot_county_miles.py` routes cached pairs and creates the
   tract-pair-by-county mileage table.
5. `run_route_exposure_pilot.py` creates county denominators, alert-date
   exposures, diagnostics, comparisons, and the feasibility report.

Downloaded files, OSRM build artifacts, and full route geometries are
reproducible caches and will remain untracked. Small manifests, diagnostics,
and regression-ready aggregate outputs will follow the repository's existing
data/output conventions.

## Failure handling and resumability

- Fail before routing if required source checksums, columns, coordinate
  coverage, CRS metadata, or county polygons are invalid.
- Record OSRM `NoRoute`, `NoSegment`, malformed response, and timeout outcomes
  separately with endpoints and commuter weight.
- Retry transient routing failures with bounded backoff; do not retry permanent
  topology failures indefinitely.
- Write route results atomically in checkpoints. A restart resumes only
  incomplete pairs.
- Never substitute straight-line geometry for a failed primary route without
  labeling the substitution in both row-level and aggregate diagnostics.
- Refuse to construct an exposure share where its denominator is zero or where
  routed-weight coverage is below the declared threshold.

## Validation and pilot success criteria

The feasibility report must contain all of the following.

### Data coverage

- Counts and worker/car-weight shares before and after tract aggregation.
- Shares omitted because an endpoint lies outside the five-state region.
- Missing ACS car-share and LODES coordinate rates, including fallback weight.
- Same-tract worker, commuter-car-mile, and imputed-mile shares.

### Routing quality

- At least 99% of selected tract-pair commuter-car weight routes successfully.
- Snapping-distance distribution, with all large snaps listed for review.
- Route/straight-line distance-ratio distribution and outlier list.
- At least 100 stratified random route maps, including border crossings,
  same-county routes, long routes, and routes crossing three or more counties.
- A worker-weighted block-pair routing sample to measure tract aggregation
  error in route length and traversed counties.

### Geometry conservation

- The sum of county-allocated miles plus explicit unallocated miles equals the
  OSRM route length within 0.5% for each route and within 0.1% in aggregate.
- No negative, duplicate, nonfinite, or unexplained zero mileage.
- County denominators are positive for every retained pilot county and no
  county is dominated by failed or imputed routes without being flagged.

### Measurement comparison

- Correlations and percentile comparisons among affected route share, current
  destination-worker-normalized dosage, destination commuter-car-mile share,
  and straight-line allocation.
- Decomposition of each route measure into own, cross, and pass-through shares.
- Re-estimation of the existing pilot-sample crash specification only after all
  construction checks pass, with raw and standardized coefficients and the
  existing fixed-effect/inference ladder retained.

## Decision rule for national scaling

Proceed to a national implementation plan only if:

1. selected-flow routing coverage is at least 99% by commuter-car weight;
2. mileage conservation meets the thresholds above;
3. tract aggregation error is acceptably small and not systematically larger
   for alerted origins;
4. same-tract imputation does not dominate the measure or reverse its main
   pilot comparison;
5. the measured routing rate, storage per pair, and preprocessing footprint
   imply a feasible national build; and
6. route exposure materially changes county allocation relative to the
   destination-only measure or provides a substantially clearer mechanism.

If the pilot fails only on computational scale, use a hybrid national design:
route the tract pairs representing at least 99% of commuter-car weight and
allocate the residual with a separately labeled straight-line method. If it
fails on routing validity, conservation, or severe aggregation bias, do not
scale it nationally.

The national design will preserve separate 2013, 2018, and 2022 LODES flow
matrices. The pilot will determine whether a common road-network vintage or
vintage-matched historical OpenStreetMap extracts produce the more stable and
reproducible measure; no national choice is made in advance.

## Tests

Unit tests will cover:

- worker-weighted representative endpoints;
- home-tract car-share attachment;
- same-tract calibration and all sensitivity modes;
- OSRM success and error-response parsing;
- multi-county route intersection and mileage conservation;
- own, cross, and pass-through classification;
- county denominator and alert-date numerator construction;
- zero/low-coverage refusal; and
- checkpoint resume behavior.

An integration fixture will use a small synthetic road geometry and county
boundary set, avoiding network and Docker dependencies in the automated test
suite. The real Wisconsin run will have a separate reproducibility command and
manifest-based validation report.

## Out of scope for the pilot

- modeling time-of-day congestion or alternate-route choice;
- assigning noncommuting trips or commercial traffic;
- claiming that LODES workers drove on the modeled date;
- treating routed commuter miles as total county vehicle-miles traveled;
- changing the reviewed alert-selection rules; and
- replacing the national headline model before pilot and national validation.
