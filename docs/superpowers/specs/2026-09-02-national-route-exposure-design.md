# National Route-Exposure Analysis Design

**Status:** Approved design direction (2026-09-02)

## Goal

Extend the validated route-exposure pilot to the full fatal-crash analysis:
all 50 states and the District of Columbia, study years 2013–2024, nearest
available LODES flow year for each state/year, and nearest available ACS
tract-level car-share vintage. The resulting measure will allocate commuter
car-miles to every county traversed by a home-to-work route and will be used
in the existing own/cross/pass-through alert models.

## Scope and estimand

- The geographic universe is the 50 states plus DC, matching the validated
  FARS panel. Puerto Rico and other territories remain out of scope.
- The flow source is LODES8 JT00. For each workplace state, both `main` and
  `aux` files are processed; this covers jobs whose workplace is in that state
  without double-counting a home/work pair.
- A flow row is a block-to-block worker count. Rows are collapsed to a
  home-tract/work-tract pair using worker-weighted LODES block internal-point
  coordinates.
- Home-tract ACS B08301 car share is attached before route aggregation. The
  primary commuter-car weight is `workers × home_car_share`; county mileage
  dosage is that weight times routed miles inside the county.
- Each route is classified for a county as `own_origin`, `cross_origin`, or
  `pass_through`, retaining all three components in the exposure output.
- Alert selection, alert geography, crash outcomes, fixed effects, controls,
  and inference procedures are unchanged. This work replaces only the
  commuting-exposure construction and its documented robustness comparison.

## Closest-vintage rule

Vintage selection is deterministic and state-specific.

1. For target analysis year `y`, enumerate the LODES8 JT00 source years that
   are actually available for each workplace state. Prefer an exact `y`.
   Otherwise select the available year minimizing `abs(source_year - y)`;
   ties select the earlier year. If no source year is available, mark that
   state/year unavailable and retain its omitted worker weight in diagnostics;
   never create a fabricated empty flow file.
2. Enumerate the ACS 5-year tract B08301 vintages that are present and valid.
   Select the vintage whose estimate window is nearest to `y`, preferring the
   window containing `y` and then the earlier midpoint on a tie.
3. Record the selection in a versioned manifest with at least:
   `analysis_year`, `state`, `lodes_source_year`, `lodes_year_gap`,
   `lodes_status`, `acs_car_share_vintage`, `acs_window_start`,
   `acs_window_end`, `acs_status`, `selection_rule`, retrieval/checksum
   identifiers, and the number and worker weight of retained/omitted rows.
4. A flow artifact carries both `analysis_year` and `lodes_source_year`; a
   car-share artifact carries `acs_car_share_vintage`. These fields prevent
   mixed-vintage rows from being mistaken for a single contemporaneous source.

The nearest-vintage mapping is used for the primary full analysis. Exact-only
and common-vintage alternatives remain available as sensitivity runs, not as
silent substitutions.

## Input preparation

The national builder processes one workplace state and one selected LODES
source year at a time, in chunks bounded by an explicit row/chunk-size
parameter. It:

1. downloads and validates the state `main`, `aux`, and geography crosswalk
   inputs with atomic files and SHA-256 manifests;
2. reads only the required OD columns (`h_geocode`, `w_geocode`, `S000`);
3. joins block internal-point coordinates and tract/county identifiers;
4. aggregates to tract pairs with worker-weighted endpoint coordinates and
   worker-weighted straight-line mileage;
5. attaches the selected tract car-share series and computes
   `commuter_car_weight` and straight-line diagnostic mileage; and
6. writes a partitioned Parquet flow artifact plus state/year diagnostics.

Partitions are keyed by `analysis_year`, `lodes_source_year`, and workplace
state. Each partition is independently resumable and can be combined with
PyArrow/DuckDB without loading the national matrix into memory.

Missing coordinates and missing tract car shares remain explicit weighted
omissions in the primary artifact. A county-share fallback is a separate
sensitivity artifact with its own manifest; a national-mean fill is not a
primary estimate.

## Road network and routing

The primary network is a reproducible merged contiguous-US-plus-DC road graph
with a car profile and a recorded OSRM/osmium manifest. All state extracts are
included so cross-state routes are not truncated at state boundaries. A
network-year argument is explicit; the primary run uses one common network
vintage to isolate changes in commuting flows, with vintage-matched historical
networks available as a sensitivity.

Routing uses the bounded parallel worker pool already validated in the pilot.
Each eligible tract pair is routed exactly once for a given flow/network
signature. Checkpoints are atomic and keyed by a route signature containing
endpoint coordinates, flow source, and network manifest, so a rerun resumes
without positional joins or stale-vintage reuse.

Route geometry is not persisted once per national pair. The response geometry
is intersected with county boundaries immediately, and only the following are
retained:

- a compact route audit table (status, distance, duration, weights, source and
  network IDs, and error text);
- a partitioned county-segment table containing route miles in each county,
  unallocated mileage, segment type, and the flow/vintage identifiers; and
- a deterministic small QA geometry sample (default 0.1% of successful
  eligible routes, configurable without changing the exposure estimate).

This preserves county allocation and auditability without the pilot's
per-route GeoJSON storage explosion. Failed routes and unallocated mileage
remain audit records and never contribute affected dosage.

## Exposure construction and model integration

For each analysis year, the runner joins the county-segment table to the
reviewed combined AMBER/missing-person/Silver alert panel. It constructs
county-date totals and affected totals for own, cross, and pass-through
origins using the existing alert geography rules. The output includes:

- total and affected commuter-car miles;
- own/cross/pass-through affected and denominator shares;
- routing, omitted-coordinate, omitted-car-share, failed-route, and
  unallocated weights;
- analysis year, LODES source year, ACS car-share vintage, network manifest,
  and same-tract mode.

The full analysis re-runs the established fatal-count and combined-rate
specifications, including county/date or day-month fixed effects, county-year
robustness specifications, holiday controls, and the existing cluster/wild
bootstrap inference ladder. Results are reported by year/vintage and pooled;
the route measure does not silently overwrite the existing destination-based
headline measure.

## Validation gates

The national output is accepted only after all gates pass separately by flow
partition and for the pooled exposure:

- at least 99% successful routing by selected commuter-car weight;
- route mileage conservation within 0.5% per evaluable route and 0.1%
  aggregate;
- positive, stable county denominators;
- no material alerted-origin coverage bias;
- no same-tract dominance or sign reversal across primary/zero/exclude
  same-tract modes;
- explicit accounting for unavailable LODES states/years, missing ACS shares,
  route failures, and unallocated miles; and
- a documented route-versus-destination comparison and measured runtime,
  storage, and restart footprint.

If only national compute is infeasible, a high-weight exact plus residual
imputation build may be produced as a labeled sensitivity. It cannot replace
the exact accepted measure without a renewed coverage and bias review.

## Reproducibility and failure handling

- Every source, network, and vintage manifest stores URL, retrieval time,
  checksum, byte count, attribution, and license metadata.
- Downloads and Parquet checkpoints use temporary sibling files followed by
  atomic rename.
- A failed source request is recorded as unavailable with its error; it does
  not become an empty successful partition.
- All route and segment partitions include a schema version and source IDs.
- The pilot artifacts remain readable and unchanged; the national artifacts
  use a separate cache root.

## Out of scope

- Modeling congestion, departure-time choice, or route choice beyond the OSRM
  car route.
- Assigning noncommuting or commercial trips.
- Treating LODES workers as observed drivers on the crash date.
- Reclassifying alerts or changing the reviewed WEA/AMBER inclusion rules.
- Replacing the headline destination-based estimate before the national gates
  and comparison are complete.
