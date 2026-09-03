# Task 1 Report: Route Exposure Core

Implemented the pure route-exposure core module at `code/route_exposure_core.py` and added focused tests in `tests/test_route_exposure_core.py`.

What changed:

- Added `weighted_tract_pairs(...)` to collapse block-level LODES flows to tract pairs with worker-weighted endpoints, tract car-share attachment, and explicit missing-endpoint/missing-share tracking.
- Added `parse_osrm_route(...)` to validate OSRM responses and return a normalized route record.
- Added `classify_route_origin(...)` to label each route segment as `own_origin`, `cross_origin`, or `pass_through`.
- Added `allocate_route_miles(...)` to allocate a routed LineString across county polygons while preserving explicit unallocated mileage.
- Added `build_county_exposure(...)` to build county/date commuter-car exposure panels with shared denominators and own/cross/pass-through splits.
- Updated `requirements-analysis.txt` to include the geometry stack used by the pilot design.

Verification:

- Focused task tests: `pytest tests/test_route_exposure_core.py -q`
- Full repository suite: `pytest -q`
- Result: 293 passed, 7 warnings

Notes:

- The first full-suite attempt failed because `pyfixest` was not installed in the local environment. I installed the pinned version used by the repo and reran the suite successfully.
- Existing untracked data outputs in `data/raw/...` and `data/processed/...` were left untouched.

## Fix round 1

Updated the core module to address the review findings:

- `weighted_tract_pairs(...)` now treats missing or nonfinite endpoint coordinates as missing coverage instead of a hard failure. Those rows still contribute to `workers` and `missing_endpoint_workers`, while valid endpoint rows drive the weighted latitude/longitude averages.
- `allocate_route_miles(...)` now uses Shapely/pyproj projected geometries and county intersections instead of midpoint sampling. It allocates miles by intersecting the projected route with projected county polygons and only retains `unallocated_miles` when a nontrivial residual remains.
- Added regression coverage for missing-coordinate worker accounting and exact county-boundary splitting.

Verification for the fix:

- Focused route tests: `pytest tests/test_route_exposure_core.py -q`
- Full suite: `pytest -q`
- Result: 295 passed, 7 warnings
