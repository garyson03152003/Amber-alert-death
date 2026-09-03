# Task 4 Report: County Mileage Allocation and Conservation

Implemented the county-mile allocation stage in `code/build_route_pilot_county_miles.py` and added focused tests in `tests/test_route_pilot_county_miles.py`.

What changed:

- Added `load_county_boundaries(path)` to read GeoJSON/JSON/parquet county geometries, normalize five-digit county FIPS codes, and repair minor invalid geometries with `buffer(0)` when needed.
- Added `allocate_cached_routes(route_cache, county_boundaries, output_path)` to intersect cached route geometries with county polygons, preserve explicit unallocated mileage, and emit both `county_fips` and `outcome_fips` for downstream compatibility.
- Added `calibrate_same_tract_distance(pairs, routed_segments, mode)` to support the three same-tract treatments from the pilot brief:
  - `primary_calibrated`
  - `zero`
  - `exclude`
- Added `validate_mileage_conservation(segments, tolerance_row=0.005, tolerance_total=0.001)` to report route-level conservation gaps, unallocated mileage, commuter-car weight coverage, and same-tract imputed mileage.
- Updated the CLI failure path so conservation diagnostics are written even when the final county-segment table is rejected, and the accepted county parquet is removed rather than left behind on failure.
- Added regression tests for:
  - county-crossing allocation,
  - same-tract calibration, including a nonzero routed same-tract row,
  - zero/exclude modes, and
  - conservation checks with explicit unallocated miles and a CLI failure-path cleanup check.

Verification:

- Focused task tests: `pytest tests/test_route_pilot_county_miles.py -q`
- Result: `7 passed, 1 warning`

Notes:

- The warning is a pandas concatenation FutureWarning in the same-tract calibration path. It does not affect the current test result.
- Existing unrelated untracked data outputs in `data/raw/...` and `data/processed/...` were left untouched.
