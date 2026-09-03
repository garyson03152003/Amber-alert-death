# Wisconsin Route Exposure Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a Wisconsin five-state pilot that routes LODES home-to-work flows over OpenStreetMap roads and allocates alert-affected commuter car-miles to every county traversed.

**Architecture:** Keep flow preparation, routing, geometry allocation, county exposure construction, and analysis reporting in separate import-safe modules. Use block internal points to construct weighted tract-pair endpoints, local OSRM for full route geometries, and Census county polygons for mileage allocation. The pilot is a gated measurement exercise; national scaling is considered only after the declared coverage, conservation, aggregation, and feasibility checks pass.

**Tech Stack:** Python 3.13; pandas, NumPy, PyArrow, requests, Shapely, pyproj, pyfixest, pytest; LODES8 JT00 OD files and crosswalks; 2022 Census TIGER/Line counties; 2022 Geofabrik OpenStreetMap extracts; OSRM car profile in Docker.

**Spec:** `docs/superpowers/specs/2026-09-02-wisconsin-route-exposure-pilot-design.md`

## Global Constraints

- Use the 2022 LODES vintage and Wisconsin, Illinois, Iowa, Minnesota, and Michigan as the pilot routing region.
- Route only flows whose endpoints are both in the five-state region; measure and report omitted external-endpoint coverage.
- Use the LODES block internal-point fields `blklatdd` and `blklondd`; they are not geometric centroids.
- Preserve home-tract ACS car share and compute commuter car-miles as `workers * home_car_share * route_miles_in_county`.
- Use local OSRM full GeoJSON routes; do not silently replace failed routes with straight-line distances.
- Treat same-tract flows explicitly and report primary calibrated, zero-mile, and excluded sensitivities.
- County-allocated miles plus explicit unallocated miles must conserve each OSRM route within 0.5% row-wise and 0.1% in aggregate.
- Require at least 99% successful routing by selected commuter-car weight before accepting the pilot.
- Use the same county denominator for own-origin and cross-origin exposure.
- Keep downloaded files, OSRM build artifacts, and full route geometries in untracked caches; commit only code, tests, manifests, diagnostics, and aggregate outputs.
- Do not change alert-selection rules or replace the national headline model during the pilot.

---

## File map

Create the following focused modules and artifacts:

- `code/route_exposure_core.py`: pure data and geometry helpers; no downloads, Docker calls, or regression side effects.
- `code/build_route_pilot_flows.py`: download/validate five-state LODES and crosswalk inputs, aggregate block rows to weighted tract-pair representatives, and write manifests.
- `code/build_route_pilot_network.py`: download/checksum/merge the five historical Geofabrik extracts and prepare the local OSRM MLD graph.
- `code/build_route_pilot_county_miles.py`: call the local OSRM service with resumable caching and write tract-pair-by-county route miles.
- `code/run_route_exposure_pilot.py`: build alert-date county exposures, diagnostics, comparison tables, and the gated pilot report; optionally rerun the existing FARS pilot model.
- `tests/test_route_exposure_core.py`: pure helper tests, including synthetic route geometries and county polygons.
- `tests/test_route_pilot_flows.py`: flow aggregation, coverage, manifest, and same-tract tests.
- `tests/test_route_pilot_routing.py`: OSRM request/response and checkpoint tests with mocked HTTP/Docker boundaries.
- `tests/test_route_pilot_county_miles.py`: route allocation and conservation integration tests.
- `tests/test_route_exposure_pilot.py`: exposure construction, alert classification, diagnostics, and model-input tests.
- `requirements-analysis.txt`: add the geometry dependencies required by the new modules.
- `docs/route_exposure_pilot.md`: concise user-facing data, limitations, run commands, and results interpretation.
- `data/processed/commuting/route_pilot/`: local untracked cache plus committed small manifests/aggregate tables only.
- `output/tables/route_pilot_*.csv` and `output/ROUTE_EXPOSURE_PILOT_REPORT.md`: diagnostics and final gate report.

### Task 1: Add pure route-exposure primitives

**Files:**
- Create: `code/route_exposure_core.py`
- Create: `tests/test_route_exposure_core.py`
- Modify: `requirements-analysis.txt: dependency list`

**Interfaces:**
- `weighted_tract_pairs(block_flows: pd.DataFrame, block_crosswalk: pd.DataFrame, tract_car_share: pd.Series) -> pd.DataFrame`
- `parse_osrm_route(payload: dict, route_id: str) -> dict`
- `allocate_route_miles(route_geojson: dict, counties: pd.DataFrame, route_id: str) -> pd.DataFrame`
- `build_county_exposure(route_segments: pd.DataFrame, alert_home_counties: pd.DataFrame, denominator_mode: str = "all_region_routes") -> pd.DataFrame`
- `classify_route_origin(home_fips: str, work_fips: str, outcome_fips: str) -> str`

- [ ] **Step 1: Write failing tests for weighted tract-pair endpoints.**

```python
def test_weighted_tract_pair_preserves_workers_and_weighted_endpoints():
    flows = pd.DataFrame({
        "h_geocode": ["550010001001001", "550010001001002"],
        "w_geocode": ["550010001002001", "550010001002002"],
        "S000": [3, 1],
    })
    crosswalk = pd.DataFrame({
        "tabblk2020": ["550010001001001", "550010001001002", "550010001002001", "550010001002002"],
        "cty": ["55001", "55001", "55001", "55001"],
        "trct": ["55001000100", "55001000100", "55001000200", "55001000200"],
        "blklatdd": [43.00, 43.04, 43.10, 43.14],
        "blklondd": [-89.40, -89.36, -89.30, -89.26],
    })
    car = pd.Series({"55001000100": 0.8, "55001000200": 0.9})

    out = weighted_tract_pairs(flows, crosswalk, car)

    assert len(out) == 1
    assert out.loc[0, "workers"] == 4
    assert out.loc[0, "home_tract"] == "55001000100"
    assert out.loc[0, "work_tract"] == "55001000200"
    assert out.loc[0, "home_car_share"] == 0.8
    assert out.loc[0, "home_lat"] == pytest.approx(43.01)
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `pytest tests/test_route_exposure_core.py::test_weighted_tract_pair_preserves_workers_and_weighted_endpoints -q`

Expected: FAIL because `route_exposure_core.py` and `weighted_tract_pairs` do not exist.

- [ ] **Step 3: Implement weighted tract-pair aggregation.**

Validate required columns and string geocodes, join home and work block crosswalk records, drop rows with nonfinite coordinates only after recording their worker weight, group by home/work tract and county, and calculate worker-weighted endpoint coordinates. Attach home-tract car share with an explicit missing-share count and do not mutate the caller's frames.

- [ ] **Step 4: Add route parsing and classification tests.**

```python
def test_parse_osrm_route_rejects_non_ok_response():
    with pytest.raises(ValueError, match="NoRoute"):
        parse_osrm_route({"code": "NoRoute", "message": "no path"}, "r1")


def test_classify_route_origin():
    assert classify_route_origin("55001", "55001", "55001") == "own_origin"
    assert classify_route_origin("55003", "55001", "55001") == "cross_origin"
    assert classify_route_origin("55003", "55005", "55001") == "pass_through"
```

- [ ] **Step 5: Implement response parsing and origin classification.**

`parse_osrm_route` must return `route_id`, `distance_m`, `duration_s`, and GeoJSON geometry only for `code == "Ok"`; preserve the original error code/message in a raised exception. `classify_route_origin` must return exactly `own_origin`, `cross_origin`, or `pass_through` using home/work/outcome county equality.

- [ ] **Step 6: Add synthetic county-allocation tests.**

```python
def test_allocate_route_miles_conserves_route_length():
    route = {
        "type": "LineString",
        "coordinates": [[-90.0, 43.0], [-89.0, 43.0]],
        "properties": {"distance_m": 80000.0},
    }
    counties = synthetic_county_boundaries_split_at_longitude(-89.5)
    out = allocate_route_miles(route, counties, "r1")

    assert out["route_miles_in_county"].sum() == pytest.approx(49.7097, rel=0.005)
    assert out["route_miles_in_county"].ge(0).all()
```

- [ ] **Step 7: Implement projected line/polygon allocation.**

Use Shapely and pyproj to transform route and county geometries to an equal-distance CRS, split the route by county intersections, and calculate line lengths in miles. Preserve `unallocated_miles` for geometry outside available counties. Reject invalid or non-LineString route geometry with a descriptive error.

- [ ] **Step 8: Add county exposure tests and implement the builder.**

```python
def test_build_county_exposure_uses_one_denominator_for_own_and_cross():
    segments = pd.DataFrame({
        "outcome_fips": ["55001", "55001"],
        "home_fips": ["55001", "55003"],
        "work_fips": ["55001", "55001"],
        "workers": [100, 50],
        "home_car_share": [0.8, 0.8],
        "route_miles_in_county": [10.0, 20.0],
    })
    alerts = pd.DataFrame({"home_fips": ["55001"], "alert_date": ["2022-01-02"]})

    out = build_county_exposure(segments, alerts)

    assert out.loc[0, "total_commuter_car_miles"] == pytest.approx(1200.0)
    assert out.loc[0, "own_affected_car_miles"] == pytest.approx(800.0)
    assert out.loc[0, "cross_affected_car_miles"] == pytest.approx(0.0)
```

Implement total, own, cross, pass-through, affected, and normalized share columns. Refuse zero denominators, preserve structural zeros, and use alert-date/home-county joins without expanding a single route more than once.

- [ ] **Step 9: Run all Task 1 tests and commit.**

Run: `pytest tests/test_route_exposure_core.py -q`

Expected: all focused tests pass.

Commit: `git add code/route_exposure_core.py tests/test_route_exposure_core.py requirements-analysis.txt && git commit -m "feat: add route exposure core primitives"`

### Task 2: Prepare and validate the five-state LODES flow inputs

**Files:**
- Create: `code/build_route_pilot_flows.py`
- Create: `tests/test_route_pilot_flows.py`
- Modify: `code/config.py: commuting/data path constants only`

**Interfaces:**
- `PILOT_STATES = ("wi", "il", "ia", "mn", "mi")`
- `download_lodes_input(state: str, file_type: str, year: int, cache_dir: Path, session: requests.Session) -> Path`
- `load_lodes_block_flows(paths: list[Path]) -> pd.DataFrame`
- `build_pilot_tract_pairs(block_flows: pd.DataFrame, crosswalks: pd.DataFrame, tract_car_share: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]`
- `write_input_manifest(records: list[dict], path: Path) -> None`

- [ ] **Step 1: Write tests for state/file URL selection and manifest fields.**

```python
def test_lodes_url_uses_2022_jt00_file():
    assert lodes_url("wi", "main", 2022).endswith("wi_od_main_JT00_2022.csv.gz")


def test_manifest_records_checksum_and_source_metadata(tmp_path):
    src = tmp_path / "input.csv.gz"
    src.write_bytes(b"fixture")
    record = manifest_record(src, "https://example.test/input.csv.gz", "wi", "main", 2022)
    assert set(["path", "url", "sha256", "bytes", "state", "file_type", "year"]).issubset(record)
    assert record["bytes"] == 7
```

- [ ] **Step 2: Run tests and confirm the missing helpers fail.**

Run: `pytest tests/test_route_pilot_flows.py::test_lodes_url_uses_2022_jt00_file -q`

Expected: FAIL because the pilot flow module does not exist.

- [ ] **Step 3: Implement resumable LODES downloads and manifests.**

Use the existing repository `requests`/retry conventions, write downloads to a temporary sibling file, atomically rename on success, and never overwrite a file whose checksum differs without recording the old manifest. Download `main` and `aux` for all five states for 2022. Record URL, retrieval time, byte size, SHA-256, state, file type, and year.

- [ ] **Step 4: Add flow validation tests.**

```python
def test_build_pilot_tract_pairs_reports_external_and_missing_weights():
    pairs, diagnostics = build_pilot_tract_pairs(fixture_block_flows(), fixture_crosswalks(), fixture_car_share())
    assert {"workers", "home_lat", "work_lat", "home_car_share"}.issubset(pairs.columns)
    assert diagnostics["input_worker_weight"] >= diagnostics["retained_worker_weight"]
    assert diagnostics["missing_coordinate_worker_weight"] >= 0
```

- [ ] **Step 5: Implement block loading, crosswalk joins, and tract aggregation.**

Read only `h_geocode`, `w_geocode`, and `S000` from OD files; join crosswalk fields `tabblk2020`, `cty`, `trct`, `blklatdd`, and `blklondd`; retain home/work state and county identifiers; call `weighted_tract_pairs`; and write diagnostics for input rows, worker weight, external-endpoint weight, missing-coordinate weight, and missing-car-share weight. Keep only endpoints in `PILOT_STATES` for the pilot pair table.

- [ ] **Step 6: Add same-tract and structural-zero tests, implement cache outputs, and commit.**

Ensure same-tract pairs are retained with their weighted block endpoints and a `same_tract` flag. Write `pilot_tract_pairs_2022.parquet`, `pilot_flow_diagnostics_2022.csv`, and `pilot_input_manifest_2022.csv` beneath the untracked route-pilot cache.

Run: `pytest tests/test_route_pilot_flows.py -q`

Expected: all flow tests pass without network access.

Commit: `git add code/build_route_pilot_flows.py tests/test_route_pilot_flows.py code/config.py && git commit -m "feat: build Wisconsin route pilot flows"`

### Task 3: Prepare the local OSRM routing network and client

**Files:**
- Create: `code/build_route_pilot_network.py`
- Create: `tests/test_route_pilot_routing.py`

**Interfaces:**
- `download_geofabrik_extract(state: str, year: int, cache_dir: Path, session: requests.Session) -> Path`
- `prepare_osrm_network(pbf_paths: list[Path], network_dir: Path, docker_runner: Callable) -> dict`
- `route_pair(home_lon: float, home_lat: float, work_lon: float, work_lat: float, route_id: str, base_url: str, session: requests.Session) -> dict`
- `route_pairs_with_checkpoints(pairs: pd.DataFrame, cache_path: Path, base_url: str, session: requests.Session) -> pd.DataFrame`

- [ ] **Step 1: Write mocked routing-client tests.**

```python
def test_route_pair_returns_distance_and_geometry(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", fake_ok_osrm_response)
    result = route_pair(-89.4, 43.0, -89.3, 43.1, "r1", "http://127.0.0.1:5000", requests.Session())
    assert result["route_id"] == "r1"
    assert result["distance_m"] == 12345.0
    assert result["geometry"]["type"] == "LineString"


def test_route_pair_records_no_segment(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", fake_error_osrm_response("NoSegment"))
    result = route_pair(0.0, 0.0, 1.0, 1.0, "r2", "http://127.0.0.1:5000", requests.Session())
    assert result["status"] == "NoSegment"
```

- [ ] **Step 2: Run the mocked tests and confirm failure.**

Run: `pytest tests/test_route_pilot_routing.py -q`

Expected: FAIL because the routing module does not yet exist.

- [ ] **Step 3: Implement OSRM request construction and bounded retries.**

Use `/route/v1/driving/{lon1},{lat1};{lon2},{lat2}` with `overview=full`, `geometries=geojson`, and `steps=false`. Set a finite connect/read timeout, retry only transport/5xx errors with bounded backoff, and preserve OSRM `NoRoute`, `NoSegment`, `TooBig`, and malformed-response statuses as row-level statuses. Do not call the public OSRM endpoint.

- [ ] **Step 4: Add network-preparation tests and implement Docker command generation.**

Mock the Docker runner and assert the pipeline invokes `osrm-extract` with the car profile, then `osrm-partition`, `osrm-customize`, and `osrm-routed --algorithm mld`. Merge the five Geofabrik state PBFs with an OSM-aware merge tool before extraction; record source checksums and OSRM profile/version in a network manifest. Fail with an actionable message if Docker is unavailable rather than attempting a public fallback.

- [ ] **Step 5: Implement resumable route checkpoints.**

Process unique tract pairs in deterministic `route_id` order, write each successful or failed result atomically, and resume only missing route IDs. Store endpoint coordinates, status, OSRM distance/duration, geometry path, error message, source/network manifest ID, and route timestamp. Keep full geometries in the untracked cache and write only aggregate references to committed outputs.

- [ ] **Step 6: Run mocked routing tests and commit.**

Run: `pytest tests/test_route_pilot_routing.py -q`

Expected: all mocked routing and checkpoint tests pass; no Docker daemon or network is required by the tests.

Commit: `git add code/build_route_pilot_network.py tests/test_route_pilot_routing.py && git commit -m "feat: add local OSRM pilot routing"`

### Task 4: Allocate route mileage to counties and validate conservation

**Files:**
- Create: `code/build_route_pilot_county_miles.py`
- Create: `tests/test_route_pilot_county_miles.py`

**Interfaces:**
- `load_county_boundaries(path: Path) -> pd.DataFrame`
- `allocate_cached_routes(route_cache: pd.DataFrame, county_boundaries: pd.DataFrame, output_path: Path) -> pd.DataFrame`
- `calibrate_same_tract_distance(pairs: pd.DataFrame, routed_segments: pd.DataFrame, mode: str) -> pd.DataFrame`
- `validate_mileage_conservation(segments: pd.DataFrame, tolerance_row: float = 0.005, tolerance_total: float = 0.001) -> dict`

- [ ] **Step 1: Write allocation and conservation tests against synthetic routes.**

```python
def test_allocate_cached_routes_keeps_multi_county_segments():
    routes = fixture_route_crossing_two_counties()
    counties = fixture_two_county_boundaries()
    out = allocate_cached_routes(routes, counties, Path("unused.parquet"))
    assert set(out["county_fips"]) == {"55001", "55003"}
    assert out.groupby("route_id")["route_miles_in_county"].sum().iloc[0] == pytest.approx(
        routes.loc[0, "route_miles_total"], rel=0.005
    )


def test_validate_mileage_conservation_flags_unallocated_route():
    result = validate_mileage_conservation(fixture_segments_with_unallocated_miles())
    assert result["n_failed_rows"] == 0
    assert result["total_unallocated_miles"] > 0
```

- [ ] **Step 2: Implement TIGER boundary loading and county intersection.**

Load 2022 county geometries, normalize five-digit FIPS, repair only known minor invalid geometry issues with a logged operation, and use the projected allocation helper from Task 1. Preserve rows for routes with no county intersection as explicit unallocated records.

- [ ] **Step 3: Implement same-tract calibration modes.**

For the primary mode, estimate the median route-to-straight-line ratio among successfully routed short trips within the same urban/rural class and apply it to worker-weighted block-pair straight-line mileage. Implement `primary_calibrated`, `zero`, and `exclude` modes with a `same_tract_mode` column and no silent default.

- [ ] **Step 4: Implement conservation and coverage diagnostics.**

Report row-level and aggregate route-mile discrepancies, successful/failed commuter-car weight, snapping/route ratio summaries, unallocated mileage, and imputed same-tract mileage. Refuse to write an accepted aggregate table if the row or aggregate conservation threshold fails.

- [ ] **Step 5: Run focused allocation tests and commit.**

Run: `pytest tests/test_route_pilot_county_miles.py -q`

Expected: all synthetic county-crossing, same-tract, and conservation tests pass.

Commit: `git add code/build_route_pilot_county_miles.py tests/test_route_pilot_county_miles.py && git commit -m "feat: allocate routed miles to counties"`

### Task 5: Build alert-date county exposures and pilot diagnostics

**Files:**
- Create: `code/run_route_exposure_pilot.py`
- Create: `tests/test_route_exposure_pilot.py`
- Create: `docs/route_exposure_pilot.md`

**Interfaces:**
- `build_alert_date_exposures(county_segments: pd.DataFrame, alerts: pd.DataFrame, same_tract_mode: str) -> pd.DataFrame`
- `build_route_pilot_diagnostics(pairs: pd.DataFrame, route_results: pd.DataFrame, county_segments: pd.DataFrame) -> dict[str, pd.DataFrame]`
- `compare_destination_and_route_exposure(route_exposures: pd.DataFrame, existing_exposure: pd.DataFrame) -> pd.DataFrame`
- `write_pilot_report(diagnostics: dict[str, pd.DataFrame], path: Path) -> None`

- [ ] **Step 1: Write exposure-builder tests.**

```python
def test_alert_date_exposure_splits_own_cross_and_passthrough():
    segments = fixture_route_segments()
    alerts = pd.DataFrame({
        "home_fips": ["55001"],
        "alert_date": [pd.Timestamp("2022-01-02")],
    })
    out = build_alert_date_exposures(segments, alerts, "primary_calibrated")
    row = out.query("outcome_fips == '55003' and alert_date == '2022-01-02'").iloc[0]
    assert row["cross_affected_car_miles"] > 0
    assert row["affected_route_share"] == pytest.approx(
        row["affected_commuter_car_miles"] / row["total_commuter_car_miles"]
    )


def test_zero_denominator_is_rejected():
    with pytest.raises(ValueError, match="zero denominator"):
        build_alert_date_exposures(fixture_zero_denominator_segments(), fixture_alerts(), "zero")
```

- [ ] **Step 2: Run focused tests to confirm failure.**

Run: `pytest tests/test_route_exposure_pilot.py -q`

Expected: FAIL because the pilot runner does not yet exist.

- [ ] **Step 3: Implement alert-date exposure construction.**

Load the reviewed combined alert panel through `load_amber_missing_alerts.py`, join home-county alert status to route segments, and produce total, affected, own, cross, and pass-through commuter-car miles plus normalized shares. Use local plus inbound/outbound route miles in the county denominator, not the cross-only denominator. Preserve structural zero county-date rows needed by the regression panel.

- [ ] **Step 4: Add comparisons and diagnostics.**

Compare route exposure with the existing `commuter_car_miles` destination dosage, simple commuter shares, straight-line allocation, and current county denominator. Report percentile distributions, correlations, route coverage by worker and commuter-car weight, omitted external-endpoint weight, failed statuses, snapping distance, route/straight-line ratios, same-tract shares, and own/cross/pass-through totals.

- [ ] **Step 5: Add model-input and report tests, implement report generation.**

Ensure output columns are finite and labeled with `route_exposure_2022`, `same_tract_mode`, and `network_manifest_id`; verify own and cross treatments share the same denominator; write diagnostic CSVs and a Markdown report with limitations, source URLs, and the exact command used.

- [ ] **Step 6: Run focused runner tests and commit.**

Run: `pytest tests/test_route_exposure_pilot.py -q`

Expected: all exposure, comparison, and report tests pass without network or Docker access.

Commit: `git add code/run_route_exposure_pilot.py tests/test_route_exposure_pilot.py docs/route_exposure_pilot.md && git commit -m "feat: build route exposure pilot diagnostics"`

### Task 6: Run the real Wisconsin pilot and apply the national-scaling gate

**Files:**
- Modify: `docs/route_exposure_pilot.md` with the observed run command and results
- Create: `output/tables/route_pilot_input_diagnostics.csv`
- Create: `output/tables/route_pilot_route_diagnostics.csv`
- Create: `output/tables/route_pilot_county_exposure_summary.csv`
- Create: `output/tables/route_pilot_exposure_comparison.csv`
- Create: `output/ROUTE_EXPOSURE_PILOT_REPORT.md`

**Interfaces:**
- CLI: `python code/build_route_pilot_flows.py --year 2022 --states wi il ia mn mi`
- CLI: `python code/build_route_pilot_network.py --year 2022 --states wi il ia mn mi`
- CLI: `python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode primary_calibrated`
- CLI: `python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode primary_calibrated --write-report`

- [ ] **Step 1: Run all automated tests before downloading data.**

Run: `pytest -q`

Expected: the existing suite and all new route-pilot tests pass. If a pre-existing test fails, record it separately and do not attribute it to route exposure.

- [ ] **Step 2: Download and validate the five-state LODES/crosswalk inputs.**

Run: `python code/build_route_pilot_flows.py --year 2022 --states wi il ia mn mi`

Expected: resumable input files, a manifest with checksums, tract-pair parquet, and diagnostics listing external endpoints and missing coordinate/car-share weight.

- [ ] **Step 3: Prepare the merged historical OpenStreetMap/OSRM network.**

Run: `python code/build_route_pilot_network.py --year 2022 --states wi il ia mn mi`

Expected: verified five-state PBF inputs, a prepared MLD graph, a network manifest, and a local OSRM endpoint. If Docker is unavailable, stop with the exact prerequisite rather than switching routing providers.

- [ ] **Step 4: Route tract pairs and allocate county miles.**

Run: `python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode primary_calibrated`

Expected: resumable route cache, row-level status table, county-mile parquet, and conservation/coverage diagnostics.

- [ ] **Step 5: Run the pilot exposure builder and report.**

Run: `python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode primary_calibrated --write-report`

Expected: the four committed diagnostic tables, aggregate county exposures, exposure comparisons, and a Markdown gate report.

- [ ] **Step 6: Repeat only the same-tract sensitivity modes.**

Run:

```bash
python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode zero --write-report
python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode exclude --write-report
```

Expected: sensitivity rows identify the mode and do not overwrite the primary outputs.

- [ ] **Step 7: Inspect validation criteria and make the gate decision.**

Accept the pilot only when successful routing is at least 99% by selected commuter-car weight, route-mile conservation meets both thresholds, tract aggregation error is not systematically worse for alerted origins, same-tract imputation is not dominant or sign-reversing, denominators are positive and stable, and the measured route exposure materially differs from or improves the destination-only measure.

- [ ] **Step 8: Run the existing pilot-sample model only after acceptance.**

Use the accepted route exposure as an additional treatment while preserving the existing fixed effects, weather/holiday/other-WEA controls, own/cross separation, analytic clustering, and wild-cluster-bootstrap conventions. Report raw and standardized coefficients; do not replace headline results automatically.

- [ ] **Step 9: Commit only reproducible diagnostics and documentation.**

Run: `git diff --check && git status --short`

Expected: no generated raw downloads or OSRM artifacts staged; only code/tests/manifests/aggregate diagnostics/documentation are committed.

Commit: `git add docs/route_exposure_pilot.md output/tables/route_pilot_*.csv output/ROUTE_EXPOSURE_PILOT_REPORT.md && git commit -m "analysis: validate Wisconsin route exposure pilot"`

If and only if the gate passes, create a separate national-scaling specification and plan. The national build must not be started in this plan when any gate criterion fails.

## Self-review checklist

- Every spec requirement maps to at least one task: five-state scope (Tasks 2/6), local OSRM (Tasks 3/6), same-tract modes (Tasks 1/4/6), route allocation (Tasks 1/4), denominator/exposure definitions (Tasks 1/5), failure handling (Tasks 2/3/4), diagnostics and visual review hooks (Tasks 4/5/6), and national gate (Task 6).
- All functions referenced by later tasks are defined in earlier task interfaces.
- No task uses public routing or silently substitutes a failed route.
- Generated caches are explicitly excluded from commits.
- The plan contains no TODO/TBD/placeholders and every code step has a concrete test or command.
