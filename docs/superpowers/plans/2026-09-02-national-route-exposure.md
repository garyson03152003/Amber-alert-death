# National Route-Exposure Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the route-exposure pilot to all 50 states plus DC and all 2013–2024 analysis years using nearest-valid LODES and ACS tract car-share vintages, streamed county allocation, and the existing full analysis models.

**Architecture:** Keep the pilot modules backward-compatible and add a national, partitioned pipeline. Resolve vintages before downloading; process one workplace state/year in bounded chunks; route tract pairs with the validated parallel pool; intersect route geometry immediately into county segments; and retain only compact audits plus a deterministic QA geometry sample. Reuse a flow/vintage segment partition for every crash year mapped to it, then feed the resulting county-date exposures into the existing model ladder.

**Tech Stack:** Python 3.13; pandas; NumPy; PyArrow/Parquet; requests; Shapely; pyproj; pytest; LODES8 JT00; ACS B08301 5-year tract tables; TIGER/Line county boundaries; OSRM car profile in Docker.

**Spec:** `docs/superpowers/specs/2026-09-02-national-route-exposure-design.md`

## Global Constraints

- Use all 50 states plus DC; exclude Puerto Rico and other territories.
- Select exact LODES/ACS vintage when available, otherwise nearest available year/window; ties choose the earlier year.
- Preserve `analysis_year`, source vintages, gaps, statuses, checksums, and retrieval metadata in every artifact.
- Keep missing coordinates and missing tract car shares as explicit weighted omissions in the primary build.
- Do not persist one GeoJSON file per national route; write county segments immediately and retain only a deterministic 0.1% QA geometry sample by default.
- Use atomic temporary-file-then-rename writes and resumable partitions; unavailable sources must never be represented by successful empty files.
- Do not change alert selection, outcomes, fixed effects, controls, or inference definitions.
- No production code is written before its failing test has been run.

---

### Task 1: Add deterministic nearest-vintage resolution

**Files:**
- Create: `code/route_vintages.py`
- Test: `tests/test_route_vintages.py`

**Interfaces:**
- Consumes: target year, state, available LODES years, available ACS vintage windows.
- Produces: immutable `VintageChoice` records and a manifest writer used by Tasks 2–6.

- [ ] **Step 1: Write failing tests for exact-year selection, tie-breaking, and unavailable inputs.**

```python
def test_resolve_nearest_year_prefers_exact_then_earlier_tie():
    from route_vintages import resolve_nearest_year

    assert resolve_nearest_year(2022, [2018, 2022, 2023]).source_year == 2022
    assert resolve_nearest_year(2021, [2019, 2023]).source_year == 2019


def test_resolve_nearest_year_rejects_empty_candidates():
    import pytest
    from route_vintages import resolve_nearest_year

    with pytest.raises(ValueError, match="no available source years"):
        resolve_nearest_year(2022, [])
```

- [ ] **Step 2: Run the focused tests and verify they fail because `route_vintages` does not exist.**

Run: `pytest tests/test_route_vintages.py -q`

Expected: FAIL with an import error for `route_vintages`.

- [ ] **Step 3: Implement the resolver and manifest schema.**

Implement these exact interfaces:

```python
@dataclass(frozen=True)
class VintageChoice:
    target_year: int
    source_year: int | None
    gap: int | None
    status: str
    reason: str


def resolve_nearest_year(target_year: int, available_years: Iterable[int]) -> VintageChoice:
    """Choose exact, otherwise nearest, with earlier-year tie breaking."""


def resolve_acs_window(target_year: int, windows: Iterable[tuple[int, int, str]]) -> VintageChoice:
    """Choose the containing ACS window, otherwise nearest midpoint."""


def write_vintage_manifest(records: Iterable[Mapping[str, object]], path: Path) -> None:
    """Atomically write sorted manifest rows with a schema version."""
```

Use integer validation, deduplicate candidate years, return `status="unavailable"` only when the candidate list is empty, and sort manifest rows by analysis year/state/source year.

- [ ] **Step 4: Run focused tests and the full unit suite.**

Run: `pytest tests/test_route_vintages.py -q && pytest -q`

Expected: focused tests and all existing tests pass.

- [ ] **Step 5: Commit the resolver.**

```bash
git add code/route_vintages.py tests/test_route_vintages.py
git commit -m "feat: resolve nearest route exposure vintages"
```

### Task 2: Build tract-level ACS car-share vintages

**Files:**
- Create: `code/build_acs_tract_car_share_vintages.py`
- Test: `tests/test_acs_tract_car_share_vintages.py`
- Modify: `code/build_acs_tract_car_share.py:1-220` only to expose its existing table parsing helpers without changing the 2020 output.

**Interfaces:**
- Consumes: ACS 5-year B08301 state files and `route_vintages.resolve_acs_window`.
- Produces: `data/processed/commuting/route_national/car_share/acs_{vintage}/state=XX.parquet` partitions with `tract`, `total_workers`, `car_total`, and `car_share`.

- [ ] **Step 1: Write failing tests for car-share calculation, tract normalization, and missing-share accounting.**

```python
def test_build_car_share_uses_car_total_over_workers():
    import pandas as pd
    from build_acs_tract_car_share_vintages import build_car_share_frame

    result = build_car_share_frame(pd.DataFrame({
        "GEO_ID": ["1400000US55001000100"],
        "B08301_E001": [100], "B08301_E002": [80],
    }))
    assert result.loc[0, "tract"] == "55001000100"
    assert result.loc[0, "car_share"] == 0.8


def test_missing_or_zero_worker_rows_are_omitted_and_counted():
    import pandas as pd
    from build_acs_tract_car_share_vintages import build_car_share_frame

    result, diagnostics = build_car_share_frame(pd.DataFrame({
        "GEO_ID": ["1400000US55001000100", "1400000US55001000200"],
        "B08301_E001": [0, 10], "B08301_E002": [0, 5],
    }), return_diagnostics=True)
    assert len(result) == 1
    assert diagnostics["zero_worker_rows"] == 1
```

- [ ] **Step 2: Run the focused tests and verify the expected import/attribute failures.**

Run: `pytest tests/test_acs_tract_car_share_vintages.py -q`

Expected: FAIL because the new builder and parser do not yet exist.

- [ ] **Step 3: Implement vintage-aware ACS acquisition and parsing.**

Implement:

```python
def build_car_share_frame(raw: pd.DataFrame, *, return_diagnostics: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Normalize one ACS B08301 table to tract-level car shares."""


def load_car_share_for_analysis_year(year: int, cache_dir: Path) -> tuple[pd.Series, str, dict[str, float]]:
    """Resolve the nearest available ACS window and return a tract-indexed share series."""
```

Use the published tract GEO_ID, require positive total workers, bound `car_share` to `[0, 1]`, preserve zero-worker/malformed rows in diagnostics, and never fill missing tract shares in the primary series. Write each state/vintage partition atomically and include URL/checksum metadata.

- [ ] **Step 4: Run focused and existing car-share tests.**

Run: `pytest tests/test_acs_tract_car_share_vintages.py tests/test_route_pilot_flows.py -q`

Expected: PASS with no changes to the existing 2020 pilot behavior.

- [ ] **Step 5: Commit the ACS vintage builder.**

```bash
git add code/build_acs_tract_car_share_vintages.py code/build_acs_tract_car_share.py tests/test_acs_tract_car_share_vintages.py
git commit -m "feat: add vintage-matched tract car shares"
```

### Task 3: Generalize LODES flow preparation to national partitions

**Files:**
- Create: `code/build_route_national_flows.py`
- Test: `tests/test_route_national_flows.py`
- Reuse: `code/build_route_pilot_flows.py`, `code/route_exposure_core.py`, `code/route_vintages.py`

**Interfaces:**
- Consumes: state `main`/`aux`/crosswalk files, tract car-share series, target year, and selected source year.
- Produces: one partition per `(analysis_year, lodes_source_year, work_state)` plus diagnostics and a national vintage manifest.

- [ ] **Step 1: Write failing tests for workplace-state coverage, chunked aggregation, source metadata, and stable IDs.**

```python
def test_national_flow_partition_keeps_cross_state_aux_rows():
    from build_route_national_flows import build_flow_partition

    pairs, diagnostics = build_flow_partition(
        block_flows=fixture_main_and_aux_rows(),
        crosswalks=fixture_crosswalk(),
        tract_car_share=fixture_car_share(),
        analysis_year=2022, lodes_source_year=2021, work_state="mi",
    )
    assert set(pairs["work_state"]) == {"mi"}
    assert diagnostics.loc[0, "lodes_source_year"] == 2021
    assert pairs["route_id"].is_unique
```

- [ ] **Step 2: Run the focused tests and verify they fail because the national builder is absent.**

Run: `pytest tests/test_route_national_flows.py -q`

Expected: FAIL with an import error for `build_route_national_flows`.

- [ ] **Step 3: Implement chunked state/year processing.**

Implement:

```python
def build_flow_partition(
    block_flows: pd.DataFrame,
    crosswalks: pd.DataFrame,
    tract_car_share: pd.Series,
    *,
    analysis_year: int,
    lodes_source_year: int,
    work_state: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one validated national flow partition."""


def build_national_flow_year(
    analysis_year: int,
    states: Sequence[str],
    cache_dir: Path,
    *,
    chunk_rows: int = 250_000,
) -> pd.DataFrame:
    """Download/validate, process, and manifest all workplace states for one year."""
```

Use workplace-state `main` and `aux` files, read only `h_geocode`, `w_geocode`, and `S000`, call the existing worker-weighted tract aggregation, attach source metadata to every row, and write temporary partition files followed by atomic rename. Use the nearest LODES resolver per state; an unavailable state produces a manifest row and diagnostics but no successful empty data partition.

- [ ] **Step 4: Run focused tests plus a small two-state integration fixture.**

Run: `pytest tests/test_route_national_flows.py tests/test_route_pilot_flows.py -q`

Expected: PASS and identical results for the pilot fixture when the same states/year/share inputs are supplied.

- [ ] **Step 5: Commit national flow preparation.**

```bash
git add code/build_route_national_flows.py tests/test_route_national_flows.py
git commit -m "feat: build partitioned national LODES route flows"
```

### Task 4: Add national road-network manifests and all-state input validation

**Files:**
- Create: `code/build_route_national_network.py`
- Test: `tests/test_route_national_network.py`
- Reuse: `code/build_route_pilot_network.py`

**Interfaces:**
- Consumes: state abbreviations, explicit network year, Geofabrik extracts, and the existing Docker runner.
- Produces: a merged contiguous-US-plus-DC OSRM graph and checksum manifest without changing the pilot network command.

- [ ] **Step 1: Write failing tests for the 51-state universe, deterministic ordering, and cross-state manifest contents.**

```python
def test_national_state_universe_has_50_states_plus_dc():
    from build_route_national_network import NATIONAL_STATES

    assert len(NATIONAL_STATES) == 51
    assert "dc" in NATIONAL_STATES
    assert "pr" not in NATIONAL_STATES


def test_network_manifest_records_all_sources_and_network_year(tmp_path):
    from build_route_national_network import build_network_manifest

    manifest = build_network_manifest([{"state": "wi", "sha256": "a"}], network_year=2022, network_id="n1")
    assert manifest["network_year"] == 2022
    assert manifest["states"] == ["wi"]
```

- [ ] **Step 2: Run focused tests and verify the expected import failure.**

Run: `pytest tests/test_route_national_network.py -q`

Expected: FAIL because the new module is absent.

- [ ] **Step 3: Implement state inventory, extract acquisition, and manifest creation.**

Implement:

```python
NATIONAL_STATES: tuple[str, ...]


def build_network_manifest(source_records: Sequence[Mapping[str, object]], *, network_year: int, network_id: str) -> dict[str, object]:
    """Return a sorted, schema-versioned network manifest."""


def prepare_national_osrm_network(states: Sequence[str], network_year: int, cache_dir: Path, docker_runner: Callable) -> dict[str, object]:
    """Download all requested extracts, merge them, and prepare the OSRM graph."""
```

Validate state names against the 51-state universe, sort inputs before merging, include `network_year`, profile, algorithm, Docker image, source URLs/checksums, and fail with an actionable error on a missing extract or unavailable Docker runtime.

- [ ] **Step 4: Run focused tests and the complete network unit suite.**

Run: `pytest tests/test_route_national_network.py tests/test_route_pilot_routing.py -q`

Expected: PASS without changing pilot manifest IDs or command behavior.

- [ ] **Step 5: Commit the national network module.**

```bash
git add code/build_route_national_network.py tests/test_route_national_network.py
git commit -m "feat: prepare national routing networks"
```

### Task 5: Stream route geometry into county segments

**Files:**
- Create: `code/build_route_national_segments.py`
- Test: `tests/test_route_national_segments.py`
- Modify: `code/build_route_pilot_network.py:route_pairs_with_checkpoints` only to expose a geometry callback while preserving the default pilot geometry behavior.
- Reuse: `code/route_exposure_core.py`, `code/build_route_pilot_county_miles.py`

**Interfaces:**
- Consumes: flow partitions, OSRM responses, county boundaries, and network/source manifests.
- Produces: compact route audits, partitioned county segments, and deterministic QA geometries.

- [ ] **Step 1: Write failing tests for immediate county allocation, no full geometry persistence, resume behavior, and conservation.**

```python
def test_streaming_router_writes_segments_and_only_sampled_geometry(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    audits, segments = route_partition_to_segments(
        fixture_pairs(), fixture_boundaries(),
        base_url="http://osrm", cache_dir=tmp_path,
        geometry_sample_rate=0.5, max_workers=1,
        route_client=fixture_route_client(),
    )
    assert len(audits) == 2
    allocated = segments.groupby("route_id")["route_miles_in_county"].sum().sum()
    unallocated = audits["unallocated_miles"].sum()
    assert pytest.approx(allocated + unallocated, rel=1e-6) == audits["route_miles_total"].sum()
    assert len(list((tmp_path / "qa_geometries").glob("*.geojson"))) == 1


def test_streaming_router_resumes_completed_route_signatures(tmp_path):
    from build_route_national_segments import route_partition_to_segments

    first = route_partition_to_segments(fixture_pairs(), fixture_boundaries(), base_url="http://osrm", cache_dir=tmp_path, route_client=fixture_route_client())
    second = route_partition_to_segments(fixture_pairs(), fixture_boundaries(), base_url="http://osrm", cache_dir=tmp_path, route_client=fixture_route_client_that_fails_if_called())
    assert len(second[0]) == len(first[0])
```

- [ ] **Step 2: Run focused tests and verify the expected import failure.**

Run: `pytest tests/test_route_national_segments.py -q`

Expected: FAIL because the streaming segment builder is absent.

- [ ] **Step 3: Implement the bounded streaming router and append-only partition writer.**

Implement:

```python
def route_partition_to_segments(
    pairs: pd.DataFrame,
    county_boundaries: object,
    *,
    base_url: str,
    cache_dir: Path,
    route_client: Callable,
    max_workers: int = 8,
    checkpoint_every: int = 10_000,
    geometry_sample_rate: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route one flow partition and immediately emit county-mile segments."""
```

Route only `routing_eligible` rows, use the existing retry/status parser, and key checkpoints by the full route signature including source and network IDs. For each successful response, call the existing county intersection allocator before discarding geometry; write only the hash-selected QA sample. Store failed routes and unallocated miles as audit records, keep a schema version, and atomically replace each checkpoint partition.

- [ ] **Step 4: Run focused, pilot, and core allocation tests.**

Run: `pytest tests/test_route_national_segments.py tests/test_route_pilot_routing.py tests/test_route_pilot_county_miles.py -q`

Expected: PASS; pilot default still writes its existing geometry paths, while national mode writes no unselected route geometries.

- [ ] **Step 5: Commit streaming routing.**

```bash
git add code/build_route_national_segments.py code/build_route_pilot_network.py tests/test_route_national_segments.py
git commit -m "feat: stream national routes into county segments"
```

### Task 6: Build national county-date exposures and integrate the full models

**Files:**
- Create: `code/run_route_exposure_national.py`
- Test: `tests/test_route_exposure_national.py`
- Modify: `code/run_route_exposure_pilot.py:build_alert_date_exposures` to accept a caller-supplied label and vintage columns while retaining the existing default label.
- Reuse: `code/load_amber_missing_alerts.py`, `code/analysis_lib.py`, and the existing model runners.

**Interfaces:**
- Consumes: national county segments, vintage manifest, reviewed alerts, existing panel, and same-tract mode.
- Produces: one county-date exposure panel and model result tables for each analysis year/vintage and pooled full analysis.

- [ ] **Step 1: Write failing tests for own/cross/pass-through dosage, source metadata, and nearest-vintage reuse.**

```python
def test_national_exposure_preserves_origin_components_and_vintages():
    from run_route_exposure_national import build_national_exposure

    result = build_national_exposure(fixture_segments(), fixture_alerts(), analysis_year=2022, flow_source_year=2021, car_share_vintage="2019-2023")
    row = result.iloc[0]
    assert row["own_affected_car_miles"] > 0
    assert row["cross_affected_car_miles"] >= 0
    assert row["pass_through_affected_car_miles"] >= 0
    assert row["lodes_source_year"] == 2021
    assert row["acs_car_share_vintage"] == "2019-2023"


def test_national_runner_reuses_one_segment_partition_for_mapped_years(monkeypatch):
    from run_route_exposure_national import required_flow_partitions

    assert required_flow_partitions([2013, 2014, 2015], {2013: 2013, 2014: 2013, 2015: 2014}) == [2013, 2014]
```

- [ ] **Step 2: Run focused tests and verify the expected import/keyword failures.**

Run: `pytest tests/test_route_exposure_national.py -q`

Expected: FAIL because the national runner and vintage-aware exposure label are absent.

- [ ] **Step 3: Implement exposure construction and model runner.**

Implement:

```python
def build_national_exposure(
    county_segments: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    analysis_year: int,
    flow_source_year: int,
    car_share_vintage: str,
    same_tract_mode: str = "primary_calibrated",
) -> pd.DataFrame:
    """Build county-date route dosage with complete source metadata."""


def required_flow_partitions(analysis_years: Sequence[int], mapping: Mapping[int, int]) -> list[int]:
    """Return sorted unique source flow years needed by the analysis years."""


def run_national_route_analysis(args: argparse.Namespace) -> pd.DataFrame:
    """Load partitioned exposures and run the established model specifications."""
```

Pass a caller-supplied `label` into the pilot exposure helper, retain the existing alert geography behavior, and join the source metadata without changing model controls or fixed effects. Produce per-year, per-vintage, and pooled result tables; keep route and destination measures side by side.

- [ ] **Step 4: Run focused tests and the existing analysis tests.**

Run: `pytest tests/test_route_exposure_national.py tests/test_route_exposure_pilot.py -q`

Expected: PASS with unchanged pilot results on the existing fixtures.

- [ ] **Step 5: Commit national exposure integration.**

```bash
git add code/run_route_exposure_national.py code/run_route_exposure_pilot.py tests/test_route_exposure_national.py
git commit -m "feat: integrate national route exposures into analysis"
```

### Task 7: Add end-to-end manifests, diagnostics, documentation, and verification

**Files:**
- Modify: `docs/route_exposure_pilot.md` to link the national workflow and distinguish pilot/national artifacts.
- Create: `docs/route_exposure_national.md`
- Create: `tests/test_route_national_end_to_end.py`
- Modify: `.gitignore` only to add the national cache root while keeping large source/network/geometry artifacts untracked.

**Interfaces:**
- Consumes: all artifacts from Tasks 1–6.
- Produces: reproducible commands, gate diagnostics, and a validated national analysis report.

- [ ] **Step 1: Write failing end-to-end fixture tests for manifests, gates, and refusal of incomplete coverage.**

```python
def test_national_gate_rejects_coverage_below_99_percent():
    from run_route_exposure_national import evaluate_national_gates

    result = evaluate_national_gates({
        "successful_commuter_car_share": 0.989,
        "aggregate_conservation_error": 0.0005,
        "positive_denominators": True,
        "all_partitions_available": True,
        "no_alerted_origin_bias": True,
        "same_tract_stable": True,
        "route_comparison_complete": True,
    })
    assert result["accepted"] is False
    assert "coverage" in result["failed_gates"]


def test_national_gate_accepts_valid_fixture():
    from run_route_exposure_national import evaluate_national_gates

    result = evaluate_national_gates({
        "successful_commuter_car_share": 0.999,
        "aggregate_conservation_error": 0.0005,
        "positive_denominators": True,
        "all_partitions_available": True,
        "no_alerted_origin_bias": True,
        "same_tract_stable": True,
        "route_comparison_complete": True,
    })
    assert result["accepted"] is True
```

- [ ] **Step 2: Run the focused tests and verify they fail because the gate function is absent.**

Run: `pytest tests/test_route_national_end_to_end.py -q`

Expected: FAIL with an import error for `evaluate_national_gates`.

- [ ] **Step 3: Implement diagnostics, gate evaluation, CLI, and documentation.**

Implement:

```python
def evaluate_national_gates(metrics: Mapping[str, float]) -> dict[str, object]:
    """Apply coverage/conservation/denominator requirements and report failures."""
```

The CLI must support `--analysis-years`, `--states`, `--network-year`, `--same-tract-mode`, `--chunk-rows`, `--route-workers`, `--checkpoint-every`, and `--geometry-sample-rate`; emit manifest paths and gate tables; and refuse to label a run accepted when any partition is unavailable or below the stated thresholds. Documentation must include exact commands, expected partition paths, nearest-vintage behavior, storage policy, and the residual-imputation fallback as a sensitivity only.

- [ ] **Step 4: Run all unit/integration tests and static checks.**

Run: `pytest -q`

Expected: all repository tests pass with zero failures.

- [ ] **Step 5: Run a small synthetic national dry run.**

Run: `python code/run_route_exposure_national.py --analysis-years 2022 --states wi il --network-year 2022 --chunk-rows 1000 --route-workers 1 --checkpoint-every 2 --geometry-sample-rate 1.0 --dry-run-fixture`

Expected: synthetic manifests, flow partitions, route audits, county segments, gate diagnostics, and model input tables are created under the national cache without network downloads.

- [ ] **Step 6: Commit the verified national pipeline.**

```bash
git add docs/route_exposure_national.md docs/route_exposure_pilot.md tests/test_route_national_end_to_end.py .gitignore
git commit -m "docs: add national route exposure workflow and gates"
```

### Task 8: Execute the real national build in bounded batches

**Files:**
- No source-file changes; write only ignored data/cache/output artifacts.

**Interfaces:**
- Consumes: verified code and synthetic dry-run output from Tasks 1–7.
- Produces: national flow/vintage manifests, route audits, county segments, exposures, and full-analysis tables.

- [ ] **Step 1: Resolve and record all 2013–2024 LODES and ACS selections without routing.**

Run the manifest-only command for all 51 states. Review unavailable state/year rows, source gaps, missing coordinates, and missing car-share weight before downloading networks.

- [ ] **Step 2: Build the common network and route one source-flow partition as a timing/storage benchmark.**

Measure routes per second, checkpoint throughput, successful commuter-car coverage, county-mile conservation, and disk growth. Stop before disk exhaustion if the observed footprint violates the storage gate.

- [ ] **Step 3: Route remaining partitions with bounded workers and resume checkpoints.**

Run one workplace-state/source-year batch at a time, keeping the route worker count and checkpoint interval in the manifest. Do not run multiple national network builds concurrently on the same disk.

- [ ] **Step 4: Build primary/zero/exclude same-tract exposure tables and run the full model ladder.**

Write per-year/vintage and pooled tables, route-versus-destination comparisons, and all gate diagnostics.

- [ ] **Step 5: Run fresh verification before claiming completion.**

Run: `pytest -q` and the national gate report command emitted by the CLI.

Expected: tests pass and the report explicitly states accepted/rejected status, coverage, conservation, omitted weights, runtime, and storage. If compute is infeasible, report the exact gate failure and produce only the labeled hybrid sensitivity.
