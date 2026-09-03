# Wisconsin route-exposure pilot — final integration fix report

Date: 2026-09-02

## Outcome

The final Important review findings were addressed without creating or
fabricating real pilot data. The real Wisconsin pilot remains **REJECTED / NOT
EVALUABLE**. Network permission was restored, ACS 2020 car shares and genuine
LODES flow tables were built, but routing was not attempted because no supported
Docker/OSRM/osmium runtime is available. No national build was started.

## Integration fixes

- Task 2 now emits a deterministic, route-ID keyed schema with endpoint and
  coverage fields, workers, car share, worker-weighted block-pair straight-line
  mileage, commuter-car weight, commuter-car miles, and explicit pre-routing
  omission weights. Its regional endpoint filter now uses the state FIPS set
  requested by `--states`; the approved WI/IL/IA/MN run excludes Michigan
  endpoints before crosswalk resolution instead of retaining them through the
  former five-state global default.
- Task 2 tract-pair aggregation now computes worker totals, valid/missing
  endpoint weights, endpoint coordinate means, and block-pair straight-line
  means with vectorized weighted sums and one grouped aggregation. It preserves
  the reviewed schema and car-share join while removing the Python callback per
  tract pair. The 5,000-pair regression's aggregation call decreased from
  3.66 seconds before the fix to 0.05 seconds after it on this environment.
- Flow outputs now derive the year in pair, diagnostic, and input-manifest
  filenames from `--year`. A sequential 2022/2021 regression confirms all six
  paths are disjoint and keep their own contents; the default 2022 filenames
  remain exactly backward compatible. The county-mile stage resolves the
  matching year-specific pair, manifest, and route-checkpoint paths as well.
- Task 3 checkpoints preserve the complete pair metadata and include it in
  checkpoint identity. Missing-coordinate/car-share pairs are recorded without
  calling OSRM. Download caches require matching URL/size/checksum manifests.
- Task 4 consumes the checkpoint artifact by exact `route_id`, validates source
  manifests and county CRS, preserves failed/unallocated audit rows, and
  computes selected coverage once per eligible route. The 99% coverage, 0.5%
  row-conservation, and 0.1% aggregate-conservation thresholds are enforced.
- Primary same-tract calibration uses worker-weighted block-pair distance,
  respects urban/rural class when present, and imputes only successfully routed
  zero/negligible same-tract routes. Rebuilt primary, zero, and exclude audit
  rows preserve route IDs, commuter weights/miles, eligibility and omission
  fields, and source/network metadata, so coverage still counts selected weight
  once per route. Primary, zero, and exclude artifacts have distinct paths.
- Task 5 uses the reviewed combined-alert loader by default. Statewide scope is
  limited to the loader-expanded counties, and exact effective dates/scopes are
  retained. Missing outcome FIPS is allowed only for failed/unallocated audit
  rows; only allocated county rows contribute dosage.
- The national-scaling gate fails closed unless routing coverage, both
  conservation checks, tract-aggregation bias, same-tract dominance/sign
  stability, denominator stability, route-versus-destination materiality, and
  computational feasibility all pass. Rejected runs retain input/route/gate
  diagnostics and remove accepted exposure/comparison outputs.
- Provenance records source URLs, retrieval time, bytes, checksums, attribution,
  and licensing. OpenStreetMap output carries ODbL metadata; Docker image tags
  are pinned. Parquet county boundaries now require verified EPSG:4326/CRS84
  metadata, and custom boundary files require their actual source URL and
  attribution instead of receiving Census provenance automatically.
- Mileage validation rejects missing, nonfinite, negative, duplicate,
  inconsistent-route-total, and unexplained-zero allocated/unallocated values.
  These validation failures preserve diagnostics and refuse the accepted
  county-segment artifact.
- Primary output tables and report use the required unsuffixed
  `output/tables/route_pilot_*.csv` and
  `output/ROUTE_EXPOSURE_PILOT_REPORT.md` paths. The `zero` and `exclude`
  sensitivity outputs retain distinct mode suffixes.

## Verification

- Focused route-core/flow-builder suite: `21 passed`; focused
  county-mile/runner suite: `39 passed`.
- Full repository suite: `359 passed, 8 warnings` in 52.04 seconds on the final
  code and test tree.
- CLI/import checks: runner and county allocator `--help` commands completed;
  all five pilot modules compiled with `py_compile`.
- Whitespace check: `git diff --check` completed cleanly and is rerun
  immediately before commit.

The warnings in the recorded full run were non-failing model/library warnings
plus existing timestamp/test warnings; no pilot production exception or failed
assertion occurred.

## Environment blockers and remaining non-blocking concerns

- Network access is restored. ACS 2020 tract car shares for the five candidate
  states were regenerated at the ignored route-pilot input path. The strict
  2022 WI/IL/IA/MN flow build retained 2,844,476 tract pairs plus diagnostics
  and source manifests.
- The official Michigan 2022 LODES OD source returned HTTP 404. The retained
  1,081,377-pair Michigan-only table therefore uses 2021 flows and is labeled a
  **mixed-vintage MI 2021 sensitivity**; it is not part of or a substitute for
  the strict 2022 primary sample.
- Routing was not attempted because Docker Desktop (including the Docker app),
  Colima, Podman, `osrm-routed`, and `osmium` are unavailable. Thus no PBF/network,
  route checkpoint, county allocation, exposure comparison, aggregation-sample
  evidence, same-tract sensitivity result, or runtime/storage feasibility
  measurement exists.
- The default county prerequisite is intentionally not silently substituted.
  It requires the official 2022 TIGER/Line county file converted to CRS84 with
  the documented `curl`/`unzip`/`ogr2ogr` commands, or an explicitly supplied
  equivalent file.
- Urban/rural calibration is used only when the source crosswalk exposes a
  recognized class; otherwise the reported overall short-route ratio is used.
- The non-automatic gate criteria require a genuine `metric,value` evidence
  table. Missing evidence rejects the pilot; users must not populate it from
  assumptions.
- No route maps or block-pair routing sample can be produced before real
  routing is available. These remain required gate evidence, not waived items.

Unrelated untracked crash-analysis and other-WEA files were neither modified
nor staged.
