# National route-exposure workflow

This workflow extends the [Wisconsin route-exposure pilot](route_exposure_pilot.md)
to the 50 states and District of Columbia for analysis years 2013–2024. It
retains the reviewed alert definitions, crash outcomes, controls, fixed
effects, and inference specifications. Only the commuting exposure is
reconstructed: LODES8 JT00 tract pairs are weighted by the home tract's ACS
B08301 car share, routed, and allocated to every county traversed.

The national files live below
`data/processed/commuting/route_national/`. They are reproducible cache files
and are intentionally excluded from Git. The pilot remains under
`data/processed/commuting/route_pilot/`; neither workflow overwrites the
other.

## Required synthetic verification

Run this before any national download or routing job:

```bash
python code/run_route_exposure_national.py \
  --analysis-years 2022 \
  --states wi il \
  --network-year 2022 \
  --same-tract-mode primary_calibrated \
  --chunk-rows 1000 \
  --route-workers 1 \
  --checkpoint-every 2 \
  --geometry-sample-rate 1.0 \
  --dry-run-fixture \
  --output-dir data/processed/commuting/route_national/dry_run
```

This command performs no network requests and starts no OSRM service. It
creates a versioned vintage manifest, Task-3-shaped flow partitions, a
checksum-based synthetic network manifest, Task 5 route audits and county
segments, a strict `route_national.segments.v1` segment manifest, Task 6 model
inputs/results, route-versus-destination and same-tract comparisons, and JSON,
CSV, and Markdown gate reports.

The expected high-level paths are:

```text
data/processed/commuting/route_national/dry_run/
  manifests/national_vintage_manifest.csv
  manifests/network_manifest.json
  flows/partitions/analysis_year=2022/lodes_source_year=2021/work_state=wi.parquet
  segments/analysis_year=2022/lodes_source_year=2021/work_state=wi/route_audits.parquet
  segments/analysis_year=2022/lodes_source_year=2021/work_state=wi/county_segments.parquet
  segments/segment_manifest.csv
  analysis/national_route_model_panel.parquet
  analysis/route_vs_destination_comparison.csv
  analysis/same_tract_mode_summary.csv
  analysis/same_tract_model_results.csv
  gates/national_gate_report.json
  gates/national_gate_table.csv
  gates/national_partition_gate_table.csv
  gates/NATIONAL_ROUTE_GATE_REPORT.md
```

The dry-run fixture deliberately makes 2021 and 2023 available for target
2022 so the closest-vintage tie rule can be verified: the earlier 2021 source
is selected. Its ACS fixture uses the containing 2018–2022 five-year window.
Those labels describe the fixture inventory; production manifests always use
years and windows actually found and validated in the source inventories.

## Production build order

1. Build and validate ACS B08301 state/vintage partitions with
   `code/build_acs_tract_car_share_vintages.py`. Each partition must contain
   tract, total workers, car total, car share, retrieval metadata, and a
   checksum. Missing or zero-worker tracts remain explicit omissions.
2. Call `build_national_flow_year` from
   `code/build_route_national_flows.py` for each analysis year. Use all 51
   workplace states and all 51 origin crosswalks. Keep the default bounded
   `chunk_rows=250_000` unless a measured memory constraint requires a smaller
   value.
3. Build the common car-profile graph once with:

   ```bash
   python code/build_route_national_network.py \
     --year 2022 \
     --cache-dir data/processed/commuting/route_national
   ```

   Omitting `--states` is intentional: the primary graph requires all 50
   states plus DC. A partial graph is only a labeled diagnostic and cannot be
   the accepted national network.
4. For every successful flow partition, call
   `route_partition_to_segments` from
   `code/build_route_national_segments.py`, supplying the validated common
   network manifest ID, national county boundaries, the existing retrying
   OSRM client, and the chosen `--route-workers`, `--checkpoint-every`, and
   `--geometry-sample-rate` values. Start with one timing partition before
   expanding nationally.
5. Build `segments/segment_manifest.csv`. Every v1 row must carry
   `analysis_year`, `lodes_source_year`, `acs_car_share_vintage`,
   `source_manifest_id`, `network_manifest_id`, `source_partition_id`,
   `flow_path`, `flow_sha256`, `segment_path`, `segment_sha256`, `audit_path`,
   `audit_sha256`, state, and status. The flow identity permits failed routes
   without county segments to retain their origin for the alerted-origin bias
   check. A missing, blank, mixed, or checksum-mismatched identity is rejected.
6. Run the established analysis after the segment gate inputs exist:

   ```bash
   python code/run_route_exposure_national.py \
     --segment-manifest data/processed/commuting/route_national/segments/segment_manifest.csv \
     --panel data/processed/panel_county_day.parquet \
     --destination-exposure output/tables/destination_exposure.parquet \
     --analysis-years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \
     --states al ak az ar ca co ct dc de fl ga hi ia id il in ks ky la ma md me mi mn mo ms mt nc nd ne nh nj nm nv ny oh ok or pa ri sc sd tn tx ut va vt wa wi wv wy \
     --network-year 2022 \
     --network-manifest data/processed/commuting/route_national/network/2022/network_manifest.json \
     --run-metrics data/processed/commuting/route_national/run_metrics.json \
     --same-tract-results output/tables/route_national/same_tract_model_results.csv \
     --same-tract-mode primary_calibrated \
     --chunk-rows 250000 \
     --route-workers 8 \
     --checkpoint-every 10000 \
     --geometry-sample-rate 0.001 \
     --output-dir output/tables/route_national
   ```

   Repeat the exposure/model stage for `--same-tract-mode zero` and
   `--same-tract-mode exclude` using separate output directories. The build
   flags are recorded in the machine-readable gate report even though the
   analysis stage consumes already-built segment partitions. Combine the
   actual pooled model rows from all three modes in the file passed through
   `--same-tract-results`. Keep both the pooled rows and the per-partition rows
   from every mode. Those rows must retain the exact analysis years, requested
   states, network year, and source/network/partition manifest IDs emitted by
   this invocation; missing or mismatched pooled or partition provenance fails
   closed.
   The JSON passed through `--run-metrics` must contain
   measured `runtime_seconds` and `restart_reused_share`; missing values fail
   closed. The production command exits nonzero and writes a rejected report
   if any requested state-year is missing, any checksum/provenance check fails,
   or any pooled or partition gate fails.

## Closest-vintage and storage rules

For each state and analysis year, choose the exact LODES year when available;
otherwise choose the available year with the smallest absolute gap, breaking
ties toward the earlier year. ACS uses a containing five-year window when
available, then the nearest window midpoint with the same earlier-tie rule.
Unavailable state-years are manifest rows with an unavailable status, never
successful empty Parquet files.

National routing never retains one geometry file per route. Geometry is
intersected with counties immediately and then discarded. The durable output
is the compact route audit, county-segment Parquet, atomic checkpoint marker,
and a deterministic 0.1% QA geometry sample by default. Route signatures
include endpoint, flow-source, source-partition, and network identities, so a
changed vintage or graph cannot reuse a stale checkpoint.

## Acceptance gates

A real run is accepted only when every flow partition and the pooled output
pass all of these checks; missing evidence fails closed:

- successful routing covers at least 99% of selected commuter-car weight;
- the maximum evaluable-route mileage error is at most 0.5%, and aggregate
  error is at most 0.1%;
- every analysis county has a finite positive route denominator;
- denominator and route-versus-destination evidence is rebuilt and checked
  separately for each source partition from `--panel` and
  `--destination-exposure`, rather than borrowed from the pooled panel;
- every requested state/year partition is available;
- alerted origins do not have materially different routing coverage;
- both alerted and non-alerted origin groups have positive selected
  commuter-car weight, otherwise the bias comparison is not evaluable and
  fails closed; each partition uses only alerts from its own analysis year;
- primary, zero, and exclude same-tract results show neither dominance nor a
  sign reversal in the actual own/cross/pass-through estimates, both pooled
  and separately within every source partition;
- source omissions, missing coordinates/shares, route failures, and
  unallocated miles are explicitly reconciled;
- the route-versus-destination comparison is present; and
- runtime, disk use, and checkpoint-restart reuse are measured and reported.

The evaluator `evaluate_national_gates` returns a machine-readable
`route_national.gates.v1` record with all gate rows and failed gate names. A
failed gate must be reported as rejected; it must not be relabeled accepted.
The synthetic report proves orchestration only and cannot satisfy a real-data
gate.

If exact national computation is infeasible, a high-weight exact-route plus
residual-imputation build may be produced only as an explicitly labeled
sensitivity. It cannot replace the exact primary measure or the existing
destination-based headline estimate without a renewed coverage and bias
review.
