# Wisconsin route-exposure pilot

For the 50-state-plus-DC extension, closest-vintage rules, streamed storage,
and national acceptance reports, see the
[national route-exposure workflow](route_exposure_national.md). Pilot and
national artifacts use separate cache roots and remain independently
reproducible.

This gated measurement pilot routes strict 2022 LODES tract-pair flows across
Wisconsin, Illinois, Iowa, and Minnesota. It allocates each route to all
traversed counties and constructs county-date commuter-car-mile exposure. A
separate Michigan-only 2021 flow build is retained solely as a clearly labeled
mixed-vintage sensitivity because the official Michigan 2022 LODES OD source
returned HTTP 404. Neither build replaces the national destination-based
measure unless every gate criterion passes.

The default runner reads the reviewed combined AMBER/missing-person/Silver
panel through `code/load_amber_missing_alerts.py`. Its statewide rows have
already been expanded to the included counties, so `statewide_same` never
broadens treatment to origins absent from that reviewed expansion.

## Prerequisites and default commands

Install the pinned analysis requirements and ensure Docker is running. Build
the default artifacts in order:

```bash
python code/build_route_pilot_flows.py --year 2022 --states wi il ia mn \
  --tract-car-share-path data/processed/commuting/route_pilot/inputs/tract_car_share_acs2020_5state.parquet
python code/build_route_pilot_network.py --year 2022 --states wi il ia mn
```

The county allocation also requires the official 2022 TIGER/Line county file
in CRS84/EPSG:4326. The default path is
`data/raw/crosswalks/tl_2022_us_county.geojson`. One operational acquisition
path is:

```bash
curl -O https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip
unzip tl_2022_us_county.zip
ogr2ogr -t_srs EPSG:4326 data/raw/crosswalks/tl_2022_us_county.geojson tl_2022_us_county.shp
```

If the file is absent, has a non-longitude/latitude CRS, or fails its generated
checksum manifest, the allocator stops with an actionable prerequisite. A
different validated GeoJSON or parquet input remains configurable with
`--county-boundaries-path`; a custom file must also provide its real
`--county-boundaries-source-url` and `--county-boundaries-attribution` the
first time its checksum manifest is created. Parquet inputs must carry CRS
metadata identifying EPSG:4326 or OGC:CRS84.

Build the three distinct same-tract artifacts:

```bash
python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode primary_calibrated --route-workers 8 --checkpoint-every 500
python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode zero
python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode exclude
```

The default documented runner now needs no explicit data paths once those
artifacts and the reviewed combined alert panel exist:

```bash
python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode primary_calibrated --write-report
```

Explicit `--pairs`, `--route-results`, `--county-segments`, `--alerts`,
`--existing-exposure`, `--gate-evidence`, `--cache-dir`, and `--output-root`
paths remain available. Imports perform no downloads, routing, or writes.

## Artifacts and accounting

The Task 2 pair artifact includes deterministic `route_id`, endpoint/coverage
fields, `workers`, `home_car_share`, worker-weighted block-pair straight-line
mileage, `commuter_car_weight`, and `commuter_car_miles`. Missing coordinates
or car shares remain as explicitly weighted, routing-ineligible rows. Task 3
checkpoints preserve this schema, and Task 4 consumes it by `route_id` without
positional joins.

Failed routes and explicit unallocated segments remain in the county-segment
artifact with missing outcome FIPS permitted only for those audit record
types. They never contribute affected dosage. Selected commuter-car weight is
counted once per eligible route; external-endpoint omissions, input omissions,
route failures, and unallocated mileage are reported separately. Missing,
nonfinite, negative, duplicate, inconsistent, or unexplained-zero mileage
records fail validation and cannot produce an accepted county-segment output.

Primary same-tract calibration uses the worker-weighted block-pair
straight-line mileage and an urban/rural-specific short-route ratio when that
class exists. It imputes only successfully routed same-tract pairs with zero
or negligible route mileage. Non-negligible routes remain routed. The `zero`
and `exclude` modes materially alter the segment output and use separate cache,
table, and report names.

The primary artifacts use the required unsuffixed paths under `output/tables/`
(`route_pilot_input_diagnostics.csv`, `route_pilot_route_diagnostics.csv`,
`route_pilot_county_exposure_summary.csv`, and
`route_pilot_exposure_comparison.csv`) and
`output/ROUTE_EXPOSURE_PILOT_REPORT.md`. The `zero` and `exclude` sensitivity
artifacts use explicit mode suffixes. Rejected runs retain input, route, and
gate diagnostics but remove accepted county-exposure and comparison outputs.

## Acceptance gate

Acceptance requires all of the following, with missing evidence treated as a
failure:

- at least 99% successful routing by selected commuter-car weight;
- every evaluable route conserved within 0.5% and aggregate mileage within
  0.1%;
- acceptable tract-aggregation error without alerted-origin bias;
- no same-tract dominance or sign reversal across the three modes;
- positive, stable denominators across modes;
- material route-versus-destination allocation differences or a clearly
  improved mechanism; and
- feasible routing rate, storage, and preprocessing requirements for national
  scaling.

The runner derives coverage and conservation directly. The remaining reviewed
results must be supplied as a two-column `metric,value` gate-evidence table;
the accepted metric names are printed in the gate diagnostic. The existing
destination exposure can be supplied with `--existing-exposure`. This makes
the gate fail closed until the comparison, sensitivity, aggregation-sample,
and feasibility exercises have actually been run.

Source provenance includes URLs, retrieval times, byte sizes, checksums, and
attribution. OpenStreetMap-derived artifacts retain “© OpenStreetMap
contributors” and ODbL 1.0 metadata. Source references:
[Census LODES](https://lehd.ces.census.gov/data/),
[2022 TIGER/Line counties](https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/),
[Geofabrik/OpenStreetMap](https://download.geofabrik.de/), and
[OSRM](https://project-osrm.org/).

## Current real-run status (2026-09-02)

The real-data result remains **REJECTED / NOT EVALUABLE**, but network access
was restored and genuine flow preparation completed. The ACS 2020 tract car
share was regenerated for all five candidate states at the ignored
`data/processed/commuting/route_pilot/inputs/tract_car_share_acs2020_5state.parquet`
path. The strict 2022 WI/IL/IA/MN build produced 2,844,476 tract pairs and
retained its diagnostics and source manifest. A separate Michigan-only 2021
build produced 1,081,377 pairs after the official Michigan 2022 LODES source
returned HTTP 404; it is a mixed-vintage sensitivity, not part of the strict
2022 primary sample.

Routing was not attempted because no supported routing toolchain is installed:
Docker Desktop (including the Docker app), Colima, Podman, `osrm-routed`, and
`osmium` are all unavailable. Consequently no PBF/network, route, county-mile,
exposure, or gate-evidence outputs were created or inferred. No data were
fabricated and no national build was started. Once a supported Docker runtime
is installed, run the four-state network command above, then the county-mile
and runner commands for all three same-tract modes.
