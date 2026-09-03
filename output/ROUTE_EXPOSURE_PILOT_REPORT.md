# Wisconsin route exposure pilot

Label: `route_exposure_2022`

## Run status

This report records the latest Task 6 dual-run attempt on 2026-09-02. Network
permission was restored, and genuine Census input preparation completed. The
final integration suite passed (`pytest -q`: **359 passed, 8 warnings**) before
this documentation-only update.

The ACS 2020 tract car-share input was regenerated for Wisconsin, Illinois,
Iowa, Minnesota, and Michigan at the ignored route-pilot input path:

`data/processed/commuting/route_pilot/inputs/tract_car_share_acs2020_5state.parquet`

The strict primary flow command used a single 2022 vintage:

```bash
python code/build_route_pilot_flows.py --year 2022 --states wi il ia mn \
  --tract-car-share-path data/processed/commuting/route_pilot/inputs/tract_car_share_acs2020_5state.parquet
```

It wrote 2,844,476 WI/IL/IA/MN tract pairs. Its retained diagnostics report
11,958,797 input rows, 13,001,750 input workers, 11,652,717 retained rows,
12,691,398 retained workers, 310,352 external-endpoint workers, zero
missing-coordinate worker weight, 1,443 missing-car-share workers, and 7,116
same-tract pairs. The year-specific source manifest was retained alongside the
pair and diagnostic artifacts.

The official Michigan 2022 LODES OD source returned HTTP 404. The separate
Michigan sensitivity therefore used 2021 LODES flows:

```bash
python code/build_route_pilot_flows.py --year 2021 --states mi \
  --tract-car-share-path data/processed/commuting/route_pilot/inputs/tract_car_share_acs2020_5state.parquet
```

It wrote 1,081,377 Michigan-only tract pairs. Its diagnostics report 3,712,582
input rows, 4,042,589 input workers, 3,641,886 retained rows, 3,971,140 retained
workers, 71,449 external-endpoint workers, zero missing-coordinate worker
weight, 6,135 missing-car-share workers, and 2,864 same-tract pairs. This is a
**mixed-vintage MI 2021 sensitivity** and is not pooled into or substituted for
the strict 2022 primary sample.

Routing was not attempted because Docker Desktop (including the Docker app),
Colima, Podman, `osrm-routed`, and `osmium` are unavailable in the environment.
No OSM PBF, OSRM graph, route checkpoint, county-mile allocation, or exposure
output was produced.

## Gate decision

**REJECTED / NOT EVALUABLE.** Genuine flow artifacts now exist, but successful
routing, both mileage-conservation thresholds, tract aggregation error,
same-tract dominance/sign stability, denominator stability,
route-versus-destination materiality, and computational routing feasibility
remain unmeasured. The national build is explicitly not started.

The required route-, county-, and exposure-dependent output tables are
intentionally absent. No route or exposure values were fabricated or inferred
from the completed flow tables.

## Re-run commands after installing a supported routing runtime

```bash
python code/build_route_pilot_network.py --year 2022 --states wi il ia mn
python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode primary_calibrated
python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode zero
python code/build_route_pilot_county_miles.py --year 2022 --same-tract-mode exclude
python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode primary_calibrated --write-report
python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode zero --write-report
python code/run_route_exposure_pilot.py --year 2022 --same-tract-mode exclude --write-report
```
