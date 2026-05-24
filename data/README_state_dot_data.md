# State DOT Crash Data — Download Instructions

This project uses FARS (already on disk) as the primary crash source.  
State DOT data adds **serious injury** coverage beyond FARS fatality-only scope.

> **Note**: The remote execution environment has no outbound Python network access.  
> Run these downloads **locally** or from any machine with internet access,  
> then upload the resulting parquet files to `data/processed/`.

---

## Priority Order (by alert count in sample)

| Rank | State | Alerts | Source | Auth | Script |
|------|-------|--------|--------|------|--------|
| 1 | **Texas** | 24,737 | CRIS | Free registration | see below |
| 2 | California | — | CCRS | None | `build_california_ccrs.py` |
| 3 | **Illinois** | 1,608 | IDOT ArcGIS | None | `build_illinois_idot.py` |
| 4 | Florida | 2,862 | FDOT | None (partial) | see below |

---

## 1. California CCRS (No Auth — Best for Immediate Use)

**Portal**: https://data.ca.gov/dataset/ccrs  
**Files**: One CSV per year, 2016–2024 (~100–200 MB each)

```bash
# Run from project root (needs internet access)
python3 code/build_california_ccrs.py
```

The script downloads all years and saves to:
`data/processed/california_ccrs_county_day.parquet`

**Direct URLs** (if script fails, download manually):
- 2016: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/3d5f2586-cf68-4213-aa1c-60df37399d10/download/crashes_2016.csv
- 2017: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/4784664d-b7cf-4427-af25-7c7307bad56c/download/crashes_2017.csv
- 2018: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/a4b57216-5110-43d3-884c-d95366b19158/download/crashes_2018.csv
- 2019: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/2b4c7d03-e684-435e-80da-17935de9499f/download/crashes_2019.csv
- 2020: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/a2e0605d-0695-4bce-806d-4d0dda7ace68/download/crashes_2020.csv
- 2021: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/d08692e2-6d36-487e-bca0-28cd127a626f/download/crashes_2021.csv
- 2022: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/7828780b-117b-455e-9275-986ad3ffde50/download/crashes_2022.csv
- 2023: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/436642c0-cd04-4a4c-b45e-564b66437476/download/crashes_2023.csv
- 2024: https://data.ca.gov/dataset/80c6a49d-c6b3-40ba-86d8-379c9741b4be/resource/f775df59-b89b-4f82-bd3d-8807fa3a22a0/download/crashes_2024.csv

**Key columns** (SWITRS/CCRS):
- `COLLISION_DATE` — crash date (YYYY-MM-DD)
- `COLLISION_TIME` — time of crash (HHMM)
- `COUNTY_CITY_LOCATION` / `COUNTY` — county identifier
- `COLLISION_SEVERITY` — K/Fatal, A/SevereInj, B/OtherVis, C/Complaint, O/PDO
- `NUMBER_KILLED`, `NUMBER_INJURED`

---

## 2. Texas CRIS (Free Registration — Most Alerts)

Texas has by far the most alerts (24,737) but requires one-time registration.

**Steps**:
1. Register (free, no cost): https://cris.dot.state.tx.us/public/Register/app/registration/selectRegistrationType
2. After receiving login credentials (~24 hours), log in at: https://cris.dot.state.tx.us/
3. Request CSV extract for years 2013–2024, statewide
4. Download the ZIP file delivered to your email
5. Place CSV files in `data/raw/texas_cris/` and run:

```bash
python3 code/build_texas_cris.py   # (to be written after data arrives)
```

**Key columns** (CRIS standard extract):
- `Crash_Date`, `Crash_Time`
- `County` (county name)
- `Crash_Sev_ID` — 1=Fatal, 2=Incapacitating, 3=Non-Incap, 4=Possible, 5=Not Injured
- `Person_Injury_Severity` at person level

---

## 3. Illinois IDOT (No Auth — ArcGIS)

**Portal**: https://gis-idot.opendata.arcgis.com/  
**Script**: `python3 code/build_illinois_idot.py`

Manual download per year if the ArcGIS API doesn't respond:
1. Go to https://gis-idot.opendata.arcgis.com/
2. Search "Crashes 2023" (one dataset per year, 2016–2024)
3. Click Download → CSV
4. Place files in `data/raw/illinois_idot/crashes_{year}.csv`

**Key columns**:
- `CRASH_DATE` — crash date
- `COUNTY_NAME` — county name
- `INJURIES_FATAL`, `INJURIES_INCAPACITATING`

---

## 4. Florida FDOT (No Auth — ArcGIS)

**Portal**: https://gis-fdot.opendata.arcgis.com/  
**Static 2017–2020 CSV**: https://gis-fdot.opendata.arcgis.com/documents/630f22996b88425a94781c597be7bc01

For 2021+, use the ArcGIS REST API with pagination:
```
https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0/query?where=1=1&outFields=CRASH_DATE,DOT_CNTY_CD,NUMBER_OF_KILLED,NUMBER_OF_SERIOUS_INJURIES&f=csv&resultOffset=0&resultRecordCount=1000
```

**Key columns**: `CRASH_DATE`, `DOT_CNTY_CD`, `NUMBER_OF_KILLED`, `NUMBER_OF_SERIOUS_INJURIES`

---

## States with Only PDF/Dashboard Data (Not Usable for Panel Analysis)

These states are in our top-10 alert list but have no bulk CSV download:
- **Georgia** (3,242 alerts): GEARS portal requires LexisNexis login
- **Tennessee** (2,990 alerts): PDFs and dashboards only
- **North Carolina** (2,374 alerts): Dashboard only; bulk requires DOT request
- **Ohio** (1,965 alerts): Restricted to agency personnel
- **Michigan** (1,730 alerts): Mi-CAT tool, limited bulk export
- **Missouri** (2,635 alerts): Limited attributes; contact MoDOT
- **Oklahoma** (1,490 alerts): Dashboard/PDF reports only

---

## After downloading: merge into main panel

Once any state parquet file is in `data/processed/`, run:
```bash
python3 code/run_state_serious_injuries.py   # (to be written)
```
This merges the state serious injury counts with the main panel and runs
the same TWFE regressions using `serious_injuries` as the outcome.
