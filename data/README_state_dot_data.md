# State DOT Crash Data — Download Summary

This directory contains county-level crash panel data downloaded from individual
state Departments of Transportation (DOTs). These datasets supplement the federal
FARS (fatality-only) data with **all police-reported crashes** (KABCO: K/A/B/C/O)
including serious injuries, providing a richer outcome variable for the study.

All scripts live in `code/` and can be re-run independently.

---

## Datasets

### California — CCRS
| Item | Value |
|------|-------|
| **File** | `processed/california_ccrs_county_day.parquet` |
| **Script** | `code/build_california_ccrs.py` |
| **Source** | California SWITRS / CCRS statewide crash database |
| **Coverage** | 2013–2023, 58 counties, county-day granularity |
| **Key fields** | `fips`, `date`, `ca_fatals`, `ca_serious_inj`, `ca_crashes` |
| **Serious inj.** | KABCO-A (Suspected Serious Injury) |

---

### Florida — FDOT
| Item | Value |
|------|-------|
| **File** | `processed/florida_fdot_county_day.parquet` |
| **Script** | `code/build_florida_fdot.py` |
| **Source** | FDOT GIS FeatureServer — gis.fdot.gov/arcgis (Crashes_All layer) |
| **Coverage** | 2013–2016 only (server doesn't support deep pagination for 2017+), 67 counties, county-day |
| **Key fields** | `fips`, `date`, `fl_fatals`, `fl_serious_inj`, `fl_crashes` |
| **Serious inj.** | `NUMBER_OF_SERIOUS_INJURIES` field (KABCO-A) |
| **Note** | FDOT FeatureServer returns errors for `resultOffset > ~100k`; 2017–2019 skipped |

---

### Illinois — IDOT
| Item | Value |
|------|-------|
| **File** | `processed/illinois_idot_county_day.parquet` |
| **Script** | `code/build_illinois_idot.py` |
| **Source** | Illinois DOT open data portal |
| **Coverage** | 2013–2022, 102 counties, county-day granularity |
| **Key fields** | `fips`, `date`, `il_fatals`, `il_serious_inj`, `il_crashes` |
| **Serious inj.** | KABCO-A (Suspected Serious Injury) |

---

### Iowa — Iowa DOT
| Item | Value |
|------|-------|
| **File** | `processed/iowa_dot_county_day.parquet` |
| **Script** | `code/build_iowa_dot.py` |
| **Source** | Iowa DOT SOR (Safety Operations Report) download API |
| **URL** | `https://swttraffic.iowadot.gov/SOR/download` (POST request) |
| **Coverage** | 2015–2024, 99 counties, county-day granularity |
| **Key fields** | `fips`, `date`, `ia_fatals`, `ia_serious_inj`, `ia_crashes` |
| **Serious inj.** | `MAJINJURY` (KABCO-A, Suspected Serious Injury) |
| **Totals** | 177,698 county-days; 3,474 fatals; 14,039 serious injuries |

---

### Massachusetts — MassDOT IMPACT
| Item | Value |
|------|-------|
| **File** | `processed/massachusetts_massdot_county_day.parquet` |
| **Script** | `code/build_massachusetts_massdot.py` |
| **Source** | MassDOT GIS ArcGIS server — gis.massdot.state.ma.us |
| **URLs** | 2013–2019: `CrashClosedYear/CrashClosedYear{yr}/FeatureServer/0` |
|  | 2020: `Dashboard/CrashClosedYear2020_Views/MapServer/4` |
| **Coverage** | 2013–2020, 14 counties, county-day granularity |
| **Key fields** | `fips`, `date`, `ma_fatals`, `ma_serious_inj`, `ma_crashes` |
| **Serious inj.** | `NUMB_NONFATAL_INJR` where `MAX_INJR_SVRTY_CL` contains "Incapacitating" (KABCO-A) |
| **Schema note** | 2013–2017 layers use `CRASH_DATETIME` field; 2018–2020 layers use `CRASH_DATE`. Script handles both automatically. |
| **No auth** | Public, no API key required |

---

### Nevada — NDOT
| Item | Value |
|------|-------|
| **File** | `processed/nevada_ndot_county_day.parquet` |
| **Script** | `code/build_nevada_ndot.py` |
| **Source** | Nevada DOT CrashData OpenData FeatureServer |
| **URL** | `gis.dot.nv.gov/arcgis/rest/services/ArcGISOnline/CrashData_OpenData/FeatureServer/0` |
| **Coverage** | 2016–2024 (2013–2015 return 0 records in this portal), 17 counties, county-day |
| **Key fields** | `fips`, `date`, `nv_fatals`, `nv_serious_inj`, `nv_all_injured`, `nv_crashes` |
| **Serious inj.** | `sum(Injured)` where `Injury_Type == 'A'` (KABCO-A) |
| **Totals** | 26,814 county-days; 3,768 fatals; 16,258 serious injuries |
| **No auth** | Public, no API key required |

---

### Pennsylvania — PennDOT
| Item | Value |
|------|-------|
| **File** | `processed/pennsylvania_penndot_county_month.parquet` |
| **Script** | `code/build_pennsylvania_penndot.py` |
| **Source** | PennDOT Socrata open data portal |
| **URL** | `https://data.pa.gov/resource/dc5b-gebx.json` |
| **Coverage** | 2013–2020, 67 counties, **county-MONTH** (no calendar day in data) |
| **Key fields** | `fips`, `date`, `pa_fatals`, `pa_serious_inj`, `pa_crashes` |
| **Serious inj.** | `maj_inj_count` (major/suspected serious injuries, KABCO-A) |
| **Totals** | 6,309 county-months; 9,036 fatals; 30,301 serious injuries |
| **Note** | Date granularity is year-month only; `date` column is set to 1st of the month. |
| **No auth** | Public, no API key required (Socrata) |

---

### Wisconsin — WisDOT Community Maps
| Item | Value |
|------|-------|
| **File** | `processed/wisconsin_dot_county_day.parquet` |
| **Script** | `code/build_wisconsin_dot.py` |
| **Source** | Wisconsin Community Maps crash API |
| **URL** | `https://transportal.cee.wisc.edu/partners/community-maps/crash/public/crashesKML.do` |
| **Coverage** | 2013–2024, 72 counties, county-day granularity |
| **Key fields** | `fips`, `date`, `wi_fatals`, `wi_serious_inj`, `wi_crashes` |
| **Serious inj.** | `sum(totinj)` where `injsvr == 'A'` (Suspected Serious Injury, KABCO-A) |
| **Totals** | 249,365 county-days; 6,840 fatals; 51,837 serious injuries |
| **API note** | Must pass `injsvr=K,A,B,C,O` explicitly — default returns fatal crashes only |
| **No auth** | Public, no API key required |

---

## Common Column Schema

All output files share a common schema (column names are state-prefixed):

| Column | Type | Description |
|--------|------|-------------|
| `fips` | str(5) | 5-digit county FIPS code (e.g. "25017" = Middlesex MA) |
| `date` | datetime64 | Date (day precision) or 1st-of-month for PA |
| `{st}_fatals` | int/float | Total fatalities in county on that date |
| `{st}_serious_inj` | int/float | Suspected serious injuries (KABCO-A proxy) |
| `{st}_crashes` | int | Total crash records (all severities) |

Where `{st}` is the two-letter state abbreviation (ca, fl, il, ia, ma, nv, pa, wi).

---

## Re-running Downloads

Each script is self-contained. From the repo root:

```bash
python code/build_iowa_dot.py
python code/build_massachusetts_massdot.py
python code/build_nevada_ndot.py
python code/build_pennsylvania_penndot.py   # slow: ~967k records
python code/build_wisconsin_dot.py          # slow: 864 API calls
python code/build_florida_fdot.py           # limited to 2013-2016
```

Dependencies: `pandas`, `numpy`, `requests`, `pyarrow`

All outputs are written to `data/processed/`.
