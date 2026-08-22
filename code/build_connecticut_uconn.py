"""
build_connecticut_uconn.py
========================================================
Download Connecticut statewide crash data from UConn CTDOT's public
"ConnecticutCrash" FeatureServer (Crash and Person layers) and build a
county-day panel of fatalities and serious injuries.

Source: UConn Connecticut Transportation Safety Research Center
URL: https://gis.cti.uconn.edu/arcgis/rest/services/Crash_Dashboards/ConnecticutCrash/FeatureServer
Coverage: 2015-2021 requested. Connecticut retired its 8 counties for 9
planning regions in January 2022; this builder only covers the legacy-county
period (2015-2021) to avoid a per-year-varying county universe, which this
project's state contract does not otherwise need to support. 2022-2025 data
exists in the source but is out of scope here.
No authentication required.

Connecticut has no crash-level fatality/injury COUNT field; severity is
person-level only (KABCO), on a separate Person layer keyed by CrashID.
Person-level `InjuryStatus` values used:
  "Fatal Injury (K)"             -> person_fatals
  "Suspected Serious Injury (A)" -> serious_injury_persons

Key fields (confirmed by probe):
  CrashDate / CrashDateYear — crash date
  CrashTownName             — Connecticut town name (not county)
  CrashID                   — joins Crash and Person layers

Output columns: fips, date, ct_fatals, ct_serious_inj, ct_crashes
Output: data/processed/connecticut_uconn_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from state_dot_sources import strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("connecticut_uconn")

OUT_PATH = DATA_PROC / "connecticut_uconn_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

BASE = "https://gis.cti.uconn.edu/arcgis/rest/services/Crash_Dashboards/ConnecticutCrash/FeatureServer"
CRASH_URL = f"{BASE}/0/query"
PERSON_URL = f"{BASE}/1/query"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS = list(range(2015, 2022))
PAGE_SIZE = 2_000  # server maxRecordCount
FETCH_FAILURES: dict[int, BaseException] = {}
CRASH_FIELDS = "CrashID,CrashDate,CrashDateYear,CrashTownName"
PERSON_FIELDS = "CrashID,InjuryStatus"

# Connecticut's 169 towns mapped to their pre-2022 legacy county, from the
# Census Bureau's own 2020 Gazetteer county-subdivision file (GEOID digits
# 3-5 are the county FIPS) -- not guessed. "Mashantucket" (the federally
# recognized Mashantucket Pequot reservation, 118 crash records) is not one
# of the 169 incorporated towns; it lies within the town of Ledyard, New
# London County, a well-documented, unambiguous location.
CT_TOWN_COUNTY: dict[str, str] = {
    "ANDOVER": "09013", "ANSONIA": "09009", "ASHFORD": "09015", "AVON": "09003",
    "BARKHAMSTED": "09005", "BEACON FALLS": "09009", "BERLIN": "09003",
    "BETHANY": "09009", "BETHEL": "09001", "BETHLEHEM": "09005",
    "BLOOMFIELD": "09003", "BOLTON": "09013", "BOZRAH": "09011",
    "BRANFORD": "09009", "BRIDGEPORT": "09001", "BRIDGEWATER": "09005",
    "BRISTOL": "09003", "BROOKFIELD": "09001", "BROOKLYN": "09015",
    "BURLINGTON": "09003", "CANAAN": "09005", "CANTERBURY": "09015",
    "CANTON": "09003", "CHAPLIN": "09015", "CHESHIRE": "09009",
    "CHESTER": "09007", "CLINTON": "09009", "COLCHESTER": "09011",
    "COLEBROOK": "09005", "COLUMBIA": "09013", "CORNWALL": "09005",
    "COVENTRY": "09013", "CROMWELL": "09007", "DANBURY": "09001",
    "DARIEN": "09001", "DEEP RIVER": "09007", "DERBY": "09009",
    "DURHAM": "09009", "EAST GRANBY": "09003", "EAST HADDAM": "09007",
    "EAST HAMPTON": "09007", "EAST HARTFORD": "09003", "EAST HAVEN": "09009",
    "EAST LYME": "09011", "EAST WINDSOR": "09003", "EASTFORD": "09015",
    "EASTON": "09001", "ELLINGTON": "09013", "ENFIELD": "09003",
    "ESSEX": "09007", "FAIRFIELD": "09001", "FARMINGTON": "09003",
    "FRANKLIN": "09011", "GLASTONBURY": "09003", "GOSHEN": "09005",
    "GRANBY": "09003", "GREENWICH": "09001", "GRISWOLD": "09011",
    "GROTON": "09011", "GUILFORD": "09009", "HADDAM": "09007",
    "HAMDEN": "09009", "HAMPTON": "09015", "HARTFORD": "09003",
    "HARTLAND": "09005", "HARWINTON": "09005", "HEBRON": "09013",
    "KENT": "09005", "KILLINGLY": "09015", "KILLINGWORTH": "09009",
    "LEBANON": "09011", "LEDYARD": "09011", "LISBON": "09011",
    "LITCHFIELD": "09005", "LYME": "09011", "MADISON": "09009",
    "MANCHESTER": "09003", "MANSFIELD": "09013", "MARLBOROUGH": "09013",
    "MERIDEN": "09009", "MIDDLEBURY": "09009", "MIDDLEFIELD": "09007",
    "MIDDLETOWN": "09007", "MILFORD": "09009", "MONROE": "09001",
    "MONTVILLE": "09011", "MORRIS": "09005", "NAUGATUCK": "09009",
    "NEW BRITAIN": "09003", "NEW CANAAN": "09001", "NEW FAIRFIELD": "09001",
    "NEW HARTFORD": "09005", "NEW HAVEN": "09009", "NEW LONDON": "09011",
    "NEW MILFORD": "09005", "NEWINGTON": "09003", "NEWTOWN": "09001",
    "NORFOLK": "09005", "NORTH BRANFORD": "09009", "NORTH CANAAN": "09005",
    "NORTH HAVEN": "09009", "NORTH STONINGTON": "09011", "NORWALK": "09001",
    "NORWICH": "09011", "OLD LYME": "09011", "OLD SAYBROOK": "09007",
    "ORANGE": "09009", "OXFORD": "09009", "PLAINFIELD": "09015",
    "PLAINVILLE": "09003", "PLYMOUTH": "09005", "POMFRET": "09015",
    "PORTLAND": "09007", "PRESTON": "09011", "PROSPECT": "09009",
    "PUTNAM": "09015", "REDDING": "09001", "RIDGEFIELD": "09001",
    "ROCKY HILL": "09003", "ROXBURY": "09005", "SALEM": "09011",
    "SALISBURY": "09005", "SCOTLAND": "09015", "SEYMOUR": "09009",
    "SHARON": "09005", "SHELTON": "09009", "SHERMAN": "09001",
    "SIMSBURY": "09003", "SOMERS": "09013", "SOUTH WINDSOR": "09003",
    "SOUTHBURY": "09009", "SOUTHINGTON": "09003", "SPRAGUE": "09011",
    "STAFFORD": "09013", "STAMFORD": "09001", "STERLING": "09015",
    "STONINGTON": "09011", "STRATFORD": "09001", "SUFFIELD": "09003",
    "THOMASTON": "09005", "THOMPSON": "09015", "TOLLAND": "09013",
    "TORRINGTON": "09005", "TRUMBULL": "09001", "UNION": "09013",
    "VERNON": "09013", "VOLUNTOWN": "09011", "WALLINGFORD": "09009",
    "WARREN": "09005", "WASHINGTON": "09005", "WATERBURY": "09009",
    "WATERFORD": "09011", "WATERTOWN": "09005", "WEST HARTFORD": "09003",
    "WEST HAVEN": "09009", "WESTBROOK": "09007", "WESTON": "09001",
    "WESTPORT": "09001", "WETHERSFIELD": "09003", "WILLINGTON": "09013",
    "WILTON": "09001", "WINCHESTER": "09005", "WINDHAM": "09015",
    "WINDSOR": "09003", "WINDSOR LOCKS": "09003", "WOLCOTT": "09009",
    "WOODBRIDGE": "09009", "WOODBURY": "09005", "WOODSTOCK": "09015",
    "MASHANTUCKET": "09011",
}


def town_to_fips(name: str) -> str | None:
    if name is None:
        return None
    return CT_TOWN_COUNTY.get(str(name).strip().upper())


def fetch_year(session: requests.Session, year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    where = f"CrashDateYear = {year}"
    try:
        crashes = strict_arcgis_dataframe(session, url=CRASH_URL, where=where,
                                          expected_count=_count(session, CRASH_URL, where),
                                          id_field="OBJECTID", out_fields=CRASH_FIELDS, page_size=PAGE_SIZE)
        persons = strict_arcgis_dataframe(session, url=PERSON_URL, where=where,
                                          expected_count=_count(session, PERSON_URL, where),
                                          id_field="OBJECTID", out_fields=PERSON_FIELDS, page_size=PAGE_SIZE)
        return crashes, persons
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return None, None


def _count(session: requests.Session, url: str, where: str) -> int:
    r = session.get(url, params={"where": where, "returnCountOnly": "true", "f": "json"}, timeout=45)
    r.raise_for_status()
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(f"count query error: {resp['error']}")
    return resp.get("count", 0)


def process_year(crashes: pd.DataFrame, persons: pd.DataFrame, year: int) -> pd.DataFrame | None:
    if crashes is None or crashes.empty:
        return None
    crashes = crashes.copy()
    crashes["crash_date"] = pd.to_datetime(crashes["CrashDate"], unit="ms", errors="coerce")
    n_bad_dt = crashes["crash_date"].isna().sum()
    if n_bad_dt:
        log.warning("  [%d] %d rows with unparseable CrashDate dropped", year, n_bad_dt)
    crashes = crashes.dropna(subset=["crash_date"])
    crashes["crash_date"] = crashes["crash_date"].dt.normalize()

    crashes["fips"] = crashes["CrashTownName"].map(town_to_fips)
    n_miss = crashes["fips"].isna().sum()
    if n_miss:
        unmapped = crashes.loc[crashes["fips"].isna(), "CrashTownName"].value_counts().head(10).to_dict()
        log.warning("  [%d] %d rows with unmapped town: %s", year, n_miss, unmapped)
    crashes = crashes.dropna(subset=["fips"])

    # Person-level counts per crash: count occurrences of each CrashID with
    # that InjuryStatus (a fatal crash can have more than one fatality).
    if persons is not None and not persons.empty:
        fatal_per_crash = persons.loc[persons["InjuryStatus"] == "Fatal Injury (K)"].groupby("CrashID").size()
        serious_per_crash = persons.loc[persons["InjuryStatus"] == "Suspected Serious Injury (A)"].groupby("CrashID").size()
    else:
        fatal_per_crash = pd.Series(dtype=int)
        serious_per_crash = pd.Series(dtype=int)
    crashes["ct_fatals"] = crashes["CrashID"].map(fatal_per_crash).fillna(0)
    crashes["ct_serious_inj"] = crashes["CrashID"].map(serious_per_crash).fillna(0)

    agg = (
        crashes.groupby(["fips", "crash_date"])
          .agg(ct_fatals=("ct_fatals", "sum"), ct_serious_inj=("ct_serious_inj", "sum"),
               ct_crashes=("CrashID", "nunique"))
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  ct_fatals=%.0f  ct_serious_inj=%.0f  ct_crashes=%d",
             year, len(agg), agg["ct_fatals"].sum(), agg["ct_serious_inj"].sum(), agg["ct_crashes"].sum())
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
log.info("Downloading Connecticut UConn crash data (2015–2021) …")
log.info("Source: %s", BASE)

session = requests.Session()
session.headers.update(HEADERS)
parts = []
coverage_rows = []

for yr in YEARS:
    log.info("=== Year %d ===", yr)
    crashes, persons = fetch_year(session, yr)
    coverage_rows.append(validate_source_frame("CT", yr, crashes,
        required_columns={"CrashID", "CrashDate", "CrashDateYear", "CrashTownName"},
        date_column="CrashDate", outcome_columns=set(), date_unit="ms",
        geography_column="CrashTownName", geography_mapper=town_to_fips,
        terminal_error=FETCH_FAILURES.get(yr)))
    agg = process_year(crashes, persons, yr)
    if agg is not None:
        parts.append(agg)
    del crashes, persons, agg
    gc.collect()
    time.sleep(1.0)

session.close()
write_state_manifest_or_raise("CT", coverage_rows, output_dir=DATA_PROC / "coverage")

if not parts:
    log.error("No Connecticut data downloaded — aborting.")
    sys.exit(1)

ct_panel = pd.concat(parts, ignore_index=True)
ct_panel["date"] = pd.to_datetime(ct_panel["date"])
ct_panel = (
    ct_panel.groupby(["fips", "date"])
      .agg(ct_fatals=("ct_fatals", "sum"), ct_serious_inj=("ct_serious_inj", "sum"),
           ct_crashes=("ct_crashes", "sum"))
      .reset_index()
)

log.info("")
log.info("Final Connecticut UConn panel:")
log.info("  Rows          : %d", len(ct_panel))
log.info("  Counties      : %d", ct_panel["fips"].nunique())
log.info("  Date range    : %s – %s", ct_panel["date"].min().date(), ct_panel["date"].max().date())
log.info("  ct_fatals     : %.0f", ct_panel["ct_fatals"].sum())
log.info("  ct_serious_inj: %.0f", ct_panel["ct_serious_inj"].sum())
log.info("  ct_crashes    : %d", int(ct_panel["ct_crashes"].sum()))

ct_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
