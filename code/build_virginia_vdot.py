"""
build_virginia_vdot.py
========================================================
Download Virginia VDOT crash data from the ArcGIS FeatureServer and
build a county-day panel of fatalities and serious injuries.

Source: Virginia DOT CrashData ArcGIS FeatureServer (layer 2)
URL: https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/CrashData_test/FeatureServer/2
Total records: ~1,121,988 (as of probe date)
maxRecordCount per query: 2,000
No authentication required.

Key fields (confirmed via API probe):
  CRASH_DT         — crash date (epoch milliseconds)
  CRASH_YEAR       — 4-digit year (string in API, e.g. '2017')
  CRASH_SEVERITY   — KABCO: K=Fatal, A=Severe, B=Visible, C=Nonvisible, O=PDO
  K_PEOPLE         — count of fatalities per crash
  PERSONS_INJURED  — count of injured persons per crash
  PHYSICAL_JURIS   — jurisdiction like "029. Fairfax County" or "121. City of Newport News"

FIPS mapping:
  PHYSICAL_JURIS contains a 3-digit prefix (the VDOT jurisdiction code).
  This code maps to Virginia county/city/town FIPS via a full lookup table.
  Towns (not independent cities) are rolled up to their parent county FIPS.
  Virginia FIPS: 51001–51199 (counties), 51510–51840 (independent cities).

Serious injuries:
  sum(PERSONS_INJURED) for crashes where CRASH_SEVERITY == 'A'

Coverage: 2017–2024

Output: data/processed/virginia_vdot_county_day.parquet
Columns: fips, date, va_fatals, va_serious_inj, va_crashes
"""
import sys, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from state_dot_sources import strict_arcgis_dataframe, validate_source_frame, write_state_manifest_or_raise

warnings.filterwarnings("ignore")
log = get_logger("virginia_vdot")

OUT_PATH = DATA_PROC / "virginia_vdot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

FEATURE_SERVER = (
    "https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/"
    "CrashData_test/FeatureServer/2/query"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}

YEARS      = list(range(2017, 2025))   # 2017–2024
OUT_FIELDS = "CRASH_DT,CRASH_YEAR,CRASH_SEVERITY,K_PEOPLE,PERSONS_INJURED,PHYSICAL_JURIS"
PAGE_SIZE  = 2000   # server maxRecordCount
FETCH_FAILURES: dict[int, BaseException] = {}

# ── Virginia FIPS mapping ──────────────────────────────────────────────────────
# Key: 3-digit VDOT jurisdiction code (zero-padded string)
# Value: 5-digit Census FIPS code
# Counties (VDOT codes 000-099) → FIPS 51001-51199
# Independent cities (VDOT codes 100-155) → FIPS 51510-51840
# Towns (VDOT codes 101+) → parent county FIPS

VA_VDOT_FIPS: dict[str, str] = {
    # ── Counties ─────────────────────────────────────────────────────────────
    "000": "51013",   # Arlington County
    "001": "51001",   # Accomack County
    "002": "51003",   # Albemarle County
    "003": "51005",   # Alleghany County
    "004": "51007",   # Amelia County
    "005": "51009",   # Amherst County
    "006": "51011",   # Appomattox County
    "007": "51015",   # Augusta County
    "008": "51017",   # Bath County
    "009": "51019",   # Bedford County
    "010": "51021",   # Bland County
    "011": "51023",   # Botetourt County
    "012": "51025",   # Brunswick County
    "013": "51027",   # Buchanan County
    "014": "51029",   # Buckingham County
    "015": "51031",   # Campbell County
    "016": "51033",   # Caroline County
    "017": "51035",   # Carroll County
    "018": "51036",   # Charles City County
    "019": "51037",   # Charlotte County
    "020": "51041",   # Chesterfield County
    "021": "51043",   # Clarke County
    "022": "51045",   # Craig County
    "023": "51047",   # Culpeper County
    "024": "51049",   # Cumberland County
    "025": "51051",   # Dickenson County
    "026": "51053",   # Dinwiddie County
    "028": "51057",   # Essex County
    "029": "51059",   # Fairfax County
    "030": "51061",   # Fauquier County
    "031": "51063",   # Floyd County
    "032": "51065",   # Fluvanna County
    "033": "51067",   # Franklin County
    "034": "51069",   # Frederick County
    "035": "51071",   # Giles County
    "036": "51073",   # Gloucester County
    "037": "51075",   # Goochland County
    "038": "51077",   # Grayson County
    "039": "51079",   # Greene County
    "040": "51081",   # Greensville County
    "041": "51083",   # Halifax County
    "042": "51085",   # Hanover County
    "043": "51087",   # Henrico County
    "044": "51089",   # Henry County
    "045": "51091",   # Highland County
    "046": "51093",   # Isle of Wight County
    "047": "51095",   # James City County
    "048": "51099",   # King George County
    "049": "51097",   # King & Queen County
    "050": "51101",   # King William County
    "051": "51103",   # Lancaster County
    "052": "51105",   # Lee County
    "053": "51107",   # Loudoun County
    "054": "51109",   # Louisa County
    "055": "51111",   # Lunenburg County
    "056": "51113",   # Madison County
    "057": "51115",   # Mathews County
    "058": "51117",   # Mecklenburg County
    "059": "51119",   # Middlesex County
    "060": "51121",   # Montgomery County
    "062": "51125",   # Nelson County
    "063": "51127",   # New Kent County
    "065": "51131",   # Northampton County
    "066": "51133",   # Northumberland County
    "067": "51135",   # Nottoway County
    "068": "51137",   # Orange County
    "069": "51139",   # Page County
    "070": "51141",   # Patrick County
    "071": "51143",   # Pittsylvania County
    "072": "51145",   # Powhatan County
    "073": "51147",   # Prince Edward County
    "074": "51149",   # Prince George County
    "076": "51153",   # Prince William County
    "077": "51155",   # Pulaski County
    "078": "51157",   # Rappahannock County
    "079": "51159",   # Richmond County
    "080": "51161",   # Roanoke County
    "081": "51163",   # Rockbridge County
    "082": "51165",   # Rockingham County
    "083": "51167",   # Russell County
    "084": "51169",   # Scott County
    "085": "51171",   # Shenandoah County
    "086": "51173",   # Smyth County
    "087": "51175",   # Southampton County
    "088": "51177",   # Spotsylvania County
    "089": "51179",   # Stafford County
    "090": "51181",   # Surry County
    "091": "51183",   # Sussex County
    "092": "51185",   # Tazewell County
    "093": "51187",   # Warren County
    "095": "51191",   # Washington County
    "096": "51193",   # Westmoreland County
    "097": "51195",   # Wise County
    "098": "51197",   # Wythe County
    "099": "51199",   # York County
    # ── Independent Cities ────────────────────────────────────────────────────
    "100": "51510",   # City of Alexandria
    "102": "51520",   # City of Bristol
    "103": "51530",   # City of Buena Vista
    "104": "51540",   # City of Charlottesville
    "105": "51005",   # Town of Clifton Forge → Alleghany County (city abolished 2001)
    "106": "51570",   # City of Colonial Heights
    "107": "51580",   # City of Covington
    "108": "51590",   # City of Danville
    "109": "51595",   # City of Emporia
    "110": "51610",   # City of Falls Church
    "111": "51630",   # City of Fredericksburg
    "113": "51640",   # City of Galax
    "114": "51650",   # City of Hampton
    "115": "51660",   # City of Harrisonburg
    "116": "51670",   # City of Hopewell
    "117": "51678",   # City of Lexington
    "118": "51680",   # City of Lynchburg
    "120": "51690",   # City of Martinsville
    "121": "51700",   # City of Newport News
    "122": "51710",   # City of Norfolk
    "123": "51730",   # City of Petersburg
    "124": "51740",   # City of Portsmouth
    "126": "51750",   # City of Radford
    "127": "51760",   # City of Richmond
    "128": "51770",   # City of Roanoke
    "129": "51775",   # City of Salem
    "131": "51550",   # City of Chesapeake
    "132": "51790",   # City of Staunton
    "133": "51800",   # City of Suffolk
    "134": "51810",   # City of Virginia Beach
    "136": "51820",   # City of Waynesboro
    "137": "51830",   # City of Williamsburg
    "138": "51840",   # City of Winchester
    "145": "51620",   # City of Franklin
    "146": "51720",   # City of Norton
    "147": "51735",   # City of Poquoson
    "151": "51600",   # City of Fairfax
    "152": "51685",   # City of Manassas Park
    "155": "51683",   # City of Manassas
    # ── Towns (mapped to parent county FIPS) ─────────────────────────────────
    "101": "51195",   # Town of Big Stone Gap → Wise County
    "112": "51187",   # Town of Front Royal → Warren County
    "119": "51173",   # Town of Marion → Smyth County
    "125": "51155",   # Town of Pulaski → Pulaski County
    "130": "51083",   # Town of South Boston → Halifax County
    "139": "51197",   # Town of Wytheville → Wythe County
    "140": "51191",   # Town of Abingdon → Washington County
    "141": "51019",   # Town of Bedford → Bedford County
    "142": "51135",   # Town of Blackstone → Nottoway County
    "143": "51185",   # Town of Bluefield → Tazewell County
    "144": "51147",   # Town of Farmville → Prince Edward County
    "148": "51185",   # Town of Richlands → Tazewell County
    "149": "51161",   # Town of Vinton → Roanoke County
    "150": "51121",   # Town of Blacksburg → Montgomery County
    "153": "51059",   # Town of Vienna → Fairfax County
    "154": "51121",   # Town of Christiansburg → Montgomery County
    "156": "51061",   # Town of Warrenton → Fauquier County
    "157": "51067",   # Town of Rocky Mount → Franklin County
    "158": "51185",   # Town of Tazewell → Tazewell County
    "159": "51139",   # Town of Luray → Page County
    "160": "51001",   # Town of Accomac → Accomack County
    "161": "51025",   # Town of Alberta → Brunswick County
    "162": "51031",   # Town of Altavista → Campbell County
    "163": "51009",   # Town of Amherst → Amherst County
    "164": "51195",   # Town of Appalachia → Wise County
    "165": "51011",   # Town of Appomattox → Appomattox County
    "166": "51085",   # Town of Ashland → Hanover County
    "167": "51001",   # Town of Belle Haven → Accomack County
    "168": "51043",   # Town of Berryville → Clarke County
    "169": "51001",   # Town of Bloxom → Accomack County
    "170": "51067",   # Town of Boones Mill → Franklin County
    "171": "51033",   # Town of Bowling Green → Caroline County
    "172": "51043",   # Town of Boyce → Clarke County
    "173": "51117",   # Town of Boydton → Mecklenburg County
    "174": "51175",   # Town of Boykins → Southampton County
    "176": "51165",   # Town of Bridgewater → Rockingham County
    "177": "51165",   # Town of Broadway → Rockingham County
    "179": "51031",   # Town of Brookneal → Campbell County
    "180": "51023",   # Town of Buchanan → Botetourt County
    "184": "51185",   # Town of Cedar Bluff → Tazewell County
    "186": "51117",   # Town of Chase City → Mecklenburg County
    "187": "51143",   # Town of Chatham → Pittsylvania County
    "188": "51131",   # Town of Cheriton → Northampton County
    "189": "51173",   # Town of Chilhowie → Smyth County
    "190": "51001",   # Town of Chincoteague → Accomack County
    "192": "51117",   # Town of Clarksville → Mecklenburg County
    "193": "51167",   # Town of Cleveland → Russell County
    "195": "51169",   # Town of Clinchport → Scott County
    "196": "51051",   # Town of Clintwood → Dickenson County
    "198": "51195",   # Town of Coeburn → Wise County
    "199": "51193",   # Town of Colonial Beach → Westmoreland County
    "202": "51015",   # Town of Craigsville → Augusta County
    "203": "51135",   # Town of Crewe → Nottoway County
    "204": "51047",   # Town of Culpeper → Culpeper County
    "206": "51165",   # Town of Dayton → Rockingham County
    "210": "51155",   # Town of Dublin → Pulaski County
    "211": "51169",   # Town of Duffield → Scott County
    "212": "51153",   # Town of Dumfries → Prince William County
    "215": "51171",   # Town of Edinburg → Shenandoah County
    "216": "51165",   # Town of Elkton → Rockingham County
    "217": "51131",   # Town of Exmore → Northampton County
    "218": "51023",   # Town of Fincastle → Botetourt County
    "219": "51063",   # Town of Floyd → Floyd County
    "220": "51077",   # Town of Fries → Grayson County
    "221": "51169",   # Town of Gate City → Scott County
    "223": "51163",   # Town of Glasgow → Rockbridge County
    "225": "51137",   # Town of Gordonsville → Orange County
    "226": "51163",   # Town of Goshen → Rockbridge County
    "229": "51027",   # Town of Grundy → Buchanan County
    "230": "51083",   # Town of Halifax → Halifax County
    "231": "51001",   # Town of Hallwood → Accomack County
    "232": "51107",   # Town of Hamilton → Loudoun County
    "233": "51153",   # Town of Haymarket → Prince William County
    "235": "51059",   # Town of Herndon → Fairfax County
    "236": "51107",   # Town of Hillsboro → Loudoun County
    "237": "51035",   # Town of Hillsville → Carroll County
    "239": "51167",   # Town of Honaker → Russell County
    "240": "51077",   # Town of Independence → Grayson County
    "242": "51103",   # Town of Irvington → Lancaster County
    "243": "51175",   # Town of Ivor → Southampton County
    "244": "51183",   # Town of Jarratt → Sussex County
    "245": "51105",   # Town of Jonesville → Lee County
    "246": "51001",   # Town of Keller → Accomack County
    "247": "51111",   # Town of Kenbridge → Lunenburg County
    "248": "51037",   # Town of Keysville → Charlotte County
    "249": "51103",   # Town of Kilmarnock → Lancaster County
    "251": "51025",   # Town of Lawrenceville → Brunswick County
    "252": "51167",   # Town of Lebanon → Russell County
    "253": "51107",   # Town of Leesburg → Loudoun County
    "254": "51109",   # Town of Louisa → Louisa County
    "255": "51107",   # Town of Lovettsville → Loudoun County
    "257": "51053",   # Town of McKenney → Dinwiddie County
    "258": "51001",   # Town of Melfa → Accomack County
    "259": "51107",   # Town of Middleburg → Loudoun County
    "260": "51069",   # Town of Middletown → Frederick County
    "261": "51109",   # Town of Mineral → Louisa County
    "262": "51091",   # Town of Monterey → Highland County
    "263": "51193",   # Town of Montross → Westmoreland County
    "264": "51165",   # Town of Mount Crawford → Rockingham County
    "265": "51171",   # Town of Mount Jackson → Shenandoah County
    "267": "51131",   # Town of Nassawadox → Northampton County
    "269": "51171",   # Town of New Market → Shenandoah County
    "272": "51153",   # Town of Occoquan → Prince William County
    "273": "51001",   # Town of Onancock → Accomack County
    "274": "51001",   # Town of Onley → Accomack County
    "275": "51137",   # Town of Orange → Orange County
    "276": "51001",   # Town of Painter → Accomack County
    "279": "51071",   # Town of Pearisburg → Giles County
    "280": "51071",   # Town of Pembroke → Giles County
    "281": "51105",   # Town of Pennington Gap → Lee County
    "285": "51195",   # Town of Pound → Wise County
    "286": "51107",   # Town of Purcellville → Loudoun County
    "288": "51061",   # Town of Remington → Fauquier County
    "289": "51071",   # Town of Rich Creek → Giles County
    "290": "51089",   # Town of Ridgeway → Henry County
    "291": "51107",   # Town of Round Hill → Loudoun County
    "292": "51197",   # Town of Rural Retreat → Wythe County
    "294": "51195",   # Town of Saint Paul → Wise County
    "298": "51003",   # Town of Scottsville → Albemarle County
    "299": "51171",   # Town of Shenandoah → Page County
    "300": "51093",   # Town of Smithfield → Isle of Wight County
    "301": "51117",   # Town of South Hill → Mecklenburg County
    "302": "51079",   # Town of Stanardsville → Greene County
    "303": "51139",   # Town of Stanley → Page County
    "304": "51069",   # Town of Stephens City → Frederick County
    "305": "51183",   # Town of Stony Creek → Sussex County
    "306": "51171",   # Town of Strasburg → Shenandoah County
    "307": "51141",   # Town of Stuart → Patrick County
    "310": "51057",   # Town of Tappahannock → Essex County
    "311": "51061",   # Town of The Plains → Fauquier County
    "312": "51165",   # Town of Timberville → Rockingham County
    "313": "51171",   # Town of Toms Brook → Shenandoah County
    "314": "51077",   # Town of Troutdale → Grayson County
    "315": "51023",   # Town of Troutville → Botetourt County
    "317": "51111",   # Town of Victoria → Lunenburg County
    "320": "51183",   # Town of Wakefield → Sussex County
    "321": "51159",   # Town of Warsaw → Richmond County
    "323": "51183",   # Town of Waverly → Sussex County
    "324": "51169",   # Town of Weber City → Scott County
    "325": "51101",   # Town of West Point → King William County
    "328": "51093",   # Town of Windsor → Isle of Wight County
    "329": "51195",   # Town of Wise → Wise County
    "330": "51171",   # Town of Woodstock → Shenandoah County
    "331": "51031",   # Town of Hurt → Campbell County
    "339": "51051",   # Town of Clinchco → Dickenson County
    # ── Additional towns (Census TIGERweb spatial join to containing county) ──
    "175": "51175",   # Town of Branchville -> Southampton County
    "178": "51025",   # Town of Brodnax -> Brunswick County
    "181": "51135",   # Town of Burkeville -> Nottoway County
    "182": "51131",   # Town of Cape Charles -> Northampton County
    "183": "51175",   # Town of Capron -> Southampton County
    "185": "51037",   # Town of Charlotte C.H. -> Charlotte County
    "191": "51181",   # Town of Claremont -> Surry County
    "194": "51059",   # Town of Clifton -> Fairfax County
    "200": "51065",   # Town of Columbia -> Fluvanna County
    "201": "51175",   # Town of Courtland -> Southampton County
    "205": "51191",   # Town of Damascus -> Washington County
    "207": "51181",   # Town of Dendron -> Surry County
    "208": "51029",   # Town of Dillwyn -> Buckingham County
    "209": "51037",   # Town of Drakes Branch -> Charlotte County
    "213": "51169",   # Town of Dungannon -> Scott County
    "214": "51131",   # Town of Eastville -> Northampton County
    "222": "51191",   # Town of Glade Spring -> Washington County
    "224": "51071",   # Town of Glen Lyn -> Giles County
    "227": "51143",   # Town of Gretna -> Pittsylvania County
    "228": "51165",   # Town of Grottoes -> Rockingham County
    "234": "51051",   # Town of Haysi -> Dickenson County
    "241": "51005",   # Town of Iron Gate -> Alleghany County
    "250": "51117",   # Town of LaCrosse -> Mecklenburg County
    "256": "51113",   # Town of Madison -> Madison County
    "266": "51071",   # Town of Narrows -> Giles County
    "268": "51045",   # Town of New Castle -> Craig County
    "270": "51175",   # Town of Newsoms -> Southampton County
    "271": "51169",   # Town of Nickelsville -> Scott County
    "277": "51011",   # Town of Pamplin City -> Appomattox County
    "278": "51001",   # Town of Parksley -> Accomack County
    "282": "51037",   # Town of Phenix -> Charlotte County
    "283": "51185",   # Town of Pocahontas -> Tazewell County
    "284": "51033",   # Town of Port Royal -> Caroline County
    "287": "51153",   # Town of Quantico -> Prince William County
    "293": "51105",   # Town of St. Charles -> Lee County
    "295": "51173",   # Town of Saltville -> Smyth County
    "296": "51001",   # Town of Saxis -> Accomack County
    "297": "51083",   # Town of Scottsburg -> Halifax County
    "308": "51181",   # Town of Surry -> Surry County
    "309": "51001",   # Town of Tangier -> Accomack County
    "316": "51119",   # Town of Urbanna -> Middlesex County
    "318": "51083",   # Town of Virgilina -> Halifax County
    "319": "51001",   # Town of Wachapreague -> Accomack County
    "322": "51157",   # Town of Washington -> Rappahannock County
    "327": "51103",   # Town of White Stone -> Lancaster County
}


def extract_vdot_code(physical_juris: str) -> str | None:
    """Extract the 3-digit VDOT jurisdiction code from PHYSICAL_JURIS string.

    E.g. '029. Fairfax County' → '029'
         '121. City of Newport News' → '121'
    """
    if not physical_juris or not isinstance(physical_juris, str):
        return None
    parts = physical_juris.split(".", 1)
    if not parts:
        return None
    code = parts[0].strip().zfill(3)
    return code if code.isdigit() else None


def fetch_year(session: requests.Session, year: int) -> pd.DataFrame | None:
    """Download all crash records for one year with offset-based pagination."""
    where_clause = f"CRASH_YEAR = '{year}'"

    # Count total records for this year
    try:
        r = session.get(FEATURE_SERVER, params={
            "where": where_clause,
            "returnCountOnly": "true",
            "f": "json",
        }, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if "error" in resp:
            log.warning("  [%d] count ArcGIS error: %s", year, resp["error"])
            return None
        total = resp.get("count", 0)
        log.info("  [%d] %d records", year, total)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.warning("  [%d] count query failed: %s", year, exc)
        return None

    if total == 0:
        log.warning("  [%d] 0 records — skipping", year)
        return None

    try:
        return strict_arcgis_dataframe(session, url=FEATURE_SERVER, where=where_clause,
                                       expected_count=total, id_field="OBJECTID",
                                       out_fields=OUT_FIELDS, page_size=PAGE_SIZE)
    except Exception as exc:
        FETCH_FAILURES[year] = exc
        log.error("  [%d] strict pagination failed: %s", year, exc)
        return None

    # Paginate with resultOffset
    parts = []
    offset = 0
    while offset < total:
        for attempt in range(3):
            try:
                r = session.get(FEATURE_SERVER, params={
                    "where":             where_clause,
                    "outFields":         OUT_FIELDS,
                    "resultOffset":      offset,
                    "resultRecordCount": PAGE_SIZE,
                    "f":                 "json",
                }, timeout=90)
                r.raise_for_status()
                page = r.json()
                # ArcGIS can return HTTP 200 with an embedded error
                if "error" in page:
                    raise ValueError(f"ArcGIS error: {page['error']}")
                break
            except Exception as exc:
                wait = 5 * (attempt + 1)
                log.warning("  [%d] offset=%d attempt %d failed: %s; retry in %ds",
                            year, offset, attempt + 1, exc, wait)
                if attempt < 2:
                    time.sleep(wait)
                else:
                    log.error("  [%d] gave up at offset=%d", year, offset)
                    offset = total   # exit outer while loop
                    page = {}
                    break

        features = page.get("features", [])
        if not features:
            break

        rows = [f["attributes"] for f in features]
        parts.append(pd.DataFrame(rows))
        offset += len(rows)

        if offset % 50_000 == 0 or len(rows) < PAGE_SIZE:
            log.info("  [%d] fetched %d / %d", year, offset, total)

        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.2)

    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    log.info("  [%d] raw total: %d rows", year, len(df))
    return df


def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Aggregate raw crash records to county-day panel."""
    if df is None or df.empty:
        return None
    df = df.copy()

    # ── Date ─────────────────────────────────────────────────────────────────
    df["crash_date"] = pd.to_datetime(df["CRASH_DT"], unit="ms", errors="coerce")
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    # ── Jurisdiction → FIPS ──────────────────────────────────────────────────
    df["vdot_code"] = df["PHYSICAL_JURIS"].apply(extract_vdot_code)
    df["fips"]      = df["vdot_code"].map(VA_VDOT_FIPS)
    n_miss = df["fips"].isna().sum()
    if n_miss:
        log.warning("  [%d] %d rows unmapped jurisdiction: %s",
                    year, n_miss,
                    df.loc[df["fips"].isna(), "PHYSICAL_JURIS"]
                      .value_counts().head(10).to_dict())
    df = df.dropna(subset=["fips"])

    # ── Severity ─────────────────────────────────────────────────────────────
    df["fatals"]   = pd.to_numeric(df["K_PEOPLE"],        errors="coerce").fillna(0)
    df["injured"]  = pd.to_numeric(df["PERSONS_INJURED"], errors="coerce").fillna(0)

    # Serious injuries = PERSONS_INJURED for crashes where CRASH_SEVERITY == 'A'
    df["crash_sev"]    = df["CRASH_SEVERITY"].astype(str).str.strip().str.upper()
    df["serious_inj"]  = df["injured"].where(df["crash_sev"] == "A", 0)

    # ── Aggregate to county-day ───────────────────────────────────────────────
    agg = (
        df.groupby(["fips", "crash_date"])
          .agg(
              va_fatals     =("fatals",      "sum"),
              va_injury_proxy=("serious_inj", "sum"),
              va_crashes    =("fatals",      "count"),
          )
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    log.info("  [%d] → %d county-days  va_fatals=%.0f  va_serious_inj=%.0f",
             year, len(agg), agg["va_fatals"].sum(), agg["va_injury_proxy"].sum())
    return agg


# Executed only as a script. Without this guard the whole download-and-write
# pipeline ran on *import*, so merely importing this module (from a test, an
# audit, or another builder) silently re-downloaded the source and overwrote
# the processed panel on disk.
if __name__ == "__main__":
    # ── Main ──────────────────────────────────────────────────────────────────────
    log.info("Downloading Virginia VDOT crash data (2017–2024) …")

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    coverage_rows = []

    for yr in YEARS:
        log.info("Year %d …", yr)
        raw = fetch_year(session, yr)
        coverage_rows.append(validate_source_frame("VA", yr, raw,
            required_columns={"CRASH_DT", "PHYSICAL_JURIS", "K_PEOPLE", "PERSONS_INJURED", "CRASH_SEVERITY"},
            date_column="CRASH_DT", outcome_columns={"K_PEOPLE", "PERSONS_INJURED"}, date_unit="ms",
            geography_column="PHYSICAL_JURIS", geography_mapper=lambda value: VA_VDOT_FIPS.get(extract_vdot_code(value)),
            terminal_error=FETCH_FAILURES.get(yr)))
        agg = process_year(raw, yr)
        if agg is not None:
            parts.append(agg)
        del raw, agg
        time.sleep(1.0)
        gc.collect()

    session.close()
    write_state_manifest_or_raise("VA", coverage_rows, output_dir=DATA_PROC / "coverage")

    if not parts:
        log.error("No Virginia data downloaded.")
        sys.exit(1)

    va_panel = pd.concat(parts, ignore_index=True)
    va_panel["date"] = pd.to_datetime(va_panel["date"])

    # Final dedup/sum in case any year overlap at boundaries
    va_panel = (
        va_panel.groupby(["fips", "date"])
          .agg(
              va_fatals     =("va_fatals",      "sum"),
              va_injury_proxy=("va_injury_proxy", "sum"),
              va_crashes    =("va_crashes",     "sum"),
          )
          .reset_index()
    )

    log.info("\nFinal Virginia VDOT panel:")
    log.info("  Rows:       %d", len(va_panel))
    log.info("  Counties:   %d", va_panel["fips"].nunique())
    log.info("  Date range: %s – %s",
             va_panel["date"].min().date(), va_panel["date"].max().date())
    log.info("  Total va_fatals:      %.0f", va_panel["va_fatals"].sum())
    va_panel["va_serious_inj"] = np.nan
    log.info("  Total va_injury_proxy: %.0f", va_panel["va_injury_proxy"].sum())
    log.info("  Total va_crashes:     %.0f", va_panel["va_crashes"].sum())

    va_panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s", OUT_PATH)
