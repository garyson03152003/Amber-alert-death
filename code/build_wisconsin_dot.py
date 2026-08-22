"""
build_wisconsin_dot.py
========================================================
Download Wisconsin crash data from the Community Maps public API and
build a county-day panel of fatalities and serious injuries.

Source: https://CommunityMaps.wi.gov  (redirects to transportal.cee.wisc.edu)
API:    https://transportal.cee.wisc.edu/partners/community-maps/crash/public/crashesKML.do
Coverage: 2013–2024 (all police-reported crashes)
No authentication required.

Query strategy:
  - One request per (county, year) with all severity levels (K/A/B/C/O)
  - Largest county (Milwaukee) ~22k rows/year — safely under 50k limit
  - 72 counties × 12 years = 864 requests at ~0.5 s each ≈ 8 min

Key response fields (in features[].properties):
  date      — crash date "MM/DD/YYYY"
  cnytname  — county name (UPPERCASE)
  totfatl   — total fatalities in crash
  totinj    — total injuries in crash (all severity levels)
  injsvr    — most-severe injury (K=Fatal, A=Serious, B=Minor, C=Possible, O=PDO)

Output columns:
  fips           — 5-digit FIPS string "55xxx"
  date           — crash date (datetime64)
  wi_fatals      — sum of totfatl per county-day
  wi_serious_inj — sum of totinj where injsvr == 'A' (serious injury crashes)
  wi_crashes     — total crash count per county-day

Output: data/processed/wisconsin_dot_county_day.parquet
"""
import sys, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger
from crash_coverage import write_manifest
from state_dot_sources import validate_wisconsin_county_year

warnings.filterwarnings("ignore")
log = get_logger("wisconsin_dot")

OUT_PATH = DATA_PROC / "wisconsin_dot_county_day.parquet"
DATA_PROC.mkdir(parents=True, exist_ok=True)

API_URL = (
    "https://transportal.cee.wisc.edu/partners/community-maps/crash/"
    "public/crashesKML.do"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}
YEARS   = list(range(2013, 2025))   # 2013–2024 inclusive

# ── Wisconsin county FIPS mapping ─────────────────────────────────────────────
# 72 counties; FIPS = "55" + odd-numbered sequence (001, 003, …, 141)
# API county parameter uses lowercase names
WI_COUNTIES = {
    "adams":       ("ADAMS",       "55001"),
    "ashland":     ("ASHLAND",     "55003"),
    "barron":      ("BARRON",      "55005"),
    "bayfield":    ("BAYFIELD",    "55007"),
    "brown":       ("BROWN",       "55009"),
    "buffalo":     ("BUFFALO",     "55011"),
    "burnett":     ("BURNETT",     "55013"),
    "calumet":     ("CALUMET",     "55015"),
    "chippewa":    ("CHIPPEWA",    "55017"),
    "clark":       ("CLARK",       "55019"),
    "columbia":    ("COLUMBIA",    "55021"),
    "crawford":    ("CRAWFORD",    "55023"),
    "dane":        ("DANE",        "55025"),
    "dodge":       ("DODGE",       "55027"),
    "door":        ("DOOR",        "55029"),
    "douglas":     ("DOUGLAS",     "55031"),
    "dunn":        ("DUNN",        "55033"),
    "eau claire":  ("EAU CLAIRE",  "55035"),
    "florence":    ("FLORENCE",    "55037"),
    "fond du lac": ("FOND DU LAC", "55039"),
    "forest":      ("FOREST",      "55041"),
    "grant":       ("GRANT",       "55043"),
    "green":       ("GREEN",       "55045"),
    "green lake":  ("GREEN LAKE",  "55047"),
    "iowa":        ("IOWA",        "55049"),
    "iron":        ("IRON",        "55051"),
    "jackson":     ("JACKSON",     "55053"),
    "jefferson":   ("JEFFERSON",   "55055"),
    "juneau":      ("JUNEAU",      "55057"),
    "kenosha":     ("KENOSHA",     "55059"),
    "kewaunee":    ("KEWAUNEE",    "55061"),
    "la crosse":   ("LA CROSSE",   "55063"),
    "lafayette":   ("LAFAYETTE",   "55065"),
    "langlade":    ("LANGLADE",    "55067"),
    "lincoln":     ("LINCOLN",     "55069"),
    "manitowoc":   ("MANITOWOC",   "55071"),
    "marathon":    ("MARATHON",    "55073"),
    "marinette":   ("MARINETTE",   "55075"),
    "marquette":   ("MARQUETTE",   "55077"),
    "menominee":   ("MENOMINEE",   "55078"),
    "milwaukee":   ("MILWAUKEE",   "55079"),
    "monroe":      ("MONROE",      "55081"),
    "oconto":      ("OCONTO",      "55083"),
    "oneida":      ("ONEIDA",      "55085"),
    "outagamie":   ("OUTAGAMIE",   "55087"),
    "ozaukee":     ("OZAUKEE",     "55089"),
    "pepin":       ("PEPIN",       "55091"),
    "pierce":      ("PIERCE",      "55093"),
    "polk":        ("POLK",        "55095"),
    "portage":     ("PORTAGE",     "55097"),
    "price":       ("PRICE",       "55099"),
    "racine":      ("RACINE",      "55101"),
    "richland":    ("RICHLAND",    "55103"),
    "rock":        ("ROCK",        "55105"),
    "rusk":        ("RUSK",        "55107"),
    "st. croix":   ("ST. CROIX",   "55109"),
    "sauk":        ("SAUK",        "55111"),
    "sawyer":      ("SAWYER",      "55113"),
    "shawano":     ("SHAWANO",     "55115"),
    "sheboygan":   ("SHEBOYGAN",   "55117"),
    "taylor":      ("TAYLOR",      "55119"),
    "trempealeau": ("TREMPEALEAU", "55121"),
    "vernon":      ("VERNON",      "55123"),
    "vilas":       ("VILAS",       "55125"),
    "walworth":    ("WALWORTH",    "55127"),
    "washburn":    ("WASHBURN",    "55129"),
    "washington":  ("WASHINGTON",  "55131"),
    "waukesha":    ("WAUKESHA",    "55133"),
    "waupaca":     ("WAUPACA",     "55135"),
    "waushara":    ("WAUSHARA",    "55137"),
    "winnebago":   ("WINNEBAGO",   "55139"),
    "wood":        ("WOOD",        "55141"),
}

# Reverse: API cnytname (uppercase) → FIPS
CNTY_NAME_TO_FIPS = {v[0]: v[1] for v in WI_COUNTIES.values()}


def fetch_county_year(
    session: requests.Session, county_param: str, year: int, retries: int = 3
) -> list[dict] | None:
    """
    Fetch all crashes for one (county, year) pair.
    Returns a list of property dicts or None on failure.
    """
    params = {
        "filetype":  "json",
        "county":    county_param,
        "startyear": str(year),
        "endyear":   str(year),
        "injsvr":    ["K", "A", "B", "C", "O"],  # all severity levels
    }
    for attempt in range(retries):
        try:
            resp = session.get(API_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            return [f["properties"] for f in features]
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("  [%s %d] attempt %d failed (%s); retrying in %ds",
                        county_param, year, attempt + 1, exc, wait)
            time.sleep(wait)
    log.error("  [%s %d] all %d attempts failed", county_param, year, retries)
    return None


# ── Main download loop ────────────────────────────────────────────────────────
log.info("Downloading Wisconsin crash data (2013–2024) via Community Maps API …")
log.info("Counties: %d  Years: %s–%s", len(WI_COUNTIES), YEARS[0], YEARS[-1])

session = requests.Session()
session.headers.update(HEADERS)

all_parts: list[pd.DataFrame] = []
coverage_rows = []
total_requests = len(WI_COUNTIES) * len(YEARS)
done = 0

for county_param, (county_upper, fips) in WI_COUNTIES.items():
    county_rows: list[dict] = []

    for yr in YEARS:
        done += 1
        props = fetch_county_year(session, county_param, yr)

        if props is None:
            log.warning("  [%s %d] skipped (all retries failed)", county_param, yr)
            coverage_rows.append(validate_wisconsin_county_year(
                fips, yr, response_kind="failed", terminal_error="request_failed",
            ))
            time.sleep(1.0)
            continue

        if not props:
            # Zero crashes this county-year is plausible for small/rural counties
            log.debug("  [%s %d] 0 crashes", county_param, yr)
            coverage_rows.append(validate_wisconsin_county_year(
                fips, yr, response_kind="empty", request_complete=True,
            ))
            time.sleep(0.3)
            continue

        raw_request = pd.DataFrame(props)
        raw_dates = raw_request["date"] if "date" in raw_request else pd.Series(pd.NaT, index=raw_request.index)
        request_dates = pd.to_datetime(raw_dates, format="%m/%d/%Y", errors="coerce")
        wrong_year = int(request_dates.notna().sum() - request_dates.dt.year.eq(yr).sum())
        coverage_rows.append(validate_wisconsin_county_year(
            fips, yr, response_kind="success", expected_records=len(raw_request),
            fetched_records=len(raw_request), retained_records=len(raw_request),
            request_complete=True,
            required_columns_ok={"date", "totfatl", "injsvr"}.issubset(raw_request.columns),
            invalid_date_count=int(request_dates.isna().sum()) + wrong_year,
            observed_min_date=request_dates.min(), observed_max_date=request_dates.max(),
        ))

        county_rows.extend(props)
        log.debug("  [%s %d] %d crashes", county_param, yr, len(props))
        time.sleep(0.4)   # polite delay

    if not county_rows:
        log.warning("[%s] no data across all years — skipping", county_param)
        continue

    df = pd.DataFrame(county_rows)

    # ── Parse date ──────────────────────────────────────────────────────────
    df["crash_date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["crash_date"])
    df["crash_date"] = df["crash_date"].dt.normalize()

    # ── Severity ────────────────────────────────────────────────────────────
    df["totfatl"] = pd.to_numeric(df.get("totfatl", 0), errors="coerce").fillna(0)
    df["totinj"]  = pd.to_numeric(df.get("totinj",  0), errors="coerce").fillna(0)
    df["injsvr"]  = df.get("injsvr", pd.Series("O", index=df.index)).astype(str)

    # The API exposes total injuries for an A-severity crash, not a verified
    # count of seriously injured people.  Retain it under an honest proxy name;
    # the comparable serious-injury outcome remains unavailable.
    df["injury_proxy"] = df["totinj"].where(df["injsvr"] == "A", 0)

    # ── Aggregate to county-day ──────────────────────────────────────────────
    agg = (
        df.groupby("crash_date")
          .agg(
              wi_fatals     =("totfatl",    "sum"),
              wi_injury_proxy=("injury_proxy", "sum"),
              wi_crashes    =("totfatl",    "count"),
          )
          .reset_index()
          .rename(columns={"crash_date": "date"})
    )
    agg["fips"] = fips
    all_parts.append(agg)

    log.info("[%d/%d] %-15s → %d county-days  fatals=%.0f  serious=%.0f",
             done // len(YEARS), len(WI_COUNTIES), county_param,
             len(agg), agg["wi_fatals"].sum(), agg["wi_injury_proxy"].sum())

    del df, agg, county_rows
    gc.collect()

session.close()

# One explicit manifest row is written for every 72 x 12 county-year request.
write_manifest(coverage_rows, DATA_PROC / "coverage", filename="wisconsin_coverage")
if any(not row.coverage_valid for row in coverage_rows):
    raise RuntimeError("Wisconsin coverage validation failed; sparse output is not valid for balancing")

# ── Combine ───────────────────────────────────────────────────────────────────
if not all_parts:
    log.error("No Wisconsin data collected.")
    sys.exit(1)

wi_panel = pd.concat(all_parts, ignore_index=True)
wi_panel["date"] = pd.to_datetime(wi_panel["date"])

# De-duplicate in case of overlap
wi_panel = (
    wi_panel.groupby(["fips", "date"])
      .agg(
          wi_fatals     =("wi_fatals",      "sum"),
          wi_injury_proxy=("wi_injury_proxy", "sum"),
          wi_crashes    =("wi_crashes",     "sum"),
      )
      .reset_index()
)

log.info("\nFinal Wisconsin panel:")
log.info("  Rows: %d  Counties: %d  Date range: %s – %s",
         len(wi_panel), wi_panel["fips"].nunique(),
         wi_panel["date"].min().date(), wi_panel["date"].max().date())
wi_panel["wi_serious_inj"] = np.nan
log.info("  Total wi_fatals: %.0f  Total wi_injury_proxy: %.0f",
         wi_panel["wi_fatals"].sum(), wi_panel["wi_injury_proxy"].sum())

wi_panel.to_parquet(OUT_PATH, index=False)
log.info("Saved → %s", OUT_PATH)
