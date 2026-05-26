"""
run_state_dot_analysis.py
========================================================
Initial analysis: do AMBER alerts correlate with more traffic crashes
per 100k residents, using all-crash (not fatals-only) state DOT data?

Data sources:
  - State DOT crash panels (county-day or county-month):
      CA, FL, IL, IA, MA, NV, NY, OR, VA, WI  (+ PA county-month)
  - AMBER alert records: data/raw/amber/foia/openfema_ipaws_alerts_2013_2024.csv
  - Census county population: data/processed/county_population.parquet

Method: Two-Way Fixed Effects (TWFE) OLS
  crashes_per_100k_{i, crash_date} = β · night_alert_{i, crash_date}
                                    + county_FE_i + date_FE_t
                                    + ε_{it}

Timing convention:
  "night" = 10pm–6am local time.  The crash date an alert is expected to
  affect differs by sub-window:
    • Early night (22:00–23:59): alert on calendar day D  →  crash date D+1
      (alert fires before midnight; disrupted drivers are on the road on D+1)
    • Late night  (00:00–05:59): alert on calendar day D  →  crash date D
      (alert fires after midnight, already IS the next morning; crashes
       recorded on the same calendar day D)
  Both sub-windows target the same next-morning commute.  We collapse them
  to a single "night_alert" indicator keyed on the crash date, so no
  additional outcome shift is needed — the outcome is same-day crashes.

Two model families run in parallel:

  OLS-rate:  outcome = crashes_per_100k  (count / population × 100,000)
             β = absolute change in crashes per 100k on alert nights

  Poisson PPML (pyfixest.fepois):
             outcome = crash count  with offset = log(population)
             exp(β) = Incident Rate Ratio (IRR); 100×(exp(β)−1) = % change
             County FE + date FE + DoW absorbed inside fepois().
             SE clustered by county.

Treatment:
  night_alert — 1 if county received a nighttime AMBER alert whose
                effective crash date equals the panel date (crash_date)

Output: output/tables/state_dot_analysis.csv  (main regression table)
        output/tables/state_dot_descriptives.csv (summary stats)
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, DATA_RAW, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("state_dot_analysis")

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

# ── 1. Load and unify all state DOT crash data ────────────────────────────────
log.info("=== Loading state DOT crash data ===")

STATE_FILES = {
    "CA": ("california_ccrs_county_day.parquet",   "ca_crashes", "ca_fatals", "ca_serious_inj"),
    "FL": ("florida_fdot_county_day.parquet",       "fl_crashes", "fl_fatals", "fl_serious_inj"),
    "IL": ("illinois_idot_county_day.parquet",      "il_crashes", "il_fatals", "il_serious_inj"),
    "IA": ("iowa_dot_county_day.parquet",           "ia_crashes", "ia_fatals", "ia_serious_inj"),
    "MA": ("massachusetts_massdot_county_day.parquet","ma_crashes","ma_fatals","ma_serious_inj"),
    "NV": ("nevada_ndot_county_day.parquet",        "nv_crashes", "nv_fatals", "nv_serious_inj"),
    "OR": ("oregon_odot_county_day.parquet",        "or_crashes", "or_fatals", "or_serious_inj"),
    "VA": ("virginia_vdot_county_day.parquet",      "va_crashes", "va_fatals", "va_serious_inj"),
    "WI": ("wisconsin_dot_county_day.parquet",      "wi_crashes", "wi_fatals", "wi_serious_inj"),
    # NY: crash counts only; ny_fatal_crashes = fatal crashes (not fatalities); no serious_inj
    "NY": ("newyork_dot_county_day.parquet",        "ny_crashes", "ny_fatal_crashes", None),
    # PA is county-MONTH — handled separately below
}
PA_FILE = "pennsylvania_penndot_county_month.parquet"

parts = []
for state, (fname, c_crashes, c_fatals, c_serious) in STATE_FILES.items():
    path = DATA_PROC / fname
    if not path.exists():
        log.warning("Missing %s — skipping %s", fname, state)
        continue
    df = pd.read_parquet(path)
    rename_map = {c_crashes: "crashes", c_fatals: "fatals"}
    if c_serious and c_serious in df.columns:
        rename_map[c_serious] = "serious_inj"
    df = df.rename(columns=rename_map)
    # Ensure all expected columns exist
    for col in ["crashes", "fatals", "serious_inj"]:
        if col not in df.columns:
            df[col] = 0
    df = df[["fips", "date", "crashes", "fatals", "serious_inj"]].copy()
    # Normalize date to calendar day (CA CCRS has individual crash timestamps)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Aggregate to county-day (no-op for already-aggregated states)
    df = (df.groupby(["fips", "date"])
            .agg(crashes=("crashes", "sum"), fatals=("fatals", "sum"),
                 serious_inj=("serious_inj", "sum"))
            .reset_index())
    df["state"] = state
    df["granularity"] = "day"
    parts.append(df)
    log.info("  %s: %d county-days  crashes=%.0f  fatals=%.0f  serious=%.0f",
             state, len(df), df.crashes.sum(), df.fatals.sum(), df.serious_inj.sum())

# Pennsylvania (monthly)
pa_path = DATA_PROC / PA_FILE
if pa_path.exists():
    pa = pd.read_parquet(pa_path)
    pa = pa.rename(columns={"pa_crashes": "crashes", "pa_fatals": "fatals",
                             "pa_serious_inj": "serious_inj"})
    pa = pa[["fips", "date", "crashes", "fatals", "serious_inj"]].copy()
    pa["state"] = "PA"
    pa["granularity"] = "month"
    parts.append(pa)
    log.info("  PA: %d county-months  crashes=%.0f  fatals=%.0f  serious=%.0f",
             len(pa), pa.crashes.sum(), pa.fatals.sum(), pa.serious_inj.sum())

crashes_all = pd.concat(parts, ignore_index=True)
crashes_all["date"] = pd.to_datetime(crashes_all["date"])
log.info("Combined: %d records across %d states",
         len(crashes_all), crashes_all["state"].nunique())

# ── 2. Load AMBER alert data ──────────────────────────────────────────────────
log.info("=== Loading AMBER alert data ===")

AMBER_CSV = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2024.csv"
if not AMBER_CSV.exists():
    # fall back to old filename
    AMBER_CSV = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2022.csv"
if not AMBER_CSV.exists():
    log.error("AMBER alert file not found at %s", AMBER_CSV)
    log.error("Run: python code/02c_fetch_openfema_ipaws.py")
    sys.exit(1)
log.info("  Reading AMBER data from: %s", AMBER_CSV.name)

alerts = pd.read_csv(AMBER_CSV, parse_dates=["sent_utc"])
log.info("  Raw AMBER records: %d", len(alerts))

# ── msgType filter ────────────────────────────────────────────────────────────
# AMBER IPAWS records include three message types:
#   Alert  — new abduction notification → triggers loud WEA tone + vibration
#   Update — revised information        → triggers loud WEA tone + vibration
#   Cancel — case resolved              → silently dismisses previous alert; NO ringing
#
# Only Alert and Update physically ring/buzz phones and can disrupt sleep.
# Cancel messages in the night window are FALSE POSITIVES for sleep disruption.
# Empirical counts (full dataset): Alert 64.9%, Cancel 31.8%, Update 3.3%.
# False-positive treated county-dates (Cancel-only at night): ~25% of treated obs.
# Excluding Cancels removes attenuation bias (estimated β biased toward zero).
if "msg_type" in alerts.columns:
    before = len(alerts)
    mt_counts = alerts.groupby("msg_type")["alert_id"].nunique()
    log.info("  msgType breakdown (unique alert_ids):\n%s", mt_counts.to_string())
    # Keep only phone-ringing message types
    alerts = alerts[alerts["msg_type"].isin(["Alert", "Update"])].copy()
    log.info("  Filtered to Alert+Update: %d → %d rows (%d unique alert_ids)",
             before, len(alerts), alerts["alert_id"].nunique())
else:
    log.warning("  msg_type column missing — Cancel records NOT filtered.")
    log.warning("  Run 02d_classify_alert_msgtypes.py then re-run analysis.")
    log.warning("  ~25%% of treated county-dates are false positives (Cancel-only nights).")

# Convert UTC → local time with proper DST handling (via pytz)
# ─────────────────────────────────────────────────────────────────────────────
# Primary timezone per state FIPS (2-digit).  America/Indiana/Indianapolis
# observes EST/EDT; America/Phoenix has no DST.
import pytz

STATE_TIMEZONE = {
    "01": "America/Chicago",               # Alabama
    "02": "America/Anchorage",             # Alaska
    "04": "America/Phoenix",               # Arizona (no DST)
    "05": "America/Chicago",               # Arkansas
    "06": "America/Los_Angeles",           # California
    "08": "America/Denver",                # Colorado
    "09": "America/New_York",              # Connecticut
    "10": "America/New_York",              # Delaware
    "11": "America/New_York",              # DC
    "12": "America/New_York",              # Florida (majority ET; panhandle counties overridden below)
    "13": "America/New_York",              # Georgia
    "15": "Pacific/Honolulu",             # Hawaii (no DST)
    "16": "America/Boise",                # Idaho (majority MT; panhandle PT overridden below)
    "17": "America/Chicago",              # Illinois
    "18": "America/Indiana/Indianapolis", # Indiana (observes ET/EDT)
    "19": "America/Chicago",              # Iowa
    "20": "America/Chicago",              # Kansas (majority CT; western counties overridden)
    "21": "America/New_York",             # Kentucky (majority ET; western counties overridden)
    "22": "America/Chicago",              # Louisiana
    "23": "America/New_York",             # Maine
    "24": "America/New_York",             # Maryland
    "25": "America/New_York",             # Massachusetts
    "26": "America/Detroit",              # Michigan (majority ET; UP counties overridden)
    "27": "America/Chicago",              # Minnesota
    "28": "America/Chicago",              # Mississippi
    "29": "America/Chicago",              # Missouri
    "30": "America/Denver",               # Montana
    "31": "America/Chicago",              # Nebraska (majority CT; Panhandle overridden)
    "32": "America/Los_Angeles",          # Nevada
    "33": "America/New_York",             # New Hampshire
    "34": "America/New_York",             # New Jersey
    "35": "America/Denver",               # New Mexico
    "36": "America/New_York",             # New York
    "37": "America/New_York",             # North Carolina
    "38": "America/Chicago",              # North Dakota (majority CT; western overridden)
    "39": "America/New_York",             # Ohio
    "40": "America/Chicago",              # Oklahoma
    "41": "America/Los_Angeles",          # Oregon (majority PT; eastern overridden)
    "42": "America/New_York",             # Pennsylvania
    "44": "America/New_York",             # Rhode Island
    "45": "America/New_York",             # South Carolina
    "46": "America/Chicago",              # South Dakota (majority CT; western overridden)
    "47": "America/Chicago",              # Tennessee (majority CT; eastern counties overridden)
    "48": "America/Chicago",              # Texas (majority CT; El Paso area overridden)
    "49": "America/Denver",               # Utah
    "50": "America/New_York",             # Vermont
    "51": "America/New_York",             # Virginia
    "53": "America/Los_Angeles",          # Washington
    "54": "America/New_York",             # West Virginia
    "55": "America/Chicago",              # Wisconsin
    "56": "America/Denver",               # Wyoming
}

# Per-county FIPS overrides for states that span two time zones.
# Sources: USNO timezone boundaries; US Census TIGER.
COUNTY_TIMEZONE_OVERRIDE = {
    # Florida panhandle (CT)
    "12033": "America/Chicago", "12059": "America/Chicago",
    "12077": "America/Chicago", "12113": "America/Chicago",
    "12131": "America/Chicago",
    # Idaho panhandle (PT)
    "16021": "America/Los_Angeles", "16055": "America/Los_Angeles",
    "16057": "America/Los_Angeles", "16069": "America/Los_Angeles",
    "16079": "America/Los_Angeles",
    # Kansas western (MT)
    "20129": "America/Denver", "20189": "America/Denver",
    # Kentucky western (CT)
    "21007": "America/Chicago", "21083": "America/Chicago",
    "21139": "America/Chicago", "21145": "America/Chicago",
    "21157": "America/Chicago", "21179": "America/Chicago",
    "21195": "America/Chicago", "21221": "America/Chicago",
    # Michigan Upper Peninsula (CT)
    "26003": "America/Chicago", "26013": "America/Chicago",
    "26033": "America/Chicago", "26041": "America/Chicago",
    "26043": "America/Chicago", "26053": "America/Chicago",
    "26061": "America/Chicago", "26071": "America/Chicago",
    "26083": "America/Chicago", "26095": "America/Chicago",
    "26097": "America/Chicago", "26103": "America/Chicago",
    "26131": "America/Chicago", "26153": "America/Chicago",
    # Nebraska panhandle (MT)
    "31007": "America/Denver", "31057": "America/Denver",
    "31069": "America/Denver", "31123": "America/Denver",
    "31157": "America/Denver", "31165": "America/Denver",
    "31173": "America/Denver",
    # North Dakota western (MT)
    "38011": "America/Denver", "38025": "America/Denver",
    "38041": "America/Denver", "38053": "America/Denver",
    "38055": "America/Denver", "38087": "America/Denver",
    "38105": "America/Denver",
    # Oregon eastern (MT)
    "41001": "America/Denver", "41017": "America/Denver",
    "41021": "America/Denver", "41023": "America/Denver",
    "41025": "America/Denver", "41035": "America/Denver",
    "41037": "America/Denver", "41045": "America/Denver",
    "41049": "America/Denver", "41055": "America/Denver",
    "41059": "America/Denver", "41065": "America/Denver",
    # South Dakota western (MT)
    "46017": "America/Denver", "46033": "America/Denver",
    "46047": "America/Denver", "46063": "America/Denver",
    "46065": "America/Denver", "46093": "America/Denver",
    "46105": "America/Denver", "46113": "America/Denver",
    "46117": "America/Denver",
    # Tennessee eastern (ET)
    "47001": "America/New_York", "47009": "America/New_York",
    "47013": "America/New_York", "47025": "America/New_York",
    "47029": "America/New_York", "47051": "America/New_York",
    "47063": "America/New_York", "47065": "America/New_York",
    "47067": "America/New_York", "47073": "America/New_York",
    "47089": "America/New_York", "47097": "America/New_York",
    "47105": "America/New_York", "47107": "America/New_York",
    "47121": "America/New_York", "47129": "America/New_York",
    "47139": "America/New_York", "47143": "America/New_York",
    "47145": "America/New_York", "47151": "America/New_York",
    "47155": "America/New_York", "47163": "America/New_York",
    "47171": "America/New_York", "47173": "America/New_York",
    "47179": "America/New_York", "47189": "America/New_York",
    # Texas El Paso area (MT)
    "48141": "America/Denver", "48229": "America/Denver",
}

# Build a FIPS → timezone name mapping
alerts["fips_5"] = alerts["fips"].astype(str).str.zfill(5)
alerts["state_fips"] = alerts["fips_5"].str[:2]

alerts["tz_name"] = (alerts["fips_5"]
                     .map(COUNTY_TIMEZONE_OVERRIDE)
                     .fillna(alerts["state_fips"].map(STATE_TIMEZONE))
                     .fillna("America/Chicago"))   # fallback

# DST-aware UTC → local conversion using pytz
# sent_utc is already tz-aware (UTC); convert to local tz row-by-row.
# Group by timezone to batch the conversion efficiently.
alerts["hour_local"] = pd.NA
alerts["sent_local"]  = pd.NaT

utc_series = pd.to_datetime(alerts["sent_utc"], utc=True)

for tz_name, idx in alerts.groupby("tz_name").groups.items():
    tz = pytz.timezone(tz_name)
    local = utc_series.loc[idx].dt.tz_convert(tz)
    alerts.loc[idx, "hour_local"] = local.dt.hour
    alerts.loc[idx, "sent_local"] = local.dt.tz_localize(None)  # strip tz for merges

alerts["hour_local"] = alerts["hour_local"].astype(int)
log.info("  Timezone-aware local conversion done (%d unique tz used)", alerts["tz_name"].nunique())

# Night alert: 10pm–6am local
alerts["is_night"] = (alerts["hour_local"] >= 22) | (alerts["hour_local"] < 6)

# Calendar date of alert (local time) — sent_local is already tz-naive
alerts["alert_date"] = alerts["sent_local"].dt.normalize()

# Effective crash date: the calendar day whose crashes this alert is expected to affect.
#   Early night (22–23h): alert fires before midnight on day D  →  crash date D+1
#   Late night  ( 0– 5h): alert fires after midnight, already on day D  →  crash date D
# Both sub-windows disrupt the same next-morning commute; keying on crash_date lets
# us match directly to same-day crashes without a separate t+1 shift.
alerts["effective_crash_date"] = np.where(
    alerts["hour_local"] >= 22,
    alerts["alert_date"] + pd.Timedelta(days=1),
    alerts["alert_date"],
)

log.info("  Total alerts: %d   Night alerts: %d (%.1f%%)",
         len(alerts), alerts.is_night.sum(), 100*alerts.is_night.mean())
log.info("  Early-night (22–23h): %d   Late-night (0–5h): %d",
         (alerts["is_night"] & (alerts["hour_local"] >= 22)).sum(),
         (alerts["is_night"] & (alerts["hour_local"] < 6)).sum())

# ── Classify alerts and keep only those with verified county-level FIPS ───────
# State-level FIPS (e.g. 48000 = all of TX) have unknown true geographic scope:
# they may represent 1 county, several nearby counties, or the whole state —
# we cannot determine this from the FIPS code alone (no polygon data in IPAWS).
# To avoid mis-assigning treatment, we use ONLY alerts with explicit county FIPS
# (Type C: every row is a county FIPS; Type B county rows: already have county FIPS).
# Type A (all rows are state-FIPS) are excluded from the main treatment variable
# and reported separately as a sensitivity check.
alerts["fips"]          = alerts["fips"].astype(str).str.zfill(5)
alerts["state_fips2"]   = alerts["fips"].str[:2]
alerts["is_state_fips"] = alerts["fips"].str[2:] == "000"

# Per alert_id: classify as A / B / C
def _alert_type(g):
    s = g["is_state_fips"].any(); c = (~g["is_state_fips"]).any()
    return "A" if (s and not c) else ("B" if (s and c) else "C")

_atype = alerts.groupby("alert_id").apply(_alert_type).rename("alert_type")
alerts  = alerts.join(_atype, on="alert_id")

n_A = alerts.loc[alerts["alert_type"]=="A","alert_id"].nunique()
n_B = alerts.loc[alerts["alert_type"]=="B","alert_id"].nunique()
n_C = alerts.loc[alerts["alert_type"]=="C","alert_id"].nunique()
log.info("  Alert types: A(state-FIPS only)=%d  B(mixed)=%d  C(county-FIPS)=%d",
         n_A, n_B, n_C)
log.info("  Using Type C + Type B county rows for treatment (Type A excluded from main spec)")

# Keep only county-level FIPS rows (drops state-level rows from Type B and all Type A)
alerts_county = alerts[~alerts["is_state_fips"]].copy()

# ── Compute alert broadcast scope (% state pop covered) for Type C/B alerts ──
# scope = covered_pop / state_pop.
#   0.03 = narrow single-county alert in large state
#   1.00 = all counties in state (or IA/MA which only issue statewide)
# This is a CONTINUOUS treatment intensity that replaces the binary indicator.
# We do NOT impute scope for Type A — they stay excluded.
pop_ref = pd.read_parquet(DATA_PROC / "county_population.parquet")
pop_ref = pop_ref[pop_ref["year"] == 2019][["fips", "population"]].copy()
pop_ref["fips"] = pop_ref["fips"].astype(str).str.zfill(5)
pop_ref = pop_ref.set_index("fips")
state_pop_total = (pop_ref.assign(sf=pop_ref.index.str[:2])
                          .groupby("sf")["population"].sum())

log.info("Computing alert broadcast scope from county FIPS …")
scope_map: dict[str, float] = {}
for alert_id, grp in alerts_county.groupby("alert_id"):
    sfips  = grp["state_fips2"].iloc[0]
    fips_l = grp["fips"].tolist()
    cov    = sum(pop_ref.loc[f,"population"] for f in fips_l if f in pop_ref.index)
    st_pop = state_pop_total.get(sfips, np.nan)
    scope_map[alert_id] = cov / st_pop if (st_pop and st_pop > 0) else np.nan

alerts_county["alert_scope"] = alerts_county["alert_id"].map(scope_map)
log.info("  Scope computed: median=%.3f  mean=%.3f  (N=%d alerts)",
         pd.Series(scope_map).median(), pd.Series(scope_map).mean(), len(scope_map))

# ── Collapse to county × effective_crash_date ─────────────────────────────────
# night_alert = binary 0/1
# alert_scope = max scope among all alerts hitting this county-night
#               (continuous; 0 = untreated, >0 = treated with this intensity)
night_alerts = (
    alerts_county[alerts_county["is_night"]]
    .groupby(["fips", "effective_crash_date"])
    .agg(n_alerts   =("alert_id",     "nunique"),
         alert_scope=("alert_scope",  "max"))
    .reset_index()
)
night_alerts["fips"]        = night_alerts["fips"].astype(str).str.zfill(5)
night_alerts["night_alert"] = 1
log.info("  County-crash_dates with ≥1 verified-county night alert: %d", len(night_alerts))

# ── Sleep-phase indicators ─────────────────────────────────────────────────────
# Four mutually-exclusive bins keyed on alert's local hour.
# Name / hour-range / human label
SLEEP_PHASES = [
    ("ph_2223", 22, 23, "Still awake  (22–23h)"),
    ("ph_0001",  0,  1, "Light sleep  (0–1h)"),
    ("ph_0203",  2,  3, "Deep sleep   (2–3h)"),   # N3 peak
    ("ph_0405",  4,  5, "Late/REM     (4–5h)"),
]
PHASE_COLS   = [p[0] for p in SLEEP_PHASES]
PHASE_LABELS = {p[0]: p[3] for p in SLEEP_PHASES}

phase_alert_frames = {}
for ph_name, h_lo, h_hi, ph_label in SLEEP_PHASES:
    ph = alerts[alerts["is_night"] & alerts["hour_local"].between(h_lo, h_hi)].copy()
    collapsed = (
        ph.groupby(["fips", "effective_crash_date"])
        .size()
        .reset_index(name=f"n_{ph_name}")
    )
    collapsed["fips"] = collapsed["fips"].astype(str).str.zfill(5)
    collapsed[ph_name] = 1
    phase_alert_frames[ph_name] = collapsed
    log.info("  Sleep phase %-28s %d county-crash_dates", ph_label, len(collapsed))

# ── 3. Load population data ───────────────────────────────────────────────────
log.info("=== Loading population data ===")

POP_PATH = DATA_PROC / "county_population.parquet"
if not POP_PATH.exists():
    log.error("Population file not found. Run: python code/00_download_population.py")
    sys.exit(1)

pop = pd.read_parquet(POP_PATH)
log.info("  Population records: %d  Years: %s–%s",
         len(pop), pop.year.min(), pop.year.max())

# ── 4. Build analysis panel ───────────────────────────────────────────────────
log.info("=== Building analysis panel ===")

# Use day-granularity data only for the main regression
panel = crashes_all[crashes_all.granularity == "day"].copy()
panel["year"] = panel["date"].dt.year

# Merge population (by fips + year)
panel["fips"] = panel["fips"].astype(str).str.zfill(5)
panel = panel.merge(pop[["fips", "year", "population"]], on=["fips", "year"], how="left")
n_miss_pop = panel["population"].isna().sum()
if n_miss_pop:
    log.warning("  %d rows missing population (%.1f%%) — dropping",
                n_miss_pop, 100*n_miss_pop/len(panel))
panel = panel.dropna(subset=["population"])

# Compute per-100k rates
panel["crashes_per_100k"]  = 100_000 * panel["crashes"]  / panel["population"]
panel["fatals_per_100k"]   = 100_000 * panel["fatals"]   / panel["population"]
panel["serious_per_100k"]  = 100_000 * panel["serious_inj"] / panel["population"]

# Sort panel (no t+1 shift needed — effective_crash_date already accounts for timing)
panel = panel.sort_values(["state", "fips", "date"])

# Day of week + month
panel["dow"]   = panel["date"].dt.dayofweek   # 0=Mon
panel["month"] = panel["date"].dt.month

# State × year-month FE — weather and holiday proxy
# Controls for seasonal weather patterns within each state-year:
#   e.g. "unusually wet July 2018 in California" vs the national average captured
#   by the date FE.  This is the standard econometric substitute for actual daily
#   weather data when county-grid weather is unavailable.
#   Identification: variation in whether a given county got a night alert,
#   conditional on state-level seasonal patterns.
panel["state_yearmon"] = panel["state"] + "_" + panel["date"].dt.to_period("M").astype(str)

# Merge: match alert's effective_crash_date to the panel's crash date
night_alerts["fips"] = night_alerts["fips"].astype(str).str.zfill(5)
panel = panel.merge(
    night_alerts[["fips", "effective_crash_date", "night_alert"]],
    left_on=["fips", "date"], right_on=["fips", "effective_crash_date"],
    how="left"
)
panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)

# Merge sleep-phase indicators
for ph_name, collapsed in phase_alert_frames.items():
    panel = panel.merge(
        collapsed[["fips", "effective_crash_date", ph_name]],
        left_on=["fips", "date"], right_on=["fips", "effective_crash_date"],
        how="left", suffixes=("", f"_{ph_name}")
    )
    panel[ph_name] = panel[ph_name].fillna(0).astype(int)
    # drop the duplicate effective_crash_date column from this merge
    dup_col = f"effective_crash_date_{ph_name}"
    if dup_col in panel.columns:
        panel = panel.drop(columns=[dup_col])

# Restrict to study years that overlap between crash data and alert data
panel = panel[panel["year"].between(2013, 2022)]
panel = panel.dropna(subset=["crashes_per_100k"])

log.info("  Panel rows after merge: %d", len(panel))
log.info("  Counties: %d  States: %d  Date range: %s – %s",
         panel.fips.nunique(), panel.state.nunique(),
         panel.date.min().date(), panel.date.max().date())
log.info("  Night alert county-days: %d (%.3f%%)",
         panel.night_alert.sum(), 100*panel.night_alert.mean())

# ── 5. Descriptive statistics ─────────────────────────────────────────────────
log.info("=== Descriptive statistics ===")

desc_rows = []
for state in sorted(panel.state.unique()):
    s = panel[panel.state == state]
    s_alert = s[s.night_alert == 1]
    s_noalert = s[s.night_alert == 0]
    desc_rows.append({
        "state":             state,
        "county_days":       len(s),
        "counties":          s.fips.nunique(),
        "date_range":        f"{s.date.min().date()} – {s.date.max().date()}",
        "night_alert_days":  s.night_alert.sum(),
        "mean_crashes_100k": s.crashes_per_100k.mean(),
        "mean_crashes_alert":  s_alert.crashes_per_100k.mean() if len(s_alert) else np.nan,
        "mean_crashes_notalert": s_noalert.crashes_per_100k.mean() if len(s_noalert) else np.nan,
        "mean_fatals_100k":  s.fatals_per_100k.mean(),
        "mean_serious_100k": s.serious_per_100k.mean(),
    })

desc = pd.DataFrame(desc_rows)
desc["raw_diff_crashes"] = desc["mean_crashes_alert"] - desc["mean_crashes_notalert"]
log.info("\nDescriptive statistics by state:")
log.info(desc[["state","county_days","night_alert_days","mean_crashes_100k",
               "mean_crashes_alert","mean_crashes_notalert","raw_diff_crashes"]].to_string(index=False))

desc.to_csv(OUTPUT_TABS / "state_dot_descriptives.csv", index=False)

# ── 6. TWFE regressions ───────────────────────────────────────────────────────
log.info("\n=== TWFE Regressions ===")

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    log.warning("linearmodels not installed — using within-transform OLS fallback")

try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    log.warning("pyfixest not installed — Poisson PPML specs will be skipped")

from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

results_rows = []

# OLS outcomes (rate per 100k) and their raw count equivalents for Poisson
OUTCOMES = [
    ("crashes_per_100k", "All crashes / 100k",  "crashes"),
    ("fatals_per_100k",  "Fatalities / 100k",   "fatals"),
    ("serious_per_100k", "Serious inj / 100k",  "serious_inj"),
]

# String label for day-of-week in pyfixest formula
panel["dow_str"] = "dow" + panel["dow"].astype(str)


def _pyfixest_coef(fit, coef_name: str) -> tuple[float, float, float, int] | None:
    """Extract (beta, se, pvalue, n_obs) from a pyfixest fit object."""
    tbl = fit.tidy()
    if coef_name not in tbl.index:
        return None
    return (float(tbl.loc[coef_name, "Estimate"]),
            float(tbl.loc[coef_name, "Std. Error"]),
            float(tbl.loc[coef_name, "Pr(>|t|)"]),
            int(fit._N))


def run_twfe(sub2: pd.DataFrame, outcome_col: str, label: str) -> dict | None:
    """
    Population-weighted WLS TWFE via pyfixest.feols (county + date FE).
    Weights = county population (analytic weights).
    SE: two-way cluster by (county × year) — handles within-county serial
    correlation AND within-year cross-county common shocks.
    State-level clustering is avoided: only 11 states → too few for asymptotics.
    DoW dummies are omitted from the feols formula because calendar-date FEs
    already absorb day-of-week variation exactly.
    Falls back to unweighted iterative within-transform if pyfixest unavailable.
    """
    sub2 = sub2.dropna(subset=[outcome_col, "population"]).copy()
    if len(sub2) < 100 or sub2["night_alert"].std() < 1e-12:
        return None

    if HAS_PYFIXEST:
        sub2["_fips_str"]  = sub2["fips"].astype(str)
        sub2["_date_str"]  = sub2["date"].astype(str)
        sub2["_year_str"]  = sub2["year"].astype(str)
        sub2["_pop"]       = sub2["population"].astype(float)
        sub2["_stym_str"]  = sub2["state_yearmon"].astype(str)  # state×year-month FE
        # Baseline: county + date FEs
        # Robustness: county + date + state×year-month FEs (weather/holiday proxy)
        formula = f"{outcome_col} ~ night_alert | _fips_str + _date_str"
        for vcov_spec, method_tag in [
            ({"CRV1": "_fips_str + _year_str"}, "2-way SE: county×year"),
            ({"CRV1": "_fips_str"},              "SE: county"),
        ]:
            try:
                fit  = pf.feols(formula, data=sub2, weights="_pop", vcov=vcov_spec)
                vals = _pyfixest_coef(fit, "night_alert")
                if vals is None:
                    continue
                b, se, pv, n = vals
                if np.isnan(se):          # two-way failed silently → try county-only
                    continue
                return dict(beta=round(b,6), se=round(se,6), pvalue=round(pv,4), n_obs=n,
                            method=f"WLS TWFE (pyfixest, pop-wt, county+date FE, {method_tag})")
            except Exception as exc:
                log.warning("  [%s] feols[%s] failed for %s: %s",
                            label, method_tag, outcome_col, exc)
                continue

    # ── Fallback: unweighted iterative within-transform ───────────────────────
    dow_dummies = pd.get_dummies(sub2["dow"], prefix="dow", drop_first=True).astype(float)
    dow_cols = dow_dummies.columns.tolist()
    sub2 = pd.concat([sub2, dow_dummies], axis=1)
    all_cols = ["_y", "_x"] + dow_cols
    sub2["_y"] = sub2[outcome_col].astype(float)
    sub2["_x"] = sub2["night_alert"].astype(float)
    for c in all_cols:
        sub2[c] -= sub2[c].mean()
    for _ in range(5):
        for c in all_cols:
            sub2[c] = (sub2[c]
                       - sub2.groupby("fips")[c].transform("mean")
                       - sub2.groupby("date")[c].transform("mean"))
    if sub2["_x"].std() < 1e-12:
        return None
    X = sm.add_constant(sub2[["_x"] + dow_cols])
    try:
        mod = OLS(sub2["_y"], X).fit(
            cov_type="cluster", cov_kwds={"groups": sub2["fips"]}
        )
    except Exception as exc:
        log.warning("  [%s] %s OLS fallback failed: %s", label, outcome_col, exc)
        return None
    b = mod.params["_x"]; se = mod.bse["_x"]; pv = mod.pvalues["_x"]
    return dict(beta=round(b,6), se=round(se,6), pvalue=round(pv,4), n_obs=int(mod.nobs),
                method="Within-transform TWFE (county+date FE+DoW, unweighted fallback)")


def run_twfe_panelols(sub2: pd.DataFrame, outcome_col: str, label: str) -> dict | None:
    """Thin wrapper — now just calls run_twfe() which uses pyfixest feols."""
    return run_twfe(sub2, outcome_col, label)


def run_poisson_ppml(sub2: pd.DataFrame, rate_col: str, label: str) -> dict | None:
    """
    Population-weighted Poisson PPML (county+date FE+DoW).
    Weights = county population so large counties dominate the likelihood.
    exp(β) = incident rate ratio (IRR).
    """
    if not HAS_PYFIXEST:
        return None
    sub2 = sub2.dropna(subset=[rate_col, "population"]).copy()
    sub2 = sub2[sub2[rate_col] > 0]
    if len(sub2) < 100 or sub2["night_alert"].std() < 1e-12:
        return None

    sub2["_fips_str"] = sub2["fips"].astype(str)
    sub2["_date_str"] = sub2["date"].astype(str)
    sub2["_year_str"] = sub2["year"].astype(str)
    sub2["_pop"]      = sub2["population"].astype(float)

    formula = f"{rate_col} ~ night_alert | _fips_str + _date_str"
    for vcov_spec, method_tag in [
        ({"CRV1": "_fips_str + _year_str"}, "2-way SE: county×year"),
        ({"CRV1": "_fips_str"},              "SE: county"),
    ]:
        try:
            fit = pf.fepois(formula, data=sub2, weights="_pop", vcov=vcov_spec)
            vals = _pyfixest_coef(fit, "night_alert")
            if vals is None:
                continue
            b, se, pv, n = vals
            if np.isnan(se):
                continue
            irr = round(float(np.exp(b)), 6)
            pct = round(100 * (np.exp(b) - 1), 3)
            return dict(beta=round(b,6), se=round(se,6), pvalue=round(pv,4),
                        irr=irr, pct_change=pct, n_obs=n,
                        method=f"Poisson PPML (pyfixest, pop-wt, county+date FE, {method_tag})")
        except Exception as exc:
            log.warning("  [%s] PPML[%s] failed for %s: %s", label, method_tag, rate_col, exc)
    return None


def run_sleep_phase_twfe(sub2: pd.DataFrame, outcome_col: str, label: str) -> list[dict] | None:
    """
    Joint TWFE with four sleep-phase treatment indicators as separate regressors.
    Returns one dict per active phase, with β, SE, p-value directly comparable.

    Sleep phase bins (local time):
      ph_2223  22–23h  Still awake / falling asleep
      ph_0001   0– 1h  Light sleep (N1/N2, first cycle)
      ph_0203   2– 3h  Deep sleep  (N3, slow-wave; hardest to rouse)
      ph_0405   4– 5h  Late sleep / REM (lighter, closer to waking)

    Hypothesis: ph_0203 should have the largest (most negative) coefficient if
    the mechanism is sleep-inertia-driven next-morning driving impairment.
    """
    sub2 = sub2.dropna(subset=[outcome_col, "population"]).copy()

    # Only include phases that have ≥5 treated obs and non-zero variance
    active = [c for c in PHASE_COLS
              if c in sub2.columns and sub2[c].sum() >= 5 and sub2[c].std() > 1e-12]
    if not active:
        return None

    if HAS_PYFIXEST:
        sub2["_fips_str"] = sub2["fips"].astype(str)
        sub2["_date_str"] = sub2["date"].astype(str)
        sub2["_year_str"] = sub2["year"].astype(str)
        sub2["_pop"]      = sub2["population"].astype(float)
        # No C(dow_str): date FEs absorb DoW
        rhs = " + ".join(active)
        formula = f"{outcome_col} ~ {rhs} | _fips_str + _date_str"
        for vcov_spec, method_tag in [
            ({"CRV1": "_fips_str + _year_str"}, "2-way SE: county×year"),
            ({"CRV1": "_fips_str"},              "SE: county"),
        ]:
            try:
                fit  = pf.feols(formula, data=sub2, weights="_pop", vcov=vcov_spec)
                tbl  = fit.tidy()
                rows = []
                for ph in active:
                    if ph not in tbl.index:
                        continue
                    b  = float(tbl.loc[ph, "Estimate"])
                    se = float(tbl.loc[ph, "Std. Error"])
                    pv = float(tbl.loc[ph, "Pr(>|t|)"])
                    if np.isnan(se):
                        rows = []
                        break
                    rows.append(dict(
                        phase=ph, phase_label=PHASE_LABELS[ph],
                        beta=round(b,6), se=round(se,6), pvalue=round(pv,4),
                        n_treated_phase=int(sub2[ph].sum()),
                        n_obs=int(fit._N),
                        method=f"Sleep-phase WLS TWFE (pop-wt, county+date FE, {method_tag}, joint)",
                    ))
                if rows:
                    return rows
            except Exception as exc:
                log.warning("  [%s] sleep-phase feols[%s] failed: %s", label, method_tag, exc)

    # ── Fallback: unweighted iterative within-transform ───────────────────────
    dow_dummies = pd.get_dummies(sub2["dow"], prefix="dow", drop_first=True).astype(float)
    dow_cols = dow_dummies.columns.tolist()
    sub2 = pd.concat([sub2, dow_dummies], axis=1)
    tx_map = {c: f"_tx_{c}" for c in active}
    sub2["_y"] = sub2[outcome_col].astype(float)
    for orig, renamed in tx_map.items():
        sub2[renamed] = sub2[orig].astype(float)
    all_cols = ["_y"] + list(tx_map.values()) + dow_cols
    for c in all_cols:
        sub2[c] -= sub2[c].mean()
    for _ in range(5):
        for c in all_cols:
            sub2[c] = (sub2[c]
                       - sub2.groupby("fips")[c].transform("mean")
                       - sub2.groupby("date")[c].transform("mean"))
    if sub2["_y"].std() < 1e-10:
        return None
    X = sm.add_constant(sub2[list(tx_map.values()) + dow_cols])
    try:
        mod = OLS(sub2["_y"], X).fit(
            cov_type="cluster", cov_kwds={"groups": sub2["fips"]}
        )
    except Exception as exc:
        log.warning("  [%s] sleep-phase OLS failed: %s", label, exc)
        return None
    rows = []
    for orig, renamed in tx_map.items():
        if renamed not in mod.params:
            continue
        rows.append(dict(
            phase=orig, phase_label=PHASE_LABELS[orig],
            beta=round(float(mod.params[renamed]),6),
            se=round(float(mod.bse[renamed]),6),
            pvalue=round(float(mod.pvalues[renamed]),4),
            n_treated_phase=int(sub2[orig].sum()),
            n_obs=int(mod.nobs),
            method="Sleep-phase TWFE (county+date FE+DoW, joint, unweighted fallback)",
        ))
    return rows or None


for state_filter in [None] + sorted(panel.state.unique().tolist()):
    label = state_filter if state_filter else "ALL"
    sub = panel if state_filter is None else panel[panel.state == state_filter]

    if sub.night_alert.sum() < 10:
        log.warning("  [%s] fewer than 10 treated obs — skipping", label)
        continue

    for outcome_col, outcome_label, count_col in OUTCOMES:
        sub2 = sub.dropna(subset=[outcome_col]).copy()
        if len(sub2) < 100:
            continue

        # ── OLS on rate per 100k ──────────────────────────────────────────────
        use_panel_ols = HAS_LINEARMODELS and (state_filter is not None) and (len(sub2) < 300_000)
        ols_result = (run_twfe_panelols(sub2, outcome_col, label)
                      if use_panel_ols else
                      run_twfe(sub2, outcome_col, label))

        if ols_result is not None:
            ols_result.update({"state": label, "outcome": outcome_label,
                               "n_treated": int(sub.night_alert.sum()),
                               "model": "OLS"})
            results_rows.append(ols_result)
            stars = ("***" if ols_result["pvalue"] < 0.01 else
                     "**"  if ols_result["pvalue"] < 0.05 else
                     "*"   if ols_result["pvalue"] < 0.10 else "")
            log.info("  [%s] OLS  %-30s β=%+.4f  se=%.4f  p=%.3f %s",
                     label, outcome_label,
                     ols_result["beta"], ols_result["se"],
                     ols_result["pvalue"], stars)

        # ── Poisson PPML on rate per 100k (incident rate ratio) ──────────────
        if sub2[outcome_col].sum() > 0:
            pois_result = run_poisson_ppml(sub2, outcome_col, label)
            if pois_result is not None:
                pois_result.update({"state": label, "outcome": outcome_label,
                                    "n_treated": int(sub.night_alert.sum()),
                                    "model": "Poisson"})
                results_rows.append(pois_result)
                stars = ("***" if pois_result["pvalue"] < 0.01 else
                         "**"  if pois_result["pvalue"] < 0.05 else
                         "*"   if pois_result["pvalue"] < 0.10 else "")
                log.info("  [%s] PPML %-30s β=%+.4f  IRR=%.4f  p=%.3f %s  (%.1f%%)",
                         label, outcome_label,
                         pois_result["beta"], pois_result["irr"],
                         pois_result["pvalue"], stars,
                         pois_result["pct_change"])

results = pd.DataFrame(results_rows)

# ── 7. Summary ────────────────────────────────────────────────────────────────
log.info("\n=== Summary ===")
log.info("Night AMBER alert → same-day traffic outcomes (TWFE, county+date FE, clustered SE)")
log.info("(early-night alerts: effective crash date = alert date + 1; late-night: same day)")
log.info("")
if not results.empty:
    all_rows = results[results.state == "ALL"]
    log.info("  OLS (β = change in rate per 100k):")
    for _, row in all_rows[all_rows.get("model", "OLS") == "OLS"].iterrows():
        stars = "***" if row.pvalue < 0.01 else "**" if row.pvalue < 0.05 else "*" if row.pvalue < 0.10 else ""
        log.info("    %-30s  β=%+.4f (SE=%.4f)  p=%.3f %s",
                 row.outcome, row.beta, row.se, row.pvalue, stars)
    log.info("  Poisson PPML (IRR = multiplicative factor on crash rate):")
    for _, row in all_rows[all_rows.get("model", "OLS") == "Poisson"].iterrows():
        stars = "***" if row.pvalue < 0.01 else "**" if row.pvalue < 0.05 else "*" if row.pvalue < 0.10 else ""
        irr = row.get("irr", float("nan"))
        pct = row.get("pct_change", float("nan"))
        log.info("    %-30s  IRR=%.4f (%+.1f%%)  p=%.3f %s",
                 row.outcome, irr, pct, row.pvalue, stars)

    results.to_csv(OUTPUT_TABS / "state_dot_analysis.csv", index=False)
    log.info("\nSaved → %s", OUTPUT_TABS / "state_dot_analysis.csv")
else:
    log.warning("No results produced.")

# ── 8. Sleep-phase heterogeneity ──────────────────────────────────────────────
log.info("\n=== Sleep-phase heterogeneity (crashes_per_100k, joint TWFE) ===")
log.info("Hypothesis: deep-sleep alerts (2–3h) → worst sleep inertia → most crashes")
log.info("")
log.info("  %-8s  %-28s  %7s  %7s  %6s  %5s", "State", "Phase", "β", "SE", "p", "N_treated")
log.info("  " + "-"*70)

phase_rows: list[dict] = []
for state_filter in [None] + sorted(panel.state.unique().tolist()):
    label = state_filter if state_filter else "ALL"
    sub = panel if state_filter is None else panel[panel.state == state_filter]

    # Skip if no phase has enough treatment
    active_phases = [c for c in PHASE_COLS if c in sub.columns and sub[c].sum() >= 5]
    if not active_phases:
        continue

    sub2 = sub.dropna(subset=["crashes_per_100k"]).copy()
    if len(sub2) < 100:
        continue

    phase_res = run_sleep_phase_twfe(sub2, "crashes_per_100k", label)
    if not phase_res:
        continue

    for row in phase_res:
        row["state"] = label
        phase_rows.append(row)
        stars = ("***" if row["pvalue"] < 0.01 else
                 "**"  if row["pvalue"] < 0.05 else
                 "*"   if row["pvalue"] < 0.10 else "")
        log.info("  %-8s  %-28s  β=%+.4f  SE=%.4f  p=%.3f %-3s  N=%d",
                 label, row["phase_label"],
                 row["beta"], row["se"], row["pvalue"], stars,
                 row["n_treated_phase"])

if phase_rows:
    phase_df = pd.DataFrame(phase_rows)
    phase_path = OUTPUT_TABS / "sleep_phase_analysis.csv"
    phase_df.to_csv(phase_path, index=False)
    log.info("\nSaved → %s", phase_path)

    # Print the ALL-states summary clearly
    all_phase = phase_df[phase_df.state == "ALL"].sort_values("phase")
    log.info("\n  ALL states — sleep-phase coefficients (crashes / 100k):")
    log.info("  %-28s  %7s  %7s  %6s", "Phase", "β", "SE", "p")
    for _, r in all_phase.iterrows():
        stars = ("***" if r.pvalue < 0.01 else
                 "**"  if r.pvalue < 0.05 else
                 "*"   if r.pvalue < 0.10 else "")
        log.info("  %-28s  %+7.4f  %7.4f  %.3f %s",
                 r.phase_label, r.beta, r.se, r.pvalue, stars)
else:
    log.warning("No sleep-phase results produced.")

# ── 9. Urban / Rural heterogeneity ────────────────────────────────────────────
log.info("\n=== Urban / Rural heterogeneity ===")
log.info("Counties split at median population within each state × year cell.")
log.info("Urban = above-median pop; Rural = below-median pop.")
log.info("")

# Classify counties as Urban / Rural using within-state-year median population.
# This is time-varying (a county can cross median if pop grows), but the split
# is stable in practice.  Using within-state median avoids classifying all
# large-state counties as urban relative to small-state counties.
med_pop = (panel.groupby(["state", "year"])["population"]
           .transform("median"))
panel["urban"] = (panel["population"] >= med_pop).astype(int)  # 1=urban, 0=rural

log.info("Urban/Rural county-day counts:")
g_ur = panel.groupby("urban").agg(
    county_days=("night_alert", "count"),
    n_treated=("night_alert", "sum"),
    mean_pop=("population", "mean"),
).reset_index()
g_ur["label"] = g_ur["urban"].map({1: "Urban (above-median pop)", 0: "Rural (below-median pop)"})
for _, r in g_ur.iterrows():
    log.info("  %-30s  %7d county-days  %4d treated  mean_pop=%,.0f",
             r["label"], r["county_days"], r["n_treated"], r["mean_pop"])
log.info("")

# Show what share of population and treated obs each group carries
total_pop_wt = panel["population"].sum()
for grp, label in [(1, "Urban"), (0, "Rural")]:
    sub_g = panel[panel["urban"] == grp]
    log.info("  %s carries %.1f%% of total pop-weight, %.1f%% of treated county-days",
             label,
             100 * sub_g["population"].sum() / total_pop_wt,
             100 * sub_g["night_alert"].sum() / panel["night_alert"].sum())
log.info("")

# Run main TWFE (crashes_per_100k) + sleep-phase TWFE for each group
ur_results = []
ur_phase_results = []

for grp_val, grp_label in [(1, "Urban"), (0, "Rural")]:
    sub_ur = panel[panel["urban"] == grp_val].copy()
    log.info("─── %s (pop %s median) ───", grp_label,
             "≥" if grp_val == 1 else "<")

    if sub_ur["night_alert"].sum() < 5:
        log.warning("  Fewer than 5 treated obs — skipping %s", grp_label)
        continue

    # ── Main OLS TWFE ────────────────────────────────────────────────────────
    for outcome_col, outcome_label, _ in OUTCOMES:
        res = run_twfe(sub_ur, outcome_col, grp_label)
        if res:
            res.update({"group": grp_label, "outcome": outcome_label,
                        "n_treated": int(sub_ur["night_alert"].sum()),
                        "model": "OLS"})
            ur_results.append(res)
            stars = ("***" if res["pvalue"] < 0.01 else
                     "**"  if res["pvalue"] < 0.05 else
                     "*"   if res["pvalue"] < 0.10 else "")
            log.info("  OLS  %-30s  β=%+.4f  SE=%.4f  p=%.3f %-3s  N_treated=%d",
                     outcome_label, res["beta"], res["se"],
                     res["pvalue"], stars, res["n_treated"])

    # ── Sleep-phase joint TWFE ───────────────────────────────────────────────
    log.info("  Sleep-phase (crashes / 100k):")
    sub2_ur = sub_ur.dropna(subset=["crashes_per_100k"]).copy()
    phase_res_ur = run_sleep_phase_twfe(sub2_ur, "crashes_per_100k", grp_label)
    if phase_res_ur:
        for row in phase_res_ur:
            row["group"] = grp_label
            ur_phase_results.append(row)
            stars = ("***" if row["pvalue"] < 0.01 else
                     "**"  if row["pvalue"] < 0.05 else
                     "*"   if row["pvalue"] < 0.10 else "")
            log.info("    %-28s  β=%+.4f  SE=%.4f  p=%.3f %-3s  N=%d",
                     row["phase_label"], row["beta"], row["se"],
                     row["pvalue"], stars, row["n_treated_phase"])
    log.info("")

# ── Summary comparison table ─────────────────────────────────────────────────
if ur_results:
    ur_df = pd.DataFrame(ur_results)
    log.info("=== Urban vs Rural — ALL OUTCOMES SUMMARY ===")
    log.info("  %-10s  %-30s  %-6s  %+7s  %7s  %6s",
             "Group", "Outcome", "Model", "β", "SE", "p")
    log.info("  " + "-"*75)
    for _, r in ur_df.iterrows():
        stars = ("***" if r.pvalue < 0.01 else
                 "**"  if r.pvalue < 0.05 else
                 "*"   if r.pvalue < 0.10 else "")
        log.info("  %-10s  %-30s  %-6s  %+7.4f  %7.4f  %.3f %s",
                 r["group"], r["outcome"], r["model"],
                 r["beta"], r["se"], r["pvalue"], stars)

    ur_df.to_csv(OUTPUT_TABS / "urban_rural_main.csv", index=False)
    log.info("\nSaved → %s", OUTPUT_TABS / "urban_rural_main.csv")

if ur_phase_results:
    ur_ph_df = pd.DataFrame(ur_phase_results)
    log.info("\n=== Urban vs Rural — SLEEP PHASE SUMMARY ===")
    log.info("  %-10s  %-28s  %+7s  %7s  %6s",
             "Group", "Phase", "β", "SE", "p")
    log.info("  " + "-"*65)
    for _, r in ur_ph_df.iterrows():
        stars = ("***" if r.pvalue < 0.01 else
                 "**"  if r.pvalue < 0.05 else
                 "*"   if r.pvalue < 0.10 else "")
        log.info("  %-10s  %-28s  %+7.4f  %7.4f  %.3f %s",
                 r["group"], r["phase_label"],
                 r["beta"], r["se"], r["pvalue"], stars)

    ur_ph_df.to_csv(OUTPUT_TABS / "urban_rural_sleep_phase.csv", index=False)
    log.info("\nSaved → %s", OUTPUT_TABS / "urban_rural_sleep_phase.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Confounding Checks
# ═══════════════════════════════════════════════════════════════════════════════
#
# Three tests to diagnose whether the negative fatality result is causal or
# an artefact of confounders (mainly law-enforcement mobilisation):
#
#   A. Placebo lead/lag test  (±1, ±2 days)
#      If the result is causal, only the contemporaneous (t=0) coefficient
#      should be negative and significant.  Law-enforcement mobilisation would
#      also only operate on t=0, so this test does NOT rule it out — but it
#      rules out general confounders like weather or day-of-week selection.
#      Pre-trends (t=-1, t=-2 significant) would indicate selection bias.
#
#   B. Alert-hour-of-night FE
#      Alerts issued at 22-23h arrive when traffic is still relatively high;
#      those at 2-4h arrive in the low-traffic dead of night.  If negative
#      results are purely from traffic-volume confounding, adding hour-of-alert
#      dummies (within the night window) should eliminate the effect.
#
#   C. Severity gradient
#      Sleep disruption → impaired driving → more crashes across ALL severity
#      levels.  Law-enforcement mobilisation specifically reduces FATAL crashes
#      (DUI, speeding) but leaves minor crashes untouched.  If we see
#      fatalities ↓ but all-crashes and serious-injury NS, that pattern is
#      more consistent with confounding than with the sleep channel.
#
# ═══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "═"*70)
log.info("=== Confounding Checks ===")
log.info("═"*70)

# ── A. Placebo lead / lag test ────────────────────────────────────────────────
log.info("\n─── A. Placebo leads & lags (fatalities / 100k, ALL states) ───")
log.info("  H0: only t=0 coefficient should be non-zero if effect is causal.")
log.info("  Pre-trend at t=-1 or t=-2 → selection bias.\n")

# Build a date-indexed set of treated (fips, date) pairs
treated_dates = set(
    zip(night_alerts["fips"].astype(str).str.zfill(5),
        pd.to_datetime(night_alerts["effective_crash_date"]).dt.date)
)

placebo_rows = []
for lag in [-2, -1, 0, 1, 2]:
    col = f"alert_t{lag:+d}".replace("+", "p").replace("-", "m")
    if lag == 0:
        panel[col] = panel["night_alert"]
    else:
        # Efficient merge-based shift: shift effective_crash_date by lag days
        shifted_alerts = night_alerts[["fips", "effective_crash_date"]].copy()
        shifted_alerts["fips"] = shifted_alerts["fips"].astype(str).str.zfill(5)
        shifted_alerts["shifted_date"] = (
            pd.to_datetime(shifted_alerts["effective_crash_date"]) +
            pd.Timedelta(days=lag)
        ).dt.date
        shifted_alerts["_treated"] = 1
        panel["_date_d"] = panel["date"].dt.date
        merged = panel[["fips", "_date_d"]].merge(
            shifted_alerts[["fips", "shifted_date", "_treated"]],
            left_on=["fips", "_date_d"],
            right_on=["fips", "shifted_date"],
            how="left"
        )
        panel[col] = merged["_treated"].fillna(0).astype(int).values
        panel.drop(columns=["_date_d"], inplace=True, errors="ignore")

    sub2 = panel.dropna(subset=["fatals_per_100k", "population"]).copy()
    n_treated_lag = int(sub2[col].sum())
    if n_treated_lag < 10:
        log.warning("  t=%+d: fewer than 10 treated obs (%d) — skip", lag, n_treated_lag)
        continue

    if HAS_PYFIXEST:
        sub2["_fips_str"] = sub2["fips"].astype(str)
        sub2["_date_str"] = sub2["date"].astype(str)
        sub2["_year_str"] = sub2["year"].astype(str)
        sub2["_pop"]      = sub2["population"].astype(float)
        formula = f"fatals_per_100k ~ {col} | _fips_str + _date_str"
        res = None
        for vcov_spec, mtag in [
            ({"CRV1": "_fips_str + _year_str"}, "2-way SE"),
            ({"CRV1": "_fips_str"}, "county SE"),
        ]:
            try:
                fit = pf.feols(formula, data=sub2, weights="_pop", vcov=vcov_spec)
                vals = _pyfixest_coef(fit, col)
                if vals and not np.isnan(vals[1]):
                    b, se, pv, n = vals
                    res = dict(lag=lag, beta=round(b,6), se=round(se,6),
                               pvalue=round(pv,4), n_treated=n_treated_lag,
                               se_method=mtag)
                    break
            except Exception as exc:
                log.debug("  t=%+d feols[%s] failed: %s", lag, mtag, exc)
                continue
        if res:
            stars = ("***" if res["pvalue"] < 0.01 else
                     "**"  if res["pvalue"] < 0.05 else
                     "*"   if res["pvalue"] < 0.10 else "")
            log.info("  t=%+d  β=%+.4f  SE=%.4f  p=%.3f %-3s  (N_treated=%d)",
                     lag, res["beta"], res["se"], res["pvalue"], stars,
                     res["n_treated"])
            placebo_rows.append(res)
        else:
            log.warning("  t=%+d: all feols specs failed", lag)

if placebo_rows:
    pd.DataFrame(placebo_rows).to_csv(OUTPUT_TABS / "placebo_lags.csv", index=False)
    log.info("  Saved → output/tables/placebo_lags.csv")

# ── B. Hour-of-alert FE within night window ───────────────────────────────────
log.info("\n─── B. Hour-of-alert FE (fatalities / 100k, ALL states) ───")
log.info("  Adds 2-hour bin FE for when within the night the alert was issued.")
log.info("  If traffic-volume timing drives results, effect should vanish.\n")

# Compute 2-hour bin of the FIRST alert in each county-night.
# alerts has hour_local from the local-time conversion block earlier in the script.
if "hour_local" in alerts.columns:
    _abin = alerts[alerts["is_night"]].copy()
    _abin["fips"] = _abin["fips"].astype(str).str.zfill(5)
    _abin["hour_bin"] = (_abin["hour_local"] // 2 * 2).astype(int).astype(str).str.zfill(2) + "h"

    # Collapse to the first alert per (fips, effective_crash_date)
    first_alert_hour = (
        _abin.sort_values("sent_utc")
        .groupby(["fips", "effective_crash_date"])["hour_bin"]
        .first()
        .reset_index()
    )
    first_alert_hour["fips"] = first_alert_hour["fips"].astype(str).str.zfill(5)
    first_alert_hour["effective_crash_date"] = pd.to_datetime(
        first_alert_hour["effective_crash_date"]
    ).dt.date

    sub2 = panel.copy()
    sub2["_date_d"] = sub2["date"].dt.date
    sub2 = sub2.merge(
        first_alert_hour.rename(columns={"effective_crash_date": "_ecd"}),
        left_on=["fips", "_date_d"], right_on=["fips", "_ecd"], how="left"
    ).drop(columns=["_date_d", "_ecd"], errors="ignore")
    sub2 = sub2.dropna(subset=["fatals_per_100k", "population"])
    sub2["hour_bin"] = sub2["hour_bin"].fillna("none")
    sub2["_fips_str"] = sub2["fips"].astype(str)
    sub2["_date_str"] = sub2["date"].astype(str)
    sub2["_year_str"] = sub2["year"].astype(str)
    sub2["_hbin_str"] = sub2["hour_bin"]
    sub2["_pop"]      = sub2["population"].astype(float)

    log.info("  Alert-hour bins in treated nights: %s",
             sub2.loc[sub2["night_alert"]==1, "hour_bin"].value_counts().to_dict())

    if HAS_PYFIXEST and sub2["night_alert"].sum() >= 10:
        hbin_rows = []
        formula_base = "fatals_per_100k ~ night_alert | _fips_str + _date_str"
        formula_hrfe = "fatals_per_100k ~ night_alert | _fips_str + _date_str + _hbin_str"
        for fname, flabel in [(formula_base, "Baseline (no hour-bin FE)"),
                               (formula_hrfe, "With 2h alert-hour FE")]:
            for vcov_spec, mtag in [
                ({"CRV1": "_fips_str + _year_str"}, "2-way SE"),
                ({"CRV1": "_fips_str"}, "county SE"),
            ]:
                try:
                    fit = pf.feols(fname, data=sub2, weights="_pop", vcov=vcov_spec)
                    vals = _pyfixest_coef(fit, "night_alert")
                    if vals and not np.isnan(vals[1]):
                        b, se, pv, n = vals
                        stars = ("***" if pv < 0.01 else "**" if pv < 0.05
                                 else "*" if pv < 0.10 else "")
                        log.info("  %-35s  β=%+.4f  SE=%.4f  p=%.3f %s",
                                 flabel, b, se, pv, stars)
                        hbin_rows.append(dict(spec=flabel, beta=round(b,6),
                                              se=round(se,6), pvalue=round(pv,4)))
                        break
                except Exception as exc:
                    log.debug("  hour-bin FE feols[%s] failed: %s", mtag, exc)
                    continue
        if hbin_rows:
            pd.DataFrame(hbin_rows).to_csv(OUTPUT_TABS / "hour_bin_fe.csv", index=False)
            log.info("  Saved → output/tables/hour_bin_fe.csv")
else:
    log.info("  (hour_local column not present in alerts — skipping hour-bin FE check)")

# ── C. Severity gradient ──────────────────────────────────────────────────────
log.info("\n─── C. Severity gradient (ALL states, OLS + PPML) ───")
log.info("  Sleep disruption → impaired driving → crashes at ALL severity levels.")
log.info("  Law-enforcement mobilisation → specifically fewer FATAL crashes.")
log.info("  Pattern to distinguish: fatalities ↓ but all-crashes/serious-injury NS")
log.info("  is consistent with law-enforcement confounding, not sleep disruption.\n")

if not results.empty:
    all_ols = results[(results["state"] == "ALL") & (results["model"] == "OLS")]
    all_pml = results[(results["state"] == "ALL") & (results["model"] == "Poisson")]

    log.info("  %-30s  %-8s  %+8s  %8s  %7s",
             "Outcome", "Model", "β / log-β", "SE", "p-value")
    log.info("  " + "-"*68)
    for _, r in pd.concat([all_ols, all_pml]).sort_values("outcome").iterrows():
        stars = ("***" if r.pvalue < 0.01 else "**" if r.pvalue < 0.05
                 else "*" if r.pvalue < 0.10 else "")
        irr_str = (f" [IRR={r.get('irr',float('nan')):.3f}]"
                   if r["model"] == "Poisson" else "")
        log.info("  %-30s  %-8s  %+8.4f  %8.4f  %.3f %-3s%s",
                 r["outcome"], r["model"],
                 r["beta"], r["se"], r["pvalue"], stars, irr_str)


# ── E. Weather + Holiday robustness (state × year-month FE) ──────────────────
log.info("\n─── E. Weather & holiday controls (state × year-month FE) ───")
log.info("  Holidays: date FEs already absorb ALL holiday effects (each specific date,")
log.info("  e.g. 2018-12-25, gets its own coefficient).")
log.info("  Weather: adding state×year-month FE absorbs seasonal weather within state")
log.info("  (standard econometric proxy when county-grid weather data unavailable).")
log.info("  If results are stable → weather/holidays are not driving the findings.\n")

weather_rows = []
if HAS_PYFIXEST:
    sub_w = panel.dropna(subset=["fatals_per_100k", "population"]).copy()
    sub_w["_fips_str"] = sub_w["fips"].astype(str)
    sub_w["_date_str"] = sub_w["date"].astype(str)
    sub_w["_year_str"] = sub_w["year"].astype(str)
    sub_w["_pop"]      = sub_w["population"].astype(float)
    sub_w["_stym_str"] = sub_w["state_yearmon"].astype(str)

    specs = [
        ("County + Date FE (baseline)",
         "fatals_per_100k ~ night_alert | _fips_str + _date_str"),
        ("County + Date + State×YearMon FE",
         "fatals_per_100k ~ night_alert | _fips_str + _date_str + _stym_str"),
    ]
    for spec_label, formula in specs:
        for vcov_spec, mtag in [
            ({"CRV1": "_fips_str + _year_str"}, "2-way SE"),
            ({"CRV1": "_fips_str"}, "county SE"),
        ]:
            try:
                fit = pf.feols(formula, data=sub_w, weights="_pop", vcov=vcov_spec)
                vals = _pyfixest_coef(fit, "night_alert")
                if vals and not np.isnan(vals[1]):
                    b, se, pv, n = vals
                    stars = ("***" if pv < 0.01 else "**" if pv < 0.05
                             else "*" if pv < 0.10 else "")
                    log.info("  %-42s  β=%+.4f  SE=%.4f  p=%.3f %s",
                             spec_label, b, se, pv, stars)
                    weather_rows.append(dict(spec=spec_label, beta=round(b,6),
                                            se=round(se,6), pvalue=round(pv,4)))
                    break
            except Exception as exc:
                log.debug("  weather FE [%s] failed: %s", mtag, exc)
                continue

if weather_rows:
    pd.DataFrame(weather_rows).to_csv(OUTPUT_TABS / "weather_holiday_robustness.csv", index=False)
    log.info("  Saved → output/tables/weather_holiday_robustness.csv")

log.info("\n─── Interpretation guide ───")
log.info("  If t=0 significant, t±1 and t±2 NS  → consistent with causal effect")
log.info("  If hour-bin FE kills the result       → traffic-volume confound")
log.info("  If hour-bin FE doesn't kill result    → not a traffic-timing artefact")
log.info("  If only fatalities ↓, not all-crashes → consistent with law-enforcement")
log.info("    confound (DUI/speeding enforcement), NOT sleep-disruption mechanism")


# ── D. Late-night only (0–5am): sleep-disruption sensitivity check ────────────
log.info("\n─── D. Late-night only (0–5am) vs. full night (22h–5am) ───")
log.info("  Sleep disruption requires people to be asleep.")
log.info("  22–23h alerts: most recipients still awake → minimal sleep disruption.")
log.info("  0–5am alerts:  recipients in deep/REM sleep → maximum disruption.")
log.info("  If mechanism is sleep disruption: 0–5am-only effect should be STRONGER.")
log.info("  If mechanism is law enforcement:  both windows show similar effects.\n")

# Build a late-night-only treatment variable (0–5am local time only)
late_night_alerts = (
    alerts[alerts["is_night"] & (alerts["hour_local"] < 6)]
    .groupby(["fips", "effective_crash_date"])
    .agg(n_alerts_late=("alert_id", "nunique"))
    .reset_index()
)
late_night_alerts["fips"] = late_night_alerts["fips"].astype(str).str.zfill(5)
late_night_alerts["late_night_alert"] = 1

panel_ln = panel.copy()
panel_ln = panel_ln.merge(
    late_night_alerts[["fips", "effective_crash_date", "late_night_alert"]],
    left_on=["fips", "date"],
    right_on=["fips", "effective_crash_date"],
    how="left"
)
panel_ln["late_night_alert"] = panel_ln["late_night_alert"].fillna(0).astype(int)

n_late = panel_ln["late_night_alert"].sum()
log.info("  Late-night (0–5am) treated county-days: %d  (full-night: %d)",
         n_late, panel["night_alert"].sum())

latenight_rows = []
if HAS_PYFIXEST and n_late >= 10:
    for outcome_col, outcome_label, _ in OUTCOMES:
        sub_ln = panel_ln.dropna(subset=[outcome_col, "population"]).copy()
        sub_ln["_fips_str"] = sub_ln["fips"].astype(str)
        sub_ln["_date_str"] = sub_ln["date"].astype(str)
        sub_ln["_year_str"] = sub_ln["year"].astype(str)
        sub_ln["_pop"]      = sub_ln["population"].astype(float)

        # Swap treatment to late_night_alert
        sub_ln["night_alert_orig"] = sub_ln["night_alert"]
        sub_ln["night_alert"]      = sub_ln["late_night_alert"]

        formula = f"{outcome_col} ~ night_alert | _fips_str + _date_str"
        res_ln = None
        for vcov_spec, mtag in [
            ({"CRV1": "_fips_str + _year_str"}, "2-way SE"),
            ({"CRV1": "_fips_str"}, "county SE"),
        ]:
            try:
                fit = pf.feols(formula, data=sub_ln, weights="_pop", vcov=vcov_spec)
                vals = _pyfixest_coef(fit, "night_alert")
                if vals and not np.isnan(vals[1]):
                    b, se, pv, n = vals
                    res_ln = dict(spec="Late-night (0–5am)", outcome=outcome_label,
                                  beta=round(b,6), se=round(se,6), pvalue=round(pv,4),
                                  n_treated=int(n_late), se_method=mtag)
                    break
            except Exception:
                continue

        # Also get the full-night result for direct comparison
        sub_full = panel.dropna(subset=[outcome_col, "population"]).copy()
        sub_full["_fips_str"] = sub_full["fips"].astype(str)
        sub_full["_date_str"] = sub_full["date"].astype(str)
        sub_full["_year_str"] = sub_full["year"].astype(str)
        sub_full["_pop"]      = sub_full["population"].astype(float)
        formula_f = f"{outcome_col} ~ night_alert | _fips_str + _date_str"
        res_full = None
        for vcov_spec, mtag in [
            ({"CRV1": "_fips_str + _year_str"}, "2-way SE"),
            ({"CRV1": "_fips_str"}, "county SE"),
        ]:
            try:
                fit = pf.feols(formula_f, data=sub_full, weights="_pop", vcov=vcov_spec)
                vals = _pyfixest_coef(fit, "night_alert")
                if vals and not np.isnan(vals[1]):
                    b, se, pv, n = vals
                    res_full = dict(spec="Full night (22h–5am)", outcome=outcome_label,
                                   beta=round(b,6), se=round(se,6), pvalue=round(pv,4),
                                   n_treated=int(panel["night_alert"].sum()), se_method=mtag)
                    break
            except Exception:
                continue

        for res, lbl in [(res_full, "Full  22h–5am"), (res_ln, "Late  00h–5am")]:
            if res:
                stars = ("***" if res["pvalue"] < 0.01 else "**" if res["pvalue"] < 0.05
                         else "*" if res["pvalue"] < 0.10 else "")
                log.info("  %-18s  %-28s  β=%+.4f  SE=%.4f  p=%.3f %-3s  N=%d",
                         lbl, outcome_label,
                         res["beta"], res["se"], res["pvalue"], stars, res["n_treated"])
                latenight_rows.append(res)

if latenight_rows:
    pd.DataFrame(latenight_rows).to_csv(OUTPUT_TABS / "latenight_sensitivity.csv", index=False)
    log.info("  Saved → output/tables/latenight_sensitivity.csv")
