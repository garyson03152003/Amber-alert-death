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

# Collapse to county-crash_date: 1 if any night alert targets that county-day
night_alerts = (
    alerts[alerts.is_night]
    .groupby(["fips", "effective_crash_date"])
    .size()
    .reset_index(name="n_alerts")
)
night_alerts["fips"] = night_alerts["fips"].astype(str).str.zfill(5)
night_alerts["night_alert"] = 1
log.info("  County-crash_dates with ≥1 night AMBER alert: %d", len(night_alerts))

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
        sub2["_fips_str"] = sub2["fips"].astype(str)
        sub2["_date_str"] = sub2["date"].astype(str)
        sub2["_year_str"] = sub2["year"].astype(str)
        sub2["_pop"]      = sub2["population"].astype(float)
        # No C(dow_str): date FEs absorb DoW perfectly
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
