"""
run_state_dot_analysis.py
========================================================
Initial analysis: do AMBER alerts correlate with more traffic crashes
per 100k residents, using all-crash (not fatals-only) state DOT data?

Data sources:
  - State DOT crash panels (county-day or county-month):
      CA, FL, IL, IA, MA, NV, PA, WI
  - AMBER alert records: data/raw/amber/foia/openfema_ipaws_alerts_2013_2022.csv
  - Census county population: data/processed/county_population.parquet

Method: Two-Way Fixed Effects (TWFE) OLS
  crashes_per_100k_{it+1} = β · night_alert_{it}
                           + county_FE_i + date_FE_t
                           + ε_{it}

where t indexes the alert date and t+1 is the next calendar day.
Standard errors clustered by county.

Outcome variables (per 100k):
  crashes_per_100k   — total crashes
  fatals_per_100k    — fatalities
  serious_per_100k   — serious injuries (KABCO-A)

Treatment:
  night_alert — 1 if county received nighttime AMBER alert (10pm-6am local)
                on day t

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
    "WI": ("wisconsin_dot_county_day.parquet",      "wi_crashes", "wi_fatals", "wi_serious_inj"),
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
    df = df.rename(columns={c_crashes: "crashes", c_fatals: "fatals", c_serious: "serious_inj"})
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

AMBER_CSV = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2022.csv"
if not AMBER_CSV.exists():
    log.error("AMBER alert file not found at %s", AMBER_CSV)
    log.error("Run: python code/02c_fetch_openfema_ipaws.py")
    sys.exit(1)

alerts = pd.read_csv(AMBER_CSV, parse_dates=["sent_utc"])
log.info("  Raw AMBER records: %d", len(alerts))

# Convert UTC → local night classification
# Use state_fips → UTC offset (approximate; ignores DST)
STATE_UTC_OFFSET = {
    "01": -6, "02": -9, "04": -7, "05": -6, "06": -8, "08": -7, "09": -5,
    "10": -5, "11": -5, "12": -5, "13": -5, "15": -10, "16": -7, "17": -6,
    "18": -5, "19": -6, "20": -6, "21": -6, "22": -6, "23": -5, "24": -5,
    "25": -5, "26": -5, "27": -6, "28": -6, "29": -6, "30": -7, "31": -6,
    "32": -8, "33": -5, "34": -5, "35": -7, "36": -5, "37": -5, "38": -6,
    "39": -5, "40": -6, "41": -8, "42": -5, "44": -5, "45": -5, "46": -6,
    "47": -6, "48": -6, "49": -7, "50": -5, "51": -5, "53": -8, "54": -5,
    "55": -6, "56": -7,
}

alerts["state_fips"] = alerts["fips"].astype(str).str.zfill(5).str[:2]
alerts["utc_offset"] = alerts["state_fips"].map(STATE_UTC_OFFSET).fillna(-6)
alerts["sent_local"] = alerts["sent_utc"] + pd.to_timedelta(alerts["utc_offset"], unit="h")
alerts["hour_local"] = alerts["sent_local"].dt.hour

# Night alert: 10pm–6am local
alerts["is_night"] = (alerts["hour_local"] >= 22) | (alerts["hour_local"] < 6)

# Alert date = calendar date of the alert (local time)
alerts["alert_date"] = alerts["sent_local"].dt.normalize()

log.info("  Total alerts: %d   Night alerts: %d (%.1f%%)",
         len(alerts), alerts.is_night.sum(), 100*alerts.is_night.mean())

# Collapse to county-day: 1 if any night alert issued to that county that day
night_alerts = (
    alerts[alerts.is_night]
    .groupby(["fips", "alert_date"])
    .size()
    .reset_index(name="n_alerts")
)
night_alerts["fips"] = night_alerts["fips"].astype(str).str.zfill(5)
night_alerts["night_alert"] = 1
log.info("  County-nights with ≥1 night AMBER alert: %d", len(night_alerts))

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

# Shift crashes: outcome is next-day crashes (t+1)
# We sort and shift within (fips, state) groups so we don't bleed across states
panel = panel.sort_values(["state", "fips", "date"])
for col in ["crashes", "fatals", "serious_inj",
            "crashes_per_100k", "fatals_per_100k", "serious_per_100k"]:
    panel[f"{col}_t1"] = panel.groupby(["state", "fips"])[col].shift(-1)

# Day of week + month for controls
panel["dow"]   = panel["date"].dt.dayofweek   # 0=Mon
panel["month"] = panel["date"].dt.month

# Merge night alerts (alert on day t → outcome on day t+1)
night_alerts["fips"] = night_alerts["fips"].astype(str).str.zfill(5)
panel = panel.merge(
    night_alerts[["fips", "alert_date", "night_alert"]],
    left_on=["fips", "date"], right_on=["fips", "alert_date"],
    how="left"
)
panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)

# Restrict to study years that overlap between crash data and alert data
panel = panel[panel["year"].between(2013, 2022)]
panel = panel.dropna(subset=["crashes_per_100k_t1"])

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
        "mean_crashes_100k": s.crashes_per_100k_t1.mean(),
        "mean_crashes_alert":  s_alert.crashes_per_100k_t1.mean() if len(s_alert) else np.nan,
        "mean_crashes_notalert": s_noalert.crashes_per_100k_t1.mean() if len(s_noalert) else np.nan,
        "mean_fatals_100k":  s.fatals_per_100k_t1.mean(),
        "mean_serious_100k": s.serious_per_100k_t1.mean(),
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
    log.warning("linearmodels not installed — using statsmodels with entity dummies (slower)")

results_rows = []

OUTCOMES = [
    ("crashes_per_100k_t1", "All crashes / 100k (t+1)"),
    ("fatals_per_100k_t1",  "Fatalities / 100k (t+1)"),
    ("serious_per_100k_t1", "Serious inj / 100k (t+1)"),
]

for state_filter in [None] + sorted(panel.state.unique().tolist()):
    label = state_filter if state_filter else "ALL"
    sub = panel if state_filter is None else panel[panel.state == state_filter]

    if sub.night_alert.sum() < 10:
        log.warning("  [%s] fewer than 10 treated obs — skipping", label)
        continue

    for outcome_col, outcome_label in OUTCOMES:
        sub2 = sub.dropna(subset=[outcome_col]).copy()
        if len(sub2) < 100:
            continue

        if HAS_LINEARMODELS:
            try:
                # Multi-index: entity=fips, time=date
                sub2 = sub2.set_index(["fips", "date"])
                mod = PanelOLS(
                    sub2[outcome_col],
                    sub2[["night_alert"]],
                    entity_effects=True,
                    time_effects=True,
                    drop_absorbed=True,
                )
                res = mod.fit(cov_type="clustered", cluster_entity=True)
                b   = res.params["night_alert"]
                se  = res.std_errors["night_alert"]
                pv  = res.pvalues["night_alert"]
                n   = int(res.nobs)
                results_rows.append({
                    "state":   label,
                    "outcome": outcome_label,
                    "beta":    round(b, 6),
                    "se":      round(se, 6),
                    "pvalue":  round(pv, 4),
                    "n_obs":   n,
                    "n_treated": int(sub.night_alert.sum()),
                    "method":  "PanelOLS TWFE (linearmodels)",
                })
                stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
                log.info("  [%s] %-35s β=%+.4f  se=%.4f  p=%.3f %s  n=%d",
                         label, outcome_label, b, se, pv, stars, n)
            except Exception as exc:
                log.warning("  [%s] %s failed: %s", label, outcome_label, exc)
        else:
            # Fallback: statsmodels with absorbed dummies via within-transform
            import statsmodels.formula.api as smf
            sub2["y"] = sub2[outcome_col]
            # Demean within county (entity FE) and within date (time FE)
            sub2["y_dm"] = (sub2["y"]
                            - sub2.groupby("fips")["y"].transform("mean")
                            - sub2.groupby("date")["y"].transform("mean")
                            + sub2["y"].mean())
            sub2["x_dm"] = (sub2["night_alert"]
                            - sub2.groupby("fips")["night_alert"].transform("mean")
                            - sub2.groupby("date")["night_alert"].transform("mean")
                            + sub2["night_alert"].mean())
            if sub2["x_dm"].std() < 1e-10:
                continue
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
            mod = OLS(sub2["y_dm"], sm.add_constant(sub2[["x_dm"]])).fit(
                cov_type="cluster", cov_kwds={"groups": sub2["fips"]}
            )
            b  = mod.params["x_dm"]
            se = mod.bse["x_dm"]
            pv = mod.pvalues["x_dm"]
            n  = int(mod.nobs)
            results_rows.append({
                "state":   label,
                "outcome": outcome_label,
                "beta":    round(b, 6),
                "se":      round(se, 6),
                "pvalue":  round(pv, 4),
                "n_obs":   n,
                "n_treated": int(sub.night_alert.sum()),
                "method":  "Within-transform OLS (fallback)",
            })
            stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
            log.info("  [%s] %-35s β=%+.4f  se=%.4f  p=%.3f %s  n=%d",
                     label, outcome_label, b, se, pv, stars, n)

results = pd.DataFrame(results_rows)

# ── 7. Summary ────────────────────────────────────────────────────────────────
log.info("\n=== Summary ===")
log.info("Night AMBER alert → next-day traffic outcomes (TWFE, county+date FE, clustered SE)")
log.info("")
if not results.empty:
    for _, row in results[results.state == "ALL"].iterrows():
        stars = ("***" if row.pvalue < 0.01 else
                 "**"  if row.pvalue < 0.05 else
                 "*"   if row.pvalue < 0.10 else "")
        log.info("  %-35s  β=%+.4f (SE=%.4f)  p=%.3f %s",
                 row.outcome, row.beta, row.se, row.pvalue, stars)

    results.to_csv(OUTPUT_TABS / "state_dot_analysis.csv", index=False)
    log.info("\nSaved → %s", OUTPUT_TABS / "state_dot_analysis.csv")
else:
    log.warning("No results produced.")
