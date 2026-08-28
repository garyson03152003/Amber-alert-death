"""
run_fars_national_analysis.py
=============================================================
TWFE analysis using FARS (national fatality data) as outcome.

Memory-efficient design:
  - Minimal dtypes (float32 outcomes, int8 indicators, category FEs)
  - Garbage-collected after panel construction
  - Only necessary columns passed to pyfixest

Outcomes:
  - total_fatals_per_100k  : all traffic fatalities / 100k pop
  - drunk_fatals_per_100k  : DR_DRINK==1 crashes / 100k pop
  - sober_fatals_per_100k  : (total - drunk) / 100k pop

Treatment:
  night_alert — county received ≥1 AMBER WEA nighttime alert
                (Alert+Update msgType only; 22:00–05:59 local)

Output:
  output/tables/fars_national_main.csv
  output/tables/fars_national_drunk_sober.csv
  output/tables/fars_national_california.csv
  output/tables/fars_national_weekend.csv
  output/tables/fars_national_weather.csv
  output/tables/fars_national_latenight.csv
"""

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("pyfixest not available — exiting")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW  = ROOT / "data" / "raw"
OUT_DIR   = ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fars_national")

# ── Timezone maps ──────────────────────────────────────────────────────────────
STATE_TIMEZONE = {
    "01": "America/Chicago",    "02": "America/Anchorage",
    "04": "America/Phoenix",    "05": "America/Chicago",
    "06": "America/Los_Angeles","08": "America/Denver",
    "09": "America/New_York",   "10": "America/New_York",
    "11": "America/New_York",   "12": "America/New_York",
    "13": "America/New_York",   "15": "Pacific/Honolulu",
    "16": "America/Boise",      "17": "America/Chicago",
    "18": "America/Indiana/Indianapolis",
    "19": "America/Chicago",    "20": "America/Chicago",
    "21": "America/New_York",   "22": "America/Chicago",
    "23": "America/New_York",   "24": "America/New_York",
    "25": "America/New_York",   "26": "America/Detroit",
    "27": "America/Chicago",    "28": "America/Chicago",
    "29": "America/Chicago",    "30": "America/Denver",
    "31": "America/Chicago",    "32": "America/Los_Angeles",
    "33": "America/New_York",   "34": "America/New_York",
    "35": "America/Denver",     "36": "America/New_York",
    "37": "America/New_York",   "38": "America/Chicago",
    "39": "America/New_York",   "40": "America/Chicago",
    "41": "America/Los_Angeles","42": "America/New_York",
    "44": "America/New_York",   "45": "America/New_York",
    "46": "America/Chicago",    "47": "America/Chicago",
    "48": "America/Chicago",    "49": "America/Denver",
    "50": "America/New_York",   "51": "America/New_York",
    "53": "America/Los_Angeles","54": "America/New_York",
    "55": "America/Chicago",    "56": "America/Denver",
    "72": "America/Puerto_Rico","78": "America/St_Thomas",
}

COUNTY_TIMEZONE_OVERRIDE = {
    "12033": "America/Chicago", "12059": "America/Chicago",
    "12077": "America/Chicago", "12113": "America/Chicago",
    "12131": "America/Chicago",
    "16021": "America/Los_Angeles", "16055": "America/Los_Angeles",
    "16057": "America/Los_Angeles", "16069": "America/Los_Angeles",
    "16079": "America/Los_Angeles",
    "20129": "America/Denver", "20189": "America/Denver",
    "21007": "America/Chicago", "21083": "America/Chicago",
    "21139": "America/Chicago", "21145": "America/Chicago",
    "21157": "America/Chicago", "21179": "America/Chicago",
    "21195": "America/Chicago", "21221": "America/Chicago",
    "26003": "America/Chicago", "26013": "America/Chicago",
    "26033": "America/Chicago", "26041": "America/Chicago",
    "26043": "America/Chicago", "26053": "America/Chicago",
    "26061": "America/Chicago", "26071": "America/Chicago",
    "26083": "America/Chicago", "26095": "America/Chicago",
    "26097": "America/Chicago", "26103": "America/Chicago",
    "26131": "America/Chicago", "26153": "America/Chicago",
    "31007": "America/Denver", "31057": "America/Denver",
    "31069": "America/Denver", "31123": "America/Denver",
    "31157": "America/Denver", "31165": "America/Denver",
    "31173": "America/Denver",
    "38011": "America/Denver", "38025": "America/Denver",
    "38041": "America/Denver", "38053": "America/Denver",
    "38055": "America/Denver", "38087": "America/Denver",
    "38105": "America/Denver",
    "41001": "America/Denver", "41017": "America/Denver",
    "41021": "America/Denver", "41023": "America/Denver",
    "41025": "America/Denver", "41035": "America/Denver",
    "41037": "America/Denver", "41045": "America/Denver",
    "41049": "America/Denver", "41055": "America/Denver",
    "41059": "America/Denver", "41065": "America/Denver",
    "46017": "America/Denver", "46033": "America/Denver",
    "46047": "America/Denver", "46063": "America/Denver",
    "46065": "America/Denver", "46093": "America/Denver",
    "46105": "America/Denver", "46113": "America/Denver",
    "46117": "America/Denver",
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
    "47153": "America/New_York", "47155": "America/New_York",
    "47163": "America/New_York", "47173": "America/New_York",
    "47179": "America/New_York",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: TWFE via pyfixest
# ═══════════════════════════════════════════════════════════════════════════════
def run_twfe(sub: pd.DataFrame, outcome_col: str, label: str,
             extra_fe: str = "", treat_col: str = "night_alert",
             vcov_spec: str = "iid") -> dict | None:
    """
    Run TWFE with county + date FE, population-weighted WLS.
    vcov_spec: 'iid' for fast initial run; '2way' for county×year clustering
    """
    # Minimal copy — only needed columns
    cols = [outcome_col, treat_col, "population", "fips", "date", "year"]
    if extra_fe:
        cols += [c for c in ["state_yearmon", "weather_adverse"] if c in sub.columns]
    sub2 = sub[cols].dropna(subset=[outcome_col, "population"]).copy()

    if sub2[treat_col].std() < 1e-12 or len(sub2) < 200:
        log.warning("  %s: insufficient variation or obs (%d rows)", label, len(sub2))
        return None

    # String FE columns (category dtype is memory-efficient)
    sub2["_fips_c"] = sub2["fips"].astype("category")
    sub2["_date_c"] = sub2["date"].astype(str).astype("category")
    sub2["_year_c"] = sub2["year"].astype(str).astype("category")
    sub2["_pop"]    = sub2["population"].astype(float)

    fe_str = "_fips_c + _date_c"
    if extra_fe == "stym":
        sub2["_stym_c"] = sub2["state_yearmon"].astype("category")
        fe_str += " + _stym_c"
    elif extra_fe == "weather":
        # weather_adverse as covariate (not FE)
        formula = f"{outcome_col} ~ {treat_col} + weather_adverse | _fips_c + _date_c"

    if extra_fe != "weather":
        formula = f"{outcome_col} ~ {treat_col} | {fe_str}"

    if vcov_spec == "2way":
        vcov = {"CRV1": "_fips_c + _year_c"}
    else:
        vcov = {"CRV1": "_fips_c"}  # county clustering (faster than 2-way)

    try:
        fit = pf.feols(fml=formula, data=sub2, weights="_pop", vcov=vcov)
        coef = fit.coef().get(treat_col, float("nan"))
        se   = fit.se().get(treat_col,   float("nan"))
        t    = fit.tstat().get(treat_col, float("nan"))
        p    = fit.pvalue().get(treat_col, float("nan"))
        n    = int(getattr(fit, "_N", len(sub2)))
        del sub2, fit
        gc.collect()
        return dict(label=label, N=n, coef=coef, se=se, t=t, p=p,
                    outcome=outcome_col, treat=treat_col)
    except Exception as e:
        log.warning("  %s: feols failed — %s", label, e)
        del sub2
        gc.collect()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load AMBER alerts → night_alert treatment
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== Loading AMBER alerts ===")
ALERT_PATH = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2024.csv"
alerts = pd.read_csv(ALERT_PATH)
log.info("  Total rows: %d  Alert+Update: %d  Cancel: %d",
         len(alerts),
         (alerts["msg_type"].isin(["Alert","Update"])).sum(),
         (alerts["msg_type"] == "Cancel").sum())

alerts = alerts[alerts["msg_type"].isin(["Alert", "Update"])].copy()

alerts["fips"]       = alerts["fips"].astype(str).str.zfill(5)
alerts["state_fips"] = alerts["fips"].str[:2]
alerts["is_state_fips"] = alerts["fips"].str[2:] == "000"

def _alert_type(g):
    s = g["is_state_fips"].any(); c = (~g["is_state_fips"]).any()
    return "A" if (s and not c) else ("B" if (s and c) else "C")

_atype = alerts.groupby("alert_id").apply(_alert_type).rename("alert_type")
alerts  = alerts.join(_atype, on="alert_id")
n_A = alerts.loc[alerts["alert_type"] == "A", "alert_id"].nunique()
n_C = alerts.loc[alerts["alert_type"] == "C", "alert_id"].nunique()
log.info("  Alert types: A(state-only)=%d  C(county)=%d  (keeping C+B county rows)", n_A, n_C)

alerts_county = alerts[~alerts["is_state_fips"]].copy()

# Timezone conversion
alerts_county["tz_name"] = (
    alerts_county["fips"].map(COUNTY_TIMEZONE_OVERRIDE)
    .fillna(alerts_county["state_fips"].map(STATE_TIMEZONE))
    .fillna("America/Chicago")
)
alerts_county["sent_utc"] = pd.to_datetime(alerts_county["sent_utc"], utc=True)
alerts_county["hour_local"] = 0
alerts_county["sent_local"]  = pd.NaT
utc_series = alerts_county["sent_utc"]
for tz_name, idx in alerts_county.groupby("tz_name").groups.items():
    tz    = pytz.timezone(tz_name)
    local = utc_series.loc[idx].dt.tz_convert(tz)
    alerts_county.loc[idx, "hour_local"] = local.dt.hour.values
    alerts_county.loc[idx, "sent_local"] = local.dt.tz_localize(None).values

alerts_county["is_night"] = (
    (alerts_county["hour_local"] >= 22) | (alerts_county["hour_local"] < 6)
)
alerts_county["alert_date"] = pd.to_datetime(alerts_county["sent_local"]).dt.normalize()
alerts_county["effective_crash_date"] = np.where(
    alerts_county["hour_local"] >= 22,
    alerts_county["alert_date"] + pd.Timedelta(days=1),
    alerts_county["alert_date"],
)
alerts_county["effective_crash_date"] = pd.to_datetime(alerts_county["effective_crash_date"])

# Night alerts: county × crash_date (any window)
night_alerts = (
    alerts_county[alerts_county["is_night"]]
    .groupby(["fips", "effective_crash_date"])
    .size().reset_index(name="n_alerts")
)
night_alerts["night_alert"] = np.int8(1)
log.info("  Night-alert county-crash_dates: %d  (%d unique counties)",
         len(night_alerts), night_alerts["fips"].nunique())

# Late-night only (0–5am) alerts
latenight_alerts = (
    alerts_county[alerts_county["is_night"] & (alerts_county["hour_local"] < 6)]
    .groupby(["fips", "effective_crash_date"])
    .size().reset_index(name="n_late")
)
latenight_alerts["latenight_alert"] = np.int8(1)
log.info("  Late-night (0–5am) county-crash_dates: %d", len(latenight_alerts))

del alerts, alerts_county, _atype
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load FARS + population; build panel
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== Loading FARS + building panel ===")
fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
fars["fips"] = fars["fips"].astype(str).str.zfill(5)
fars["date"] = pd.to_datetime(fars["date"])
log.info("  FARS: %d county-days  %d counties  %d–%d  total_fatals=%d  drunk=%.1f%%",
         len(fars), fars["fips"].nunique(),
         fars["date"].dt.year.min(), fars["date"].dt.year.max(),
         fars["total_fatals"].sum(),
         100 * fars["drunk_fatals"].sum() / max(fars["total_fatals"].sum(), 1))

pop = pd.read_parquet(DATA_PROC / "county_population.parquet")
pop["fips"] = pop["fips"].astype(str).str.zfill(5)

all_fips   = sorted(fars["fips"].unique())
date_range = pd.date_range(start=fars["date"].min(), end=fars["date"].max(), freq="D")
log.info("  Building %d counties × %d dates = ~%.1fM rows",
         len(all_fips), len(date_range), len(all_fips)*len(date_range)/1e6)

# Build panel efficiently using MultiIndex
panel_idx = pd.MultiIndex.from_product([all_fips, date_range], names=["fips", "date"])
panel = (pd.DataFrame(index=panel_idx).reset_index()
         .astype({"fips": "category"}))  # category saves memory

# Merge FARS outcomes (fill zeros)
panel = panel.merge(
    fars[["fips", "date", "total_fatals", "drunk_fatals", "sober_fatals", "weather_adverse"]],
    on=["fips", "date"], how="left"
)
for col in ["total_fatals", "drunk_fatals", "sober_fatals"]:
    panel[col] = panel[col].fillna(0).astype(np.int16)
panel["weather_adverse"] = panel["weather_adverse"].fillna(0).astype(np.int8)

del fars
gc.collect()
log.info("  Panel: %d rows  Non-zero: %d (%.1f%%)",
         len(panel), (panel["total_fatals"] > 0).sum(),
         100*(panel["total_fatals"] > 0).mean())

# Merge population by year
panel["year"] = panel["date"].dt.year.astype(np.int16)
panel = panel.merge(pop[["fips","year","population"]], on=["fips","year"], how="left")

# Fill missing population by county median
miss = panel["population"].isna()
if miss.sum():
    pop_median = panel.groupby("fips")["population"].transform("median")
    panel.loc[miss, "population"] = pop_median[miss]
panel = panel.dropna(subset=["population"])
panel["population"] = panel["population"].astype(np.float32)

del pop, pop_median, miss
gc.collect()
log.info("  Panel after pop merge: %d rows  %d counties", len(panel), panel["fips"].nunique())

# Per-100k rates (float32 saves memory)
pop_f = panel["population"].values
panel["total_fatals_per_100k"] = (panel["total_fatals"].values / pop_f * 100_000).astype(np.float32)
panel["drunk_fatals_per_100k"] = (panel["drunk_fatals"].values / pop_f * 100_000).astype(np.float32)
panel["sober_fatals_per_100k"] = (panel["sober_fatals"].values / pop_f * 100_000).astype(np.float32)

# Covariates
panel["state"]      = panel["fips"].astype(str).str[:2].astype("category")
panel["dow"]        = panel["date"].dt.dayofweek.astype(np.int8)
panel["is_weekend"] = (panel["dow"].isin([4, 5, 6])).astype(np.int8)
panel["state_yearmon"] = (panel["state"].astype(str) + "_" +
                          panel["date"].dt.to_period("M").astype(str)).astype("category")

# Merge night alerts
panel["_dk"] = panel["date"]
panel = panel.merge(
    night_alerts.rename(columns={"effective_crash_date": "_dk"})[["fips", "_dk", "night_alert"]],
    on=["fips", "_dk"], how="left"
)
panel["night_alert"] = panel["night_alert"].fillna(0).astype(np.int8)

panel = panel.merge(
    latenight_alerts.rename(columns={"effective_crash_date": "_dk"})[["fips", "_dk", "latenight_alert"]],
    on=["fips", "_dk"], how="left"
)
panel["latenight_alert"] = panel["latenight_alert"].fillna(0).astype(np.int8)
panel = panel.drop(columns=["_dk"])

# String versions of fips/date for pyfixest (avoid repeated conversion)
panel["fips"] = panel["fips"].astype(str)

del night_alerts, latenight_alerts
gc.collect()

log.info("  Night-alert county-days: %d (%.3f%%)",
         int(panel["night_alert"].sum()), 100*panel["night_alert"].mean())
log.info("  Late-night county-days:  %d (%.3f%%)",
         int(panel["latenight_alert"].sum()), 100*panel["latenight_alert"].mean())
log.info("  Memory: panel dtypes → %s", {c: str(d) for c, d in panel.dtypes.items()
                                           if c in ["total_fatals_per_100k","night_alert",
                                                    "population","fips","date"]})


# ═══════════════════════════════════════════════════════════════════════════════
# 3. National TWFE — main results
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== SECTION 1: National TWFE ===")
results_main = []
for outcome, olabel in [
    ("total_fatals_per_100k", "Total fatalities/100k"),
    ("drunk_fatals_per_100k", "Drunk fatalities/100k"),
    ("sober_fatals_per_100k", "Sober fatalities/100k"),
]:
    log.info("  Running: %s …", olabel)
    res = run_twfe(panel, outcome, olabel, vcov_spec="2way")
    if res:
        results_main.append(res)
        stars = "***" if res["p"] < 0.01 else ("**" if res["p"] < 0.05 else
                ("*"  if res["p"] < 0.10 else ""))
        log.info("  %-38s  β=%+.5f  SE=%.5f  p=%.4f%s  N=%d",
                 olabel, res["coef"], res["se"], res["p"], stars, res["N"])

df_main = pd.DataFrame(results_main)
df_main.to_csv(OUT_DIR / "fars_national_main.csv", index=False)
log.info("  → output/tables/fars_national_main.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. California validation
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== SECTION 2: California validation ===")
ca_panel = panel[panel["fips"].str.startswith("06")].copy()
gc.collect()
log.info("  CA: %d rows  %d counties  night_alerts=%d",
         len(ca_panel), ca_panel["fips"].nunique(), int(ca_panel["night_alert"].sum()))

results_ca = []
for outcome, olabel in [
    ("total_fatals_per_100k", "CA Total fatalities/100k"),
    ("drunk_fatals_per_100k", "CA Drunk fatalities/100k"),
    ("sober_fatals_per_100k", "CA Sober fatalities/100k"),
]:
    res = run_twfe(ca_panel, outcome, olabel, vcov_spec="2way")
    if res:
        results_ca.append(res)
        stars = "***" if res["p"] < 0.01 else ("**" if res["p"] < 0.05 else
                ("*"  if res["p"] < 0.10 else ""))
        log.info("  %-38s  β=%+.5f  SE=%.5f  p=%.4f%s", olabel, res["coef"], res["se"], res["p"], stars)

df_ca = pd.DataFrame(results_ca)
df_ca.to_csv(OUT_DIR / "fars_national_california.csv", index=False)
log.info("  → output/tables/fars_national_california.csv")
del ca_panel
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Weekend vs Weekday (DUI deterrence test)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== SECTION 3: Weekend vs Weekday ===")
results_wknd = []
for grp_label, mask in [
    ("Weekday (Mon–Thu)",  panel["is_weekend"] == 0),
    ("Weekend (Fri–Sun)",  panel["is_weekend"] == 1),
]:
    sub = panel[mask].copy()
    gc.collect()
    for outcome, olabel in [
        ("total_fatals_per_100k", "Total"),
        ("drunk_fatals_per_100k", "Drunk"),
        ("sober_fatals_per_100k", "Sober"),
    ]:
        res = run_twfe(sub, outcome, f"{grp_label}  {olabel}", vcov_spec="2way")
        if res:
            res["group"] = grp_label
            results_wknd.append(res)
            stars = "***" if res["p"] < 0.01 else ("**" if res["p"] < 0.05 else
                    ("*"  if res["p"] < 0.10 else ""))
            log.info("  %-45s  β=%+.5f  SE=%.5f  p=%.4f%s",
                     res["label"], res["coef"], res["se"], res["p"], stars)
    del sub
    gc.collect()

df_wknd = pd.DataFrame(results_wknd)
df_wknd.to_csv(OUT_DIR / "fars_national_weekend.csv", index=False)
log.info("  → output/tables/fars_national_weekend.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Weather robustness
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== SECTION 4: Weather Robustness ===")
results_wx = []
for spec_label, extra_fe in [
    ("Baseline (county+date FE)",          ""),
    ("+ weather_adverse (FARS field)",     "weather"),
    ("+ state×year-month FE",              "stym"),
]:
    log.info("  %s …", spec_label)
    sub = panel[["fips", "date", "year", "population", "night_alert",
                 "total_fatals_per_100k", "state_yearmon", "weather_adverse"]].copy()
    gc.collect()

    sub["_fips_c"] = sub["fips"].astype("category")
    sub["_date_c"] = sub["date"].astype(str).astype("category")
    sub["_year_c"] = sub["year"].astype(str).astype("category")
    sub["_pop"]    = sub["population"].astype(float)

    if extra_fe == "stym":
        sub["_stym_c"] = sub["state_yearmon"].astype("category")
        formula = "total_fatals_per_100k ~ night_alert | _fips_c + _date_c + _stym_c"
    elif extra_fe == "weather":
        formula = "total_fatals_per_100k ~ night_alert + weather_adverse | _fips_c + _date_c"
    else:
        formula = "total_fatals_per_100k ~ night_alert | _fips_c + _date_c"

    try:
        fit = pf.feols(fml=formula, data=sub, weights="_pop",
                       vcov={"CRV1": "_fips_c + _year_c"})
        coef = fit.coef().get("night_alert", float("nan"))
        se   = fit.se().get("night_alert",   float("nan"))
        p    = fit.pvalue().get("night_alert", float("nan"))
        n    = int(getattr(fit, "_N", len(sub)))
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log.info("  %-42s  β=%+.5f  SE=%.5f  p=%.4f%s", spec_label, coef, se, p, stars)
        results_wx.append({"spec": spec_label, "coef": coef, "se": se, "p": p, "N": n})
        del fit
    except Exception as e:
        log.warning("  %s: failed — %s", spec_label, e)
    del sub
    gc.collect()

df_wx = pd.DataFrame(results_wx)
df_wx.to_csv(OUT_DIR / "fars_national_weather.csv", index=False)
log.info("  → output/tables/fars_national_weather.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Late-night sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== SECTION 5: Late-night (0–5am) sensitivity ===")
results_ln = []
for outcome, olabel in [
    ("total_fatals_per_100k", "Total fatalities/100k"),
    ("drunk_fatals_per_100k", "Drunk fatalities/100k"),
    ("sober_fatals_per_100k", "Sober fatalities/100k"),
]:
    res = run_twfe(panel, outcome, f"Late-night: {olabel}",
                   treat_col="latenight_alert", vcov_spec="2way")
    if res:
        results_ln.append(res)
        stars = "***" if res["p"] < 0.01 else ("**" if res["p"] < 0.05 else
                ("*"  if res["p"] < 0.10 else ""))
        log.info("  %-45s  β=%+.5f  SE=%.5f  p=%.4f%s",
                 res["label"], res["coef"], res["se"], res["p"], stars)

df_ln = pd.DataFrame(results_ln)
df_ln.to_csv(OUT_DIR / "fars_national_latenight.csv", index=False)
log.info("  → output/tables/fars_national_latenight.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Summary print
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 90)
print("FARS NATIONAL ANALYSIS — RESULTS SUMMARY")
print("═" * 90)
print(f"\n{'─'*90}")
print("SECTION 1: National TWFE  (county + date FE, pop-weighted, SE clustered by county×year)")
print(f"{'─'*90}")
print(f"  {'Outcome':<40} {'β':>10} {'SE':>9} {'p':>8}  {'N':>12}")
for r in results_main:
    stars = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
    print(f"  {r['outcome']:<40} {r['coef']:>+10.5f} {r['se']:>9.5f} {r['p']:>7.4f}{stars}  {r['N']:>12,}")

print(f"\n{'─'*90}")
print("SECTION 2: California TWFE (FARS drunk/sober; compare with CCRS all-crash)")
print(f"{'─'*90}")
for r in results_ca:
    stars = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
    print(f"  {r['label']:<40} {r['coef']:>+10.5f} {r['se']:>9.5f} {r['p']:>7.4f}{stars}")

print(f"\n{'─'*90}")
print("SECTION 3: Weekend vs Weekday")
print(f"{'─'*90}")
for r in results_wknd:
    stars = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
    print(f"  {r['label']:<48} β={r['coef']:>+.5f}  SE={r['se']:.5f}  p={r['p']:.4f}{stars}")

print(f"\n{'─'*90}")
print("SECTION 4: Weather Robustness (total_fatals_per_100k)")
print(f"{'─'*90}")
for r in results_wx:
    stars = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
    print(f"  {r['spec']:<48} β={r['coef']:>+.5f}  SE={r['se']:.5f}  p={r['p']:.4f}{stars}")

print(f"\n{'─'*90}")
print("SECTION 5: Late-night (0–5am) sensitivity")
print(f"{'─'*90}")
for r in results_ln:
    stars = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
    print(f"  {r['label']:<48} β={r['coef']:>+.5f}  SE={r['se']:.5f}  p={r['p']:.4f}{stars}")

print("═" * 90)
log.info("All outputs saved to output/tables/")
