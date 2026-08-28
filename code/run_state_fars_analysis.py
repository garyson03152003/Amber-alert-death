"""
run_state_fars_analysis.py
=================================================================
Run individual-state TWFE analyses using FARS (fatality-only) data.

Covers states where no DOT all-crash panel exists yet:
  TX, MO, GA, TN, NC, MI, OH, OK, CO, WA, AZ
Plus re-validates states we already have DOT data for (for comparison):
  CA, FL, IL, NY, WI, VA, OR

Uses the existing data/processed/fars_county_day.parquet (built by
build_fars_county_day.py). Each state is processed independently so
memory usage stays modest (~0.5–1M rows per state).

Output: output/tables/fars_state_individual.csv
        output/tables/fars_state_summary.csv   (pivot: state × metric)
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
except ImportError:
    print("pyfixest not available — install with: pip install pyfixest")
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
log = logging.getLogger("state_fars")

# ── States to analyse ─────────────────────────────────────────────────────────
# Format: fips_prefix → (abbrev, name, tz_default)
# Prioritized by AMBER alert volume
TARGET_STATES = {
    "48": ("TX", "Texas"),
    "29": ("MO", "Missouri"),
    "13": ("GA", "Georgia"),
    "47": ("TN", "Tennessee"),
    "37": ("NC", "North Carolina"),
    "26": ("MI", "Michigan"),
    "39": ("OH", "Ohio"),
    "40": ("OK", "Oklahoma"),
    "08": ("CO", "Colorado"),
    "53": ("WA", "Washington"),
    "04": ("AZ", "Arizona"),
    "22": ("LA", "Louisiana"),
    # States already in DOT panel (for FARS cross-validation)
    "06": ("CA", "California"),
    "12": ("FL", "Florida"),
    "17": ("IL", "Illinois"),
    "55": ("WI", "Wisconsin"),
    "51": ("VA", "Virginia"),
    "41": ("OR", "Oregon"),
    "36": ("NY", "New York"),
}

# ── Timezone maps (same as national script) ───────────────────────────────────
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
    # Michigan UP (CT)
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
    "47153": "America/New_York", "47155": "America/New_York",
    "47163": "America/New_York", "47173": "America/New_York",
    "47179": "America/New_York",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: per-state TWFE
# ═══════════════════════════════════════════════════════════════════════════════
def run_twfe_state(panel: pd.DataFrame, outcome_col: str, label: str,
                   treat_col: str = "night_alert") -> dict | None:
    """
    Run county + date TWFE with population weights.
    SE clustered by county (CRV1).
    Returns dict with coef, se, p, N or None on failure.
    """
    cols = [outcome_col, treat_col, "population", "fips", "date", "year"]
    sub  = panel[cols].dropna(subset=[outcome_col, "population"]).copy()

    if sub[treat_col].std() < 1e-12 or len(sub) < 200:
        log.warning("  %s: no treatment variation or too few rows (%d)", label, len(sub))
        return None

    sub["_fips_c"] = sub["fips"].astype("category")
    sub["_date_c"] = sub["date"].astype(str).astype("category")
    sub["_year_c"] = sub["year"].astype(str).astype("category")
    sub["_pop"]    = sub["population"].astype(float)

    formula = f"{outcome_col} ~ {treat_col} | _fips_c + _date_c"
    vcov    = {"CRV1": "_fips_c"}

    try:
        fit   = pf.feols(fml=formula, data=sub, weights="_pop", vcov=vcov)
        coef  = fit.coef().get(treat_col, float("nan"))
        se    = fit.se().get(treat_col,   float("nan"))
        t     = fit.tstat().get(treat_col, float("nan"))
        p     = fit.pvalue().get(treat_col, float("nan"))
        n     = int(getattr(fit, "_N", len(sub)))
        del sub, fit
        gc.collect()
        return dict(label=label, N=n, coef=coef, se=se, t=t, p=p,
                    outcome=outcome_col, treat=treat_col)
    except Exception as e:
        log.warning("  %s: feols failed — %s", label, e)
        del sub
        gc.collect()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Build night_alerts treatment (county × effective_crash_date)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== Loading AMBER alerts ===")
ALERT_PATH = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2024.csv"
alerts = pd.read_csv(ALERT_PATH)
alerts = alerts[alerts["msg_type"].isin(["Alert", "Update"])].copy()

alerts["fips"]          = alerts["fips"].astype(str).str.zfill(5)
alerts["state_fips"]    = alerts["fips"].str[:2]
alerts["is_state_fips"] = alerts["fips"].str[2:] == "000"

# Keep only county-level rows (not state-broadcast)
alerts_county = alerts[~alerts["is_state_fips"]].copy()

# Timezone conversion
alerts_county["tz_name"] = (
    alerts_county["fips"].map(COUNTY_TIMEZONE_OVERRIDE)
    .fillna(alerts_county["state_fips"].map(STATE_TIMEZONE))
    .fillna("America/Chicago")
)
alerts_county["sent_utc"] = pd.to_datetime(alerts_county["sent_utc"], utc=True)
alerts_county["hour_local"] = 0
utc_series = alerts_county["sent_utc"]
sent_local_vals = pd.Series(pd.NaT, index=alerts_county.index)

for tz_name, idx in alerts_county.groupby("tz_name").groups.items():
    tz    = pytz.timezone(tz_name)
    local = utc_series.loc[idx].dt.tz_convert(tz)
    alerts_county.loc[idx, "hour_local"] = local.dt.hour.values
    sent_local_vals.loc[idx]             = local.dt.tz_localize(None).values

alerts_county["sent_local"] = pd.to_datetime(sent_local_vals)
alerts_county["is_night"]   = (
    (alerts_county["hour_local"] >= 22) | (alerts_county["hour_local"] < 6)
)
alerts_county["alert_date"] = alerts_county["sent_local"].dt.normalize()
alerts_county["effective_crash_date"] = np.where(
    alerts_county["hour_local"] >= 22,
    alerts_county["alert_date"] + pd.Timedelta(days=1),
    alerts_county["alert_date"],
)
alerts_county["effective_crash_date"] = pd.to_datetime(alerts_county["effective_crash_date"])

# Night alerts: county × crash_date
night_alerts = (
    alerts_county[alerts_county["is_night"]]
    .groupby(["fips", "effective_crash_date"])
    .size().reset_index(name="n_alerts")
)
night_alerts["night_alert"] = np.int8(1)
log.info("  Night-alert county-crash_dates: %d  (%d unique counties)  %d states",
         len(night_alerts), night_alerts["fips"].nunique(),
         night_alerts["fips"].str[:2].nunique())

del alerts, alerts_county
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load FARS and population
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== Loading FARS + population ===")
fars = pd.read_parquet(DATA_PROC / "fars_county_day.parquet")
fars["fips"] = fars["fips"].astype(str).str.zfill(5)
fars["date"] = pd.to_datetime(fars["date"])

pop = pd.read_parquet(DATA_PROC / "county_population.parquet")
pop["fips"] = pop["fips"].astype(str).str.zfill(5)

DATE_MIN = fars["date"].min()
DATE_MAX = fars["date"].max()
date_range = pd.date_range(start=DATE_MIN, end=DATE_MAX, freq="D")

log.info("  FARS: %d county-days  %d counties  %s – %s",
         len(fars), fars["fips"].nunique(), DATE_MIN.date(), DATE_MAX.date())
log.info("  Population: %d rows  %d counties", len(pop), pop["fips"].nunique())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Per-state loop
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== Running per-state TWFE ===")
all_results = []

for state_fips, (abbrev, name) in TARGET_STATES.items():
    log.info("── %s (%s) ──", abbrev, name)

    # ── Filter FARS to state ──
    s_fars = fars[fars["fips"].str.startswith(state_fips)].copy()
    if s_fars.empty:
        log.warning("  %s: no FARS data — skip", abbrev)
        continue

    s_fips_list = sorted(s_fars["fips"].unique())
    log.info("  FARS rows: %d  counties: %d  fatals: %d",
             len(s_fars), len(s_fips_list), int(s_fars["total_fatals"].sum()))

    # ── Build state panel (all county-days including zeros) ──
    idx = pd.MultiIndex.from_product([s_fips_list, date_range], names=["fips", "date"])
    panel = pd.DataFrame(index=idx).reset_index()

    panel = panel.merge(
        s_fars[["fips", "date", "total_fatals", "drunk_fatals", "sober_fatals"]],
        on=["fips", "date"], how="left"
    )
    for col in ["total_fatals", "drunk_fatals", "sober_fatals"]:
        panel[col] = panel[col].fillna(0).astype(np.int16)

    # ── Merge population ──
    s_pop = pop[pop["fips"].str.startswith(state_fips)].copy()
    if s_pop.empty:
        log.warning("  %s: no population data — skip", abbrev)
        del panel, s_fars, s_pop
        gc.collect()
        continue

    panel = panel.merge(
        s_pop[["fips", "year", "population"]],
        left_on=["fips", panel["date"].dt.year.rename("year")],
        right_on=["fips", "year"],
        how="left"
    )
    panel["population"] = panel["population"].astype("float32")

    # Drop rows without population
    n_before = len(panel)
    panel = panel.dropna(subset=["population"])
    if len(panel) < n_before * 0.5:
        log.warning("  %s: lost >50%% rows on pop merge (%d → %d)", abbrev, n_before, len(panel))

    # ── Merge treatment ──
    s_night = night_alerts[night_alerts["fips"].str.startswith(state_fips)].copy()
    s_night = s_night.rename(columns={"effective_crash_date": "date"})
    panel = panel.merge(s_night[["fips", "date", "night_alert"]], on=["fips", "date"], how="left")
    panel["night_alert"] = panel["night_alert"].fillna(0).astype(np.int8)

    n_treated = int(panel["night_alert"].sum())
    log.info("  Panel: %d rows  treated: %d  (%.3f%%)",
             len(panel), n_treated, 100 * n_treated / max(len(panel), 1))

    if n_treated < 10:
        log.warning("  %s: only %d treated county-days — skip", abbrev, n_treated)
        del panel, s_fars, s_pop, s_night
        gc.collect()
        continue

    # ── Compute per-100k outcomes ──
    pop_f = panel["population"].values
    pop_f = np.where(pop_f < 1, np.nan, pop_f)

    panel["total_fatals_per_100k"] = (panel["total_fatals"].values / pop_f * 100_000).astype("float32")
    panel["drunk_fatals_per_100k"] = (panel["drunk_fatals"].values / pop_f * 100_000).astype("float32")
    panel["sober_fatals_per_100k"] = (panel["sober_fatals"].values / pop_f * 100_000).astype("float32")

    # Year column for clustering
    panel["year"] = panel["date"].dt.year.astype("int16")

    del s_fars, s_pop, s_night
    gc.collect()

    # ── Run TWFE for three outcomes ──
    for outcome, olabel in [
        ("total_fatals_per_100k", "Total fatals/100k"),
        ("drunk_fatals_per_100k", "Drunk fatals/100k"),
        ("sober_fatals_per_100k", "Sober fatals/100k"),
    ]:
        label = f"{abbrev} {olabel}"
        log.info("  Running: %s …", label)
        res = run_twfe_state(panel, outcome, label)
        if res:
            res["state_fips"] = state_fips
            res["state_abbrev"] = abbrev
            res["state_name"]   = name
            res["n_treated_cd"] = n_treated
            all_results.append(res)
            log.info("  %-40s β=%+.5f  SE=%.5f  p=%.4f  N=%d",
                     label, res["coef"], res["se"], res["p"], res["N"])

    del panel
    gc.collect()
    log.info("")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Save results
# ═══════════════════════════════════════════════════════════════════════════════
if not all_results:
    log.error("No results produced.")
    sys.exit(1)

results_df = pd.DataFrame(all_results)

# ── Long-format table ──
out_long = OUT_DIR / "fars_state_individual.csv"
results_df.to_csv(out_long, index=False)
log.info("→ %s", out_long)

# ── Pivot to state × outcome summary ──
pivot = results_df.pivot_table(
    index=["state_fips", "state_abbrev", "state_name", "N", "n_treated_cd"],
    columns="outcome",
    values=["coef", "se", "p"],
)
pivot.columns = ["_".join(c) for c in pivot.columns]
pivot = pivot.reset_index()
pivot = pivot.sort_values("n_treated_cd", ascending=False)

out_pivot = OUT_DIR / "fars_state_summary.csv"
pivot.to_csv(out_pivot, index=False)
log.info("→ %s", out_pivot)

# ── Console summary ──
print()
print("=" * 90)
print("INDIVIDUAL STATE FARS RESULTS  (TWFE: county + date FE, pop-weighted, SE clustered county)")
print("=" * 90)
print(f"{'State':<6}  {'Treated CDs':>11}  {'Total β':>10}  {'p':>7}  {'Drunk β':>10}  {'p':>7}  {'Sober β':>10}  {'p':>7}")
print("-" * 90)

for _, row in pivot.sort_values("n_treated_cd", ascending=False).iterrows():
    print(f"{row['state_abbrev']:<6}  {int(row['n_treated_cd']):>11,d}  "
          f"{row.get('coef_total_fatals_per_100k', float('nan')):>+10.5f}  "
          f"{row.get('p_total_fatals_per_100k', float('nan')):>7.4f}  "
          f"{row.get('coef_drunk_fatals_per_100k', float('nan')):>+10.5f}  "
          f"{row.get('p_drunk_fatals_per_100k', float('nan')):>7.4f}  "
          f"{row.get('coef_sober_fatals_per_100k', float('nan')):>+10.5f}  "
          f"{row.get('p_sober_fatals_per_100k', float('nan')):>7.4f}")

print("=" * 90)
log.info("Done.")
