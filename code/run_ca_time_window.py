"""
run_ca_time_window.py
============================================================
Time-window × sleep-phase analysis for California.

Uses county-HOUR crash data (california_ccrs_county_hour.parquet) to
determine WHEN during the effective crash day the negative effect operates:

  W0  pre-dawn     effective_crash_date hours  0–5   (same night / very early morning)
  W1  commute      effective_crash_date hours  6–10  (morning commute)
  W2  midday       effective_crash_date hours 11–15  (control)
  W3  evening      effective_crash_date hours 16–22  (control)

Cross-tabulated with alert sleep phase:
  ph_2223  22–23h  Still awake / falling asleep
  ph_0001   0– 1h  Light sleep  (N1/N2)
  ph_0203   2– 3h  Deep sleep   (N3 — peak effect in daily analysis)
  ph_0405   4– 5h  Late / REM

If the mechanism is morning-commute impairment (sleep disruption or
behavioral lockdown), the effect should concentrate in W1 (commute).
If it is same-night driving suppression, the effect should concentrate
in W0 (pre-dawn) and be strongest for the high-traffic 22–23h window.

Output:
  output/tables/ca_time_window_results.csv
  (logged summary table)
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import pyfixest as pf
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, DATA_PROC, OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("ca_time_window")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

# ── 1. Load CA county-hour crash data ─────────────────────────────────────────
log.info("=== Loading CA CCRS county-hour data ===")
HOURLY_PATH = DATA_PROC / "california_ccrs_county_hour.parquet"
if not HOURLY_PATH.exists():
    log.error("Run build_ca_ccrs_hourly.py first.")
    sys.exit(1)

hourly = pd.read_parquet(HOURLY_PATH)
hourly["date"] = pd.to_datetime(hourly["date"])
log.info("  %d county-hour rows, %d counties, %s–%s",
         len(hourly), hourly.fips.nunique(),
         hourly.date.min().date(), hourly.date.max().date())

# ── 2. Build crash-window columns per county-day ───────────────────────────────
# Crash time windows on the effective crash date:
WINDOWS = [
    ("W0_predawn",  0,  5,  "Pre-dawn    (0–5h)"),
    ("W1_commute",  6, 10,  "Commute     (6–10h)"),
    ("W2_midday",  11, 15,  "Midday      (11–15h)"),
    ("W3_evening", 16, 22,  "Evening     (16–22h)"),
]
WIN_COLS   = [w[0] for w in WINDOWS]
WIN_LABELS = {w[0]: w[3] for w in WINDOWS}

log.info("Aggregating to county-day crash windows …")
window_dfs = {}
for wname, h_lo, h_hi, _ in WINDOWS:
    mask = hourly["hour"].between(h_lo, h_hi)
    agg = (hourly[mask]
           .groupby(["fips", "date"])["ca_crashes"]
           .sum()
           .reset_index()
           .rename(columns={"ca_crashes": wname}))
    window_dfs[wname] = agg
    log.info("  %s: %.0f crashes across %d county-days",
             wname, agg[wname].sum(), len(agg))

# Full county-day grid from the hourly data
all_dates = hourly["date"].unique()
all_fips  = hourly["fips"].unique()

# Start panel from hourly aggregated to county-day (total crashes for merge check)
panel = (hourly.groupby(["fips", "date"])["ca_crashes"]
               .sum()
               .reset_index()
               .rename(columns={"ca_crashes": "total_crashes"}))

# Merge each window
for wname, agg in window_dfs.items():
    panel = panel.merge(agg, on=["fips", "date"], how="left")
    panel[wname] = panel[wname].fillna(0)

log.info("Panel shape after window merge: %s", panel.shape)

# ── 3. Load population ────────────────────────────────────────────────────────
log.info("=== Loading population ===")
pop = pd.read_parquet(DATA_PROC / "county_population.parquet")
pop = pop[pop["fips"].str.startswith("06")]   # CA only
panel["year"] = panel["date"].dt.year
panel = panel.merge(pop[["fips", "year", "population"]], on=["fips", "year"], how="left")
panel = panel.dropna(subset=["population"])

# Per-100k rates for each window
for wname in WIN_COLS:
    panel[f"{wname}_100k"] = 100_000 * panel[wname] / panel["population"]

# ── 4. Load AMBER alerts and apply DST-aware CA timezone conversion ────────────
log.info("=== Loading AMBER alerts ===")
AMBER_PATH = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2024.csv"
alerts = pd.read_csv(AMBER_PATH, low_memory=False)

# Keep only CA alerts (state_fips = 6)
alerts["state_fips"] = alerts["state_fips"].astype(str).str.zfill(2)
alerts = alerts[alerts["state_fips"] == "06"].copy()
log.info("  CA alerts: %d", len(alerts))

# DST-aware UTC → Pacific time (CA is always America/Los_Angeles)
alerts["sent_utc"]  = pd.to_datetime(alerts["sent_utc"], utc=True)
tz_la               = pytz.timezone("America/Los_Angeles")
local               = alerts["sent_utc"].dt.tz_convert(tz_la)
alerts["hour_local"]= local.dt.hour
alerts["sent_local"]= local.dt.tz_localize(None)
alerts["alert_date"]= alerts["sent_local"].dt.normalize()

# Night filter: 22h–5h local
alerts["is_night"]  = (alerts["hour_local"] >= 22) | (alerts["hour_local"] < 6)

# Effective crash date (same logic as run_state_dot_analysis.py)
alerts["effective_crash_date"] = np.where(
    alerts["hour_local"] >= 22,
    alerts["alert_date"] + pd.Timedelta(days=1),
    alerts["alert_date"],
)

log.info("  Night alerts: %d (%.1f%%)",
         alerts["is_night"].sum(), 100 * alerts["is_night"].mean())

# ── 5. Build sleep-phase indicators ───────────────────────────────────────────
SLEEP_PHASES = [
    ("ph_2223", 22, 23, "Still awake  (22–23h)"),
    ("ph_0001",  0,  1, "Light sleep  (0–1h)"),
    ("ph_0203",  2,  3, "Deep sleep   (2–3h)"),
    ("ph_0405",  4,  5, "Late/REM     (4–5h)"),
]
PHASE_COLS   = [p[0] for p in SLEEP_PHASES]
PHASE_LABELS = {p[0]: p[3] for p in SLEEP_PHASES}

# Overall night alert indicator
night_alerts = (
    alerts[alerts.is_night]
    .groupby(["fips", "effective_crash_date"])
    .size().reset_index(name="n_alerts")
)
night_alerts["fips"]        = night_alerts["fips"].astype(str).str.zfill(5)
night_alerts["night_alert"] = 1

panel["fips"] = panel["fips"].astype(str).str.zfill(5)
panel = panel.merge(
    night_alerts[["fips", "effective_crash_date", "night_alert"]],
    left_on=["fips", "date"], right_on=["fips", "effective_crash_date"],
    how="left"
)
panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)

# Per-phase indicators
for ph_name, h_lo, h_hi, ph_label in SLEEP_PHASES:
    ph = alerts[alerts["is_night"] & alerts["hour_local"].between(h_lo, h_hi)]
    collapsed = (ph.groupby(["fips", "effective_crash_date"])
                   .size().reset_index(name=f"n_{ph_name}"))
    collapsed["fips"]   = collapsed["fips"].astype(str).str.zfill(5)
    collapsed[ph_name]  = 1
    panel = panel.merge(
        collapsed[["fips", "effective_crash_date", ph_name]],
        left_on=["fips", "date"], right_on=["fips", "effective_crash_date"],
        how="left", suffixes=("", f"_{ph_name}")
    )
    panel[ph_name] = panel[ph_name].fillna(0).astype(int)
    dup = f"effective_crash_date_{ph_name}"
    if dup in panel.columns:
        panel = panel.drop(columns=[dup])
    log.info("  Phase %-28s %d treated county-days in CA panel",
             ph_label, panel[ph_name].sum())

# Restrict to study years overlapping alert data
panel = panel[panel["year"].between(2016, 2022)].copy()
panel["dow"] = panel["date"].dt.dayofweek
panel["fips"] = panel["fips"].astype(str)

log.info("Final panel: %d rows  treated nights: %d",
         len(panel), panel["night_alert"].sum())

# ── 6. TWFE regression function (population-weighted) ─────────────────────────
def run_twfe_window(sub2: pd.DataFrame,
                    outcome_col: str,
                    treatment_col: str,
                    label: str) -> dict | None:
    """
    Population-weighted WLS TWFE via pyfixest.feols (county+date FE+DoW).
    Weights = county population so large counties (LA, SF) dominate.
    """
    sub2 = sub2.dropna(subset=[outcome_col, "population"]).copy()
    if len(sub2) < 50 or sub2[treatment_col].std() < 1e-12:
        return None

    sub2["_fips_str"] = sub2["fips"].astype(str)
    sub2["_date_str"] = sub2["date"].astype(str)
    sub2["_pop"]      = sub2["population"].astype(float)
    sub2["_dow_str"]  = "dow" + sub2["dow"].astype(str)

    formula = f"{outcome_col} ~ {treatment_col} + C(_dow_str) | _fips_str + _date_str"
    try:
        fit = pf.feols(formula, data=sub2, weights="_pop",
                       vcov={"CRV1": "_fips_str"})
        tbl = fit.tidy()
        if treatment_col not in tbl.index:
            return None
        return dict(
            beta     = round(float(tbl.loc[treatment_col, "Estimate"]), 6),
            se       = round(float(tbl.loc[treatment_col, "Std. Error"]), 6),
            pvalue   = round(float(tbl.loc[treatment_col, "Pr(>|t|)"]), 4),
            n_obs    = int(fit._N),
            n_treated= int(sub2[treatment_col].sum()),
        )
    except Exception as e:
        log.warning("  feols failed [%s/%s/%s]: %s", label, outcome_col, treatment_col, e)
        return None


# ── 7. Run regressions: each sleep-phase × each crash-window ──────────────────
log.info("\n=== Time-window × Sleep-phase regressions (CA) ===")
log.info("Outcome: crashes per 100k in each time window of the effective crash date")
log.info("")

results_rows = []

# A: Overall night alert × each window (baseline — total night, no phase split)
log.info("─── Overall night alert (all phases pooled) ───")
for wname, _, _, wlabel in WINDOWS:
    out_col = f"{wname}_100k"
    res = run_twfe_window(panel, out_col, "night_alert", "CA-overall")
    if res:
        stars = ("***" if res["pvalue"] < 0.01 else
                 "**"  if res["pvalue"] < 0.05 else
                 "*"   if res["pvalue"] < 0.10 else "")
        log.info("  %-28s  β=%+.4f  SE=%.4f  p=%.3f %-3s  N_treated=%d",
                 wlabel, res["beta"], res["se"], res["pvalue"], stars, res["n_treated"])
        res.update({"treatment": "night_alert", "phase": "ALL",
                    "phase_label": "All night (pooled)",
                    "window": wname, "window_label": wlabel})
        results_rows.append(res)

# B: Joint regression — all 4 phases simultaneously, for each window
log.info("")
log.info("─── Per sleep-phase, per crash-window (joint TWFE) ───")

def run_joint_phase_window(sub2, outcome_col, label):
    """Population-weighted TWFE with all 4 phase indicators simultaneously."""
    active = [c for c in PHASE_COLS
              if c in sub2.columns and sub2[c].sum() >= 3]
    if not active:
        return None
    sub2 = sub2.dropna(subset=[outcome_col, "population"]).copy()

    sub2["_fips_str"] = sub2["fips"].astype(str)
    sub2["_date_str"] = sub2["date"].astype(str)
    sub2["_pop"]      = sub2["population"].astype(float)
    sub2["_dow_str"]  = "dow" + sub2["dow"].astype(str)

    rhs     = " + ".join(active) + " + C(_dow_str)"
    formula = f"{outcome_col} ~ {rhs} | _fips_str + _date_str"
    try:
        fit  = pf.feols(formula, data=sub2, weights="_pop",
                        vcov={"CRV1": "_fips_str"})
        tbl  = fit.tidy()
        rows = []
        for ph in active:
            if ph not in tbl.index:
                continue
            rows.append(dict(
                phase       = ph,
                phase_label = PHASE_LABELS[ph],
                beta        = round(float(tbl.loc[ph, "Estimate"]), 6),
                se          = round(float(tbl.loc[ph, "Std. Error"]), 6),
                pvalue      = round(float(tbl.loc[ph, "Pr(>|t|)"]), 4),
                n_obs       = int(fit._N),
                n_treated   = int(sub2[ph].sum()),
                treatment   = "sleep_phase_joint",
            ))
        return rows or None
    except Exception as e:
        log.warning("  Joint feols failed [%s]: %s — skipping", label, e)
        return None

for wname, _, _, wlabel in WINDOWS:
    out_col = f"{wname}_100k"
    log.info("  Window: %s", wlabel)
    phase_res = run_joint_phase_window(panel, out_col, wname)
    if phase_res:
        for row in phase_res:
            stars = ("***" if row["pvalue"] < 0.01 else
                     "**"  if row["pvalue"] < 0.05 else
                     "*"   if row["pvalue"] < 0.10 else "")
            log.info("    %-28s  β=%+.4f  SE=%.4f  p=%.3f %-3s  N=%d",
                     row["phase_label"], row["beta"], row["se"],
                     row["pvalue"], stars, row["n_treated"])
            row["window"]       = wname
            row["window_label"] = wlabel
            results_rows.append(row)
    log.info("")

# ── 8. Summary matrix ─────────────────────────────────────────────────────────
results = pd.DataFrame(results_rows)
phase_results = results[results.treatment == "sleep_phase_joint"].copy()

log.info("=== SUMMARY: β matrix (crashes/100k) ===")
log.info("Rows = crash-time window; Cols = alert sleep phase")
log.info("(*** p<0.01, ** p<0.05, * p<0.10)")
log.info("")

# Wide table: window × phase
if not phase_results.empty:
    def fmt(sub, wname, ph):
        r = sub[(sub.window == wname) & (sub.phase == ph)]
        if r.empty:
            return "  —  "
        b = r.iloc[0]["beta"]; p = r.iloc[0]["pvalue"]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.3f}{stars}"

    phases_order = ["ph_2223", "ph_0001", "ph_0203", "ph_0405"]
    hdr = f"  {'Window':<22}" + "".join(f"  {PHASE_LABELS[p]:<22}" for p in phases_order)
    log.info(hdr)
    log.info("  " + "-" * (22 + 26 * len(phases_order)))
    for wname, _, _, wlabel in WINDOWS:
        row_str = f"  {wlabel:<22}"
        for ph in phases_order:
            row_str += f"  {fmt(phase_results, wname, ph):<22}"
        log.info(row_str)

results.to_csv(OUTPUT_TABS / "ca_time_window_results.csv", index=False)
log.info("\nSaved → %s", OUTPUT_TABS / "ca_time_window_results.csv")
