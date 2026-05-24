"""
run_time_window_analysis.py
=============================================================
Time-window analysis of AMBER Alert effects on traffic fatalities.

Mechanism tests
---------------
Two competing hypotheses for WHY Amber Alerts increase crashes:

  H1 (Immediate disruption): alert fires while people are still driving,
     distracting them → crash same night (within ~6 hours of alert)

  H2 (Sleep disruption): alert wakes people up at night, reducing sleep
     quality → they are impaired on the following morning commute
     → crashes 6–12 hours after a night alert

These are empirically distinguishable by looking at WHEN crashes occur
relative to the alert night.

Time windows (relative to panel date D = night of Amber Alert)
--------------------------------------------------------------
  W0  same night:    D 20:00 – D+1 05:59   (immediate disruption zone)
  W1  morning comm:  D+1 06:00 – 09:59     (peak sleep-disruption window)
  W2  midday ctrl:   D+1 10:00 – 15:59     (control: fatigue mostly resolved)
  W3  evening ctrl:  D+1 16:00 – 19:59     (control: further from alert)
  W4  placebo:       D+2 06:00 – 09:59     (same commute window, 24h later)

Additionally, serious injuries (INJ_SEV = 3 in FARS Person.CSV) are extracted
to test whether effects extend beyond fatal crashes.

Data: FARS 2013–2023 raw ZIPs (already on disk)
  - accident.CSV: STATE, COUNTY, YEAR, MONTH, DAY, HOUR, FATALS
  - Person.CSV: STATE, COUNTY, ST_CASE, INJ_SEV (serious injury count)

All regressions: OLS TWFE, county + year FE, state-clustered SE.

Output: output/tables/reg_time_window.csv
         data/processed/fars_hourly.parquet  (cached hourly crash panel)
"""
import sys, warnings, importlib.util, gc, os, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("time_window")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

FARS_DIR     = DATA_RAW / "fars"
HOURLY_CACHE = DATA_PROC / "fars_hourly.parquet"

# ── Step 1: Build or load hourly FARS crash panel ─────────────────────────────
def load_fars_year(yr: int) -> pd.DataFrame:
    """Load accident.CSV and Person.CSV for one FARS year, return tidy rows."""
    zp = FARS_DIR / f"FARS{yr}NationalCSV.zip"
    if not zp.exists():
        log.warning("  Missing %s — skipping", zp.name)
        return pd.DataFrame()

    rows = []
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()

        # ── Accident file ──────────────────────────────────────────────────────
        acc_name = next((f for f in names if "accident" in f.lower()), None)
        if not acc_name:
            log.warning("  No accident file in %s", zp.name)
            return pd.DataFrame()

        with z.open(acc_name) as f:
            acc = pd.read_csv(f, encoding="latin1", low_memory=False)

        # Normalise BOM-prefixed column names (appears in 2021–2022)
        acc.columns = [c.replace("ï»¿", "").strip() for c in acc.columns]

        # Keep rows with valid hour (0–23) and county (1–999 — exclude 0/998/999=unknown)
        acc = acc[
            acc["HOUR"].between(0, 23) &
            acc["COUNTY"].between(1, 997)
        ].copy()

        acc["fips"] = (acc["STATE"].astype(str).str.zfill(2) +
                       acc["COUNTY"].astype(str).str.zfill(3))
        acc["crash_date"] = pd.to_datetime(
            dict(year=acc["YEAR"], month=acc["MONTH"], day=acc["DAY"]),
            errors="coerce"
        )
        acc = acc.dropna(subset=["crash_date"])
        acc_slim = acc[["ST_CASE", "fips", "crash_date", "HOUR", "FATALS"]].copy()

        # ── Person file (serious injuries, INJ_SEV == 3) ──────────────────────
        per_name = next((f for f in names if "person" in f.lower()), None)
        serious_counts = None
        if per_name:
            with z.open(per_name) as f:
                per = pd.read_csv(
                    f, encoding="latin1", low_memory=False,
                    usecols=lambda c: c.replace("ï»¿", "").strip() in
                             ["STATE", "COUNTY", "ST_CASE", "INJ_SEV"]
                )
            per.columns = [c.replace("ï»¿", "").strip() for c in per.columns]
            # INJ_SEV == 3 → "Suspected Serious Injury (A)"
            serious = (per[per["INJ_SEV"] == 3]
                       .groupby("ST_CASE")
                       .size()
                       .reset_index(name="serious_inj"))
            serious_counts = serious

        # Merge person injuries back to accident level
        if serious_counts is not None:
            acc_slim = acc_slim.merge(serious_counts, on="ST_CASE", how="left")
            acc_slim["serious_inj"] = acc_slim["serious_inj"].fillna(0).astype(int)
        else:
            acc_slim["serious_inj"] = 0

    return acc_slim

if HOURLY_CACHE.exists():
    log.info("Loading cached hourly FARS data from %s …", HOURLY_CACHE)
    crashes = pd.read_parquet(HOURLY_CACHE)
else:
    log.info("Building hourly FARS crash panel from raw ZIPs …")
    parts = []
    for yr in range(2013, 2024):
        log.info("  Loading %d …", yr)
        part = load_fars_year(yr)
        if not part.empty:
            parts.append(part)
    crashes = pd.concat(parts, ignore_index=True)
    crashes.to_parquet(HOURLY_CACHE, index=False)
    log.info("Saved hourly FARS cache → %s  (%d rows)", HOURLY_CACHE, len(crashes))

log.info("Hourly FARS: %d crash records, %d unique counties",
         len(crashes), crashes["fips"].nunique())

# ── Step 2: Assign each crash to panel windows ────────────────────────────────
#
# For a panel row with date = D (alert night):
#   W0 same night:   crash_date=D  AND hour 20–23
#                  OR crash_date=D+1 AND hour 0–5
#   W1 morning:      crash_date=D+1 AND hour 6–9
#   W2 midday:       crash_date=D+1 AND hour 10–15
#   W3 evening:      crash_date=D+1 AND hour 16–19
#   W4 placebo:      crash_date=D+2 AND hour 6–9
#
# We build separate aggregations for each window using date offsets.

def agg_window(crashes, date_offset: int, hour_lo: int, hour_hi: int,
               col_prefix: str) -> pd.DataFrame:
    """
    Sum FATALS and serious_inj for crashes in [hour_lo, hour_hi] on
    (panel_date + date_offset).  Returns county × panel_date frame.
    """
    sub = crashes[crashes["HOUR"].between(hour_lo, hour_hi)].copy()
    # panel_date = crash_date - date_offset
    sub["panel_date"] = sub["crash_date"] - pd.Timedelta(days=date_offset)
    agg = (sub.groupby(["fips", "panel_date"])
               .agg(fatals=("FATALS", "sum"),
                    serious=("serious_inj", "sum"))
               .reset_index()
               .rename(columns={"panel_date": "date",
                                 "fatals":     f"{col_prefix}_fatals",
                                 "serious":    f"{col_prefix}_serious"}))
    return agg

log.info("Building time-window aggregations …")

# W0: same-night crashes (8pm–midnight on D plus midnight–6am on D+1)
w0a = agg_window(crashes, date_offset=0, hour_lo=20, hour_hi=23, col_prefix="w0a")
w0b = agg_window(crashes, date_offset=1, hour_lo=0,  hour_hi=5,  col_prefix="w0b")

# W1 morning commute: D+1 6am–10am
w1  = agg_window(crashes, date_offset=1, hour_lo=6,  hour_hi=9,  col_prefix="w1")
# W2 midday control: D+1 10am–4pm
w2  = agg_window(crashes, date_offset=1, hour_lo=10, hour_hi=15, col_prefix="w2")
# W3 evening control: D+1 4pm–8pm
w3  = agg_window(crashes, date_offset=1, hour_lo=16, hour_hi=19, col_prefix="w3")
# W4 placebo: D+2 6am–10am (same morning window 24h later)
w4  = agg_window(crashes, date_offset=2, hour_lo=6,  hour_hi=9,  col_prefix="w4")

# ── Step 3: Load main panel and merge windows ─────────────────────────────────
log.info("Loading main panel …")
spec_mod = importlib.util.spec_from_file_location("a05", Path(__file__).parent / "05_analysis.py")
a05 = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(a05)

df = a05.load_panel()
df = prep_panel(df)
df.sort_values(["fips", "date"], inplace=True)
df["fips"]       = df["fips"].astype(str)
df["state_code"] = df["state_code"].astype(str)
df["year_str"]   = pd.to_datetime(df["date"]).dt.year.astype(str)
df["date"]       = pd.to_datetime(df["date"])

log.info("Panel: %d rows, %d counties", len(df), df["fips"].nunique())

for win_df in [w0a, w0b, w1, w2, w3, w4]:
    win_df["date"] = pd.to_datetime(win_df["date"])
    df = df.merge(win_df, on=["fips", "date"], how="left")

# Combine W0a and W0b into a single same-night window
for sfx in ["fatals", "serious"]:
    df[f"w0_{sfx}"] = df[f"w0a_{sfx}"].fillna(0) + df[f"w0b_{sfx}"].fillna(0)
    for col in [f"w0a_{sfx}", f"w0b_{sfx}", f"w1_{sfx}",
                f"w2_{sfx}", f"w3_{sfx}", f"w4_{sfx}"]:
        df[col] = df[col].fillna(0)

log.info("Window crash counts added. Sample treated row:")
tr = df[df["night_alert"] > 0].iloc[0]
log.info("  date=%s  w0_fatals=%.0f  w1_fatals=%.0f  w2_fatals=%.0f  w3_fatals=%.0f  w4_fatals=%.0f",
         tr["date"], tr["w0_fatals"], tr["w1_fatals"],
         tr["w2_fatals"], tr["w3_fatals"], tr["w4_fatals"])

# ── Step 4: Regressions ───────────────────────────────────────────────────────
WEATHER    = [c for c in ["prcp_mm", "tmax_c"] if c in df.columns and df[c].notna().mean() > 0.01]
HOL        = [c for c in ["is_holiday"] if c in df.columns]
ctrl_parts = HOL + WEATHER
CTRL_STR   = " + ".join(ctrl_parts) if ctrl_parts else "1"

results = []

def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."

def run(label, outcome, treat, data, spec_tag):
    log.info("  %s …", label)
    sub = data.dropna(subset=[treat, outcome]).copy()
    sub = sub[sub[outcome] >= 0]
    formula = f"{outcome} ~ {treat} + {CTRL_STR} | fips + year_str"
    try:
        fit = pf.feols(formula, data=sub,
                       vcov={"CRV1": "state_code"}, lean=True)
        td   = fit.tidy()
        row  = (td.loc[treat] if treat in td.index
                else td.loc[[i for i in td.index if treat in i][0]])
        coef = float(row["Estimate"])
        se   = float(row["Std. Error"])
        pval = float(row["Pr(>|t|)"])
        nobs = int(getattr(fit, "_N", None) or 0)
        log.info("  %-60s β=%+.5f  se=%.5f  p=%.3f  n=%d  %s",
                 label, coef, se, pval, nobs, _sig(pval))
        results.append({"label": label, "outcome": outcome, "spec": spec_tag,
                        "treatment": treat,
                        "coef": coef, "se": se, "pval": pval, "nobs": nobs})
        del fit, sub; gc.collect()
    except Exception as e:
        log.warning("  %s FAILED: %s", label, e)
        del sub; gc.collect()

TREAT = "night_alert"

log.info("\n=== Time-Window Regressions: Fatal Crashes ===")
log.info("Treatment: night_alert  |  FE: county + year  |  SE: state-clustered")
log.info("")
run("W0 Same-night fatals     (D 20:00–D+1 06:00)",
    "w0_fatals", TREAT, df, "w0_fatal")
run("W1 Morning-commute fatals (D+1 06:00–10:00)  [H2: sleep disruption]",
    "w1_fatals", TREAT, df, "w1_fatal")
run("W2 Midday control fatals  (D+1 10:00–16:00)",
    "w2_fatals", TREAT, df, "w2_fatal")
run("W3 Evening control fatals (D+1 16:00–20:00)",
    "w3_fatals", TREAT, df, "w3_fatal")
run("W4 Placebo: D+2 morning  (D+2 06:00–10:00)",
    "w4_fatals", TREAT, df, "w4_fatal")
gc.collect()

log.info("\n=== Time-Window Regressions: Serious Injuries (in fatal crashes) ===")
run("W0 Same-night serious inj (D 20:00–D+1 06:00)",
    "w0_serious", TREAT, df, "w0_serious")
run("W1 Morning-commute serious (D+1 06:00–10:00) [H2: sleep disruption]",
    "w1_serious", TREAT, df, "w1_serious")
run("W2 Midday control serious  (D+1 10:00–16:00)",
    "w2_serious", TREAT, df, "w2_serious")
run("W3 Evening control serious (D+1 16:00–20:00)",
    "w3_serious", TREAT, df, "w3_serious")
run("W4 Placebo: D+2 morning   (D+2 06:00–10:00)",
    "w4_serious", TREAT, df, "w4_serious")
gc.collect()

# ── Baseline daily for reference ──────────────────────────────────────────────
log.info("\n=== Baseline: Full-day fatals (fatals_t1, standard spec) ===")
if "fatals_t1" in df.columns:
    run("Baseline fatals_t1 (full next day, all hours)",
        "fatals_t1", TREAT, df, "baseline_t1")
if "fatals_next_commute" not in df.columns and hasattr(a05, "add_aligned_outcome"):
    df = a05.add_aligned_outcome(df)
if "fatals_next_commute" in df.columns:
    run("Baseline fatals_next_commute (timing-aligned)",
        "fatals_next_commute", TREAT, df, "baseline_nextcomm")

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("\n=== Summary: Time-Window Effects ===")
log.info("%-62s  %8s  %6s  %6s  %6s", "Label", "β", "se", "p", "sig")
log.info("-" * 90)
fatal_rows = [r for r in results if "serious" not in r["outcome"]]
serious_rows = [r for r in results if "serious" in r["outcome"]]

log.info("--- Fatal crashes ---")
for r in fatal_rows:
    log.info("  %-60s  %+8.5f  %.5f  %.3f  %s",
             r["label"], r["coef"], r["se"], r["pval"], _sig(r["pval"]))

log.info("--- Serious injuries (within fatal crashes) ---")
for r in serious_rows:
    log.info("  %-60s  %+8.5f  %.5f  %.3f  %s",
             r["label"], r["coef"], r["se"], r["pval"], _sig(r["pval"]))

log.info("\nInterpretation guide:")
log.info("  H1 (immediate disruption): W0 significant, W1 not → disruption while driving")
log.info("  H2 (sleep disruption):     W1 significant, W0 not → impairment next morning")
log.info("  Both H1+H2:                W0 and W1 both significant")
log.info("  Neither:                   no time-window effect (consistent with null)")
log.info("  Placebo check (W4):        should be null if effect is real")

# ── Save ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = OUTPUT_TABS / "reg_time_window.csv"
out.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
