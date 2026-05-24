"""
run_weather_robustness.py
Runs the baseline regressions with and without PRISM weather controls
and saves a side-by-side comparison table.

Weather controls: prcp_mm (daily precipitation, mm) and tmax_c (daily
max temperature, °C) at county centroid from the ACIS/PRISM 4-km grid.

Specs compared:
  (A) Baseline          — county + DoW×Month FE, no weather
  (B) + Weather         — same + prcp_mm + tmax_c
  (C) TWFE2             — county×year FE + lag(fatals_tm1), no weather
  (D) TWFE2 + Weather   — same + prcp_mm + tmax_c
  (E) WLS rate          — combined/100k, log-pop WLS, no weather
  (F) WLS rate + Wx     — same + prcp_mm + tmax_c

Output: output/tables/reg_weather_robustness.csv
"""
import sys, warnings, importlib.util, gc
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel, fe_ols_from_panel

warnings.filterwarnings("ignore")
log = get_logger("run_weather_robust")

WEATHER_PATH = DATA_PROC / "weather_county_day.parquet"

# ── Load analysis module ────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("a05", Path(__file__).parent / "05_analysis.py")
a05  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

# ── Check weather availability ──────────────────────────────────────────────
if not WEATHER_PATH.exists():
    log.error("Weather file not found: %s", WEATHER_PATH)
    log.error("Run code/01c_fetch_weather.py first.")
    sys.exit(1)

wx_counties = pd.read_parquet(WEATHER_PATH, columns=["fips"])["fips"].nunique()
log.info("Weather file covers %d counties", wx_counties)

# ── Load panel (load_panel already merges weather if file exists) ───────────
log.info("Loading panel (with PRISM weather)…")
df = a05.load_panel()
df = prep_panel(df)

avail_wx  = [c for c in ["prcp_mm","tmax_c"] if df[c].notna().mean() > 0.01]
avail_hol = [c for c in ["is_holiday"] if c in df.columns]
log.info("Available weather controls: %s  (coverage %.1f%%)",
         avail_wx, df[avail_wx[0]].notna().mean()*100 if avail_wx else 0)

if not avail_wx:
    log.error("No weather data in panel — cannot run weather robustness.")
    sys.exit(1)

# ── Aligned outcome + combined + population ─────────────────────────────────
df_al = a05.add_aligned_outcome(df)
df_al.sort_values(["fips","date"], inplace=True)

has_combined = "combined_next_commute" in df_al.columns
has_pop      = "population" in df_al.columns

if has_pop:
    county_mean_pop = df_al.groupby("fips")["population"].transform("mean")
    pop = df_al["population"].fillna(county_mean_pop)
    df_al["log_pop"]  = np.log(pop.clip(lower=1))
    df_al["pop_100k"] = pop / 100_000
    df_al["comb_rate"] = df_al["combined_next_commute"] / df_al["pop_100k"]

df_al["county_year_code"] = pd.Categorical(
    df_al["county_code"].astype(str) + "_" + df_al["year"].astype(str)
).codes.astype(np.int32)

has_lag  = "fatals_tm1" in df_al.columns
hol_ctrl = avail_hol
twfe_ctrl = hol_ctrl + (["fatals_tm1"] if has_lag else [])

results = []

def rec(label, r, has_weather):
    if "error" not in r:
        results.append({
            "label":       label,
            "has_weather": has_weather,
            "coef":        r["coef"],
            "se":          r["se"],
            "pval":        r["pval"],
            "n_obs":       r.get("n_obs", np.nan),
        })
    else:
        log.warning("Error in %s: %s", label, r.get("error","?"))

# ── (A/B) Baseline count ────────────────────────────────────────────────────
log.info("(A) Baseline count, no weather")
rec("Baseline count", fe_ols_from_panel(
    df_al, "fatals_next_commute", controls=hol_ctrl,
    county=True, dm=True, cluster_col="state_code",
    label="(A) Baseline"), has_weather=False)

log.info("(B) Baseline count + PRISM weather")
rec("Baseline count", fe_ols_from_panel(
    df_al, "fatals_next_commute", controls=hol_ctrl + avail_wx,
    county=True, dm=True, cluster_col="state_code",
    label="(B) + Weather"), has_weather=True)

# ── (C/D) TWFE2 count ───────────────────────────────────────────────────────
log.info("(C) TWFE2 count, no weather")
rec("TWFE2 count", fe_ols_from_panel(
    df_al, "fatals_next_commute", controls=twfe_ctrl,
    county=False, dm=True, extra_fes=["county_year_code"],
    cluster_col="state_code", label="(C) TWFE2"), has_weather=False)

log.info("(D) TWFE2 count + PRISM weather")
rec("TWFE2 count", fe_ols_from_panel(
    df_al, "fatals_next_commute", controls=twfe_ctrl + avail_wx,
    county=False, dm=True, extra_fes=["county_year_code"],
    cluster_col="state_code", label="(D) TWFE2 + Wx"), has_weather=True)

# ── (E/F) WLS combined rate ──────────────────────────────────────────────────
if has_combined and has_pop:
    sub_r = df_al.dropna(subset=["comb_rate","log_pop"])

    log.info("(E) WLS rate, no weather")
    rec("WLS comb/100k", fe_ols_from_panel(
        sub_r, "comb_rate", controls=hol_ctrl,
        county=True, dm=True, weights_col="log_pop",
        cluster_col="state_code", label="(E) WLS"), has_weather=False)

    log.info("(F) WLS rate + PRISM weather")
    rec("WLS comb/100k", fe_ols_from_panel(
        sub_r, "comb_rate", controls=hol_ctrl + avail_wx,
        county=True, dm=True, weights_col="log_pop",
        cluster_col="state_code", label="(F) WLS + Wx"), has_weather=True)

del df_al; gc.collect()

# ── Print and save ───────────────────────────────────────────────────────────
out = pd.DataFrame(results)
log.info("\n=== Weather robustness results ===")
for spec_label in out["label"].unique():
    sub = out[out["label"] == spec_label]
    no_wx = sub[~sub["has_weather"]].iloc[0]
    with_wx = sub[sub["has_weather"]].iloc[0]
    def fmt(r):
        sig = ("***" if r["pval"]<0.01 else "**" if r["pval"]<0.05
               else "*" if r["pval"]<0.10 else "")
        return f"coef={r['coef']:+.4f}  se={r['se']:.4f}  p={r['pval']:.3f}  {sig}"
    log.info("\n  %-20s  No weather:   %s", spec_label, fmt(no_wx))
    log.info("  %-20s  + PRISM:      %s", spec_label, fmt(with_wx))
    coef_chg = (with_wx["coef"] - no_wx["coef"]) / abs(no_wx["coef"]) * 100
    log.info("  %-20s  Coef change:  %+.1f%%", spec_label, coef_chg)

out_path = OUTPUT_TABS / "reg_weather_robustness.csv"
out.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
