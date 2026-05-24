"""
run_affected_commuters.py
==========================================================================
Construct and test the "affected commuter" dosage at the county-day level.

Conceptual framework
--------------------
An Amber Alert fires in county j at night t.  Every car commuter who:
  (a) LIVES in county j  →  received the WEA on their phone that night
  (b) drives to work on morning t+1

...is a potentially distracted driver.  The roads they drive on depend on
where they work:
  - If they work in county i, they drive through / into county i → i exposed
  - If they work in county j (same as home), they drive within j → j exposed

So the distracted drivers who will be on county i's roads on morning t+1 are:

    affected_{i,t} =  car_total_i  × alert_{i,t}          (own-county)
                    + Σ_{j ≠ i : alert_{j,t}=1}  flow_{j→i}  (cross-county)

where:
  car_total_i     = total car commuters living in county i   (ACS B08301)
  alert_{i,t}     = 1 if county i received an alert on night t
  flow_{j→i}      = car commuters living in county j, working in county i
                    (ACS 2016-2020 county-to-county flows)

Own-county note: we use ALL car workers living in county i (not just those
who work in i), because even outbound commuters drive on county i's roads
for the first leg of their trip.

This variable is:
  • County-level: varies across counties within the same alert night
  • Mechanistically grounded: counts actual commuters on the roads
  • Not subject to the multi-county averaging collapse (unlike raw pop. reached)

Output: output/tables/reg_affected_commuters.csv
"""
import sys, warnings, importlib.util, gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("affected_commuters")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

CAR_PATH     = DATA_PROC / "county_car_commuters.parquet"
FLOWS_PATH   = DATA_PROC / "commuting" / "county_commuting_weights.parquet"

# ── Load panel ────────────────────────────────────────────────────────────────
log.info("Loading panel …")
spec = importlib.util.spec_from_file_location("a05", Path(__file__).parent / "05_analysis.py")
a05  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

df = a05.load_panel()
df = prep_panel(df)
df = a05.add_aligned_outcome(df)
df.sort_values(["fips", "date"], inplace=True)
df["fips"]       = df["fips"].astype(str)
df["state_code"] = df["state_code"].astype(str)
df["year_str"]   = pd.to_datetime(df["date"]).dt.year.astype(str)

log.info("Panel: %d rows, %d counties", len(df), df["fips"].nunique())

# ── Build affected_commuters ──────────────────────────────────────────────────
log.info("\nBuilding affected_commuters dosage …")

# ── Part A: Own-county exposure ───────────────────────────────────────────────
#   car_total_i × alert_{i,t}
#   All car workers who live in county i and received the alert there
car = pd.read_parquet(CAR_PATH)[["fips", "car_total"]].copy()
car["fips"] = car["fips"].astype(str).str.zfill(5)
df = df.merge(car.rename(columns={"car_total": "own_car_total"}),
              on="fips", how="left")
df["own_car_total"] = df["own_car_total"].fillna(0)

# own exposure: car_total × 1(alerted); zero when not alerted
df["own_exposure"] = df["own_car_total"] * (df["night_alert"] > 0).astype(float)
log.info("  Own-county car workers: mean=%.0f  (alerted nights only: %.0f)",
         df["own_car_total"].mean(),
         df.loc[df["night_alert"] > 0, "own_car_total"].mean())

# ── Part B: Cross-county inbound exposure ─────────────────────────────────────
#   Σ_{j≠i, alert_j=1}  flow_{j→i}
#   Car commuters living in alerted county j who drive to work in county i
flows = pd.read_parquet(FLOWS_PATH)[["fips_home", "fips_work", "workers"]].copy()
# Exclude own-county flows (already in Part A, don't double-count)
flows = flows[flows["fips_home"] != flows["fips_work"]]

fips_in_sample = set(df["fips"].unique())
alert_events   = df.loc[df["night_alert"] > 0, ["fips", "date"]].copy()
alert_events["fips_home"] = alert_events["fips"].astype(int)

# Each alerted home county fans out workers to all work counties
pairs = alert_events.merge(flows, on="fips_home", how="inner")
pairs["fips_work_str"] = pairs["fips_work"].astype(str).str.zfill(5)
pairs = pairs[pairs["fips_work_str"].isin(fips_in_sample)]

cross_agg = (pairs.groupby(["fips_work_str", "date"])["workers"]
             .sum().reset_index()
             .rename(columns={"workers": "cross_exposure", "fips_work_str": "fips"}))

df = df.merge(cross_agg, on=["fips", "date"], how="left")
df["cross_exposure"] = df["cross_exposure"].fillna(0)
log.info("  Cross-county inbound exposure: %d non-zero county-days  (mean=%.0f)",
         (df["cross_exposure"] > 0).sum(), df.loc[df["cross_exposure"] > 0, "cross_exposure"].mean())

# ── Total affected commuters ──────────────────────────────────────────────────
df["affected_commuters"]     = df["own_exposure"] + df["cross_exposure"]
df["log_affected_commuters"] = np.log1p(df["affected_commuters"])

# ── Diagnostics ───────────────────────────────────────────────────────────────
treated = df[df["night_alert"] > 0]
log.info("\nDosage summary (on treated county-nights only):")
log.info("  own_exposure:    mean=%8.0f  median=%6.0f  std=%8.0f",
         treated["own_exposure"].mean(), treated["own_exposure"].median(),
         treated["own_exposure"].std())
log.info("  cross_exposure:  mean=%8.0f  median=%6.0f  std=%8.0f",
         treated["cross_exposure"].mean(), treated["cross_exposure"].median(),
         treated["cross_exposure"].std())
log.info("  total affected:  mean=%8.0f  median=%6.0f  std=%8.0f",
         treated["affected_commuters"].mean(), treated["affected_commuters"].median(),
         treated["affected_commuters"].std())
log.info("  log_affected:    mean=%7.3f  std=%5.3f",
         treated["log_affected_commuters"].mean(),
         treated["log_affected_commuters"].std())

# Show within-alert variation (the key property)
sample_night = df[df["night_alert"] > 0]["date"].value_counts().idxmax()
sample       = df[(df["date"] == sample_night) & (df["night_alert"] > 0)]
if "alert_breadth" in df.columns:
    breadth = sample["alert_breadth"].iloc[0]
else:
    breadth = sample["night_alert"].sum()

log.info("\nWithin-alert variation (date=%s, %d alerted counties, breadth=%d):",
         sample_night, len(sample), int(breadth))
lac = sample["log_affected_commuters"]
log.info("  log_affected_commuters: mean=%.2f  std=%.3f  cv=%.3f  [min=%.2f, max=%.2f]",
         lac.mean(), lac.std(),
         lac.std()/lac.mean() if lac.mean() > 0 else 0,
         lac.min(), lac.max())
log.info("  (For comparison: log_breadth std=0.000 — constant across counties in same alert)")
log.info("  Top counties by affected commuters:")
top = sample.nlargest(5, "affected_commuters")[
    ["fips", "own_exposure", "cross_exposure", "affected_commuters"]]
log.info("\n%s", top.to_string(index=False))

# Correlation checks
if "alert_breadth" in df.columns:
    df["log_breadth"] = np.log1p(df["alert_breadth"])
treated = df[df["night_alert"] > 0]   # refresh after adding log_breadth

log.info("\nCorrelation (on treated rows):")
if "log_breadth" in df.columns:
    log.info("  log_affected vs log_breadth:     %.3f",
             treated["log_affected_commuters"].corr(treated["log_breadth"]))
log.info("  log_affected vs night_alert:     %.3f",
         df["log_affected_commuters"].corr(df["night_alert"]))

# ── Controls ──────────────────────────────────────────────────────────────────
WEATHER    = [c for c in ["prcp_mm", "tmax_c"]
              if c in df.columns and df[c].notna().mean() > 0.01]
HOL        = [c for c in ["is_holiday"] if c in df.columns]
ctrl_parts = HOL + WEATHER
CTRL_STR   = " + ".join(ctrl_parts) if ctrl_parts else "1"
lag_col    = next((c for c in ["fatals_tm1", "lag_fatals"] if c in df.columns), None)

pop_col = next((c for c in ["population", "pop"] if c in df.columns), None)
if pop_col:
    df["pop_w"]            = df[pop_col].clip(lower=1)
    df["fatals_rate_100k"] = df["fatals_next_commute"] * 100_000 / df["pop_w"]

results = []

def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "n.s."

def _fit(formula, treat, data, model="ols", wt=None):
    out_var = formula.split("~")[0].strip()
    sub = data.dropna(subset=[treat, out_var]).copy()
    sub = sub[sub[out_var] >= 0]
    kw  = {"vcov": {"CRV1": "state_code"}, "lean": True}
    if wt:
        kw["weights"] = wt
    fit = (pf.fepois(formula, data=sub, **kw) if model == "ppml"
           else pf.feols(formula, data=sub, **kw))
    td  = fit.tidy()
    row = (td.loc[treat] if treat in td.index
           else td.loc[[i for i in td.index if treat in i][0]])
    coef = float(row["Estimate"])
    se   = float(row["Std. Error"])
    pval = float(row["Pr(>|t|)"])
    nobs = int(getattr(fit, "_N", None) or 0)
    del fit, sub; gc.collect()
    return coef, se, pval, nobs

def run(label, formula, treat, data, spec_tag, model="ols", wt=None):
    log.info("  %s …", label)
    try:
        coef, se, pval, nobs = _fit(formula, treat, data, model, wt)
        irr_str = f"  IRR={np.exp(coef):.4f}" if model == "ppml" else ""
        log.info("  %-58s β=%+.5f  se=%.5f  p=%.3f  n=%d%s  %s",
                 label, coef, se, pval, nobs, irr_str, _sig(pval))
        results.append({"label": label, "spec": spec_tag, "treatment": treat,
                        "coef": coef, "se": se, "pval": pval, "nobs": nobs,
                        "model": model,
                        "irr": np.exp(coef) if model == "ppml" else np.nan})
    except Exception as e:
        log.warning("  %s FAILED: %s", label, e)

TREAT = "log_affected_commuters"

# ── Main regressions ──────────────────────────────────────────────────────────
log.info("\n=== OLS count TWFE1 ===")
run("AC1 log_affected_commuters [count TWFE1]",
    f"fatals_next_commute ~ {TREAT} + {CTRL_STR} | fips + year_str",
    TREAT, df, "ols_count_twfe1")

if pop_col:
    log.info("\n=== WLS rate/100k ===")
    run("AC2 log_affected_commuters [WLS rate/100k]",
        f"fatals_rate_100k ~ {TREAT} + {CTRL_STR} | fips + year_str",
        TREAT, df, "ols_rate_wls", wt="pop_w")

log.info("\n=== Poisson PPML count ===")
run("AC3 log_affected_commuters [Poisson count]",
    f"fatals_next_commute ~ {TREAT} + {CTRL_STR} | fips + year_str",
    TREAT, df, "ppml_count", model="ppml")

if pop_col:
    log.info("\n=== Quasi-Poisson rate/100k ===")
    run("AC4 log_affected_commuters [Poisson rate/100k, pop-wt]",
        f"fatals_rate_100k ~ {TREAT} + {CTRL_STR} | fips + year_str",
        TREAT, df, "ppml_rate", model="ppml", wt="pop_w")

# ── Benchmark specs (binary) ──────────────────────────────────────────────────
log.info("\n=== Benchmark: binary night_alert ===")
run("BIN night_alert [count TWFE1]",
    f"fatals_next_commute ~ night_alert + {CTRL_STR} | fips + year_str",
    "night_alert", df, "ols_count_twfe1_bin")
if pop_col:
    run("BIN night_alert [WLS rate/100k]",
        f"fatals_rate_100k ~ night_alert + {CTRL_STR} | fips + year_str",
        "night_alert", df, "ols_rate_wls_bin", wt="pop_w")

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("\n=== Summary ===")
for r in results:
    irr = f"  IRR={r['irr']:.4f}" if not np.isnan(r.get("irr", np.nan)) else ""
    log.info("  %-58s β=%+.5f  se=%.5f  p=%.3f%s  %s",
             r["label"], r["coef"], r["se"], r["pval"], irr, _sig(r["pval"]))

# ── Save ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = OUTPUT_TABS / "reg_affected_commuters.csv"
out.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
