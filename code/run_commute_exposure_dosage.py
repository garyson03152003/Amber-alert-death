"""
run_commute_exposure_dosage.py
=====================================================================
Commuter-flow weighted county-level dosage analysis.

Motivation
----------
log_breadth (total counties in alert) is an ALERT-level property — the
same for every county in the same alert night, so it cannot identify
why county i is more exposed than county k on the same alert night.

This script builds a genuine COUNTY-level dosage:

    commute_dosage_{i,t} = Σ_{j : night_alert_{j,t}=1}  workers_{j → i}

where workers_{j→i} = ACS county-to-county car commuters from home county j
to work county i.  This counts the total number of car commuters who:
  1. Live in an alerted county (received WEA on phone at night)
  2. Drive to work in county i the next morning (potential distracted drivers)

Includes own-county commuters (j=i) when county i itself is alerted.

Why this works where population-scaling failed:
  - Population averaged across 97 counties → collapsed to ~constant per alert
  - commute_dosage varies at the COUNTY level because different counties have
    different sets of commuter-supplying neighbors that happen to be alerted:
    * Metro core county: large inflows from many alerted suburbs → high dosage
    * Rural isolated county: few cross-county workers, mostly own → low dosage
    * Edge-of-alert county: fewer alerted neighbors than a central county

Units: workers (raw ACS count).  Log-transform: log(1 + commute_dosage).

Comparison with existing cross_spillover:
  - cross_spillover uses weight = workers/total_workforce (fraction)
  - commute_dosage uses raw workers count
  - commute_dosage includes own county (j=i); cross_spillover excludes it

Regressions (all: state-clustered SE):
  CD1  count TWFE1:   log_commute_dosage ~ county + year FE
  CD2  count TWFE2:   + lagged fatalities
  CD3  WLS rate/100k: population-weighted rate
  CD4  Poisson PPML:  count model (P1-analog with new dosage)
  CD5  Poisson rate:  quasi-Poisson on rate/100k with pop weights

Outputs: output/tables/reg_commute_dosage.csv
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
log = get_logger("commute_dosage")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = DATA_PROC / "commuting" / "county_commuting_weights.parquet"

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

# ── Build commute exposure dosage ─────────────────────────────────────────────
log.info("Building commuter-flow dosage …")
weights = pd.read_parquet(WEIGHTS_PATH)     # fips_home, fips_work, workers, weight
log.info("  Commuting flows: %d OD pairs", len(weights))

# Get all alerted county-days (county i received a WEA on night t)
fips_in_sample = set(df["fips"].unique())
alert_events   = df.loc[df["night_alert"] > 0, ["fips", "date"]].copy()
log.info("  Alert events: %d county-nights", len(alert_events))

# For each alerted home county j: find all work counties i that receive commuters
# from j, and add workers_{j→i} to dosage of (i, t)
alert_events["fips_home"] = alert_events["fips"].astype(int)

# Merge alert events × flows: each alert event fans out to all work counties
pairs = alert_events.merge(weights[["fips_home", "fips_work", "workers"]],
                           on="fips_home", how="inner")
pairs["fips_work_str"] = pairs["fips_work"].astype(str).str.zfill(5)

# Restrict to counties in analysis sample (both home and work counties)
pairs = pairs[pairs["fips_work_str"].isin(fips_in_sample)]

log.info("  Spillover pairs (alert × OD links): %d", len(pairs))

# Aggregate: for each (work county i, date t) sum workers from all alerted home counties
dosage_df = (
    pairs.groupby(["fips_work_str", "date"])["workers"]
    .sum()
    .reset_index()
    .rename(columns={"workers": "commute_dosage", "fips_work_str": "fips"})
)

log.info("  Non-zero county-days with dosage: %d", len(dosage_df))
log.info("  Dosage stats — mean: %.0f  median: %.0f  max: %d",
         dosage_df["commute_dosage"].mean(),
         dosage_df["commute_dosage"].median(),
         dosage_df["commute_dosage"].max())

# Merge back: county-days without any alerted commuting inflow get 0
df = df.merge(dosage_df, on=["fips", "date"], how="left")
df["commute_dosage"]     = df["commute_dosage"].fillna(0.0)
df["log_commute_dosage"] = np.log1p(df["commute_dosage"])

# ── Diagnostics ───────────────────────────────────────────────────────────────
nonzero = df[df["commute_dosage"] > 0]
log.info("Non-zero dosage rows: %d (%.2f%% of panel)",
         len(nonzero), len(nonzero) / len(df) * 100)
log.info("Correlation: log_commute_dosage vs night_alert:  %.3f",
         df["log_commute_dosage"].corr(df["night_alert"]))

# Compare with log_breadth if available
if "alert_breadth" in df.columns:
    df["log_breadth"] = np.log1p(df["alert_breadth"])
    log.info("Correlation: log_commute_dosage vs log_breadth: %.3f",
             df[df["night_alert"] > 0]["log_commute_dosage"].corr(
             df[df["night_alert"] > 0]["log_breadth"]))

# Show variation WITHIN a sample alert night (key: does it vary across counties?)
sample_night = df[df["night_alert"] > 0]["date"].value_counts().idxmax()
sample = df[(df["date"] == sample_night) & (df["night_alert"] > 0)].copy()
log.info("Within-alert variation example (date=%s, %d alerted counties):",
         sample_night, len(sample))
log.info("  log_commute_dosage: mean=%.2f  std=%.2f  cv=%.2f",
         sample["log_commute_dosage"].mean(),
         sample["log_commute_dosage"].std(),
         sample["log_commute_dosage"].std() / sample["log_commute_dosage"].mean()
         if sample["log_commute_dosage"].mean() > 0 else 0)
if "alert_breadth" in df.columns:
    log.info("  log_breadth (same for all):   mean=%.2f  std=%.4f",
             sample["log_breadth"].mean(), sample["log_breadth"].std())
log.info("  Top-5 counties by dosage:\n%s",
         sample.nlargest(5, "commute_dosage")[
             ["fips", "commute_dosage", "log_commute_dosage"]].to_string(index=False))

# ── Controls ──────────────────────────────────────────────────────────────────
WEATHER   = [c for c in ["prcp_mm", "tmax_c"]  if c in df.columns and df[c].notna().mean() > 0.01]
HOL       = [c for c in ["is_holiday"]          if c in df.columns]
ctrl_parts = HOL + WEATHER
CTRL_STR   = " + ".join(ctrl_parts) if ctrl_parts else "1"
lag_col    = next((c for c in ["fatals_tm1", "lag_fatals"] if c in df.columns), None)

# Population column for WLS / rate
pop_col = next((c for c in ["population", "pop"] if c in df.columns), None)
if pop_col:
    df["pop_w"]           = df[pop_col].clip(lower=1)
    df["fatals_rate_100k"]= df["fatals_next_commute"] * 100_000 / df["pop_w"]

# ── Result accumulator ────────────────────────────────────────────────────────
results = []

def _sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def _extract(fit, treatment, tidy=None):
    if tidy is None:
        tidy = fit.tidy()
    row = tidy.loc[treatment] if treatment in tidy.index else \
          tidy.loc[[i for i in tidy.index if treatment in i][0]]
    coef = float(row["Estimate"])
    se   = float(row["Std. Error"])
    pval = float(row["Pr(>|t|)"])
    nobs = int(getattr(fit, "_N", None) or getattr(fit, "n_obs", None) or 0)
    return coef, se, pval, nobs

def run_ols(label, formula, treatment, data, spec_tag, weights_col=None):
    outcome_var = formula.split("~")[0].strip()
    sub = data.dropna(subset=[treatment, outcome_var]).copy()
    fit = None
    try:
        kw = {"vcov": {"CRV1": "state_code"}, "lean": True}
        if weights_col:
            kw["weights"] = weights_col
        fit  = pf.feols(formula, data=sub, **kw)
        tidy = fit.tidy()
        coef, se, pval, nobs = _extract(fit, treatment, tidy)
        log.info("  %-52s β=%+.4f  se=%.4f  p=%.3f  n=%d  %s",
                 label, coef, se, pval, nobs, _sig(pval))
        results.append({"label": label, "spec": spec_tag, "treatment": treatment,
                        "coef": coef, "se": se, "pval": pval, "nobs": nobs,
                        "model": "ols_wls", "outcome": outcome_var})
    except Exception as e:
        log.warning("  %s FAILED: %s", label, e)
    finally:
        del fit, sub; gc.collect()

def run_ppml(label, formula, treatment, data, spec_tag, weights_col=None):
    outcome_var = formula.split("~")[0].strip()
    sub = data.dropna(subset=[treatment, outcome_var]).copy()
    sub = sub[sub[outcome_var] >= 0]
    fit = None
    try:
        kw = {"vcov": {"CRV1": "state_code"}, "lean": True}
        if weights_col:
            kw["weights"] = weights_col
        fit  = pf.fepois(formula, data=sub, **kw)
        tidy = fit.tidy()
        coef, se, pval, nobs = _extract(fit, treatment, tidy)
        irr  = np.exp(coef)
        log.info("  %-52s β=%+.4f  se=%.4f  p=%.3f  IRR=%.4f  %s",
                 label, coef, se, pval, irr, _sig(pval))
        results.append({"label": label, "spec": spec_tag, "treatment": treatment,
                        "coef": coef, "se": se, "pval": pval, "nobs": nobs,
                        "irr": irr, "model": "poisson_ppml",
                        "outcome": outcome_var})
    except Exception as e:
        log.warning("  %s FAILED: %s", label, e)
    finally:
        del fit, sub; gc.collect()

# ── CD1: OLS count TWFE1 ──────────────────────────────────────────────────────
log.info("\n=== OLS count (raw fatalities) ===")
run_ols("CD1 log_commute_dosage [count TWFE1]",
        f"fatals_next_commute ~ log_commute_dosage + {CTRL_STR} | fips + year_str",
        "log_commute_dosage", df, "ols_count_twfe1")

# CD2 (TWFE2 + lag) skipped to avoid OOM at 7.2M rows; OLS TWFE2 in main analysis covers this

# ── CD3: WLS rate/100k ────────────────────────────────────────────────────────
if pop_col:
    log.info("\n=== WLS rate / 100k population ===")
    run_ols("CD3 log_commute_dosage [WLS rate/100k]",
            f"fatals_rate_100k ~ log_commute_dosage + {CTRL_STR} | fips + year_str",
            "log_commute_dosage", df, "ols_rate_wls",
            weights_col="pop_w")

# ── CD4: Poisson PPML raw count ───────────────────────────────────────────────
log.info("\n=== Poisson PPML (county + year FE) ===")
run_ppml("CD4 log_commute_dosage [Poisson count]",
         f"fatals_next_commute ~ log_commute_dosage + {CTRL_STR} | fips + year_str",
         "log_commute_dosage", df, "ppml_count")

# ── CD5: Poisson rate/100k ────────────────────────────────────────────────────
if pop_col:
    log.info("\n--- Quasi-Poisson on rate/100k with pop weights ---")
    run_ppml("CD5 log_commute_dosage [Poisson rate/100k, pop-wt]",
             f"fatals_rate_100k ~ log_commute_dosage + {CTRL_STR} | fips + year_str",
             "log_commute_dosage", df, "ppml_rate",
             weights_col="pop_w")

# ── Benchmark: binary night_alert ────────────────────────────────────────────
log.info("\n=== Benchmark: binary night_alert ===")
run_ols("BIN count TWFE1",
        f"fatals_next_commute ~ night_alert + {CTRL_STR} | fips + year_str",
        "night_alert", df, "ols_count_twfe1_bin")
gc.collect()
if pop_col:
    run_ols("BIN WLS rate/100k",
            f"fatals_rate_100k ~ night_alert + {CTRL_STR} | fips + year_str",
            "night_alert", df, "ols_rate_wls_bin", weights_col="pop_w")
gc.collect()

# ── Full summary ──────────────────────────────────────────────────────────────
log.info("\n=== Full summary ===")
for r in results:
    sig = _sig(r["pval"])
    irr_str = f"  IRR={r.get('irr', float('nan')):.4f}" if "irr" in r else ""
    log.info("  %-52s β=%+.4f  se=%.4f  p=%.3f%s  %s",
             r["label"], r["coef"], r["se"], r["pval"], irr_str, sig or "n.s.")

# ── Save ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = OUTPUT_TABS / "reg_commute_dosage.csv"
out.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
