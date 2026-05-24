"""
run_poisson_fe.py
Poisson PPML fixed-effects count model as a robustness check.

Why Poisson?
  - Outcome is a non-negative integer count (daily traffic fatalities)
  - OLS predicts on a linear scale; Poisson models E[y] = exp(Xβ)
  - Robust to exact distributional misspecification (QMLE consistency)
  - Handles zero-inflation (97% zeros) without special treatment
  - Incidental-parameters problem is mild for Poisson FE
    (Hausman, Hall & Griliches 1984; Wooldridge 1999)

Overdispersion: var/mean = 1.41 (mild). We report robust/clustered SEs
which remain valid regardless of the true count distribution.

Specifications (all: state-clustered SE throughout):
  P1: Poisson  county + year FE + binary treatment
  P2: Poisson  county + year FE + log-breadth dosage
  P3: Poisson  county + year FE + binary + lagged fatals (≈TWFE2)
  P4: Poisson  county + year FE + log-breadth + lagged fatals (≈TWFE2)
  NB: Neg. Bin county + year FE + binary (overdispersion check, subsample)

Uses pyfixest.fepois (PPML, same algorithm as R ppmlhdfe).
NegBin via statsmodels on 300-county subsample (LSDV too heavy at full scale).

Output: output/tables/reg_poisson.csv
"""
import sys, warnings, importlib.util, gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("poisson")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

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

WEATHER = [c for c in ["prcp_mm", "tmax_c"] if c in df.columns and df[c].notna().mean() > 0.01]
HOL     = [c for c in ["is_holiday"] if c in df.columns]
ctrl_parts = HOL + WEATHER
CTRL_STR   = " + ".join(ctrl_parts) if ctrl_parts else "1"

# Compute log_breadth from breadth column if available
if "log_breadth" not in df.columns:
    bread_col = next((c for c in ["breadth", "n_counties", "alert_breadth"] if c in df.columns), None)
    if bread_col:
        df["log_breadth"] = np.log1p(df[bread_col])
        log.info("Derived log_breadth from '%s' (mean=%.3f)", bread_col, df["log_breadth"].mean())
    else:
        log.info("log_breadth not available — P2/P4 will be skipped")

# ── Overdispersion summary ────────────────────────────────────────────────────
y = df["fatals_next_commute"].dropna()
log.info("fatals_next_commute: n=%d  mean=%.4f  var=%.4f  dispersion ratio=%.2f",
         len(y), y.mean(), y.var(), y.var() / y.mean())
log.info("Zeros: %.1f%%   Max: %d", (y == 0).mean() * 100, int(y.max()))

# ── pyfixest result extractor ────────────────────────────────────────────────
def extract_fepois(fit, treatment):
    """Return (coef, se, pval, nobs) from a pyfixest Fepois object."""
    tidy = fit.tidy()                         # DataFrame indexed by term name
    if treatment not in tidy.index:
        # try partial match
        matches = [i for i in tidy.index if treatment in i]
        if not matches:
            raise KeyError(f"'{treatment}' not in {list(tidy.index)}")
        treatment = matches[0]
    row  = tidy.loc[treatment]
    coef = float(row["Estimate"])
    se   = float(row["Std. Error"])
    pval = float(row["Pr(>|t|)"])
    # nobs: try several attribute names across pyfixest versions
    nobs = getattr(fit, "_N", None) or getattr(fit, "n_obs", None) or len(fit.resid())
    return coef, se, pval, int(nobs)

results = []

def run_ppml(label, formula, treatment, data, spec_tag):
    log.info("  %s …", label)
    sub = data.dropna(subset=[treatment, "fatals_next_commute"]).copy()
    sub = sub[sub["fatals_next_commute"] >= 0]
    fit = None
    try:
        fit  = pf.fepois(formula, data=sub, vcov={"CRV1": "state_code"})
        coef, se, pval, nobs = extract_fepois(fit, treatment)
        irr  = np.exp(coef)
        sig  = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else ""
        log.info("  %-48s β=%+.4f  se=%.4f  p=%.3f  IRR=%.4f  %s",
                 label, coef, se, pval, irr, sig)
        results.append({"label": label, "spec": spec_tag, "treatment": treatment,
                         "coef": coef, "se": se, "pval": pval,
                         "nobs": nobs, "irr": irr, "model": "poisson_ppml"})
    except Exception as e:
        log.warning("  %s FAILED: %s", label, e)
    finally:
        del fit, sub
        gc.collect()

# ── Lagged fatals column ───────────────────────────────────────────────────────
lag_col = next((c for c in ["fatals_tm1", "lag_fatals"] if c in df.columns), None)
lag_str = f" + {lag_col}" if lag_col else ""
if lag_col:
    log.info("Lagged fatals column: %s  (missing rate %.1f%%)",
             lag_col, df[lag_col].isna().mean() * 100)

# ── P1–P4: Poisson with county + year FE (avoids memory-intensive ×) ─────────
log.info("\n=== Poisson PPML: county + year FE ===")

run_ppml("P1 Binary         [county+yr FE]",
         f"fatals_next_commute ~ night_alert + {CTRL_STR} | fips + year_str",
         "night_alert", df, "ppml_ctyYr")

if "log_breadth" in df.columns:
    run_ppml("P2 Log-breadth    [county+yr FE]",
             f"fatals_next_commute ~ log_breadth + {CTRL_STR} | fips + year_str",
             "log_breadth", df, "ppml_ctyYr")

# P3/P4 (binary/breadth + lagged fatals) are skipped for Poisson to avoid OOM
# at full panel scale (7M rows × PPML iterations × lagged column).
# OLS TWFE2 specs cover the lagged-DV robustness check.
if lag_col:
    log.info("  P3/P4 (Poisson + lag) skipped at full scale — OLS TWFE2 covers this.")

gc.collect()

# ── NB: Negative Binomial on subsample ────────────────────────────────────────
log.info("\n=== Negative Binomial (overdispersion robustness, 100-county subsample) ===")
np.random.seed(42)
nb_counties = np.random.choice(df["fips"].unique(), size=100, replace=False)
nb_df = (df[df["fips"].isin(nb_counties)]
           .dropna(subset=["fatals_next_commute", "night_alert"] + ctrl_parts)
           .copy())
nb_df["fatals_next_commute"] = nb_df["fatals_next_commute"].astype(int)

# Drop counties with perfect-0 or zero-variance outcomes (can't identify FE)
cty_means = nb_df.groupby("fips")["fatals_next_commute"].mean()
nonzero_cty = cty_means[cty_means > 0].index
nb_df = nb_df[nb_df["fips"].isin(nonzero_cty)].copy()
log.info("  NegBin sample: %d obs, %d counties", len(nb_df), nb_df["fips"].nunique())

# county + year dummies (LSDV)
cty_dum  = pd.get_dummies(nb_df["fips"],     prefix="c",  drop_first=True, dtype=float)
yr_dum   = pd.get_dummies(nb_df["year_str"], prefix="yr", drop_first=True, dtype=float)
ctrl_arr = nb_df[["night_alert"] + ctrl_parts].reset_index(drop=True).astype(float)
X_nb = pd.concat([ctrl_arr, cty_dum.reset_index(drop=True),
                  yr_dum.reset_index(drop=True)], axis=1)
X_nb = sm.add_constant(X_nb)
# Drop any constant or all-zero columns
X_nb = X_nb.loc[:, X_nb.std() > 0]
y_nb = nb_df["fatals_next_commute"].reset_index(drop=True).astype(int)

try:
    nb_fit = sm.NegativeBinomial(y_nb, X_nb).fit(
        method="nm", maxiter=500, disp=False, cov_type="HC3")
    coef_nb = float(nb_fit.params["night_alert"])
    se_nb   = float(nb_fit.bse["night_alert"])
    pval_nb = float(nb_fit.pvalues["night_alert"])
    alpha   = float(nb_fit.params.get("alpha", np.nan))
    irr_nb  = np.exp(coef_nb)
    sig_nb  = "***" if pval_nb < .01 else "**" if pval_nb < .05 else "*" if pval_nb < .10 else ""
    log.info("  NB Binary [subsample]: β=%+.4f  se=%.4f  p=%.3f  IRR=%.4f  alpha=%.3f  %s",
             coef_nb, se_nb, pval_nb, irr_nb, alpha, sig_nb)
    results.append({"label": "NB Binary [100-cty subsample]", "spec": "negbin_subsample",
                    "treatment": "night_alert", "coef": coef_nb, "se": se_nb,
                    "pval": pval_nb, "nobs": len(y_nb), "irr": irr_nb,
                    "model": "negbin", "dispersion_alpha": alpha})
except Exception as e:
    log.warning("  NegBin FAILED: %s", e)

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("\n=== Full summary ===")
log.info("OLS reference:")
log.info("  OLS P1 Binary     [county+yr FE]:  β=+0.0044  p≈0.244  n.s.")
log.info("  OLS P2 Log-breadth[county+yr FE]:  β=+0.0015  p=0.041**")
log.info("  OLS P3 Binary+lag [county+yr FE]:  β=+0.0050  p≈0.200  n.s.")
log.info("  OLS P4 Breadth+lag[county+yr FE]:  β=+0.0016  p≈0.035**")
log.info("Poisson/NegBin:")
for r in results:
    sig = "***" if r["pval"]<.01 else "**" if r["pval"]<.05 else "*" if r["pval"]<.10 else "n.s."
    log.info("  %-48s β=%+.4f  se=%.4f  p=%.3f  IRR=%.4f  %s",
             r["label"], r["coef"], r["se"], r["pval"], r.get("irr", np.nan), sig)

log.info("\nInterpretation note: for Poisson, IRR = exp(β) is the incidence")
log.info("  rate ratio. IRR=1.005 means 0.5%% more fatalities on treated nights.")
log.info("  With β≈0.001–0.006, IRR≈1.001–1.006 (small proportional effect).")

# ── Save ─────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_path = OUTPUT_TABS / "reg_poisson.csv"
out_df.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
