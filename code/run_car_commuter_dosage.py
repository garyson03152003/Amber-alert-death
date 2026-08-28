"""
run_car_commuter_dosage.py
Compares four dosage variables:
  (A) Binary treatment (night_alert)
  (B) log(1 + n_counties)         — geographic breadth
  (C) log(1 + pop_reached)        — raw population in alert counties
  (D) log(1 + car_commuters_reached) — car-commuting workers in alert counties

car_commuters_reached = Σ_{county in alert} car_total_c
  where car_total_c = ACS 2020 5-yr B08301_002 (drove alone + carpooled)

Two specs each: count baseline FE + WLS combined/100k.

Output: output/tables/reg_dosage_car_commuters.csv
"""
import sys, warnings, gc, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel, fe_ols_from_panel

warnings.filterwarnings("ignore")
log = get_logger("car_dosage")

CAR_PATH  = DATA_PROC / "county_car_commuters.parquet"
CELL_PATH = DATA_PROC / "county_cell_connectivity.parquet"
COV_PATH  = DATA_PROC / "county_coverage_weight.parquet"
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

if not CAR_PATH.exists():
    log.error("Car commuter file not found: %s", CAR_PATH)
    log.error("Run code/01e_fetch_car_commuters.py first.")
    sys.exit(1)

# ── Load data ────────────────────────────────────────────────────────────────
log.info("Loading panel …")
spec = importlib.util.spec_from_file_location("a05", Path(__file__).parent / "05_analysis.py")
a05  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

df = a05.load_panel()
df = prep_panel(df)
df = a05.add_aligned_outcome(df)
df.sort_values(["fips","date"], inplace=True)

WEATHER = [c for c in ["prcp_mm","tmax_c"] if c in df.columns and df[c].notna().mean() > 0.01]
HOL     = [c for c in ["is_holiday"] if c in df.columns]
BASE_CTRL = HOL + WEATHER

# Population weights
if "population" in df.columns:
    cpop = df.groupby("fips")["population"].transform("mean")
    pop  = df["population"].fillna(cpop)
    df["log_pop"]  = np.log(pop.clip(lower=1))
    df["pop_100k"] = pop / 100_000
    has_pop = True
else:
    has_pop = False

if "combined_next_commute" in df.columns and has_pop:
    df["comb_rate"] = df["combined_next_commute"] / df["pop_100k"]
    has_comb = True
else:
    has_comb = False

# ── Load car commuter data ───────────────────────────────────────────────────
log.info("Loading ACS car-commuter data …")
car = pd.read_parquet(CAR_PATH, columns=["fips","car_total","total_workers"])
car["fips"] = car["fips"].astype(str).str.zfill(5)
log.info("Car commuter counties: %d, mean car_total: %.0f", len(car), car["car_total"].mean())
log.info("Car share: mean=%.1f%%, range %.1f–%.1f%%",
         (car["car_total"]/car["total_workers"].clip(lower=1)).mean()*100,
         (car["car_total"]/car["total_workers"].clip(lower=1)).min()*100,
         (car["car_total"]/car["total_workers"].clip(lower=1)).max()*100)

# ── Build alert-level dosage variables ───────────────────────────────────────
log.info("Building car-commuter dosage variable …")

# Merge car_total into the panel
df = df.merge(car[["fips","car_total"]].rename(columns={"car_total":"county_car_commuters"}),
              on="fips", how="left")
df["county_car_commuters"] = df["county_car_commuters"].fillna(0)

# For each alert-county-day, aggregate to get total car commuters across alert footprint
# The panel has `log_breadth` = log(1 + n_counties_in_alert)
# and `pop_reached` if available

# Build car_commuters_reached similarly to how breadth is built:
# On alert days, each county c sees an alert covering n counties.
# We need: Σ_{j in alert} car_commuters_j for each alert event.
# Since each county in the panel knows its own alert breadth, but not which OTHER counties
# are in the same alert, we need to aggregate from the alerts data.

# Load amber alerts to reconstruct which counties were in each alert
alerts_raw = pd.read_parquet(DATA_PROC / "amber_alerts_clean.parquet")
alerts_raw["issued_local"] = pd.to_datetime(alerts_raw["issued_local"])
alerts_raw["date"]         = alerts_raw["issued_local"].dt.normalize()
alerts_raw["county_fips"]  = alerts_raw["county_fips"].astype(str).str.zfill(5)

# Merge car commuters into alert counties
alerts_raw = alerts_raw.merge(car[["fips","car_total"]], left_on="county_fips", right_on="fips", how="left")
alerts_raw["car_total"] = alerts_raw["car_total"].fillna(0)

# Only keep night alerts
night_alerts_car = alerts_raw[alerts_raw["is_night"]].copy()

# For each (date, county_fips) pair in the panel, compute total car commuters in the alert
# An alert event covers multiple counties on the same date; sum car commuters across counties
alert_car = (night_alerts_car.groupby(["date","county_fips"])["car_total"]
             .sum().reset_index()
             .rename(columns={"car_total":"car_commuters_in_alert","county_fips":"fips"}))

# But wait: each county_fips row is one county in the alert, so we want the TOTAL
# across ALL counties in the alert that covered county c on date t.
# The panel's night_alert=1 already flags county c if it received the alert.
# What we want: for county c on day t with night_alert=1, total car_commuters across
# ALL counties in that alert (including c itself).
#
# Since we don't have an alert_id per county-day in the panel, we use date as proxy:
# For each date, aggregate all alert counties' car commuters.
# This is the same aggregation used for log_breadth (per date).

alert_day_car = (night_alerts_car.groupby("date").agg(
    car_commuters_reached=("car_total", "sum"),
    n_alert_counties=("county_fips", "nunique")
).reset_index())

log.info("Alert-day car commuter aggregation:")
log.info("  Night alert days: %d", len(alert_day_car))
log.info("  Mean car_commuters_reached: {:,.0f}".format(alert_day_car["car_commuters_reached"].mean()))
log.info("  Corr(car_reached, n_counties): %.3f",
         alert_day_car["car_commuters_reached"].corr(alert_day_car["n_alert_counties"]))

# Merge into panel
df["date"] = pd.to_datetime(df["date"])
alert_day_car["date"] = pd.to_datetime(alert_day_car["date"])
df = df.merge(alert_day_car[["date","car_commuters_reached"]], on="date", how="left")
df["car_commuters_reached"] = df["car_commuters_reached"].fillna(0)
# FIX: zero out non-treated counties (date-level merge assigns to ALL counties on alert dates)
df["car_commuters_reached"] *= (df["night_alert"] > 0).astype(float)

# Build log dosage
df["log_car_reached"] = np.log1p(df["car_commuters_reached"])
df["log_pop_reached"] = np.log1p(df["car_commuters_reached"])  # placeholder until pop_reached

# If log_breadth already exists in panel
if "log_breadth" not in df.columns and "n_counties_covered" in df.columns:
    df["log_breadth"] = np.log1p(df["n_counties_covered"])
elif "log_breadth" not in df.columns:
    # reconstruct from alert_day_car
    df = df.merge(alert_day_car[["date","n_alert_counties"]].rename(
        columns={"n_alert_counties":"n_counties_covered"}), on="date", how="left")
    df["n_counties_covered"] = df["n_counties_covered"].fillna(0)
    # FIX: only treated counties should have non-zero breadth
    df["n_counties_covered"] *= (df["night_alert"] > 0).astype(float)
    df["log_breadth"] = np.log1p(df["n_counties_covered"])

# Population reached (if not already in panel)
if "pop_reached" not in df.columns:
    # Build from alerts + population
    pop_data = df[["fips","population"]].drop_duplicates("fips").copy()
    pop_data["population"] = pop_data["population"].fillna(pop_data["population"].median())
    alerts_raw2 = alerts_raw.merge(pop_data, left_on="county_fips", right_on="fips", how="left")
    alerts_raw2["population"] = alerts_raw2["population"].fillna(0)
    night_alerts2 = alerts_raw2[alerts_raw2["is_night"]].copy()
    alert_day_pop = (night_alerts2.groupby("date")["population"].sum().reset_index()
                     .rename(columns={"population":"pop_reached"}))
    alert_day_pop["date"] = pd.to_datetime(alert_day_pop["date"])
    df = df.merge(alert_day_pop, on="date", how="left")
    df["pop_reached"] = df["pop_reached"].fillna(0)
    # FIX: zero out non-treated counties
    df["pop_reached"] *= (df["night_alert"] > 0).astype(float)

df["log_pop_reached"] = np.log1p(df["pop_reached"])

# ── (E) Cellular-connectivity-weighted population reached ────────────────────
has_cell = False
if CELL_PATH.exists():
    log.info("Loading ACS cellular connectivity data …")
    cell = pd.read_parquet(CELL_PATH, columns=["fips","hh_total","hh_cell_plan"])
    cell["fips"] = cell["fips"].astype(str).str.zfill(5)
    cell["cell_share"] = cell["hh_cell_plan"] / cell["hh_total"].clip(lower=1)
    log.info("Cell plan counties: %d, mean share: %.1f%%",
             len(cell), cell["cell_share"].mean() * 100)

    # Merge cell_share into panel for per-county weight
    df = df.merge(cell[["fips","cell_share"]], on="fips", how="left")
    df["cell_share"] = df["cell_share"].fillna(df["cell_share"].median())

    # For each alert, compute Σ pop_j × cell_share_j across alert counties
    pop_data2 = df[["fips","population","cell_share"]].drop_duplicates("fips").copy()
    pop_data2["population"] = pop_data2["population"].fillna(pop_data2["population"].median())
    pop_data2["cell_pop"] = pop_data2["population"] * pop_data2["cell_share"]

    alerts_cell = alerts_raw.merge(pop_data2[["fips","cell_pop"]],
                                   left_on="county_fips", right_on="fips", how="left")
    alerts_cell["cell_pop"] = alerts_cell["cell_pop"].fillna(0)
    night_alerts_cell = alerts_cell[alerts_cell["is_night"]].copy()

    alert_day_cell = (night_alerts_cell.groupby("date")["cell_pop"]
                      .sum().reset_index()
                      .rename(columns={"cell_pop": "cell_pop_reached"}))
    alert_day_cell["date"] = pd.to_datetime(alert_day_cell["date"])

    df = df.merge(alert_day_cell, on="date", how="left")
    df["cell_pop_reached"] = df["cell_pop_reached"].fillna(0)
    # Zero out non-treated counties
    df["cell_pop_reached"] *= (df["night_alert"] > 0).astype(float)
    df["log_cell_reached"] = np.log1p(df["cell_pop_reached"])
    has_cell    = True
    has_cellcar = False   # set True after (F) block below
    log.info("Built log_cell_reached: mean=%.2f on treated", df.loc[df["night_alert"]>0,"log_cell_reached"].mean())

    # ── (F) Cell-connectivity × car-commuter combined dosage ─────────────────
    # cell_car_pop_c = car_total_c × cell_share_c
    #   = "car commuters with cellular service" — the most targeted WEA dose:
    #   people who (1) received the WEA blast at night, and (2) drive to work
    # car_share and cell_share have Corr = −0.110 (negatively correlated:
    #   urban → low car, high cell; rural → high car, moderate cell)
    # so the product std (9.9pp) > car alone (7.5pp) — more county-level variation
    car_cell = car[["fips","car_total"]].merge(cell[["fips","cell_share"]], on="fips", how="inner")
    car_cell["cell_car_pop"] = car_cell["car_total"].astype(float) * car_cell["cell_share"]

    alerts_cellcar = alerts_raw.merge(car_cell[["fips","cell_car_pop"]],
                                       left_on="county_fips", right_on="fips", how="left")
    alerts_cellcar["cell_car_pop"] = alerts_cellcar["cell_car_pop"].fillna(0)
    night_alerts_cellcar = alerts_cellcar[alerts_cellcar["is_night"]].copy()

    alert_day_cellcar = (night_alerts_cellcar.groupby("date")["cell_car_pop"]
                         .sum().reset_index()
                         .rename(columns={"cell_car_pop": "cell_car_reached"}))
    alert_day_cellcar["date"] = pd.to_datetime(alert_day_cellcar["date"])

    df = df.merge(alert_day_cellcar, on="date", how="left")
    df["cell_car_reached"] = df["cell_car_reached"].fillna(0)
    df["cell_car_reached"] *= (df["night_alert"] > 0).astype(float)
    df["log_cell_car_reached"] = np.log1p(df["cell_car_reached"])
    has_cellcar = True   # both (E) and (F) are ready
    log.info("Built log_cell_car_reached (car_commuters × cell_share): mean=%.2f on treated",
             df.loc[df["night_alert"]>0,"log_cell_car_reached"].mean())
else:
    has_cell    = False
    has_cellcar = False
    log.warning("Cell connectivity file not found (%s); skipping (E) and (F)", CELL_PATH)
    log.warning("Run code/01f_fetch_cell_connectivity.py first.")

# ── (G) Population-weighted coverage (density-based, not subscription) ───────
# coverage_fraction_c = f(pop_density_c) calibrated to FCC Form 477 aggregates
# coverage_pop_c = population_c × coverage_fraction_c
# National pop-wtd average = 97.5%; range 73%–99.5% across counties
# Addresses the subscription-vs-coverage distinction: WEA needs no plan,
# only tower proximity. Dense counties (NYC) → 99.5%; frontier → 73%.
has_cov = False
if COV_PATH.exists():
    log.info("Loading density-based coverage weight …")
    cov = pd.read_parquet(COV_PATH, columns=["fips","coverage_fraction","coverage_pop"])
    cov["fips"] = cov["fips"].astype(str).str.zfill(5)
    log.info("  Coverage fraction: mean=%.1f%%  range %.1f%%–%.1f%%",
             cov.coverage_fraction.mean()*100,
             cov.coverage_fraction.min()*100,
             cov.coverage_fraction.max()*100)

    # Aggregate: sum coverage_pop across alert counties per date
    pop_data3 = cov[["fips","coverage_pop"]].copy()
    alerts_cov = alerts_raw.merge(pop_data3, left_on="county_fips", right_on="fips", how="left")
    alerts_cov["coverage_pop"] = alerts_cov["coverage_pop"].fillna(0)
    night_alerts_cov = alerts_cov[alerts_cov["is_night"]].copy()

    alert_day_cov = (night_alerts_cov.groupby("date")["coverage_pop"]
                     .sum().reset_index()
                     .rename(columns={"coverage_pop": "cov_pop_reached"}))
    alert_day_cov["date"] = pd.to_datetime(alert_day_cov["date"])
    df = df.merge(alert_day_cov, on="date", how="left")
    df["cov_pop_reached"] = df["cov_pop_reached"].fillna(0)
    df["cov_pop_reached"] *= (df["night_alert"] > 0).astype(float)
    df["log_cov_reached"] = np.log1p(df["cov_pop_reached"])
    has_cov = True
    log.info("Built log_cov_reached: mean=%.2f on treated",
             df.loc[df["night_alert"]>0,"log_cov_reached"].mean())
else:
    log.warning("Coverage weight file not found; run code/01g_build_coverage_weight.py")

# Diagnostics
treated = df[df["night_alert"] > 0]
log.info("\n=== Dosage variable diagnostics (treated county-days only) ===")
log.info("N treated county-days: %d", len(treated))
if "log_breadth" in df.columns:
    log.info("Corr(log_breadth, log_car_reached):  %.3f",
             treated["log_breadth"].corr(treated["log_car_reached"]))
    log.info("Corr(log_breadth, log_pop_reached):  %.3f",
             treated["log_breadth"].corr(treated["log_pop_reached"]))
    if has_cell:
        log.info("Corr(log_breadth, log_cell_reached): %.3f",
                 treated["log_breadth"].corr(treated["log_cell_reached"]))
log.info("Corr(log_car_reached, log_pop_reached): %.3f",
         treated["log_car_reached"].corr(treated["log_pop_reached"]))
if has_cell:
    log.info("Corr(log_pop_reached, log_cell_reached):    %.3f",
             treated["log_pop_reached"].corr(treated["log_cell_reached"]))
if has_cellcar:
    log.info("Corr(log_breadth, log_cell_car_reached):    %.3f",
             treated["log_breadth"].corr(treated["log_cell_car_reached"]))
    log.info("Corr(log_pop_reached, log_cell_car_reached):%.3f",
             treated["log_pop_reached"].corr(treated["log_cell_car_reached"]))
    log.info("Corr(log_car_reached, log_cell_car_reached):%.3f",
             treated["log_car_reached"].corr(treated["log_cell_car_reached"]))

if has_cov:
    log.info("Corr(log_pop_reached, log_cov_reached):  %.3f",
             treated["log_pop_reached"].corr(treated["log_cov_reached"]))
    log.info("Corr(log_breadth,     log_cov_reached):  %.3f",
             treated["log_breadth"].corr(treated["log_cov_reached"]))

log.info("Within-county variance:")
extra = []
if has_cell:    extra.append("log_cell_reached")
if has_cellcar: extra.append("log_cell_car_reached")
if has_cov:     extra.append("log_cov_reached")
for v in ["log_breadth","log_car_reached","log_pop_reached"] + extra:
    if v in df.columns:
        within_var = df.groupby("fips")[v].var().mean()
        log.info("  %s: %.4f", v, within_var)

gc.collect()

# ── Run regressions ──────────────────────────────────────────────────────────
results = []

def run_spec(label, treatment, outcome, weight_col=None):
    sub = df.dropna(subset=[treatment, outcome])
    if weight_col and weight_col not in sub.columns:
        weight_col = None
    r = fe_ols_from_panel(sub, outcome, treatment=treatment, controls=BASE_CTRL,
                          county=True, dm=True, cluster_col="state_code",
                          weights_col=weight_col, label=label)
    if "error" not in r:
        r["label"] = label
        r["treatment"] = treatment
        r["outcome"] = outcome
        results.append(r)
        log.info("  %-45s β=%+.4f  se=%.4f  p=%.3f",
                 label, r["coef"], r["se"], r["pval"])
    else:
        log.warning("  Error in %s: %s", label, r.get("error","?"))

log.info("\n=== Count outcome (fatals_next_commute) ===")
run_spec("(A) Binary [count]",         "night_alert",     "fatals_next_commute")
if "log_breadth" in df.columns:
    run_spec("(B) Log-breadth [count]", "log_breadth",     "fatals_next_commute")
run_spec("(C) Log-pop-reached [count]","log_pop_reached",  "fatals_next_commute")
run_spec("(D) Log-car-reached [count]",     "log_car_reached",      "fatals_next_commute")
if has_cell:
    run_spec("(E) Log-cell-reached [count]",   "log_cell_reached",     "fatals_next_commute")
if has_cellcar:
    run_spec("(F) Log-cell×car-reached [count]", "log_cell_car_reached","fatals_next_commute")
if has_cov:
    run_spec("(G) Log-coverage-pop [count]",      "log_cov_reached",    "fatals_next_commute")

if has_comb and has_pop:
    log.info("\n=== WLS rate (combined/100k) ===")
    run_spec("(A) Binary [WLS]",            "night_alert",      "comb_rate", "log_pop")
    if "log_breadth" in df.columns:
        run_spec("(B) Log-breadth [WLS]",   "log_breadth",      "comb_rate", "log_pop")
    run_spec("(C) Log-pop-reached [WLS]",   "log_pop_reached",  "comb_rate", "log_pop")
    run_spec("(D) Log-car-reached [WLS]",      "log_car_reached",      "comb_rate", "log_pop")
    if has_cell:
        run_spec("(E) Log-cell-reached [WLS]",   "log_cell_reached",     "comb_rate", "log_pop")
    if has_cellcar:
        run_spec("(F) Log-cell×car-reached [WLS]", "log_cell_car_reached","comb_rate", "log_pop")
    if has_cov:
        run_spec("(G) Log-coverage-pop [WLS]",      "log_cov_reached",    "comb_rate", "log_pop")

# ── Save ─────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = OUTPUT_TABS / "reg_dosage_extended.csv"
out.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
log.info("\n=== Summary: which dosage variable works best? ===")
for row in results:
    sig = "***" if row["pval"]<0.01 else "**" if row["pval"]<0.05 else "*" if row["pval"]<0.10 else "n.s."
    log.info("  %-45s β=%+.4f  p=%.3f  %s", row["label"], row["coef"], row["pval"], sig)
