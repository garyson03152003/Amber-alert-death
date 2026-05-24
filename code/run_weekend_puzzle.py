"""
run_weekend_puzzle.py
Investigates why weekend-night AMBER Alerts show a larger (and significant)
effect on next-day traffic fatalities compared to workday-night alerts.

Hypotheses tested:
  H1. Alert timing: Weekend-night alerts fire earlier in the evening (10pm),
      while weekday-night alerts fire deeper into the night (2–5am). The 10pm
      alerts catch people still awake and driving, causing same-night crashes
      in the early morning hours (captured in fatals_t0/aligned outcome).

  H2. Baseline crash level: Weekend days have higher baseline fatals; does the
      rate spec (÷ population) fully absorb the difference?

  H3. Alert breadth: Are weekend-night alerts broader (more counties) and
      hence more disruptive?

  H4. Deep-night restriction: If we restrict to deep/late-night alerts only
      (midnight–5am), does the workday effect become larger than the weekend
      effect (consistent with sleep disruption)?

  H5. Urban/rural composition: Weekend alerts may disproportionately cover
      rural counties (more recreational driving, longer distances).

Output:
  output/tables/reg_weekend_puzzle.csv
  Printed diagnostics to stdout
"""
import sys, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel, fe_ols_from_panel

warnings.filterwarnings("ignore")
log = get_logger("weekend_puzzle")

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

# ── Load panel ───────────────────────────────────────────────────────────────
log.info("Loading panel …")
import importlib.util
spec = importlib.util.spec_from_file_location("a05", Path(__file__).parent / "05_analysis.py")
a05  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

df_raw = a05.load_panel()
df_raw = prep_panel(df_raw)
df_raw = a05.add_aligned_outcome(df_raw)
df_raw.sort_values(["fips","date"], inplace=True)

# Population weights
if "population" in df_raw.columns:
    cpop = df_raw.groupby("fips")["population"].transform("mean")
    pop  = df_raw["population"].fillna(cpop)
    df_raw["log_pop"]  = np.log(pop.clip(lower=1))
    df_raw["pop_100k"] = pop / 100_000
    has_pop = True
else:
    has_pop = False

if "combined_next_commute" in df_raw.columns and has_pop:
    df_raw["comb_rate"] = df_raw["combined_next_commute"] / df_raw["pop_100k"]
    has_comb = True
else:
    has_comb = False

WEATHER = [c for c in ["prcp_mm","tmax_c"] if c in df_raw.columns
           and df_raw[c].notna().mean() > 0.01]
HOL     = [c for c in ["is_holiday"] if c in df_raw.columns]
BASE_CTRL = HOL + WEATHER

# ── 1. Alert timing descriptives ─────────────────────────────────────────────
log.info("\n=== H1: Alert timing by weekday vs weekend ===")
alerts_raw = pd.read_parquet(DATA_PROC / "amber_alerts_clean.parquet")
alerts_raw["issued_local"] = pd.to_datetime(alerts_raw["issued_local"])
alerts_raw["dow"] = alerts_raw["issued_local"].dt.dayofweek
night_alerts = alerts_raw[alerts_raw["is_night"]].copy()
night_alerts["dow_type"] = night_alerts["dow"].apply(
    lambda x: "weekend" if x in [4,5] else "weekday")

# Night hours: 22,23,0,1,2,3,4,5 — map to circular order (22→-2, 23→-1, 0→0, …)
def circular_hour(h):
    return h if h <= 12 else h - 24   # 22→-2, 23→-1, 0→0 … 5→5

night_alerts["hour_circ"] = night_alerts["hour_local"].apply(circular_hour)
h1 = night_alerts.groupby("dow_type")["hour_circ"].describe()
log.info("  Night alert timing (circular: -2=10pm, -1=11pm, 0=midnight, …5=5am):")
log.info("  %s", h1.to_string())

log.info("\n  Night band breakdown:")
nb = night_alerts.groupby(["dow_type","night_band"]).size().unstack(fill_value=0)
nb["pct_early"] = nb["early_night"] / nb.sum(axis=1) * 100
log.info("  %s", nb.to_string())

# ── 2. Baseline crash levels ──────────────────────────────────────────────────
log.info("\n=== H2: Baseline crash levels ===")
no_alert = df_raw[df_raw["night_alert"]==0]
wd_base  = no_alert[no_alert["dow"].isin([0,1,2,3,6])]["fatals_next_commute"].mean()
we_base  = no_alert[no_alert["dow"].isin([4,5])]["fatals_next_commute"].mean()
log.info("  Non-alert baseline fatals_next_commute:")
log.info("    Workday nights (Sun-Thu): %.4f", wd_base)
log.info("    Weekend nights (Fri-Sat): %.4f", we_base)
log.info("    Weekend premium: %.1f%%", (we_base/wd_base - 1)*100)

if has_comb and has_pop:
    wd_base_r = no_alert[no_alert["dow"].isin([0,1,2,3,6])]["comb_rate"].mean()
    we_base_r = no_alert[no_alert["dow"].isin([4,5])]["comb_rate"].mean()
    log.info("  Non-alert baseline comb_rate (after ÷ population):")
    log.info("    Workday nights: %.5f", wd_base_r)
    log.info("    Weekend nights: %.5f", we_base_r)
    log.info("    Weekend premium after rate adjustment: %.1f%%",
             (we_base_r/wd_base_r - 1)*100)

# ── 3. Alert breadth (H3) ─────────────────────────────────────────────────────
log.info("\n=== H3: Alert breadth by weekday vs weekend ===")
# Count counties per alert event
event_cols = [c for c in ["alert_id","county_fips","night_band","dow"] if c in night_alerts.columns]
if "alert_id" in night_alerts.columns:
    breadth = night_alerts.groupby(["alert_id","dow_type"])["county_fips"].count().reset_index()
    breadth.columns = ["alert_id","dow_type","n_counties"]
    log.info("  Mean counties per alert:")
    log.info("  %s", breadth.groupby("dow_type")["n_counties"].describe().to_string())

# ── 4. Deep-night restriction (H4) ───────────────────────────────────────────
log.info("\n=== H4: Deep-night-only restriction (midnight–5am alerts only) ===")

def add_wd_we_split(df, restrict_deep=False):
    """Add workday/weekend night alert indicators; optionally restrict to deep/late night."""
    df = df.copy()
    early_night   = df["night_band"] == "early_night"
    midnight_band = df["night_band"].isin(["deep_night","late_night"])

    if restrict_deep:
        # Only midnight+late-night alerts (midnight–5am)
        alert_mask = df["night_alert"].astype(bool) & midnight_band
    else:
        alert_mask = df["night_alert"].astype(bool)

    # Workday: alert before a workday morning (Sun night = dow6 → Mon, …, Thu night = dow3 → Fri)
    #          midnight-band on Mon(0)→Tue, Tue(1)→Wed, Wed(2)→Thu, Thu(3)→Fri, Fri(4)→Sat [workday for midnight]
    early_workday    = early_night   & df["dow"].isin([0,1,2,3,6])
    midnight_workday = midnight_band & df["dow"].isin([0,1,2,3,4])
    workday_mask     = early_workday | midnight_workday

    df["night_alert_workday"] = (alert_mask & workday_mask).astype(int)
    df["night_alert_weekend"] = (alert_mask & ~workday_mask).astype(int)
    return df

results = []

for restrict_deep in [False, True]:
    label_suffix = " [deep-night only]" if restrict_deep else " [all night]"
    df2 = add_wd_we_split(df_raw, restrict_deep=restrict_deep)
    n_wd = int(df2["night_alert_workday"].sum())
    n_we = int(df2["night_alert_weekend"].sum())
    log.info("  %s: %d workday, %d weekend treated county-days", label_suffix, n_wd, n_we)

    for spec_name, outcome, weight_col in [
        ("count", "fatals_next_commute", None),
        ("WLS",   "comb_rate" if has_comb else "fatals_next_commute",
                  "log_pop" if has_pop else None),
    ]:
        if outcome not in df2.columns:
            continue
        sub = df2.dropna(subset=[outcome])
        if weight_col and weight_col not in sub.columns:
            weight_col = None

        # Workday effect (controlling for weekend)
        r_wd = fe_ols_from_panel(
            sub, outcome, treatment="night_alert_workday",
            controls=["night_alert_weekend"] + BASE_CTRL,
            county=True, dm=True, cluster_col="state_code",
            weights_col=weight_col,
            label=f"workday{label_suffix} [{spec_name}]")
        if "error" not in r_wd:
            r_wd["split"] = "workday"
            r_wd["spec"] = spec_name
            r_wd["restrict_deep"] = restrict_deep
            results.append(r_wd)

        # Weekend effect (controlling for workday)
        r_we = fe_ols_from_panel(
            sub, outcome, treatment="night_alert_weekend",
            controls=["night_alert_workday"] + BASE_CTRL,
            county=True, dm=True, cluster_col="state_code",
            weights_col=weight_col,
            label=f"weekend{label_suffix} [{spec_name}]")
        if "error" not in r_we:
            r_we["split"] = "weekend"
            r_we["spec"] = spec_name
            r_we["restrict_deep"] = restrict_deep
            results.append(r_we)

# ── 5. Urban/rural composition (H5) ──────────────────────────────────────────
log.info("\n=== H5: Urban/rural composition of weekend vs weekday alerts ===")
if "pop_quartile" in df_raw.columns:
    wd_events = df_raw[(df_raw["night_alert"]>0) & (df_raw["dow"].isin([0,1,2,3,6]))]
    we_events = df_raw[(df_raw["night_alert"]>0) & (df_raw["dow"].isin([4,5]))]
    log.info("  Population quartile distribution (alert days):")
    log.info("  Workday nights:")
    log.info("  %s", wd_events["pop_quartile"].value_counts(normalize=True).round(3).to_string())
    log.info("  Weekend nights:")
    log.info("  %s", we_events["pop_quartile"].value_counts(normalize=True).round(3).to_string())

# ── 6. Print summary table ───────────────────────────────────────────────────
log.info("\n=== Summary: workday vs weekend night effect ===")
log.info("  All-night:                    workday β | weekend β  | weekend/workday ratio")
log.info("  ─────────────────────────────────────────────────────────────────────")

out = pd.DataFrame([r for r in results if isinstance(r, dict) and "coef" in r])

for spec_name in ["count","WLS"]:
    for restrict in [False, True]:
        sub_r = out[(out["spec"]==spec_name) & (out["restrict_deep"]==restrict)]
        if sub_r.empty:
            continue
        r_wd = sub_r[sub_r["split"]=="workday"]
        r_we = sub_r[sub_r["split"]=="weekend"]
        if r_wd.empty or r_we.empty:
            continue
        r_wd = r_wd.iloc[0]
        r_we = r_we.iloc[0]

        def stars(p):
            return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

        label = f"  [{spec_name}] {'deep-night' if restrict else 'all-night  '}"
        ratio = abs(r_we["coef"]) / max(abs(r_wd["coef"]), 1e-9)
        log.info("%s  wd=%+.4f%s  we=%+.4f%s  ratio=%.2fx",
                 label,
                 r_wd["coef"], stars(r_wd["pval"]),
                 r_we["coef"], stars(r_we["pval"]),
                 ratio)

# Save
out_path = OUTPUT_TABS / "reg_weekend_puzzle.csv"
out.to_csv(out_path, index=False)
log.info("\nSaved → %s", out_path)
