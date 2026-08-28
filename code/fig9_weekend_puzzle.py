"""
fig9_weekend_puzzle.py
Four-panel figure diagnosing the weekend > workday anomaly.

Panel A: Coefficient comparison (WLS) — all-night vs deep-night
Panel B: WLS baseline crash rates by DoW (no-alert days)
Panel C: Alert-hour distributions by weekday vs weekend (stacked bars)
Panel D: Event study k=0 workday vs weekend with CIs
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_FIGS, OUTPUT_TABS
warnings.filterwarnings("ignore")

OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
puzzle = pd.read_csv(OUTPUT_TABS / "reg_weekend_puzzle.csv")
evs    = pd.read_csv(OUTPUT_TABS / "reg_event_study_split.csv")
panel  = pd.read_parquet(DATA_PROC / "panel_county_day.parquet",
                         columns=["fips","date","fatals_t1","night_alert","dow",
                                  "night_band","population"])
panel["date"] = pd.to_datetime(panel["date"])

# WLS results from puzzle table
wls_all    = puzzle[puzzle["spec"]=="WLS"][["split","restrict_deep","coef","se","pval"]].copy()
wls_all_sub = wls_all[~wls_all["restrict_deep"]]
wls_deep    = wls_all[wls_all["restrict_deep"]]

# ── Colours ──────────────────────────────────────────────────────────────────
C_WD = "#2166ac"   # workday blue
C_WE = "#d6604d"   # weekend red
C_GRID = "#e0e0e0"

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.42, wspace=0.38)

# ── Panel A: Coefficient comparison ─────────────────────────────────────────
ax = axes[0, 0]

specs = ["all night", "deep night only\n(midnight–5am)"]
wd_coefs   = [wls_all_sub[wls_all_sub["split"]=="workday"]["coef"].iloc[0],
              wls_deep   [wls_deep["split"]=="workday"]["coef"].iloc[0]]
we_coefs   = [wls_all_sub[wls_all_sub["split"]=="weekend"]["coef"].iloc[0],
              wls_deep   [wls_deep["split"]=="weekend"]["coef"].iloc[0]]
wd_ses     = [wls_all_sub[wls_all_sub["split"]=="workday"]["se"].iloc[0],
              wls_deep   [wls_deep["split"]=="workday"]["se"].iloc[0]]
we_ses     = [wls_all_sub[wls_all_sub["split"]=="weekend"]["se"].iloc[0],
              wls_deep   [wls_deep["split"]=="weekend"]["se"].iloc[0]]
wd_pvals   = [wls_all_sub[wls_all_sub["split"]=="workday"]["pval"].iloc[0],
              wls_deep   [wls_deep["split"]=="workday"]["pval"].iloc[0]]
we_pvals   = [wls_all_sub[wls_all_sub["split"]=="weekend"]["pval"].iloc[0],
              wls_deep   [wls_deep["split"]=="weekend"]["pval"].iloc[0]]

x   = np.arange(len(specs))
w   = 0.32
bar_wd = ax.bar(x - w/2, wd_coefs, width=w, color=C_WD, alpha=0.85, label="Workday night")
bar_we = ax.bar(x + w/2, we_coefs, width=w, color=C_WE, alpha=0.85, label="Weekend night")

# Error bars (95% CI)
ax.errorbar(x - w/2, wd_coefs, yerr=1.96*np.array(wd_ses),
            fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
ax.errorbar(x + w/2, we_coefs, yerr=1.96*np.array(we_ses),
            fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

# Stars
def stars(p): return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
for i, (c, s) in enumerate(zip(wd_coefs, wd_ses)):
    ax.text(i - w/2, c + 1.96*s + 0.001, stars(wd_pvals[i]),
            ha="center", va="bottom", fontsize=9, color=C_WD)
for i, (c, s) in enumerate(zip(we_coefs, we_ses)):
    ax.text(i + w/2, c + 1.96*s + 0.001, stars(we_pvals[i]),
            ha="center", va="bottom", fontsize=9, color=C_WE)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(specs, fontsize=9)
ax.set_ylabel("β (comb/100k)", fontsize=9)
ax.set_title("A. WLS effect by sample restriction", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)

# ── Panel B: Baseline crash rates by DOW (no-alert days) ────────────────────
ax = axes[0, 1]
no_alert = panel[panel["night_alert"] == 0]
if "population" in no_alert.columns:
    cpop  = no_alert.groupby("fips")["population"].transform("mean")
    pop   = no_alert["population"].fillna(cpop).clip(lower=1)
    no_alert = no_alert.copy()
    no_alert["rate"] = no_alert["fatals_t1"] / (pop / 100_000)
    by_dow = no_alert.groupby("dow")["rate"].mean()
else:
    by_dow = no_alert.groupby("dow")["fatals_t1"].mean()

dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
colors = [C_WD]*5 + [C_WE]*2
ax.bar(range(7), by_dow.values, color=colors, alpha=0.85, edgecolor="white")
ax.set_xticks(range(7)); ax.set_xticklabels(dow_labels)
ax.set_ylabel("Crashes per 100k pop", fontsize=9)
ax.set_title("B. Baseline crash rate by day of week\n(no-alert days)", fontsize=10, fontweight="bold")
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)
wd_patch = mpatches.Patch(color=C_WD, label="Workday")
we_patch = mpatches.Patch(color=C_WE, label="Weekend")
ax.legend(handles=[wd_patch, we_patch], fontsize=8)

# ── Panel C: Alert-hour distribution (night window) ─────────────────────────
ax = axes[1, 0]
alerts_raw = pd.read_parquet(DATA_PROC / "amber_alerts_clean.parquet")
alerts_raw["issued_local"] = pd.to_datetime(alerts_raw["issued_local"])
alerts_raw["dow"] = alerts_raw["issued_local"].dt.dayofweek
night_a = alerts_raw[alerts_raw["is_night"]].copy()
night_a["dow_type"] = night_a["dow"].apply(lambda x: "weekend" if x in [4,5] else "weekday")

# Night hours in correct order: 22, 23, 0, 1, 2, 3, 4, 5
night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
hour_labels = ["10pm","11pm","12am","1am","2am","3am","4am","5am"]
wd_counts = [night_a[(night_a["dow_type"]=="weekday") & (night_a["hour_local"]==h)].shape[0]
             for h in night_hours]
we_counts = [night_a[(night_a["dow_type"]=="weekend") & (night_a["hour_local"]==h)].shape[0]
             for h in night_hours]

# Normalize to fractions
wd_frac = np.array(wd_counts) / sum(wd_counts)
we_frac = np.array(we_counts) / sum(we_counts)

x_h = np.arange(len(night_hours))
w_h = 0.35
ax.bar(x_h - w_h/2, wd_frac, width=w_h, color=C_WD, alpha=0.85, label="Workday night")
ax.bar(x_h + w_h/2, we_frac, width=w_h, color=C_WE, alpha=0.85, label="Weekend night")
ax.set_xticks(x_h); ax.set_xticklabels(hour_labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Fraction of alerts", fontsize=9)
ax.set_title("C. Night-alert hour distribution\n(workday vs weekend night)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)

# ── Panel D: Event study k=0 comparison ─────────────────────────────────────
ax = axes[1, 1]
wls_evs = evs[evs["spec"]=="comb_wls"].copy()
k_vals  = sorted(wls_evs["k"].unique())

wd_c = [wls_evs[(wls_evs["k"]==k) & (wls_evs["split"]=="workday")]["coef"].iloc[0] for k in k_vals]
we_c = [wls_evs[(wls_evs["k"]==k) & (wls_evs["split"]=="weekend")]["coef"].iloc[0] for k in k_vals]
wd_s = [wls_evs[(wls_evs["k"]==k) & (wls_evs["split"]=="workday")]["se"].iloc[0] for k in k_vals]
we_s = [wls_evs[(wls_evs["k"]==k) & (wls_evs["split"]=="weekend")]["se"].iloc[0] for k in k_vals]

x_k = np.arange(len(k_vals))
ax.fill_between(x_k,
                np.array(wd_c) - 1.96*np.array(wd_s),
                np.array(wd_c) + 1.96*np.array(wd_s),
                alpha=0.15, color=C_WD)
ax.fill_between(x_k,
                np.array(we_c) - 1.96*np.array(we_s),
                np.array(we_c) + 1.96*np.array(we_s),
                alpha=0.15, color=C_WE)
ax.plot(x_k, wd_c, "o-", color=C_WD, label="Workday night", markersize=5)
ax.plot(x_k, we_c, "s-", color=C_WE, label="Weekend night", markersize=5)
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.axvline(k_vals.index(0), color="gray", linewidth=0.8, linestyle=":")

ax.set_xticks(x_k)
ax.set_xticklabels([f"k={k}" for k in k_vals], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("β (comb/100k)", fontsize=9)
ax.set_title("D. Event study: WLS effect at each lag\n(workday vs weekend night)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.suptitle("The Weekend Puzzle: Why Do Weekend-Night Alerts Cause Larger Effects?",
             fontsize=11, fontweight="bold", y=1.01)

for ext in ["png", "pdf"]:
    out_path = OUTPUT_FIGS / f"fig9_weekend_puzzle.{ext}"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

plt.close(fig)
