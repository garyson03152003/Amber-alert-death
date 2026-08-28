"""
fig10_twfe2_event_study.py
TWFE2 event study: county×year FE + lagged fatalities.
Two-panel figure: count (left) and WLS rate (right).
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_FIGS, OUTPUT_TABS
warnings.filterwarnings("ignore")

OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUTPUT_TABS / "reg_event_study_twfe2.csv")
k_vals = sorted(df["k"].unique())

specs_info = [
    ("count_ctyYr_lagFat",       "Count (TWFE2 + lag fatals)", "#2166ac"),
    ("rate_ctyYr_lagFat_WLS",    "Rate per 100k, WLS (TWFE2 + lag)", "#d6604d"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.subplots_adjust(wspace=0.35)
C_GRID = "#e0e0e0"

def stars(p):
    return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

for ax, (spec, title, color) in zip(axes, specs_info):
    sub = df[df["spec"] == spec].set_index("k").loc[k_vals]
    coefs = sub["coef"].values
    ses   = sub["se"].values
    pvals = sub["pval"].values

    x = np.arange(len(k_vals))
    ci95_lo = coefs - 1.96*ses
    ci95_hi = coefs + 1.96*ses

    ax.fill_between(x, ci95_lo, ci95_hi, alpha=0.18, color=color, label="95% CI")
    ax.plot(x, coefs, "o-", color=color, markersize=6, linewidth=2, label=r"$\hat\beta_k$")

    # Stars above peaks
    for i, (c, p) in enumerate(zip(coefs, pvals)):
        st = stars(p)
        if st:
            ax.text(x[i], c + 1.96*ses[i] + abs(c)*0.05 + 0.001,
                    st, ha="center", va="bottom", fontsize=9, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(k_vals.index(0), color="gray", linewidth=0.8, linestyle=":",
               label="Alert night (k=0)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_vals], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Days relative to night alert (k)", fontsize=10)
    ax.set_ylabel(r"$\hat\beta_k$", fontsize=11)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.yaxis.grid(True, color=C_GRID, zorder=0)
    ax.set_axisbelow(True)

fig.suptitle("Event Study: TWFE2 Specification (County×Year FE + Lagged Fatalities)",
             fontsize=11, fontweight="bold", y=1.01)

for ext in ["png", "pdf"]:
    out = OUTPUT_FIGS / f"fig10_twfe2_event_study.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")

plt.close(fig)
