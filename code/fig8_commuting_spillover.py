"""
fig8_commuting_spillover.py
Commuting-flow spillover results figure.

Shows β_own and β_spillover with 95% CI bars across three specs:
  (I)   Raw count, county+DoW×Month FE, state-clustered SEs
  (II)  Raw count, TWFE2 (county×year FE + lag), state-clustered SEs
  (III) Combined (fatal+serious)/100k, log-pop WLS, state-clustered SEs

Interpretation:
  β_own      = direct effect of own-county night alert on next-day crashes
  β_spillover = effect of commuter-weighted alerts in NEIGHBOUR counties
                (workers who commuted from a county that had an alert)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_TABS, OUTPUT_FIGS
from utils import get_logger

log = get_logger("fig8")
OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_TABS / "reg_commuting_spillover.csv"


def get_results() -> pd.DataFrame:
    if CSV_PATH.exists():
        log.info("Loading cached results from %s", CSV_PATH)
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError(
        f"Results not found at {CSV_PATH}. "
        "Run code/run_commuting_spillover.py first."
    )


SPEC_LABELS = {
    "count_baseline": "(I) Count\nBaseline FE",
    "count_twfe2":    "(II) Count\nTWFE2",
    "comb_wls":       "(III) Comb./100k\nLog-pop WLS",
}
COLORS = {
    "own":       "#2166ac",   # blue
    "spillover": "#d6604d",   # orange-red
}
LABELS = {
    "own":       "Own-county alert ($\\hat{\\beta}_{own}$)",
    "spillover": "Cross-county spillover ($\\hat{\\beta}_{spill}$)",
}


def make_figure(df: pd.DataFrame):
    specs = ["count_baseline", "count_twfe2", "comb_wls"]
    n_specs  = len(specs)
    n_types  = 2         # own + spillover
    width    = 0.35
    gap      = 0.08
    x_center = np.arange(n_specs)

    offsets = {"own": -(width / 2 + gap / 2), "spillover": +(width / 2 + gap / 2)}

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "AMBER Alert Effects: Own-County vs Commuter-Spillover\n"
        "(ACS county-to-county commuting weights; state-clustered SEs; 95% CI)",
        fontsize=12
    )

    for ctype in ["own", "spillover"]:
        xs, ys, errs, pvs = [], [], [], []
        for i, spec in enumerate(specs):
            row = df[(df["spec"] == spec) & (df["coef_type"] == ctype)]
            if row.empty:
                continue
            coef = float(row["coef"].iloc[0])
            se   = float(row["se"].iloc[0])
            pval = float(row["pval"].iloc[0])
            xs.append(x_center[i] + offsets[ctype])
            ys.append(coef)
            errs.append(1.96 * se)
            pvs.append(pval)

        xs   = np.array(xs)
        ys   = np.array(ys)
        errs = np.array(errs)

        bars = ax.bar(xs, ys, width=width, color=COLORS[ctype],
                      alpha=0.75, label=LABELS[ctype], zorder=3)
        ax.errorbar(xs, ys, yerr=errs, fmt="none",
                    color=COLORS[ctype], linewidth=1.5, capsize=4, zorder=4)

        # Significance stars
        for x, y, e, p in zip(xs, ys, errs, pvs):
            star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            if star:
                ypos = (y + e + 0.0008) if y >= 0 else (y - e - 0.003)
                ax.text(x, ypos, star, ha="center", va="bottom",
                        fontsize=9, color=COLORS[ctype], fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x_center)
    ax.set_xticklabels([SPEC_LABELS[s] for s in specs], fontsize=9)
    ax.set_ylabel("Coefficient (fatalities per county-day)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")

    # Annotation box explaining spillover variable
    note = (
        "Spillover = $\\sum_{j \\neq c}\\, w_{j \\rightarrow c} \\times \\text{Alert}_{j,t}$\n"
        "where $w_{j \\rightarrow c}$ = share of $c$'s workforce commuting from $j$\n"
        "(ACS 2016-2020, 5-year estimates; 119k county-pair flows)"
    )
    ax.text(0.02, 0.97, note, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    for ext in ("png", "pdf"):
        out = OUTPUT_FIGS / f"fig8_commuting_spillover.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)

    plt.close(fig)
    return OUTPUT_FIGS / "fig8_commuting_spillover.png"


if __name__ == "__main__":
    df = get_results()
    log.info("Results:\n%s",
             df[["spec","coef_type","coef","se","pval"]].to_string(index=False))
    out = make_figure(df)
    log.info("Figure: %s", out)
