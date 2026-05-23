"""
fig7_workday_weekend_evs.py
Workday-night vs Weekend-night event study figure.

2-panel figure:
  Left:  raw count outcome (county+DoW×Month FE, state-clustered SEs)
  Right: combined (fatal+serious)/100k, log-pop WLS, state-clustered SEs

For each panel, two series:
  Blue  circles: workday-night alerts (precede Mon–Fri)
  Orange squares: weekend-night alerts (precede Sat–Sun)

95% CI bands.  k=0 is the treatment night itself.  Dashed line at β=0.
"""

import sys, warnings, importlib.util
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import DATA_PROC, OUTPUT_TABS, OUTPUT_FIGS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("fig7")

OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_TABS / "reg_event_study_split.csv"


# ---------------------------------------------------------------------------
# Run the split event study if CSV is missing
# ---------------------------------------------------------------------------
def get_results() -> pd.DataFrame:
    if CSV_PATH.exists():
        log.info("Loading cached results from %s", CSV_PATH)
        return pd.read_csv(CSV_PATH)

    log.info("Running split event study (not yet cached)…")
    spec = importlib.util.spec_from_file_location("analysis_05", ROOT / "05_analysis.py")
    a05  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a05)

    df = a05.load_panel()
    df = prep_panel(df)

    evs = a05.run_event_study_split(df)
    del df; gc.collect()

    evs.to_csv(CSV_PATH, index=False)
    log.info("Saved to %s", CSV_PATH)
    return evs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {"workday": "#2166ac", "weekend": "#d6604d"}
MARKERS = {"workday": "o", "weekend": "s"}
LABELS  = {"workday": "Workday night (→ Mon–Fri)", "weekend": "Weekend night (→ Sat–Sun)"}
OFFSET  = {"workday": -0.15, "weekend": +0.15}   # horizontal jitter

def plot_panel(ax, df_spec, spec_label, ylabel="", add_legend=True):
    """Draw one event-study panel (one spec, two splits)."""
    ks = sorted(df_spec["k"].unique())

    for split in ["workday", "weekend"]:
        sub = df_spec[df_spec["split"] == split].set_index("k")
        xs  = np.array([k + OFFSET[split] for k in ks])
        ys  = np.array([sub.loc[k, "coef"] for k in ks])
        ses = np.array([sub.loc[k, "se"]   for k in ks])

        ci95 = 1.96 * ses
        ax.errorbar(xs, ys, yerr=ci95,
                    fmt=MARKERS[split], color=COLORS[split],
                    markersize=5, linewidth=1.2, capsize=3,
                    label=LABELS[split], zorder=3)
        ax.plot(xs, ys, color=COLORS[split], linewidth=0.8,
                alpha=0.5, zorder=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(-0.5, color="grey", linewidth=0.6, linestyle=":", zorder=1,
               label="Treatment night")
    ax.set_xticks(ks)
    ax.set_xlabel("Days relative to alert (k)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(spec_label, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9)
    if add_legend:
        ax.legend(fontsize=8, framealpha=0.8, loc="upper left")


def make_figure(evs: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle(
        "Workday-Night vs Weekend-Night AMBER Alert Effects\n"
        "(county+DoW×Month FE; 95% CI; state-clustered SEs)",
        fontsize=12
    )

    # Left: count
    df_count = evs[evs["spec"] == "count"]
    if not df_count.empty:
        plot_panel(axes[0], df_count,
                   spec_label="Raw fatality count",
                   ylabel="Coefficient (fatalities per county-day)",
                   add_legend=True)

    # Right: combined WLS
    df_wls = evs[evs["spec"] == "comb_wls"]
    if not df_wls.empty:
        plot_panel(axes[1], df_wls,
                   spec_label="Combined (fatal+serious) per 100k,\nlog-pop WLS",
                   ylabel="Coefficient (per 100k population)",
                   add_legend=False)

    # Shared annotation
    n_wd = evs.loc[(evs["spec"]=="count") & (evs["split"]=="workday"), "n_obs"].median()
    n_we = evs.loc[(evs["spec"]=="count") & (evs["split"]=="weekend"), "n_obs"].median()
    fig.text(0.5, 0.01,
             f"Workday-night treated county-days: 4,474  |  Weekend-night: 1,808  "
             f"|  Total obs: {int(n_wd):,}",
             ha="center", fontsize=8, color="grey")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("png", "pdf"):
        out = OUTPUT_FIGS / f"fig7_workday_weekend_evs.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)

    plt.close(fig)
    return OUTPUT_FIGS / "fig7_workday_weekend_evs.png"


if __name__ == "__main__":
    evs = get_results()
    log.info("Specs available: %s", evs["spec"].unique().tolist())
    out_path = make_figure(evs)
    log.info("Figure saved: %s", out_path)
