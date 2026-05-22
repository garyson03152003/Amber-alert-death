"""
06_figures.py — Generate all paper figures.

Imports the regression function from 05_analysis to ensure consistent FE handling.

Figures:
  Fig 1: Event study  (β at t−3…t+3 around night alert)
  Fig 2: Alert timing histogram
  Fig 3: Heterogeneity forest plot
  Fig 4: Geographic bar chart (top states by night alert count)
  Fig 5: Placebo coefficient plot

Run: python code/06_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_FIGS, OUTPUT_TABS, EVENT_WINDOW
from utils import get_logger

# Reuse the efficient two-way FE estimator from 05_analysis
from analysis_lib import fe_ols_from_panel

log = get_logger("06_figures")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
BLUE = "#2166ac"
RED  = "#d6604d"
GRAY = "#888888"


def save(fig, stem: str) -> None:
    out = OUTPUT_FIGS / stem
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    log.info("Saved %s", out.with_suffix(".png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: Event study
# ---------------------------------------------------------------------------

def event_study(panel: pd.DataFrame) -> None:
    log.info("Fig 1: Event study")
    window = range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1)
    estimates = []

    for k in window:
        # Build outcome for this lag: shift fatals_t0 by k within county
        # negative k → look back (placebo), positive → look forward
        panel["_yk"] = panel.groupby("fips")["fatals_t0"].shift(-k)
        sub = panel.dropna(subset=["_yk"]).copy()
        r = fe_ols_from_panel(sub, "_yk", county=True, dm=True,
                              label=f"k={k}")
        estimates.append({
            "k": k,
            "coef":  r.get("coef", np.nan),
            "ci_lo": r.get("ci_lo", np.nan),
            "ci_hi": r.get("ci_hi", np.nan),
        })
        log.info("  k=%+d  β=%.4f  (%.4f, %.4f)",
                 k, r.get("coef",np.nan), r.get("ci_lo",np.nan), r.get("ci_hi",np.nan))

    panel.drop(columns=["_yk"], inplace=True, errors="ignore")

    est = pd.DataFrame(estimates)
    mask = est["coef"].notna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvspan(0.5, 1.5, alpha=0.07, color=BLUE)   # highlight t+1 estimate

    ax.errorbar(
        est.loc[mask, "k"], est.loc[mask, "coef"],
        yerr=[est.loc[mask,"coef"]-est.loc[mask,"ci_lo"],
              est.loc[mask,"ci_hi"]-est.loc[mask,"coef"]],
        fmt="o", color=BLUE, capsize=4, lw=1.5, ms=6,
        label="Point estimate (95% CI)",
    )
    ax.set_xlabel("Days relative to nighttime AMBER Alert")
    ax.set_ylabel("Additional traffic fatalities (county-day, FE-adjusted)")
    ax.set_title("Fig. 1  Event Study Around Nighttime AMBER Alerts")
    ax.set_xticks(list(window))
    ax.set_xticklabels([f"t{k:+d}" if k != 0 else "t (alert night)" for k in window],
                       fontsize=9)
    ax.legend(frameon=False)
    save(fig, "fig1_event_study")


# ---------------------------------------------------------------------------
# Fig 2: Alert timing histogram
# ---------------------------------------------------------------------------

def alert_timing(amber: pd.DataFrame) -> None:
    log.info("Fig 2: Alert timing")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    hours = amber["hour_local"].dropna()
    ax.hist(hours, bins=24, range=(0, 24), color=BLUE, edgecolor="white", lw=0.4)
    ax.axvspan(22, 24, alpha=0.15, color=RED, label="Nighttime window (10pm–5am)")
    ax.axvspan(0,   5, alpha=0.15, color=RED)
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Number of alerts")
    ax.set_title("Fig. 2  AMBER Alert Issuance Times")
    ax.set_xticks(range(0, 25, 2))
    ax.legend(frameon=False)
    save(fig, "fig2_alert_timing")


# ---------------------------------------------------------------------------
# Fig 3: Heterogeneity forest plot
# ---------------------------------------------------------------------------

def heterogeneity_forest() -> None:
    log.info("Fig 3: Heterogeneity forest plot")
    h_path = OUTPUT_TABS / "reg_hetero.csv"
    if not h_path.exists():
        log.warning("Heterogeneity CSV not found — skipping Fig 3.")
        return
    h = pd.read_csv(h_path).dropna(subset=["coef"])

    fig, ax = plt.subplots(figsize=(6, 0.55 * len(h) + 1.5))
    y_pos = range(len(h))
    ax.errorbar(
        h["coef"], list(y_pos),
        xerr=[h["coef"]-h["ci_lo"], h["ci_hi"]-h["coef"]],
        fmt="o", color=BLUE, capsize=4, lw=1.4, ms=7,
    )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(h["model"].tolist(), fontsize=9)
    ax.set_xlabel("Effect on next-day traffic fatalities (county-day)")
    ax.set_title("Fig. 3  Heterogeneity Analysis")
    ax.invert_yaxis()
    save(fig, "fig3_heterogeneity")


# ---------------------------------------------------------------------------
# Fig 4: Geographic bar chart
# ---------------------------------------------------------------------------

def geographic_bar(amber: pd.DataFrame) -> None:
    log.info("Fig 4: Geographic distribution")
    night = amber[amber["is_night"]].copy()
    sc = (night.groupby("state_fips")["alert_id"].nunique()
              .reset_index().rename(columns={"alert_id": "n"})
              .sort_values("n", ascending=False).head(20))
    try:
        import us
        sc["abbr"] = sc["state_fips"].map({s.fips: s.abbr for s in us.states.STATES}
                                          ).fillna(sc["state_fips"])
    except ImportError:
        sc["abbr"] = sc["state_fips"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(sc["abbr"], sc["n"], color=BLUE, edgecolor="white")
    ax.set_xlabel("Unique nighttime AMBER Alerts")
    ax.set_title("Fig. 4  Top 20 States by Nighttime AMBER Alert Count")
    ax.invert_yaxis()
    save(fig, "fig4_geographic")


# ---------------------------------------------------------------------------
# Fig 5: Placebo coefficient plot
# ---------------------------------------------------------------------------

def placebo_plot() -> None:
    log.info("Fig 5: Placebo coefficient plot")
    p_path = OUTPUT_TABS / "reg_placebo.csv"
    if not p_path.exists():
        log.warning("Placebo CSV not found — skipping Fig 5.")
        return
    p = pd.read_csv(p_path).dropna(subset=["coef"])

    label_to_k = {"Placebo: t−1": -1, "Same-day: t": 0,
                  "Main: t+1": 1, "Placebo: t+2": 2}
    p["k"] = p["model"].map(label_to_k)
    p = p.dropna(subset=["k"]).sort_values("k")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvspan(0.5, 1.5, alpha=0.06, color=BLUE)

    ax.errorbar(p["k"], p["coef"],
                yerr=[p["coef"]-p["ci_lo"], p["ci_hi"]-p["coef"]],
                fmt="none", ecolor=GRAY, capsize=4, lw=1.2)

    for _, row in p.iterrows():
        color = RED if "Placebo" in row["model"] else BLUE
        ax.scatter(row["k"], row["coef"], color=color, s=60, zorder=5)

    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels(["t−1\n(placebo)", "t\n(same day)",
                         "t+1\n(main)", "t+2\n(placebo)"])
    ax.set_ylabel("Effect on fatalities (county-day, FE-adjusted)")
    ax.set_title("Fig. 5  Main Estimate and Placebo Tests")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=BLUE,ms=8,label="Main / same-day"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=RED, ms=8,label="Placebo"),
    ], frameon=False)
    save(fig, "fig5_placebo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    panel_path = DATA_PROC / "panel_county_day.parquet"
    amber_path = DATA_PROC / "amber_alerts_clean.parquet"

    if not panel_path.exists():
        log.error("Panel missing — run 04_build_panel.py first.")
        return

    # Import the prep helper to get FE codes
    from analysis_lib import prep_panel
    panel = pd.read_parquet(panel_path)
    panel = prep_panel(panel)

    amber = pd.read_parquet(amber_path) if amber_path.exists() else pd.DataFrame()

    event_study(panel)

    if not amber.empty and "hour_local" in amber.columns:
        alert_timing(amber)

    heterogeneity_forest()

    if not amber.empty:
        geographic_bar(amber)

    placebo_plot()

    log.info("All figures saved to output/figures/")


if __name__ == "__main__":
    main()
