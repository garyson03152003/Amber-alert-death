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
# Fig 1: Event study  (reads pre-computed reg_event_study.csv)
# ---------------------------------------------------------------------------

def event_study(panel: pd.DataFrame) -> None:
    log.info("Fig 1: Event study")
    es_path = OUTPUT_TABS / "reg_event_study.csv"
    if not es_path.exists():
        log.warning("reg_event_study.csv not found — re-computing (slow).")
        # Fallback: re-compute using timing-aligned outcome (raw count only)
        from analysis_lib import prep_panel
        from analysis_lib import fe_ols_from_panel as _ols
        df = prep_panel(panel.copy())
        df["fatals_next_commute"] = df["fatals_t1"]
        mask_mid = df["night_band"].isin(["deep_night", "late_night"])
        df.loc[mask_mid, "fatals_next_commute"] = df.loc[mask_mid, "fatals_t0"]
        df = df.sort_values(["fips", "date"]).copy()
        rows = []
        for k in range(-3, 4):
            col = f"aligned_k{k:+d}"
            df[col] = df.groupby("fips")["fatals_next_commute"].shift(-k)
            sub = df.dropna(subset=[col]).copy()
            r = _ols(sub, col, county=True, dm=True,
                     cluster_col="state_code", label=f"k={k:+d}")
            r["k"] = k
            r["spec"] = "count"
            rows.append(r)
        est = pd.DataFrame(rows)
    else:
        est = pd.read_csv(es_path)

    est = est.dropna(subset=["coef"])

    # Detect window from data (e.g. -5..+5 or -3..+3)
    ks_all   = sorted(est["k"].unique()) if "k" in est.columns else list(range(-3, 4))
    k_min, k_max = int(min(ks_all)), int(max(ks_all))

    def _xtick_labels(ks):
        return [f"k={k}" if k != 0 else "k=0\n(alert)" for k in ks]

    # Spec groups: (base_key, odm_key, y-axis label, panel title)
    spec_groups = [
        ("count", "count_outDM",
         "Additional fatalities (county-day, FE-adjusted)",
         "Raw fatality count"),
        ("comb_rate_logWLS", "comb_rate_logWLS_outDM",
         "Combined injuries per 100k (log-pop WLS)",
         "Combined (fatal + serious inj.) per 100k, log-pop WLS"),
    ]
    has_spec_col = "spec" in est.columns
    available_specs = set(est["spec"].unique()) if has_spec_col else set()

    has_base = any(s in available_specs for s, *_ in spec_groups)
    has_odm  = any(s in available_specs for _, s, *_ in spec_groups)

    ORANGE = "#e08214"

    def _draw_es(ax, sub, color, label):
        sub = sub.sort_values("k")
        ax.errorbar(
            sub["k"], sub["coef"],
            yerr=[sub["coef"] - sub["ci_lo"], sub["ci_hi"] - sub["coef"]],
            fmt="o", color=color, capsize=3, lw=1.4, ms=5, label=label,
        )

    def _decorate_ax(ax, ks, ylabel, title):
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvspan(-0.5, 0.5, alpha=0.07, color=BLUE)
        ax.set_xticks(ks)
        ax.set_xticklabels(_xtick_labels(ks), fontsize=7, rotation=45, ha="right")
        ax.set_xlabel("Days relative to nighttime AMBER Alert", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8, pad=4)
        ax.legend(frameon=False, fontsize=7)

    if has_base and has_odm:
        # 2×2: row = outcome spec, col = baseline vs +outDM
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
        col_titles = ["Baseline (treatment-date DoW×Month FE)",
                      "+ Outcome-date DoW×Month FE (robustness)"]
        for row, (base_key, odm_key, ylabel, row_lbl) in enumerate(spec_groups):
            for col, (spec_key, col_title) in enumerate(
                    [(base_key, col_titles[0]), (odm_key, col_titles[1])]):
                ax = axes[row][col]
                sub = est[est["spec"] == spec_key] if has_spec_col else est
                if not sub.empty:
                    _draw_es(ax, sub, BLUE if col == 0 else ORANGE,
                             "Point est. (95% CI, state-cl.)")
                _decorate_ax(ax, ks_all, ylabel if col == 0 else "",
                             col_title if row == 0 else "")
                if col == 0:
                    ax.text(-0.18, 0.5, row_lbl, transform=ax.transAxes,
                            fontsize=7, rotation=90, va="center", ha="center")
        fig.suptitle("Fig. 1  Event Study Around Nighttime AMBER Alerts "
                     "(Timing-Aligned Outcome)", fontsize=10)
        fig.tight_layout()

    elif has_spec_col and has_base:
        # 1×2: two base specs side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (base_key, _, ylabel, title) in zip(axes, spec_groups):
            sub = est[est["spec"] == base_key] if has_spec_col else est
            if not sub.empty:
                _draw_es(ax, sub, BLUE, "Point est. (95% CI, state-cl.)")
            _decorate_ax(ax, ks_all, ylabel, title)
        fig.suptitle("Fig. 1  Event Study Around Nighttime AMBER Alerts",
                     fontsize=11, y=1.01)
        fig.tight_layout()

    else:
        # Single-spec fallback
        sub = est.sort_values("k")
        fig, ax = plt.subplots(figsize=(8, 4))
        _draw_es(ax, sub, BLUE, "Point estimate (95% CI, state-clustered)")
        _decorate_ax(ax, ks_all,
                     "Additional traffic fatalities (county-day, FE-adjusted)",
                     "Fig. 1  Event Study Around Nighttime AMBER Alerts")

    save(fig, "fig1_event_study")


# ---------------------------------------------------------------------------
# Fig 2: Alert timing histogram
# ---------------------------------------------------------------------------

def alert_timing(amber: pd.DataFrame) -> None:
    log.info("Fig 2: Alert timing")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    hours = amber["hour_local"].dropna()
    ax.hist(hours, bins=24, range=(0, 24), color=BLUE, edgecolor="white", lw=0.4)
    ax.axvspan(22, 24, alpha=0.15, color=RED, label="Nighttime window (10pm–6am)")
    ax.axvspan(0,   6, alpha=0.15, color=RED)
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
# Fig 6: Daytime vs Nighttime alert comparison
# ---------------------------------------------------------------------------

def daytime_placebo_plot() -> None:
    log.info("Fig 6: Daytime alert placebo")
    dp_path = OUTPUT_TABS / "reg_daytime_placebo.csv"
    if not dp_path.exists():
        log.warning("reg_daytime_placebo.csv not found — skipping Fig 6.")
        return
    dp = pd.read_csv(dp_path).dropna(subset=["coef"])

    label_order = [
        "Same-day (daytime alert)",
        "Next-day (daytime alert)",
        "Next-commute (night alert) [ref]",
    ]
    colors = [RED, RED, BLUE]
    dp["_order"] = dp["model"].map({l: i for i, l in enumerate(label_order)})
    dp = dp.dropna(subset=["_order"]).sort_values("_order")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axvline(0, color="black", lw=0.8, ls="--")

    for i, (_, row) in enumerate(dp.iterrows()):
        col = BLUE if "night" in row["model"].lower() else RED
        ax.errorbar(
            row["coef"], i,
            xerr=[[row["coef"] - row["ci_lo"]], [row["ci_hi"] - row["coef"]]],
            fmt="o", color=col, capsize=4, lw=1.4, ms=7,
        )

    ax.set_yticks(range(len(dp)))
    ax.set_yticklabels(dp["model"].tolist(), fontsize=9)
    ax.set_xlabel("Coefficient (county + DoW×Month FE; state-clustered SE)")
    ax.set_title("Fig. 6  Daytime Alert Placebo vs Nighttime Alert Effect")
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=BLUE,ms=8,label="Nighttime (treatment)"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=RED, ms=8,label="Daytime (placebo)"),
    ], frameon=False, loc="lower right")
    save(fig, "fig6_daytime_placebo")


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
    daytime_placebo_plot()

    log.info("All figures saved to output/figures/")


if __name__ == "__main__":
    main()
