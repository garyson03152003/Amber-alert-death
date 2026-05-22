"""
06_figures.py — Generate all figures for the paper.

Figures produced:
    Fig 1: Event-study plot  — fatalities in days t-3 through t+3 around alert days
    Fig 2: Alert timing distribution — histogram of alert issuance hour (local time)
    Fig 3: Heterogeneity forest plot — β by night band
    Fig 4: Geographic distribution — state-level alert counts (choropleth)
    Fig 5: Coefficient plot — placebo + main + second-day estimates

Output directory: output/figures/

Run: python code/06_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_FIGS, OUTPUT_TABS, EVENT_WINDOW
from utils import get_logger

log = get_logger("06_figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})
BLUE = "#2166ac"
RED  = "#d6604d"
GRAY = "#888888"


# ---------------------------------------------------------------------------
# Fig 1: Event study
# ---------------------------------------------------------------------------

def event_study(panel: pd.DataFrame) -> None:
    """
    For each alert county-day, compute mean fatalities at lags t-3 to t+3
    relative to the alert, controlling for county and dow×month means.

    We use a regression-based event study: estimate
        fatals_{c,t+k} = β_k · NightAlert_{c,t} + γ_c + δ_{dow×month}
    for k ∈ {-3, …, +3} and plot β_k with 95% CIs.
    """
    log.info("Building event study...")

    # Columns we need
    need = ["fips", "date", "night_alert", "dow_x_month",
            "fatals_t0", "fatals_t1", "fatals_t2", "fatals_tm1"]
    sub = panel[[c for c in need if c in panel.columns]].dropna()

    # Build dataset with fatalities at each event-window lag/lead
    # We compute this by shifting within county
    sub = sub.sort_values(["fips", "date"])

    window = range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1)
    estimates = []

    for k in window:
        # Shift fatals_t0 by k periods within county (negative k = look back)
        sub[f"y_k{k}"] = sub.groupby("fips")["fatals_t0"].shift(-k)

    # For each k, regress y_k ~ night_alert + county_dummies + dow_x_month_dummies
    # We do a quick within-estimator: demean by county, then regress
    sub_wide = sub[["fips", "date", "night_alert", "dow_x_month"]
                   + [f"y_k{k}" for k in window]].dropna()

    # County demeaning
    county_means = sub_wide.groupby("fips")[
        ["night_alert"] + [f"y_k{k}" for k in window]
    ].transform("mean")
    demeaned = sub_wide[["night_alert"] + [f"y_k{k}" for k in window]] - county_means

    # dow×month dummies
    dm_dummies = pd.get_dummies(sub_wide["dow_x_month"], drop_first=True).astype(float)

    import statsmodels.api as sm

    for k in window:
        y = demeaned[f"y_k{k}"].dropna()
        X_raw = demeaned.loc[y.index, "night_alert"]
        dm = dm_dummies.loc[y.index]
        X = sm.add_constant(pd.concat([X_raw, dm], axis=1).astype(float))
        try:
            res = sm.OLS(y, X).fit(
                cov_type="cluster", cov_kwds={"groups": sub_wide.loc[y.index, "fips"]}
            )
            coef = res.params["night_alert"]
            ci   = res.conf_int().loc["night_alert"]
            estimates.append({"k": k, "coef": coef, "ci_lo": ci[0], "ci_hi": ci[1]})
        except Exception as exc:
            log.warning("Event study k=%d failed: %s", k, exc)
            estimates.append({"k": k, "coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})

    est = pd.DataFrame(estimates)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color=GRAY, linewidth=0.6, linestyle=":")   # alert day separator
    ax.fill_between([0.5, 1.5], -10, 10, alpha=0.06, color=BLUE, label="_")

    mask = est["coef"].notna()
    ax.errorbar(
        est.loc[mask, "k"],
        est.loc[mask, "coef"],
        yerr=[
            est.loc[mask, "coef"] - est.loc[mask, "ci_lo"],
            est.loc[mask, "ci_hi"] - est.loc[mask, "coef"],
        ],
        fmt="o",
        color=BLUE,
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label="Point estimate (95% CI)",
    )

    ax.set_xlabel("Days relative to nighttime AMBER Alert (day 0 = alert night)")
    ax.set_ylabel("Additional fatalities (county-day, demeaned)")
    ax.set_title("Fig. 1  Event Study: Traffic Fatalities Around Nighttime AMBER Alerts")
    ax.set_xticks(list(window))
    ax.set_xticklabels([f"t{k:+d}" if k != 0 else "t" for k in window])
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    out = OUTPUT_FIGS / "fig1_event_study.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    log.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Alert timing histogram
# ---------------------------------------------------------------------------

def alert_timing(amber: pd.DataFrame) -> None:
    """Histogram of AMBER Alert issuance hours (local time)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    hours = amber["hour_local"].dropna()
    ax.hist(hours, bins=24, range=(0, 24), color=BLUE, edgecolor="white", linewidth=0.4)

    # Shade nighttime window
    ax.axvspan(22, 24, alpha=0.15, color=RED, label="Nighttime window")
    ax.axvspan(0,   5, alpha=0.15, color=RED)

    ax.set_xlabel("Hour of day (local time, 24-hour)")
    ax.set_ylabel("Number of alerts")
    ax.set_title("Fig. 2  Distribution of AMBER Alert Issuance Times")
    ax.set_xticks(range(0, 25, 2))
    ax.legend(frameon=False)

    out = OUTPUT_FIGS / "fig2_alert_timing.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    log.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: Heterogeneity forest plot
# ---------------------------------------------------------------------------

def heterogeneity_forest(hetero_csv: Path) -> None:
    """Forest plot of β by sub-group from reg_hetero.csv."""
    if not hetero_csv.exists():
        log.warning("Heterogeneity CSV not found — skipping Fig 3.")
        return

    h = pd.read_csv(hetero_csv)
    h = h.dropna(subset=["coef"])

    fig, ax = plt.subplots(figsize=(6, 0.6 * len(h) + 1.5))

    y_pos = range(len(h))
    ax.errorbar(
        h["coef"], y_pos,
        xerr=[h["coef"] - h["ci_lo"], h["ci_hi"] - h["coef"]],
        fmt="o", color=BLUE, capsize=4, linewidth=1.4, markersize=7,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(h["model"].tolist())
    ax.set_xlabel("Effect on next-day fatalities (county-day)")
    ax.set_title("Fig. 3  Heterogeneity Analysis")
    ax.invert_yaxis()

    out = OUTPUT_FIGS / "fig3_heterogeneity.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    log.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Geographic distribution of alerts
# ---------------------------------------------------------------------------

def geographic_distribution(amber: pd.DataFrame) -> None:
    """
    Choropleth of night-alert counts by state.
    Uses only matplotlib (no geopandas) — plots a simple bar chart as fallback
    when shapefiles are unavailable.
    """
    state_counts = (
        amber[amber["is_night"]]
        .groupby("state_fips")["alert_id"]
        .nunique()
        .reset_index()
        .rename(columns={"alert_id": "n_night_alerts"})
        .sort_values("n_night_alerts", ascending=False)
        .head(20)
    )

    # Try to get state abbreviations
    try:
        import us
        state_counts["state_abbr"] = state_counts["state_fips"].map(
            {s.fips: s.abbr for s in us.states.STATES}
        ).fillna(state_counts["state_fips"])
    except ImportError:
        state_counts["state_abbr"] = state_counts["state_fips"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(
        state_counts["state_abbr"],
        state_counts["n_night_alerts"],
        color=BLUE, edgecolor="white",
    )
    ax.set_xlabel("Number of unique nighttime AMBER Alerts")
    ax.set_title("Fig. 4  Top 20 States by Nighttime AMBER Alert Count")
    ax.invert_yaxis()

    out = OUTPUT_FIGS / "fig4_geographic.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    log.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Placebo coefficient plot
# ---------------------------------------------------------------------------

def placebo_plot(placebo_csv: Path) -> None:
    """Coefficient plot showing t-1, t, t+1, t+2 estimates side by side."""
    if not placebo_csv.exists():
        log.warning("Placebo CSV not found — skipping Fig 5.")
        return

    p = pd.read_csv(placebo_csv)
    p = p.dropna(subset=["coef"])

    # Assign x-axis position by model label
    label_to_k = {
        "Placebo: t-1": -1,
        "Same-day: t":   0,
        "Main: t+1":     1,
        "Placebo: t+2":  2,
    }
    p["k"] = p["model"].map(label_to_k)
    p = p.dropna(subset=["k"])

    colors = [RED if "Placebo" in m else BLUE for m in p["model"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvspan(0.5, 1.5, alpha=0.06, color=BLUE)   # highlight main estimate

    ax.errorbar(
        p["k"], p["coef"],
        yerr=[p["coef"] - p["ci_lo"], p["ci_hi"] - p["coef"]],
        fmt="none", ecolor=GRAY, capsize=4, linewidth=1.2,
    )
    for _, row in p.iterrows():
        color = RED if "Placebo" in row["model"] else BLUE
        ax.scatter(row["k"], row["coef"], color=color, s=60, zorder=5)

    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels(["t−1\n(placebo)", "t\n(same day)", "t+1\n(main)", "t+2\n(placebo)"])
    ax.set_ylabel("Effect on fatalities")
    ax.set_title("Fig. 5  Main and Placebo Estimates")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Main/same-day"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,  markersize=8, label="Placebo"),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    out = OUTPUT_FIGS / "fig5_placebo.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    log.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    panel_path = DATA_PROC / "panel_county_day.parquet"
    amber_path = DATA_PROC / "amber_alerts_clean.parquet"

    if not panel_path.exists():
        log.error("Panel not found — run 04_build_panel.py first.")
        return

    panel = pd.read_parquet(panel_path)
    amber = pd.read_parquet(amber_path) if amber_path.exists() else pd.DataFrame()

    # Figure 1: Event study
    event_study(panel)

    # Figure 2: Alert timing histogram
    if not amber.empty and "hour_local" in amber.columns:
        alert_timing(amber)
    else:
        log.warning("Amber data missing or lacks hour_local — skipping Fig 2.")

    # Figure 3: Heterogeneity forest plot
    heterogeneity_forest(OUTPUT_TABS / "reg_hetero.csv")

    # Figure 4: Geographic distribution
    if not amber.empty:
        geographic_distribution(amber)
    else:
        log.warning("Amber data not available — skipping Fig 4.")

    # Figure 5: Placebo coefficient plot
    placebo_plot(OUTPUT_TABS / "reg_placebo.csv")

    log.info("All figures saved to output/figures/")


if __name__ == "__main__":
    main()
