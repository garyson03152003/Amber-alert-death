"""
make_latex_tables.py
Generates all publication-quality LaTeX tables for the AMBER Alert paper.

Tables produced:
  Tab 1  — Main results: state-clustered SE (7 specs, count + combined/100k)
  Tab 2  — Dosage: log-breadth vs binary (10 specs)
  Tab 3  — Heterogeneity: night band, weekday/weekend, pop quartile
  Tab 4  — Commuting spillover (own + cross)
  Tab 5  — Weather robustness (6 specs ±PRISM weather)
  Tab 6  — Placebo: daytime alerts, pre-period, aligned

Output: output/tables/paper_tables.tex  (standalone compilable LaTeX)
        Individual .tex files per table for easy \input{}
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_TABS
warnings.filterwarnings("ignore")

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.01:  return r"^{***}"
    if p < 0.05:  return r"^{**}"
    if p < 0.10:  return r"^{*}"
    return ""

def fmt_coef(c, p, decimals=4):
    """Format coefficient with stars as superscript."""
    s = f"{c:+.{decimals}f}"
    st = stars(p)
    return f"${s}{st}$"

def fmt_se(se, decimals=4):
    return f"$({se:.{decimals}f})$"

def fmt_n(n):
    return f"{int(n):,}"

def fmt_pval(p):
    if p < 0.001: return "$<$0.001"
    return f"${p:.3f}$"

def hline(): return r"\hline"
def dline(): return r"\hline\hline"

def table_wrapper(body_lines, caption, label, footnote, col_spec):
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        dline(),
    ] + body_lines
    # Close table
    ncol = len(col_spec.replace("|","").replace(" ",""))
    lines += [
        dline(),
        f"\\multicolumn{{{ncol}}}{{p{{0.96\\textwidth}}}}{{\\footnotesize {footnote}}} \\\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)

# ── Table 1: Main Results ─────────────────────────────────────────────────────
def make_table1():
    sc  = pd.read_csv(OUTPUT_TABS / "reg_state_clustered.csv")
    comb = pd.read_csv(OUTPUT_TABS / "reg_combined.csv")

    col_spec = "lcccccc"
    header = r"Specification & $\hat\beta$ & SE & $p$-value & $N$ & Mean $y$ & $R^2$ \\"
    rows = [header, hline()]

    # Panel A: Count outcome, state-clustered SE
    rows.append(r"\multicolumn{7}{l}{\textit{Panel A: fatals\_next\_commute (count), SE clustered by state}} \\")
    rows.append(hline())
    count_specs = [
        ("(2) County FE [state-cl]",      "County FE"),
        ("(3) Baseline [state-cl]",        r"Baseline: county + DoW$\times$Month FE"),
        ("(5) + Year FE [state-cl]",       r"Baseline + Year FE"),
    ]
    for key, lbl in count_specs:
        r = sc[sc["model"] == key]
        if r.empty: continue
        r = r.iloc[0]
        rows.append(
            f"\\quad {lbl} & {fmt_coef(r['coef'], r['pval'])} & "
            f"{fmt_se(r['se'])} & {fmt_pval(r['pval'])} & "
            f"{fmt_n(r['n_obs'])} & {r['mean_y']:.4f} & {r['r2']:.2e} \\\\"
        )

    rows.append(hline())
    # Panel B: Combined/100k, log-pop WLS, county-clustered SE
    rows.append(r"\multicolumn{7}{l}{\textit{Panel B: combined (fatal+serious inj.) per 100k, log-pop WLS}} \\")
    rows.append(hline())
    wls_specs = [
        ("(2) County FE [logWLS]",    "County FE"),
        ("(3) Baseline [logWLS]",      r"Baseline: county + DoW$\times$Month FE"),
        ("(5) + Year FE [logWLS]",     r"Baseline + Year FE"),
    ]
    for key, lbl in wls_specs:
        r = comb[comb["model"] == key]
        if r.empty: continue
        r = r.iloc[0]
        rows.append(
            f"\\quad {lbl} & {fmt_coef(r['coef'], r['pval'])} & "
            f"{fmt_se(r['se'])} & {fmt_pval(r['pval'])} & "
            f"{fmt_n(r['n_obs'])} & {r['mean_y']:.4f} & {r['r2']:.2e} \\\\"
        )

    footnote = (
        "Panel A: SE clustered by state (51 clusters). "
        "Panel B: SE clustered by county (log-population WLS). "
        r"County FE and DoW$\times$Month FE absorbed via HDFE (pyhdfe). "
        r"Outcome \textit{fatals\_next\_commute} = fatals$_{t+0}$ for midnight--6am alerts, "
        r"fatals$_{t+1}$ otherwise. Combined = fatalities + serious injuries. "
        r"$^{***}$ $p{<}0.01$, $^{**}$ $p{<}0.05$, $^{*}$ $p{<}0.10$."
    )
    return table_wrapper(rows,
        "Effect of Nighttime AMBER Alert on Traffic Fatalities and Injuries",
        "main_results", footnote, col_spec)

# ── Table 2: Dosage (log-breadth) ────────────────────────────────────────────
def make_table2():
    df = pd.read_csv(OUTPUT_TABS / "reg_dosage_breadth.csv")

    col_spec = "lcccc"
    header = r"Specification & $\hat\beta$ & SE & $p$-value & $N$ \\"

    rows = [header, hline()]

    spec_labels = {
        "(1) Binary, baseline FE":           "(1) Binary treatment, baseline FE",
        "(2) Log-breadth, baseline FE":      "(2) Log(1+breadth), baseline FE",
        "(3) Log-breadth + lags, baseline FE": "(3) Log(1+breadth) + lags",
        "(4) Log-breadth, TWFE2":            "(4) Log(1+breadth), TWFE2",
        "(5) Log-breadth, TWFE2 + Year FE":  "(5) Log(1+breadth), TWFE2 + Year FE",
        "(6) Log-breadth, WLS rate":         "(6) Log(1+breadth), WLS rate",
        "(7) Log-breadth, WLS + Year FE":    "(7) Log(1+breadth), WLS + Year FE",
        "(8) Log-breadth, WLS + TWFE2":      "(8) Log(1+breadth), WLS + TWFE2",
        "(9) Log-breadth, binary joint":     "(9) Binary + log-breadth joint",
        "(10) Log-breadth, state FE WLS":    "(10) Log(1+breadth), state FE WLS",
    }

    for _, row in df.iterrows():
        lbl = spec_labels.get(str(row["label"]), str(row["label"]))
        rows.append(
            f"{lbl} & {fmt_coef(row['coef'], row['pval'])} & "
            f"{fmt_se(row['se'])} & {fmt_pval(row['pval'])} & {fmt_n(row['n_obs'])} \\\\"
        )

    footnote = (
        r"Log-breadth $= \log(1 + \text{counties covered})$, the count of counties receiving the WEA alert. "
        "Coefficient is interpretable as: fatalities per 1-unit increase in log counties covered. "
        r"$^{***}$ $p{<}0.01$, $^{**}$ $p{<}0.05$, $^{*}$ $p{<}0.10$. SE clustered by state."
    )
    return table_wrapper(rows,
        "Dose--Response: Effect of Alert Geographic Breadth on Traffic Fatalities",
        "dosage_breadth", footnote, col_spec)

# ── Table 3: Heterogeneity ───────────────────────────────────────────────────
def make_table3():
    df = pd.read_csv(OUTPUT_TABS / "reg_hetero.csv")

    col_spec = "lcccccc"
    header = r"Subgroup & $\hat\beta$ & SE & $p$-value & $N$ & Mean $y$ & Counties \\"
    rows = [header, hline()]

    group_labels = {
        "Band: early_night":  "Alert band: 10pm--midnight",
        "Band: deep_night":   "Alert band: midnight--3am",
        "Band: late_night":   "Alert band: 3am--6am",
        "Weekday":            "Day type: Weekday (Mon--Fri)",
        "Weekend":            "Day type: Weekend (Sat--Sun)",
        "Pop Q1 (lowest)":    "Population: Q1 (lowest density)",
        "Pop Q2":             "Population: Q2",
        "Pop Q3":             "Population: Q3",
        "Pop Q4 (highest)":   "Population: Q4 (highest density)",
    }

    sections = [
        ("Alert timing", ["Band: early_night","Band: deep_night","Band: late_night"]),
        ("Day of week",  ["Weekday","Weekend"]),
        ("Population quartile", ["Pop Q1 (lowest)","Pop Q2","Pop Q3","Pop Q4 (highest)"]),
    ]

    for sec_title, models in sections:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{sec_title}}}}} \\\\")
        rows.append(hline())
        for m in models:
            r = df[df["model"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            lbl = group_labels.get(m, m)
            rows.append(
                f"\\quad {lbl} & {fmt_coef(r['coef'], r['pval'])} & "
                f"{fmt_se(r['se'])} & {fmt_pval(r['pval'])} & "
                f"{fmt_n(r['n_obs'])} & {r['mean_y']:.4f} & {int(r['n_counties'])} \\\\"
            )
        rows.append(hline())

    footnote = (
        "Baseline specification (county FE + DoW$\\times$Month FE) estimated on each subgroup separately. "
        "SE clustered by state. "
        r"$^{***}$ $p{<}0.01$, $^{**}$ $p{<}0.05$, $^{*}$ $p{<}0.10$."
    )
    return table_wrapper(rows,
        "Heterogeneity Analysis: Effect by Alert Band, Day Type, and Population Density",
        "heterogeneity", footnote, col_spec)

# ── Table 4: Commuting Spillover ─────────────────────────────────────────────
def make_table4():
    df = pd.read_csv(OUTPUT_TABS / "reg_commuting_spillover.csv")

    # Columns: spec, coef_type, coef, se, pval, n_obs
    spec_labels = {
        "count_baseline": r"Count, baseline FE (county + DoW$\times$Month)",
        "count_twfe2":    r"Count, TWFE2 (county$\times$year + lag)",
        "wls_rate":       r"WLS, combined/100k (log-pop weights)",
    }

    col_spec = "llcccc"
    header = r"Specification & Treatment & $\hat\beta$ & SE & $p$-value & $N$ \\"
    rows = [header, hline()]

    for spec in df["spec"].unique():
        sub = df[df["spec"] == spec]
        spec_lbl = spec_labels.get(spec, spec)
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{spec_lbl}}}}} \\\\")
        rows.append(hline())
        for _, row in sub.iterrows():
            treat_lbl = "Own-county alert" if row["coef_type"] == "own" else "Cross-county spillover"
            n_obs = row["n_obs"]
            rows.append(
                f"\\quad & {treat_lbl} & {fmt_coef(row['coef'], row['pval'])} & "
                f"{fmt_se(row['se'])} & {fmt_pval(row['pval'])} & {fmt_n(n_obs)} \\\\"
            )
        rows.append(hline())

    footnote = (
        r"Cross-spillover $= \sum_{j \neq c} w_{j \to c} \cdot \mathbf{1}[\text{night alert}_j]$, "
        "where weights are ACS 2016--2020 commuting flows (column-normalized). "
        "Both own-county alert and cross-county spillover enter the same regression. "
        "SE clustered by state. "
        r"$^{***}$ $p{<}0.01$, $^{**}$ $p{<}0.05$, $^{*}$ $p{<}0.10$."
    )
    return table_wrapper(rows,
        "Commuting-Flow Spillover: Own-County and Cross-County Alert Effects",
        "spillover", footnote, col_spec)

# ── Table 5: Weather Robustness ──────────────────────────────────────────────
def make_table5():
    df = pd.read_csv(OUTPUT_TABS / "reg_weather_robustness.csv")

    col_spec = "llcccc"
    header = r"Specification & Weather & $\hat\beta$ & SE & $p$-value & $N$ \\"
    rows = [header, hline()]

    spec_order = ["Baseline count", "TWFE2 count", "WLS comb/100k"]
    spec_labels = {
        "Baseline count": r"Baseline (count, county + DoW$\times$Month FE)",
        "TWFE2 count":    r"TWFE2 (count, county$\times$year FE + lag)",
        "WLS comb/100k":  r"WLS (combined/100k, log-pop weights)",
    }

    for spec in spec_order:
        sub = df[df["label"] == spec]
        if sub.empty:
            continue
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{spec_labels.get(spec, spec)}}}}} \\\\")
        rows.append(hline())
        for _, row in sub.iterrows():
            wx_label = "PRISM (prcp + tmax)" if row["has_weather"] else "None"
            rows.append(
                f"\\quad & {wx_label} & {fmt_coef(row['coef'], row['pval'])} & "
                f"{fmt_se(row['se'])} & {fmt_pval(row['pval'])} & "
                f"{fmt_n(row['n_obs'])} \\\\"
            )
        rows.append(hline())

    footnote = (
        "PRISM 4-km gridded daily precipitation (mm) and maximum temperature ($^\\circ$C) "
        "from the ACIS GridData API (Parameter-elevation Regressions on Independent Slopes Model). "
        "SE clustered by state. "
        r"$^{***}$ $p{<}0.01$, $^{**}$ $p{<}0.05$, $^{*}$ $p{<}0.10$."
    )
    return table_wrapper(rows,
        "Weather Robustness: PRISM Daily Climate Controls",
        "weather_robust", footnote, col_spec)

# ── Table 6: Placebo ─────────────────────────────────────────────────────────
def make_table6():
    plac = pd.read_csv(OUTPUT_TABS / "reg_placebo.csv")
    day_plac = pd.read_csv(OUTPUT_TABS / "reg_daytime_placebo.csv")

    col_spec = "lcccc"
    header = r"Placebo test & $\hat\beta$ & SE & $p$-value & $N$ \\"
    rows = [header, hline()]

    rows.append(r"\multicolumn{5}{l}{\textit{Temporal placebo (shifted outcomes)}} \\")
    rows.append(hline())
    for _, row in plac.iterrows():
        rows.append(
            f"\\quad {row['model']} & {fmt_coef(row['coef'], row['pval'])} & "
            f"{fmt_se(row['se'])} & {fmt_pval(row['pval'])} & {fmt_n(row['n_obs'])} \\\\"
        )

    rows.append(hline())
    rows.append(r"\multicolumn{5}{l}{\textit{Daytime alert placebo (non-night alerts)}} \\")
    rows.append(hline())
    for _, row in day_plac.iterrows():
        rows.append(
            f"\\quad {row['model']} & {fmt_coef(row['coef'], row['pval'])} & "
            f"{fmt_se(row['se'])} & {fmt_pval(row['pval'])} & {fmt_n(row['n_obs'])} \\\\"
        )

    footnote = (
        "Placebo tests: outcome shifted to 1--7 days before alert "
        "(should be zero if pre-trends are absent), and daytime alerts (6am--10pm) "
        "as placebo treatment (should be zero if night-specific channel is absent). "
        "SE clustered by state. "
        r"$^{***}$ $p{<}0.01$, $^{**}$ $p{<}0.05$, $^{*}$ $p{<}0.10$."
    )
    return table_wrapper(rows,
        "Placebo Tests: Pre-Trends and Daytime Alerts",
        "placebo", footnote, col_spec)

# ── Assemble full LaTeX document ──────────────────────────────────────────────
preamble = r"""\documentclass[12pt]{article}
\usepackage{booktabs,longtable,geometry,caption,array,lscape,pdflscape}
\geometry{margin=1in}
\usepackage{amsmath}
\captionsetup{labelfont=bf, labelsep=period}

\begin{document}
\pagestyle{empty}
"""

postamble = r"""
\end{document}
"""

tables = []
print("Building Table 1: Main results …")
tables.append(("tab1_main_results",      make_table1()))

print("Building Table 2: Dosage …")
tables.append(("tab2_dosage",            make_table2()))

print("Building Table 3: Heterogeneity …")
tables.append(("tab3_heterogeneity",     make_table3()))

print("Building Table 4: Spillover …")
tables.append(("tab4_spillover",         make_table4()))

print("Building Table 5: Weather robustness …")
tables.append(("tab5_weather",           make_table5()))

print("Building Table 6: Placebo …")
tables.append(("tab6_placebo",           make_table6()))

# Save individual .tex files
for fname, content in tables:
    out_path = OUTPUT_TABS / f"{fname}.tex"
    out_path.write_text(content)
    print(f"  Saved → {out_path}")

# Save combined document
combined_path = OUTPUT_TABS / "paper_tables.tex"
combined_parts = [preamble]
for _, content in tables:
    combined_parts.append(content)
    combined_parts.append("\n\\clearpage\n")
combined_parts.append(postamble)
combined_path.write_text("\n\n".join(combined_parts))
print(f"\nCombined document → {combined_path}")
print("Done.")
