"""
05_analysis.py — Main regression analysis.

Baseline:  fatals_{c,t+1} = α + β·NightAlert_{c,t} + γ_c + δ_{dow×month} + X_{c,t}·θ + ε
Estimator: two-way FE via pyhdfe; SE clustered at county level.

Models  (1) Pooled OLS   (2) County FE   (3) Baseline   (4) + Weather
Hetero  night band · weekday/weekend · alert hour
Placebo outcomes at t−1, same-day, t+2

Output: output/tables/reg_baseline.csv/.tex  reg_hetero.csv  reg_placebo.csv

Run: python code/05_analysis.py
"""

import sys, warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel, fe_ols_from_panel

log = get_logger("05_analysis")
warnings.filterwarnings("ignore")

WEATHER = ["prcp_mm", "tmax_c"]


def load_panel() -> pd.DataFrame:
    path = DATA_PROC / "panel_county_day.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Panel not found: {path}")
    df = pd.read_parquet(path)
    log.info("Panel: {:,} rows, {:,} counties".format(len(df), df["fips"].nunique()))
    return df


# ---------------------------------------------------------------------------
# Model suites  (thin wrappers around fe_ols_from_panel)
# ---------------------------------------------------------------------------

def run_baseline(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    avail_w = [c for c in WEATHER if df[c].notna().mean() > 0.01]

    log.info("(1) Pooled OLS")
    results.append(fe_ols_from_panel(df, "fatals_t1", county=False, dm=False,
                                     label="(1) Pooled OLS"))
    log.info("(2) County FE")
    results.append(fe_ols_from_panel(df, "fatals_t1", county=True, dm=False,
                                     label="(2) County FE"))
    log.info("(3) County FE + DoW×Month FE  [baseline]")
    results.append(fe_ols_from_panel(df, "fatals_t1", county=True, dm=True,
                                     label="(3) Baseline"))
    if avail_w:
        log.info("(4) Baseline + weather")
        results.append(fe_ols_from_panel(df, "fatals_t1", controls=avail_w,
                                         county=True, dm=True, label="(4) + Weather"))
    else:
        log.warning("Weather controls sparse — skipping model (4).")

    return pd.DataFrame(results)


def run_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for band in ["early_night", "deep_night", "late_night"]:
        sub = df.copy()
        sub["night_alert"] = (sub["night_band"] == band).astype(int)
        results.append(fe_ols_from_panel(sub, "fatals_t1",
                                         label=f"Band: {band}"))

    for lbl, mask in [
        ("Next-day: weekday", df["dow"].isin([0, 1, 2, 3])),
        ("Next-day: weekend", df["dow"].isin([4, 5, 6])),
    ]:
        results.append(fe_ols_from_panel(df[mask], "fatals_t1", label=lbl))

    for lbl, hrs in [
        ("Alert 10pm–midnight", list(range(22, 24))),
        ("Alert midnight–3am",  list(range(0,  3))),
        ("Alert 3am–5am",       list(range(3,  5))),
    ]:
        sub = df.copy()
        sub["night_alert"] = (
            df["night_alert"].astype(bool) & df["alert_hour"].isin(hrs)
        ).astype(int)
        results.append(fe_ols_from_panel(sub, "fatals_t1", label=lbl))

    return pd.DataFrame(results)


def run_placebo(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for outcome, lbl in [
        ("fatals_tm1", "Placebo: t−1"),
        ("fatals_t0",  "Same-day: t"),
        ("fatals_t1",  "Main: t+1"),
        ("fatals_t2",  "Placebo: t+2"),
    ]:
        if outcome not in df.columns:
            continue
        sub = df.dropna(subset=[outcome]).copy()
        results.append(fe_ols_from_panel(sub, outcome, label=lbl))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def to_latex(results: pd.DataFrame, title: str, note: str = "") -> str:
    STARS = [(0.01, "***"), (0.05, "**"), (0.10, "*")]
    def star(p):
        for t, s in STARS:
            if p < t: return s
        return ""

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{" + title + r"}",
        r"\begin{tabular}{lccccc}",
        r"\hline\hline",
        r"Model & $\hat\beta$ & SE & $p$-value & $N$ & Mean $y$ \\",
        r"\midrule",
    ]
    for _, r in results.iterrows():
        if "error" in r and pd.notna(r.get("error")):
            lines.append(f"{r['model']} & \\multicolumn{{5}}{{c}}{{—}} \\\\")
            continue
        s    = star(r.get("pval", 1.0))
        coef = f"{r['coef']:.4f}{s}"   if pd.notna(r.get("coef"))  else "—"
        se   = f"({r['se']:.4f})"      if pd.notna(r.get("se"))    else "—"
        pv   = f"{r['pval']:.3f}"      if pd.notna(r.get("pval"))  else "—"
        n    = f"{int(r['n_obs']):,}"  if pd.notna(r.get("n_obs")) else "—"
        my   = f"{r['mean_y']:.4f}"    if pd.notna(r.get("mean_y")) else "—"
        lines.append(f"{r['model']} & {coef} & {se} & {pv} & {n} & {my} \\\\")

    lines += [r"\hline\hline"]
    if note:
        lines.append(r"\multicolumn{6}{p{0.9\textwidth}}{\footnotesize " + note + r"} \\")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    df = prep_panel(df)

    note = ("SE clustered by county. County FE and DoW$\\times$Month FE absorbed "
            "via pyhdfe. *** $p{<}$0.01, ** $p{<}$0.05, * $p{<}$0.10.")

    log.info("=== BASELINE ===")
    base = run_baseline(df)
    log.info("\n%s", base[["model","coef","se","pval","n_obs"]].to_string(index=False))
    base.to_csv(OUTPUT_TABS / "reg_baseline.csv", index=False)
    (OUTPUT_TABS / "reg_baseline.tex").write_text(
        to_latex(base, "Effect of Nighttime AMBER Alert on Next-Day Traffic Fatalities", note))

    log.info("=== HETEROGENEITY ===")
    het = run_heterogeneity(df)
    log.info("\n%s", het[["model","coef","se","pval","n_obs"]].to_string(index=False))
    het.to_csv(OUTPUT_TABS / "reg_hetero.csv", index=False)
    (OUTPUT_TABS / "reg_hetero.tex").write_text(to_latex(het, "Heterogeneity Analysis", note))

    log.info("=== PLACEBO ===")
    plac = run_placebo(df)
    log.info("\n%s", plac[["model","coef","se","pval","n_obs"]].to_string(index=False))
    plac.to_csv(OUTPUT_TABS / "reg_placebo.csv", index=False)
    (OUTPUT_TABS / "reg_placebo.tex").write_text(to_latex(plac, "Placebo Tests", note))

    log.info("Done. Results saved to output/tables/")


if __name__ == "__main__":
    main()
