"""
Standalone runner: commuting-flow weighted spillover analysis.

Tests whether AMBER Alerts in neighbouring counties raise next-day crashes
in the work county, via sleep-disrupted commuter spillover.

Data required:
  data/processed/commuting/county_commuting_weights.parquet
  (built automatically by build_commuting_weights.py if missing)

Output:
  output/tables/reg_commuting_spillover.csv
"""
import sys, warnings, importlib.util, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("run_commuting_spillover")

WEIGHTS_PATH = (Path(__file__).parent.parent / "data" / "processed" /
                "commuting" / "county_commuting_weights.parquet")

# ---- Build weights if missing ----
if not WEIGHTS_PATH.exists():
    log.info("Commuting weights not found — building from ACS data…")
    spec = importlib.util.spec_from_file_location(
        "build_weights", Path(__file__).parent / "build_commuting_weights.py")
    bw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bw)

# ---- Load analysis module ----
spec = importlib.util.spec_from_file_location(
    "analysis_05", Path(__file__).parent / "05_analysis.py")
a05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

log.info("Loading panel…")
df = a05.load_panel()
df = prep_panel(df)

log.info("Running commuting spillover analysis…")
results = a05.run_commuting_spillover(df)
del df; gc.collect()

if not results.empty:
    out_path = OUTPUT_TABS / "reg_commuting_spillover.csv"
    results.to_csv(out_path, index=False)
    log.info("Saved to %s", out_path)

    # Pretty print
    for spec_label in results["spec"].unique():
        sub = results[results["spec"] == spec_label]
        log.info("\n--- %s ---", spec_label)
        for _, row in sub.iterrows():
            sig = ""
            if row["pval"] < 0.01:  sig = "***"
            elif row["pval"] < 0.05: sig = "**"
            elif row["pval"] < 0.10: sig = "*"
            log.info("  %-12s  coef=%+.5f  se=%.5f  p=%.3f  %s",
                     row["coef_type"], row["coef"], row["se"], row["pval"], sig)
else:
    log.warning("No results returned!")

log.info("Done.")
