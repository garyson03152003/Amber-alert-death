"""Standalone: run weekday/weekend interaction spec and save CSV."""
import sys, warnings, importlib.util
from pathlib import Path
import gc

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("run_wd_interaction")

spec = importlib.util.spec_from_file_location("analysis_05", Path(__file__).parent / "05_analysis.py")
a05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

log.info("Loading panel…")
df = a05.load_panel()
df = prep_panel(df)

log.info("Running weekday/weekend interaction…")
wd_int = a05.run_weekday_interaction(df)
if not wd_int.empty:
    log.info("\n%s", wd_int[["model", "split", "coef", "se", "pval", "n_obs"]].to_string(index=False))
    wd_int.to_csv(OUTPUT_TABS / "reg_weekday_interaction.csv", index=False)
    log.info("Saved to output/tables/reg_weekday_interaction.csv")
else:
    log.warning("No results returned!")

del df, wd_int; gc.collect()
log.info("Done.")
