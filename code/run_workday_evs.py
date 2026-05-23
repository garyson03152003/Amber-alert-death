"""Standalone: run workday-night event study and save CSV."""
import sys, warnings, importlib.util
from pathlib import Path
import gc

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

warnings.filterwarnings("ignore")
log = get_logger("run_workday_evs")

# ---- import 05_analysis as a module ----
spec = importlib.util.spec_from_file_location("analysis_05", ROOT / "05_analysis.py")
a05  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

log.info("Loading panel…")
df = a05.load_panel()
df = prep_panel(df)

# Serious injuries already merged in load_panel via a05
log.info("Running workday-night event study (Sun–Thu only)…")
evs_wd = a05.run_event_study_workday(df)
if not evs_wd.empty:
    log.info("\n%s", evs_wd[["k", "model", "coef", "se", "pval"]].to_string(index=False))
    evs_wd.to_csv(OUTPUT_TABS / "reg_event_study_workday.csv", index=False)
    log.info("Saved to output/tables/reg_event_study_workday.csv")
else:
    log.warning("No results returned!")

del df, evs_wd; gc.collect()
log.info("Done.")
