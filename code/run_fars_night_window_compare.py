"""FARS national night-alert effect under both night-window definitions.

The overnight exposure window runs night_start -> 06:00 and the outcome is
day D+1's crashes, which is what the effective_crash_date mapping already
encodes. What was wrong is where the window STARTS: the legacy cutoff of
22:00 discards the 20:00 and 21:00 hours, which hold 4,982 alerts. Moving
the boundary to 20:00 raises night county-dates from 6,072 to 10,430 (+72%)
-- a larger power gain than pooling five states delivered.

Both definitions are run side by side so the choice is visible rather than
buried, on the national FARS panel (12.6M county-days, and immune to the
UTC-date bug that affected the state sources).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as base
import run_state_dot_analysis_share as share
import run_validated_fars_share as fars
from config import OUTPUT_TABS

log = base.log


def main() -> None:
    rows = []
    for night_start in (22, 20):
        # build_panel() reaches for load_verified_night_alerts(); point it at
        # the requested window boundary for this pass.
        base_loader = base.load_verified_night_alerts
        fars.base.load_verified_night_alerts = (
            lambda *, detail=False, _ns=night_start: base.load_verified_alerts(
                window="night", detail=detail, night_start=_ns)
        )
        try:
            panel = fars.build_panel(direct_only=False)
        finally:
            fars.base.load_verified_night_alerts = base_loader

        n_treated = int(panel["night_alert"].sum())
        log.info("[night_start=%d] panel %s county-days | %s treated",
                 night_start, f"{len(panel):,}", f"{n_treated:,}")

        for rate_col, count_col in (("fatals_per_100k", "fatals"),):
            for r in share.run_ppml(panel, count_col, f"FARS_NIGHT{night_start}"):
                if r.get("record_type") == "estimate":
                    r.update({"night_start": night_start, "n_treated_county_days": n_treated})
                    rows.append(r)
            for r in share.run_wls(panel, rate_col, f"FARS_NIGHT{night_start}"):
                if r.get("record_type") == "estimate":
                    r.update({"night_start": night_start, "n_treated_county_days": n_treated})
                    rows.append(r)

    out = pd.DataFrame(rows)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_TABS / "fars_night_window_compare.csv", index=False)
    keep = [c for c in ["night_start", "model", "outcome", "term", "pct_change",
                        "beta", "se", "pvalue", "n_treated_county_days", "n_obs"]
            if c in out.columns]
    print("\n" + out[keep].to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
