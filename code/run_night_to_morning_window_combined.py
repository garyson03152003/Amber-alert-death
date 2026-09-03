"""Run the overnight sleep/spillover model with combined treatment.

The treatment is the union of the verified CAE AMBER records and the
high-confidence missing-person/Silver records in
``openfema_ipaws_alerts_amber_missing_2013_2024.csv``.  Cancellations are
included in the headline run because they are phone-delivered WEA messages;
an Alert/Update-only sensitivity is written alongside it.  Non-AMBER WEA
controls come from ``other_wea_night_controls.parquet`` after the same
person/Silver records have been removed.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import load_amber_missing_alerts as combined_loader
import run_night_to_morning_window as ntm
from config import OUTPUT_TABS
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("night_to_morning_combined")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)


def attach_combined_night_alert(grid: pd.DataFrame, *, include_cancel: bool = True) -> pd.DataFrame:
    """Merge combined AMBER/missing-person night exposure onto the grid."""
    alerts = combined_loader.load_combined_alerts(
        window="night", detail=False, include_cancel=include_cancel,
    )
    events = alerts.rename(columns={"effective_crash_date": "date"})[
        ["fips", "date"]
    ].drop_duplicates()
    events["date"] = pd.to_datetime(events["date"])
    events["night_alert"] = 1

    out = grid.copy()
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(events, on=["fips", "date"], how="left", validate="one_to_one")
    out["night_alert"] = out["night_alert"].fillna(0).astype(int)
    log.info(
        "Combined night-alert county-dates matched to active grid: %d (include_cancel=%s)",
        int(out["night_alert"].sum()), include_cancel,
    )

    out = out.sort_values(["fips", "date"]).reset_index(drop=True)
    out["night_alert_lag1"] = out.groupby("fips")["night_alert"].shift(1).fillna(0).astype(int)
    out["night_alert_lead1"] = out.groupby("fips")["night_alert"].shift(-1).fillna(0).astype(int)
    out["alert_last2nights_any"] = (
        (out["night_alert"] + out["night_alert_lag1"]) > 0
    ).astype(int)
    out["alert_last2nights_dose"] = out["night_alert"] + out["night_alert_lag1"]
    return out


def _add_model_columns(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["year_str"] = out["date"].dt.year.astype(str)
    out["dow"] = out["date"].dt.dayofweek.astype(str)
    out["month_str"] = out["date"].dt.month.astype(str)
    out["fips_dow"] = out["fips"] + "_" + out["dow"]
    out["fips_year"] = out["fips"] + "_" + out["year_str"]
    out["state_code"] = out["fips"].str[:2]
    out["date_str"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def run_combined(
    *, include_cancel: bool = True, control_path: Path | None = None
) -> pd.DataFrame:
    """Estimate core own/cross models for one cancellation policy."""
    grid = ntm.build_outcome_grid()
    grid = attach_combined_night_alert(grid, include_cancel=include_cancel)
    grid = ntm.attach_cross_spillover(grid)
    grid = ntm.attach_other_wea_control(grid, path=control_path)
    grid = _add_model_columns(grid)

    results: list[dict] = []
    controls = ntm.OTHER_WEA_CONTROL_SPECS
    fe_specs = (
        ("naive", "fips + year_str + dow + month_str"),
        ("robust", "fips_year + fips_dow + month_str"),
    )
    for control_label, control_cols in controls:
        for _, fe in fe_specs:
            ntm.run(
                grid,
                f"OWN night_alert -> fatals, combined ({'with' if include_cancel else 'without'} cancellations; {control_label} WEA control)",
                "fatals_0623", "night_alert", fe, results,
                extra_controls=["cross_spillover", *control_cols],
            )
            ntm.run(
                grid,
                f"CROSS_SPILLOVER -> fatals, combined ({'with' if include_cancel else 'without'} cancellations; {control_label} WEA control)",
                "fatals_0623", "cross_spillover", fe, results,
                extra_controls=["night_alert", *control_cols],
            )
            ntm.run(
                grid,
                f"CROSS_SPILLOVER -> serious injuries, combined ({'with' if include_cancel else 'without'} cancellations; {control_label} WEA control)",
                "serious_0623", "cross_spillover", fe, results,
                extra_controls=["night_alert", *control_cols],
            )
    out = pd.DataFrame(results)
    out["include_cancel"] = include_cancel
    out["control_columns"] = out["label"].str.extract(r"; (.+) WEA control\)", expand=False)
    out["control_source"] = "all_non_amber_wea" if control_path is None else "non_weather_wea"
    return out


def main() -> None:
    headline = run_combined(include_cancel=True)
    sensitivity = run_combined(include_cancel=False)
    out = pd.concat([headline, sensitivity], ignore_index=True)
    path = OUTPUT_TABS / "reg_night_to_morning_window_combined.csv"
    out.to_csv(path, index=False)
    log.info("Saved combined estimates -> %s (%d rows)", path, len(out))


if __name__ == "__main__":
    main()
