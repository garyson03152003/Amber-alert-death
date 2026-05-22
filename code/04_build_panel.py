"""
04_build_panel.py — Construct the county-day analysis panel.

Inputs (from data/processed/):
    fars_county_day.parquet
    amber_alerts_clean.parquet
    weather_county_day.parquet   (optional; rows with missing weather kept)

Outputs (data/processed/):
    panel_county_day.parquet     — main analysis dataset

Panel structure:
    Unit of observation: county × calendar day
    Universe: all US counties × all days in STUDY_YEARS, restricted to
              counties observed in FARS at least once.

Key variables:
    fatals_t1       traffic fatalities on day t+1 (outcome)
    fatals_t0       traffic fatalities on day t   (same-day, for falsification)
    fatals_tm1      traffic fatalities on day t-1
    night_alert     1 if county received nighttime AMBER alert on day t
    alert_any       1 if county received any AMBER alert on day t
    alert_hour      hour of the nighttime alert (missing if no alert)
    night_band      "early_night" / "deep_night" / "late_night" / None
    prcp_mm         precipitation (mm)
    tmax_c          max temperature (°C)
    dow             day of week (0=Mon … 6=Sun)
    month           calendar month (1–12)
    year            calendar year
    fips            5-digit county FIPS

Fixed effect groups (not stored as dummies; used as index in linearmodels):
    fips
    dow × month  (interaction, stored as dow_x_month string)

Run: python code/04_build_panel.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, STUDY_YEARS, NIGHT_START_HOUR, NIGHT_END_HOUR
from utils import get_logger

log = get_logger("04_panel")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}.\n"
            f"Run the corresponding download script first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded %s: %d rows", label, len(df))
    return df


def build_full_grid(fips_codes: list[str], years: list[int]) -> pd.DataFrame:
    """
    Build a balanced county × day grid for all counties in fips_codes
    over all days in years.  This is the backbone of the panel.

    Returns DataFrame with columns: fips, date
    """
    log.info("Building balanced panel grid: %d counties × %d years...",
             len(fips_codes), len(years))
    dates = pd.date_range(
        start=f"{min(years)}-01-01",
        end=f"{max(years)}-12-31",
        freq="D",
    )
    grid = pd.MultiIndex.from_product(
        [fips_codes, dates], names=["fips", "date"]
    ).to_frame(index=False)
    log.info("Grid: %d rows", len(grid))
    return grid


def lag_fatals(fatals: pd.Series, days: int, panel: pd.DataFrame) -> pd.Series:
    """
    Return fatals shifted by `days` within each county, preserving county
    boundaries (no bleeding across counties).

    positive days → lead (future), negative → lag (past)
    """
    idx = panel.set_index(["fips", "date"])
    shifted = (
        panel.groupby("fips", group_keys=False)
        .apply(lambda g: g.set_index("date")["fatals_t0"]
               .shift(-days)   # shift(-1) gives next-day
               .rename(f"fatals_shift{days}"))
    )
    return shifted.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load inputs
    # -----------------------------------------------------------------------
    fars = load_required(DATA_PROC / "fars_county_day.parquet", "FARS")
    amber = load_required(DATA_PROC / "amber_alerts_clean.parquet", "AMBER alerts")

    weather_path = DATA_PROC / "weather_county_day.parquet"
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        log.info("Loaded weather: %d rows", len(weather))
        has_weather = True
    else:
        log.warning("Weather file missing — proceeding without weather controls.")
        has_weather = False

    # -----------------------------------------------------------------------
    # 2. Build balanced grid
    # -----------------------------------------------------------------------
    # Only include counties that appear in FARS at least once
    active_fips = sorted(fars["fips"].unique().tolist())
    grid = build_full_grid(active_fips, STUDY_YEARS)

    # -----------------------------------------------------------------------
    # 3. Merge FARS fatalities onto grid (missing = 0 fatalities that day)
    # -----------------------------------------------------------------------
    fars_clean = fars.rename(columns={"fatals": "fatals_t0"})
    panel = grid.merge(fars_clean, on=["fips", "date"], how="left")
    panel["fatals_t0"] = panel["fatals_t0"].fillna(0).astype(int)

    # -----------------------------------------------------------------------
    # 4. Build leads/lags of fatalities (outcome and placebo variables)
    # -----------------------------------------------------------------------
    # Sort by county then date so shift() works correctly within county
    panel = panel.sort_values(["fips", "date"]).reset_index(drop=True)

    for shift_days, colname in [
        (-1, "fatals_tm1"),  # t-1  (past-day placebo)
        (-1, None),          # placeholder
        ( 1, "fatals_t1"),   # t+1  (main outcome)
        ( 2, "fatals_t2"),   # t+2  (second-day placebo)
    ]:
        if colname is None:
            continue
        panel[colname] = (
            panel.groupby("fips")["fatals_t0"]
            .shift(-shift_days)   # shift(-1) → bring next day's value to today
        )

    # -----------------------------------------------------------------------
    # 5. Merge AMBER Alert treatment indicators
    # -----------------------------------------------------------------------
    # We need: for each (county, date), did a nighttime alert fire?
    # amber_alerts_clean has one row per (alert_id, county_fips).
    # A county receives a night alert if county_fips == fips and is_night == True.

    amber["date"] = pd.to_datetime(amber["issued_local"].dt.date)

    # Night alert indicator
    amber_night = amber[amber["is_night"]].copy()
    amber_night = amber_night.dropna(subset=["county_fips"])

    # Aggregate: one row per (fips, date) with indicator columns
    alert_agg = (
        amber.groupby(["county_fips", "date"])
        .agg(
            alert_any=("alert_id", "count"),
            night_alert=("is_night", "max"),
            # For heterogeneity: hour of the first nighttime alert
            alert_hour=("hour_local", lambda s: s[amber.loc[s.index, "is_night"]].min()
                        if amber.loc[s.index, "is_night"].any() else np.nan),
            night_band=("night_band", lambda s:
                        amber.loc[s.index[amber.loc[s.index, "is_night"]], "night_band"].mode()[0]
                        if amber.loc[s.index, "is_night"].any() else None),
        )
        .reset_index()
        .rename(columns={"county_fips": "fips", "alert_any": "_n_alerts"})
    )
    alert_agg["alert_any"] = (alert_agg["_n_alerts"] > 0).astype(int)
    alert_agg["night_alert"] = alert_agg["night_alert"].fillna(False).astype(int)
    alert_agg = alert_agg.drop(columns=["_n_alerts"])

    panel = panel.merge(alert_agg, on=["fips", "date"], how="left")
    panel["alert_any"]   = panel["alert_any"].fillna(0).astype(int)
    panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)

    log.info(
        "Alert coverage: %d county-days with any alert, %d with night alert",
        (panel["alert_any"] > 0).sum(),
        (panel["night_alert"] > 0).sum(),
    )

    # -----------------------------------------------------------------------
    # 6. Merge weather
    # -----------------------------------------------------------------------
    if has_weather:
        panel = panel.merge(weather, on=["fips", "date"], how="left")
        coverage = panel["prcp_mm"].notna().mean()
        log.info("Weather coverage: %.1f%%", coverage * 100)

    # -----------------------------------------------------------------------
    # 7. Add calendar fixed-effect identifiers
    # -----------------------------------------------------------------------
    panel["year"]  = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["dow"]   = panel["date"].dt.dayofweek          # 0=Mon, 6=Sun
    panel["dow_x_month"] = (
        panel["dow"].astype(str) + "_" + panel["month"].astype(str)
    )
    panel["is_weekend"]  = (panel["dow"] >= 5).astype(int)

    # -----------------------------------------------------------------------
    # 8. Population density (proxy for shock intensity)
    # -----------------------------------------------------------------------
    # We load the ACS 5-year county population estimates from a cached Census file.
    # If not available, skip (the variable is used only in heterogeneity analysis).
    pop_path = DATA_PROC / "county_population.parquet"
    if pop_path.exists():
        pop = pd.read_parquet(pop_path)
        panel = panel.merge(pop[["fips", "year", "population"]], on=["fips", "year"], how="left")
        # Density quartile (county-level, time-fixed for simplicity)
        county_pop = pop.groupby("fips")["population"].mean()
        quartiles = pd.qcut(county_pop, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        panel["pop_quartile"] = panel["fips"].map(quartiles)
        log.info("Population data merged.")
    else:
        log.warning(
            "County population file not found (%s).\n"
            "  → Download from Census ACS and save as county_population.parquet.\n"
            "  → Population density heterogeneity analysis will be skipped.",
            pop_path,
        )

    # -----------------------------------------------------------------------
    # 9. Sample restrictions
    # -----------------------------------------------------------------------
    # Keep only study-year observations
    panel = panel[panel["year"].isin(STUDY_YEARS)]

    # Drop Dec 31 rows where t+1 would cross a year boundary into the next year
    # (not strictly wrong, but keeps the panel balanced within years)
    panel = panel[~((panel["month"] == 12) & (panel["date"].dt.day == 31))]

    # Drop rows where next-day outcome is missing (happens at year-end boundary)
    panel = panel.dropna(subset=["fatals_t1"])
    panel["fatals_t1"] = panel["fatals_t1"].astype(int)

    # -----------------------------------------------------------------------
    # 10. Save
    # -----------------------------------------------------------------------
    panel = panel.sort_values(["fips", "date"]).reset_index(drop=True)

    out_path = DATA_PROC / "panel_county_day.parquet"
    panel.to_parquet(out_path, index=False)

    log.info("=== Panel summary ===")
    log.info("Rows: {:,}".format(len(panel)))
    log.info("Counties: {:,}".format(panel["fips"].nunique()))
    log.info("Date range: %s – %s", panel["date"].min().date(), panel["date"].max().date())
    log.info("Night-alert county-days: {:,}".format(panel["night_alert"].sum()))
    log.info("Mean next-day fatalities: %.4f", panel["fatals_t1"].mean())
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
