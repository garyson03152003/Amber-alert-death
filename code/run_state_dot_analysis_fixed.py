"""Corrected state-DOT AMBER-alert analysis.

Key corrections relative to ``run_state_dot_analysis.py``:
1. unavailable/non-comparable outcomes stay missing instead of becoming zeros;
2. Poisson PPML uses raw counts and retains zero outcomes;
3. population exposure is an offset when supported, otherwise an explicit
   log-population control (never a population-weight substitute for an offset);
4. commuter-flow spillovers are modeled explicitly, so non-targeted counties
   with inbound commuters from alerted counties are not treated as clean controls.

Outputs
-------
output/tables/state_dot_analysis_fixed.csv
output/tables/state_dot_descriptives_fixed.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, DATA_RAW, OUTPUT_TABS
from utils import get_logger
from state_dot_analysis_core import (
    normalize_state_outcomes,
    prepare_ppml_sample,
    build_ppml_call_spec,
    build_commuter_spillover,
    add_spillover_classes,
)

warnings.filterwarnings("ignore")
log = get_logger("state_dot_analysis_fixed")
OUTPUT_TABS.mkdir(parents=True, exist_ok=True)

STATE_FILES = {
    "CA": dict(file="california_ccrs_county_day.parquet", crashes="ca_crashes", fatals="ca_fatals", serious="ca_serious_inj"),
    "FL": dict(file="florida_fdot_county_day.parquet", crashes="fl_crashes", fatals="fl_fatals", serious="fl_serious_inj"),
    "IL": dict(file="illinois_idot_county_day.parquet", crashes="il_crashes", fatals="il_fatals", serious="il_serious_inj"),
    "IA": dict(file="iowa_dot_county_day.parquet", crashes="ia_crashes", fatals="ia_fatals", serious="ia_serious_inj"),
    "MA": dict(file="massachusetts_massdot_county_day.parquet", crashes="ma_crashes", fatals="ma_fatals", serious="ma_serious_inj"),
    "NV": dict(file="nevada_ndot_county_day.parquet", crashes="nv_crashes", fatals="nv_fatals", serious="nv_serious_inj"),
    "NY": dict(file="newyork_dot_county_day.parquet", crashes="ny_crashes", fatals="ny_fatal_crashes", serious=None, fatals_comparable=False),
    "OR": dict(file="oregon_odot_county_day.parquet", crashes="or_crashes", fatals="or_fatals", serious="or_serious_inj"),
    "TN": dict(file="tennessee_tdot_county_day.parquet", crashes="tn_crashes", fatals="tn_fatals", serious="tn_serious_inj"),
    "TX": dict(file="texas_txdot_county_day.parquet", crashes="tx_crashes", fatals="tx_fatals", serious="tx_serious_inj"),
    "VA": dict(file="virginia_vdot_county_day.parquet", crashes="va_crashes", fatals="va_fatals", serious="va_serious_inj"),
    "WI": dict(file="wisconsin_dot_county_day.parquet", crashes="wi_crashes", fatals="wi_fatals", serious="wi_serious_inj"),
}

STATE_TIMEZONE = {
    "06": "America/Los_Angeles", "12": "America/New_York",
    "17": "America/Chicago", "19": "America/Chicago",
    "25": "America/New_York", "32": "America/Los_Angeles",
    "36": "America/New_York", "41": "America/Los_Angeles",
    "47": "America/Chicago", "48": "America/Chicago",
    "51": "America/New_York", "55": "America/Chicago",
}

COUNTY_TIMEZONE_OVERRIDE = {
    "12033": "America/Chicago", "12059": "America/Chicago",
    "12077": "America/Chicago", "12113": "America/Chicago",
    "12131": "America/Chicago",
    "41001": "America/Denver", "41017": "America/Denver",
    "41021": "America/Denver", "41023": "America/Denver",
    "41025": "America/Denver", "41035": "America/Denver",
    "41037": "America/Denver", "41045": "America/Denver",
    "41049": "America/Denver", "41055": "America/Denver",
    "41059": "America/Denver", "41065": "America/Denver",
    "47001": "America/New_York", "47009": "America/New_York",
    "47013": "America/New_York", "47025": "America/New_York",
    "47029": "America/New_York", "47051": "America/New_York",
    "47063": "America/New_York", "47065": "America/New_York",
    "47067": "America/New_York", "47073": "America/New_York",
    "47089": "America/New_York", "47097": "America/New_York",
    "47105": "America/New_York", "47107": "America/New_York",
    "47121": "America/New_York", "47129": "America/New_York",
    "47139": "America/New_York", "47143": "America/New_York",
    "47145": "America/New_York", "47151": "America/New_York",
    "47155": "America/New_York", "47163": "America/New_York",
    "47171": "America/New_York", "47173": "America/New_York",
    "47179": "America/New_York", "47189": "America/New_York",
    "48141": "America/Denver", "48229": "America/Denver",
}


def load_state_crashes() -> pd.DataFrame:
    parts = []
    for state, meta in STATE_FILES.items():
        path = DATA_PROC / meta["file"]
        if not path.exists():
            log.warning("Missing %s — skipping %s", meta["file"], state)
            continue
        raw = pd.read_parquet(path)
        clean = normalize_state_outcomes(
            raw,
            crashes_col=meta.get("crashes"),
            fatals_col=meta.get("fatals"),
            serious_col=meta.get("serious"),
            fatals_comparable=meta.get("fatals_comparable", True),
        )
        clean["state"] = state
        parts.append(clean)
        log.info(
            "%s: %,d county-days; available crash/fatal/serious rows = %,d / %,d / %,d",
            state, len(clean), clean["crashes"].notna().sum(), clean["fatals"].notna().sum(),
            clean["serious_inj"].notna().sum(),
        )
    if not parts:
        raise FileNotFoundError("No state-DOT county-day files were found")
    return pd.concat(parts, ignore_index=True)


def load_verified_night_alerts() -> pd.DataFrame:
    path = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2024.csv"
    if not path.exists():
        path = DATA_RAW / "amber" / "foia" / "openfema_ipaws_alerts_2013_2022.csv"
    if not path.exists():
        raise FileNotFoundError("OpenFEMA IPAWS AMBER file not found")

    alerts = pd.read_csv(path, parse_dates=["sent_utc"])
    if "msg_type" in alerts.columns:
        alerts = alerts[alerts["msg_type"].isin(["Alert", "Update"])].copy()
    else:
        log.warning("msg_type is missing; Cancel messages cannot be excluded")

    alerts["fips"] = alerts["fips"].astype(str).str.zfill(5)
    alerts = alerts[alerts["fips"].str.match(r"^\d{5}$")].copy()
    alerts["state_fips"] = alerts["fips"].str[:2]
    alerts = alerts[alerts["fips"].str[2:] != "000"].copy()
    alerts = alerts[alerts["state_fips"].isin(STATE_TIMEZONE)].copy()
    alerts["tz_name"] = alerts["fips"].map(COUNTY_TIMEZONE_OVERRIDE).fillna(
        alerts["state_fips"].map(STATE_TIMEZONE)
    )

    utc = pd.to_datetime(alerts["sent_utc"], utc=True, errors="coerce")
    alerts["sent_local"] = pd.NaT
    alerts["hour_local"] = pd.NA
    for tz_name, idx in alerts.groupby("tz_name").groups.items():
        local = utc.loc[idx].dt.tz_convert(pytz.timezone(tz_name))
        alerts.loc[idx, "sent_local"] = local.dt.tz_localize(None)
        alerts.loc[idx, "hour_local"] = local.dt.hour

    alerts = alerts.dropna(subset=["sent_local", "hour_local"]).copy()
    alerts["hour_local"] = alerts["hour_local"].astype(int)
    alerts["is_night"] = (alerts["hour_local"] >= 22) | (alerts["hour_local"] < 6)
    alerts = alerts[alerts["is_night"]].copy()
    alerts["alert_date"] = pd.to_datetime(alerts["sent_local"]).dt.normalize()
    alerts["effective_crash_date"] = alerts["alert_date"]
    early = alerts["hour_local"] >= 22
    alerts.loc[early, "effective_crash_date"] += pd.Timedelta(days=1)

    out = alerts.groupby(["fips", "effective_crash_date"], as_index=False).agg(
        n_alerts=("alert_id", "nunique")
    )
    out["night_alert"] = 1
    log.info("Verified county-level night-alert county-dates: %,d", len(out))
    return out


def build_panel() -> pd.DataFrame:
    crashes = load_state_crashes()
    crashes["date"] = pd.to_datetime(crashes["date"]).dt.normalize()
    crashes["year"] = crashes["date"].dt.year

    pop_path = DATA_PROC / "county_population.parquet"
    if not pop_path.exists():
        raise FileNotFoundError("county_population.parquet not found")
    pop = pd.read_parquet(pop_path)[["fips", "year", "population"]].copy()
    pop["fips"] = pop["fips"].astype(str).str.zfill(5)

    panel = crashes.merge(pop, on=["fips", "year"], how="left")
    panel = panel.dropna(subset=["population"])
    panel = panel[panel["population"] > 0].copy()
    for count, rate in [
        ("crashes", "crashes_per_100k"),
        ("fatals", "fatals_per_100k"),
        ("serious_inj", "serious_per_100k"),
    ]:
        panel[rate] = 100_000 * panel[count] / panel["population"]

    night_alerts = load_verified_night_alerts()
    panel = panel.merge(
        night_alerts[["fips", "effective_crash_date", "night_alert"]],
        left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
    )
    panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)
    panel = panel.drop(columns=["effective_crash_date"], errors="ignore")

    flows_path = DATA_PROC / "commuting" / "county_commuting_weights.parquet"
    if flows_path.exists():
        flows = pd.read_parquet(flows_path)
        spill = build_commuter_spillover(night_alerts, flows)
        panel = panel.merge(
            spill,
            left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
        )
        panel = panel.drop(columns=["effective_crash_date"], errors="ignore")
        log.info("Commuter spillover exposure merged from %s", flows_path)
    else:
        log.warning(
            "Commuting flows not found; run code/build_commuting_weights.py. "
            "Spillover-aware models will reduce to direct-alert models."
        )
        panel["spillover_commuters"] = 0.0
        panel["log_spillover_commuters"] = 0.0

    panel = add_spillover_classes(panel)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    return panel[panel["year"].between(2013, 2024)].copy()


def _coef_row(fit, name: str):
    tbl = fit.tidy()
    if name not in tbl.index:
        return None
    return (
        float(tbl.loc[name, "Estimate"]),
        float(tbl.loc[name, "Std. Error"]),
        float(tbl.loc[name, "Pr(>|t|)"]),
    )


def _treatments(panel: pd.DataFrame) -> tuple[str, ...]:
    if panel["log_spillover_commuters"].fillna(0).gt(0).any():
        return ("night_alert", "log_spillover_commuters")
    return ("night_alert",)


def run_wls(panel: pd.DataFrame, rate_col: str, label: str, *, clean_controls=False) -> list[dict]:
    import pyfixest as pf

    sub = panel.dropna(subset=[rate_col, "population", "night_alert"]).copy()
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    if len(sub) < 100 or sub["night_alert"].nunique() < 2:
        return []

    sub["_fips_str"] = sub["fips"].astype(str)
    sub["_date_str"] = sub["date"].astype(str)
    sub["_year_str"] = sub["year"].astype(str)
    sub["_pop"] = sub["population"].astype(float)
    treatments = ("night_alert",) if clean_controls else _treatments(sub)
    formula = f"{rate_col} ~ {' + '.join(treatments)} | _fips_str + _date_str"
    fit = pf.feols(formula, data=sub, weights="_pop", vcov={"CRV1": "_fips_str + _year_str"})

    rows = []
    for term in treatments:
        vals = _coef_row(fit, term)
        if vals is None:
            continue
        b, se, p = vals
        rows.append({
            "sample": "direct_vs_clean" if clean_controls else "spillover_joint",
            "state": label, "model": "WLS_TWFE", "outcome": rate_col, "term": term,
            "beta": b, "se": se, "pvalue": p, "n_obs": int(fit._N),
            "exposure_mode": "rate_per_100k_population_weighted",
        })
    return rows


def run_ppml(panel: pd.DataFrame, count_col: str, label: str, *, clean_controls=False) -> list[dict]:
    import pyfixest as pf

    treatments = ("night_alert",) if clean_controls else _treatments(panel)
    sub = prepare_ppml_sample(panel, count_col, treatment_cols=treatments)
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    if len(sub) < 100 or sub["night_alert"].nunique() < 2:
        return []

    spec = build_ppml_call_spec(pf.fepois, count_col=count_col, treatment_cols=treatments)
    kwargs = {"vcov": {"CRV1": "_fips_str + _year_str"}}
    if spec["offset"] is not None:
        kwargs["offset"] = spec["offset"]

    zero_share = float((sub[count_col] == 0).mean())
    log.info(
        "[%s %s] PPML sample %,d obs; %.1f%% zero outcomes; exposure=%s",
        label, count_col, len(sub), 100 * zero_share, spec["exposure_mode"],
    )
    fit = pf.fepois(spec["formula"], data=sub, **kwargs)

    rows = []
    for term in treatments:
        vals = _coef_row(fit, term)
        if vals is None:
            continue
        b, se, p = vals
        rows.append({
            "sample": "direct_vs_clean" if clean_controls else "spillover_joint",
            "state": label, "model": "PPML_raw_count", "outcome": count_col, "term": term,
            "beta": b, "se": se, "pvalue": p, "irr": float(np.exp(b)),
            "pct_change": float(100 * (np.exp(b) - 1)), "n_obs": int(fit._N),
            "zero_share_input": zero_share, "exposure_mode": spec["exposure_mode"],
        })
    return rows


def main() -> None:
    panel = build_panel()
    log.info(
        "Panel: %,d rows, %d counties, %d direct-alert days, %d spillover-only days, %d clean controls",
        len(panel), panel["fips"].nunique(), int((panel["exposure_class"] == "direct").sum()),
        int((panel["exposure_class"] == "spillover").sum()),
        int((panel["exposure_class"] == "clean_control").sum()),
    )

    desc = panel.groupby("state", as_index=False).agg(
        county_days=("fips", "size"),
        counties=("fips", "nunique"),
        direct_alert_days=("night_alert", "sum"),
        spillover_only_days=("exposure_class", lambda s: int((s == "spillover").sum())),
        crash_rows_available=("crashes", lambda s: int(s.notna().sum())),
        fatal_rows_available=("fatals", lambda s: int(s.notna().sum())),
        serious_rows_available=("serious_inj", lambda s: int(s.notna().sum())),
    )
    desc.to_csv(OUTPUT_TABS / "state_dot_descriptives_fixed.csv", index=False)

    outcomes = [
        ("crashes_per_100k", "crashes"),
        ("fatals_per_100k", "fatals"),
        ("serious_per_100k", "serious_inj"),
    ]
    results = []
    for state_filter in [None] + sorted(panel["state"].unique().tolist()):
        label = "ALL" if state_filter is None else state_filter
        sub = panel if state_filter is None else panel[panel["state"] == state_filter]
        if sub["night_alert"].sum() < 10:
            continue
        for rate_col, count_col in outcomes:
            if sub[count_col].notna().sum() < 100:
                continue
            results.extend(run_wls(sub, rate_col, label, clean_controls=False))
            results.extend(run_ppml(sub, count_col, label, clean_controls=False))
            results.extend(run_wls(sub, rate_col, label, clean_controls=True))
            results.extend(run_ppml(sub, count_col, label, clean_controls=True))

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_TABS / "state_dot_analysis_fixed.csv", index=False)
    log.info("Saved %d result rows → %s", len(out), OUTPUT_TABS / "state_dot_analysis_fixed.csv")


if __name__ == "__main__":
    main()
