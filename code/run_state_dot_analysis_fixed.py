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

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent))
from config import CROSSWALK_RAW, DATA_PROC, DATA_RAW, OUTPUT_TABS
from utils import get_logger
from state_dot_analysis_core import (
    normalize_state_outcomes,
    prepare_ppml_sample,
    build_ppml_call_spec,
    build_commuter_spillover,
    add_spillover_classes,
    validate_analysis_inputs,
    extract_finite_coefficients,
    fit_status_row,
    summarize_fit_statuses,
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

# Alert-origin counties are nationwide.  Outcome panels are narrowed only
# after spillovers have been calculated from every valid US origin represented
# in the commuting-flow matrix.
NATIONWIDE_STATE_TIMEZONE = {
    "01": "America/Chicago", "02": "America/Anchorage", "04": "America/Phoenix",
    "05": "America/Chicago", "06": "America/Los_Angeles", "08": "America/Denver",
    "09": "America/New_York", "10": "America/New_York", "11": "America/New_York",
    "12": "America/New_York", "13": "America/New_York", "15": "Pacific/Honolulu",
    "16": "America/Boise", "17": "America/Chicago", "18": "America/Indiana/Indianapolis",
    "19": "America/Chicago", "20": "America/Chicago", "21": "America/New_York",
    "22": "America/Chicago", "23": "America/New_York", "24": "America/New_York",
    "25": "America/New_York", "26": "America/New_York", "27": "America/Chicago",
    "28": "America/Chicago", "29": "America/Chicago", "30": "America/Denver",
    "31": "America/Chicago", "32": "America/Los_Angeles", "33": "America/New_York",
    "34": "America/New_York", "35": "America/Denver", "36": "America/New_York",
    "37": "America/New_York", "38": "America/Chicago", "39": "America/New_York",
    "40": "America/Chicago", "41": "America/Los_Angeles", "42": "America/New_York",
    "44": "America/New_York", "45": "America/New_York", "46": "America/Chicago",
    "47": "America/Chicago", "48": "America/Chicago", "49": "America/Denver",
    "50": "America/New_York", "51": "America/New_York", "53": "America/Los_Angeles",
    "54": "America/New_York", "55": "America/Chicago", "56": "America/Denver",
}
ACCEPTED_STATE_YEARS = Path(__file__).resolve().parent.parent / "config" / "accepted_state_years.csv"

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
            "%s: %s county-days; available crash/fatal/serious rows = %s / %s / %s",
            state, f"{len(clean):,}", f"{clean['crashes'].notna().sum():,}",
            f"{clean['fatals'].notna().sum():,}", f"{clean['serious_inj'].notna().sum():,}",
        )
    if not parts:
        raise FileNotFoundError("No state-DOT county-day files were found")
    return pd.concat(parts, ignore_index=True)


def _load_coverage_manifests(coverage_dir: Path, states: set[str]) -> pd.DataFrame:
    """Read only the source manifests needed by the selected validated panels."""
    if not coverage_dir.is_dir():
        raise FileNotFoundError(f"validated coverage manifest directory not found: {coverage_dir}")
    parts = [pd.read_parquet(path) for path in sorted(coverage_dir.glob("*_coverage.parquet"))]
    if not parts:
        raise FileNotFoundError(f"no validated coverage manifests found under {coverage_dir}")
    combined = pd.concat(parts, ignore_index=True)
    if "state" not in combined.columns:
        raise ValueError("coverage manifests are missing state")
    combined["state"] = combined["state"].astype(str).str.upper()
    # write_manifest() normalizes the state column to numeric Census FIPS
    # (crash_coverage._manifest_frame's _state_code), so the 2-letter
    # abbreviations from the accepted-state-years review file must be
    # translated the same way before filtering.
    from state_dot_sources import STATE_SOURCE_SPECS
    fips_by_state = {
        state: STATE_SOURCE_SPECS[state].state_fips if state in STATE_SOURCE_SPECS else state
        for state in states
    }
    # Filtering on state FIPS alone is not enough: a diagnostic row from an
    # unrelated source can share the same FIPS code as a state-DOT source
    # (observed with Connecticut -- FARS's own "excluded from longitudinal
    # panel" policy row for state 09 is a different reporting unit entirely,
    # not a real CT_UCONN failure). Also require the row's source to match
    # this state's own spec.
    source_by_state = {
        state: STATE_SOURCE_SPECS[state].source for state in states if state in STATE_SOURCE_SPECS
    }
    fips_source_pairs = {
        (fips_by_state[state], source_by_state[state])
        for state in states if state in source_by_state
    }
    if "source" in combined.columns and fips_source_pairs:
        keep = combined.apply(
            lambda row: (row["state"], row["source"]) in fips_source_pairs, axis=1
        )
        selected = combined.loc[keep].copy()
    else:
        selected = combined.loc[combined["state"].isin(fips_by_state.values())].copy()
    selected["state"] = selected["state"].map(
        {fips: state for state, fips in fips_by_state.items()}
    ).fillna(selected["state"])
    missing = sorted(states - set(selected["state"]))
    if missing:
        raise FileNotFoundError(f"missing coverage manifests for validated states: {', '.join(missing)}")
    return selected


def load_validated_state_crashes(*, direct_only: bool = False, flows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load reviewed zero-balanced panels; never fall back to legacy sparse files."""
    if not ACCEPTED_STATE_YEARS.is_file():
        raise FileNotFoundError(f"reviewed accepted state-years file not found: {ACCEPTED_STATE_YEARS}")
    review = pd.read_csv(ACCEPTED_STATE_YEARS)
    if not {"state", "year"}.issubset(review.columns):
        raise ValueError("accepted state-years file must contain state and year")
    review = review.copy()
    review["state"] = review["state"].astype(str).str.upper()
    if "review_status" not in review.columns:
        review["review_status"] = "accepted"
    accepted = review.loc[review["review_status"].astype(str).str.lower().eq("accepted")].copy()
    if accepted.empty:
        raise ValueError("accepted state-years file contains no reviewed accepted units")

    panels: list[pd.DataFrame] = []
    states = set(accepted["state"])
    validated_dir = DATA_PROC / "validated"
    for state in sorted(states):
        path = validated_dir / f"{state.lower()}_county_day.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"validated balanced panel not found for {state}: {path}")
        panel = pd.read_parquet(path).copy()
        panel["state"] = state
        panels.append(panel)
    result = pd.concat(panels, ignore_index=True)
    manifest = _load_coverage_manifests(DATA_PROC / "coverage", states)
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    accepted_keys = accepted.loc[:, ["state", "year"]].copy()
    accepted_keys["year"] = pd.to_numeric(accepted_keys["year"], errors="raise").astype(int)
    manifest["year"] = pd.to_numeric(manifest["year"], errors="raise").astype(int)
    manifest = manifest.merge(accepted_keys.drop_duplicates(), on=["state", "year"], how="inner")
    panel_keys = set(map(tuple, result.loc[:, ["state", "year"]].drop_duplicates().to_records(index=False)))
    manifest_keys = set(map(tuple, manifest.loc[:, ["state", "year"]].drop_duplicates().to_records(index=False)))
    missing_manifest = sorted(panel_keys - manifest_keys)
    if missing_manifest:
        rendered = ", ".join(f"{state} {year}" for state, year in missing_manifest)
        raise ValueError(f"missing coverage manifest rows for validated panel units: {rendered}")
    validate_analysis_inputs(result, manifest, review, flows=flows, direct_only=direct_only)

    # Canonical Task 5 names are the only names accepted at this boundary.
    expected = {"crashes", "person_fatals", "serious_injury_persons"}
    missing = expected - set(result.columns)
    if missing:
        raise ValueError(f"validated panel missing canonical outcome columns: {sorted(missing)}")
    result = result.rename(columns={
        "person_fatals": "fatals", "serious_injury_persons": "serious_inj",
    })
    return result


NIGHT_START_HOUR = 22   # legacy default; see load_verified_alerts
NIGHT_END_HOUR = 6      # night runs [night_start, 24) U [0, NIGHT_END_HOUR)

_STATE_COUNTY_MAP: dict[str, list[str]] | None = None  # lazy-loaded singleton


def _state_county_map() -> dict[str, list[str]]:
    """state_fips -> every real county 5-digit FIPS in that state, built from
    the Census population-estimates file already in data/raw/crosswalks/."""
    global _STATE_COUNTY_MAP
    if _STATE_COUNTY_MAP is not None:
        return _STATE_COUNTY_MAP
    path = CROSSWALK_RAW / "co-est2023-alldata.csv"
    if not path.exists():
        log.warning("County crosswalk not found at %s — statewide alerts "
                    "will be dropped instead of expanded.", path)
        _STATE_COUNTY_MAP = {}
        return _STATE_COUNTY_MAP
    cw = pd.read_csv(path, encoding="latin1", usecols=["STATE", "COUNTY"])
    cw = cw[cw["COUNTY"] != 0]
    out: dict[str, list[str]] = {}
    for _, row in cw.iterrows():
        state_fips = f"{int(row['STATE']):02d}"
        out.setdefault(state_fips, []).append(f"{state_fips}{int(row['COUNTY']):03d}")
    _STATE_COUNTY_MAP = out
    return out


def _expand_statewide_rows(alerts: pd.DataFrame) -> pd.DataFrame:
    """Expand rows whose fips is a state-level SAME code ("XX000") into one
    row per real county in that state.

    ~65% of AMBER alerts are broadcast statewide rather than to a specific
    county. A prior version of this function excluded every such row
    outright (`alerts["fips"].str[2:] != "000"`), so every county-level
    causal analysis built on load_verified_alerts (the hourly event study,
    the state-DOT crash-share analyses, the commuting-spillover analyses)
    was estimated from only the minority of alerts issued at exact county
    granularity. Expanding restores the missing statewide alert-nights to
    the treatment definition instead of silently dropping them.
    """
    is_statewide = alerts["fips"].str[2:] == "000"
    if not is_statewide.any():
        return alerts
    state_map = _state_county_map()
    other = alerts[~is_statewide]
    statewide = alerts[is_statewide]
    if not state_map:
        return other
    expanded = []
    for _, row in statewide.iterrows():
        for cfips in state_map.get(row["fips"][:2], []):
            new_row = row.copy()
            new_row["fips"] = cfips
            expanded.append(new_row)
    if not expanded:
        return other
    log.info("Expanded %d statewide alert rows -> %d county-level rows",
             len(statewide), len(expanded))
    return pd.concat([other, pd.DataFrame(expanded)], ignore_index=True)


def load_verified_alerts(*, window: str = "night", detail: bool = False,
                         night_start: int = NIGHT_START_HOUR,
                         night_end: int = NIGHT_END_HOUR) -> pd.DataFrame:
    """Verified county-level AMBER-alert exposure in a time-of-day window.

    ``window="night"`` keeps alerts sent ``night_start``:00-05:59 local, and
    assigns any alert sent at or after ``night_start`` to the *following*
    calendar date -- the overnight window spans two dates and the driving it
    could plausibly affect is day D+1's.

    ``night_start`` matters more than it looks. The legacy default of 22
    excludes the 20:00 and 21:59 hours, which hold 4,982 alerts -- 69% more
    night exposure than the 22:00 cutoff admits. An evening alert at 20:30 is
    an overnight alert by any behavioural reading (it reaches drivers before
    the overnight period and the outcome of interest is the next day), so
    ``night_start=20`` is the substantively motivated choice and 22 is
    retained only for continuity with earlier runs.

    ``window="day"`` keeps the complement -- alerts sent 06:00 up to
    ``night_start`` -- and assigns them to the *same* calendar date, because a
    daytime alert affects that day's driving and the next-day shift used for
    overnight alerts would be wrong here.

    This same-day assignment is why the two windows cannot share one flag:
    they are different exposure timings, not just different hour filters.

    ``detail=False`` returns one row per (fips, effective_crash_date) with a
    binary treatment flag named ``night_alert`` or ``day_alert``.
    ``detail=True`` returns the underlying per-alert rows with their local
    send timestamp/hour, for analyses (e.g. the traffic-volume station-hour
    panel) that need sub-day alert timing rather than a county-day flag.

    The verification, FIPS-validation, and DST-aware timezone logic is shared
    across both windows rather than duplicated.
    """
    if window not in {"night", "day"}:
        raise ValueError(f"window must be 'night' or 'day', got {window!r}")
    if not night_end < night_start <= 23:
        raise ValueError(
            f"night_start must be in ({night_end}, 23], got {night_start!r}")
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
    alerts = _expand_statewide_rows(alerts)
    alerts = alerts[alerts["state_fips"].isin(NATIONWIDE_STATE_TIMEZONE)].copy()
    alerts["tz_name"] = alerts["fips"].map(COUNTY_TIMEZONE_OVERRIDE).fillna(
        alerts["state_fips"].map(NATIONWIDE_STATE_TIMEZONE)
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
    is_night = (alerts["hour_local"] >= night_start) | (alerts["hour_local"] < night_end)
    alerts = alerts[is_night if window == "night" else ~is_night].copy()
    alerts["alert_date"] = pd.to_datetime(alerts["sent_local"]).dt.normalize()
    alerts["effective_crash_date"] = alerts["alert_date"]
    if window == "night":
        # An alert sent in the evening belongs to the NEXT day's driving: the
        # overnight window runs night_start -> 06:00 across a date boundary.
        evening = alerts["hour_local"] >= night_start
        alerts.loc[evening, "effective_crash_date"] += pd.Timedelta(days=1)

    if detail:
        msg_type_col = ["msg_type"] if "msg_type" in alerts.columns else []
        return alerts[msg_type_col + [
            "alert_id", "fips", "state_fips", "tz_name",
            "sent_local", "hour_local", "effective_crash_date",
        ]].reset_index(drop=True)

    flag = "night_alert" if window == "night" else "day_alert"
    out = alerts.groupby(["fips", "effective_crash_date"], as_index=False).agg(
        n_alerts=("alert_id", "nunique")
    )
    out[flag] = 1
    log.info("Verified county-level %s-alert county-dates: %s", window, f"{len(out):,}")
    return out


def load_verified_night_alerts(*, detail: bool = False) -> pd.DataFrame:
    """Night-window alerts; thin wrapper preserving the original entry point."""
    return load_verified_alerts(window="night", detail=detail)


def build_panel(*, direct_only: bool = False) -> pd.DataFrame:
    flows_path = DATA_PROC / "commuting" / "county_commuting_weights.parquet"
    flows = pd.read_parquet(flows_path) if flows_path.is_file() else None
    crashes = load_validated_state_crashes(direct_only=direct_only, flows=flows)
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

    night_alerts = load_verified_alerts(window="night")
    panel = panel.merge(
        night_alerts[["fips", "effective_crash_date", "night_alert"]],
        left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
    )
    panel["night_alert"] = panel["night_alert"].fillna(0).astype(int)
    panel = panel.drop(columns=["effective_crash_date"], errors="ignore")

    # Daytime alerts are carried alongside the night flag so a specification
    # can hold one constant while estimating the other: a county-day can have
    # both, and attributing such a day to a single window would confound them.
    day_alerts = load_verified_alerts(window="day")
    panel = panel.merge(
        day_alerts[["fips", "effective_crash_date", "day_alert"]],
        left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
    )
    panel["day_alert"] = panel["day_alert"].fillna(0).astype(int)
    panel = panel.drop(columns=["effective_crash_date"], errors="ignore")

    if not direct_only:
        assert flows is not None  # validated above; makes the branch explicit.
        spill = build_commuter_spillover(night_alerts, flows)
        panel = panel.merge(
            spill,
            left_on=["fips", "date"], right_on=["fips", "effective_crash_date"], how="left",
        )
        panel = panel.drop(columns=["effective_crash_date"], errors="ignore")
        log.info("Commuter spillover exposure merged from %s", flows_path)
    else:
        panel["spillover_commuters"] = 0.0
        panel["spillover_share"] = 0.0
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
    if panel.get("log_spillover_commuters", pd.Series(0.0, index=panel.index)).fillna(0).gt(0).any():
        return ("night_alert", "log_spillover_commuters")
    return ("night_alert",)


def _diagnostic(
    *, label: str, model: str, outcome: str, sample: str, status: str,
    input_n: int, fitted_n: int, zero_share: float | None,
    terms_requested: tuple[str, ...], terms_produced: tuple[str, ...] = (),
    error_reason: str | None = None,
) -> dict[str, object]:
    return {
        "state": label, "model": model, "outcome": outcome, "sample": sample,
        **fit_status_row(
            status=status, input_n=input_n, fitted_n=fitted_n,
            zero_share=zero_share, terms_requested=terms_requested,
            terms_produced=terms_produced, error_reason=error_reason,
        ),
    }


def run_wls(panel: pd.DataFrame, rate_col: str, label: str, *, clean_controls=False, direct_only: bool = False) -> list[dict]:
    import pyfixest as pf

    sub = panel.dropna(subset=[rate_col, "population", "night_alert"]).copy()
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    sample = "direct_vs_clean" if clean_controls else ("direct_only" if direct_only else "spillover_joint")
    treatments = ("night_alert",) if (clean_controls or direct_only) else _treatments(sub)
    zero_share = float((sub[rate_col] == 0).mean()) if len(sub) else None
    if len(sub) < 100 or sub["night_alert"].nunique() < 2:
        return [_diagnostic(
            label=label, model="WLS_TWFE", outcome=rate_col, sample=sample,
            status="skipped", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason="insufficient_estimable_sample",
        )]
    sub["_fips_str"] = sub["fips"].astype(str)
    sub["_date_str"] = sub["date"].astype(str)
    sub["_year_str"] = sub["year"].astype(str)
    sub["_pop"] = sub["population"].astype(float)
    formula = f"{rate_col} ~ {' + '.join(treatments)} | _fips_str + _date_str"
    try:
        fit = pf.feols(formula, data=sub, weights="_pop", vcov={"CRV1": "_fips_str + _year_str"})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return [_diagnostic(
            label=label, model="WLS_TWFE", outcome=rate_col, sample=sample,
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason=str(exc),
        )]

    coefficients, produced, errors = extract_finite_coefficients(fit, treatments)
    rows = []
    for coefficient in coefficients:
        b, se, p = coefficient["beta"], coefficient["se"], coefficient["pvalue"]
        rows.append({
            "record_type": "estimate", "status": "ok", "sample": sample,
            "state": label, "model": "WLS_TWFE", "outcome": rate_col, "term": coefficient["term"],
            "beta": b, "se": se, "pvalue": p, "n_obs": int(fit._N),
            "exposure_mode": "rate_per_100k_population_weighted",
        })
    error_reason = next(iter(set(errors.values()))) if len(set(errors.values())) == 1 else "; ".join(
        f"{term}:{reason}" for term, reason in errors.items()
    )
    rows.append(_diagnostic(
        label=label, model="WLS_TWFE", outcome=rate_col, sample=sample,
        status="ok" if not errors else ("partial" if produced else "failed"),
        input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=treatments, terms_produced=produced, error_reason=error_reason,
    ))
    return rows


def run_ppml(panel: pd.DataFrame, count_col: str, label: str, *, clean_controls=False, direct_only: bool = False) -> list[dict]:
    import pyfixest as pf

    treatments = ("night_alert",) if (clean_controls or direct_only) else _treatments(panel)
    sub = prepare_ppml_sample(panel, count_col, treatment_cols=treatments)
    if clean_controls:
        sub = sub[(sub["night_alert"] == 1) | (sub["clean_control"] == 1)].copy()
    sample = "direct_vs_clean" if clean_controls else ("direct_only" if direct_only else "spillover_joint")
    zero_share = float((sub[count_col] == 0).mean()) if len(sub) else None
    if len(sub) < 100 or sub["night_alert"].nunique() < 2:
        return [_diagnostic(
            label=label, model="PPML_raw_count", outcome=count_col, sample=sample,
            status="skipped", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason="insufficient_estimable_sample",
        )]

    spec = build_ppml_call_spec(pf.fepois, count_col=count_col, treatment_cols=treatments)
    kwargs = {"vcov": {"CRV1": "_fips_str + _year_str"}}
    if spec["offset"] is not None:
        kwargs["offset"] = spec["offset"]

    log.info(
        "[%s %s] PPML sample %s obs; %.1f%% zero outcomes; exposure=%s",
        label, count_col, f"{len(sub):,}", 100 * zero_share, spec["exposure_mode"],
    )
    try:
        fit = pf.fepois(spec["formula"], data=sub, **kwargs)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return [_diagnostic(
            label=label, model="PPML_raw_count", outcome=count_col, sample=sample,
            status="failed", input_n=len(sub), fitted_n=0, zero_share=zero_share,
            terms_requested=treatments, error_reason=str(exc),
        )]

    coefficients, produced, errors = extract_finite_coefficients(fit, treatments)
    rows = []
    for coefficient in coefficients:
        b, se, p = coefficient["beta"], coefficient["se"], coefficient["pvalue"]
        rows.append({
            "record_type": "estimate", "status": "ok", "sample": sample,
            "state": label, "model": "PPML_raw_count", "outcome": count_col, "term": coefficient["term"],
            "beta": b, "se": se, "pvalue": p, "irr": float(np.exp(b)),
            "pct_change": float(100 * (np.exp(b) - 1)), "n_obs": int(fit._N),
            "zero_share_input": zero_share, "exposure_mode": spec["exposure_mode"],
        })
    error_reason = next(iter(set(errors.values()))) if len(set(errors.values())) == 1 else "; ".join(
        f"{term}:{reason}" for term, reason in errors.items()
    )
    rows.append(_diagnostic(
        label=label, model="PPML_raw_count", outcome=count_col, sample=sample,
        status="ok" if not errors else ("partial" if produced else "failed"),
        input_n=len(sub), fitted_n=int(fit._N), zero_share=zero_share,
        terms_requested=treatments, terms_produced=produced, error_reason=error_reason,
    ))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-only", action="store_true",
        help="run the explicitly requested direct-alert-only sensitivity; do not relabel it as spillover_joint",
    )
    args = parser.parse_args(argv)
    panel = build_panel(direct_only=args.direct_only)
    log.info(
        "Panel: %s rows, %d counties, %d direct-alert days, %d spillover-only days, %d clean controls",
        f"{len(panel):,}", panel["fips"].nunique(), int((panel["exposure_class"] == "direct").sum()),
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
        for rate_col, count_col in outcomes:
            results.extend(run_wls(sub, rate_col, label, clean_controls=False, direct_only=args.direct_only))
            results.extend(run_ppml(sub, count_col, label, clean_controls=False, direct_only=args.direct_only))
            results.extend(run_wls(sub, rate_col, label, clean_controls=True))
            results.extend(run_ppml(sub, count_col, label, clean_controls=True))

    all_rows = pd.DataFrame(results)
    out = all_rows.loc[all_rows.get("record_type", pd.Series(dtype=str)).eq("estimate")].copy()
    statuses = all_rows.loc[all_rows.get("record_type", pd.Series(dtype=str)).eq("fit_status")].copy()
    statuses = pd.concat([statuses, pd.DataFrame([{
        "record_type": "model_count_summary", **summarize_fit_statuses(statuses.to_dict("records")),
    }])], ignore_index=True)
    out.to_csv(OUTPUT_TABS / "state_dot_analysis_fixed.csv", index=False)
    statuses.to_csv(OUTPUT_TABS / "state_dot_analysis_fixed_status.csv", index=False)
    log.info("Saved %d estimates and %d fit diagnostics → %s", len(out), len(statuses) - 1, OUTPUT_TABS)


if __name__ == "__main__":
    main()
