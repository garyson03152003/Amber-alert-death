"""Build collision-level California CCRS severity panels.

The legacy California builders use ``NumberInjured`` as a broad proxy.  This
adapter joins the collision file to ``InjuredWitnessPassengers`` and assigns
each collision exactly one class: fatal, serious, or nonserious.  Downloads
are cached with checksums and the module is import-safe; network I/O occurs
only through :func:`build` or the CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DEFAULT_RAW_DIR = DATA_RAW / "california_ccrs"
DEFAULT_OUTPUT_DIR = DATA_PROC
CATALOG_URL = "https://data.ca.gov/api/3/action/package_show"
PACKAGE_ID = "80c6a49d-c6b3-40ba-86d8-379c9741b4be"
YEARS = tuple(range(2016, 2023))

FATAL_INJURY_CODES = frozenset({"FATAL"})
SERIOUS_INJURY_CODES = frozenset({"SEVEREINACTIVE", "SUSPECTSERIOUS"})

DAY_OUTPUT_NAME = "california_ccrs_severity_county_day.parquet"
HOUR_OUTPUT_NAME = "california_ccrs_severity_county_hour.parquet"
COVERAGE_OUTPUT_NAME = "ca_severity_coverage.parquet"

CA_COUNTY_FIPS = {
    code: f"060{code:03d}"[-5:]
    for code in range(1, 116, 2)
}


class CaliforniaSeverityBuildError(RuntimeError):
    """Raised when a California severity source year cannot be validated."""


def _normal_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _column(frame: pd.DataFrame, *names: str, required: bool = True) -> str | None:
    lookup = {_normal_name(name): name for name in frame.columns}
    for name in names:
        found = lookup.get(_normal_name(name))
        if found is not None:
            return found
    if required:
        raise ValueError(f"missing required column; expected one of {names}")
    return None


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def county_to_fips(value: object) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or int(numeric) != numeric or not 1 <= int(numeric) <= 58:
        return None
    suffix = 2 * int(numeric) - 1
    return f"06{suffix:03d}"


def normalize_crashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a CCRS collision extract to one row per collision."""
    collision = _column(frame, "Collision Id", "CollisionId")
    date = _column(frame, "Crash Date Time")
    county = _column(frame, "County Code")
    killed = _column(frame, "NumberKilled", "Number Killed")
    version = _column(frame, "Report Version", required=False)

    crash_dt = pd.to_datetime(frame[date], errors="coerce")
    out = pd.DataFrame({
        "collision_id": frame[collision].astype("string").str.strip(),
        "date": crash_dt.dt.normalize(),
        "hour": crash_dt.dt.hour,
        "fips": frame[county].map(county_to_fips),
        "number_killed": pd.to_numeric(frame[killed], errors="coerce").fillna(0),
    })
    out["report_version"] = (
        pd.to_numeric(frame[version], errors="coerce").fillna(-np.inf)
        if version is not None else -np.inf
    )
    out = out.loc[
        out["collision_id"].notna()
        & out["collision_id"].ne("")
        & out["date"].notna()
        & out["fips"].notna()
    ].copy()
    out["date"] = out["date"].dt.normalize()
    out["number_killed"] = out["number_killed"].clip(lower=0)
    # CCRS occasionally republishes a collision with a higher report version.
    # Keep the latest version deterministically, then the first row on ties.
    duplicate_count = int(len(out) - out["collision_id"].nunique())
    out = (
        out.sort_values(["collision_id", "report_version"], kind="mergesort")
        .drop_duplicates("collision_id", keep="last")
        .reset_index(drop=True)
    )
    out.attrs["duplicate_crash_ids"] = duplicate_count
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype("int8")
    selected = out[["collision_id", "fips", "date", "hour", "number_killed"]].copy()
    selected.attrs["duplicate_crash_ids"] = duplicate_count
    return selected


def normalize_injuries(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize injury rows and retain only fields used for severity."""
    collision = _column(frame, "CollisionId", "Collision Id")
    witness = _column(frame, "IsWitnessOnly", required=False)
    extent = _column(frame, "ExtentOfInjuryCode")
    out = pd.DataFrame({
        "collision_id": frame[collision].astype("string").str.strip(),
        "is_witness": (_as_bool(frame[witness]) if witness is not None
                       else pd.Series(False, index=frame.index)),
        "injury_code": frame[extent].astype("string").str.strip(),
    })
    out["injury_code_norm"] = out["injury_code"].map(_normal_name)
    out.loc[out["injury_code"].isna(), "injury_code_norm"] = pd.NA
    return out


def classify_collisions(
    crashes: pd.DataFrame, injuries: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign one mutually exclusive severity to every retained collision."""
    required = {"collision_id", "fips", "date", "hour", "number_killed"}
    if not required.issubset(crashes.columns):
        try:
            crashes = normalize_crashes(crashes)
        except ValueError as exc:
            raise ValueError(f"crashes missing required columns: {sorted(required - set(crashes.columns))}") from exc
    else:
        crashes = crashes.copy()
    duplicate_count = int(crashes.attrs.get("duplicate_crash_ids", 0))
    injuries = normalize_injuries(injuries) if "injury_code_norm" not in injuries.columns else injuries.copy()
    valid_ids = set(crashes["collision_id"].astype("string"))
    orphan_count = int((~injuries["collision_id"].isin(valid_ids)).sum())
    injuries = injuries.loc[injuries["collision_id"].isin(valid_ids)].copy()
    diagnostics = {
        "crash_rows": int(len(crashes)),
        "duplicate_crash_ids": duplicate_count,
        "injury_rows": int(len(injuries)),
        "orphan_injury_rows": orphan_count,
        "witness_injury_rows_excluded": int(injuries["is_witness"].sum()),
        "unknown_injury_code_rows": 0,
    }
    # When called with raw fixtures, normalize_crashes performs duplicate
    # resolution and exposes the count only through the input row count.  The
    # production builder supplies this diagnostic explicitly below.
    involved = injuries.loc[~injuries["is_witness"]].copy()
    known = FATAL_INJURY_CODES | SERIOUS_INJURY_CODES | {
        "SUSPECTMINOR", "POSSIBLEINJURY", "COMPLAINTOFPAININACTIVE", "OTHERVISIBLEINACTIVE", "NOINJURY",
    }
    diagnostics["unknown_injury_code_rows"] = int((
        involved["injury_code_norm"].notna()
        & ~involved["injury_code_norm"].isin(known)
    ).sum())

    flags = (
        involved.assign(
            fatal_injury=involved["injury_code_norm"].isin(FATAL_INJURY_CODES),
            serious_injury=involved["injury_code_norm"].isin(SERIOUS_INJURY_CODES),
        )
        .groupby("collision_id", as_index=True)[["fatal_injury", "serious_injury"]]
        .max()
    )
    out = crashes.copy()
    out = out.join(flags, on="collision_id")
    out[["fatal_injury", "serious_injury"]] = (
        out[["fatal_injury", "serious_injury"]].astype("boolean").fillna(False)
    )
    out["severity"] = np.select(
        [out["number_killed"].gt(0) | out["fatal_injury"], out["serious_injury"]],
        ["fatal", "serious"],
        default="nonserious",
    )
    out["ca_fatal_crashes"] = out["severity"].eq("fatal").astype("int8")
    out["ca_serious_crashes"] = out["severity"].eq("serious").astype("int8")
    out["ca_nonserious_crashes"] = out["severity"].eq("nonserious").astype("int8")
    out["ca_crashes"] = 1
    out["ca_fatals"] = out["number_killed"].astype("int64")
    return out, diagnostics


def _empty_aggregate(grain: str) -> pd.DataFrame:
    keys = ["fips", "date"] + (["hour"] if grain == "hour" else [])
    cols = keys + [
        "ca_crashes", "ca_fatal_crashes", "ca_serious_crashes",
        "ca_nonserious_crashes", "ca_fatals",
    ]
    return pd.DataFrame(columns=cols)


def aggregate_severity(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate classified collisions to county-day and county-hour panels."""
    required = {"fips", "date", "hour", "ca_crashes", "ca_fatal_crashes", "ca_serious_crashes", "ca_nonserious_crashes", "ca_fatals"}
    missing = required - set(classified.columns)
    if missing:
        raise ValueError(f"classified collisions missing required columns: {sorted(missing)}")
    value_cols = ["ca_crashes", "ca_fatal_crashes", "ca_serious_crashes", "ca_nonserious_crashes", "ca_fatals"]
    if classified.empty:
        return _empty_aggregate("day"), _empty_aggregate("hour")
    day = classified.groupby(["fips", "date"], as_index=False)[value_cols].sum()
    hour = classified.groupby(["fips", "date", "hour"], as_index=False)[value_cols].sum()
    for frame in (day, hour):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["fips"] = frame["fips"].astype("string")
        for col in value_cols:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("int64")
        if not (frame["ca_crashes"] == frame[["ca_fatal_crashes", "ca_serious_crashes", "ca_nonserious_crashes"]].sum(axis=1)).all():
            raise ValueError("severity aggregation does not conserve crash counts")
    return day, hour


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, encoding="utf-8") as handle:
        tmp = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def discover_resources(session: requests.Session) -> dict[int, dict[str, str]]:
    response = session.get(CATALOG_URL, params={"id": PACKAGE_ID}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise CaliforniaSeverityBuildError("California CCRS catalog returned success=false")
    resources: dict[int, dict[str, str]] = {}
    for resource in payload["result"].get("resources", []):
        name = str(resource.get("name", ""))
        match = re.search(r"(Crashes|InjuredWitnessPassengers)_(\d{4})", name, flags=re.I)
        if not match or not resource.get("url"):
            continue
        year = int(match.group(2))
        kind = "crashes" if match.group(1).lower() == "crashes" else "injuries"
        resources.setdefault(year, {})[kind] = str(resource["url"])
    return resources


def download_resource(session: requests.Session, url: str, path: Path) -> tuple[Path, str]:
    """Stream one source file to an atomic cache path and return its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        digest = _sha256(path)
        sidecar = path.with_name(path.name + ".manifest.json")
        if sidecar.exists():
            try:
                prior = json.loads(sidecar.read_text(encoding="utf-8"))
                if prior.get("url") == url and prior.get("sha256") == digest:
                    return path, digest
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    last_error: BaseException | None = None
    for delay in (0, 5, 15, 30):
        if delay:
            time.sleep(delay)
        try:
            with session.get(
                url,
                stream=True,
                timeout=1200,
                headers={"User-Agent": "amber-alert-death/1.0", "Accept": "text/csv,*/*"},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
                            digest.update(block)
            temporary.replace(path)
            last_error = None
            break
        except BaseException as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
    if last_error is not None:
        raise last_error
    digest_text = digest.hexdigest()
    _atomic_json({"url": url, "sha256": digest_text, "bytes": path.stat().st_size}, path.with_name(path.name + ".manifest.json"))
    return path, digest_text


def _read_chunks(path: Path, *, usecols: list[str], chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    wanted = {_normal_name(name) for name in usecols}
    return pd.read_csv(
        path,
        usecols=lambda name: _normal_name(name) in wanted,
        chunksize=chunksize,
        encoding="latin1",
        skipinitialspace=True,
        low_memory=False,
    )


def build_year(
    year: int,
    session: requests.Session,
    raw_dir: Path = DEFAULT_RAW_DIR,
    resources: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Download and build one validated California severity year."""
    year = int(year)
    resources = resources or discover_resources(session)
    source = resources.get(year, {})
    if "crashes" not in source or "injuries" not in source:
        raise CaliforniaSeverityBuildError(f"missing CCRS crash/injury resource for {year}")
    year_dir = Path(raw_dir) / str(year)
    crash_path, crash_sha = download_resource(session, source["crashes"], year_dir / f"crashes_{year}.csv")
    injury_path, injury_sha = download_resource(session, source["injuries"], year_dir / f"injuredwitnesspassengers_{year}.csv")

    crash_parts = []
    duplicate_crash_ids = 0
    for chunk in _read_chunks(crash_path, usecols=["Collision Id", "Report Version", "Crash Date Time", "County Code", "NumberKilled"]):
        normalized = normalize_crashes(chunk)
        duplicate_crash_ids += int(normalized.attrs.get("duplicate_crash_ids", 0))
        crash_parts.append(normalized)
    crashes = pd.concat(crash_parts, ignore_index=True) if crash_parts else pd.DataFrame()
    crashes.attrs["duplicate_crash_ids"] = duplicate_crash_ids
    injury_parts = []
    for chunk in _read_chunks(injury_path, usecols=["CollisionId", "IsWitnessOnly", "ExtentOfInjuryCode"]):
        injury_parts.append(normalize_injuries(chunk))
    injuries = pd.concat(injury_parts, ignore_index=True) if injury_parts else pd.DataFrame()
    classified, diagnostics = classify_collisions(crashes, injuries)
    day, hour = aggregate_severity(classified)
    diagnostics.update({
        "year": year,
        "crash_sha256": crash_sha,
        "injury_sha256": injury_sha,
        "crash_url": source["crashes"],
        "injury_url": source["injuries"],
        "retained_collisions": int(len(classified)),
        "fatal_crashes": int(classified["ca_fatal_crashes"].sum()),
        "serious_crashes": int(classified["ca_serious_crashes"].sum()),
        "nonserious_crashes": int(classified["ca_nonserious_crashes"].sum()),
        "coverage_valid": bool(len(classified) > 0 and len(day) > 0),
    })
    if not diagnostics["coverage_valid"]:
        raise CaliforniaSeverityBuildError(f"no valid California collisions retained for {year}")
    return day, hour, diagnostics


def build(
    years: Iterable[int] = YEARS,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    """Build combined county-day/hour severity panels and coverage manifest."""
    wanted = sorted({int(year) for year in years})
    if not wanted:
        raise ValueError("at least one California source year is required")
    own_session = session or requests.Session()
    resources = discover_resources(own_session)
    days, hours, coverage = [], [], []
    for year in wanted:
        day, hour, diag = build_year(year, own_session, raw_dir=Path(raw_dir), resources=resources)
        days.append(day.assign(year=year))
        hours.append(hour.assign(year=year))
        coverage.append(diag)
    day_out = pd.concat(days, ignore_index=True).sort_values(["fips", "date"]).reset_index(drop=True)
    hour_out = pd.concat(hours, ignore_index=True).sort_values(["fips", "date", "hour"]).reset_index(drop=True)
    # Year is provenance only; the date already identifies it and the legacy
    # panel schema does not need a duplicate year column.
    day_out = day_out.drop(columns=["year"])
    hour_out = hour_out.drop(columns=["year"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    day_path = output_dir / DAY_OUTPUT_NAME
    hour_path = output_dir / HOUR_OUTPUT_NAME
    coverage_path = output_dir / "coverage" / COVERAGE_OUTPUT_NAME
    day_out.to_parquet(day_path, index=False)
    hour_out.to_parquet(hour_path, index=False)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).sort_values("year").to_parquet(coverage_path, index=False)
    return {"day": day_path, "hour": hour_path, "coverage": coverage_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    outputs = build(args.years, raw_dir=args.raw_dir, output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
