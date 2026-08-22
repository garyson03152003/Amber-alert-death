"""Parse FHWA TMAS Hourly Traffic Volume (.VOL) and Station Description (.STA)
files, in both formats FHWA has published data in:

  - "legacy" fixed-width format (pre-2020 monthly archives): a 2-digit year
    field, confirmed byte-for-byte against the 2013 Traffic Monitoring Guide
    (Chapter 7) record layout except for a 2-digit (not 4-digit) year --
    verified against ground truth by checking the day-of-week field for a
    known calendar date (Jan 1, 2015 parses to day_of_week=5/Thursday, which
    is correct).
  - "modern" pipe-delimited format (2020-present monthly archives): a
    header row followed by pipe-delimited fields, confirmed directly
    against a live download.

Both formats are converted to the same long station-date-hour schema.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Legacy fixed-width format (pre-2020 monthly archives)
# ---------------------------------------------------------------------------
# Byte offsets are 0-indexed [start, end) python slice bounds, derived from
# the 2013 TMG Chapter 7 Table 7-9 layout, with the year field shortened to
# 2 digits (columns 14-15 instead of 14-17) to match what FHWA's own
# pre-2020 archived files actually contain.
_LEGACY_VOL_FIELDS = {
    "record_type": (0, 1),
    "state_fips": (1, 3),
    "f_system": (3, 5),
    "station_id": (5, 11),
    "travel_dir": (11, 12),
    "travel_lane": (12, 13),
    "year": (13, 15),
    "month": (15, 17),
    "day": (17, 19),
    "day_of_week": (19, 20),
}
_LEGACY_VOL_HOUR_START = 20  # 0-indexed start of the first 5-char hour field
_LEGACY_VOL_HOUR_WIDTH = 5
_LEGACY_VOL_N_HOURS = 24


def _to_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_legacy_vol_line(line: str) -> dict | None:
    """Parse one fixed-width Hourly Traffic Volume record ("legacy" format).

    Returns None for non-volume record types (record_type != "3") so mixed
    files can be filtered without raising.
    """
    if len(line) < _LEGACY_VOL_HOUR_START + _LEGACY_VOL_HOUR_WIDTH * _LEGACY_VOL_N_HOURS:
        return None
    record_type = line[0:1]
    if record_type != "3":
        return None
    out: dict[str, object] = {}
    for name, (start, end) in _LEGACY_VOL_FIELDS.items():
        out[name] = line[start:end].strip()
    year2 = _to_int(out["year"])
    if year2 is None:
        return None
    # 2-digit years in this archive are all 2000s (data starts 2010).
    out["year"] = 2000 + year2
    out["month"] = _to_int(out["month"])
    out["day"] = _to_int(out["day"])
    out["day_of_week"] = _to_int(out["day_of_week"])
    hours = []
    pos = _LEGACY_VOL_HOUR_START
    for _ in range(_LEGACY_VOL_N_HOURS):
        hours.append(_to_int(line[pos:pos + _LEGACY_VOL_HOUR_WIDTH]))
        pos += _LEGACY_VOL_HOUR_WIDTH
    out["hours"] = hours
    return out


# ---------------------------------------------------------------------------
# Station Description format: unlike Volume data, FHWA has retroactively
# re-released station-description archives for every year (2010-present) in
# a single normalized pipe-delimited layout (confirmed directly against a
# downloaded 2015 archive: `Station_Data_Extract_Pipe_Delimited_CleanData_
# 2015.txt`, same column set as a 2023 archive, just different casing) --
# there is no separate legacy fixed-width Station format to support.
# ---------------------------------------------------------------------------
def parse_modern_vol_bytes(raw: bytes) -> pd.DataFrame:
    """Parse a pipe-delimited (2020+) .VOL file into long station-date-hour rows."""
    text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    hour_cols = [f"hour_{h:02d}" for h in range(24)]
    missing_hours = [c for c in hour_cols if c not in df.columns]
    if missing_hours:
        raise ValueError(f"modern VOL file missing expected hour columns: {missing_hours}")
    keep = ["state_code", "station_id", "travel_dir", "travel_lane",
            "year_record", "month_record", "day_record", "day_of_week"] + hour_cols
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"modern VOL file missing expected columns: {missing}")
    out = df[keep].copy()
    out = out.rename(columns={
        "state_code": "state_fips", "year_record": "year",
        "month_record": "month", "day_record": "day",
    })
    for col in ("year", "month", "day", "day_of_week", *hour_cols):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["hours"] = out[hour_cols].apply(lambda row: [None if pd.isna(v) else int(v) for v in row], axis=1)
    return out.drop(columns=hour_cols)


def _decode_coord(raw: str, *, negate: bool) -> float | None:
    """Decode a TMG lat/lon field: an implied decimal 6 places from the
    right (e.g. "31076964" -> 31.076964), north/east assumed positive per
    the spec -- longitude is always west in the US, so `negate=True` there.
    """
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text) / 1_000_000
    except ValueError:
        return None
    return -value if negate else value


def parse_modern_sta_bytes(raw: bytes) -> pd.DataFrame:
    text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    keep = {"state_code": "state_fips", "station_id": "station_id",
            "county_code": "county_fips", "latitude": "latitude", "longitude": "longitude"}
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"modern STA file missing expected columns: {missing}")
    out = df[list(keep)].rename(columns=keep).copy()
    out["latitude"] = out["latitude"].map(lambda v: _decode_coord(v, negate=False))
    out["longitude"] = out["longitude"].map(lambda v: _decode_coord(v, negate=True))
    return out


# ---------------------------------------------------------------------------
# Whole-zip readers: each monthly/annual zip contains one file per state.
# ---------------------------------------------------------------------------
def read_legacy_vol_zip(zip_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.upper().endswith(".VOL"):
                continue
            with zf.open(name) as fh:
                for raw_line in io.TextIOWrapper(fh, encoding="latin-1"):
                    line = raw_line.rstrip("\n\r")
                    if not line:
                        continue
                    parsed = parse_legacy_vol_line(line)
                    if parsed is not None:
                        rows.append(parsed)
    return pd.DataFrame(rows)


def read_modern_vol_zip(zip_path: Path) -> pd.DataFrame:
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.upper().endswith(".VOL"):
                continue
            with zf.open(name) as fh:
                raw = fh.read()
            if not raw.strip():
                continue
            parts.append(parse_modern_vol_bytes(raw))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def read_sta_zip(zip_path: Path) -> pd.DataFrame:
    """Read every non-directory member of a station-data zip.

    Station-data archives are not consistently named `.STA` -- FHWA's
    per-year re-releases use e.g. `Station_Data_Extract_Pipe_Delimited_
    CleanData_2015.txt` for older years and `<state>_<year> (TMAS).STA` for
    newer ones -- so every member is attempted rather than filtering by a
    specific extension.
    """
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            with zf.open(name) as fh:
                raw = fh.read()
            if not raw.strip():
                continue
            parts.append(parse_modern_sta_bytes(raw))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def read_vol_zip(zip_path: Path, *, modern: bool) -> pd.DataFrame:
    return read_modern_vol_zip(zip_path) if modern else read_legacy_vol_zip(zip_path)
