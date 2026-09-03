"""Deterministic source-vintage selection for route exposure inputs."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class VintageChoice:
    """The selected source year and the rule used to select it."""

    target_year: int
    source_year: int | None
    gap: int | None
    status: str
    reason: str
    window_start: int | None = None
    window_end: int | None = None
    vintage: str | None = None


def _year(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def resolve_nearest_year(target_year: int, available_years: Iterable[int]) -> VintageChoice:
    """Choose an exact year, or the nearest year with earlier-year tie breaking."""

    target = _year(target_year, name="target_year")
    years = sorted({_year(year, name="available source year") for year in available_years})
    if not years:
        raise ValueError("no available source years")
    source = min(years, key=lambda year: (abs(year - target), year))
    gap = abs(source - target)
    if gap == 0:
        return VintageChoice(target, source, 0, "exact", "exact source year available")
    return VintageChoice(
        target, source, gap, "nearest", "nearest source year; earlier year wins ties"
    )


def resolve_acs_window(
    target_year: int, windows: Iterable[tuple[int, int, str]]
) -> VintageChoice:
    """Choose a containing ACS window, otherwise the nearest midpoint."""

    target = _year(target_year, name="target_year")
    candidates: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for window in windows:
        try:
            start, end, label = window
        except (TypeError, ValueError) as exc:
            raise ValueError("ACS windows must be (start, end, vintage) tuples") from exc
        start = _year(start, name="ACS window start")
        end = _year(end, name="ACS window end")
        if start > end:
            raise ValueError("ACS window start must not exceed end")
        if not isinstance(label, str) or not label.strip():
            raise TypeError("ACS vintage must be a non-empty string")
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        midpoint = (start + end) // 2
        candidates.append((start, end, midpoint, label))

    if not candidates:
        return VintageChoice(target, None, None, "unavailable", "no available ACS windows")

    containing = [row for row in candidates if row[0] <= target <= row[1]]
    if containing:
        start, end, midpoint, label = min(containing, key=lambda row: (abs(row[2] - target), row[2], row[3]))
        return VintageChoice(
            target,
            midpoint,
            abs(midpoint - target),
            "exact",
            f"target year contained in ACS window {label}",
            start,
            end,
            label,
        )

    start, end, midpoint, label = min(candidates, key=lambda row: (abs(row[2] - target), row[2], row[3]))
    return VintageChoice(
        target,
        midpoint,
        abs(midpoint - target),
        "nearest",
        f"nearest ACS window midpoint; earlier midpoint wins ties ({label})",
        start,
        end,
        label,
    )


def _sort_key(row: Mapping[str, object]) -> tuple[tuple[int, int], tuple[int, str], tuple[int, int]]:
    """Return comparable keys, placing absent sort fields after present ones."""

    analysis = row.get("analysis_year")
    if analysis is None:
        analysis_key = (1, 0)
    else:
        analysis_key = (0, _year(analysis, name="analysis_year"))

    state = row.get("state")
    state_key = (1, "") if state is None else (0, str(state).strip())

    source = row.get("source_year")
    if source is None:
        source = row.get("lodes_source_year")
    source_key = (1, 0) if source is None else (0, _year(source, name="source_year"))
    return (analysis_key, state_key, source_key)


def write_vintage_manifest(records: Iterable[Mapping[str, object]], path: Path) -> None:
    """Atomically write a schema-versioned, deterministically sorted manifest.

    JSON is the canonical format.  CSV paths are also supported for consumers
    that require row-oriented interchange.
    """

    destination = Path(path)
    rows = [dict(record) for record in records]
    rows.sort(key=_sort_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            if suffix == ".csv":
                csv_rows = [dict(row, schema_version="route_vintages.v1") for row in rows]
                fieldnames = sorted({key for row in csv_rows for key in row}) or ["schema_version"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            else:
                json.dump({"schema_version": "route_vintages.v1", "records": rows}, handle, indent=2, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
