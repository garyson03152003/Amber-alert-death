"""Strict download primitives used by crash-source adapters.

The APIs used by state transportation departments are generally forgiving:
they can return an HTTP 200 containing an error, silently stop paging, or
return the same row twice.  These helpers deliberately fail closed so that a
source adapter can mark the reporting unit incomplete instead of treating a
partial extract as a valid zero-balanced panel.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class IncompleteDownloadError(RuntimeError):
    """A paginated request did not produce the complete expected extract.

    The attributes are intentionally machine-readable for coverage manifests;
    the string includes the expected and fetched counts for useful logs.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_count: int | None = None,
        fetched_count: int = 0,
        terminal_error: object | None = None,
    ) -> None:
        self.expected_count = expected_count
        self.fetched_count = fetched_count
        self.terminal_error = terminal_error
        super().__init__(message)


def _failure(
    detail: str,
    *,
    expected_count: int | None,
    fetched_count: int,
    terminal_error: object,
) -> IncompleteDownloadError:
    count_text = (
        f"expected {expected_count}, fetched {fetched_count}"
        if expected_count is not None
        else f"fetched {fetched_count}"
    )
    return IncompleteDownloadError(
        f"{detail} ({count_text})",
        expected_count=expected_count,
        fetched_count=fetched_count,
        terminal_error=terminal_error,
    )


def _validate_count(value: object, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if number < 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _response_json(
    session: Any,
    url: str,
    *,
    params: Mapping[str, object],
    timeout: float,
    expected_count: int | None,
    fetched_count: int,
) -> object:
    """Request JSON and convert transport/HTTP/API failures to diagnostics."""
    try:
        response = session.get(url, params=dict(params), timeout=timeout)
        raise_for_status = getattr(response, "raise_for_status", None)
        if raise_for_status is not None:
            raise_for_status()
        payload = response.json()
    except IncompleteDownloadError:
        raise
    except Exception as exc:
        raise _failure(
            f"terminal request failure: {exc}",
            expected_count=expected_count,
            fetched_count=fetched_count,
            terminal_error=exc,
        ) from exc

    if isinstance(payload, Mapping) and payload.get("error") not in (None, "", False):
        error = payload["error"]
        if isinstance(error, Mapping):
            detail = error.get("message") or error.get("details") or error
        else:
            detail = error
        raise _failure(
            f"embedded API error: {detail}",
            expected_count=expected_count,
            fetched_count=fetched_count,
            terminal_error=error,
        )
    return payload


def _record_id(record: Mapping[str, object], id_field: str) -> object:
    if id_field not in record or record[id_field] is None or record[id_field] == "":
        raise KeyError(f"missing unique ID field {id_field!r}")
    value = record[id_field]
    try:
        hash(value)
    except TypeError as exc:
        raise KeyError(f"unhashable unique ID field {id_field!r}") from exc
    return value


def _check_records(
    records: list[Mapping[str, object]],
    *,
    id_field: str,
    seen_ids: set[object],
    expected_count: int,
) -> None:
    for record in records:
        if not isinstance(record, Mapping):
            raise _failure(
                "record is not an object",
                expected_count=expected_count,
                fetched_count=len(seen_ids),
                terminal_error="invalid_record",
            )
        try:
            record_id = _record_id(record, id_field)
        except KeyError as exc:
            raise _failure(
                str(exc),
                expected_count=expected_count,
                fetched_count=len(seen_ids),
                terminal_error="missing_id",
            ) from exc
        if record_id in seen_ids:
            raise _failure(
                f"duplicate record ID {record_id!r} in {id_field}",
                expected_count=expected_count,
                fetched_count=len(seen_ids),
                terminal_error="duplicate_id",
            )
        seen_ids.add(record_id)


def _arcgis_records(payload: object, *, expected_count: int, fetched_count: int) -> list[Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        raise _failure(
            "ArcGIS response is not an object",
            expected_count=expected_count,
            fetched_count=fetched_count,
            terminal_error="invalid_response",
        )
    features = payload.get("features")
    if not isinstance(features, list):
        raise _failure(
            "ArcGIS response has no feature list",
            expected_count=expected_count,
            fetched_count=fetched_count,
            terminal_error="invalid_response",
        )
    records: list[Mapping[str, object]] = []
    for feature in features:
        if not isinstance(feature, Mapping):
            records.append(feature)  # type: ignore[arg-type]
        elif isinstance(feature.get("attributes"), Mapping):
            records.append(feature["attributes"])  # type: ignore[arg-type]
        else:
            records.append(feature)
    return records


def fetch_arcgis_pages(
    session: Any,
    *,
    url: str,
    where: str | None = None,
    expected_count: int,
    page_size: int = 2_000,
    id_field: str,
    timeout: float = 120,
    out_fields: str = "*",
) -> list[dict[str, object]]:
    """Fetch all ArcGIS features, reconciling count and unique IDs."""
    expected = _validate_count(expected_count, "expected_count")
    size = _validate_count(page_size, "page_size")
    if size == 0:
        raise ValueError("page_size must be positive")
    if not id_field:
        raise ValueError("id_field is required for strict ArcGIS paging")
    if expected == 0:
        return []

    rows: list[dict[str, object]] = []
    seen_ids: set[object] = set()
    offset = 0
    while len(rows) < expected:
        params: dict[str, object] = {
            "f": "json",
            "where": where if where is not None else "1=1",
            "outFields": out_fields,
            "returnGeometry": "false",
            "orderByFields": f"{id_field} ASC",
            "resultOffset": offset,
            "resultRecordCount": size,
        }
        payload = _response_json(
            session,
            url,
            params=params,
            timeout=timeout,
            expected_count=expected,
            fetched_count=len(rows),
        )
        page = _arcgis_records(payload, expected_count=expected, fetched_count=len(rows))
        if not page:
            raise _failure(
                "empty page before expected count",
                expected_count=expected,
                fetched_count=len(rows),
                terminal_error="empty_page",
            )
        _check_records(page, id_field=id_field, seen_ids=seen_ids, expected_count=expected)
        rows.extend(dict(record) for record in page)
        offset += len(page)

    if len(rows) != expected:
        raise _failure(
            "final count mismatch",
            expected_count=expected,
            fetched_count=len(rows),
            terminal_error="count_mismatch",
        )
    return rows


def _socrata_count(payload: object) -> int:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], Mapping):
        raise ValueError("Socrata count response must be a non-empty row list")
    row = payload[0]
    value = row.get("count")
    if value is None:
        for key, candidate in row.items():
            if str(key).lower().startswith("count"):
                value = candidate
                break
    return _validate_count(value, "Socrata count")


def _candidate_socrata_id(records: list[Mapping[str, object]]) -> str | None:
    if not records:
        return None
    candidates = (":id", "id", "objectid", "OBJECTID", "crash_id", "case_id", "st_case")
    for candidate in candidates:
        if all(candidate in record for record in records):
            return candidate
    return None


def fetch_socrata_pages(
    session: Any,
    *,
    url: str,
    where: str | None = None,
    expected_count: int | None = None,
    page_size: int = 50_000,
    id_field: str | None = None,
    stable_id_field: str | None = None,
    timeout: float = 120,
) -> list[dict[str, object]]:
    """Count, then strictly page a Socrata dataset.

    ``id_field`` (or its descriptive alias ``stable_id_field``) should be
    supplied when the dataset has a stable source identifier.  If omitted, a
    conventional identifier is detected in returned rows for duplicate
    checking, but no unsupported ``$order`` expression is sent to Socrata.
    """
    size = _validate_count(page_size, "page_size")
    if size == 0:
        raise ValueError("page_size must be positive")
    if id_field and stable_id_field and id_field != stable_id_field:
        raise ValueError("id_field and stable_id_field disagree")
    requested_id = id_field or stable_id_field

    count_params: dict[str, object] = {"$select": "count(*)"}
    if where is not None:
        count_params["$where"] = where
    count_payload = _response_json(
        session,
        url,
        params=count_params,
        timeout=timeout,
        expected_count=expected_count,
        fetched_count=0,
    )
    try:
        counted = _socrata_count(count_payload)
    except (TypeError, ValueError) as exc:
        raise _failure(
            f"invalid Socrata count response: {exc}",
            expected_count=expected_count,
            fetched_count=0,
            terminal_error="invalid_count",
        ) from exc
    if expected_count is not None and _validate_count(expected_count, "expected_count") != counted:
        raise _failure(
            f"Socrata count query returned {counted}",
            expected_count=_validate_count(expected_count, "expected_count"),
            fetched_count=0,
            terminal_error="count_mismatch",
        )
    expected = counted
    if expected == 0:
        return []

    rows: list[dict[str, object]] = []
    seen_ids: set[object] = set()
    discovered_id = requested_id
    offset = 0
    while len(rows) < expected:
        params: dict[str, object] = {
            "$limit": size,
            "$offset": offset,
        }
        if where is not None:
            params["$where"] = where
        if requested_id:
            params["$order"] = f"{requested_id} ASC"
        payload = _response_json(
            session,
            url,
            params=params,
            timeout=timeout,
            expected_count=expected,
            fetched_count=len(rows),
        )
        if not isinstance(payload, list):
            raise _failure(
                "Socrata page response is not a row list",
                expected_count=expected,
                fetched_count=len(rows),
                terminal_error="invalid_response",
            )
        page = payload
        if not page:
            raise _failure(
                "empty page before expected count",
                expected_count=expected,
                fetched_count=len(rows),
                terminal_error="empty_page",
            )
        if any(not isinstance(record, Mapping) for record in page):
            raise _failure(
                "Socrata page contains a non-object row",
                expected_count=expected,
                fetched_count=len(rows),
                terminal_error="invalid_record",
            )
        if discovered_id is None:
            discovered_id = _candidate_socrata_id(page) or _candidate_socrata_id(rows)
        if discovered_id:
            _check_records(
                page,
                id_field=discovered_id,
                seen_ids=seen_ids,
                expected_count=expected,
            )
        rows.extend(dict(record) for record in page)
        offset += len(page)

    if len(rows) != expected:
        raise _failure(
            "final count mismatch",
            expected_count=expected,
            fetched_count=len(rows),
            terminal_error="count_mismatch",
        )
    return rows


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_bulk_file(
    session: Any,
    url: str,
    destination: str | os.PathLike[str],
    *,
    params: Mapping[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 120,
    chunk_size: int = 1 << 20,
) -> str:
    """Stream a bulk file atomically and return its SHA-256 digest.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic on the target filesystem.  Existing destinations are never changed
    if the request or stream fails.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(
        url,
        params=None if params is None else dict(params),
        headers=None if headers is None else dict(headers),
        stream=True,
        timeout=timeout,
    )
    raise_for_status = getattr(response, "raise_for_status", None)
    if raise_for_status is not None:
        raise_for_status()

    fd, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest()
