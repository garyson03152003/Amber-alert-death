import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from crash_download import (  # noqa: E402
    IncompleteDownloadError,
    download_bulk_file,
    fetch_arcgis_pages,
    fetch_socrata_pages,
    sha256_file,
)


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", error=None):
        self.payload = payload
        self.content = content
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self.content), max(1, chunk_size)):
            yield self.content[start:start + max(1, chunk_size)]


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_arcgis_pages_extracts_attributes_and_orders_by_unique_id():
    session = FakeSession([
        FakeResponse({"features": [
            {"attributes": {"OBJECTID": 1, "YEAR": 2024}},
            {"attributes": {"OBJECTID": 2, "YEAR": 2024}},
        ]}),
        FakeResponse({"features": [
            {"attributes": {"OBJECTID": 3, "YEAR": 2024}},
        ]}),
    ])

    rows = fetch_arcgis_pages(
        session,
        url="https://example.test/query",
        where="YEAR=2024",
        expected_count=3,
        page_size=2,
        id_field="OBJECTID",
    )

    assert rows == [
        {"OBJECTID": 1, "YEAR": 2024},
        {"OBJECTID": 2, "YEAR": 2024},
        {"OBJECTID": 3, "YEAR": 2024},
    ]
    first_params = session.calls[0][1]["params"]
    assert first_params["orderByFields"] == "OBJECTID ASC"
    assert first_params["resultOffset"] == 0
    assert first_params["resultRecordCount"] == 2
    assert first_params["where"] == "YEAR=2024"


def test_arcgis_empty_page_before_expected_count_is_incomplete():
    session = FakeSession([
        FakeResponse({"features": [
            {"attributes": {"OBJECTID": 1}},
            {"attributes": {"OBJECTID": 2}},
        ]}),
        FakeResponse({"features": []}),
    ])

    with pytest.raises(IncompleteDownloadError, match="expected 3, fetched 2") as caught:
        fetch_arcgis_pages(
            session,
            url="https://example.test/query",
            where="YEAR=2024",
            expected_count=3,
            page_size=2,
            id_field="OBJECTID",
        )

    assert caught.value.expected_count == 3
    assert caught.value.fetched_count == 2
    assert caught.value.terminal_error == "empty_page"


def test_arcgis_rejects_duplicate_ids_and_embedded_api_errors():
    duplicate_session = FakeSession([
        FakeResponse({"features": [
            {"attributes": {"OBJECTID": 1}},
            {"attributes": {"OBJECTID": 1}},
        ]}),
    ])
    with pytest.raises(IncompleteDownloadError, match="duplicate.*OBJECTID"):
        fetch_arcgis_pages(
            duplicate_session,
            url="https://example.test/query",
            expected_count=2,
            page_size=2,
            id_field="OBJECTID",
        )

    api_error_session = FakeSession([
        FakeResponse({"error": {"code": 400, "message": "bad where"}}),
    ])
    with pytest.raises(IncompleteDownloadError, match="bad where") as caught:
        fetch_arcgis_pages(
            api_error_session,
            url="https://example.test/query",
            expected_count=1,
            page_size=1,
            id_field="OBJECTID",
        )
    assert caught.value.fetched_count == 0
    assert caught.value.terminal_error is not None


def test_socrata_count_query_precedes_ordered_pages_and_reconciles_count():
    session = FakeSession([
        FakeResponse([{"count": "3"}]),
        FakeResponse([
            {"id": "a", "year": "2024"},
            {"id": "b", "year": "2024"},
        ]),
        FakeResponse([{"id": "c", "year": "2024"}]),
    ])

    rows = fetch_socrata_pages(
        session,
        url="https://example.test/resource/abcd.json",
        where="year = 2024",
        page_size=2,
        id_field="id",
    )

    assert rows == [
        {"id": "a", "year": "2024"},
        {"id": "b", "year": "2024"},
        {"id": "c", "year": "2024"},
    ]
    assert session.calls[0][1]["params"]["$select"] == "count(*)"
    assert session.calls[1][1]["params"] == {
        "$where": "year = 2024",
        "$limit": 2,
        "$offset": 0,
        "$order": "id ASC",
    }


def test_socrata_rejects_duplicate_ids_and_early_empty_page():
    duplicate_session = FakeSession([
        FakeResponse([{"count": "2"}]),
        FakeResponse([{"id": "a"}, {"id": "a"}]),
    ])
    with pytest.raises(IncompleteDownloadError, match="duplicate.*id"):
        fetch_socrata_pages(
            duplicate_session,
            url="https://example.test/resource/abcd.json",
            page_size=2,
            id_field="id",
        )

    incomplete_session = FakeSession([
        FakeResponse([{"count": "3"}]),
        FakeResponse([{"id": "a"}, {"id": "b"}]),
        FakeResponse([]),
    ])
    with pytest.raises(IncompleteDownloadError, match="expected 3, fetched 2") as caught:
        fetch_socrata_pages(
            incomplete_session,
            url="https://example.test/resource/abcd.json",
            page_size=2,
            id_field="id",
        )
    assert caught.value.terminal_error == "empty_page"


def test_bulk_download_is_atomic_and_returns_sha256(tmp_path):
    destination = tmp_path / "nested" / "download.zip"
    payload = b"first\nsecond\n"
    session = FakeSession([FakeResponse(content=payload)])

    checksum = download_bulk_file(
        session,
        "https://example.test/download.zip",
        destination,
        chunk_size=5,
    )

    expected = hashlib.sha256(payload).hexdigest()
    assert checksum == expected
    assert destination.read_bytes() == payload
    assert sha256_file(destination) == expected
    assert list(destination.parent.iterdir()) == [destination]
    assert session.calls[0][1]["stream"] is True


def test_bulk_download_failure_preserves_existing_destination(tmp_path):
    destination = tmp_path / "download.zip"
    destination.write_bytes(b"known-good")
    session = FakeSession([FakeResponse(error=RuntimeError("network down"))])

    with pytest.raises(RuntimeError, match="network down"):
        download_bulk_file(session, "https://example.test/download.zip", destination)

    assert destination.read_bytes() == b"known-good"
    assert list(tmp_path.iterdir()) == [destination]
