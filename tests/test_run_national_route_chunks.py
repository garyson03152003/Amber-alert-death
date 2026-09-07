from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_iter_flow_batches_preserves_partition_rows(tmp_path):
    from run_national_route_chunks import iter_flow_batches

    source = tmp_path / "flow.parquet"
    frame = pd.DataFrame({"route_id": ["a", "b", "c"], "workers": [1.0, 2.0, 3.0]})
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), source)

    batches = list(iter_flow_batches(source, batch_rows=2))

    assert [len(batch) for batch in batches] == [2, 1]
    assert pd.concat(batches, ignore_index=True).equals(frame)


def test_chunk_manifest_is_atomic_and_records_checksums(tmp_path):
    from run_national_route_chunks import write_chunk_manifest

    segment = tmp_path / "segments.parquet"
    audit = tmp_path / "audits.parquet"
    segment.write_bytes(b"segment")
    audit.write_bytes(b"audit")

    path = write_chunk_manifest(
        tmp_path / "chunk_manifest.json",
        analysis_year=2013,
        source_year=2013,
        work_state="wi",
        chunk_index=4,
        segment_path=segment,
        audit_path=audit,
        route_count=12,
    )

    payload = __import__("json").loads(path.read_text())
    assert payload["chunk_index"] == 4
    assert payload["route_count"] == 12
    assert len(payload["segment_sha256"]) == 64
    assert len(payload["audit_sha256"]) == 64


def test_wait_for_osrm_retries_until_endpoint_is_ready(monkeypatch):
    import requests
    from run_national_route_chunks import wait_for_osrm

    responses = [requests.ConnectionError("not listening"), type("Response", (), {"status_code": 200})()]
    calls = []

    def get(_self, url, timeout):
        calls.append((url, timeout))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    session = type("Session", (), {"get": get})()
    monkeypatch.setattr("run_national_route_chunks.time.sleep", lambda seconds: None)

    wait_for_osrm("http://127.0.0.1:5000", session=session, timeout_seconds=1, poll_seconds=0)

    assert len(calls) == 2
    assert calls[0][0].startswith("http://127.0.0.1:5000/")
