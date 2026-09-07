"""Run a resumable national LODES flow collection with durable progress state."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from build_route_national_flows import NATIONAL_STATES, build_national_flow_year


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-year", type=int, default=2013)
    parser.add_argument("--states", nargs="+", default=list(NATIONAL_STATES))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/commuting/route_national/flows",
    )
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--crosswalk-batch-states", type=int, default=4)
    parser.add_argument("--stream-bucket-count", type=int, default=8)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    status_path = output_dir.parent / "execution" / f"national_flow_run_{int(args.analysis_year)}.json"
    status: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "started_at_utc": _utc_now(),
        "analysis_year": int(args.analysis_year),
        "states": sorted({str(state).lower() for state in args.states}),
        "chunk_rows": int(args.chunk_rows),
        "crosswalk_batch_states": int(args.crosswalk_batch_states),
        "stream_bucket_count": int(args.stream_bucket_count),
        "output_dir": str(output_dir),
    }
    _write_status(status_path, status)
    print(json.dumps(status, sort_keys=True), flush=True)
    try:
        manifest = build_national_flow_year(
            int(args.analysis_year), args.states, output_dir,
            chunk_rows=int(args.chunk_rows),
            crosswalk_batch_states=int(args.crosswalk_batch_states),
            stream_bucket_count=int(args.stream_bucket_count),
        )
        counts = manifest["partition_status"].value_counts().to_dict() if not manifest.empty else {}
        status.update({
            "status": "completed",
            "finished_at_utc": _utc_now(),
            "partition_status_counts": {str(key): int(value) for key, value in counts.items()},
        })
        _write_status(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        status.update({
            "status": "failed",
            "finished_at_utc": _utc_now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        _write_status(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
