"""Rewrite Parquet files into a reader-compatible representation.

Some processed artifacts in this repository were written with Arrow 24/25,
while the analysis environment may use an older Arrow reader.  The older
reader can decode the source files, but the active reader raises a repetition
level histogram error.  This utility rewrites each file through the legacy
reader and replaces it only after a row-count/schema check succeeds.

Run this module with a Python environment that has a reader able to decode the
source files (the local pyarrow 12 environment is suitable here), for example:

    /Users/chinghinyang/opt/anaconda3/bin/python \
        code/repair_parquet_compat.py data/processed/fars_balanced_county_day.parquet
"""
from __future__ import annotations

import argparse
import os
import stat
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


def rewrite_file(path: Path) -> tuple[int, int]:
    """Atomically rewrite one Parquet file and return (row_groups, rows)."""
    source = pq.ParquetFile(path)
    expected_rows = int(source.metadata.num_rows)
    expected_names = tuple(source.schema_arrow.names)
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    writer = None
    try:
        for row_group in range(source.num_row_groups):
            table = source.read_row_group(row_group)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    version="2.6",
                )
            writer.write_table(table)
        if writer is None:
            # Parquet files with zero rows still have a schema.
            writer = pq.ParquetWriter(
                temporary,
                source.schema_arrow,
                compression="zstd",
                version="2.6",
            )
        writer.close()
        writer = None

        repaired = pq.ParquetFile(temporary)
        actual_rows = int(repaired.metadata.num_rows)
        actual_names = tuple(repaired.schema_arrow.names)
        if actual_rows != expected_rows or actual_names != expected_names:
            raise RuntimeError(
                f"validation failed for {path}: rows {actual_rows}/{expected_rows}, "
                f"columns {actual_names!r}/{expected_names!r}"
            )
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        return source.num_row_groups, expected_rows
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for raw_path in args.paths:
        path = raw_path.resolve()
        groups, rows = rewrite_file(path)
        print(f"rewrote {path} ({rows:,} rows, {groups} row groups)")


if __name__ == "__main__":
    main()
