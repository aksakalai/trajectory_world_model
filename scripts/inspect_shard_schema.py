from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pyarrow.parquet as parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect authoritative shard schemas")
    parser.add_argument("shard", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.shard, mode="r") as archive:
        for name in ("episodes.parquet", "transitions.parquet", "frames.parquet"):
            extracted = archive.extractfile(name)
            if extracted is None:
                raise RuntimeError(f"Could not extract {name}")
            table = parquet.read_table(io.BytesIO(extracted.read()))
            print(f"{name}: rows={table.num_rows}", flush=True)
            print(table.schema, flush=True)
            if name == "episodes.parquet":
                print(f"split_values={table.column('split').unique().to_pylist()}")


if __name__ == "__main__":
    main()
