#!/usr/bin/env python3
"""Normalize the nine-profile Stage61 serial and double-buffer surfaces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from stage60_parse_pipeline import parse


RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256, 768)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for kind in ("serial", "double_buffer"):
            path = args.input / f"r{resolution}_{kind}.tsv"
            rows, metadata = parse(path, resolution, kind)
            summaries.extend(rows)
            identities.append(
                {
                    "resolution": resolution,
                    "pipeline_kind": kind,
                    "output_hash": metadata["output_hash"],
                    "package_manifest_sha256": metadata["package_manifest_sha256"],
                    "detections": metadata["detections"],
                    "cpu4_7_ime_count": metadata["cpu4_7_ime_count"],
                    "preprocessor_cpus": metadata.get("preprocessor_cpus", "controller-thread"),
                    "executor_cpus": metadata.get("executor_cpus", "0-4"),
                    "steady_state_fps": metadata.get("steady_state_fps", ""),
                    "source_file": path.name,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output / "resolution_pipeline_summary_v2.tsv", summaries)
    write_tsv(args.output / "resolution_pipeline_identity_v2.tsv", identities)
    print(f"summary_rows={len(summaries)}")
    print(f"identity_rows={len(identities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
