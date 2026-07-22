#!/usr/bin/env python3
"""Normalize Stage60 serial and double-buffer benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256)


def parse(path: Path, resolution: int, kind: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] == "metadata" and len(fields) == 3:
            metadata[fields[1]] = fields[2]
            continue
        if fields[0] != "summary" or fields[1] == "phase":
            continue
        if kind == "serial":
            if len(fields) != 11:
                raise ValueError(f"invalid serial summary row in {path}: {line}")
            _, phase, mean, stddev, cv_pct, minimum, maximum, median, p90, p95, p99 = fields
            samples = 500
        else:
            if len(fields) != 10:
                raise ValueError(f"invalid double-buffer summary row in {path}: {line}")
            _, phase, samples, mean, stddev, median, p90, p95, p99, maximum = fields
            minimum = ""
            cv_pct = f"{100.0 * float(stddev) / float(mean):.9f}"
        rows.append({
            "resolution": resolution,
            "pipeline_kind": kind,
            "phase": phase,
            "samples": samples,
            "mean_us": mean,
            "stddev_us": stddev,
            "cv_pct": cv_pct,
            "minimum_us": minimum,
            "median_us": median,
            "p90_us": p90,
            "p95_us": p95,
            "p99_us": p99,
            "maximum_us": maximum,
            "fps_from_mean": f"{1_000_000.0 / float(mean):.9f}",
            "source_file": path.name,
        })
    if not rows:
        raise ValueError(f"no summary rows in {path}")
    if metadata.get("resolution") != str(resolution):
        raise ValueError(f"resolution metadata mismatch in {path}")
    if metadata.get("cpu4_7_ime_count") != "0":
        raise ValueError(f"IME ownership evidence missing in {path}")
    output_hash = metadata.get("output_hash", "")
    if re.fullmatch(r"0x[0-9a-f]+", output_hash) is None:
        raise ValueError(f"invalid output hash in {path}")
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for kind in ("serial", "double_buffer"):
            path = args.input / f"r{resolution}_{kind}.tsv"
            parsed, metadata = parse(path, resolution, kind)
            rows.extend(parsed)
            identities.append({
                "resolution": resolution,
                "pipeline_kind": kind,
                "output_hash": metadata["output_hash"],
                "package_manifest_sha256": metadata["package_manifest_sha256"],
                "detections": metadata["detections"],
                "cpu4_7_ime_count": metadata["cpu4_7_ime_count"],
                "preprocessor_cpus": metadata.get("preprocessor_cpus", "controller-thread"),
                "executor_cpus": metadata.get("executor_cpus", "0-4"),
                "steady_state_fps": metadata.get("steady_state_fps", ""),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    for name, materialized in (("resolution_pipeline_summary.tsv", rows),
                               ("resolution_pipeline_identity.tsv", identities)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(materialized[0]), delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(materialized)
    print(f"summary_rows={len(rows)}")
    print(f"identity_rows={len(identities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
