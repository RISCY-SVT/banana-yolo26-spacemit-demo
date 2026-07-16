#!/usr/bin/env python3
"""Merge selected Stage58 camera detections into the operating-envelope record."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def size_bin(value: float) -> str:
    for name, upper in (
        ("0-8", 8.0), ("8-12", 12.0), ("12-16", 16.0),
        ("16-24", 24.0), ("24-32", 32.0), ("32-48", 48.0),
        ("48-64", 64.0), ("64-96", 96.0),
    ):
        if value < upper:
            return name
    return ">=96"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="release-c5-r*-detections.tsv")
    args = parser.parse_args()

    sources = sorted(args.detections_dir.glob(args.pattern))
    if len(sources) != 3:
        raise ValueError(f"expected three selected-run files, found {len(sources)}")

    rows: list[dict[str, str]] = []
    input_fields: list[str] = []
    for source in sources:
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"missing header: {source}")
            if not input_fields:
                input_fields = reader.fieldnames
            elif input_fields != reader.fieldnames:
                raise ValueError(f"incompatible detection schema: {source}")
            for row in reader:
                if row["measured"] != "1":
                    continue
                shorter_side = min(float(row["letterbox_width"]),
                                   float(row["letterbox_height"]))
                rows.append({
                    "source_run": source.stem.removesuffix("-detections"),
                    "scene_id": "fixed-wall-poster",
                    "camera_mode": "1280x720@60_MJPG",
                    "illumination_note": "stable-indoor-board-session",
                    "distance_m": "",
                    "distance_status": "not-measured",
                    "letterbox_shorter_side_bin": size_bin(shorter_side),
                    **row,
                })
    if not rows:
        raise ValueError("no measured detections")

    fields = [
        "source_run", "scene_id", "camera_mode", "illumination_note",
        "distance_m", "distance_status", "letterbox_shorter_side_bin",
        *input_fields,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"sources={len(sources)} observations={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
