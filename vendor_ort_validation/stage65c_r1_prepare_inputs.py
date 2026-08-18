#!/usr/bin/env python3
"""Freeze exact project-preprocessed inputs for selected R1 boundary cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from stage64_preprocess import letterbox_rgb_nchw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {options.output_dir}")
    inputs = options.output_dir / "inputs"
    inputs.mkdir(parents=True)
    rows = list(csv.DictReader(options.selection.open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise ValueError("selection is empty")
    manifest = []
    seen: set[int] = set()
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate selected image: {image_id}")
        seen.add(image_id)
        source = Path(row["image_path"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: {source}")
        height, width = image.shape[:2]
        scale = min(np.float32(640) / np.float32(width), np.float32(640) / np.float32(height))
        resized_width = int(np.rint(float(width) * float(scale)))
        resized_height = int(np.rint(float(height) * float(scale)))
        tensor = letterbox_rgb_nchw(source)[None, ...]
        destination = inputs / f"{image_id:012d}.nchw-f32.bin"
        destination.write_bytes(np.ascontiguousarray(tensor).tobytes(order="C"))
        manifest.append(
            {
                "selection_group": row["selection_group"],
                "rank": row["rank"],
                "image_id": image_id,
                "source_path": source,
                "source_sha256": sha256(source),
                "input_path": destination,
                "input_sha256": sha256(destination),
                "shape": "1x3x640x640",
                "dtype": "float32",
                "source_width": width,
                "source_height": height,
                "scale": float(scale),
                "pad_x": float(np.float32(640 - resized_width) / np.float32(2)),
                "pad_y": float(np.float32(640 - resized_height) / np.float32(2)),
            }
        )
    fields = list(manifest[0])
    with (options.output_dir / "input_manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
