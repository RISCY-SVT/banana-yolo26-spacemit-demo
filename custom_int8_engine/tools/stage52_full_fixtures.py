#!/usr/bin/env python3
"""Generate deterministic full-input fixtures for Stage 52 boundary parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shape = (3, 640, 640)
    f0 = np.fromfile(args.f0, dtype="<f4")
    if f0.size != np.prod(shape):
        raise ValueError("F0 must contain 1x3x640x640 float32 values")
    f0 = f0.reshape(shape)
    y, x = np.indices((640, 640), dtype=np.int32)
    x_ramp = np.broadcast_to(np.linspace(0.0, 1.0, 640, dtype=np.float32), shape)
    y_ramp = np.broadcast_to(
        np.linspace(0.0, 1.0, 640, dtype=np.float32)[None, :, None], shape)
    channel_scale = np.asarray([0.25, 0.5, 1.0], dtype=np.float32)[:, None, None]
    rng = np.random.default_rng(52001)
    fixtures = [
        ("F0", "real_bus_preprocessed", f0),
        ("F1", "all_zero", np.zeros(shape, dtype=np.float32)),
        ("F2", "all_one", np.ones(shape, dtype=np.float32)),
        ("F3", "letterbox_padding_114_over_255", np.full(shape, 114.0 / 255.0, dtype=np.float32)),
        ("F4", "checkerboard_zero_one", np.broadcast_to(((x + y) & 1).astype(np.float32), shape)),
        ("F5", "horizontal_ramp", x_ramp),
        ("F6", "channel_scaled_vertical_ramp", y_ramp * channel_scale),
        ("F7", "uniform_seed_52001", rng.random(shape, dtype=np.float32)),
    ]
    rows = []
    for fixture_id, description, values in fixtures:
        path = args.output / f"{fixture_id}_nchw_f32.bin"
        np.asarray(values, dtype="<f4").tofile(path)
        rows.append({
            "fixture": fixture_id,
            "description": description,
            "shape": "1x3x640x640",
            "dtype": "float32-little-endian",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "path": str(path),
        })
    with (args.output / "fixture_manifest.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"fixtures={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
