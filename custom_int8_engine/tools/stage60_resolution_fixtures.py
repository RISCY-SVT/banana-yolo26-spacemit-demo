#!/usr/bin/env python3
"""Generate deterministic image and adversarial fixtures for Stage60 profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_image(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("images must use NAME=PATH")
    return name, Path(path)


def letterbox(image: np.ndarray, resolution: int) -> tuple[np.ndarray, dict[str, int | float]]:
    height, width = image.shape[:2]
    ratio = min(resolution / width, resolution / height)
    resized_width = int(np.rint(width * ratio))
    resized_height = int(np.rint(height * ratio))
    pad_x = int(np.rint((resolution - resized_width) / 2.0 - 0.1))
    pad_y = int(np.rint((resolution - resized_height) / 2.0 - 0.1))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((resolution, resolution, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, {
        "source_width": width,
        "source_height": height,
        "ratio": ratio,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def adversarial_fixtures(f0: np.ndarray, resolution: int) -> list[tuple[str, str, np.ndarray]]:
    shape = (3, resolution, resolution)
    y, x = np.indices((resolution, resolution), dtype=np.int32)
    x_ramp = np.broadcast_to(np.linspace(0.0, 1.0, resolution, dtype=np.float32), shape)
    y_ramp = np.broadcast_to(
        np.linspace(0.0, 1.0, resolution, dtype=np.float32)[None, :, None], shape
    )
    channel_scale = np.asarray([0.25, 0.5, 1.0], dtype=np.float32)[:, None, None]
    rng = np.random.default_rng(52001)
    return [
        ("F0", "real_bus_preprocessed", f0),
        ("F1", "all_zero", np.zeros(shape, dtype=np.float32)),
        ("F2", "all_one", np.ones(shape, dtype=np.float32)),
        ("F3", "letterbox_padding_114_over_255",
         np.full(shape, 114.0 / 255.0, dtype=np.float32)),
        ("F4", "checkerboard_zero_one",
         np.broadcast_to(((x + y) & 1).astype(np.float32), shape)),
        ("F5", "horizontal_ramp", x_ramp),
        ("F6", "channel_scaled_vertical_ramp", y_ramp * channel_scale),
        ("F7", "uniform_seed_52001", rng.random(shape, dtype=np.float32)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolutions", default="640,512,448,416,384,352,320,256")
    parser.add_argument("--image", action="append", type=parse_image, required=True)
    args = parser.parse_args()
    resolutions = [int(value) for value in args.resolutions.split(",")]
    if any(value <= 0 or value % 32 for value in resolutions):
        raise ValueError("resolutions must be positive multiples of 32")

    source_images: dict[str, tuple[Path, np.ndarray]] = {}
    for name, path in args.image:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read image: {path}")
        source_images[name] = (path.resolve(), image)
    if "bus" not in source_images:
        raise ValueError("a bus=PATH image is required for F0")

    rows: list[dict[str, object]] = []
    for resolution in resolutions:
        output = args.output_root / f"r{resolution}"
        output.mkdir(parents=True, exist_ok=True)
        prepared: dict[str, np.ndarray] = {}
        for name, (source, bgr) in source_images.items():
            canvas, geometry = letterbox(bgr, resolution)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            f32 = np.ascontiguousarray(
                rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0), dtype="<f4"
            )
            rgb_path = output / f"{name}_r{resolution}_rgb_u8.bin"
            f32_path = output / f"{name}_r{resolution}_nchw_f32.bin"
            rgb.tofile(rgb_path)
            f32.tofile(f32_path)
            prepared[name] = f32
            for surface, path in (("rgb-u8", rgb_path), ("nchw-f32", f32_path)):
                rows.append({
                    "resolution": resolution,
                    "fixture": name,
                    "description": "real-image-letterbox",
                    "surface": surface,
                    "source": str(source),
                    **geometry,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "path": str(path),
                })

        for fixture, description, values in adversarial_fixtures(prepared["bus"], resolution):
            path = output / f"{fixture}_nchw_f32.bin"
            np.asarray(values, dtype="<f4").tofile(path)
            rows.append({
                "resolution": resolution,
                "fixture": fixture,
                "description": description,
                "surface": "nchw-f32",
                "source": str(source_images["bus"][0]),
                "source_width": source_images["bus"][1].shape[1],
                "source_height": source_images["bus"][1].shape[0],
                "ratio": "",
                "resized_width": "",
                "resized_height": "",
                "pad_x": "",
                "pad_y": "",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "path": str(path),
            })

    manifest = args.output_root / "stage60_fixture_manifest.tsv"
    fields = list(rows[0])
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"resolutions={len(resolutions)}")
    print(f"fixtures={len(rows)}")
    print(f"manifest_sha256={sha256(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
