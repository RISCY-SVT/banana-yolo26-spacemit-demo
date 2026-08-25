#!/usr/bin/env python3
"""Measure board file-read, JPEG-decode, and letterbox/tensor components."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(640.0 / width, 640.0 / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_x = (640 - resized_width) / 2.0
    pad_y = (640 - resized_height) / 2.0
    left = round(pad_x - 0.1)
    right = round(pad_x + 0.1)
    top = round(pad_y - 0.1)
    bottom = round(pad_y + 0.1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if padded.shape[:2] != (640, 640):
        raise RuntimeError(f"letterbox shape mismatch: {padded.shape}")
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    options = parser.parse_args()
    if options.output.exists():
        raise RuntimeError(f"refusing existing output: {options.output}")
    images = sorted(options.images.glob("*.jpg"))[: options.limit]
    if len(images) != options.limit or options.repeats < 1:
        raise RuntimeError("insufficient images or invalid repeat count")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    tensor_hashes: dict[str, str] = {}
    for repeat in range(options.repeats):
        for image_path in images:
            begin = time.perf_counter_ns()
            payload = image_path.read_bytes()
            read_done = time.perf_counter_ns()
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            decode_done = time.perf_counter_ns()
            if image is None:
                raise RuntimeError(f"JPEG decode failed: {image_path}")
            tensor = preprocess(image)
            prepared = time.perf_counter_ns()
            tensor_sha = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
            previous = tensor_hashes.setdefault(image_path.name, tensor_sha)
            if previous != tensor_sha:
                raise RuntimeError(f"tensor changed across repeats: {image_path.name}")
            rows.append(
                {
                    "repeat": repeat,
                    "image": image_path.name,
                    "file_bytes": len(payload),
                    "file_read_us": (read_done - begin) / 1000.0,
                    "jpeg_decode_us": (decode_done - read_done) / 1000.0,
                    "letterbox_tensor_us": (prepared - decode_done) / 1000.0,
                    "preprocess_total_us": (prepared - begin) / 1000.0,
                    "tensor_sha256": tensor_sha,
                }
            )
    with options.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
