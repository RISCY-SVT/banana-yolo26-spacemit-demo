#!/usr/bin/env python3
"""Exact YOLO26 letterbox preprocessing and XSlim callback."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def letterbox_rgb_nchw(path: str | Path, size: int = 640) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    resized_width = int(np.rint(width * ratio))
    resized_height = int(np.rint(height * ratio))
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

    delta_width = size - resized_width
    delta_height = size - resized_height
    half_width = delta_width / 2.0
    half_height = delta_height / 2.0
    left = int(np.rint(half_width - 0.1))
    right = int(np.rint(half_width + 0.1))
    top = int(np.rint(half_height - 0.1))
    bottom = int(np.rint(half_height + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if image.shape != (size, size, 3):
        raise RuntimeError(f"letterbox shape mismatch: {image.shape}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(
        rgb.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    )


def vendor_literal_rgb_nchw(path: str | Path, size: int = 640) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(
        image.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    )


def preprocess_impl(
    file_names: Sequence[str], input_info: dict[str, Any]
) -> "torch.Tensor":
    import torch

    input_shape = input_info.get("input_shape") or [1, 3, 640, 640]
    size = int(input_shape[-1])
    tensors = [letterbox_rgb_nchw(path, size) for path in file_names]
    return torch.from_numpy(np.stack(tensors, axis=0))


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    paths = [
        Path(line.strip())
        for line in options.list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if options.limit > 0:
        paths = paths[: options.limit]
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8", newline="") as output:
        fields = [
            "path",
            "mode",
            "shape",
            "dtype",
            "minimum",
            "maximum",
            "sha256",
            "mismatch_count",
            "max_abs_difference",
        ]
        writer = csv.DictWriter(
            output, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for path in paths:
            exact = letterbox_rgb_nchw(path)
            literal = vendor_literal_rgb_nchw(path)
            difference = np.abs(exact - literal)
            for mode, array in (("project-exact", exact), ("vendor-literal", literal)):
                writer.writerow(
                    {
                        "path": str(path),
                        "mode": mode,
                        "shape": "x".join(map(str, array.shape)),
                        "dtype": str(array.dtype),
                        "minimum": f"{float(array.min()):.9g}",
                        "maximum": f"{float(array.max()):.9g}",
                        "sha256": digest(array),
                        "mismatch_count": int(np.count_nonzero(difference)),
                        "max_abs_difference": f"{float(difference.max()):.9g}",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
