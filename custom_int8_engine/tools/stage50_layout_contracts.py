#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from stage49_slice_package import write_tsv


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def encode(value: np.ndarray, layout: str) -> np.ndarray:
    _, channels, height, width = value.shape
    channel_block = 8 if layout in {"NCHWc8_SPATIAL_INNER_V1", "M4C8_KMAJOR_V1"} else 16
    channel_blocks = (channels + channel_block - 1) // channel_block
    spatial = height * width
    if layout == "M4C8_KMAJOR_V1":
        storage = np.zeros(((spatial + 3) // 4, channel_blocks, 4, 8), dtype=np.uint8)
        for channel in range(channels):
            for position in range(spatial):
                storage[position // 4, channel // 8, position % 4, channel % 8] = value.reshape(1, channels, spatial)[0, channel, position]
        return storage.reshape(-1)
    if layout == "NCHWc8_SPATIAL_INNER_V1":
        storage = np.zeros((channel_blocks, height, width, 8), dtype=np.uint8)
        for channel in range(channels):
            storage[channel // 8, :, :, channel % 8] = value[0, channel]
        return storage.reshape(-1)
    if layout == "NCHWc16_SPATIAL_INNER_V1":
        storage = np.zeros((channel_blocks, height, width, 16), dtype=np.uint8)
        for channel in range(channels):
            storage[channel // 16, :, :, channel % 16] = value[0, channel]
        return storage.reshape(-1)
    if layout == "NHWC16_SPATIAL_INNER_V1":
        storage = np.zeros((height, width, channel_blocks, 16), dtype=np.uint8)
        for channel in range(channels):
            storage[:, :, channel // 16, channel % 16] = value[0, channel]
        return storage.reshape(-1)
    raise ValueError(layout)


def decode(storage: np.ndarray, shape: tuple[int, int, int, int], layout: str) -> np.ndarray:
    _, channels, height, width = shape
    channel_block = 8 if layout in {"NCHWc8_SPATIAL_INNER_V1", "M4C8_KMAJOR_V1"} else 16
    channel_blocks = (channels + channel_block - 1) // channel_block
    spatial = height * width
    output = np.empty(shape, dtype=np.uint8)
    if layout == "M4C8_KMAJOR_V1":
        view = storage.reshape((spatial + 3) // 4, channel_blocks, 4, 8)
        flat = output.reshape(1, channels, spatial)
        for channel in range(channels):
            for position in range(spatial):
                flat[0, channel, position] = view[position // 4, channel // 8, position % 4, channel % 8]
        return output
    if layout == "NCHWc8_SPATIAL_INNER_V1":
        view = storage.reshape(channel_blocks, height, width, 8)
        for channel in range(channels):
            output[0, channel] = view[channel // 8, :, :, channel % 8]
        return output
    if layout == "NCHWc16_SPATIAL_INNER_V1":
        view = storage.reshape(channel_blocks, height, width, 16)
        for channel in range(channels):
            output[0, channel] = view[channel // 16, :, :, channel % 16]
        return output
    if layout == "NHWC16_SPATIAL_INNER_V1":
        view = storage.reshape(height, width, channel_blocks, 16)
        for channel in range(channels):
            output[0, channel] = view[:, :, channel // 16, channel % 16]
        return output
    raise ValueError(layout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tensors = read_tsv(args.package / "tensors.tsv")
    layouts = [
        "NCHWc8_SPATIAL_INNER_V1",
        "M4C8_KMAJOR_V1",
        "NCHWc16_SPATIAL_INNER_V1",
        "NHWC16_SPATIAL_INNER_V1",
    ]
    rows: list[dict[str, object]] = []
    for tensor in tensors:
        tensor_id = int(tensor["id"])
        shape = (1, int(tensor["c"]), int(tensor["h"]), int(tensor["w"]))
        logical = np.fromfile(args.package / "oracles" / "F0" / f"tensor_{tensor_id:03d}_nchw_u8.bin", dtype=np.uint8).reshape(shape)
        for layout in layouts:
            physical = encode(logical, layout)
            reconstructed = decode(physical, shape, layout)
            rows.append({
                "tensor_id": tensor_id,
                "tensor_key": tensor["key"],
                "shape": "x".join(str(item) for item in shape),
                "layout": layout,
                "logical_bytes": logical.size,
                "physical_bytes": physical.size,
                "padding_bytes": physical.size - logical.size,
                "roundtrip_exact": int(np.array_equal(logical, reconstructed)),
                "physical_sha256": sha256_bytes(physical),
                "persistent_conv_pair_implemented": int(layout == "NCHWc8_SPATIAL_INNER_V1"),
            })
    write_tsv(args.output, rows)
    print(f"rows={len(rows)}")
    print(f"exact_rows={sum(int(row['roundtrip_exact']) for row in rows)}")
    print("selected_layout=NCHWc8_SPATIAL_INNER_V1")
    print("alternative_pair_status=contract-proven-consumer-kernels-not-implemented")
    return 0 if all(int(row["roundtrip_exact"]) for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
