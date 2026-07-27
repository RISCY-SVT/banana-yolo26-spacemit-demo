#!/usr/bin/env python3
"""Summarize fixed-fixture output0 files using runtime-reported dtypes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np


TENSOR_RE = re.compile(
    r"stage46_tensor .*output_shape=(?P<shape>[0-9x-]+) "
    r"output_dtype=(?P<dtype>\d+) output_bytes=(?P<bytes>\d+) "
    r"output_fnv1a64=(?P<fnv>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--comparisons", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    options = parse_args()
    rows: list[dict[str, object]] = []
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for log in sorted((options.root / "raw").glob("*.log")):
        match = TENSOR_RE.search(log.read_text(encoding="utf-8", errors="replace"))
        if match is None:
            continue
        stem = log.stem
        fixture, arm = stem.split("__", 1)
        surface, provider = arm.rsplit("_", 1)
        output = options.root / "outputs" / f"{stem}.bin"
        dtype_id = int(match["dtype"])
        if dtype_id == 1:
            dtype = np.dtype("<f4")
            dtype_name = "float32"
        elif dtype_id == 10:
            dtype = np.dtype("<f2")
            dtype_name = "float16"
        else:
            raise ValueError(f"{log}: unsupported output dtype id {dtype_id}")
        values = np.fromfile(output, dtype=dtype).astype(np.float32)
        shape = tuple(int(item) for item in match["shape"].split("x"))
        expected = int(np.prod(shape))
        if len(values) != expected:
            raise ValueError(f"{output}: expected {expected} elements, got {len(values)}")
        tensor = values.reshape(shape)
        arrays[(fixture, surface, provider)] = tensor
        rows.append(
            {
                "fixture": fixture,
                "surface": surface,
                "provider": provider,
                "output_shape": match["shape"],
                "output_dtype": dtype_name,
                "output_bytes": match["bytes"],
                "output_sha256": sha256(output),
                "output_fnv1a64": match["fnv"],
                "nonfinite": int((~np.isfinite(tensor)).sum()),
                "score_ge_0_001": int((tensor[..., 4] >= 0.001).sum()),
                "class_noninteger": int(
                    (np.abs(tensor[..., 5] - np.rint(tensor[..., 5])) > 1e-6).sum()
                ),
                "class_out_of_range": int(
                    ((tensor[..., 5] < 0) | (tensor[..., 5] >= 80)).sum()
                ),
                "status": "pass",
            }
        )

    comparisons: list[dict[str, object]] = []
    for fixture, surface, provider in sorted(arrays):
        if provider != "spacemit":
            continue
        cpu_key = (fixture, surface, "cpu")
        if cpu_key not in arrays:
            continue
        ep = arrays[(fixture, surface, provider)]
        cpu = arrays[cpu_key]
        diff = np.abs(ep - cpu)
        comparisons.append(
            {
                "fixture": fixture,
                "surface": surface,
                "comparison": "rt206_spacemit_vs_rt206_cpu",
                "element_count": diff.size,
                "exact_element_count": int((diff == 0).sum()),
                "mean_abs_diff": f"{float(diff.mean()):.9f}",
                "max_abs_diff": f"{float(diff.max()):.9f}",
                "cpu_score_ge_0_001": int((cpu[..., 4] >= 0.001).sum()),
                "ep_score_ge_0_001": int((ep[..., 4] >= 0.001).sum()),
                "interpretation": "task-level comparison; byte identity is not required",
            }
        )

    options.results.parent.mkdir(parents=True, exist_ok=True)
    with options.results.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, delimiter="\t", fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    with options.comparisons.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            delimiter="\t",
            fieldnames=list(comparisons[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
