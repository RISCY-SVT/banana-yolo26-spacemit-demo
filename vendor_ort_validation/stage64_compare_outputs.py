#!/usr/bin/env python3
"""Compare Stage64 two-stage runner outputs and split-boundary tensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


BOUNDARY_SHAPES = [
    (1, 4, 80, 80),
    (1, 80, 80, 80),
    (1, 4, 40, 40),
    (1, 80, 40, 40),
    (1, 4, 20, 20),
    (1, 80, 20, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface",
        action="append",
        required=True,
        help="NAME=runner-output-directory",
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--comparisons", required=True, type=Path)
    return parser.parse_args()


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def load_float32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, got {values.size}")
    return values.reshape(shape)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            delimiter="\t",
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    options = parse_args()
    roots: dict[str, Path] = {}
    for value in options.surface:
        name, separator, path = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --surface value: {value}")
        roots[name] = Path(path)
    if len(roots) < 2:
        raise ValueError("at least two surfaces are required")

    arrays: dict[tuple[str, str], np.ndarray] = {}
    result_rows: list[dict[str, object]] = []
    for surface, root in roots.items():
        tensor_files = [("output0", root / "output.bin", (1, 300, 6))]
        tensor_files.extend(
            (
                f"boundary-{index}",
                root / "boundaries" / f"boundary-{index}.bin",
                shape,
            )
            for index, shape in enumerate(BOUNDARY_SHAPES)
        )
        for tensor_name, path, shape in tensor_files:
            tensor = load_float32(path, shape)
            arrays[(surface, tensor_name)] = tensor
            result_rows.append(
                {
                    "surface": surface,
                    "tensor": tensor_name,
                    "shape": "x".join(map(str, shape)),
                    "dtype": "float32",
                    "sha256": digest(tensor),
                    "nonfinite_count": int((~np.isfinite(tensor)).sum()),
                    "minimum": f"{float(np.nanmin(tensor)):.9g}",
                    "maximum": f"{float(np.nanmax(tensor)):.9g}",
                    "mean": f"{float(np.nanmean(tensor)):.9g}",
                    "stddev": f"{float(np.nanstd(tensor)):.9g}",
                    "score_ge_0_25": (
                        int(np.count_nonzero(tensor[..., 4] >= 0.25))
                        if tensor_name == "output0"
                        else ""
                    ),
                }
            )

    comparison_rows: list[dict[str, object]] = []
    reference = next(iter(roots))
    for candidate in list(roots)[1:]:
        for tensor_name in ["output0", *[f"boundary-{i}" for i in range(6)]]:
            left = arrays[(reference, tensor_name)].astype(np.float64).reshape(-1)
            right = arrays[(candidate, tensor_name)].astype(np.float64).reshape(-1)
            difference = np.abs(left - right)
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            cosine = (
                float(np.dot(left, right) / denominator)
                if denominator
                else float("nan")
            )
            comparison_rows.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "tensor": tensor_name,
                    "element_count": left.size,
                    "exact_element_count": int(np.count_nonzero(difference == 0)),
                    "mae": f"{float(difference.mean()):.9g}",
                    "max_abs_difference": f"{float(difference.max()):.9g}",
                    "cosine_similarity": f"{cosine:.12g}",
                }
            )

    write_tsv(options.results, result_rows)
    write_tsv(options.comparisons, comparison_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
