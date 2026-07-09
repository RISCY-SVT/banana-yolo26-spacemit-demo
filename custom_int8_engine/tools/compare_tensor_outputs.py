#!/usr/bin/env python3
"""Compare two tensor files for Stage40 reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage40_skeleton_common import compare_arrays, load_nhwc_bin, write_json


def load_tensor(path: Path, dtype: str | None, shape: str | None, layout: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if dtype is None or shape is None:
        raise ValueError("--dtype and --shape are required for raw binary inputs")
    shape_values = [int(v) for v in shape.replace(",", "x").split("x") if v]
    np_dtype = np.dtype(dtype)
    if layout == "nhwc_bin_from_nchw_shape":
        return load_nhwc_bin(path, shape_values, np_dtype)
    arr = np.fromfile(path, dtype=np_dtype)
    expected = int(np.prod(shape_values))
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} elements, expected {expected}")
    return arr.reshape(tuple(shape_values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs", required=True)
    parser.add_argument("--rhs", required=True)
    parser.add_argument("--lhs-dtype")
    parser.add_argument("--rhs-dtype")
    parser.add_argument("--lhs-shape")
    parser.add_argument("--rhs-shape")
    parser.add_argument("--lhs-layout", default="npy_or_raw")
    parser.add_argument("--rhs-layout", default="npy_or_raw")
    parser.add_argument("--name", default="tensor_compare")
    parser.add_argument("--json", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    lhs = load_tensor(Path(args.lhs), args.lhs_dtype, args.lhs_shape, args.lhs_layout)
    rhs = load_tensor(Path(args.rhs), args.rhs_dtype, args.rhs_shape, args.rhs_layout)
    summary = compare_arrays(args.name, lhs, rhs)
    write_json(Path(args.json), summary.__dict__)
    Path(args.report_md).write_text(
        "# Tensor Output Comparison\n\n"
        f"- name: `{summary.name}`\n"
        f"- status: `{summary.status}`\n"
        f"- lhs_shape: `{summary.lhs_shape}` dtype `{summary.lhs_dtype}` sha256 `{summary.lhs_sha256}`\n"
        f"- rhs_shape: `{summary.rhs_shape}` dtype `{summary.rhs_dtype}` sha256 `{summary.rhs_sha256}`\n"
        f"- total: `{summary.total}`\n"
        f"- mismatches: `{summary.mismatches}`\n"
        f"- max_abs_diff: `{summary.max_abs_diff}`\n",
        encoding="utf-8",
    )
    print(summary)
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
