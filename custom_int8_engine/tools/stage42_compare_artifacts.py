#!/usr/bin/env python3
"""Offline Stage42 host/board tensor comparison and output0 diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part) for part in value.split("x"))
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid concrete shape: {value}")
    return shape


def dtype_for(value: str) -> np.dtype:
    mapping = {"UINT8": np.dtype(np.uint8), "FLOAT": np.dtype(np.float32)}
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {value}") from exc


def load_raw(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {expected_bytes}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def compare(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, str]:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise ValueError("shape/dtype mismatch")
    lhs64 = lhs.astype(np.float64, copy=False)
    rhs64 = rhs.astype(np.float64, copy=False)
    equal = lhs == rhs
    if np.issubdtype(lhs.dtype, np.floating):
        equal |= np.isnan(lhs) & np.isnan(rhs)
        equal |= np.isposinf(lhs) & np.isposinf(rhs)
        equal |= np.isneginf(lhs) & np.isneginf(rhs)
    flat_equal = equal.reshape(-1)
    mismatch = np.flatnonzero(~flat_equal)
    finite = np.isfinite(lhs64) & np.isfinite(rhs64)
    diff = np.zeros(lhs.size, dtype=np.float64)
    flat_finite = finite.reshape(-1)
    diff[flat_finite] = (lhs64 - rhs64).reshape(-1)[flat_finite]
    abs_diff = np.abs(diff)
    percentiles = np.percentile(abs_diff, [50, 90, 95, 99, 99.9])
    histogram = "not-applicable"
    if np.issubdtype(lhs.dtype, np.integer):
        values, counts = np.unique(diff.astype(np.int64), return_counts=True)
        histogram = ",".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))
    return {
        "element_count": str(lhs.size),
        "mismatch_count": str(mismatch.size),
        "mismatch_ratio": f"{mismatch.size / lhs.size:.12f}",
        "max_abs_diff": f"{abs_diff.max(initial=0.0):.12f}",
        "mean_abs_diff": f"{abs_diff.mean():.12f}",
        "rmse": f"{np.sqrt(np.mean(diff * diff)):.12f}",
        "first_mismatch_index": "none" if mismatch.size == 0 else str(int(mismatch[0])),
        "p50_abs_diff": f"{percentiles[0]:.12f}",
        "p90_abs_diff": f"{percentiles[1]:.12f}",
        "p95_abs_diff": f"{percentiles[2]:.12f}",
        "p99_abs_diff": f"{percentiles[3]:.12f}",
        "p999_abs_diff": f"{percentiles[4]:.12f}",
        "signed_difference_histogram": histogram,
        "lhs_min": f"{np.nanmin(lhs64):.12f}",
        "lhs_max": f"{np.nanmax(lhs64):.12f}",
        "lhs_mean": f"{np.nanmean(lhs64):.12f}",
        "lhs_sum": f"{np.nansum(lhs64):.12f}",
        "lhs_nonfinite_count": str(int(np.count_nonzero(~np.isfinite(lhs64)))),
        "rhs_min": f"{np.nanmin(rhs64):.12f}",
        "rhs_max": f"{np.nanmax(rhs64):.12f}",
        "rhs_mean": f"{np.nanmean(rhs64):.12f}",
        "rhs_sum": f"{np.nansum(rhs64):.12f}",
        "rhs_nonfinite_count": str(int(np.count_nonzero(~np.isfinite(rhs64)))),
    }


def boundary_matrix(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    board_dir = Path(args.board_dir)
    output_path = Path(args.output)
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    output_rows: list[dict[str, str]] = []
    for row in rows:
        index = int(row["index"])
        dtype = dtype_for(row["dtype"])
        shape = parse_shape(row["shape"])
        host_path = Path(row["output_raw"])
        board_path = board_dir / f"board_{index:02d}.bin"
        host = load_raw(host_path, dtype, shape)
        board = load_raw(board_path, dtype, shape)
        result = compare(host, board)
        output_rows.append(
            {
                "index": str(index),
                "tensor_name": row["tensor_name"],
                "producer": row["producer"],
                "consumers": row["consumers"],
                "dtype": row["dtype"],
                "shape": row["shape"],
                "layout": row["layout"],
                "scale": row["scale"],
                "zero_point": row["zero_point"],
                "host_path": str(host_path),
                "host_sha256": sha256(host_path),
                "board_path": str(board_path),
                "board_sha256": sha256(board_path),
                **result,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(output_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)


def box_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    x1 = max(float(lhs[0]), float(rhs[0]))
    y1 = max(float(lhs[1]), float(rhs[1]))
    x2 = min(float(lhs[2]), float(rhs[2]))
    y2 = min(float(lhs[3]), float(rhs[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    lhs_area = max(0.0, float(lhs[2] - lhs[0])) * max(0.0, float(lhs[3] - lhs[1]))
    rhs_area = max(0.0, float(rhs[2] - rhs[0])) * max(0.0, float(rhs[3] - rhs[1]))
    union = lhs_area + rhs_area - intersection
    return intersection / union if union > 0.0 else 0.0


def multiset_overlap(lhs: np.ndarray, rhs: np.ndarray, count: int) -> tuple[int, float]:
    lhs_classes = Counter(int(value) for value in lhs[:count, 5])
    rhs_classes = Counter(int(value) for value in rhs[:count, 5])
    overlap = sum((lhs_classes & rhs_classes).values())
    return overlap, overlap / count


def output0_diagnostic(args: argparse.Namespace) -> None:
    host_path = Path(args.host)
    board_path = Path(args.board)
    host = load_raw(host_path, np.dtype(np.float32), (300, 6))
    board = load_raw(board_path, np.dtype(np.float32), (300, 6))
    exact_rows = int(np.count_nonzero(np.all(host == board, axis=1)))
    same_class_by_row = int(np.count_nonzero(host[:, 5] == board[:, 5]))
    class_host = Counter(int(value) for value in host[:, 5])
    class_board = Counter(int(value) for value in board[:, 5])
    class_l1 = sum(abs(class_host[key] - class_board[key]) for key in set(class_host) | set(class_board))
    lines = [
        "# Cross-runtime output0 diagnostic",
        "",
        "This compares one deterministic input across different ORT runtimes. It is not COCO/mAP or model accuracy.",
        "",
        f"- host_raw: `{host_path}`",
        f"- host_sha256: `{sha256(host_path)}`",
        f"- board_raw: `{board_path}`",
        f"- board_sha256: `{sha256(board_path)}`",
        f"- exact_rows_same_index: {exact_rows}/300",
        f"- same_class_same_index: {same_class_by_row}/300",
        f"- class_count_l1_distance: {class_l1}",
        "",
        "## Score distribution",
        "",
        "| runtime | min | p50 | p90 | p95 | p99 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, tensor in (("host", host), ("board", board)):
        scores = tensor[:, 4]
        q = np.percentile(scores, [50, 90, 95, 99])
        lines.append(
            f"| {name} | {scores.min():.9f} | {q[0]:.9f} | {q[1]:.9f} | "
            f"{q[2]:.9f} | {q[3]:.9f} | {scores.max():.9f} | {scores.mean():.9f} |"
        )
    lines.extend(["", "## Top-k class multiset overlap", "", "| k | overlap | ratio |", "|---:|---:|---:|"])
    for count in (10, 50, 100, 300):
        overlap, ratio = multiset_overlap(host, board, count)
        lines.append(f"| {count} | {overlap} | {ratio:.6f} |")

    board_unused = set(range(300))
    matches: list[tuple[int, int, float]] = []
    for host_index in np.argsort(-host[:, 4]):
        same_class = [index for index in board_unused if board[index, 5] == host[host_index, 5]]
        if not same_class:
            continue
        board_index = max(same_class, key=lambda index: box_iou(host[host_index], board[index]))
        iou = box_iou(host[host_index], board[board_index])
        if iou >= args.iou_threshold:
            board_unused.remove(board_index)
            matches.append((int(host_index), int(board_index), iou))
    if matches:
        coordinate_diff = np.concatenate(
            [np.abs(host[host_index, :4] - board[board_index, :4]) for host_index, board_index, _ in matches]
        )
        score_diff = np.asarray(
            [abs(float(host[host_index, 4] - board[board_index, 4])) for host_index, board_index, _ in matches]
        )
        ious = np.asarray([iou for _, _, iou in matches])
        lines.extend(
            [
                "",
                f"## Greedy class+IoU matching (IoU >= {args.iou_threshold:.2f})",
                "",
                f"- matched_rows: {len(matches)}/300",
                f"- mean_iou: {ious.mean():.9f}",
                f"- p50_iou: {np.percentile(ious, 50):.9f}",
                f"- mean_abs_coordinate_diff: {coordinate_diff.mean():.9f}",
                f"- max_abs_coordinate_diff: {coordinate_diff.max():.9f}",
                f"- mean_abs_score_diff: {score_diff.mean():.9f}",
                f"- max_abs_score_diff: {score_diff.max():.9f}",
            ]
        )
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def pairwise_matrix(args: argparse.Namespace) -> None:
    shape = parse_shape(args.shape)
    dtype = dtype_for(args.dtype)
    arms: list[tuple[str, Path, np.ndarray]] = []
    for arm in args.arm:
        if "=" not in arm:
            raise ValueError(f"arm must be NAME=PATH: {arm}")
        name, path_text = arm.split("=", 1)
        path = Path(path_text)
        arms.append((name, path, load_raw(path, dtype, shape)))
    rows: list[dict[str, str]] = []
    for lhs_index, (lhs_name, lhs_path, lhs) in enumerate(arms):
        for rhs_name, rhs_path, rhs in arms[lhs_index + 1 :]:
            rows.append(
                {
                    "lhs": lhs_name,
                    "rhs": rhs_name,
                    "lhs_path": str(lhs_path),
                    "rhs_path": str(rhs_path),
                    "lhs_sha256": sha256(lhs_path),
                    "rhs_sha256": sha256(rhs_path),
                    **compare(lhs, rhs),
                }
            )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    boundaries = subparsers.add_parser("boundaries")
    boundaries.add_argument("--manifest", required=True)
    boundaries.add_argument("--board-dir", required=True)
    boundaries.add_argument("--output", required=True)
    boundaries.set_defaults(func=boundary_matrix)
    output0 = subparsers.add_parser("output0")
    output0.add_argument("--host", required=True)
    output0.add_argument("--board", required=True)
    output0.add_argument("--output", required=True)
    output0.add_argument("--iou-threshold", type=float, default=0.5)
    output0.set_defaults(func=output0_diagnostic)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--dtype", choices=("UINT8", "FLOAT"), required=True)
    matrix.add_argument("--shape", required=True)
    matrix.add_argument("--arm", action="append", required=True)
    matrix.add_argument("--output", required=True)
    matrix.set_defaults(func=pairwise_matrix)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
