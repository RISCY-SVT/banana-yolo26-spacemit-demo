#!/usr/bin/env python3
"""Replay the common float tail and quantify rank/threshold discontinuities."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

SURFACES = ("B2-cpu", "B2-spacemit", "A1-cpu", "A1-spacemit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def compare(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    delta = left.astype(np.float64) - right.astype(np.float64)
    denominator = float(np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel()))
    cosine = float(np.dot(left.ravel(), right.ravel()) / denominator) if denominator else 1.0
    return float(np.max(np.abs(delta))), float(np.mean(np.abs(delta))), cosine


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    union = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1) + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1) - intersection
    return intersection / union if union > 0 else 0.0


def greedy_matches(left: np.ndarray, right: np.ndarray, limit: int) -> list[tuple[int, int]]:
    left_order = np.argsort(-left[:, 4], kind="stable")[:limit]
    right_order = np.argsort(-right[:, 4], kind="stable")[:limit]
    candidates = []
    for left_rank, left_index in enumerate(left_order):
        left_class = int(np.rint(left[left_index, 5]))
        for right_rank, right_index in enumerate(right_order):
            if left_class != int(np.rint(right[right_index, 5])):
                continue
            overlap = bbox_iou(left[left_index, :4], right[right_index, :4])
            if overlap >= 0.5:
                candidates.append((-overlap, left_rank, right_rank, int(left_index), int(right_index)))
    matches = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _negative_iou, _left_rank, _right_rank, left_index, right_index in sorted(candidates):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--boundary-root", required=True, type=Path)
    parser.add_argument("--tail", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {options.output_dir}")
    host_outputs = options.output_dir / "host-outputs"
    host_outputs.mkdir(parents=True)
    selected = list(csv.DictReader(options.selection.open(encoding="utf-8"), delimiter="\t"))

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    tail = ort.InferenceSession(str(options.tail), sess_options=session_options, providers=["CPUExecutionProvider"])
    inputs = tail.get_inputs()
    if len(inputs) != 6:
        raise ValueError("tail does not expose six inputs")

    replay_rows = []
    outputs: dict[tuple[int, str], np.ndarray] = {}
    host_outputs_by_surface: dict[tuple[int, str], np.ndarray] = {}
    boundary_sets: dict[tuple[int, str], list[np.ndarray]] = {}
    for case in selected:
        image_id = int(case["image_id"])
        for surface in SURFACES:
            directory = options.boundary_root / str(image_id) / surface
            boundaries = []
            boundary_hashes = []
            for index, item in enumerate(inputs):
                path = directory / "boundaries" / f"boundary-{index}.bin"
                array = np.fromfile(path, dtype=np.float32).reshape(item.shape)
                boundaries.append(array)
                boundary_hashes.append(sha256(path))
            boundary_sets[(image_id, surface)] = boundaries
            feed = {item.name: value for item, value in zip(inputs, boundaries)}
            first = tail.run(None, feed)[0]
            second = tail.run(None, feed)[0]
            if not np.array_equal(first, second):
                raise RuntimeError(f"host tail replay is nondeterministic: {image_id} {surface}")
            if first.shape != (1, 300, 6) or not np.isfinite(first).all():
                raise RuntimeError(f"invalid host tail output: {image_id} {surface}")
            destination = host_outputs / str(image_id) / f"{surface}.bin"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(np.ascontiguousarray(first).tobytes(order="C"))
            board_original_path = directory / "output.bin"
            board_replay_path = directory / "tail-replay.bin"
            board_original = np.fromfile(board_original_path, dtype=np.float32).reshape(1, 300, 6)
            board_replay = np.fromfile(board_replay_path, dtype=np.float32).reshape(1, 300, 6)
            original_replay = compare(board_original, board_replay)
            board_host = compare(board_original, first)
            if not np.array_equal(board_original, board_replay):
                raise RuntimeError(f"board tail replay contract failed: {image_id} {surface}")
            outputs[(image_id, surface)] = board_original[0]
            host_outputs_by_surface[(image_id, surface)] = first[0]
            replay_rows.append(
                {
                    "selection_group": case["selection_group"], "image_id": image_id, "surface": surface,
                    "boundary_manifest_sha256": sha256_bytes("".join(boundary_hashes).encode()),
                    "board_original_sha256": sha256(board_original_path),
                    "board_replay_sha256": sha256(board_replay_path),
                    "host_replay_sha256": sha256(destination),
                    "board_original_vs_replay_max_abs": original_replay[0],
                    "board_original_vs_replay_mean_abs": original_replay[1],
                    "board_original_vs_replay_cosine": original_replay[2],
                    "board_original_vs_host_max_abs": board_host[0],
                    "board_original_vs_host_mean_abs": board_host[1],
                    "board_original_vs_host_cosine": board_host[2],
                    "host_repeat_byte_identical": "yes",
                    "status": "pass",
                }
            )
    write_tsv(options.output_dir / "tail_replay_results.tsv", replay_rows)

    topk_rows = []
    crossing_rows = []
    for case in selected:
        image_id = int(case["image_id"])
        for model in ("B2", "A1"):
            cpu = outputs[(image_id, f"{model}-cpu")]
            ep = outputs[(image_id, f"{model}-spacemit")]
            for limit in (10, 50, 100, 300):
                matches = greedy_matches(cpu, ep, limit)
                cpu_order = np.argsort(-cpu[:, 4], kind="stable")[:limit]
                ep_order = np.argsort(-ep[:, 4], kind="stable")[:limit]
                cpu_rank = {int(value): rank for rank, value in enumerate(cpu_order)}
                ep_rank = {int(value): rank for rank, value in enumerate(ep_order)}
                displacement = [abs(cpu_rank[left] - ep_rank[right]) for left, right in matches]
                topk_rows.append(
                    {
                        "selection_group": case["selection_group"], "image_id": image_id, "model": model,
                        "top_k": limit, "matched_members": len(matches),
                        "cpu_only_members": limit - len(matches), "ep_only_members": limit - len(matches),
                        "mean_rank_displacement": float(np.mean(displacement)) if displacement else "NA",
                        "maximum_rank_displacement": int(max(displacement)) if displacement else "NA",
                    }
                )
            matches = greedy_matches(cpu, ep, 300)
            for threshold in (0.001, 0.01, 0.05, 0.25, 0.5):
                cpu_to_ep = sum(cpu[left, 4] >= threshold and ep[right, 4] < threshold for left, right in matches)
                ep_to_cpu = sum(ep[right, 4] >= threshold and cpu[left, 4] < threshold for left, right in matches)
                crossing_rows.append(
                    {
                        "selection_group": case["selection_group"], "image_id": image_id, "model": model,
                        "score_threshold": threshold,
                        "cpu_above_ep_below": cpu_to_ep, "ep_above_cpu_below": ep_to_cpu,
                        "cpu_above_count": int(np.count_nonzero(cpu[:, 4] >= threshold)),
                        "ep_above_count": int(np.count_nonzero(ep[:, 4] >= threshold)),
                    }
                )
    write_tsv(options.output_dir / "tail_topk_membership_delta.tsv", topk_rows)
    write_tsv(options.output_dir / "tail_threshold_crossings.tsv", crossing_rows)

    splice_rows = []
    for case in selected:
        image_id = int(case["image_id"])
        for model in ("B2", "A1"):
            cpu_key = (image_id, f"{model}-cpu")
            ep_key = (image_id, f"{model}-spacemit")
            cpu_output = host_outputs_by_surface[cpu_key]
            ep_output = host_outputs_by_surface[ep_key]
            cpu_ep_mean_abs = compare(cpu_output, ep_output)[1]
            for boundary_index, boundary_input in enumerate(inputs):
                hybrid_boundaries = list(boundary_sets[ep_key])
                hybrid_boundaries[boundary_index] = boundary_sets[cpu_key][boundary_index]
                feed = {item.name: value for item, value in zip(inputs, hybrid_boundaries)}
                hybrid = tail.run(None, feed)[0]
                repeated = tail.run(None, feed)[0]
                if not np.array_equal(hybrid, repeated):
                    raise RuntimeError(
                        f"hybrid tail replay is nondeterministic: {image_id} {model} {boundary_index}"
                    )
                hybrid = hybrid[0]
                hybrid_ep = compare(hybrid, ep_output)
                hybrid_cpu = compare(hybrid, cpu_output)
                recovery = (
                    (cpu_ep_mean_abs - hybrid_cpu[1]) / cpu_ep_mean_abs
                    if cpu_ep_mean_abs > 0
                    else 0.0
                )
                hybrid_cpu_matches = greedy_matches(hybrid, cpu_output, 100)
                hybrid_ep_matches = greedy_matches(hybrid, ep_output, 100)
                splice_rows.append(
                    {
                        "selection_group": case["selection_group"],
                        "image_id": image_id,
                        "model": model,
                        "replaced_boundary_index": boundary_index,
                        "replaced_boundary_name": boundary_input.name,
                        "hybrid_output_sha256": sha256_bytes(
                            np.ascontiguousarray(hybrid).tobytes(order="C")
                        ),
                        "hybrid_vs_ep_max_abs": hybrid_ep[0],
                        "hybrid_vs_ep_mean_abs": hybrid_ep[1],
                        "hybrid_vs_ep_cosine": hybrid_ep[2],
                        "hybrid_vs_cpu_max_abs": hybrid_cpu[0],
                        "hybrid_vs_cpu_mean_abs": hybrid_cpu[1],
                        "hybrid_vs_cpu_cosine": hybrid_cpu[2],
                        "cpu_ep_mean_abs": cpu_ep_mean_abs,
                        "mean_abs_recovery_fraction": recovery,
                        "top100_matches_with_cpu": len(hybrid_cpu_matches),
                        "top100_matches_with_ep": len(hybrid_ep_matches),
                        "score_ge_0_001": int(np.count_nonzero(hybrid[:, 4] >= 0.001)),
                        "score_ge_0_01": int(np.count_nonzero(hybrid[:, 4] >= 0.01)),
                        "score_ge_0_05": int(np.count_nonzero(hybrid[:, 4] >= 0.05)),
                    }
                )
    write_tsv(options.output_dir / "tail_boundary_splice.tsv", splice_rows)

    ranking_rows = []
    groups = ("large-loss", "small-loss", "matched-control", "all")
    for group in groups:
        for model in ("B2", "A1"):
            for boundary_index, boundary_input in enumerate(inputs):
                rows = [
                    row
                    for row in splice_rows
                    if row["model"] == model
                    and row["replaced_boundary_index"] == boundary_index
                    and (group == "all" or row["selection_group"] == group)
                ]
                if not rows:
                    raise RuntimeError(f"empty tail ranking group: {group} {model}")
                ranking_rows.append(
                    {
                        "selection_group": group,
                        "model": model,
                        "boundary_index": boundary_index,
                        "boundary_name": boundary_input.name,
                        "selected_cases": len(rows),
                        "mean_abs_recovery_fraction": float(
                            np.mean([float(row["mean_abs_recovery_fraction"]) for row in rows])
                        ),
                        "median_abs_recovery_fraction": float(
                            np.median([float(row["mean_abs_recovery_fraction"]) for row in rows])
                        ),
                        "mean_top100_matches_with_cpu": float(
                            np.mean([int(row["top100_matches_with_cpu"]) for row in rows])
                        ),
                    }
                )
    ranking_rows.sort(
        key=lambda row: (
            groups.index(row["selection_group"]),
            row["model"],
            -row["mean_abs_recovery_fraction"],
            row["boundary_index"],
        )
    )
    write_tsv(options.output_dir / "tail_boundary_splice_ranking.tsv", ranking_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
