#!/usr/bin/env python3
"""Run exact split controls and Stage65B-R3 cut/splice host arms."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from stage65b_r1_evaluate import (
    PredictionWriter,
    decode,
    image_id,
    image_tensor_and_geometry,
    paths_from_list,
    score_row,
    session,
    sha256_array,
    tail_run,
)
from stage65b_r3_common import sha256, write_tsv


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "byte_identical": 0,
            "mae": float("nan"),
            "max_abs": float("nan"),
            "cosine": float("nan"),
        }
    left64 = left.astype(np.float64, copy=False).reshape(-1)
    right64 = right.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(left64 - right64)
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    return {
        "byte_identical": int(
            np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
        ),
        "mae": float(delta.mean()) if delta.size else 0.0,
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "cosine": float(np.dot(left64, right64) / denominator)
        if denominator
        else float("nan"),
    }


def compare_lists(
    left: list[np.ndarray], right: list[np.ndarray]
) -> dict[str, Any]:
    if len(left) != len(right):
        return {"byte_identical": 0, "mae": float("nan"), "max_abs": float("nan")}
    rows = [compare(a, b) for a, b in zip(left, right)]
    total = sum(a.size for a in left)
    weighted_mae = (
        sum(float(row["mae"]) * array.size for row, array in zip(rows, left)) / total
        if total
        else 0.0
    )
    return {
        "byte_identical": int(all(row["byte_identical"] for row in rows)),
        "mae": weighted_mae,
        "max_abs": max(float(row["max_abs"]) for row in rows),
    }


def frontier_ids(model_root: Path, requested: list[str]) -> list[str]:
    available = sorted(path.name for path in model_root.iterdir() if path.is_dir())
    result = requested or available
    missing = sorted(set(result) - set(available))
    if missing:
        raise ValueError(f"unknown frontiers: {missing}")
    if not result:
        raise ValueError("no frontiers selected")
    return result


def model_path(root: Path, frontier: str, role: str) -> Path:
    path = root / frontier / f"{frontier}.{role}.onnx"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_controls(options: argparse.Namespace) -> int:
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)
    frontiers = frontier_ids(options.model_root, options.frontier)
    paths = paths_from_list(options.image_list, options.limit)
    fp32 = session(options.fp32, options.threads)
    b2 = session(options.b2, options.threads)
    d8 = session(options.d8, options.threads)
    tail = session(options.tail, options.threads)
    split_sessions: dict[str, dict[str, Any]] = {}
    for frontier in frontiers:
        split_sessions[frontier] = {
            role: session(model_path(options.model_root, frontier, role), options.threads)
            for role in (
                "fp32-prefix",
                "fp32-suffix",
                "b2-prefix",
                "b2-suffix",
                "b2-d8-suffix",
            )
        }
    rows: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for frontier in frontiers:
        for control in ("fp32", "b2", "d8"):
            aggregates[(frontier, control)] = {
                "images": 0,
                "boundary_exact": 0,
                "tail_exact": 0,
                "max_boundary_abs": 0.0,
                "max_tail_abs": 0.0,
                "non_finite": 0,
            }
    for image_index, path in enumerate(paths):
        tensor, _ = image_tensor_and_geometry(path)
        fp_reference = fp32.run(None, {fp32.get_inputs()[0].name: tensor})
        b2_reference = b2.run(None, {b2.get_inputs()[0].name: tensor})
        d8_reference = d8.run(None, {d8.get_inputs()[0].name: tensor})
        tail_references = {
            "fp32": tail_run(tail, fp_reference),
            "b2": tail_run(tail, b2_reference),
            "d8": tail_run(tail, d8_reference),
        }
        for frontier in frontiers:
            arms = split_sessions[frontier]
            fp_cut = arms["fp32-prefix"].run(
                None, {arms["fp32-prefix"].get_inputs()[0].name: tensor}
            )
            fp_split = arms["fp32-suffix"].run(
                None,
                {item.name: value for item, value in zip(arms["fp32-suffix"].get_inputs(), fp_cut)},
            )
            b2_cut = arms["b2-prefix"].run(
                None, {arms["b2-prefix"].get_inputs()[0].name: tensor}
            )
            b2_split = arms["b2-suffix"].run(
                None,
                {item.name: value for item, value in zip(arms["b2-suffix"].get_inputs(), b2_cut)},
            )
            d8_split = arms["b2-d8-suffix"].run(
                None,
                {item.name: value for item, value in zip(arms["b2-d8-suffix"].get_inputs(), b2_cut)},
            )
            for control, reference, reconstructed in (
                ("fp32", fp_reference, fp_split),
                ("b2", b2_reference, b2_split),
                ("d8", d8_reference, d8_split),
            ):
                boundary = compare_lists(reference, reconstructed)
                reconstructed_tail = tail_run(tail, reconstructed)
                tail_comparison = compare(tail_references[control], reconstructed_tail)
                non_finite = int(
                    sum(array.size - np.count_nonzero(np.isfinite(array)) for array in reconstructed)
                    + reconstructed_tail.size
                    - np.count_nonzero(np.isfinite(reconstructed_tail))
                )
                row = {
                    "scope": options.scope,
                    "frontier": frontier,
                    "control": control,
                    "image_index": image_index,
                    "image": path.name,
                    "input_sha256": sha256_array(tensor),
                    "boundary_byte_identical": boundary["byte_identical"],
                    "boundary_mae": boundary["mae"],
                    "boundary_max_abs": boundary["max_abs"],
                    "tail_byte_identical": tail_comparison["byte_identical"],
                    "tail_mae": tail_comparison["mae"],
                    "tail_max_abs": tail_comparison["max_abs"],
                    "non_finite": non_finite,
                    "status": "pass"
                    if boundary["byte_identical"] and tail_comparison["byte_identical"] and non_finite == 0
                    else "fail",
                }
                rows.append(row)
                aggregate = aggregates[(frontier, control)]
                aggregate["images"] += 1
                aggregate["boundary_exact"] += int(boundary["byte_identical"])
                aggregate["tail_exact"] += int(tail_comparison["byte_identical"])
                aggregate["max_boundary_abs"] = max(aggregate["max_boundary_abs"], boundary["max_abs"])
                aggregate["max_tail_abs"] = max(aggregate["max_tail_abs"], tail_comparison["max_abs"])
                aggregate["non_finite"] += non_finite
        if options.log_every and (image_index + 1) % options.log_every == 0:
            print(f"controls {options.scope}: {image_index + 1}/{len(paths)}", flush=True)
    write_tsv(options.output_dir / "split_control_per_image.tsv", rows)
    summaries: list[dict[str, Any]] = []
    for (frontier, control), aggregate in aggregates.items():
        status = (
            aggregate["boundary_exact"] == aggregate["images"]
            and aggregate["tail_exact"] == aggregate["images"]
            and aggregate["non_finite"] == 0
        )
        summaries.append(
            {
                "scope": options.scope,
                "frontier": frontier,
                "control": control,
                **aggregate,
                "status": "pass" if status else "fail",
            }
        )
    write_tsv(options.output_dir / "split_control_results.tsv", summaries)
    failures = [row for row in rows if row["status"] != "pass"]
    write_tsv(
        options.output_dir / "split_control_failures.tsv",
        failures or [{"scope": options.scope, "frontier": "none", "status": "no-failures"}],
    )
    return 1 if failures else 0


def run_arms(options: argparse.Namespace) -> int:
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)
    frontiers = frontier_ids(options.model_root, options.frontier)
    paths = paths_from_list(options.image_list, options.limit)
    tail = session(options.tail, options.threads)
    arm_sessions: dict[str, tuple[Any, Any]] = {}
    for frontier in frontiers:
        if options.direction == "FQ8":
            prefix_role, suffix_role = "fp32-prefix", "b2-d8-suffix"
        elif options.direction == "QF":
            prefix_role, suffix_role = "b2-prefix", "fp32-suffix"
        else:
            raise ValueError(f"unsupported direction: {options.direction}")
        arm_sessions[frontier] = (
            session(model_path(options.model_root, frontier, prefix_role), options.threads),
            session(model_path(options.model_root, frontier, suffix_role), options.threads),
        )
    aggregate_hash = {frontier: hashlib.sha256() for frontier in frontiers}
    score_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    non_finite = {frontier: 0 for frontier in frontiers}
    with ExitStack() as stack:
        writers = {
            frontier: stack.enter_context(
                PredictionWriter(options.output_dir / frontier / "predictions.json")
            )
            for frontier in frontiers
        }
        for index, path in enumerate(paths):
            begin = time.perf_counter_ns()
            tensor, geometry = image_tensor_and_geometry(path)
            preprocessed = time.perf_counter_ns()
            for frontier in frontiers:
                prefix, suffix = arm_sessions[frontier]
                started = time.perf_counter_ns()
                cut = prefix.run(None, {prefix.get_inputs()[0].name: tensor})
                prefixed = time.perf_counter_ns()
                boundaries = suffix.run(
                    None,
                    {item.name: value for item, value in zip(suffix.get_inputs(), cut)},
                )
                suffixed = time.perf_counter_ns()
                output = tail_run(tail, boundaries)
                tailed = time.perf_counter_ns()
                if output.shape != (1, 300, 6):
                    raise ValueError(f"{frontier}: invalid final shape {output.shape}")
                invalid = int(output.size - np.count_nonzero(np.isfinite(output)))
                detections, invalid_rows = decode(output, geometry, options.threshold)
                non_finite[frontier] += invalid + invalid_rows
                writers[frontier].add(image_id(path), detections)
                aggregate_hash[frontier].update(np.ascontiguousarray(output).tobytes())
                score_rows.append(score_row(f"{options.direction}-{frontier}", index, path, output))
                timing_rows.append(
                    {
                        "direction": options.direction,
                        "frontier": frontier,
                        "image_index": index,
                        "image_id": image_id(path),
                        "image": path.name,
                        "input_sha256": sha256_array(tensor),
                        "preprocess_shared_ms": (preprocessed - begin) / 1e6,
                        "prefix_ms": (prefixed - started) / 1e6,
                        "suffix_ms": (suffixed - prefixed) / 1e6,
                        "tail_ms": (tailed - suffixed) / 1e6,
                        "detection_count": len(detections),
                        "non_finite": invalid + invalid_rows,
                    }
                )
            if options.log_every and (index + 1) % options.log_every == 0:
                print(
                    f"{options.direction}: {index + 1}/{len(paths)} "
                    f"frontiers={len(frontiers)}",
                    flush=True,
                )
        prediction_counts = {frontier: writers[frontier].count for frontier in frontiers}
    write_tsv(options.output_dir / "score.tsv", score_rows)
    write_tsv(options.output_dir / "timing.tsv", timing_rows)
    summary = []
    for frontier in frontiers:
        prediction = options.output_dir / frontier / "predictions.json"
        collapsed = sum(
            int(row["collapsed"])
            for row in score_rows
            if row["surface"] == f"{options.direction}-{frontier}"
        )
        summary.append(
            {
                "direction": options.direction,
                "frontier": frontier,
                "images": len(paths),
                "prediction_count": prediction_counts[frontier],
                "prediction_sha256": sha256(prediction),
                "aggregate_output_sha256": aggregate_hash[frontier].hexdigest(),
                "non_finite": non_finite[frontier],
                "score_collapsed_images": collapsed,
                "status": "pass" if non_finite[frontier] == 0 and collapsed == 0 else "fail",
            }
        )
    write_tsv(options.output_dir / "arm_summary.tsv", summary)
    return 1 if any(row["status"] != "pass" for row in summary) else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    controls = subparsers.add_parser("controls")
    controls.add_argument("--fp32", required=True, type=Path)
    controls.add_argument("--b2", required=True, type=Path)
    controls.add_argument("--d8", required=True, type=Path)
    controls.add_argument("--tail", required=True, type=Path)
    controls.add_argument("--model-root", required=True, type=Path)
    controls.add_argument("--image-list", required=True, type=Path)
    controls.add_argument("--output-dir", required=True, type=Path)
    controls.add_argument("--scope", required=True)
    controls.add_argument("--frontier", action="append", default=[])
    controls.add_argument("--limit", type=int, default=0)
    controls.add_argument("--threads", type=int, default=4)
    controls.add_argument("--log-every", type=int, default=10)

    arms = subparsers.add_parser("arms")
    arms.add_argument("--direction", required=True, choices=("FQ8", "QF"))
    arms.add_argument("--tail", required=True, type=Path)
    arms.add_argument("--model-root", required=True, type=Path)
    arms.add_argument("--image-list", required=True, type=Path)
    arms.add_argument("--output-dir", required=True, type=Path)
    arms.add_argument("--frontier", action="append", default=[])
    arms.add_argument("--limit", type=int, default=0)
    arms.add_argument("--threads", type=int, default=4)
    arms.add_argument("--threshold", type=float, default=0.001)
    arms.add_argument("--log-every", type=int, default=25)
    return result


def main() -> int:
    options = parser().parse_args()
    if options.limit < 0 or options.threads < 1:
        raise ValueError("invalid limit or thread count")
    if options.command == "controls":
        return run_controls(options)
    if not math.isfinite(options.threshold):
        raise ValueError("non-finite threshold")
    return run_arms(options)


if __name__ == "__main__":
    raise SystemExit(main())
