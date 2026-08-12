#!/usr/bin/env python3
"""Run F0/F1/H8 through one deterministic host runner and compare raw outputs."""

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
    session,
    sha256_array,
    tail_run,
)
from stage65b_r2_common import sha256, write_tsv


SURFACES = ("F0", "F1", "H8")


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "byte_identical": 0,
            "mae": float("nan"),
            "max_abs": float("nan"),
            "cosine": float("nan"),
        }
    left = reference.astype(np.float64, copy=False).reshape(-1)
    right = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(left - right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "byte_identical": int(
            reference.dtype == candidate.dtype
            and np.ascontiguousarray(reference).tobytes()
            == np.ascontiguousarray(candidate).tobytes()
        ),
        "mae": float(delta.mean()),
        "max_abs": float(delta.max()),
        "cosine": float(np.dot(left, right) / denominator)
        if denominator
        else float("nan"),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-model", required=True, type=Path)
    result.add_argument("--fp32-inference", required=True, type=Path)
    result.add_argument("--candidate-inference", required=True, type=Path)
    result.add_argument("--tail", required=True, type=Path)
    result.add_argument("--image-list", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--threshold", type=float, default=0.001)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--log-every", type=int, default=100)
    return result


def main() -> int:
    options = parser().parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    if options.limit < 0 or options.threads < 1 or not math.isfinite(options.threshold):
        raise ValueError("invalid limit, thread count, or threshold")
    options.output_dir.mkdir(parents=True)

    source = session(options.source_model, options.threads)
    fp32 = session(options.fp32_inference, options.threads)
    candidate = session(options.candidate_inference, options.threads)
    tail = session(options.tail, options.threads)
    paths = paths_from_list(options.image_list, options.limit)
    if len(fp32.get_outputs()) != 6 or len(candidate.get_outputs()) != 6:
        raise ValueError("F1/H8 contract requires exactly six inference outputs")
    if [item.name for item in fp32.get_outputs()] != [
        item.name for item in candidate.get_outputs()
    ]:
        raise ValueError("FP32 and candidate boundary order/name mismatch")

    raw_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    aggregate = {surface: hashlib.sha256() for surface in SURFACES}
    exact_f0_f1 = 0
    exact_f1_h8 = 0
    non_finite = {surface: 0 for surface in SURFACES}
    prediction_counts: dict[str, int] = {}

    with ExitStack() as stack:
        writers = {
            surface: stack.enter_context(
                PredictionWriter(options.output_dir / surface / "predictions.json")
            )
            for surface in SURFACES
        }
        for index, path in enumerate(paths):
            begin = time.perf_counter_ns()
            tensor, geometry = image_tensor_and_geometry(path)
            preprocessed = time.perf_counter_ns()
            f0 = source.run(None, {"images": tensor})[0]
            source_done = time.perf_counter_ns()
            fp_boundaries = fp32.run(None, {"images": tensor})
            fp_done = time.perf_counter_ns()
            candidate_boundaries = candidate.run(None, {"images": tensor})
            candidate_done = time.perf_counter_ns()
            f1 = tail_run(tail, fp_boundaries)
            f1_done = time.perf_counter_ns()

            # Preserve the accepted H8 mechanics: begin with candidate outputs,
            # then replace all six slots with the independently executed FP32 outputs.
            h8_boundaries = list(candidate_boundaries)
            for boundary_index in range(6):
                h8_boundaries[boundary_index] = fp_boundaries[boundary_index]
            h8 = tail_run(tail, h8_boundaries)
            h8_done = time.perf_counter_ns()

            outputs = {"F0": f0, "F1": f1, "H8": h8}
            comparisons = {
                "f0_f1": compare(f0, f1),
                "f1_h8": compare(f1, h8),
            }
            exact_f0_f1 += int(comparisons["f0_f1"]["byte_identical"])
            exact_f1_h8 += int(comparisons["f1_h8"]["byte_identical"])
            for surface, output in outputs.items():
                if output.shape != (1, 300, 6):
                    raise ValueError(f"{surface} output shape is {output.shape}")
                non_finite[surface] += int(output.size - np.isfinite(output).sum())
                output_bytes = np.ascontiguousarray(output).tobytes()
                aggregate[surface].update(output_bytes)
                detections, invalid_rows = decode(output, geometry, options.threshold)
                non_finite[surface] += invalid_rows
                writers[surface].add(image_id(path), detections)

            raw_rows.append(
                {
                    "image_index": index,
                    "image_id": image_id(path),
                    "image": path.name,
                    "input_sha256": sha256_array(tensor),
                    "f0_output_sha256": sha256_array(f0),
                    "f1_output_sha256": sha256_array(f1),
                    "h8_output_sha256": sha256_array(h8),
                    "f0_f1_byte_identical": comparisons["f0_f1"]["byte_identical"],
                    "f0_f1_mae": comparisons["f0_f1"]["mae"],
                    "f0_f1_max_abs": comparisons["f0_f1"]["max_abs"],
                    "f0_f1_cosine": comparisons["f0_f1"]["cosine"],
                    "f1_h8_byte_identical": comparisons["f1_h8"]["byte_identical"],
                    "f1_h8_mae": comparisons["f1_h8"]["mae"],
                    "f1_h8_max_abs": comparisons["f1_h8"]["max_abs"],
                    "f1_h8_cosine": comparisons["f1_h8"]["cosine"],
                }
            )
            for boundary_index, (fp_value, candidate_value, output_info) in enumerate(
                zip(fp_boundaries, candidate_boundaries, fp32.get_outputs())
            ):
                boundary_rows.append(
                    {
                        "image_index": index,
                        "image_id": image_id(path),
                        "boundary_index": boundary_index,
                        "boundary_name": output_info.name,
                        "fp32_sha256": sha256_array(fp_value),
                        "candidate_sha256": sha256_array(candidate_value),
                        "shape": "x".join(map(str, fp_value.shape)),
                        "dtype": str(fp_value.dtype),
                    }
                )
            timing_rows.append(
                {
                    "image_index": index,
                    "image_id": image_id(path),
                    "preprocess_ms": (preprocessed - begin) / 1e6,
                    "f0_ms": (source_done - preprocessed) / 1e6,
                    "fp32_inference_ms": (fp_done - source_done) / 1e6,
                    "candidate_inference_ms": (candidate_done - fp_done) / 1e6,
                    "f1_tail_ms": (f1_done - candidate_done) / 1e6,
                    "h8_tail_ms": (h8_done - f1_done) / 1e6,
                }
            )
            if options.log_every and (index + 1) % options.log_every == 0:
                print(f"reconcile: {index + 1}/{len(paths)}", flush=True)
        prediction_counts = {
            surface: writer.count for surface, writer in writers.items()
        }

    write_tsv(options.output_dir / "raw_comparison.tsv", raw_rows)
    write_tsv(options.output_dir / "boundary_hashes.tsv", boundary_rows)
    write_tsv(options.output_dir / "timing.tsv", timing_rows)
    summary = {
        "images": len(paths),
        "source_model_sha256": sha256(options.source_model),
        "fp32_inference_sha256": sha256(options.fp32_inference),
        "candidate_inference_sha256": sha256(options.candidate_inference),
        "tail_sha256": sha256(options.tail),
        "f0_aggregate_output_sha256": aggregate["F0"].hexdigest(),
        "f1_aggregate_output_sha256": aggregate["F1"].hexdigest(),
        "h8_aggregate_output_sha256": aggregate["H8"].hexdigest(),
        "f0_f1_exact_images": exact_f0_f1,
        "f1_h8_exact_images": exact_f1_h8,
        "f0_non_finite": non_finite["F0"],
        "f1_non_finite": non_finite["F1"],
        "h8_non_finite": non_finite["H8"],
        "f0_prediction_count": prediction_counts["F0"],
        "f1_prediction_count": prediction_counts["F1"],
        "h8_prediction_count": prediction_counts["H8"],
        "status": "pass"
        if exact_f1_h8 == len(paths) and not any(non_finite.values())
        else "fail",
    }
    write_tsv(options.output_dir / "reconcile_summary.tsv", [summary])
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
