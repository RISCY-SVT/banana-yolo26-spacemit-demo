#!/usr/bin/env python3
"""Validate a diagnostic direct-E2E XSlim model against its FP32 source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from stage64_preprocess import letterbox_rgb_nchw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        return {
            "mae": math.nan,
            "max_abs_difference": math.nan,
            "cosine_similarity": math.nan,
        }
    left = reference.astype(np.float64).reshape(-1)
    right = candidate.astype(np.float64).reshape(-1)
    difference = np.abs(left - right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "mae": float(difference.mean()),
        "max_abs_difference": float(difference.max()),
        "cosine_similarity": (
            float(np.dot(left, right) / denominator)
            if denominator
            else math.nan
        ),
    }


def main() -> int:
    options = parse_args()
    image_paths = [
        Path(line)
        for line in options.image_list.read_text(encoding="utf-8").splitlines()
        if line
    ][: options.limit]
    source = session(options.source_model)
    candidate = session(options.candidate_model)
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        input_tensor = letterbox_rgb_nchw(image_path)[None, ...]
        reference = source.run(None, {"images": input_tensor})[0]
        value = candidate.run(None, {"images": input_tensor})[0]
        finite = np.isfinite(value)
        scores = value[..., 4] if value.ndim == 3 and value.shape[-1] >= 5 else value
        collapsed = (
            not finite.all()
            or np.count_nonzero(scores) == 0
            or float(np.nanstd(scores)) == 0.0
        )
        rows.append(
            {
                "candidate": options.candidate_name,
                "image_index": index,
                "image": image_path.name,
                "source_shape": "x".join(map(str, reference.shape)),
                "candidate_shape": "x".join(map(str, value.shape)),
                "source_sha256": digest(reference),
                "candidate_sha256": digest(value),
                "non_finite_count": int(value.size - np.count_nonzero(finite)),
                "score_nonzero_count": int(np.count_nonzero(scores)),
                "score_minimum": float(np.nanmin(scores)),
                "score_maximum": float(np.nanmax(scores)),
                "score_stddev": float(np.nanstd(scores)),
                "score_ge_0_25": int(np.count_nonzero(scores >= 0.25)),
                "score_collapsed": int(collapsed),
                **compare(reference, value),
                "status": (
                    "pass"
                    if value.shape == (1, 300, 6)
                    and finite.all()
                    and not collapsed
                    else "fail"
                ),
            }
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["status"] == "pass" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
