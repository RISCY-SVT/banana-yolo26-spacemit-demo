#!/usr/bin/env python3
"""Run source and two-stage ONNX pipelines on deterministic image lists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from stage64_preprocess import letterbox_rgb_nchw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--fp32-tail", required=True, type=Path)
    parser.add_argument("--candidate-inference", required=True, type=Path)
    parser.add_argument("--candidate-tail", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate-name", required=True)
    return parser.parse_args()


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def stats(array: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(array)
    values = array[finite]
    return {
        "shape": "x".join(map(str, array.shape)),
        "dtype": str(array.dtype),
        "sha256": digest(array),
        "non_finite_count": int(array.size - values.size),
        "minimum": float(values.min()) if values.size else float("nan"),
        "maximum": float(values.max()) if values.size else float("nan"),
        "mean": float(values.mean()) if values.size else float("nan"),
        "stddev": float(values.std()) if values.size else float("nan"),
        "zero_count": int(np.count_nonzero(array == 0)),
        "nonzero_count": int(np.count_nonzero(array)),
        "negative_count": int(np.count_nonzero(array < 0)),
        "positive_count": int(np.count_nonzero(array > 0)),
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        return {
            "mae": float("nan"),
            "max_abs_difference": float("nan"),
            "cosine_similarity": float("nan"),
        }
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    difference = np.abs(ref - cand)
    denominator = np.linalg.norm(ref) * np.linalg.norm(cand)
    cosine = float(np.dot(ref, cand) / denominator) if denominator else float("nan")
    return {
        "mae": float(difference.mean()),
        "max_abs_difference": float(difference.max()),
        "cosine_similarity": cosine,
    }


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_two_stage(
    inference: ort.InferenceSession,
    tail: ort.InferenceSession,
    input_tensor: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, float, float]:
    start = time.perf_counter_ns()
    boundaries = inference.run(None, {"images": input_tensor})
    middle = time.perf_counter_ns()
    tail_inputs = {
        item.name: value for item, value in zip(tail.get_inputs(), boundaries)
    }
    final = tail.run(None, tail_inputs)[0]
    end = time.perf_counter_ns()
    return boundaries, final, (middle - start) / 1e6, (end - middle) / 1e6


def run_tail(
    tail: ort.InferenceSession,
    boundaries: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    start = time.perf_counter_ns()
    tail_inputs = {
        item.name: value for item, value in zip(tail.get_inputs(), boundaries)
    }
    final = tail.run(None, tail_inputs)[0]
    return final, (time.perf_counter_ns() - start) / 1e6


def main() -> int:
    options = parse_args()
    paths = [
        Path(line.strip())
        for line in options.image_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if options.limit > 0:
        paths = paths[: options.limit]
    options.output_dir.mkdir(parents=True, exist_ok=True)

    source = session(options.source_model)
    fp32_inference = session(options.fp32_inference)
    fp32_tail = session(options.fp32_tail)
    candidate_inference = session(options.candidate_inference)
    candidate_tail = session(options.candidate_tail)

    semantic_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for image_index, path in enumerate(paths):
        input_tensor = letterbox_rgb_nchw(path)[None, ...]
        source_output = source.run(None, {"images": input_tensor})[0]
        fp_boundaries, fp_output, fp_inference_ms, fp_tail_ms = run_two_stage(
            fp32_inference, fp32_tail, input_tensor
        )
        q_boundaries, q_output, q_inference_ms, q_tail_ms = run_two_stage(
            candidate_inference, candidate_tail, input_tensor
        )
        candidate_tail_fp_output, candidate_tail_fp_ms = run_tail(
            candidate_tail, fp_boundaries
        )
        confidence_branch_collapsed = False
        for index, (reference, candidate, output_info) in enumerate(
            zip(fp_boundaries, q_boundaries, candidate_inference.get_outputs())
        ):
            branch_kind = "bbox" if index % 2 == 0 else "confidence"
            branch_collapsed = (
                branch_kind == "confidence"
                and np.count_nonzero(reference) > 0
                and (
                    not np.isfinite(candidate).all()
                    or np.count_nonzero(candidate) == 0
                    or float(np.nanstd(candidate)) == 0.0
                )
            )
            confidence_branch_collapsed |= branch_collapsed
            row = {
                "candidate": options.candidate_name,
                "image_index": image_index,
                "image": path.name,
                "boundary_index": index,
                "boundary_name": output_info.name,
                "branch_kind": branch_kind,
                "branch_collapsed": int(branch_collapsed),
                **{f"reference_{k}": v for k, v in stats(reference).items()},
                **{f"candidate_{k}": v for k, v in stats(candidate).items()},
                **compare(reference, candidate),
            }
            boundary_rows.append(row)

        source_compare = compare(source_output, fp_output)
        candidate_compare = compare(source_output, q_output)
        candidate_tail_compare = compare(source_output, candidate_tail_fp_output)
        for surface, value, timing in (
            ("source-fp32", source_output, (float("nan"), float("nan"))),
            ("split-fp32", fp_output, (fp_inference_ms, fp_tail_ms)),
            (
                "candidate-tail-on-fp32-boundaries",
                candidate_tail_fp_output,
                (float("nan"), candidate_tail_fp_ms),
            ),
            ("split-s8", q_output, (q_inference_ms, q_tail_ms)),
        ):
            final_rows.append(
                {
                    "candidate": options.candidate_name,
                    "image_index": image_index,
                    "image": path.name,
                    "surface": surface,
                    **stats(value),
                    "inference_ms": timing[0],
                    "tail_ms": timing[1],
                    "total_ms": sum(timing),
                }
            )
        semantic_rows.append(
            {
                "candidate": options.candidate_name,
                "image_index": image_index,
                "image": path.name,
                "fp32_split_vs_unsplit_mae": source_compare["mae"],
                "fp32_split_vs_unsplit_max_abs": source_compare[
                    "max_abs_difference"
                ],
                "fp32_split_vs_unsplit_cosine": source_compare[
                    "cosine_similarity"
                ],
                "s8_vs_fp32_mae": candidate_compare["mae"],
                "s8_vs_fp32_max_abs": candidate_compare["max_abs_difference"],
                "s8_vs_fp32_cosine": candidate_compare["cosine_similarity"],
                "candidate_tail_vs_source_mae": candidate_tail_compare["mae"],
                "candidate_tail_vs_source_max_abs": candidate_tail_compare[
                    "max_abs_difference"
                ],
                "candidate_tail_vs_source_cosine": candidate_tail_compare[
                    "cosine_similarity"
                ],
                "confidence_branch_collapsed": int(confidence_branch_collapsed),
                "status": (
                    "pass"
                    if np.isfinite(q_output).all()
                    and q_output.shape == (1, 300, 6)
                    and np.isfinite(candidate_tail_fp_output).all()
                    and candidate_tail_fp_output.shape == (1, 300, 6)
                    and not confidence_branch_collapsed
                    else "fail"
                ),
            }
        )
        for surface, value in (
            ("source-fp32", source_output),
            ("split-fp32", fp_output),
            ("candidate-tail-on-fp32-boundaries", candidate_tail_fp_output),
            ("split-s8", q_output),
        ):
            scores = value[..., 4]
            classes = value[..., 5]
            score_rows.append(
                {
                    "candidate": options.candidate_name,
                    "image_index": image_index,
                    "image": path.name,
                    "surface": surface,
                    "score_min": float(np.nanmin(scores)),
                    "score_max": float(np.nanmax(scores)),
                    "score_mean": float(np.nanmean(scores)),
                    "positive_score_count": int(np.count_nonzero(scores > 0)),
                    "score_over_0_25_count": int(np.count_nonzero(scores >= 0.25)),
                    "unique_class_count": int(np.unique(classes).size),
                    "collapsed": int(
                        not np.isfinite(scores).all()
                        or float(np.nanmax(scores)) <= 0.0
                    ),
                }
            )

    def output(name: str, rows: list[dict[str, Any]]) -> None:
        with (options.output_dir / name).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    output("host_cpu_semantic_matrix.tsv", semantic_rows)
    output("host_boundary_comparison.tsv", boundary_rows)
    output("host_final_output_comparison.tsv", final_rows)
    output("score_channel_range.tsv", score_rows)
    output("score_collapse_gate.tsv", score_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
