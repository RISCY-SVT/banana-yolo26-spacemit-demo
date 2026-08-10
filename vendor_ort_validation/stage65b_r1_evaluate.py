#!/usr/bin/env python3
"""Run deterministic host COCO and hybrid-boundary YOLO26 evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import onnxruntime as ort

from stage64_preprocess import letterbox_rgb_nchw


COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

HYBRID_REPLACEMENTS = {
    "H0": (),
    "H1": (5,),
    "H2": (4,),
    "H3": (3,),
    "H4": (1,),
    "H5": (4, 5),
    "H6": (1, 3, 5),
    "H7": (0, 2, 4),
    "H8": (0, 1, 2, 3, 4, 5),
}


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def paths_from_list(path: Path, limit: int) -> list[Path]:
    paths = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        paths = paths[:limit]
    missing = [item for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"image list contains {len(missing)} missing files")
    return paths


def image_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"COCO image filename has no numeric stem: {path}") from exc


def image_tensor_and_geometry(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    height, width = image.shape[:2]
    scale = min(
        np.float32(640.0) / np.float32(width),
        np.float32(640.0) / np.float32(height),
    )
    resized_width = int(np.rint(float(width) * float(scale)))
    resized_height = int(np.rint(float(height) * float(scale)))
    geometry = {
        "width": float(width),
        "height": float(height),
        "scale": np.float32(scale),
        "pad_x": np.float32(640 - resized_width) / np.float32(2.0),
        "pad_y": np.float32(640 - resized_height) / np.float32(2.0),
    }
    return letterbox_rgb_nchw(path)[None, ...], geometry


def tail_run(
    tail: ort.InferenceSession, boundaries: list[np.ndarray]
) -> np.ndarray:
    inputs = {item.name: value for item, value in zip(tail.get_inputs(), boundaries)}
    return tail.run(None, inputs)[0]


def decode(
    output: np.ndarray, geometry: dict[str, float], threshold: float
) -> tuple[list[tuple[int, list[float], float]], int]:
    if output.shape != (1, 300, 6):
        raise ValueError(f"expected output shape 1x300x6, got {output.shape}")
    detections: list[tuple[int, list[float], float]] = []
    non_finite = 0
    for item in output[0]:
        if not np.isfinite(item).all():
            non_finite += 1
            continue
        score = float(item[4])
        if score < threshold:
            continue
        rounded = float(np.rint(item[5]))
        if abs(float(item[5]) - rounded) > 1.0e-4:
            continue
        class_id = int(rounded)
        if not 0 <= class_id < len(COCO_CATEGORY_IDS):
            continue
        x1, y1, x2, y2 = np.clip(
            item[:4], np.float32(0.0), np.float32(640.0)
        ).astype(np.float32, copy=False)
        x1 = np.clip(
            (x1 - geometry["pad_x"]) / geometry["scale"],
            np.float32(0.0),
            np.float32(geometry["width"]),
        )
        y1 = np.clip(
            (y1 - geometry["pad_y"]) / geometry["scale"],
            np.float32(0.0),
            np.float32(geometry["height"]),
        )
        x2 = np.clip(
            (x2 - geometry["pad_x"]) / geometry["scale"],
            np.float32(0.0),
            np.float32(geometry["width"]),
        )
        y2 = np.clip(
            (y2 - geometry["pad_y"]) / geometry["scale"],
            np.float32(0.0),
            np.float32(geometry["height"]),
        )
        if x2 > x1 and y2 > y1:
            detections.append(
                (
                    COCO_CATEGORY_IDS[class_id],
                    [
                        float(x1),
                        float(y1),
                        float(np.float32(x2 - x1)),
                        float(np.float32(y2 - y1)),
                    ],
                    score,
                )
            )
    return detections, non_finite


class PredictionWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = None
        self.first = True
        self.count = 0

    def __enter__(self) -> "PredictionWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        self.stream.write("[\n")
        return self

    def add(
        self, image: int, detections: Iterable[tuple[int, list[float], float]]
    ) -> None:
        assert self.stream is not None
        for category, bbox, score in detections:
            if not self.first:
                self.stream.write(",\n")
            self.first = False
            self.count += 1
            self.stream.write(
                "  {\"image_id\":%d,\"category_id\":%d,"
                "\"bbox\":[%.6f,%.6f,%.6f,%.6f],\"score\":%.6f}"
                % (image, category, *bbox, score)
            )

    def __exit__(self, *_: object) -> None:
        assert self.stream is not None
        self.stream.write("\n]\n")
        self.stream.close()


def score_row(
    surface: str, index: int, path: Path, output: np.ndarray
) -> dict[str, Any]:
    scores = output[..., 4]
    classes = output[..., 5]
    return {
        "surface": surface,
        "image_index": index,
        "image_id": image_id(path),
        "image": path.name,
        "output_sha256": sha256_array(output),
        "shape": "x".join(map(str, output.shape)),
        "non_finite_count": int(np.size(output) - np.count_nonzero(np.isfinite(output))),
        "score_min": float(np.nanmin(scores)),
        "score_max": float(np.nanmax(scores)),
        "score_mean": float(np.nanmean(scores)),
        "positive_scores": int(np.count_nonzero(scores > 0)),
        "scores_ge_0_25": int(np.count_nonzero(scores >= 0.25)),
        "unique_classes": int(np.unique(classes[np.isfinite(classes)]).size),
        "collapsed": int(
            not np.isfinite(scores).all()
            or float(np.nanstd(scores)) == 0.0
            or float(np.nanmax(scores)) <= 0.0
        ),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_candidate(options: argparse.Namespace) -> int:
    inference = session(options.candidate_inference, options.threads)
    tail = session(options.tail, options.threads)
    paths = paths_from_list(options.image_list, options.limit)
    timings: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    with PredictionWriter(options.output_dir / "predictions.json") as writer:
        for index, path in enumerate(paths):
            begin = time.perf_counter_ns()
            tensor, geometry = image_tensor_and_geometry(path)
            preprocessed = time.perf_counter_ns()
            boundaries = inference.run(None, {"images": tensor})
            inferred = time.perf_counter_ns()
            output = tail_run(tail, boundaries)
            tailed = time.perf_counter_ns()
            detections, non_finite = decode(output, geometry, options.threshold)
            decoded = time.perf_counter_ns()
            writer.add(image_id(path), detections)
            timings.append(
                {
                    "surface": options.name,
                    "image_index": index,
                    "image_id": image_id(path),
                    "image": path.name,
                    "input_sha256": sha256_array(tensor),
                    "preprocess_ms": (preprocessed - begin) / 1e6,
                    "inference_ms": (inferred - preprocessed) / 1e6,
                    "tail_ms": (tailed - inferred) / 1e6,
                    "decode_ms": (decoded - tailed) / 1e6,
                    "detection_count": len(detections),
                    "non_finite_rows": non_finite,
                }
            )
            scores.append(score_row(options.name, index, path, output))
            if options.log_every and (index + 1) % options.log_every == 0:
                print(f"{options.name}: {index + 1}/{len(paths)}", flush=True)
    write_tsv(options.output_dir / "timing.tsv", timings)
    write_tsv(options.output_dir / "score.tsv", scores)
    return 0


def run_hybrid(options: argparse.Namespace) -> int:
    fp32 = session(options.fp32_inference, options.threads)
    candidate = session(options.candidate_inference, options.threads)
    tail = session(options.tail, options.threads)
    paths = paths_from_list(options.image_list, options.limit)
    arms = [item.strip() for item in options.arms.split(",") if item.strip()]
    unknown = sorted(set(arms) - HYBRID_REPLACEMENTS.keys())
    if unknown:
        raise ValueError(f"unknown hybrid arms: {unknown}")
    timings: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    with ExitStack() as stack:
        writers = {
            arm: stack.enter_context(
                PredictionWriter(options.output_dir / arm / "predictions.json")
            )
            for arm in arms
        }
        for index, path in enumerate(paths):
            begin = time.perf_counter_ns()
            tensor, geometry = image_tensor_and_geometry(path)
            preprocessed = time.perf_counter_ns()
            fp_boundaries = fp32.run(None, {"images": tensor})
            fp_inferred = time.perf_counter_ns()
            q_boundaries = candidate.run(None, {"images": tensor})
            q_inferred = time.perf_counter_ns()
            for arm in arms:
                boundaries = list(q_boundaries)
                for boundary_index in HYBRID_REPLACEMENTS[arm]:
                    boundaries[boundary_index] = fp_boundaries[boundary_index]
                tail_start = time.perf_counter_ns()
                output = tail_run(tail, boundaries)
                tail_end = time.perf_counter_ns()
                detections, non_finite = decode(output, geometry, options.threshold)
                decode_end = time.perf_counter_ns()
                writers[arm].add(image_id(path), detections)
                timings.append(
                    {
                        "arm": arm,
                        "image_index": index,
                        "image_id": image_id(path),
                        "image": path.name,
                        "input_sha256": sha256_array(tensor),
                        "preprocess_ms": (preprocessed - begin) / 1e6,
                        "fp32_inference_ms": (fp_inferred - preprocessed) / 1e6,
                        "candidate_inference_ms": (q_inferred - fp_inferred) / 1e6,
                        "tail_ms": (tail_end - tail_start) / 1e6,
                        "decode_ms": (decode_end - tail_end) / 1e6,
                        "detection_count": len(detections),
                        "non_finite_rows": non_finite,
                        "replaced_boundary_indices": ",".join(
                            map(str, HYBRID_REPLACEMENTS[arm])
                        ),
                    }
                )
                scores.append(score_row(arm, index, path, output))
            if options.log_every and (index + 1) % options.log_every == 0:
                print(f"hybrid: {index + 1}/{len(paths)}", flush=True)
    write_tsv(options.output_dir / "timing.tsv", timings)
    write_tsv(options.output_dir / "score.tsv", scores)
    for arm in arms:
        write_tsv(
            options.output_dir / arm / "timing.tsv",
            [row for row in timings if row["arm"] == arm],
        )
        write_tsv(
            options.output_dir / arm / "score.tsv",
            [row for row in scores if row["surface"] == arm],
        )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--candidate-inference", required=True, type=Path)
    common.add_argument("--tail", required=True, type=Path)
    common.add_argument("--image-list", required=True, type=Path)
    common.add_argument("--output-dir", required=True, type=Path)
    common.add_argument("--limit", type=int, default=0)
    common.add_argument("--threshold", type=float, default=0.001)
    common.add_argument("--log-every", type=int, default=100)
    common.add_argument("--threads", type=int, default=4)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("--name", required=True)
    hybrid = commands.add_parser("hybrid", parents=[common])
    hybrid.add_argument("--fp32-inference", required=True, type=Path)
    hybrid.add_argument("--arms", default=",".join(HYBRID_REPLACEMENTS))
    return root


def main() -> int:
    options = parser().parse_args()
    if options.limit < 0 or options.threads < 1 or not math.isfinite(options.threshold):
        raise ValueError("invalid limit or threshold")
    options.output_dir.mkdir(parents=True, exist_ok=False)
    if options.command == "run":
        return run_candidate(options)
    return run_hybrid(options)


if __name__ == "__main__":
    raise SystemExit(main())
