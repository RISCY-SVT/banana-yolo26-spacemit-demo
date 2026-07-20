#!/usr/bin/env python3
"""Evaluate one Stage60 prediction surface and explicit object-size bins."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PIXEL_BINS = ((0.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0),
              (24.0, 32.0), (32.0, 48.0), (48.0, 64.0), (64.0, 96.0),
              (96.0, float("inf")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_mean(values: np.ndarray) -> float:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else float("nan")


def iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    intersection_width = max(0.0, min(lx1 + lw, rx1 + rw) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly1 + lh, ry1 + rh) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


def pixel_bin(value: float) -> str:
    for lower, upper in PIXEL_BINS:
        if lower <= value < upper:
            return f">={int(lower)}" if np.isinf(upper) else f"{int(lower)}-{int(upper)}"
    raise AssertionError("pixel bin missing")


def area_bin(area: float) -> str:
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def match_detections(coco: COCO, predictions: list[dict[str, Any]], confidence: float,
                     resolution: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detections: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction["score"]) >= confidence:
            detections[(int(prediction["image_id"]), int(prediction["category_id"]))].append(prediction)
    annotations: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.dataset["annotations"]:
        if annotation.get("iscrowd", 0) == 0:
            annotations[(int(annotation["image_id"]), int(annotation["category_id"]))].append(annotation)

    gt_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    keys = sorted(set(annotations) | set(detections))
    for image_id, category_id in keys:
        image = coco.imgs[image_id]
        ratio = min(resolution / int(image["width"]), resolution / int(image["height"]))
        ground_truth = annotations[(image_id, category_id)]
        matched: set[int] = set()
        for detection in sorted(detections[(image_id, category_id)],
                                key=lambda item: -float(item["score"])):
            best_index = -1
            best_iou = 0.5
            for index, annotation in enumerate(ground_truth):
                if index in matched:
                    continue
                overlap = iou(detection["bbox"], annotation["bbox"])
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0:
                matched.add(best_index)
            else:
                width, height = float(detection["bbox"][2]), float(detection["bbox"][3])
                unmatched_rows.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "shorter_side_bin": pixel_bin(min(width, height) * ratio),
                    "height_bin": pixel_bin(height * ratio),
                    "area_class": area_bin(width * height),
                })
        for index, annotation in enumerate(ground_truth):
            width, height = float(annotation["bbox"][2]), float(annotation["bbox"][3])
            gt_rows.append({
                "image_id": image_id,
                "category_id": category_id,
                "matched": int(index in matched),
                "shorter_side_bin": pixel_bin(min(width, height) * ratio),
                "height_bin": pixel_bin(height * ratio),
                "area_class": area_bin(float(annotation.get("area", width * height))),
            })
    return gt_rows, unmatched_rows


def grouped_size_rows(gt_rows: list[dict[str, Any]], unmatched: list[dict[str, Any]],
                      field: str, order: list[str], resolution: int) -> list[dict[str, Any]]:
    gt_count = Counter(row[field] for row in gt_rows)
    true_positive = Counter(row[field] for row in gt_rows if row["matched"])
    false_positive = Counter(row[field] for row in unmatched)
    rows: list[dict[str, Any]] = []
    for label in order:
        count = gt_count[label]
        tp = true_positive[label]
        fp = false_positive[label]
        rows.append({
            "resolution": resolution,
            "bin_kind": field,
            "bin": label,
            "ground_truth_count": count,
            "true_positive_count": tp,
            "false_negative_count": count - tp,
            "unmatched_detection_count": fp,
            "recall_iou50": tp / count if count else "",
            "precision_iou50_diagnostic": tp / (tp + fp) if tp + fp else "",
            "confidence_threshold": 0.001,
            "matching": "greedy-score-descending-same-class-iou>=0.50",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.001)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        coco = COCO(str(args.annotations))
        result = coco.loadRes(str(args.predictions))
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (args.output_dir / "evaluator.log").write_text(capture.getvalue(), encoding="utf-8")

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    metrics = [{
        "resolution": args.resolution,
        "quantization_arm": "Q0",
        "images": len(evaluator.params.imgIds),
        "failures": 0,
        "prediction_count": len(predictions),
        "prediction_sha256": sha256(args.predictions),
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]),
        "ap_large": float(evaluator.stats[5]),
        "ar_100": float(evaluator.stats[8]),
        "mean_precision_grid": valid_mean(precision[:, :, :, 0, 2]),
        "mean_recall_grid": valid_mean(recall[:, :, 0, 2]),
    }]
    write_tsv(args.output_dir / "results.tsv", metrics)

    counts = Counter(annotation["category_id"] for annotation in coco.dataset["annotations"])
    gt_rows, unmatched = match_detections(coco, predictions, args.confidence, args.resolution)
    matched_by_class = Counter(row["category_id"] for row in gt_rows if row["matched"])
    per_class: list[dict[str, Any]] = []
    for index, category_id in enumerate(evaluator.params.catIds):
        count = counts[category_id]
        per_class.append({
            "resolution": args.resolution,
            "category_id": category_id,
            "class_name": coco.cats[category_id]["name"],
            "instances": count,
            "ap50_95": valid_mean(precision[:, :, index, 0, 2]),
            "ap50": valid_mean(precision[0, :, index, 0, 2]),
            "recall_iou50_conf001": matched_by_class[category_id] / count if count else "",
        })
    write_tsv(args.output_dir / "per_class.tsv", per_class)

    pixel_order = [pixel_bin(lower) for lower, _ in PIXEL_BINS]
    size_rows = grouped_size_rows(gt_rows, unmatched, "area_class",
                                  ["small", "medium", "large"], args.resolution)
    size_rows += grouped_size_rows(gt_rows, unmatched, "shorter_side_bin",
                                   pixel_order, args.resolution)
    size_rows += grouped_size_rows(gt_rows, unmatched, "height_bin", pixel_order, args.resolution)
    for row in size_rows:
        row["confidence_threshold"] = args.confidence
    write_tsv(args.output_dir / "size_bins.tsv", size_rows)
    print(json.dumps(metrics[0], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
