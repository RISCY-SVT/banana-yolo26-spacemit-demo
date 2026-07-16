#!/usr/bin/env python3
"""Measure COCO IoU=0.50 detection coverage by exact 640-letterboxed object size."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


EXPECTED_PREDICTION_SHA256 = "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"


@dataclass
class GroundTruth:
    annotation_id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    letterbox_width: float
    letterbox_height: float
    matched: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


SIZE_BINS = (
    ("0-8", 0.0, 8.0),
    ("8-12", 8.0, 12.0),
    ("12-16", 12.0, 16.0),
    ("16-24", 16.0, 24.0),
    ("24-32", 24.0, 32.0),
    ("32-48", 32.0, 48.0),
    ("48-64", 48.0, 64.0),
    ("64-96", 64.0, 96.0),
    (">=96", 96.0, float("inf")),
)


def numeric_bin(value: float) -> str:
    return next(name for name, lower, upper in SIZE_BINS if lower <= value < upper)


def area_bin(area: float) -> str:
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def keys_for(gt: GroundTruth) -> tuple[tuple[str, str], ...]:
    return (
        ("coco_area", area_bin(gt.area)),
        ("shorter_side_letterbox_px", numeric_bin(min(gt.letterbox_width, gt.letterbox_height))),
        ("box_height_letterbox_px", numeric_bin(gt.letterbox_height)),
    )


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-bins", type=Path, required=True)
    parser.add_argument("--output-classes", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    prediction_sha = sha256(args.predictions)
    if prediction_sha != EXPECTED_PREDICTION_SHA256:
        raise ValueError(f"unexpected prediction SHA-256: {prediction_sha}")
    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in dataset["images"]}
    categories = {int(row["id"]): str(row["name"]) for row in dataset["categories"]}

    grouped_gt: dict[tuple[int, int], list[GroundTruth]] = collections.defaultdict(list)
    all_gt: list[GroundTruth] = []
    for row in dataset["annotations"]:
        if int(row.get("iscrowd", 0)) != 0:
            continue
        image = images[int(row["image_id"])]
        ratio = min(640.0 / float(image["width"]), 640.0 / float(image["height"]))
        bbox = tuple(float(value) for value in row["bbox"])
        gt = GroundTruth(
            annotation_id=int(row["id"]),
            image_id=int(row["image_id"]),
            category_id=int(row["category_id"]),
            bbox=bbox,
            area=float(row.get("area", bbox[2] * bbox[3])),
            letterbox_width=bbox[2] * ratio,
            letterbox_height=bbox[3] * ratio,
        )
        grouped_gt[(gt.image_id, gt.category_id)].append(gt)
        all_gt.append(gt)

    filtered = [row for row in predictions if float(row["score"]) >= args.confidence]
    filtered.sort(key=lambda row: (-float(row["score"]), int(row["image_id"]), int(row["category_id"])))
    associated: collections.Counter[tuple[str, str]] = collections.Counter()
    true_positive: collections.Counter[tuple[str, str]] = collections.Counter()
    class_predictions: collections.Counter[int] = collections.Counter()
    class_tp: collections.Counter[int] = collections.Counter()
    unmatched_predictions = 0

    for prediction in filtered:
        group = grouped_gt.get((int(prediction["image_id"]), int(prediction["category_id"])), [])
        pred_bbox = tuple(float(value) for value in prediction["bbox"])
        ranked = sorted(((iou(pred_bbox, gt.bbox), gt) for gt in group),
                        key=lambda item: (-item[0], item[1].annotation_id))
        best_any = ranked[0][1] if ranked else None
        best_unmatched = next(((score, gt) for score, gt in ranked if not gt.matched), None)
        class_id = int(prediction["category_id"])
        class_predictions[class_id] += 1
        if best_any is None:
            unmatched_predictions += 1
            continue
        for key in keys_for(best_any):
            associated[key] += 1
        if best_unmatched is not None and best_unmatched[0] >= 0.50:
            matched_gt = best_unmatched[1]
            matched_gt.matched = True
            class_tp[class_id] += 1
            for key in keys_for(matched_gt):
                true_positive[key] += 1

    gt_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    matched_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    class_gt: collections.Counter[int] = collections.Counter()
    class_matched: collections.Counter[int] = collections.Counter()
    for gt in all_gt:
        class_gt[gt.category_id] += 1
        class_matched[gt.category_id] += int(gt.matched)
        for key in keys_for(gt):
            gt_counts[key] += 1
            matched_counts[key] += int(gt.matched)

    dimension_order = {
        "coco_area": ["small", "medium", "large"],
        "shorter_side_letterbox_px": [row[0] for row in SIZE_BINS],
        "box_height_letterbox_px": [row[0] for row in SIZE_BINS],
    }
    bin_rows: list[dict[str, object]] = []
    for dimension, bins in dimension_order.items():
        for name in bins:
            key = (dimension, name)
            gt_count = gt_counts[key]
            tp = matched_counts[key]
            prediction_count = associated[key]
            bin_rows.append({
                "dimension": dimension,
                "bin": name,
                "gt_count": gt_count,
                "true_positive": tp,
                "recall": f"{tp / gt_count:.9f}" if gt_count else "",
                "associated_predictions": prediction_count,
                "precision": f"{true_positive[key] / prediction_count:.9f}" if prediction_count else "",
                "iou_threshold": "0.50",
                "confidence_threshold": f"{args.confidence:.6f}",
            })
    write_rows(args.output_bins, bin_rows, list(bin_rows[0]))

    class_rows: list[dict[str, object]] = []
    for category_id, name in sorted(categories.items(), key=lambda item: item[0]):
        gt_count = class_gt[category_id]
        prediction_count = class_predictions[category_id]
        tp = class_tp[category_id]
        class_rows.append({
            "category_id": category_id,
            "class": name,
            "gt_count": gt_count,
            "true_positive": tp,
            "recall": f"{class_matched[category_id] / gt_count:.9f}" if gt_count else "",
            "predictions": prediction_count,
            "precision": f"{tp / prediction_count:.9f}" if prediction_count else "",
            "iou_threshold": "0.50",
            "confidence_threshold": f"{args.confidence:.6f}",
        })
    write_rows(args.output_classes, class_rows, list(class_rows[0]))
    print(json.dumps({
        "confidence_threshold": args.confidence,
        "ground_truth_noncrowd": len(all_gt),
        "predictions_filtered": len(filtered),
        "prediction_sha256": prediction_sha,
        "true_positive": sum(class_tp.values()),
        "unmatched_prediction_without_same_class_gt": unmatched_predictions,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
