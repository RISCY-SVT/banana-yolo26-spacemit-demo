#!/usr/bin/env python3
"""Evaluate one Stage63 predictor output with the exact processed image set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processed_image_ids(timing_path: Path) -> list[int]:
    with timing_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    ids = [int(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("timing TSV contains duplicate image IDs")
    return ids


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--per-class", required=True, type=Path)
    args = parser.parse_args()

    image_ids = processed_image_ids(args.timing)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a JSON array")

    coco = COCO(str(args.annotations))
    result = coco.loadRes(predictions)
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats.tolist()
    precision = evaluator.eval["precision"]
    categories = coco.loadCats(evaluator.params.catIds)
    per_class: list[dict[str, Any]] = []
    for class_index, category in enumerate(categories):
        values = precision[:, :, class_index, 0, 2]
        values = values[values > -1]
        per_class.append(
            {
                "category_id": category["id"],
                "name": category["name"],
                "ap50_95": float(np.mean(values)) if values.size else None,
            }
        )

    summary = {
        "annotations_sha256": sha256(args.annotations),
        "predictions_sha256": sha256(args.predictions),
        "timing_sha256": sha256(args.timing),
        "image_count": len(image_ids),
        "prediction_count": len(predictions),
        "map50_95": stats[0],
        "map50": stats[1],
        "map75": stats[2],
        "ap_small": stats[3],
        "ap_medium": stats[4],
        "ap_large": stats[5],
        "ar_maxdet1": stats[6],
        "ar_maxdet10": stats[7],
        "ar_maxdet100": stats[8],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.per_class, per_class, ["category_id", "name", "ap50_95"])


if __name__ == "__main__":
    main()
