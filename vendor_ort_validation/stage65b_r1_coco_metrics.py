#!/usr/bin/env python3
"""Evaluate full or explicitly listed COCO prediction surfaces."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_mean(values: np.ndarray) -> float:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else float("nan")


def image_ids(path: Path | None, coco: COCO) -> list[int]:
    if path is None:
        return sorted(coco.imgs)
    result: list[int] = []
    seen: set[int] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        image_id = int(Path(value).stem)
        if image_id in seen:
            raise ValueError(f"duplicate image id in list: {image_id}")
        if image_id not in coco.imgs:
            raise ValueError(f"image id absent from annotations: {image_id}")
        seen.add(image_id)
        result.append(image_id)
    if not result:
        raise ValueError("empty image list")
    return result


def finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=False)

    predictions = json.loads(options.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("prediction payload must be a JSON array")
    non_finite_predictions = sum(
        not finite_numbers(prediction) for prediction in predictions
    )
    if non_finite_predictions:
        raise ValueError(
            f"prediction payload has {non_finite_predictions} non-finite rows"
        )

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        coco = COCO(str(options.annotations))
        selected_ids = image_ids(options.image_list, coco)
        selected_set = set(selected_ids)
        prediction_ids = {int(item["image_id"]) for item in predictions}
        unexpected = sorted(prediction_ids - selected_set)
        if unexpected:
            raise ValueError(
                f"predictions contain {len(unexpected)} unselected image ids"
            )
        result = coco.loadRes(str(options.predictions))
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = selected_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (options.output_dir / "evaluator.log").write_text(
        capture.getvalue(), encoding="utf-8"
    )

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    result_row = {
        "surface": options.surface,
        "images": len(selected_ids),
        "images_with_predictions": len(prediction_ids),
        "failures": 0,
        "non_finite_predictions": non_finite_predictions,
        "prediction_count": len(predictions),
        "prediction_sha256": sha256(options.predictions),
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]),
        "ap_large": float(evaluator.stats[5]),
        "ar_1": float(evaluator.stats[6]),
        "ar_10": float(evaluator.stats[7]),
        "ar_100": float(evaluator.stats[8]),
        "ar_small": float(evaluator.stats[9]),
        "ar_medium": float(evaluator.stats[10]),
        "ar_large": float(evaluator.stats[11]),
        "mean_precision_grid": valid_mean(precision[:, :, :, 0, 2]),
        "mean_recall_grid": valid_mean(recall[:, :, 0, 2]),
    }
    write_tsv(options.output_dir / "results.tsv", [result_row])

    selected_annotations = coco.loadAnns(
        coco.getAnnIds(imgIds=selected_ids, iscrowd=None)
    )
    instance_counts = Counter(
        int(item["category_id"])
        for item in selected_annotations
        if int(item.get("iscrowd", 0)) == 0
    )
    per_class: list[dict[str, Any]] = []
    for index, category_id in enumerate(evaluator.params.catIds):
        per_class.append(
            {
                "surface": options.surface,
                "category_id": category_id,
                "class_name": coco.cats[category_id]["name"],
                "instances": instance_counts[category_id],
                "ap50_95": valid_mean(precision[:, :, index, 0, 2]),
                "ap50": valid_mean(precision[0, :, index, 0, 2]),
                "ar_100": valid_mean(recall[:, index, 0, 2]),
            }
        )
    write_tsv(options.output_dir / "per_class.tsv", per_class)

    size_rows = [
        {
            "surface": options.surface,
            "size_bin": size,
            "ap50_95": float(evaluator.stats[ap_index]),
            "ar_100": float(evaluator.stats[ar_index]),
        }
        for size, ap_index, ar_index in (
            ("small", 3, 9),
            ("medium", 4, 10),
            ("large", 5, 11),
        )
    ]
    write_tsv(options.output_dir / "size_bins.tsv", size_rows)
    print(json.dumps(result_row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
