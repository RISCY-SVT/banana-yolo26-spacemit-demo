#!/usr/bin/env python3
"""Evaluate Stage 52 COCO predictions and a paired image-level quality bootstrap."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(coco: COCO, prediction_path: Path) -> tuple[COCOeval, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        predictions = coco.loadRes(str(prediction_path))
        evaluator = COCOeval(coco, predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator, output.getvalue()


def valid_mean(values: np.ndarray) -> float:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else float("nan")


def global_metrics(evaluator: COCOeval) -> dict[str, float]:
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    return {
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]),
        "ap_large": float(evaluator.stats[5]),
        "ar_100": float(evaluator.stats[8]),
        "mean_precision_grid": valid_mean(precision[:, :, :, 0, 2]),
        "mean_recall_grid": valid_mean(recall[:, :, 0, 2]),
    }


def per_class(coco: COCO, evaluator: COCOeval) -> list[dict[str, object]]:
    precision = evaluator.eval["precision"]
    category_ids = evaluator.params.catIds
    counts = Counter(annotation["category_id"] for annotation in coco.dataset["annotations"])
    rows: list[dict[str, object]] = []
    for index, category_id in enumerate(category_ids):
        category = coco.cats[category_id]
        rows.append({
            "category_id": category_id,
            "class_name": category["name"],
            "instances": counts[category_id],
            "ap50_95": valid_mean(precision[:, :, index, 0, 2]),
            "ap50": valid_mean(precision[0, :, index, 0, 2]),
        })
    return rows


def image_quality(evaluator: COCOeval) -> dict[int, float]:
    """Return per-image IoU-averaged F1 over all categories.

    This is a paired bootstrap diagnostic, not a substitute for global COCO mAP.
    """
    image_ids = evaluator.params.imgIds
    category_count = len(evaluator.params.catIds)
    image_count = len(image_ids)
    area_count = len(evaluator.params.areaRng)
    by_image: dict[int, list[object]] = defaultdict(list)
    for category_index in range(category_count):
        base = category_index * area_count * image_count
        for image_index, image_id in enumerate(image_ids):
            entry = evaluator.evalImgs[base + image_index]
            if entry is not None:
                by_image[image_id].append(entry)
    result: dict[int, float] = {}
    thresholds = len(evaluator.params.iouThrs)
    for image_id in image_ids:
        true_positive = np.zeros(thresholds, dtype=np.float64)
        false_positive = np.zeros(thresholds, dtype=np.float64)
        ground_truth = 0
        for entry in by_image.get(image_id, []):
            ground_truth += int(np.count_nonzero(np.logical_not(entry["gtIgnore"])))
            matches = entry["dtMatches"]
            ignored = entry["dtIgnore"]
            true_positive += np.count_nonzero((matches > 0) & np.logical_not(ignored), axis=1)
            false_positive += np.count_nonzero((matches == 0) & np.logical_not(ignored), axis=1)
        false_negative = np.maximum(0.0, ground_truth - true_positive)
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1 = np.divide(2.0 * true_positive, denominator,
                       out=np.ones_like(denominator), where=denominator != 0)
        result[image_id] = float(f1.mean())
    return result


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty TSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--k1x", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=52001)
    args = parser.parse_args()
    if args.resamples < 2000:
        raise ValueError("Stage 52 requires at least 2000 bootstrap resamples")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        coco = COCO(str(args.annotations))
    k1x_eval, k1x_log = evaluate(coco, args.k1x)
    semantic_eval, semantic_log = evaluate(coco, args.semantic)
    k1x_metrics = global_metrics(k1x_eval)
    semantic_metrics = global_metrics(semantic_eval)
    (args.output_dir / "evaluator.log").write_text(
        "COCO_LOAD\n" + capture.getvalue() + "\nK1X\n" + k1x_log +
        "\nSEMANTIC_INT8\n" + semantic_log, encoding="utf-8")

    surfaces = []
    for name, path, metrics in (
        ("legacy_semantic_int8", args.semantic, semantic_metrics),
        ("k1x_int8_v1", args.k1x, k1x_metrics),
    ):
        row: dict[str, object] = {
            "surface": name,
            "prediction_json": str(path),
            "prediction_sha256": sha256(path),
        }
        row.update(metrics)
        surfaces.append(row)
    write_tsv(args.output_dir / "results.tsv", surfaces)
    write_tsv(args.output_dir / "per_class.tsv", per_class(coco, k1x_eval))

    k1x_quality = image_quality(k1x_eval)
    semantic_quality = image_quality(semantic_eval)
    image_ids = np.asarray(sorted(set(k1x_quality) & set(semantic_quality)), dtype=np.int64)
    if image_ids.size != 5000:
        raise ValueError(f"paired image set is {image_ids.size}, expected 5000")
    difference = np.asarray(
        [k1x_quality[int(image_id)] - semantic_quality[int(image_id)] for image_id in image_ids],
        dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    samples = np.empty(args.resamples, dtype=np.float64)
    for index in range(args.resamples):
        selected = rng.integers(0, image_ids.size, size=image_ids.size)
        samples[index] = float(difference[selected].mean())
    lower, upper = np.quantile(samples, [0.025, 0.975])
    bootstrap_rows = [{
        "resample": index,
        "seed": args.seed,
        "metric": "per_image_iou_averaged_f1_delta",
        "delta_k1x_minus_semantic": value,
    } for index, value in enumerate(samples)]
    write_tsv(args.output_dir / "bootstrap.tsv", bootstrap_rows)

    summary = {
        "bootstrap_metric": "per_image_iou_averaged_f1_delta",
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_delta_mean": float(samples.mean()),
        "bootstrap_delta_ci95_low": float(lower),
        "bootstrap_delta_ci95_high": float(upper),
        "k1x": k1x_metrics,
        "semantic": semantic_metrics,
        "map50_95_delta": k1x_metrics["map50_95"] - semantic_metrics["map50_95"],
        "prediction_count_k1x": len(json.loads(args.k1x.read_text(encoding="utf-8"))),
        "prediction_count_semantic": len(json.loads(args.semantic.read_text(encoding="utf-8"))),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
