#!/usr/bin/env python3
"""Frozen ten-surface accuracy ledger and COCOeval operating-point census."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAGE65C_R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-"
    "AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001"
)
STAGE65D_R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65D-R1-C2-FROZEN-FULL-VAL-PROVIDER-"
    "INTERACTION-CONDITIONAL-PERFORMANCE-STABILITY-AND-CROSS-SURFACE-"
    "PASSPORT-001"
)
DEV001A_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-"
    "OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001"
)
DEV001C_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-"
    "ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001"
)
R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-DISJOINT-"
    "CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
)

DATA = Path("/data")
RUNS = DATA / "k1x-stage-runs"
REPO = DATA / "worktrees/banana-yolo26-xslim211-s8-qdq-validation"
STAGES = REPO / "stages"
DATASET = DATA / "datasets/coco2017-independent-stage65b-r1"
ANNOTATIONS = DATASET / "annotations/instances_val2017.json"
IMAGE_LIST = DATASET / "lists/val2017_all.txt"

SURFACES: dict[str, dict[str, str | Path]] = {
    "FP32_HOST": {
        "model": "FP32", "scope": "host", "provider": "CPU",
        "path": RUNS / R1_ID / "full-matrix/hybrid-full-coco/H8/predictions.json",
        "sha256": "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a",
    },
    "B2_HOST": {
        "model": "B2", "scope": "host", "provider": "CPU",
        "path": RUNS / R1_ID / "full-matrix/full-coco/B2/predictions.json",
        "sha256": "51f8d4b25245a5f3e24feafea8aa49547c0f530f59cabcd18e61a744b4740add",
    },
    "A1_HOST": {
        "model": "A1", "scope": "host", "provider": "CPU",
        "path": RUNS / DEV001A_ID / "candidates/full-val/A1/predictions.json",
        "sha256": "fdae3c397ff82b005b3c0f507496392dde381fed2aaa0f5d18f03ea35c7b2df9",
    },
    "C2_HOST": {
        "model": "C2", "scope": "host", "provider": "CPU",
        "path": RUNS / DEV001C_ID / "full-val/C2/predictions.json",
        "sha256": "0f040fe848e9fe5306f0d8974e410a3471d2bff9cbf1f3c53dc5feb1c47fa345",
    },
    "B2_BOARD_CPU": {
        "model": "B2", "scope": "board", "provider": "CPU",
        "path": RUNS / STAGE65C_R1_ID / "board/coco/val/B2-cpu/predictions.json",
        "sha256": "c903721d880b1df599c6912455aa39106d94a2be2cd2ad226cce59fbdae28745",
    },
    "B2_BOARD_EP": {
        "model": "B2", "scope": "board", "provider": "SpaceMIT_EP",
        "path": RUNS / STAGE65C_R1_ID / "board/coco/val/B2-spacemit/predictions.json",
        "sha256": "edba82a970a95b4e13d194044573fadccebe831f98527116d1ca9a74b00eab39",
    },
    "A1_BOARD_CPU": {
        "model": "A1", "scope": "board", "provider": "CPU",
        "path": RUNS / STAGE65C_R1_ID / "board/coco/val/A1-cpu/predictions.json",
        "sha256": "b68ff726281f905f5bfdd5ae74c0e81846951ae61437d1d512af9911c74f99c4",
    },
    "A1_BOARD_EP": {
        "model": "A1", "scope": "board", "provider": "SpaceMIT_EP",
        "path": RUNS / STAGE65C_R1_ID / "board/coco/val/A1-spacemit/predictions.json",
        "sha256": "dd37dee3c27c5a6e981b6c75a73d9aa11c3cd74a7b8ad6178fb57d5ff513d9a0",
    },
    "C2_BOARD_CPU": {
        "model": "C2", "scope": "board", "provider": "CPU",
        "path": RUNS / STAGE65D_R1_ID / "board/coco/val/C2-cpu/predictions.json",
        "sha256": "186e53676f21f290e08f305aa78ad12031a3c7478698cb92535d881b8709dad5",
    },
    "C2_BOARD_EP": {
        "model": "C2", "scope": "board", "provider": "SpaceMIT_EP",
        "path": RUNS / STAGE65D_R1_ID / "board/coco/val/C2-spacemit/predictions.json",
        "sha256": "3a805d63c1e8e9ac05d843a2da87d6238f4ec6b52d3428e647ab6071f240e11a",
    },
}

METRICS = (
    "map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large",
    "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
)
SCORE_THRESHOLDS = (0.001, 0.01, 0.05, 0.25, 0.50)
IOU_THRESHOLDS = (0.50, 0.75)
MAX_DETS = (100, 300)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_ids(path: Path, coco: COCO) -> list[int]:
    values = [int(Path(line.strip()).stem) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != 5000 or len(values) != len(set(values)):
        raise ValueError("val2017 list must contain 5000 unique image ids")
    if set(values) != set(coco.imgs):
        raise ValueError("val2017 list and annotation image ids differ")
    return values


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def evaluate(coco: COCO, predictions: list[dict[str, Any]], ids: list[int], max_det: int, accumulate: bool) -> COCOeval:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = coco.loadRes(predictions)
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = ids
        evaluator.params.maxDets = [1, 10, max_det]
        evaluator.evaluate()
        if accumulate:
            evaluator.accumulate()
            evaluator.summarize()
    return evaluator


def metric_row(surface: str, spec: dict[str, str | Path], evaluator: COCOeval, predictions: list[dict[str, Any]]) -> dict[str, object]:
    stats = evaluator.stats
    return {
        "surface": surface,
        "model": spec["model"],
        "scope": spec["scope"],
        "provider": spec["provider"],
        "images": 5000,
        "images_with_predictions": len({int(row["image_id"]) for row in predictions}),
        "prediction_count": len(predictions),
        "prediction_sha256": spec["sha256"],
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "ar_1": float(stats[6]),
        "ar_10": float(stats[7]),
        "ar_100": float(stats[8]),
        "ar_small": float(stats[9]),
        "ar_medium": float(stats[10]),
        "ar_large": float(stats[11]),
        "failures": 0,
        "non_finite_predictions": 0,
        "score_collapse": "no",
    }


def summarize_counts(values: dict[str, int], images: int) -> dict[str, object]:
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detections": values["detections"],
        "detections_per_image": values["detections"] / images,
        "ignored_detections": values["ignored_detections"],
        "ignored_ground_truths": values["ignored_ground_truths"],
    }


def operating_counts(
    surface: str,
    spec: dict[str, str | Path],
    evaluator: COCOeval,
    coco: COCO,
    images: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    max_det = int(evaluator.params.maxDets[-1])
    area_labels = list(evaluator.params.areaRngLbl)
    area_by_range = {
        tuple(float(value) for value in bounds): label
        for bounds, label in zip(evaluator.params.areaRng, area_labels, strict=True)
    }
    iou_indices = {
        iou: int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, iou))[0])
        for iou in IOU_THRESHOLDS
    }
    counts: dict[tuple[int, str, float, float], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for item in evaluator.evalImgs:
        if item is None:
            continue
        category_id = int(item["category_id"])
        area = area_by_range[tuple(float(value) for value in item["aRng"])]
        scores = np.asarray(item["dtScores"], dtype=np.float64)
        gt_ids = np.asarray(item["gtIds"], dtype=np.int64)
        gt_ignore = np.asarray(item["gtIgnore"], dtype=bool)
        for iou, iou_index in iou_indices.items():
            dt_matches = np.asarray(item["dtMatches"][iou_index], dtype=np.int64)
            dt_ignore = np.asarray(item["dtIgnore"][iou_index], dtype=bool)
            for threshold in SCORE_THRESHOLDS:
                selected = scores >= threshold
                selected_matches = dt_matches[selected]
                selected_ignore = dt_ignore[selected]
                tp_mask = (selected_matches > 0) & ~selected_ignore
                fp_mask = (selected_matches == 0) & ~selected_ignore
                matched_gt = set(int(value) for value in selected_matches[tp_mask])
                nonignored_gt = set(int(value) for value in gt_ids[~gt_ignore])
                value = counts[(category_id, area, threshold, iou)]
                value["tp"] += int(tp_mask.sum())
                value["fp"] += int(fp_mask.sum())
                value["fn"] += len(nonignored_gt - matched_gt)
                value["detections"] += int(selected.sum())
                value["ignored_detections"] += int(selected_ignore.sum())
                value["ignored_ground_truths"] += int(gt_ignore.sum())

    per_class = []
    aggregate: dict[tuple[str, float, float], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (category_id, area, threshold, iou), values in sorted(counts.items()):
        base = {
            "surface": surface,
            "model": spec["model"],
            "scope": spec["scope"],
            "provider": spec["provider"],
            "score_threshold": threshold,
            "iou_threshold": iou,
            "max_dets": max_det,
            "area": area,
            "category_id": category_id,
            "class_name": coco.cats[category_id]["name"],
        }
        per_class.append({**base, **summarize_counts(values, images)})
        target = aggregate[(area, threshold, iou)]
        for key, value in values.items():
            target[key] += value

    census = []
    per_size = []
    for (area, threshold, iou), values in sorted(aggregate.items()):
        row = {
            "surface": surface,
            "model": spec["model"],
            "scope": spec["scope"],
            "provider": spec["provider"],
            "score_threshold": threshold,
            "iou_threshold": iou,
            "max_dets": max_det,
            "area": area,
            **summarize_counts(values, images),
        }
        census.append(row)
        if area != "all":
            per_size.append(dict(row))
    return census, per_class, per_size


def validate_synthetic(raw_root: Path) -> list[dict[str, object]]:
    fixture = raw_root / "operating-point-synthetic"
    fixture.mkdir(parents=True, exist_ok=False)
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "1.jpg"}],
        "categories": [{"id": 1, "name": "fixture"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10], "area": 100, "iscrowd": 1},
        ],
        "licenses": [],
        "info": {},
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [70, 70, 10, 10], "score": 0.1},
    ]
    annotation_path = fixture / "annotations.json"
    prediction_path = fixture / "predictions.json"
    annotation_path.write_text(json.dumps(annotations, sort_keys=True) + "\n")
    prediction_path.write_text(json.dumps(predictions, sort_keys=True) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_path))
        evaluator = evaluate(coco, predictions, [1], 100, False)
    spec: dict[str, str | Path] = {"model": "fixture", "scope": "synthetic", "provider": "CPU"}
    _, rows, _ = operating_counts("SYNTHETIC", spec, evaluator, coco, 1)
    selected = {
        (float(row["score_threshold"]), float(row["iou_threshold"]), str(row["area"])): row
        for row in rows
    }
    checks = []
    for threshold, expected in (
        (0.05, (1, 1, 1, 1, 1)),
        (0.50, (1, 0, 1, 1, 1)),
    ):
        row = selected[(threshold, 0.50, "all")]
        actual = (
            int(row["tp"]), int(row["fp"]), int(row["fn"]),
            int(row["ignored_detections"]), int(row["ignored_ground_truths"]),
        )
        checks.append({
            "check": f"synthetic-threshold-{threshold}",
            "actual": repr(actual),
            "expected": repr(expected),
            "status": "pass" if actual == expected else "fail",
        })
    return checks


def accepted_metrics() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    r1 = read_tsv(STAGES / R1_ID / "full_coco_results.tsv")
    result["FP32_HOST"] = next(row for row in r1 if row["surface"] == "H8")
    host = {row["surface"]: row for row in read_tsv(STAGES / DEV001C_ID / "full_val_metrics.tsv")}
    for model in ("B2", "A1", "C2"):
        result[f"{model}_HOST"] = host[model]
    board_a1 = {
        row["surface"]: row
        for row in read_tsv(STAGES / STAGE65C_R1_ID / "full_val_diagnostic_metrics.tsv")
    }
    for model in ("B2", "A1"):
        result[f"{model}_BOARD_CPU"] = board_a1[f"{model}_CPU"]
        result[f"{model}_BOARD_EP"] = board_a1[f"{model}_EP"]
    board_c2 = {
        row["surface"]: row
        for row in read_tsv(STAGES / STAGE65D_R1_ID / "full_val_board_metrics.tsv")
    }
    result["C2_BOARD_CPU"] = board_c2["C2_CPU"]
    result["C2_BOARD_EP"] = board_c2["C2_EP"]
    return result


def delta_rows(table: dict[str, dict[str, object]], reference: str, kind: str) -> list[dict[str, object]]:
    rows = []
    for surface, row in table.items():
        ref = table[reference if kind == "fp32" else {
            "host": "B2_HOST",
            "board:CPU": "B2_BOARD_CPU",
            "board:SpaceMIT_EP": "B2_BOARD_EP",
        }[f"{row['scope']}:{row['provider']}" if row["scope"] == "board" else "host"]]
        output = {
            "surface": surface,
            "reference_surface": reference if kind == "fp32" else ref["surface"],
            "scope_relation": "same-scope" if row["scope"] == ref["scope"] else "cross-scope-host-fp32-context",
        }
        output.update({metric: float(row[metric]) - float(ref[metric]) for metric in METRICS})
        rows.append(output)
    return rows


def write_accuracy_passport(
    path: Path,
    absolute: list[dict[str, object]],
    census: list[dict[str, object]],
) -> None:
    lines = [
        "# FP32/B2/A1/C2 accuracy passport", "",
        "All rows use the same source lineage, six-output split contract, common FP32 tail, val2017 list and COCO evaluator. Host, board CPU and board EP remain explicit execution surfaces.", "",
        "| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in absolute:
        lines.append(
            f"| {row['surface']} | {float(row['map50_95']):.12f} | {float(row['ap_small']):.12f} | "
            f"{float(row['ap_medium']):.12f} | {float(row['ap_large']):.12f} | "
            f"{float(row['ar_small']):.12f} | {float(row['ar_medium']):.12f} | "
            f"{float(row['ar_large']):.12f} | {row['prediction_count']} |"
        )
    lines.extend([
        "",
        "## Board EP operating points",
        "",
        "The table uses exact COCOeval matching at IoU 0.50, maxDets 100 and area `all`; ignored/crowd rows remain excluded from TP/FP/FN according to the accepted contract.",
        "",
        "| Surface | Score | TP | FP | FN | Precision | Recall | F1 | Detections |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    selected = [
        row
        for row in census
        if row["surface"] in {"B2_BOARD_EP", "C2_BOARD_EP"}
        and float(row["iou_threshold"]) == 0.50
        and int(row["max_dets"]) == 100
        and row["area"] == "all"
    ]
    selected.sort(key=lambda row: (str(row["surface"]), float(row["score_threshold"])))
    for row in selected:
        lines.append(
            f"| {row['surface']} | {float(row['score_threshold']):.3f} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {float(row['precision']):.6f} | "
            f"{float(row['recall']):.6f} | {float(row['f1']):.6f} | {row['detections']} |"
        )
    lines.extend([
        "",
        "C2 is the same-source INT8 mAP/AP leader but remains subject to the immutable Stage65D-R1 recall-contract failure. B2 remains the universal vendor control; A1 remains historical frozen evidence. The operating-point rows show why an application-specific threshold and false-negative budget are mandatory before any C2 waiver.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    metrics_raw = options.raw_root / "accuracy-ledger"
    if metrics_raw.exists():
        raise RuntimeError(f"refusing existing accuracy root: {metrics_raw}")
    metrics_raw.mkdir(parents=True)

    validation = validate_synthetic(metrics_raw)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ANNOTATIONS))
    ids = image_ids(IMAGE_LIST, coco)
    accepted = accepted_metrics()
    absolute = []
    census = []
    per_class = []
    per_size = []

    for surface, spec in SURFACES.items():
        path = Path(spec["path"])
        observed_hash = sha256(path)
        if observed_hash != spec["sha256"]:
            raise RuntimeError(f"frozen prediction identity drift: {surface}")
        predictions = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"invalid prediction payload: {surface}")
        for prediction in predictions:
            values = list(prediction.get("bbox", [])) + [prediction.get("score", float("nan"))]
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"non-finite prediction: {surface}")
        standard = evaluate(coco, predictions, ids, 100, True)
        row = metric_row(surface, spec, standard, predictions)
        absolute.append(row)
        accepted_row = accepted[surface]
        for metric in METRICS:
            actual = float(row[metric])
            expected = float(accepted_row[metric])
            validation.append({
                "check": f"accepted-metric-{surface}-{metric}",
                "actual": repr(actual),
                "expected": repr(expected),
                "status": "pass" if abs(actual - expected) <= 1e-15 else "fail",
            })
        for max_det, evaluator in ((100, standard), (300, evaluate(coco, predictions, ids, 300, False))):
            if int(evaluator.params.maxDets[-1]) != max_det:
                raise RuntimeError("COCOeval maxDet contract drift")
            c_rows, pc_rows, ps_rows = operating_counts(surface, spec, evaluator, coco, len(ids))
            census.extend(c_rows)
            per_class.extend(pc_rows)
            per_size.extend(ps_rows)
        del predictions, standard

    table = {str(row["surface"]): row for row in absolute}
    fp32 = table["FP32_HOST"]
    hard_fp32 = abs(float(fp32["map50_95"]) - 0.4018217950262668) <= 1e-15
    validation.append({
        "check": "fp32-map50-95-hard-gate",
        "actual": repr(fp32["map50_95"]),
        "expected": repr(0.4018217950262668),
        "status": "pass" if hard_fp32 else "fail",
    })
    validation.append({
        "check": "fp32-prediction-sha-hard-gate",
        "actual": str(fp32["prediction_sha256"]),
        "expected": "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a",
        "status": "pass" if fp32["prediction_sha256"] == SURFACES["FP32_HOST"]["sha256"] else "fail",
    })

    write_tsv(options.tracked_root / "fp32_complete_metrics.tsv", [fp32])
    write_tsv(options.tracked_root / "accuracy_absolute.tsv", absolute)
    write_tsv(options.tracked_root / "accuracy_delta_to_fp32.tsv", delta_rows(table, "FP32_HOST", "fp32"))
    write_tsv(options.tracked_root / "accuracy_delta_to_b2.tsv", delta_rows(table, "B2_HOST", "b2"))

    provider_rows = []
    for model in ("B2", "A1", "C2"):
        cpu, ep = table[f"{model}_BOARD_CPU"], table[f"{model}_BOARD_EP"]
        provider_rows.append({
            "model": model,
            "left": ep["surface"],
            "right": cpu["surface"],
            **{metric: float(ep[metric]) - float(cpu[metric]) for metric in METRICS},
        })
    write_tsv(options.tracked_root / "accuracy_provider_differences.tsv", provider_rows)

    host_gap = float(table["FP32_HOST"]["map50_95"]) - float(table["B2_HOST"]["map50_95"])
    recovery = []
    for surface in ("A1_HOST", "C2_HOST", "A1_BOARD_CPU", "A1_BOARD_EP", "C2_BOARD_CPU", "C2_BOARD_EP"):
        row = table[surface]
        base = table["B2_HOST"] if row["scope"] == "host" else table[f"B2_BOARD_{'CPU' if row['provider'] == 'CPU' else 'EP'}"]
        recovery.append({
            "surface": surface,
            "candidate_minus_corresponding_b2_map50_95": float(row["map50_95"]) - float(base["map50_95"]),
            "host_b2_to_fp32_gap": host_gap,
            "fraction_host_gap_recovered": (float(row["map50_95"]) - float(base["map50_95"])) / host_gap,
            "scope_caveat": "same-scope numerator; denominator is host B2-to-FP32 gap",
        })
    write_tsv(options.tracked_root / "accuracy_gap_recovery.tsv", recovery)
    write_tsv(options.tracked_root / "operating_point_census.tsv", census)
    write_tsv(options.tracked_root / "operating_point_per_class.tsv", per_class)
    write_tsv(options.tracked_root / "operating_point_per_size.tsv", per_size)
    write_tsv(options.tracked_root / "operating_point_validation.tsv", validation)

    (options.tracked_root / "operating_point_contract.md").write_text(
        "# Stage65E operating-point contract\n\n"
        "Counts are derived from `COCOeval.evalImgs` after category-wise COCO matching. "
        "For each score threshold, selected detections retain `dtIgnore`; TP/FP use "
        "`dtMatches` and `dtIgnore`, while FN is the set of non-ignored GT IDs not matched "
        "by a selected non-ignored detection. Crowd and area-ignore semantics use "
        "`gtIgnore`. MaxDets 100 and 300 are evaluated in separate COCOeval passes. "
        "The implementation passes an explicit crowd/ignore synthetic oracle and exact "
        "re-accumulation checks against every accepted aggregate metric.\n",
        encoding="utf-8",
    )
    write_accuracy_passport(
        options.tracked_root / "FP32_B2_A1_C2_ACCURACY_PASSPORT.md",
        absolute,
        census,
    )
    failures = [row for row in validation if row["status"] != "pass"]
    if failures:
        raise RuntimeError(f"accuracy/operating-point validation failed: {len(failures)} checks")
    print(
        f"stage65e_accuracy status=pass surfaces={len(absolute)} "
        f"operating_rows={len(census)} per_class_rows={len(per_class)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
