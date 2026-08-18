#!/usr/bin/env python3
"""Attribute H500 recall changes with exact COCOeval per-object matches."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SURFACES = ("B2_CPU", "B2_EP", "A1_CPU", "A1_EP")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    values = list(rows)
    if fields is None:
        if not values:
            raise ValueError(f"fields required for empty TSV: {path}")
        fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def parse_surface(raw: str) -> tuple[str, Path]:
    name, separator, value = raw.partition("=")
    if not separator or name not in SURFACES:
        raise ValueError(f"invalid --surface: {raw}")
    return name, Path(value).resolve()


def bbox_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def size_bin(area: float) -> str:
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def prepare(coco: COCO, ids: list[int], path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        result = coco.loadRes(payload)
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
    entries: dict[tuple[int, int], dict[str, Any]] = {}
    i_count = len(ids)
    a_count = len(evaluator._paramsEval.areaRng)
    for category_index, category in enumerate(evaluator._paramsEval.catIds):
        offset = category_index * a_count * i_count
        for image_index, image in enumerate(ids):
            item = evaluator.evalImgs[offset + image_index]
            if item is not None:
                entries[(image, category)] = item
    return {
        "sha256": sha256(path),
        "result": result,
        "entries": entries,
        "iou_thresholds": evaluator._paramsEval.iouThrs,
    }


def match_record(base: dict[str, Any], image: int, category: int, gt_id: int, threshold_index: int, gt_bbox: list[float]) -> dict[str, Any]:
    entry = base["entries"].get((image, category))
    if entry is None:
        return {"matched": 0, "detection_id": 0, "score": "", "rank": "", "iou": ""}
    gt_ids = [int(value) for value in entry["gtIds"]]
    try:
        gt_index = gt_ids.index(gt_id)
    except ValueError:
        return {"matched": 0, "detection_id": 0, "score": "", "rank": "", "iou": ""}
    detection_id = int(entry["gtMatches"][threshold_index, gt_index])
    if detection_id == 0:
        return {"matched": 0, "detection_id": 0, "score": "", "rank": "", "iou": ""}
    detection_ids = [int(value) for value in entry["dtIds"]]
    rank = detection_ids.index(detection_id) + 1
    detection = base["result"].anns[detection_id]
    return {
        "matched": 1,
        "detection_id": detection_id,
        "score": float(detection["score"]),
        "rank": rank,
        "iou": bbox_iou(gt_bbox, detection["bbox"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--surface", required=True, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)
    paths = dict(parse_surface(value) for value in options.surface)
    if tuple(sorted(paths)) != tuple(sorted(SURFACES)) or len(options.surface) != 4:
        raise ValueError("exactly four unique surfaces are required")

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(options.annotations))
    image_paths = [Path(line) for line in options.image_list.read_text().splitlines() if line]
    ids = sorted(int(path.stem) for path in image_paths)
    path_by_id = {int(path.stem): path for path in image_paths}
    bases = {name: prepare(coco, ids, path) for name, path in paths.items()}
    thresholds = bases["B2_CPU"]["iou_thresholds"]
    if any(not np.array_equal(base["iou_thresholds"], thresholds) for base in bases.values()):
        raise RuntimeError("COCO IoU threshold surfaces differ")

    delta_rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[int, int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_class: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for image in ids:
        for annotation in coco.imgToAnns.get(image, []):
            if int(annotation.get("iscrowd", 0)):
                continue
            category = int(annotation["category_id"])
            gt_id = int(annotation["id"])
            area = float(annotation["area"])
            bucket = size_bin(area)
            for threshold_index, threshold in enumerate(thresholds):
                matches = {
                    surface: match_record(base, image, category, gt_id, threshold_index, annotation["bbox"])
                    for surface, base in bases.items()
                }
                flags = {
                    "a1_cpu_matched_a1_ep_missed": matches["A1_CPU"]["matched"] and not matches["A1_EP"]["matched"],
                    "b2_ep_matched_a1_ep_missed": matches["B2_EP"]["matched"] and not matches["A1_EP"]["matched"],
                    "b2_cpu_matched_a1_cpu_missed": matches["B2_CPU"]["matched"] and not matches["A1_CPU"]["matched"],
                    "a1_ep_recovered_vs_b2_ep": matches["A1_EP"]["matched"] and not matches["B2_EP"]["matched"],
                    "a1_ep_lost_vs_b2_ep": matches["B2_EP"]["matched"] and not matches["A1_EP"]["matched"],
                    "a1_cpu_recovered_vs_b2_cpu": matches["A1_CPU"]["matched"] and not matches["B2_CPU"]["matched"],
                    "a1_cpu_lost_vs_b2_cpu": matches["B2_CPU"]["matched"] and not matches["A1_CPU"]["matched"],
                    "any_surface_disagreement": len(
                        {int(matches[surface]["matched"]) for surface in SURFACES}
                    )
                    > 1,
                }
                stats = aggregate[(image, category, bucket)]
                class_stats = per_class[(category, bucket)]
                stats["objects"] += 1
                class_stats["objects"] += 1
                for surface in SURFACES:
                    stats[f"{surface}_matched"] += int(matches[surface]["matched"])
                    class_stats[f"{surface}_matched"] += int(matches[surface]["matched"])
                for flag, value in flags.items():
                    stats[flag] += int(value)
                    class_stats[flag] += int(value)
                if not any(flags.values()):
                    continue
                row: dict[str, Any] = {
                    "image_id": image,
                    "image_sha256": sha256(path_by_id[image]),
                    "category_id": category,
                    "category": coco.cats[category]["name"],
                    "gt_id": gt_id,
                    "gt_bbox": json.dumps(annotation["bbox"], separators=(",", ":")),
                    "gt_area": area,
                    "size_bin": bucket,
                    "pyramid_proxy": {"small": "P3", "medium": "P4", "large": "P5"}[bucket],
                    "iou_threshold": float(threshold),
                }
                row.update({name: int(value) for name, value in flags.items()})
                for surface in SURFACES:
                    for field, value in matches[surface].items():
                        row[f"{surface}_{field}"] = value
                delta_rows.append(row)

    write_tsv(options.output_dir / "h500_object_match_delta.tsv", delta_rows)
    write_tsv(
        options.output_dir / "h500_large_recall_loss_objects.tsv",
        [row for row in delta_rows if row["size_bin"] == "large" and row["a1_cpu_matched_a1_ep_missed"]],
        list(delta_rows[0]) if delta_rows else None,
    )
    write_tsv(
        options.output_dir / "h500_small_recall_loss_objects.tsv",
        [row for row in delta_rows if row["size_bin"] == "small" and row["a1_cpu_matched_a1_ep_missed"]],
        list(delta_rows[0]) if delta_rows else None,
    )

    per_image_rows = []
    for (image, category, bucket), values in sorted(aggregate.items()):
        total = values["objects"]
        row: dict[str, Any] = {
            "image_id": image,
            "image_sha256": sha256(path_by_id[image]),
            "category_id": category,
            "category": coco.cats[category]["name"],
            "size_bin": bucket,
            "object_threshold_events": total,
        }
        for surface in SURFACES:
            row[f"{surface}_recall"] = values[f"{surface}_matched"] / total
        row["a1_provider_interaction"] = (
            row["A1_EP_recall"] - row["A1_CPU_recall"]
            - row["B2_EP_recall"] + row["B2_CPU_recall"]
        )
        for flag in (
            "a1_cpu_matched_a1_ep_missed", "b2_ep_matched_a1_ep_missed",
            "b2_cpu_matched_a1_cpu_missed", "a1_ep_recovered_vs_b2_ep",
            "a1_ep_lost_vs_b2_ep", "a1_cpu_recovered_vs_b2_cpu", "a1_cpu_lost_vs_b2_cpu",
        ):
            row[flag] = values[flag]
        per_image_rows.append(row)
    write_tsv(options.output_dir / "h500_per_image_recall_delta.tsv", per_image_rows)

    class_rows = []
    for (category, bucket), values in sorted(per_class.items()):
        total = values["objects"]
        row = {
            "category_id": category,
            "category": coco.cats[category]["name"],
            "size_bin": bucket,
            "object_threshold_events": total,
        }
        for surface in SURFACES:
            row[f"{surface}_recall"] = values[f"{surface}_matched"] / total
        row["a1_provider_interaction"] = (
            row["A1_EP_recall"] - row["A1_CPU_recall"]
            - row["B2_EP_recall"] + row["B2_CPU_recall"]
        )
        class_rows.append(row)
    write_tsv(options.output_dir / "h500_per_class_interaction.tsv", class_rows)

    by_image: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in delta_rows:
        image = int(row["image_id"])
        bucket = str(row["size_bin"])
        if row["any_surface_disagreement"]:
            by_image[image]["any_disagreement"] += 1
        if row["a1_cpu_matched_a1_ep_missed"]:
            by_image[image][f"{bucket}_provider_loss"] += 1
        if row["a1_ep_lost_vs_b2_ep"]:
            by_image[image][f"{bucket}_a1_ep_loss"] += 1
        if row["a1_cpu_lost_vs_b2_cpu"]:
            by_image[image][f"{bucket}_a1_cpu_loss"] += 1
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    groups = (
        ("large-loss", 32, "large"),
        ("small-loss", 16, "small"),
    )
    for group, limit, bucket in groups:
        ranked = sorted(
            (
                (
                    -(values[f"{bucket}_provider_loss"] + values[f"{bucket}_a1_ep_loss"] + values[f"{bucket}_a1_cpu_loss"]),
                    image,
                    values,
                )
                for image, values in by_image.items()
                if values[f"{bucket}_provider_loss"] + values[f"{bucket}_a1_ep_loss"] + values[f"{bucket}_a1_cpu_loss"] > 0
                and image not in used
            )
        )
        for rank, (_, image, values) in enumerate(ranked[:limit], 1):
            used.add(image)
            selected.append(
                {
                    "selection_group": group,
                    "rank": rank,
                    "image_id": image,
                    "image_path": path_by_id[image],
                    "image_sha256": sha256(path_by_id[image]),
                    "provider_loss_events": values[f"{bucket}_provider_loss"],
                    "a1_ep_loss_events": values[f"{bucket}_a1_ep_loss"],
                    "a1_cpu_loss_events": values[f"{bucket}_a1_cpu_loss"],
                    "selection_rule": "descending summed loss events then image_id",
                }
            )

    control_scores: dict[int, int] = defaultdict(int)
    for (image, _category, bucket), values in aggregate.items():
        if image in used or image in by_image or bucket != "large":
            continue
        if all(values[f"{surface}_matched"] == values["B2_CPU_matched"] for surface in SURFACES):
            control_scores[image] += values["B2_CPU_matched"]
    for rank, (image, score) in enumerate(sorted(control_scores.items(), key=lambda item: (-item[1], item[0]))[:16], 1):
        selected.append(
            {
                "selection_group": "matched-control",
                "rank": rank,
                "image_id": image,
                "image_path": path_by_id[image],
                "image_sha256": sha256(path_by_id[image]),
                "provider_loss_events": 0,
                "a1_ep_loss_events": 0,
                "a1_cpu_loss_events": 0,
                "selection_rule": "descending identical large-object matched events then image_id",
            }
        )
    selection_path = options.output_dir / "selected_for_boundary_diagnostic.tsv"
    write_tsv(selection_path, selected)
    (options.output_dir / "selected_for_boundary_diagnostic.sha256").write_text(
        f"{sha256(selection_path)}  {selection_path.name}\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
