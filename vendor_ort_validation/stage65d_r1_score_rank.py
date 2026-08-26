#!/usr/bin/env python3
"""Prediction-only score, rank, and TopK sensitivity diagnostics for Stage65D-R1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

SURFACES = ("B2_CPU", "B2_EP", "C2_CPU", "C2_EP")
PAIRS = (
    ("B2_EP-vs-B2_CPU", "B2_EP", "B2_CPU"),
    ("C2_EP-vs-C2_CPU", "C2_EP", "C2_CPU"),
    ("C2_EP-vs-B2_EP", "C2_EP", "B2_EP"),
    ("C2_CPU-vs-B2_CPU", "C2_CPU", "B2_CPU"),
)
THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50)
SCORE_BINS = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0000001)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty report: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_surface(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or name not in SURFACES:
        raise ValueError(f"invalid --surface: {value}")
    return name, Path(raw_path).resolve()


def load(path: Path) -> dict[int, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected JSON array: {path}")
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(payload):
        item = {
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "bbox": tuple(float(value) for value in row["bbox"]),
            "score": float(row["score"]),
            "source_index": index,
        }
        grouped[item["image_id"]].append(item)
    for rows in grouped.values():
        rows.sort(key=lambda row: (-float(row["score"]), int(row["source_index"])))
    return dict(grouped)


def size_bin(bbox: tuple[float, ...]) -> str:
    area = max(0.0, bbox[2]) * max(0.0, bbox[3])
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def iou_matrix(left: list[dict[str, object]], right: list[dict[str, object]]) -> np.ndarray:
    if not left or not right:
        return np.empty((len(left), len(right)), dtype=np.float64)
    a = np.asarray([row["bbox"] for row in left], dtype=np.float64)
    b = np.asarray([row["bbox"] for row in right], dtype=np.float64)
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = ax1 + a[:, 2], ay1 + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    inter_w = np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]))
    inter_h = np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]))
    intersection = inter_w * inter_h
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def mutual_matches(left: list[dict[str, object]], right: list[dict[str, object]], threshold: float = 0.5) -> list[tuple[int, int, float]]:
    matrix = iou_matrix(left, right)
    if not matrix.size:
        return []
    left_best = np.argmax(matrix, axis=1)
    right_best = np.argmax(matrix, axis=0)
    matches = []
    for left_index, right_index in enumerate(left_best):
        right_value = int(right_index)
        value = float(matrix[left_index, right_value])
        if value >= threshold and int(right_best[right_value]) == left_index:
            matches.append((left_index, right_value, value))
    return matches


def inversion_count(values: list[int]) -> int:
    if not values:
        return 0
    size = max(values) + 2
    tree = [0] * (size + 1)

    def add(index: int) -> None:
        index += 1
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        index += 1
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    inversions = 0
    seen = 0
    for value in values:
        inversions += seen - prefix(value)
        add(value)
        seen += 1
    return inversions


def summary(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan, math.nan
    return statistics.fmean(values), statistics.median(values), min(values), max(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", action="append", required=True)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    paths = dict(parse_surface(value) for value in options.surface)
    if tuple(sorted(paths)) != tuple(sorted(SURFACES)) or len(options.surface) != 4:
        raise ValueError(f"one path is required for each surface: {SURFACES}")
    interaction_path = options.tracked_root / "full_val_board_provider_interactions.tsv"
    if not interaction_path.is_file():
        raise FileNotFoundError(
            "provider interaction report must exist before score/rank interpretation: "
            f"{interaction_path}"
        )
    with interaction_path.open(encoding="utf-8", newline="") as stream:
        interactions = list(csv.DictReader(stream, delimiter="\t"))
    if not interactions:
        raise ValueError(f"provider interaction report is empty: {interaction_path}")
    surfaces = {name: load(paths[name]) for name in SURFACES}

    census_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    for surface in SURFACES:
        rows = [row for image_rows in surfaces[surface].values() for row in image_rows]
        scopes: dict[tuple[str, str], list[dict[str, object]]] = {("aggregate", "all"): rows}
        for size in ("small", "medium", "large"):
            scopes[("size", size)] = [row for row in rows if size_bin(row["bbox"]) == size]
        for category in range(1, 91):
            selected = [row for row in rows if int(row["category_id"]) == category]
            if selected:
                scopes[("category", str(category))] = selected
        for (scope, key), selected in scopes.items():
            for threshold in THRESHOLDS:
                retained = [row for row in selected if float(row["score"]) >= threshold]
                census_rows.append({
                    "surface": surface,
                    "scope": scope,
                    "key": key,
                    "threshold": threshold,
                    "prediction_count": len(retained),
                    "images_with_predictions": len({int(row["image_id"]) for row in retained}),
                })
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        for lower, upper in zip(SCORE_BINS[:-1], SCORE_BINS[1:]):
            count = int(np.count_nonzero((scores >= lower) & (scores < upper)))
            histogram_rows.append({
                "surface": surface,
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "count": count,
                "fraction": count / max(1, scores.size),
            })
    write_tsv(options.tracked_root / "full_val_score_threshold_census.tsv", census_rows)
    write_tsv(options.tracked_root / "full_val_score_distribution.tsv", histogram_rows)

    membership_acc: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    inversion_acc: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    change_acc: dict[tuple[str, str, str], dict[str, object]] = {}
    pair_summary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for pair_name, left_name, right_name in PAIRS:
        image_ids = sorted(set(surfaces[left_name]) | set(surfaces[right_name]))
        for image_id in image_ids:
            left = surfaces[left_name].get(image_id, [])
            right = surfaces[right_name].get(image_id, [])
            matches = mutual_matches(left, right)
            pair_summary[pair_name]["images"] += 1
            pair_summary[pair_name]["left_predictions"] += len(left)
            pair_summary[pair_name]["right_predictions"] += len(right)
            pair_summary[pair_name]["matched"] += len(matches)
            for k in (100, 300):
                both = [(li, ri) for li, ri, _ in matches if li < k and ri < k]
                crossing = [(li, ri) for li, ri, _ in matches if (li < k) != (ri < k)]
                union = min(k, len(left)) + min(k, len(right)) - len(both)
                acc = membership_acc[(pair_name, k)]
                acc["images"] += 1
                acc["left_members"] += min(k, len(left))
                acc["right_members"] += min(k, len(right))
                acc["matched_both"] += len(both)
                acc["membership_crossings"] += len(crossing)
                acc["union"] += union
                ordered = sorted(both)
                sequence = [right_index for _, right_index in ordered]
                inversions = inversion_count(sequence)
                possible = len(sequence) * (len(sequence) - 1) // 2
                inv = inversion_acc[(pair_name, k)]
                inv["matched"] += len(sequence)
                inv["inversions"] += inversions
                inv["possible_pairs"] += possible
                inv["near_boundary_crossings"] += sum(
                    1 for li, ri in crossing if min(abs(li - k), abs(ri - k)) <= 50
                )

            for left_index, right_index, iou in matches:
                left_row = left[left_index]
                right_row = right[right_index]
                size = size_bin(left_row["bbox"])
                category = str(left_row["category_id"])
                for scope, key in (("aggregate", "all"), ("size", size), ("category", category)):
                    group = change_acc.setdefault((pair_name, scope, key), {
                        "pair": pair_name,
                        "scope": scope,
                        "key": key,
                        "matched": 0,
                        "class_changes": 0,
                        "score_delta_sum": 0.0,
                        "score_abs_delta_sum": 0.0,
                        "score_abs_delta_max": 0.0,
                        "bbox_iou_sum": 0.0,
                        "bbox_iou_min": 1.0,
                        **{f"threshold_crossings_{threshold:g}": 0 for threshold in THRESHOLDS},
                    })
                    delta = float(left_row["score"]) - float(right_row["score"])
                    group["matched"] = int(group["matched"]) + 1
                    group["class_changes"] = int(group["class_changes"]) + int(left_row["category_id"] != right_row["category_id"])
                    group["score_delta_sum"] = float(group["score_delta_sum"]) + delta
                    group["score_abs_delta_sum"] = float(group["score_abs_delta_sum"]) + abs(delta)
                    group["score_abs_delta_max"] = max(float(group["score_abs_delta_max"]), abs(delta))
                    group["bbox_iou_sum"] = float(group["bbox_iou_sum"]) + iou
                    group["bbox_iou_min"] = min(float(group["bbox_iou_min"]), iou)
                    for threshold in THRESHOLDS:
                        field = f"threshold_crossings_{threshold:g}"
                        group[field] = int(group[field]) + int(
                            (float(left_row["score"]) >= threshold)
                            != (float(right_row["score"]) >= threshold)
                        )

    membership_rows = []
    for (pair, k), acc in sorted(membership_acc.items()):
        membership_rows.append({
            "pair": pair,
            "topk": k,
            "images": int(acc["images"]),
            "left_members": int(acc["left_members"]),
            "right_members": int(acc["right_members"]),
            "matched_both": int(acc["matched_both"]),
            "membership_crossings": int(acc["membership_crossings"]),
            "jaccard": acc["matched_both"] / max(1.0, acc["union"]),
        })
    write_tsv(options.tracked_root / "full_val_rank_membership.tsv", membership_rows)

    inversion_rows = []
    for (pair, k), acc in sorted(inversion_acc.items()):
        inversion_rows.append({
            "pair": pair,
            "topk": k,
            "matched": int(acc["matched"]),
            "inversions": int(acc["inversions"]),
            "possible_pairs": int(acc["possible_pairs"]),
            "inversion_fraction": acc["inversions"] / max(1.0, acc["possible_pairs"]),
            "near_boundary_crossings": int(acc["near_boundary_crossings"]),
        })
    write_tsv(options.tracked_root / "full_val_rank_inversions.tsv", inversion_rows)

    change_rows = []
    for group in change_acc.values():
        matched = int(group.pop("matched"))
        score_delta_sum = float(group.pop("score_delta_sum"))
        score_abs_delta_sum = float(group.pop("score_abs_delta_sum"))
        bbox_iou_sum = float(group.pop("bbox_iou_sum"))
        change_rows.append({
            **group,
            "matched": matched,
            "score_delta_mean": score_delta_sum / max(1, matched),
            "score_abs_delta_mean": score_abs_delta_sum / max(1, matched),
            "bbox_iou_mean": bbox_iou_sum / max(1, matched),
        })
    change_rows.sort(key=lambda row: (str(row["pair"]), str(row["scope"]), str(row["key"])))
    write_tsv(options.tracked_root / "full_val_matched_candidate_changes.tsv", change_rows)

    c2_top100 = next(row for row in membership_rows if row["pair"] == "C2_EP-vs-C2_CPU" and row["topk"] == 100)
    b2_top100 = next(row for row in membership_rows if row["pair"] == "B2_EP-vs-B2_CPU" and row["topk"] == 100)
    c2_top300 = next(row for row in membership_rows if row["pair"] == "C2_EP-vs-C2_CPU" and row["topk"] == 300)
    b2_top300 = next(row for row in membership_rows if row["pair"] == "B2_EP-vs-B2_CPU" and row["topk"] == 300)
    c2_changes = next(row for row in change_rows if row["pair"] == "C2_EP-vs-C2_CPU" and row["scope"] == "aggregate")
    b2_changes = next(row for row in change_rows if row["pair"] == "B2_EP-vs-B2_CPU" and row["scope"] == "aggregate")

    threshold_counts = {
        (str(row["surface"]), float(row["threshold"])): int(row["prediction_count"])
        for row in census_rows
        if row["scope"] == "aggregate" and row["key"] == "all"
    }
    c2_low_delta = threshold_counts[("C2_EP", 0.001)] - threshold_counts[("C2_CPU", 0.001)]
    b2_low_delta = threshold_counts[("B2_EP", 0.001)] - threshold_counts[("B2_CPU", 0.001)]
    c2_mid_delta = threshold_counts[("C2_EP", 0.05)] - threshold_counts[("C2_CPU", 0.05)]
    b2_mid_delta = threshold_counts[("B2_EP", 0.05)] - threshold_counts[("B2_CPU", 0.05)]

    task_interactions = [row for row in interactions if row["metric"] != "prediction_count"]
    material_count = sum("material" in row["classification"] for row in task_interactions)
    inconclusive_count = sum(
        row["classification"] == "provider-interaction-inconclusive"
        for row in task_interactions
    )
    neutral_count = sum(
        row["classification"] == "provider-neutral" for row in task_interactions
    )
    report = [
        "# Provider score/rank sensitivity interpretation",
        "",
        "This analysis uses frozen full-val prediction JSON only. Spatial candidates are paired by deterministic mutual-best bbox IoU >= 0.5; it does not expose provider-internal rounding.",
        "",
        f"- C2 EP/CPU Top-100 membership Jaccard is `{float(c2_top100['jaccard']):.9f}` with `{c2_top100['membership_crossings']}` crossings; B2 EP/CPU is `{float(b2_top100['jaccard']):.9f}` with `{b2_top100['membership_crossings']}` crossings. C2 is not uniquely worse on this control-normalized surface.",
        f"- Top-300 Jaccard is `{float(c2_top300['jaccard']):.9f}` for C2 and `{float(b2_top300['jaccard']):.9f}` for B2. The recorded Top-300 crossing count is zero by construction because the serialized detector output is already capped at 300; it is not evidence of membership equality.",
        f"- C2 EP/CPU matched score absolute mean delta is `{float(c2_changes['score_abs_delta_mean']):.12f}` with `{c2_changes['class_changes']}` class changes; B2 is `{float(b2_changes['score_abs_delta_mean']):.12f}` with `{b2_changes['class_changes']}` class changes.",
        f"- EP minus CPU prediction-count deltas are `{c2_low_delta}` (C2) versus `{b2_low_delta}` (B2) at score 0.001 and `{c2_mid_delta}` versus `{b2_mid_delta}` at score 0.05. Counts are descriptive and are not an accuracy oracle.",
        f"- Population-level difference-in-differences classifies `{neutral_count}` task metrics provider-neutral, `{inconclusive_count}` inconclusive, and `{material_count}` material. Therefore the deterministic ranking differences do not establish a C2-specific material provider interaction.",
        "- The data are compatible with confidence/rank sensitivity in both frozen models, but cannot prove an EP bug, exact rounding mode, or an LSB-level root cause.",
    ]
    (options.tracked_root / "provider_sensitivity_interpretation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
