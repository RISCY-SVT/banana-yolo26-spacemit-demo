#!/usr/bin/env python3
"""Stage65C paired COCO bootstrap including AP and AR size-bin metrics."""

from __future__ import annotations

import numpy as np
import stage65b_r2_bootstrap as base

base.METRICS = (
    "map50_95",
    "map50",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def valid_mean(values: np.ndarray) -> float:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else float("nan")


def accumulate_metrics(eval_imgs, params, draws: np.ndarray) -> dict[str, float]:
    t_count = len(params.iouThrs)
    r_count = len(params.recThrs)
    k_count = len(params.catIds)
    a_count = len(params.areaRng)
    max_det = 100
    precision = -np.ones((t_count, r_count, k_count, a_count), dtype=np.float64)
    recall_grid = -np.ones((t_count, k_count, a_count), dtype=np.float64)
    i_count = len(params.imgIds)
    for category_index in range(k_count):
        category_offset = category_index * a_count * i_count
        for area_index in range(a_count):
            offset = category_offset + area_index * i_count
            selected = [eval_imgs[offset + int(index)] for index in draws]
            selected = [item for item in selected if item is not None]
            if not selected:
                continue
            scores = np.concatenate([item["dtScores"][:max_det] for item in selected])
            order = np.argsort(-scores, kind="mergesort")
            matches = np.concatenate(
                [item["dtMatches"][:, :max_det] for item in selected], axis=1
            )[:, order]
            ignored = np.concatenate(
                [item["dtIgnore"][:, :max_det] for item in selected], axis=1
            )[:, order]
            gt_ignored = np.concatenate([item["gtIgnore"] for item in selected])
            positive = np.count_nonzero(gt_ignored == 0)
            if positive == 0:
                continue
            true_positive = np.logical_and(matches, np.logical_not(ignored))
            false_positive = np.logical_and(
                np.logical_not(matches), np.logical_not(ignored)
            )
            tp_sum = np.cumsum(true_positive, axis=1, dtype=np.float64)
            fp_sum = np.cumsum(false_positive, axis=1, dtype=np.float64)
            for threshold_index, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                recall = tp / positive
                curve = tp / (tp + fp + np.spacing(1))
                curve = np.maximum.accumulate(curve[::-1])[::-1]
                positions = np.searchsorted(recall, params.recThrs, side="left")
                valid = positions < curve.size
                values = np.zeros(r_count, dtype=np.float64)
                values[valid] = curve[positions[valid]]
                precision[threshold_index, :, category_index, area_index] = values
                recall_grid[threshold_index, category_index, area_index] = (
                    recall[-1] if recall.size else 0.0
                )
    return {
        "map50_95": valid_mean(precision[:, :, :, 0]),
        "map50": valid_mean(precision[0, :, :, 0]),
        "ap_small": valid_mean(precision[:, :, :, 1]),
        "ap_medium": valid_mean(precision[:, :, :, 2]),
        "ap_large": valid_mean(precision[:, :, :, 3]),
        "ar_small": valid_mean(recall_grid[:, :, 1]),
        "ar_medium": valid_mean(recall_grid[:, :, 2]),
        "ar_large": valid_mean(recall_grid[:, :, 3]),
    }


base.accumulate_metrics = accumulate_metrics


if __name__ == "__main__":
    raise SystemExit(base.main())
