#!/usr/bin/env python3
"""Exact DEV-001C bootstrap with one stable score sort per category.

COCOeval uses the same detection scores for every area range.  The baseline
bootstrap sorts those scores independently for all four area ranges.  This
wrapper preserves the baseline accumulation contract while reusing the exact
stable order across the area ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import xslim_dev_001c_bootstrap as core


def accumulate_fast(
    eval_imgs: list[Any], params: Any, draws: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    """Accumulate one bootstrap draw with exact baseline tie ordering."""
    threshold_count = len(params.iouThrs)
    recall_count = len(params.recThrs)
    category_count = len(params.catIds)
    area_count = len(params.areaRng)
    image_count = len(params.imgIds)
    precision = -np.ones(
        (threshold_count, recall_count, category_count, area_count),
        dtype=np.float64,
    )
    recall_by_max = {
        limit: -np.ones(
            (threshold_count, category_count, area_count), dtype=np.float64
        )
        for limit in (1, 10, 100)
    }
    for category_index in range(category_count):
        category_offset = category_index * area_count * image_count
        selected_by_area: list[list[Any]] = []
        for area_index in range(area_count):
            offset = category_offset + area_index * image_count
            selected = [eval_imgs[offset + int(index)] for index in draws]
            selected_by_area.append([item for item in selected if item is not None])

        score_source = selected_by_area[0]
        if not score_source:
            if any(selected_by_area[1:]):
                raise RuntimeError("COCOeval area ranges disagree on image coverage")
            continue
        scores = np.concatenate([item["dtScores"][:100] for item in score_source])
        order = np.argsort(-scores, kind="mergesort")

        for area_index, selected in enumerate(selected_by_area):
            if not selected:
                raise RuntimeError("COCOeval area range unexpectedly has no records")
            gt_ignored = np.concatenate([item["gtIgnore"] for item in selected])
            positive = np.count_nonzero(gt_ignored == 0)
            if positive == 0:
                continue
            for limit in (1, 10, 100):
                matches = np.concatenate(
                    [item["dtMatches"][:, :limit] for item in selected], axis=1
                )
                ignored = np.concatenate(
                    [item["dtIgnore"][:, :limit] for item in selected], axis=1
                )
                true_positive = np.logical_and(matches, np.logical_not(ignored))
                recall_by_max[limit][:, category_index, area_index] = (
                    np.count_nonzero(true_positive, axis=1) / positive
                )

            matches = np.concatenate(
                [item["dtMatches"][:, :100] for item in selected], axis=1
            )
            ignored = np.concatenate(
                [item["dtIgnore"][:, :100] for item in selected], axis=1
            )
            if matches.shape[1] != order.size or ignored.shape[1] != order.size:
                raise RuntimeError("COCOeval area ranges disagree on detection coverage")
            matches = matches[:, order]
            ignored = ignored[:, order]
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
                values = np.zeros(recall_count, dtype=np.float64)
                values[valid] = curve[positions[valid]]
                precision[threshold_index, :, category_index, area_index] = values

    iou75 = int(np.flatnonzero(np.isclose(params.iouThrs, 0.75))[0])
    return {
        "map50_95": core.valid_mean(precision[:, :, :, 0]),
        "map50": core.valid_mean(precision[0, :, :, 0]),
        "map75": core.valid_mean(precision[iou75, :, :, 0]),
        "ap_small": core.valid_mean(precision[:, :, :, 1]),
        "ap_medium": core.valid_mean(precision[:, :, :, 2]),
        "ap_large": core.valid_mean(precision[:, :, :, 3]),
        "ar1": core.valid_mean(recall_by_max[1][:, :, 0]),
        "ar10": core.valid_mean(recall_by_max[10][:, :, 0]),
        "ar100": core.valid_mean(recall_by_max[100][:, :, 0]),
        "ar_small": core.valid_mean(recall_by_max[100][:, :, 1]),
        "ar_medium": core.valid_mean(recall_by_max[100][:, :, 2]),
        "ar_large": core.valid_mean(recall_by_max[100][:, :, 3]),
        "prediction_count": float(np.sum(counts[draws], dtype=np.int64)),
    }


_BASE_RESUME_CONTRACT = core.resume_contract


def resume_contract_fast(
    options: Any, parsed: dict[str, Path], draw_sha: str
) -> dict[str, Any]:
    contract = _BASE_RESUME_CONTRACT(options, parsed, draw_sha)
    contract["contract_version"] = "xslim-dev-001c-bootstrap-fast-resume-v1"
    contract["tool_sha256"] = core.sha256(Path(__file__).resolve())
    contract["core_tool_sha256"] = core.sha256(Path(core.__file__).resolve())
    contract["optimization"] = "reuse-stable-score-order-across-coco-area-ranges"
    return contract


def main() -> int:
    core.accumulate = accumulate_fast
    core.resume_contract = resume_contract_fast
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
