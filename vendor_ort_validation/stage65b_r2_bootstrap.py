#!/usr/bin/env python3
"""Deterministic paired image-level COCO bootstrap with exact cached matching."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from stage65b_r2_common import sha256, write_tsv


METRICS = ("map50_95", "map50", "ap_small", "ap_medium", "ap_large")
_BASES: dict[str, dict[str, Any]] = {}
_DRAWS: np.ndarray | None = None
_PAIRS: list[tuple[str, str, str]] = []


def selected_ids(path: Path, coco: COCO) -> list[int]:
    ids = sorted(int(Path(line).stem) for line in path.read_text().splitlines() if line)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("image list is empty or contains duplicate IDs")
    missing = set(ids) - set(coco.imgs)
    if missing:
        raise ValueError(f"{len(missing)} selected IDs absent from annotations")
    return ids


def prediction_payload(path: Path, allowed: set[int]) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"prediction payload is not an array: {path}")
    unexpected = {int(row["image_id"]) for row in rows} - allowed
    if unexpected:
        raise ValueError(f"prediction payload has {len(unexpected)} unexpected IDs")
    return rows


def prepare_base(
    coco: COCO, ids: list[int], path: Path, log: io.StringIO
) -> dict[str, Any]:
    predictions = prediction_payload(path, set(ids))
    with contextlib.redirect_stdout(log):
        result = coco.loadRes(predictions)
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
    params = evaluator._paramsEval
    if params.imgIds != ids:
        raise RuntimeError("COCOeval changed the sorted image-ID contract")
    point = accumulate_metrics(evaluator.evalImgs, params, np.arange(len(ids)))
    by_image: dict[int, list[dict[str, Any]]] = {}
    for row in predictions:
        by_image.setdefault(int(row["image_id"]), []).append(row)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "evalImgs": evaluator.evalImgs,
        "params": params,
        "point": point,
        "predictions_by_image": by_image,
    }


def valid_mean(values: np.ndarray) -> float:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else float("nan")


def accumulate_metrics(
    eval_imgs: list[dict[str, Any] | None], params: Any, draws: np.ndarray
) -> dict[str, float]:
    """Re-run COCO accumulation over a duplicate-preserving image draw.

    The image matching phase is invariant per source image. Reusing those exact
    COCOeval match records and repeating their references is equivalent to
    assigning each draw a fresh synthetic image ID and matching it again.
    """
    t_count = len(params.iouThrs)
    r_count = len(params.recThrs)
    k_count = len(params.catIds)
    a_count = len(params.areaRng)
    max_det = 100
    precision = -np.ones((t_count, r_count, k_count, a_count), dtype=np.float64)
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
    return {
        "map50_95": valid_mean(precision[:, :, :, 0]),
        "map50": valid_mean(precision[0, :, :, 0]),
        "ap_small": valid_mean(precision[:, :, :, 1]),
        "ap_medium": valid_mean(precision[:, :, :, 2]),
        "ap_large": valid_mean(precision[:, :, :, 3]),
    }


def worker(task: tuple[int, int]) -> tuple[int, int, list[float]]:
    pair_index, replicate = task
    assert _DRAWS is not None
    _, left_key, right_key = _PAIRS[pair_index]
    draw = _DRAWS[replicate]
    left = _BASES[left_key]
    right = _BASES[right_key]
    left_metrics = accumulate_metrics(left["evalImgs"], left["params"], draw)
    right_metrics = accumulate_metrics(right["evalImgs"], right["params"], draw)
    return (
        pair_index,
        replicate,
        [left_metrics[name] - right_metrics[name] for name in METRICS],
    )


def remapped_validation(
    coco: COCO,
    ids: list[int],
    base: dict[str, Any],
    draw: np.ndarray,
) -> dict[str, float]:
    """Run one literal synthetic-ID remap to validate cached-match equivalence."""
    dataset = {
        key: copy.deepcopy(coco.dataset.get(key, []))
        for key in ("info", "licenses", "categories")
    }
    dataset["images"] = []
    dataset["annotations"] = []
    predictions: list[dict[str, Any]] = []
    next_annotation = 1
    for synthetic_index, original_index in enumerate(draw, 1):
        original_id = ids[int(original_index)]
        image = copy.deepcopy(coco.imgs[original_id])
        image["id"] = synthetic_index
        dataset["images"].append(image)
        for annotation in coco.imgToAnns.get(original_id, []):
            remapped = copy.deepcopy(annotation)
            remapped["id"] = next_annotation
            remapped["image_id"] = synthetic_index
            next_annotation += 1
            dataset["annotations"].append(remapped)
        for prediction in base["predictions_by_image"].get(original_id, []):
            remapped_prediction = copy.deepcopy(prediction)
            remapped_prediction["image_id"] = synthetic_index
            predictions.append(remapped_prediction)
    synthetic = COCO()
    synthetic.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        synthetic.createIndex()
        result = synthetic.loadRes(predictions)
        evaluator = COCOeval(synthetic, result, "bbox")
        evaluator.params.imgIds = list(range(1, len(draw) + 1))
        evaluator.evaluate()
    return accumulate_metrics(
        evaluator.evalImgs, evaluator._paramsEval, np.arange(len(draw))
    )


def parse_pair(raw: str) -> tuple[str, str, str]:
    name, separator, paths = raw.partition("=")
    left, comma, right = paths.partition(",")
    if not separator or not comma or not name or not left or not right:
        raise ValueError(f"invalid --pair: {raw}")
    return name, str(Path(left).resolve()), str(Path(right).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--pair", required=True, action="append")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=65002)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-remap", action="store_true")
    options = parser.parse_args()
    if options.replicates < 1000:
        raise ValueError("at least 1000 replicates are required")
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco = COCO(str(options.annotations))
    ids = selected_ids(options.image_list, coco)
    pairs = [parse_pair(raw) for raw in options.pair]
    unique_paths = sorted({path for _, left, right in pairs for path in (left, right)})
    global _BASES, _DRAWS, _PAIRS
    _BASES = {
        path: prepare_base(coco, ids, Path(path), log) for path in unique_paths
    }
    _PAIRS = pairs
    rng = np.random.default_rng(options.seed)
    _DRAWS = rng.integers(
        0,
        len(ids),
        size=(options.replicates, len(ids)),
        dtype=np.int32,
    )
    draw_sha = hashlib.sha256(_DRAWS.tobytes(order="C")).hexdigest()

    validation_rows: list[dict[str, Any]] = []
    if options.validate_remap:
        first_pair = pairs[0]
        for side, key in (("left", first_pair[1]), ("right", first_pair[2])):
            literal = remapped_validation(coco, ids, _BASES[key], _DRAWS[0])
            cached = accumulate_metrics(
                _BASES[key]["evalImgs"], _BASES[key]["params"], _DRAWS[0]
            )
            for metric in METRICS:
                difference = abs(literal[metric] - cached[metric])
                validation_rows.append(
                    {
                        "pair": first_pair[0],
                        "side": side,
                        "metric": metric,
                        "literal_synthetic_id": literal[metric],
                        "cached_match": cached[metric],
                        "absolute_difference": difference,
                        "status": "pass" if difference <= 1e-12 else "fail",
                    }
                )
        if any(row["status"] != "pass" for row in validation_rows):
            raise RuntimeError("cached bootstrap differs from literal synthetic-ID remap")
        write_tsv(options.output_dir / "synthetic_id_validation.tsv", validation_rows)

    values = np.empty(
        (len(pairs), options.replicates, len(METRICS)), dtype=np.float64
    )
    tasks = [
        (pair_index, replicate)
        for pair_index in range(len(pairs))
        for replicate in range(options.replicates)
    ]
    if options.workers == 1:
        results = map(worker, tasks)
        pool = None
    else:
        context = mp.get_context("fork")
        pool = context.Pool(processes=options.workers)
        results = pool.imap_unordered(worker, tasks, chunksize=1)
    try:
        for completed, (pair_index, replicate, row) in enumerate(results, 1):
            values[pair_index, replicate, :] = row
            if completed % 100 == 0:
                print(f"bootstrap: {completed}/{len(tasks)}", flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    replicate_file = options.output_dir / "paired_bootstrap_replicates.npz"
    np.savez_compressed(
        replicate_file,
        deltas=values,
        metrics=np.asarray(METRICS),
        pairs=np.asarray([pair[0] for pair in pairs]),
        draw_sha256=np.asarray(draw_sha),
        seed=np.asarray(options.seed),
    )
    summary_rows: list[dict[str, Any]] = []
    for pair_index, (name, left, right) in enumerate(pairs):
        for metric_index, metric in enumerate(METRICS):
            delta = values[pair_index, :, metric_index]
            point = _BASES[left]["point"][metric] - _BASES[right]["point"][metric]
            summary_rows.append(
                {
                    "pair": name,
                    "left_prediction_sha256": _BASES[left]["sha256"],
                    "right_prediction_sha256": _BASES[right]["sha256"],
                    "metric": metric,
                    "replicates": options.replicates,
                    "seed": options.seed,
                    "point_delta": point,
                    "bootstrap_mean": float(np.mean(delta)),
                    "bootstrap_median": float(np.median(delta)),
                    "percentile_2_5": float(np.percentile(delta, 2.5)),
                    "percentile_97_5": float(np.percentile(delta, 97.5)),
                    "probability_delta_gt_zero": float(np.mean(delta > 0.0)),
                }
            )
    write_tsv(options.output_dir / "paired_bootstrap_results.tsv", summary_rows)
    (options.output_dir / "paired_bootstrap_replicates.sha256").write_text(
        f"{sha256(replicate_file)}  {replicate_file.name}\n"
        f"{draw_sha}  bootstrap-draw-index-matrix-int32-le\n",
        encoding="utf-8",
    )
    (options.output_dir / "bootstrap.log").write_text(log.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
