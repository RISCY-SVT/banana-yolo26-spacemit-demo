#!/usr/bin/env python3
"""Shared-draw four-surface COCO bootstrap and C2 provider interaction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SURFACES = ("B2_CPU", "B2_EP", "C2_CPU", "C2_EP")
PAIRS = (
    ("C2_EP-vs-B2_EP", "C2_EP", "B2_EP"),
    ("C2_CPU-vs-B2_CPU", "C2_CPU", "B2_CPU"),
    ("C2_EP-vs-C2_CPU", "C2_EP", "C2_CPU"),
    ("B2_EP-vs-B2_CPU", "B2_EP", "B2_CPU"),
)
METRICS = (
    "map50_95",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
    "ar_small",
    "ar_medium",
    "ar_large",
    "prediction_count",
)

_BASES: dict[str, dict[str, Any]] = {}
_DRAWS: np.ndarray | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields = list(values[0])
    path.write_text(
        "\t".join(fields)
        + "\n"
        + "".join("\t".join(str(row[field]) for field in fields) + "\n" for row in values),
        encoding="utf-8",
    )


def parse_surface(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or name not in SURFACES or not path:
        raise ValueError(f"invalid --surface: {raw}")
    return name, Path(path).resolve()


def selected_ids(path: Path, coco: COCO) -> list[int]:
    values = sorted(int(Path(line).stem) for line in path.read_text().splitlines() if line)
    if not values or len(values) != len(set(values)):
        raise ValueError("image list is empty or contains duplicate IDs")
    missing = set(values) - set(coco.imgs)
    if missing:
        raise ValueError(f"{len(missing)} image IDs are absent from annotations")
    return values


def valid_mean(values: np.ndarray) -> float:
    selected = values[values >= 0]
    return float(selected.mean()) if selected.size else float("nan")


def accumulate(eval_imgs: list[Any], params: Any, draws: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    t_count = len(params.iouThrs)
    r_count = len(params.recThrs)
    k_count = len(params.catIds)
    a_count = len(params.areaRng)
    i_count = len(params.imgIds)
    precision = -np.ones((t_count, r_count, k_count, a_count), dtype=np.float64)
    recall_by_max = {
        limit: -np.ones((t_count, k_count, a_count), dtype=np.float64)
        for limit in (1, 10, 100)
    }
    for category_index in range(k_count):
        category_offset = category_index * a_count * i_count
        for area_index in range(a_count):
            offset = category_offset + area_index * i_count
            selected = [eval_imgs[offset + int(index)] for index in draws]
            selected = [item for item in selected if item is not None]
            if not selected:
                continue
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

            scores = np.concatenate([item["dtScores"][:100] for item in selected])
            order = np.argsort(-scores, kind="mergesort")
            matches = np.concatenate(
                [item["dtMatches"][:, :100] for item in selected], axis=1
            )[:, order]
            ignored = np.concatenate(
                [item["dtIgnore"][:, :100] for item in selected], axis=1
            )[:, order]
            true_positive = np.logical_and(matches, np.logical_not(ignored))
            false_positive = np.logical_and(np.logical_not(matches), np.logical_not(ignored))
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

    iou75 = int(np.flatnonzero(np.isclose(params.iouThrs, 0.75))[0])
    return {
        "map50_95": valid_mean(precision[:, :, :, 0]),
        "map50": valid_mean(precision[0, :, :, 0]),
        "map75": valid_mean(precision[iou75, :, :, 0]),
        "ap_small": valid_mean(precision[:, :, :, 1]),
        "ap_medium": valid_mean(precision[:, :, :, 2]),
        "ap_large": valid_mean(precision[:, :, :, 3]),
        "ar1": valid_mean(recall_by_max[1][:, :, 0]),
        "ar10": valid_mean(recall_by_max[10][:, :, 0]),
        "ar100": valid_mean(recall_by_max[100][:, :, 0]),
        "ar_small": valid_mean(recall_by_max[100][:, :, 1]),
        "ar_medium": valid_mean(recall_by_max[100][:, :, 2]),
        "ar_large": valid_mean(recall_by_max[100][:, :, 3]),
        "prediction_count": float(np.sum(counts[draws], dtype=np.int64)),
    }


def prepare(coco: COCO, ids: list[int], path: Path, log: io.StringIO) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"prediction payload is not an array: {path}")
    allowed = set(ids)
    unexpected = {int(row["image_id"]) for row in payload} - allowed
    if unexpected:
        raise ValueError(f"prediction payload contains {len(unexpected)} unexpected IDs")
    counts_by_id: dict[int, int] = {}
    for row in payload:
        image = int(row["image_id"])
        counts_by_id[image] = counts_by_id.get(image, 0) + 1
    counts = np.asarray([counts_by_id.get(image, 0) for image in ids], dtype=np.int32)
    with contextlib.redirect_stdout(log):
        result = coco.loadRes(payload)
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
    params = evaluator._paramsEval
    if params.imgIds != ids:
        raise RuntimeError("COCOeval changed the sorted image-ID contract")
    point = accumulate(evaluator.evalImgs, params, np.arange(len(ids)), counts)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "eval_imgs": evaluator.evalImgs,
        "params": params,
        "counts": counts,
        "point": point,
    }


def worker(replicate: int) -> tuple[int, np.ndarray]:
    assert _DRAWS is not None
    draw = _DRAWS[replicate]
    values = np.empty((len(SURFACES), len(METRICS)), dtype=np.float64)
    for surface_index, surface in enumerate(SURFACES):
        base = _BASES[surface]
        metrics = accumulate(base["eval_imgs"], base["params"], draw, base["counts"])
        values[surface_index] = [metrics[name] for name in METRICS]
    return replicate, values


def summarize(delta: np.ndarray) -> dict[str, float]:
    return {
        "bootstrap_mean": float(np.mean(delta)),
        "bootstrap_median": float(np.median(delta)),
        "percentile_2_5": float(np.percentile(delta, 2.5)),
        "percentile_97_5": float(np.percentile(delta, 97.5)),
        "probability_gt_zero": float(np.mean(delta > 0)),
        "probability_lt_zero": float(np.mean(delta < 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--surface", required=True, action="append")
    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.replicates < 10000:
        raise ValueError("Stage65D requires at least 10000 replicates")
    if options.output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {options.output_dir}")
    parsed = dict(parse_surface(value) for value in options.surface)
    if tuple(sorted(parsed)) != tuple(sorted(SURFACES)) or len(options.surface) != 4:
        raise ValueError(f"exactly one path is required for each of {SURFACES}")
    options.output_dir.mkdir(parents=True)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco = COCO(str(options.annotations))
    ids = selected_ids(options.image_list, coco)
    global _BASES, _DRAWS
    _BASES = {name: prepare(coco, ids, parsed[name], log) for name in SURFACES}
    rng = np.random.default_rng(options.seed)
    _DRAWS = rng.integers(0, len(ids), size=(options.replicates, len(ids)), dtype=np.int32)
    draw_sha = hashlib.sha256(_DRAWS.tobytes(order="C")).hexdigest()

    values = np.empty((options.replicates, len(SURFACES), len(METRICS)), dtype=np.float64)
    if options.workers == 1:
        results = map(worker, range(options.replicates))
        pool = None
    else:
        pool = mp.get_context("fork").Pool(options.workers)
        results = pool.imap_unordered(worker, range(options.replicates), chunksize=1)
    try:
        for completed, (replicate, result) in enumerate(results, 1):
            values[replicate] = result
            if completed % 100 == 0:
                print(f"bootstrap {completed}/{options.replicates}", flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    point_rows = []
    for surface in SURFACES:
        row = {
            "surface": surface,
            "images": len(ids),
            "prediction_sha256": _BASES[surface]["sha256"],
        }
        row.update(_BASES[surface]["point"])
        point_rows.append(row)
    write_tsv(options.output_dir / "complete_metrics.tsv", point_rows)

    pair_rows = []
    for pair_name, left, right in PAIRS:
        left_index = SURFACES.index(left)
        right_index = SURFACES.index(right)
        for metric_index, metric in enumerate(METRICS):
            delta = values[:, left_index, metric_index] - values[:, right_index, metric_index]
            row = {
                "pair": pair_name,
                "metric": metric,
                "replicates": options.replicates,
                "seed": options.seed,
                "draw_matrix_sha256": draw_sha,
                "point_delta": _BASES[left]["point"][metric] - _BASES[right]["point"][metric],
            }
            row.update(summarize(delta))
            pair_rows.append(row)
    write_tsv(options.output_dir / "complete_bootstrap.tsv", pair_rows)

    c2_ep = SURFACES.index("C2_EP")
    c2_cpu = SURFACES.index("C2_CPU")
    b2_ep = SURFACES.index("B2_EP")
    b2_cpu = SURFACES.index("B2_CPU")
    interaction_rows = []
    for metric_index, metric in enumerate(METRICS):
        interaction = (
            values[:, c2_ep, metric_index]
            - values[:, c2_cpu, metric_index]
            - values[:, b2_ep, metric_index]
            + values[:, b2_cpu, metric_index]
        )
        point = (
            _BASES["C2_EP"]["point"][metric]
            - _BASES["C2_CPU"]["point"][metric]
            - _BASES["B2_EP"]["point"][metric]
            + _BASES["B2_CPU"]["point"][metric]
        )
        row = {
            "metric": metric,
            "replicates": options.replicates,
            "seed": options.seed,
            "draw_matrix_sha256": draw_sha,
            "point_interaction": point,
        }
        row.update(summarize(interaction))
        interaction_rows.append(row)
    write_tsv(options.output_dir / "interaction_bootstrap.tsv", interaction_rows)

    replicate_file = options.output_dir / "four_surface_replicates.npz"
    np.savez_compressed(
        replicate_file,
        surface_metrics=values,
        surfaces=np.asarray(SURFACES),
        metrics=np.asarray(METRICS),
        seed=np.asarray(options.seed),
        draw_sha256=np.asarray(draw_sha),
    )
    (options.output_dir / "replicates.sha256").write_text(
        f"{sha256(replicate_file)}  {replicate_file.name}\n"
        f"{draw_sha}  bootstrap-draw-index-matrix-int32-le\n",
        encoding="utf-8",
    )
    (options.output_dir / "bootstrap.log").write_text(log.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
