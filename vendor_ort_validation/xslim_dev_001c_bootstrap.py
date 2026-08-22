#!/usr/bin/env python3
"""Shared-draw paired COCO bootstrap for frozen DEV-001C surfaces."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
_SURFACES: tuple[str, ...] = ()


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
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.write_text(
        "\t".join(fields)
        + "\n"
        + "".join(
            "\t".join(str(row.get(field, "")) for field in fields) + "\n"
            for row in values
        ),
        encoding="utf-8",
    )


def parse_surface(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
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


def metric_difference(left: float, right: float) -> float:
    if np.isnan(left) and np.isnan(right):
        return 0.0
    if not np.isfinite(left) or not np.isfinite(right):
        return float("inf")
    return abs(left - right)


def official_metrics(evaluator: COCOeval) -> dict[str, float]:
    """Read aggregate metrics from an accumulated literal COCOeval object."""
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])
    return {
        "map50_95": valid_mean(precision[:, :, :, 0, 2]),
        "map50": valid_mean(precision[0, :, :, 0, 2]),
        "map75": valid_mean(precision[iou75, :, :, 0, 2]),
        "ap_small": valid_mean(precision[:, :, :, 1, 2]),
        "ap_medium": valid_mean(precision[:, :, :, 2, 2]),
        "ap_large": valid_mean(precision[:, :, :, 3, 2]),
        "ar1": valid_mean(recall[:, :, 0, 0]),
        "ar10": valid_mean(recall[:, :, 0, 1]),
        "ar100": valid_mean(recall[:, :, 0, 2]),
        "ar_small": valid_mean(recall[:, :, 1, 2]),
        "ar_medium": valid_mean(recall[:, :, 2, 2]),
        "ar_large": valid_mean(recall[:, :, 3, 2]),
    }


def accumulate(
    eval_imgs: list[Any], params: Any, draws: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
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
        for area_index in range(area_count):
            offset = category_offset + area_index * image_count
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


def prepare_surface(
    coco: COCO, ids: list[int], path: Path, log: io.StringIO
) -> dict[str, Any]:
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
        evaluator.accumulate()
    params = evaluator._paramsEval
    if params.imgIds != ids:
        raise RuntimeError("COCOeval changed the sorted image-ID contract")
    point = accumulate(evaluator.evalImgs, params, np.arange(len(ids)), counts)
    literal_point = official_metrics(evaluator)
    literal_point["prediction_count"] = float(len(payload))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "payload": payload,
        "eval_imgs": evaluator.evalImgs,
        "params": params,
        "counts": counts,
        "point": point,
        "literal_point": literal_point,
    }


def literal_bootstrap(
    coco: COCO,
    ids: list[int],
    payload: list[dict[str, Any]],
    draw: np.ndarray,
    log: io.StringIO,
) -> dict[str, float]:
    """Run one exact synthetic-image-ID COCO bootstrap for cross-checking."""
    predictions_by_image: dict[int, list[dict[str, Any]]] = {}
    for row in payload:
        predictions_by_image.setdefault(int(row["image_id"]), []).append(row)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    annotation_id = 1
    for synthetic_id, index in enumerate(draw, 1):
        original_id = ids[int(index)]
        image = dict(coco.imgs[original_id])
        image["id"] = synthetic_id
        images.append(image)
        for source in coco.imgToAnns.get(original_id, []):
            annotation = dict(source)
            annotation["id"] = annotation_id
            annotation["image_id"] = synthetic_id
            annotations.append(annotation)
            annotation_id += 1
        for source in predictions_by_image.get(original_id, []):
            prediction = dict(source)
            prediction["image_id"] = synthetic_id
            predictions.append(prediction)

    synthetic = COCO()
    synthetic.dataset = {
        "info": dict(coco.dataset.get("info", {})),
        "licenses": list(coco.dataset.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": list(coco.dataset["categories"]),
    }
    with contextlib.redirect_stdout(log):
        synthetic.createIndex()
        result = synthetic.loadRes(predictions)
        evaluator = COCOeval(synthetic, result, "bbox")
        evaluator.params.imgIds = list(range(1, len(draw) + 1))
        evaluator.evaluate()
        evaluator.accumulate()
    metrics = official_metrics(evaluator)
    metrics["prediction_count"] = float(len(predictions))
    return metrics


def worker(replicate: int) -> tuple[int, np.ndarray]:
    assert _DRAWS is not None
    draw = _DRAWS[replicate]
    values = np.empty((len(_SURFACES), len(METRICS)), dtype=np.float64)
    for surface_index, surface in enumerate(_SURFACES):
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
        "probability_ge_minus_0_005": float(np.mean(delta >= -0.005)),
        "probability_gt_plus_0_002": float(np.mean(delta > 0.002)),
    }


def sync_memmap(array: np.memmap) -> None:
    array.flush()
    descriptor = os.open(array.filename, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resume_contract(
    options: argparse.Namespace,
    parsed: dict[str, Path],
    draw_sha: str,
) -> dict[str, Any]:
    return {
        "contract_version": "xslim-dev-001c-bootstrap-resume-v1",
        "tool_sha256": sha256(Path(__file__).resolve()),
        "annotations": str(options.annotations.resolve()),
        "annotations_sha256": sha256(options.annotations),
        "image_list": str(options.image_list.resolve()),
        "image_list_sha256": sha256(options.image_list),
        "surfaces": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sorted(parsed.items())
        },
        "surface_order": list(parsed),
        "metrics": list(METRICS),
        "replicates": options.replicates,
        "seed": options.seed,
        "draw_matrix_raw_sha256": draw_sha,
    }


def initialize_resume_state(
    options: argparse.Namespace,
    parsed: dict[str, Path],
    draws: np.ndarray,
    expected_nan: np.ndarray,
) -> tuple[np.memmap, np.memmap]:
    draw_sha = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    contract = resume_contract(options, parsed, draw_sha)
    contract_path = options.output_dir / "resume_contract.json"
    state = options.output_dir / "resume-state"
    if options.output_dir.exists():
        if not contract_path.is_file():
            raise RuntimeError("existing bootstrap root has no resume contract")
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError("bootstrap resume contract differs")
    else:
        options.output_dir.mkdir(parents=True)
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    state.mkdir(exist_ok=True)

    draw_file = options.output_dir / "draw_matrix_int32.npy"
    if draw_file.exists():
        observed_draws = np.load(draw_file, mmap_mode="r", allow_pickle=False)
        if observed_draws.shape != draws.shape or observed_draws.dtype != draws.dtype:
            raise RuntimeError("saved draw matrix shape/dtype differs")
        observed_sha = hashlib.sha256(
            np.asarray(observed_draws).tobytes(order="C")
        ).hexdigest()
        if observed_sha != draw_sha:
            raise RuntimeError("saved draw matrix bytes differ")
    else:
        np.save(draw_file, draws, allow_pickle=False)

    value_path = state / "surface_metrics.float64.memmap"
    done_path = state / "done.uint8.memmap"
    value_mode = "r+" if value_path.exists() else "w+"
    done_mode = "r+" if done_path.exists() else "w+"
    values = np.memmap(
        value_path,
        dtype=np.float64,
        mode=value_mode,
        shape=(options.replicates, len(parsed), len(METRICS)),
    )
    done = np.memmap(
        done_path,
        dtype=np.uint8,
        mode=done_mode,
        shape=(options.replicates,),
    )
    if value_mode == "w+":
        values[:] = np.nan
        sync_memmap(values)
    if done_mode == "w+":
        done[:] = 0
        sync_memmap(done)
    completed_values = np.asarray(values[np.asarray(done, dtype=bool)])
    if completed_values.size and invalid_metric_values(completed_values, expected_nan):
        raise RuntimeError("completed bootstrap resume rows violate the value contract")
    return values, done


def invalid_metric_values(values: np.ndarray, expected_nan: np.ndarray) -> bool:
    expected = np.broadcast_to(expected_nan, values.shape)
    return bool(np.any(np.isinf(values)) or np.any(np.isnan(values) != expected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--surface", required=True, action="append")
    parser.add_argument("--baseline", default="B2")
    parser.add_argument("--pareto-reference", default="A1")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=65007)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.replicates < 10_000:
        raise ValueError("DEV-001C requires at least 10000 shared draws")
    if options.checkpoint_every < 1:
        raise ValueError("checkpoint interval must be positive")
    parsed_entries = [parse_surface(value) for value in options.surface]
    parsed = dict(parsed_entries)
    if len(parsed) != len(parsed_entries) or set(parsed) != {"B2", "A1", "C2"}:
        raise ValueError("DEV-001C requires exactly B2, A1 and C2 surfaces")
    if options.baseline not in parsed or options.pareto_reference not in parsed:
        raise ValueError("baseline and Pareto reference surfaces are required")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco = COCO(str(options.annotations))
    ids = selected_ids(options.image_list, coco)
    global _BASES, _DRAWS, _SURFACES
    _SURFACES = tuple(parsed)
    _BASES = {
        name: prepare_surface(coco, ids, parsed[name], log) for name in _SURFACES
    }
    rng = np.random.default_rng(options.seed)
    _DRAWS = rng.integers(
        0, len(ids), size=(options.replicates, len(ids)), dtype=np.int32
    )
    draw_sha = hashlib.sha256(_DRAWS.tobytes(order="C")).hexdigest()
    expected_nan = np.asarray(
        [
            [np.isnan(_BASES[surface]["point"][metric]) for metric in METRICS]
            for surface in _SURFACES
        ],
        dtype=bool,
    )
    values, done = initialize_resume_state(options, parsed, _DRAWS, expected_nan)

    validation_rows: list[dict[str, Any]] = []
    for surface in _SURFACES:
        for metric in METRICS:
            vectorized = _BASES[surface]["point"][metric]
            literal = _BASES[surface]["literal_point"][metric]
            difference = metric_difference(vectorized, literal)
            validation_rows.append(
                {
                    "check": "point-literal-vs-vectorized",
                    "surface": surface,
                    "metric": metric,
                    "literal": literal,
                    "vectorized": vectorized,
                    "absolute_difference": difference,
                    "tolerance": 1.0e-12,
                    "status": "pass" if difference <= 1.0e-12 else "fail",
                }
            )
        literal = literal_bootstrap(
            coco, ids, _BASES[surface]["payload"], _DRAWS[0], log
        )
        vectorized = accumulate(
            _BASES[surface]["eval_imgs"],
            _BASES[surface]["params"],
            _DRAWS[0],
            _BASES[surface]["counts"],
        )
        for metric in METRICS:
            difference = metric_difference(vectorized[metric], literal[metric])
            validation_rows.append(
                {
                    "check": "bootstrap-draw-0-literal-vs-vectorized",
                    "surface": surface,
                    "metric": metric,
                    "literal": literal[metric],
                    "vectorized": vectorized[metric],
                    "absolute_difference": difference,
                    "tolerance": 1.0e-12,
                    "status": "pass" if difference <= 1.0e-12 else "fail",
                }
            )
    write_tsv(options.output_dir / "literal_vectorized_crosscheck.tsv", validation_rows)
    failed_validation = [row for row in validation_rows if row["status"] != "pass"]
    if failed_validation:
        raise RuntimeError(
            f"literal/vectorized COCO cross-check failed in {len(failed_validation)} rows"
        )

    metrics_rows = []
    for surface in _SURFACES:
        row = {
            "surface": surface,
            "images": len(ids),
            "prediction_sha256": _BASES[surface]["sha256"],
        }
        row.update(_BASES[surface]["point"])
        metrics_rows.append(row)
    write_tsv(options.output_dir / "complete_metrics.tsv", metrics_rows)

    pending = [int(index) for index in np.flatnonzero(done == 0)]
    print(
        f"bootstrap resume: completed={int(done.sum())} "
        f"pending={len(pending)} total={options.replicates}",
        flush=True,
    )
    if options.workers == 1:
        results = map(worker, pending)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(options.workers)
        results = pool.imap_unordered(worker, pending, chunksize=1)
    checkpoint: list[int] = []
    try:
        for replicate, result in results:
            values[replicate] = result
            checkpoint.append(replicate)
            if len(checkpoint) >= options.checkpoint_every:
                sync_memmap(values)
                done[checkpoint] = 1
                sync_memmap(done)
                checkpoint.clear()
            completed = int(done.sum()) + len(checkpoint)
            if completed % 100 == 0:
                print(f"bootstrap {completed}/{options.replicates}", flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        if checkpoint:
            sync_memmap(values)
            done[checkpoint] = 1
            sync_memmap(done)
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    if checkpoint:
        sync_memmap(values)
        done[checkpoint] = 1
        sync_memmap(done)
    if not np.all(done == 1) or invalid_metric_values(values, expected_nan):
        raise RuntimeError("cannot finalize incomplete bootstrap resume state")

    pair_rows = []
    pairs = [
        ("C2-vs-B2", "C2", "B2"),
        ("C2-vs-A1", "C2", "A1"),
        ("A1-vs-B2", "A1", "B2"),
    ]
    for pair_name, left, right in pairs:
        left_index = _SURFACES.index(left)
        right_index = _SURFACES.index(right)
        for metric_index, metric in enumerate(METRICS):
            delta = values[:, left_index, metric_index] - values[:, right_index, metric_index]
            row = {
                "pair": pair_name,
                "left": left,
                "right": right,
                "metric": metric,
                "replicates": options.replicates,
                "seed": options.seed,
                "draw_matrix_sha256": draw_sha,
                "point_delta": _BASES[left]["point"][metric] - _BASES[right]["point"][metric],
            }
            row.update(summarize(delta))
            pair_rows.append(row)
    write_tsv(options.output_dir / "complete_bootstrap.tsv", pair_rows)

    replicate_file = options.output_dir / "surface_replicates.npz"
    np.savez_compressed(
        replicate_file,
        surface_metrics=values,
        surfaces=np.asarray(_SURFACES),
        metrics=np.asarray(METRICS),
        seed=np.asarray(options.seed),
        draw_sha256=np.asarray(draw_sha),
    )
    draw_file = options.output_dir / "draw_matrix_int32.npy"
    (options.output_dir / "replicates.sha256").write_text(
        f"{sha256(replicate_file)}  {replicate_file.name}\n"
        f"{sha256(draw_file)}  {draw_file.name}\n"
        f"{draw_sha}  bootstrap-draw-index-matrix-int32-le\n",
        encoding="utf-8",
    )
    (options.output_dir / "bootstrap.log").write_text(log.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
