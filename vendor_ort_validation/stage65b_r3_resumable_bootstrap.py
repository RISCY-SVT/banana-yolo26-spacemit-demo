#!/usr/bin/env python3
"""Resume an exact Stage65B paired bootstrap without changing its worker math."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO

import stage65b_r2_bootstrap as accepted


ACCEPTED_TOOL_SHA256 = "5d3649908f7f0cf3ff02e133dfbeb58504baa14afac102ea0de9d62af916d245"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_pair(raw: str) -> tuple[str, str, str]:
    return accepted.parse_pair(raw)


def sync_memmap(array: np.memmap) -> None:
    array.flush()
    descriptor = os.open(array.filename, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def expected_contract(options: argparse.Namespace, pair: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "contract_version": "stage65b-r3-resumable-bootstrap-v1",
        "accepted_tool_sha256": ACCEPTED_TOOL_SHA256,
        "annotations": str(options.annotations.resolve()),
        "annotations_sha256": sha256(options.annotations),
        "image_list": str(options.image_list.resolve()),
        "image_list_sha256": sha256(options.image_list),
        "pair": pair[0],
        "left": pair[1],
        "left_sha256": sha256(Path(pair[1])),
        "right": pair[2],
        "right_sha256": sha256(Path(pair[2])),
        "seed": options.seed,
        "replicates": options.replicates,
        "metrics": list(accepted.METRICS),
        "screening_npz": str(options.screening_npz.resolve()),
        "screening_npz_sha256": sha256(options.screening_npz),
    }


def initialize_state(
    options: argparse.Namespace,
    pair: tuple[str, str, str],
    draws: np.ndarray,
) -> tuple[np.memmap, np.memmap, int]:
    state = options.output_dir / "resume-state"
    state.mkdir(parents=True, exist_ok=True)
    contract_path = state / "contract.json"
    contract = expected_contract(options, pair)
    if contract_path.exists():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError("resume contract differs from the accepted state")
    else:
        temporary = contract_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(contract_path)

    value_path = state / "deltas.float64.memmap"
    done_path = state / "done.uint8.memmap"
    value_mode = "r+" if value_path.exists() else "w+"
    done_mode = "r+" if done_path.exists() else "w+"
    values = np.memmap(
        value_path,
        dtype=np.float64,
        mode=value_mode,
        shape=(options.replicates, len(accepted.METRICS)),
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

    with np.load(options.screening_npz, allow_pickle=False) as screening:
        if list(screening["metrics"]) != list(accepted.METRICS):
            raise RuntimeError("screening metric contract differs")
        names = [str(item) for item in screening["pairs"]]
        if pair[0] not in names:
            raise RuntimeError(f"pair absent from screening NPZ: {pair[0]}")
        pair_index = names.index(pair[0])
        seed = int(screening["seed"])
        if seed != options.seed:
            raise RuntimeError("screening seed differs")
        prefix = screening["deltas"][pair_index]
        prefix_count = int(prefix.shape[0])
        if prefix_count >= options.replicates:
            raise RuntimeError("screening prefix must be smaller than final replicate count")
        draw_prefix_sha = hashlib.sha256(
            draws[:prefix_count].tobytes(order="C")
        ).hexdigest()
        if draw_prefix_sha != str(screening["draw_sha256"]):
            raise RuntimeError("screening draw matrix is not a prefix of final draws")
        existing_prefix = np.asarray(values[:prefix_count])
        completed_prefix = np.asarray(done[:prefix_count], dtype=bool)
        if np.any(completed_prefix) and not np.array_equal(
            existing_prefix[completed_prefix], prefix[completed_prefix]
        ):
            raise RuntimeError("resume state differs from screening prefix")
        values[:prefix_count] = prefix
        sync_memmap(values)
        done[:prefix_count] = 1
        sync_memmap(done)
    return values, done, prefix_count


def literal_validation(
    coco: COCO,
    ids: list[int],
    base: dict[str, Any],
    draws: np.ndarray,
    pair_name: str,
    side: str,
) -> list[dict[str, Any]]:
    literal = accepted.remapped_validation(coco, ids, base, draws[0])
    cached = accepted.accumulate_metrics(base["evalImgs"], base["params"], draws[0])
    rows: list[dict[str, Any]] = []
    for metric in accepted.METRICS:
        difference = abs(literal[metric] - cached[metric])
        rows.append(
            {
                "pair": pair_name,
                "side": side,
                "metric": metric,
                "literal_synthetic_id": literal[metric],
                "cached_match": cached[metric],
                "absolute_difference": difference,
                "status": "pass" if difference <= 1e-12 else "fail",
            }
        )
    return rows


def finalize(
    options: argparse.Namespace,
    pair: tuple[str, str, str],
    values: np.memmap,
    done: np.memmap,
    draws: np.ndarray,
    bases: dict[str, dict[str, Any]],
    validation: list[dict[str, Any]],
    prefix_count: int,
) -> None:
    if not np.all(done == 1) or not np.all(np.isfinite(values)):
        raise RuntimeError("cannot finalize incomplete bootstrap state")
    materialized = np.asarray(values).copy()
    replicate_file = options.output_dir / "paired_bootstrap_replicates.npz"
    np.savez_compressed(
        replicate_file,
        deltas=materialized[np.newaxis, :, :],
        metrics=np.asarray(accepted.METRICS),
        pairs=np.asarray([pair[0]]),
        draw_sha256=np.asarray(hashlib.sha256(draws.tobytes(order="C")).hexdigest()),
        seed=np.asarray(options.seed),
    )
    left, right = bases[pair[1]], bases[pair[2]]
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(accepted.METRICS):
        delta = materialized[:, metric_index]
        rows.append(
            {
                "pair": pair[0],
                "left_prediction_sha256": left["sha256"],
                "right_prediction_sha256": right["sha256"],
                "metric": metric,
                "replicates": options.replicates,
                "seed": options.seed,
                "reused_screening_prefix_replicates": prefix_count,
                "point_delta": left["point"][metric] - right["point"][metric],
                "bootstrap_mean": float(np.mean(delta)),
                "bootstrap_median": float(np.median(delta)),
                "percentile_2_5": float(np.percentile(delta, 2.5)),
                "percentile_97_5": float(np.percentile(delta, 97.5)),
                "probability_delta_gt_zero": float(np.mean(delta > 0.0)),
            }
        )
    write_tsv(options.output_dir / "paired_bootstrap_results.tsv", rows)
    write_tsv(options.output_dir / "synthetic_id_validation.tsv", validation)
    draw_sha = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    (options.output_dir / "paired_bootstrap_replicates.sha256").write_text(
        f"{sha256(replicate_file)}  {replicate_file.name}\n"
        f"{draw_sha}  bootstrap-draw-index-matrix-int32-le\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-tool", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--screening-npz", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=65003)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.replicates < 10000:
        raise ValueError("final top-region comparison requires at least 10000 replicates")
    if sha256(options.accepted_tool) != ACCEPTED_TOOL_SHA256:
        raise RuntimeError("accepted Stage65B-R2 bootstrap source hash differs")
    pair = parse_pair(options.pair)
    options.output_dir.mkdir(parents=True, exist_ok=True)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco = COCO(str(options.annotations))
    ids = accepted.selected_ids(options.image_list, coco)
    rng = np.random.default_rng(options.seed)
    draws = rng.integers(
        0,
        len(ids),
        size=(options.replicates, len(ids)),
        dtype=np.int32,
    )
    values, done, prefix_count = initialize_state(options, pair, draws)

    unique_paths = sorted({pair[1], pair[2]})
    bases = {
        path: accepted.prepare_base(coco, ids, Path(path), log)
        for path in unique_paths
    }
    accepted._BASES = bases
    accepted._DRAWS = draws
    accepted._PAIRS = [pair]
    validation: list[dict[str, Any]] = []
    for side, key in (("left", pair[1]), ("right", pair[2])):
        validation.extend(literal_validation(coco, ids, bases[key], draws, pair[0], side))
    if any(row["status"] != "pass" for row in validation):
        raise RuntimeError("cached bootstrap differs from literal synthetic-ID remap")

    pending = [int(index) for index in np.flatnonzero(done == 0)]
    print(
        f"resume-bootstrap: reused={prefix_count} completed={int(done.sum())} "
        f"pending={len(pending)} total={options.replicates}",
        flush=True,
    )
    if pending:
        context = mp.get_context("fork")
        pool = context.Pool(processes=options.workers)
        tasks = ((0, replicate) for replicate in pending)
        try:
            results = pool.imap_unordered(accepted.worker, tasks, chunksize=1)
            since_sync = 0
            for _, replicate, row in results:
                values[replicate, :] = row
                values.flush()
                done[replicate] = 1
                since_sync += 1
                if since_sync >= options.checkpoint_every:
                    sync_memmap(values)
                    sync_memmap(done)
                    since_sync = 0
                completed = int(done.sum())
                if completed % 100 == 0:
                    print(
                        f"resume-bootstrap: {completed}/{options.replicates}", flush=True
                    )
        except BaseException:
            pool.terminate()
            pool.join()
            sync_memmap(values)
            sync_memmap(done)
            raise
        else:
            pool.close()
            pool.join()
    sync_memmap(values)
    sync_memmap(done)
    finalize(options, pair, values, done, draws, bases, validation, prefix_count)
    (options.output_dir / "bootstrap.log").write_text(log.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
