#!/usr/bin/env python3
"""Run the gated Stage65B-R1 H500, hybrid, and full-COCO matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


LANES = ("B1", "B2", "B3", "B4", "B5", "B6")
HYBRID_ARMS = ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")
FULL_HYBRID_ARMS = ("H0", "H1", "H3", "H5", "H6", "H8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--postprocess-root", required=True, type=Path)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--fp32-tail", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--scout-list", required=True, type=Path)
    parser.add_argument("--val-list", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def model_paths(root: Path, lane: str) -> tuple[Path, Path]:
    base = root / lane / "candidate-gate" / lane / "models"
    prefix = f"stage65b_r1_{lane.lower()}"
    inference = base / f"{prefix}.inference.onnx"
    tail = base / f"{prefix}.postprocess.onnx"
    if not inference.is_file() or not tail.is_file():
        raise FileNotFoundError(f"missing gated model for {lane}: {base}")
    return inference, tail


def selected_lane(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for lane in LANES:
        summary = read_tsv(root / lane / "postprocess-summary.tsv")
        if len(summary) != 1 or summary[0]["status"] != "pass":
            raise RuntimeError(f"postprocess gate did not pass for {lane}")
        metrics = read_tsv(root / lane / "scout500" / "metrics" / "results.tsv")
        if len(metrics) != 1:
            raise RuntimeError(f"invalid scout metrics for {lane}")
        metric = metrics[0]
        rows.append(
            {
                "lane": lane,
                "model_sha256": summary[0]["inference_model_sha256"],
                "scout_map50_95": metric["map50_95"],
                "scout_map50": metric["map50"],
                "scout_ap_small": metric["ap_small"],
                "scout_ap_medium": metric["ap_medium"],
                "scout_ap_large": metric["ap_large"],
                "prediction_sha256": metric["prediction_sha256"],
            }
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["scout_map50_95"]),
            -float(row["scout_ap_large"]),
            row["lane"],
        ),
    )
    for rank, row in enumerate(ordered, 1):
        row["selection_rank"] = rank
        row["selected_global_candidate"] = int(rank == 1)
    return str(ordered[0]["lane"]), ordered


class Recorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def run(self, name: str, command: list[str]) -> None:
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row = {
            "step": name,
            "returncode": process.returncode,
            "elapsed_seconds": f"{time.monotonic() - started:.6f}",
            "log": str(log),
            "command_json": json.dumps(command, separators=(",", ":")),
        }
        with self.lock:
            self.rows.append(row)
            write_tsv(self.root / "command-log.tsv", self.rows)
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit {process.returncode}")


def evaluate_command(
    options: argparse.Namespace,
    lane: str,
    image_list: Path,
    output: Path,
    limit: int = 0,
) -> list[str]:
    inference, tail = model_paths(options.postprocess_root, lane)
    command = [
        str(options.python),
        str(options.repo / "vendor_ort_validation/stage65b_r1_evaluate.py"),
        "run",
        "--candidate-inference",
        str(inference),
        "--tail",
        str(tail),
        "--image-list",
        str(image_list),
        "--output-dir",
        str(output),
        "--name",
        lane,
        "--threads",
        str(options.threads),
        "--log-every",
        "100",
    ]
    if limit:
        command.extend(["--limit", str(limit)])
    return command


def hybrid_command(
    options: argparse.Namespace,
    lane: str,
    image_list: Path,
    output: Path,
    arms: tuple[str, ...],
    limit: int = 0,
) -> list[str]:
    inference, tail = model_paths(options.postprocess_root, lane)
    command = [
        str(options.python),
        str(options.repo / "vendor_ort_validation/stage65b_r1_evaluate.py"),
        "hybrid",
        "--candidate-inference",
        str(inference),
        "--fp32-inference",
        str(options.fp32_inference),
        "--tail",
        str(tail),
        "--image-list",
        str(image_list),
        "--output-dir",
        str(output),
        "--arms",
        ",".join(arms),
        "--threads",
        str(options.threads),
        "--log-every",
        "100",
    ]
    if limit:
        command.extend(["--limit", str(limit)])
    return command


def metrics_command(
    options: argparse.Namespace,
    surface: str,
    predictions: Path,
    image_list: Path,
    output: Path,
) -> list[str]:
    return [
        str(options.python),
        str(options.repo / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
        "--annotations",
        str(options.annotations),
        "--predictions",
        str(predictions),
        "--image-list",
        str(image_list),
        "--surface",
        surface,
        "--output-dir",
        str(output),
    ]


def parallel_pair(
    left: Callable[[], None], right: Callable[[], None]
) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(left), executor.submit(right)]
        for future in futures:
            future.result()


def main() -> int:
    options = parse_args()
    if options.threads < 1:
        raise ValueError("--threads must be positive")
    root = options.output_root
    if root.exists():
        raise RuntimeError(f"refusing to reuse full-matrix root: {root}")
    root.mkdir(parents=True)
    recorder = Recorder(root)
    best, scout_rows = selected_lane(options.postprocess_root)
    write_tsv(root / "scout-selection.tsv", scout_rows)
    (root / "selected-global-candidate.txt").write_text(
        f"{best}\n", encoding="utf-8"
    )

    best_inference, best_tail = model_paths(options.postprocess_root, best)
    h500_semantic = root / "host-h500" / best
    h500_100 = root / "hybrid-h500-100"
    parallel_pair(
        lambda: recorder.run(
            f"host-h500-{best}",
            [
                str(options.python),
                str(options.repo / "vendor_ort_validation/stage64_host_validate.py"),
                "--source-model",
                str(options.source_model),
                "--fp32-inference",
                str(options.fp32_inference),
                "--fp32-tail",
                str(options.fp32_tail),
                "--candidate-inference",
                str(best_inference),
                "--candidate-tail",
                str(best_tail),
                "--image-list",
                str(options.h500_list),
                "--output-dir",
                str(h500_semantic),
                "--candidate-name",
                best,
            ],
        ),
        lambda: recorder.run(
            "hybrid-h500-100",
            hybrid_command(
                options, best, options.h500_list, h500_100, HYBRID_ARMS, 100
            ),
        ),
    )

    h500_all = root / "hybrid-h500-all"
    scout_hybrid = root / "hybrid-scout500"
    parallel_pair(
        lambda: recorder.run(
            "hybrid-h500-all",
            hybrid_command(
                options, best, options.h500_list, h500_all, HYBRID_ARMS
            ),
        ),
        lambda: recorder.run(
            "hybrid-scout500",
            hybrid_command(
                options, best, options.scout_list, scout_hybrid, HYBRID_ARMS
            ),
        ),
    )
    for arm in HYBRID_ARMS:
        recorder.run(
            f"hybrid-scout500-metrics-{arm}",
            metrics_command(
                options,
                arm,
                scout_hybrid / arm / "predictions.json",
                options.scout_list,
                scout_hybrid / arm / "metrics",
            ),
        )

    def full_candidate(lane: str) -> None:
        output = root / "full-coco" / lane
        recorder.run(
            f"full-coco-predict-{lane}",
            evaluate_command(options, lane, options.val_list, output),
        )
        recorder.run(
            f"full-coco-metrics-{lane}",
            metrics_command(
                options,
                lane,
                output / "predictions.json",
                options.val_list,
                output / "metrics",
            ),
        )

    parallel_pair(lambda: full_candidate("B1"), lambda: full_candidate("B2"))
    parallel_pair(lambda: full_candidate("B3"), lambda: full_candidate("B4"))

    hybrid_full = root / "hybrid-full-coco"

    def full_hybrid() -> None:
        recorder.run(
            "hybrid-full-coco-predict",
            hybrid_command(
                options,
                best,
                options.val_list,
                hybrid_full,
                FULL_HYBRID_ARMS,
            ),
        )
        for arm in FULL_HYBRID_ARMS:
            recorder.run(
                f"hybrid-full-coco-metrics-{arm}",
                metrics_command(
                    options,
                    arm,
                    hybrid_full / arm / "predictions.json",
                    options.val_list,
                    hybrid_full / arm / "metrics",
                ),
            )

    def final_candidates() -> None:
        full_candidate("B5")
        full_candidate("B6")

    parallel_pair(full_hybrid, final_candidates)

    summary = {
        "status": "pass",
        "selected_global_candidate": best,
        "selected_inference_sha256": sha256(best_inference),
        "selected_tail_sha256": sha256(best_tail),
        "candidate_full_coco_surfaces": ",".join(LANES),
        "hybrid_full_coco_surfaces": ",".join(FULL_HYBRID_ARMS),
        "h500_semantic_images": 500,
        "hybrid_h500_images": 500,
        "hybrid_scout_images": 500,
        "full_coco_images": 5000,
    }
    write_tsv(root / "full-matrix-summary.tsv", [summary])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
