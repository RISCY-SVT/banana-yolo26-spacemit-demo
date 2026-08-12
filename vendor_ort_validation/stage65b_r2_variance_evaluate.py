#!/usr/bin/env python3
"""Run the bounded Stage65B-R2 host gates for one B2 variance probe."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from stage65b_r2_common import sha256, write_tsv


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lane", required=True)
    result.add_argument("--run-root", required=True, type=Path)
    result.add_argument("--python", required=True, type=Path)
    result.add_argument("--repo", required=True, type=Path)
    result.add_argument("--source-model", required=True, type=Path)
    result.add_argument("--fp32-inference", required=True, type=Path)
    result.add_argument("--fp32-tail", required=True, type=Path)
    result.add_argument("--expected-tail-sha256", required=True)
    result.add_argument("--h500-list", required=True, type=Path)
    result.add_argument("--scout-list", required=True, type=Path)
    result.add_argument("--train-annotations", required=True, type=Path)
    result.add_argument("--val-annotations", required=True, type=Path)
    result.add_argument("--tensor-list", required=True, type=Path)
    result.add_argument("--preprocess", required=True)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--threads", type=int, default=4)
    return result


def invoke(
    name: str,
    command: list[str],
    log: Path,
    rows: list[dict[str, Any]],
) -> None:
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    rows.append(
        {
            "step": name,
            "returncode": process.returncode,
            "elapsed_seconds": f"{time.monotonic() - started:.6f}",
            "log": str(log),
            "command_json": json.dumps(command, separators=(",", ":")),
        }
    )
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}")


def read_one(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def main() -> int:
    options = parser().parse_args()
    lane = options.lane.upper()
    root = options.output_root / lane
    if root.exists():
        raise RuntimeError(f"refusing to reuse variance evaluation root: {root}")
    root.mkdir(parents=True)
    logs = root / "logs"
    commands: list[dict[str, Any]] = []

    postprocess_root = root / "postprocess"
    invoke(
        "r1-postprocess-gates",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_postprocess.py"),
            "--lane",
            lane,
            "--run-root",
            str(options.run_root),
            "--python",
            str(options.python),
            "--repo",
            str(options.repo),
            "--source-model",
            str(options.source_model),
            "--fp32-inference",
            str(options.fp32_inference),
            "--fp32-tail",
            str(options.fp32_tail),
            "--expected-tail-sha256",
            options.expected_tail_sha256,
            "--h500-list",
            str(options.h500_list),
            "--scout-list",
            str(options.scout_list),
            "--annotations",
            str(options.val_annotations),
            "--tensor-list",
            str(options.tensor_list),
            "--preprocess",
            options.preprocess,
            "--output-root",
            str(postprocess_root),
            "--threads",
            str(options.threads),
        ],
        logs / "r1-postprocess-gates.log",
        commands,
    )

    lane_root = postprocess_root / lane
    summary = read_one(lane_root / "postprocess-summary.tsv")
    inference = Path(summary["inference_model"])
    tail = Path(summary["tail_model"])
    h500 = root / "h500"
    invoke(
        "h500-predict",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_evaluate.py"),
            "run",
            "--candidate-inference",
            str(inference),
            "--tail",
            str(tail),
            "--image-list",
            str(options.h500_list),
            "--output-dir",
            str(h500),
            "--name",
            lane,
            "--threads",
            str(options.threads),
            "--log-every",
            "50",
        ],
        logs / "h500-predict.log",
        commands,
    )
    metrics = root / "h500-metrics"
    invoke(
        "h500-metrics",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
            "--annotations",
            str(options.train_annotations),
            "--predictions",
            str(h500 / "predictions.json"),
            "--image-list",
            str(options.h500_list),
            "--surface",
            lane,
            "--output-dir",
            str(metrics),
        ],
        logs / "h500-metrics.log",
        commands,
    )

    h500_result = read_one(metrics / "results.tsv")
    scout_result = read_one(lane_root / "scout500/metrics/results.tsv")
    result = {
        "lane": lane,
        "status": "pass",
        "generated_model": summary["generated_model"],
        "generated_model_sha256": summary["generated_model_sha256"],
        "inference_model": str(inference),
        "inference_model_sha256": sha256(inference),
        "tail_model": str(tail),
        "tail_model_sha256": sha256(tail),
        "h500_predictions": str(h500 / "predictions.json"),
        "h500_prediction_sha256": sha256(h500 / "predictions.json"),
        "h500_map": h500_result["map50_95"],
        "h500_map50": h500_result["map50"],
        "h500_ap_small": h500_result["ap_small"],
        "h500_ap_medium": h500_result["ap_medium"],
        "h500_ap_large": h500_result["ap_large"],
        "h500_prediction_count": h500_result["prediction_count"],
        "scout_prediction_sha256": summary["scout_predictions_sha256"],
        "scout_map": scout_result["map50_95"],
        "scout_map50": scout_result["map50"],
        "scout_ap_small": scout_result["ap_small"],
        "scout_ap_medium": scout_result["ap_medium"],
        "scout_ap_large": scout_result["ap_large"],
        "selection_surface": "H500-only",
        "provider_claim": "none-host-only",
    }
    write_tsv(root / "variance-evaluation-summary.tsv", [result])
    write_tsv(root / "command-log.tsv", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
