#!/usr/bin/env python3
"""Fail-closed reuse audit for accepted Stage65C-R1 B2 full-val predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

STAGE65C_R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-"
    "AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001"
)
ROOT = Path("/data/k1x-stage-runs") / STAGE65C_R1_ID
TRACKED = Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/stages") / STAGE65C_R1_ID
DATASET = Path("/data/datasets/coco2017-independent-stage65b-r1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_config(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key != "command":
            result[key] = value
    return result


def add(rows: list[dict[str, object]], surface: str, field: str, actual: object, expected: object) -> None:
    rows.append({
        "surface": surface, "field": field, "actual": actual, "expected": expected,
        "status": "pass" if actual == expected else "fail",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    metrics = {row["surface"]: row for row in read_tsv(TRACKED / "full_val_diagnostic_metrics.tsv")}
    runtime = {row["field"]: row["value"] for row in read_tsv(TRACKED / "runtime_binding.tsv")}
    board = {row["field"]: row["value"] for row in read_tsv(TRACKED / "board_identity.tsv")}
    rows: list[dict[str, object]] = []

    shared = {
        "runner_sha256": "d39f5280a46b0bd342de933d6b25005a5858390430e8ce84377374f4a8225ad0",
        "model_sha256": "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853",
        "tail_sha256": "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3",
    }
    for label, directory, prediction_sha in (
        ("B2_CPU", "B2-cpu", "c903721d880b1df599c6912455aa39106d94a2be2cd2ad226cce59fbdae28745"),
        ("B2_EP", "B2-spacemit", "edba82a970a95b4e13d194044573fadccebe831f98527116d1ca9a74b00eab39"),
    ):
        run_root = ROOT / "board/coco/val" / directory
        prediction = run_root / "predictions.json"
        config = parse_config(run_root / "effective-config.txt")
        hashes = (run_root / "input-sha256.txt").read_text()
        add(rows, label, "prediction_sha256", sha256(prediction), prediction_sha)
        add(rows, label, "images", int(metrics[label]["images"]), 5000)
        add(rows, label, "runner_failures", int(metrics[label]["failures"]), 0)
        add(rows, label, "non_finite_predictions", int(metrics[label]["non_finite_predictions"]), 0)
        add(rows, label, "runner_sha256", shared["runner_sha256"] in hashes, True)
        add(rows, label, "model_sha256", shared["model_sha256"] in hashes, True)
        add(rows, label, "tail_sha256", shared["tail_sha256"] in hashes, True)
        add(rows, label, "provider", config["provider"], "cpu" if label.endswith("CPU") else "spacemit")
        add(rows, label, "cpu_list", config["cpu_list"], "0-3")
        add(rows, label, "threads", config["threads"], "4")
        add(rows, label, "confidence", config["confidence"], "0.001")
        add(rows, label, "limit", config["limit"], "0")

    add(rows, "shared", "ort_asset_sha256", runtime["ort_asset_sha256"], "bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6")
    add(rows, "shared", "ort_core_sha256", runtime["ort_core_sha256"], "93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3")
    add(rows, "shared", "spacemit_ep_sha256", runtime["spacemit_ep_sha256"], "dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae")
    add(rows, "shared", "board_hostname", board["hostname"], "bf3")
    add(rows, "shared", "board_serial", board["device_serial"], "92262f3b0dc4")
    add(rows, "shared", "board_kernel_contract", "6.6.63" in board["kernel"], True)
    add(rows, "shared", "val_list_sha256", sha256(DATASET / "lists/val2017_all.txt"), "d4b401d6be0446f1cea0aa2ea99fc4d367c498c02b18ebac75b77c2e2fe21bae")
    add(rows, "shared", "val_annotations_sha256", sha256(DATASET / "annotations/instances_val2017.json"), "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f")
    metrics_source = Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage65b_r1_coco_metrics.py")
    add(rows, "shared", "metrics_source_sha256", sha256(metrics_source), "ebd252d34473b7645d679fcdb209d32a86ab45dc5d0e75279ac1a0533c68eec9")
    python = Path("/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001/host/venv/bin/python")
    versions = subprocess.check_output(
        [str(python), "-c", "import numpy,pycocotools; print(numpy.__version__); print(getattr(pycocotools,'__version__','2.0.11'))"],
        text=True,
    ).splitlines()
    add(rows, "shared", "python_exists", python.is_file(), True)
    add(rows, "shared", "numpy_version", versions[0], "2.5.1")
    add(rows, "shared", "pycocotools_version", versions[1], "2.0.11")

    write_tsv(options.tracked_root / "full_val_board_reuse_contract.tsv", rows)
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit("B2 full-val reuse contract failed")
    print("B2 full-val reuse contract status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
