#!/usr/bin/env python3
"""Run the ordered Stage65B-R1 host gates for one generated PTQ lane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from stage64_preprocess import letterbox_rgb_nchw


FIXED_FIXTURES = (
    {
        "aliases": "F0,bus",
        "filename": "ultralytics_bus.jpg",
        "jpeg_sha256": (
            "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"
        ),
        "tensor_sha256": (
            "64d11ef4c1e470282a385f7d293607b639da2f40405c92238897253dd1e23f14"
        ),
    },
    {
        "aliases": "Zidane",
        "filename": "ultralytics_zidane.jpg",
        "jpeg_sha256": (
            "16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb"
        ),
        "tensor_sha256": (
            "a6c0d7d9e5403eec51446a5a61daa1e93bb746bc3692e8b743b41ed271812c48"
        ),
    },
    {
        "aliases": "canonical-project-image",
        "filename": "canonical_photo.jpg",
        "jpeg_sha256": (
            "4af1d821413a058744a01a9d2b20d91afa705afb9a8b98010d8f1193bd907639"
        ),
        "tensor_sha256": (
            "db87cbf5e3f43ee122f2617d08c7921f4f660dab9533470642e42a14985ba4ed"
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--fp32-tail", required=True, type=Path)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--scout-list", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--tensor-list", required=True, type=Path)
    parser.add_argument("--preprocess", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--boundary-audit", action="store_true")
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


def invoke(
    name: str,
    command: list[str],
    log: Path,
    rows: list[dict[str, Any]],
) -> None:
    started = time.monotonic()
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
    write_tsv(log.parent.parent / "command-log.tsv", rows)
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}")


def generated_model(run_root: Path) -> Path:
    config = json.loads(
        (run_root / "effective-config.json").read_text(encoding="utf-8")
    )
    prefix = config["model_parameters"]["output_prefix"]
    model = run_root / "output" / f"{prefix}.onnx"
    if not model.is_file():
        raise FileNotFoundError(model)
    return model


def graphwise_report(run_root: Path) -> Path:
    matches = sorted((run_root / "output").glob("*_report.md"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Graphwise report under {run_root / 'output'}, "
            f"found {len(matches)}"
        )
    return matches[0]


def prepare_fixed_fixtures(source_model: Path, root: Path) -> tuple[Path, Path]:
    project_root = source_model.resolve().parents[4]
    fixture_root = project_root / ".deps/images/oracle"
    fixture_list = root / "fixed-fixture-list.txt"
    identity_path = root / "fixed-fixture-identity.tsv"
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for fixture in FIXED_FIXTURES:
        path = fixture_root / str(fixture["filename"])
        if not path.is_file():
            raise FileNotFoundError(path)
        jpeg_sha256 = sha256(path)
        tensor_sha256 = hashlib.sha256(
            letterbox_rgb_nchw(path).tobytes(order="C")
        ).hexdigest()
        status = (
            "pass"
            if jpeg_sha256 == fixture["jpeg_sha256"]
            and tensor_sha256 == fixture["tensor_sha256"]
            else "fail"
        )
        rows.append(
            {
                "aliases": fixture["aliases"],
                "filename": path.name,
                "expected_jpeg_sha256": fixture["jpeg_sha256"],
                "observed_jpeg_sha256": jpeg_sha256,
                "expected_tensor_sha256": fixture["tensor_sha256"],
                "observed_tensor_sha256": tensor_sha256,
                "tensor_shape": "3x640x640",
                "tensor_dtype": "float32",
                "status": status,
            }
        )
        paths.append(path)
    if any(row["status"] != "pass" for row in rows):
        raise RuntimeError("fixed fixture identity mismatch")
    fixture_list.write_text(
        "".join(f"{path}\n" for path in paths), encoding="utf-8"
    )
    write_tsv(identity_path, rows)
    return fixture_list, identity_path


def main() -> int:
    options = parse_args()
    if options.threads < 1:
        raise ValueError("--threads must be positive")
    lane = options.lane.upper()
    root = options.output_root / lane
    if root.exists():
        raise RuntimeError(f"refusing to reuse postprocess root: {root}")
    root.mkdir(parents=True)
    logs = root / "logs"
    logs.mkdir()
    commands: list[dict[str, Any]] = []
    model = generated_model(options.run_root)
    candidate_root = root / "candidate-gate"

    invoke(
        "candidate-gate",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_candidate_gate.py"),
            "--lane",
            lane,
            "--python",
            str(options.python),
            "--repo",
            str(options.repo),
            "--model",
            str(model),
            "--source-model",
            str(options.source_model),
            "--fp32-inference",
            str(options.fp32_inference),
            "--fp32-tail",
            str(options.fp32_tail),
            "--image-list",
            str(options.h500_list),
            "--output-root",
            str(candidate_root),
            "--expected-tail-sha256",
            options.expected_tail_sha256,
            "--limit",
            "100",
        ],
        logs / "candidate-gate.log",
        commands,
    )
    gate_path = candidate_root / lane / "candidate-gate.tsv"
    gate = read_tsv(gate_path)
    if len(gate) != 1 or gate[0]["status"] != "pass":
        raise RuntimeError(f"candidate gate did not pass: {gate_path}")
    inference = candidate_root / lane / "models" / (
        f"stage65b_r1_{lane.lower()}.inference.onnx"
    )
    tail = candidate_root / lane / "models" / (
        f"stage65b_r1_{lane.lower()}.postprocess.onnx"
    )

    fixture_list, fixture_identity = prepare_fixed_fixtures(
        options.source_model, root
    )
    fixed_root = root / "fixed-fixtures"
    invoke(
        "fixed-fixture-semantics",
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
            str(inference),
            "--candidate-tail",
            str(tail),
            "--image-list",
            str(fixture_list),
            "--output-dir",
            str(fixed_root),
            "--candidate-name",
            lane,
        ],
        logs / "fixed-fixture-semantics.log",
        commands,
    )
    fixed_semantics = read_tsv(fixed_root / "host_cpu_semantic_matrix.tsv")
    fixed_scores = read_tsv(fixed_root / "score_collapse_gate.tsv")
    if len(fixed_semantics) != len(FIXED_FIXTURES) or any(
        row["status"] != "pass" for row in fixed_semantics
    ):
        raise RuntimeError("fixed fixture semantic gate did not pass")
    if any(
        row["surface"] == "split-s8" and row["collapsed"] != "0"
        for row in fixed_scores
    ):
        raise RuntimeError("fixed fixture split-S8 score collapse detected")

    invoke(
        "precision-audit",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_precision_audit.py"),
            "--model",
            f"{lane}={inference}",
            "--output",
            str(root / "precision-surface.tsv"),
        ],
        logs / "precision-audit.log",
        commands,
    )
    invoke(
        "graphwise-normalize",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_graphwise.py"),
            "--input",
            f"{lane}={graphwise_report(options.run_root)}",
            "--output",
            str(root / "graphwise-normalized.tsv"),
        ],
        logs / "graphwise-normalize.log",
        commands,
    )
    scout = root / "scout500"
    invoke(
        "scout500-predict",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_evaluate.py"),
            "run",
            "--candidate-inference",
            str(inference),
            "--tail",
            str(tail),
            "--image-list",
            str(options.scout_list),
            "--output-dir",
            str(scout),
            "--name",
            lane,
            "--threads",
            str(options.threads),
            "--log-every",
            "50",
        ],
        logs / "scout500-predict.log",
        commands,
    )
    invoke(
        "scout500-metrics",
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
            "--annotations",
            str(options.annotations),
            "--predictions",
            str(scout / "predictions.json"),
            "--image-list",
            str(options.scout_list),
            "--surface",
            lane,
            "--output-dir",
            str(scout / "metrics"),
        ],
        logs / "scout500-metrics.log",
        commands,
    )
    if options.boundary_audit:
        invoke(
            "boundary-audit-h500",
            [
                str(options.python),
                "-m",
                "xslim.tools.qdq_boundary_audit",
                "--float-model",
                str(options.fp32_inference),
                "--quant-model",
                str(inference),
                "--tensor-list",
                str(options.tensor_list),
                "--image-list",
                str(options.h500_list),
                "--preprocess",
                options.preprocess,
                "--report",
                str(root / "boundary-saturation.tsv"),
            ],
            logs / "boundary-audit-h500.log",
            commands,
        )

    write_tsv(root / "command-log.tsv", commands)
    summary = {
        "lane": lane,
        "status": "pass",
        "run_root": str(options.run_root),
        "generated_model": str(model),
        "generated_model_sha256": sha256(model),
        "inference_model": str(inference),
        "inference_model_sha256": sha256(inference),
        "tail_model": str(tail),
        "tail_model_sha256": sha256(tail),
        "candidate_gate": str(gate_path),
        "fixed_fixture_identity": str(fixture_identity),
        "fixed_fixture_unique_images": len(FIXED_FIXTURES),
        "fixed_fixture_alias_contract": "F0-is-accepted-bus-tensor",
        "fixed_fixture_semantic_status": "pass",
        "scout_predictions": str(scout / "predictions.json"),
        "scout_predictions_sha256": sha256(scout / "predictions.json"),
        "boundary_audit": int(options.boundary_audit),
    }
    write_tsv(root / "postprocess-summary.tsv", [summary])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
