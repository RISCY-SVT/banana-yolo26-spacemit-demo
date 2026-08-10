#!/usr/bin/env python3
"""Apply the Stage65B-R1 split, conformance, and host semantic gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import onnx


SPLIT_NAMES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--fp32-tail", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--limit", type=int, default=100)
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
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    command: list[str], log: Path, name: str, rows: list[dict[str, Any]]
) -> None:
    start = time.monotonic()
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
            "elapsed_seconds": f"{time.monotonic() - start:.6f}",
            "log": str(log),
        }
    )
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}")


def main() -> int:
    options = parse_args()
    lane = options.lane.upper()
    root = options.output_root / lane
    if root.exists():
        raise RuntimeError(f"refusing to reuse candidate gate root: {root}")
    root.mkdir(parents=True)
    model_dir = root / "models"
    audit_dir = root / "audit"
    semantic_dir = root / "semantic-100"
    logs = root / "logs"
    model_dir.mkdir()
    audit_dir.mkdir()
    semantic_dir.mkdir()
    logs.mkdir()
    commands: list[dict[str, Any]] = []

    prefix = f"stage65b_r1_{lane.lower()}"
    inference = model_dir / f"{prefix}.inference.onnx"
    tail = model_dir / f"{prefix}.postprocess.onnx"
    run(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage64_split_model.py"),
            str(options.model),
            "--output-dir",
            str(model_dir),
            "--prefix",
            prefix,
            "--manifest",
            str(root / "split-manifest.tsv"),
        ],
        logs / "split.log",
        "split",
        commands,
    )
    run(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage64_onnx_audit.py"),
            str(options.model),
            str(inference),
            str(tail),
            "--output-dir",
            str(audit_dir),
        ],
        logs / "audit.log",
        "audit",
        commands,
    )

    checker = read_tsv(audit_dir / "generated_model_checker.tsv")
    checker_pass = all(
        row["checker_status"] == "pass"
        and row["shape_inference"] == "pass"
        for row in checker
    )
    qlinear = read_tsv(audit_dir / "qlinear_operator_census.tsv")
    qlinear_count = sum(int(row["count"]) for row in qlinear)
    qdq = read_tsv(audit_dir / "qdq_schema_census.tsv")
    inference_qdq = [row for row in qdq if row["model"] == inference.stem]
    uint8_zero_points = sum(
        "uint8" in row["zero_point_dtype"].lower() for row in inference_qdq
    )
    signed_dtype_failures = sum(
        row["zero_point_dtype"].lower() != "int8" for row in inference_qdq
    )
    weight_symmetry_failures = sum(
        row["classification"] == "weight"
        and (
            float(row["zero_point_min"]) != 0.0
            or float(row["zero_point_max"]) != 0.0
        )
        for row in inference_qdq
    )
    activation_granularity_failures = sum(
        row["classification"] == "activation"
        and (
            row["scale_shape"] != "scalar"
            or row["zero_point_shape"] != "scalar"
        )
        for row in inference_qdq
    )
    conv = read_tsv(audit_dir / "conv_kernel_shape_audit.tsv")
    inference_conv = [row for row in conv if row["model"] == inference.stem]
    conv_failures = sum(row["status"] != "pass" for row in conv)
    inference_model = onnx.load(inference, load_external_data=False)
    tail_model = onnx.load(tail, load_external_data=False)
    inference_outputs = [item.name for item in inference_model.graph.output]
    tail_inputs = [item.name for item in tail_model.graph.input]
    tail_outputs = [item.name for item in tail_model.graph.output]
    boundary_contract = (
        inference_outputs == SPLIT_NAMES
        and tail_inputs == SPLIT_NAMES
        and tail_outputs == ["output0"]
    )
    tail_hash = sha256(tail)

    presemantic_pass = all(
        (
            checker_pass,
            qlinear_count == 0,
            uint8_zero_points == 0,
            signed_dtype_failures == 0,
            weight_symmetry_failures == 0,
            activation_granularity_failures == 0,
            conv_failures == 0,
            boundary_contract,
            tail_hash == options.expected_tail_sha256,
        )
    )
    if presemantic_pass:
        run(
            [
                str(options.python),
                str(
                    options.repo
                    / "vendor_ort_validation/stage64_host_validate.py"
                ),
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
                str(options.image_list),
                "--output-dir",
                str(semantic_dir),
                "--limit",
                str(options.limit),
                "--candidate-name",
                lane,
            ],
            logs / "semantic-100.log",
            "semantic-100",
            commands,
        )
        semantic = read_tsv(semantic_dir / "host_cpu_semantic_matrix.tsv")
        scores = read_tsv(semantic_dir / "score_collapse_gate.tsv")
        semantic_failures = sum(row["status"] != "pass" for row in semantic)
        score_collapses = sum(
            row["surface"] == "split-s8" and int(row["collapsed"])
            for row in scores
        )
    else:
        semantic_failures = -1
        score_collapses = -1

    status = (
        "pass"
        if presemantic_pass
        and semantic_failures == 0
        and score_collapses == 0
        else "fail"
    )
    gate = {
        "lane": lane,
        "status": status,
        "merged_model_sha256": sha256(options.model),
        "inference_model_sha256": sha256(inference),
        "tail_model_sha256": tail_hash,
        "tail_identity_match": int(tail_hash == options.expected_tail_sha256),
        "checker_pass": int(checker_pass),
        "qdq_site_count": len(inference_qdq),
        "qlinear_count": qlinear_count,
        "uint8_zero_point_count": uint8_zero_points,
        "non_int8_zero_point_count": signed_dtype_failures,
        "weight_symmetry_failures": weight_symmetry_failures,
        "activation_granularity_failures": activation_granularity_failures,
        "conv_count": len(inference_conv),
        "conv_kernel_shape_failures": conv_failures,
        "boundary_contract": int(boundary_contract),
        "semantic_images": options.limit if presemantic_pass else 0,
        "semantic_failures": semantic_failures,
        "score_collapses": score_collapses,
    }
    write_tsv(root / "candidate-gate.tsv", [gate])
    write_tsv(root / "command-log.tsv", commands)
    print("\t".join(str(value) for value in gate.values()))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
