#!/usr/bin/env python3
"""Audit reproducibility, topology, profile, and host semantics for A1-A6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper

LANES = ("A1", "A2", "A3", "A4", "A5", "A6")
EXPECTED_TARGET_COUNTS = {"A1": 6, "A2": 4, "A3": 3, "A4": 6, "A5": 13, "A6": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dev-python", required=True, type=Path)
    parser.add_argument("--eval-python", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--fp32-tail", required=True, type=Path)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--b2-inference", required=True, type=Path)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--fixed-list", required=True, type=Path)
    parser.add_argument("--output-contract", required=True, type=Path)
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
        raise ValueError(f"refusing to write empty report: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    return process.returncode


def model_from_run(root: Path, lane: str, run_id: str) -> Path:
    config = json.loads(
        (root / "quantization" / lane / run_id / "effective-config.json").read_text(
            encoding="utf-8"
        )
    )
    return (
        root
        / "quantization"
        / lane
        / run_id
        / "output"
        / f"{config['model_parameters']['output_prefix']}.onnx"
    )


def report_from_run(root: Path, lane: str, run_id: str) -> Path:
    matches = sorted(
        (root / "quantization" / lane / run_id / "output").glob("*_report.md")
    )
    if len(matches) != 1:
        raise ValueError(f"expected one analysis report for {lane}/{run_id}")
    return matches[0]


def qparam_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for node in model.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            names.update(value for value in node.input[1:3] if value)
    return names


def initializer_hashes(model: onnx.ModelProto) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(
            numpy_helper.to_array(item).tobytes(order="C")
        ).hexdigest()
        for item in model.graph.initializer
    }


def node_signature(model: onnx.ModelProto) -> str:
    payload = [
        {
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": [
                attribute.SerializeToString(deterministic=True).hex()
                for attribute in node.attribute
            ],
        }
        for node in model.graph.node
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def graph_audit(candidate_path: Path, b2_path: Path) -> dict[str, Any]:
    candidate = onnx.load(candidate_path, load_external_data=True)
    baseline = onnx.load(b2_path, load_external_data=True)
    onnx.checker.check_model(candidate)
    onnx.shape_inference.infer_shapes(candidate)
    candidate_qparams = qparam_names(candidate)
    baseline_qparams = qparam_names(baseline)
    candidate_initializers = initializer_hashes(candidate)
    baseline_initializers = initializer_hashes(baseline)
    same_initializer_names = set(candidate_initializers) == set(baseline_initializers)
    non_qparam_names = set(baseline_initializers) - baseline_qparams
    non_qparam_differences = sorted(
        name
        for name in non_qparam_names
        if candidate_initializers.get(name) != baseline_initializers[name]
    )
    changed_qparams = sorted(
        name
        for name in baseline_qparams & candidate_qparams
        if candidate_initializers.get(name) != baseline_initializers.get(name)
    )
    qdq_nodes = [
        node
        for node in candidate.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    ]
    zero_point_dtypes: list[int] = []
    by_initializer = {item.name: item for item in candidate.graph.initializer}
    for node in qdq_nodes:
        if len(node.input) < 3 or node.input[2] not in by_initializer:
            zero_point_dtypes.append(TensorProto.UINT8)
        else:
            zero_point_dtypes.append(by_initializer[node.input[2]].data_type)
    conv_nodes = [node for node in candidate.graph.node if node.op_type == "Conv"]
    invalid_kernel = []
    for node in conv_nodes:
        attributes = {
            item.name: helper.get_attribute_value(item) for item in node.attribute
        }
        if not attributes.get("kernel_shape"):
            invalid_kernel.append(node.name)
    fp16_initializers = [
        item.name
        for item in candidate.graph.initializer
        if item.data_type == TensorProto.FLOAT16
    ]
    topology_identity = node_signature(candidate) == node_signature(baseline)
    status = all(
        (
            topology_identity,
            same_initializer_names,
            not non_qparam_differences,
            len(qdq_nodes) == 812,
            not any(
                node.op_type.startswith("QLinear") for node in candidate.graph.node
            ),
            all(value == TensorProto.INT8 for value in zero_point_dtypes),
            len(conv_nodes) == 102,
            not invalid_kernel,
            not fp16_initializers,
        )
    )
    return {
        "status": "pass" if status else "fail",
        "topology_identity": int(topology_identity),
        "initializer_name_identity": int(same_initializer_names),
        "non_qparam_initializer_differences": len(non_qparam_differences),
        "non_qparam_difference_names": ";".join(non_qparam_differences),
        "changed_qparam_initializers": len(changed_qparams),
        "changed_qparam_names": ";".join(changed_qparams),
        "qdq_node_count": len(qdq_nodes),
        "quantize_linear_count": sum(
            node.op_type == "QuantizeLinear" for node in qdq_nodes
        ),
        "dequantize_linear_count": sum(
            node.op_type == "DequantizeLinear" for node in qdq_nodes
        ),
        "qlinear_count": sum(
            node.op_type.startswith("QLinear") for node in candidate.graph.node
        ),
        "uint8_or_implicit_zero_point_count": sum(
            value != TensorProto.INT8 for value in zero_point_dtypes
        ),
        "conv_count": len(conv_nodes),
        "invalid_kernel_shape_count": len(invalid_kernel),
        "fp16_initializer_count": len(fp16_initializers),
    }


def main() -> int:
    options = parse_args()
    options.report_dir.mkdir(parents=True, exist_ok=True)
    logs = options.raw_root / "audit-logs"
    postprocess = options.raw_root / "postprocess"
    b2 = onnx.load(options.b2_inference, load_external_data=True)
    b2_signature = node_signature(b2)

    model_rows: list[dict[str, Any]] = []
    reproducibility_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    conformance_rows: list[dict[str, Any]] = []
    qdq_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for lane in LANES:
        run1 = model_from_run(options.raw_root, lane, "run1")
        run2 = model_from_run(options.raw_root, lane, "run2")
        analysis1 = report_from_run(options.raw_root, lane, "run1")
        analysis2 = report_from_run(options.raw_root, lane, "run2")
        manifest1 = (
            options.raw_root
            / "quantization"
            / lane
            / "run1"
            / "range-policy-manifest.json"
        )
        manifest2 = (
            options.raw_root
            / "quantization"
            / lane
            / "run2"
            / "range-policy-manifest.json"
        )
        run1_hash = sha256(run1)
        run2_hash = sha256(run2)
        analysis_identity = sha256(analysis1) == sha256(analysis2)
        manifest_identity = sha256(manifest1) == sha256(manifest2)
        manifest = json.loads(manifest1.read_text(encoding="utf-8"))
        final_qparams = manifest.get("final_qparams", [])
        reproducible = (
            run1_hash == run2_hash
            and analysis_identity
            and manifest_identity
            and len(final_qparams) == EXPECTED_TARGET_COUNTS[lane]
        )
        model_rows.append(
            {
                "lane": lane,
                "run": "run1",
                "deployable": str(run1),
                "deployable_sha256": run1_hash,
                "bytes": run1.stat().st_size,
                "range_policy_manifest_sha256": sha256(manifest1),
                "matched_final_qparams": len(final_qparams),
            }
        )
        model_rows.append(
            {
                "lane": lane,
                "run": "run2",
                "deployable": str(run2),
                "deployable_sha256": run2_hash,
                "bytes": run2.stat().st_size,
                "range_policy_manifest_sha256": sha256(manifest2),
                "matched_final_qparams": len(final_qparams),
            }
        )
        reproducibility_rows.append(
            {
                "lane": lane,
                "deployable_byte_identical": int(run1_hash == run2_hash),
                "analysis_byte_identical": int(analysis_identity),
                "range_manifest_byte_identical": int(manifest_identity),
                "expected_matched_targets": EXPECTED_TARGET_COUNTS[lane],
                "observed_matched_targets": len(final_qparams),
                "status": "pass" if reproducible else "fail",
            }
        )
        analysis_rows.append(
            {
                "lane": lane,
                "run1_sha256": sha256(analysis1),
                "run2_sha256": sha256(analysis2),
                "normalized_identity": int(analysis_identity),
                "normalization": "none-required-byte-identical"
                if analysis_identity
                else "failed",
            }
        )
        for run_id in ("run1", "run2"):
            runtime_config = (
                options.raw_root / "runtime-configs" / f"{lane}-{run_id}.json"
            )
            summary = options.raw_root / "quantization" / f"{lane}-{run_id}.tsv"
            summary_row = read_tsv(summary)[0]
            command_rows.append(
                {
                    "lane": lane,
                    "run": run_id,
                    "runtime_config": str(runtime_config),
                    "runtime_config_sha256": sha256(runtime_config),
                    "seed": summary_row["random_seed"],
                    "launcher": summary_row["launcher"],
                    "python_realpath": summary_row["python_realpath"],
                    "elapsed_seconds": summary_row["elapsed_seconds"],
                    "returncode": summary_row["returncode"],
                }
            )
        if not reproducible:
            conformance_rows.append(
                {"lane": lane, "status": "rejected-nonreproducible"}
            )
            continue

        gate_path = postprocess / lane / "candidate-gate.tsv"
        if not gate_path.exists():
            returncode = run(
                [
                    str(options.eval_python),
                    str(
                        options.repo
                        / "vendor_ort_validation/stage65b_r1_candidate_gate.py"
                    ),
                    "--lane",
                    lane,
                    "--python",
                    str(options.eval_python),
                    "--repo",
                    str(options.repo),
                    "--model",
                    str(run1),
                    "--source-model",
                    str(options.source_model),
                    "--fp32-inference",
                    str(options.fp32_inference),
                    "--fp32-tail",
                    str(options.fp32_tail),
                    "--image-list",
                    str(options.h500_list),
                    "--output-root",
                    str(postprocess),
                    "--expected-tail-sha256",
                    options.expected_tail_sha256,
                    "--limit",
                    "100",
                ],
                logs / f"{lane}-candidate-gate.log",
            )
            if returncode:
                conformance_rows.append(
                    {"lane": lane, "status": "rejected-candidate-gate"}
                )
                continue
        gate = read_tsv(gate_path)[0]
        candidate_root = postprocess / lane
        model_dir = candidate_root / "models"
        inference = model_dir / f"stage65b_r1_{lane.lower()}.inference.onnx"
        tail = model_dir / f"stage65b_r1_{lane.lower()}.postprocess.onnx"
        audit = graph_audit(inference, options.b2_inference)
        audit["lane"] = lane
        audit["b2_node_signature"] = b2_signature
        audit["candidate_node_signature"] = node_signature(
            onnx.load(inference, load_external_data=True)
        )
        topology_rows.append(dict(audit))

        profile_report = candidate_root / "spacemit-profile.json"
        if not profile_report.exists():
            profile_rc = run(
                [
                    str(options.dev_python),
                    "-m",
                    "xslim.tools.spacemit_profile",
                    "--model",
                    str(inference),
                    "--output-contract",
                    str(options.output_contract),
                    "--tail",
                    str(tail),
                    "--tail-sha256",
                    options.expected_tail_sha256,
                    "--report",
                    str(profile_report),
                ],
                logs / f"{lane}-profile.log",
            )
        else:
            profile_rc = 0
        profile = json.loads(profile_report.read_text(encoding="utf-8"))
        profile_rows.append(
            {
                "lane": lane,
                "returncode": profile_rc,
                "passed": int(bool(profile.get("passed"))),
                "profile": profile.get("profile", ""),
                "report": str(profile_report),
                "report_sha256": sha256(profile_report),
                "structural_risk": profile.get("structural_risk", ""),
            }
        )

        fixed_root = candidate_root / "fixed-fixtures"
        if not (fixed_root / "host_cpu_semantic_matrix.tsv").exists():
            fixed_rc = run(
                [
                    str(options.eval_python),
                    str(
                        options.repo / "vendor_ort_validation/stage64_host_validate.py"
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
                    str(options.fixed_list),
                    "--output-dir",
                    str(fixed_root),
                    "--candidate-name",
                    lane,
                ],
                logs / f"{lane}-fixed-fixtures.log",
            )
        else:
            fixed_rc = 0
        for row in read_tsv(fixed_root / "host_cpu_semantic_matrix.tsv"):
            fixed_rows.append({"lane": lane, "returncode": fixed_rc, **row})
        semantic_root = candidate_root / "semantic-100"
        for row in read_tsv(semantic_root / "host_cpu_semantic_matrix.tsv"):
            holdout_rows.append({"lane": lane, **row})
        score_rows = read_tsv(semantic_root / "score_collapse_gate.tsv")
        collapse = sum(
            int(row["collapsed"]) for row in score_rows if row["surface"] == "split-s8"
        )
        semantic_failures = sum(
            row["status"] != "pass"
            for row in read_tsv(semantic_root / "host_cpu_semantic_matrix.tsv")
        )
        passed = all(
            (
                gate["status"] == "pass",
                audit["status"] == "pass",
                bool(profile.get("passed")),
                fixed_rc == 0,
                semantic_failures == 0,
                collapse == 0,
            )
        )
        conformance_rows.append(
            {
                "lane": lane,
                "status": "pass" if passed else "reject",
                "deployable_sha256": run1_hash,
                "inference_sha256": sha256(inference),
                "tail_sha256": sha256(tail),
                "candidate_gate": gate["status"],
                "topology_audit": audit["status"],
                "profile_pass": int(bool(profile.get("passed"))),
                "fixed_fixture_returncode": fixed_rc,
                "semantic_failures": semantic_failures,
                "score_collapses": collapse,
            }
        )
        qdq_rows.append(
            {
                "lane": lane,
                "qdq_node_count": audit["qdq_node_count"],
                "quantize_linear_count": audit["quantize_linear_count"],
                "dequantize_linear_count": audit["dequantize_linear_count"],
                "qlinear_count": audit["qlinear_count"],
                "uint8_zero_point_count": audit["uint8_or_implicit_zero_point_count"],
                "fp16_initializer_count": audit["fp16_initializer_count"],
            }
        )

    write_tsv(options.report_dir / "candidate_model_identity.tsv", model_rows)
    write_tsv(
        options.report_dir / "candidate_reproducibility.tsv", reproducibility_rows
    )
    write_tsv(options.report_dir / "candidate_normalized_analysis.tsv", analysis_rows)
    write_tsv(options.report_dir / "candidate_generation_commands.tsv", command_rows)
    write_tsv(options.report_dir / "candidate_conformance.tsv", conformance_rows)
    write_tsv(options.report_dir / "candidate_qdq_census.tsv", qdq_rows)
    write_tsv(options.report_dir / "candidate_topology_diff.tsv", topology_rows)
    write_tsv(options.report_dir / "candidate_fixed_fixtures.tsv", fixed_rows)
    write_tsv(options.report_dir / "candidate_holdout_semantics.tsv", holdout_rows)
    write_tsv(options.report_dir / "candidate_profile_validation.tsv", profile_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
