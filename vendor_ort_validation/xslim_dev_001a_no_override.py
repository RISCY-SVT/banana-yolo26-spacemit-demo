#!/usr/bin/env python3
"""Compare a development no-override model with the frozen B2 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--frozen-inference", required=True, type=Path)
    parser.add_argument("--candidate-inference", required=True, type=Path)
    parser.add_argument("--tail", required=True, type=Path)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attributes(node: onnx.NodeProto) -> list[str]:
    return [
        attribute.SerializeToString(deterministic=True).hex()
        for attribute in node.attribute
    ]


def graph_identity(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model)
    nodes = [
        {
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": attributes(node),
        }
        for node in model.graph.node
    ]
    initializers = {
        item.name: {
            "dtype": item.data_type,
            "shape": list(item.dims),
            "sha256": hashlib.sha256(
                numpy_helper.to_array(item).tobytes(order="C")
            ).hexdigest(),
        }
        for item in model.graph.initializer
    }
    payload = {
        "ir_version": model.ir_version,
        "opsets": [(item.domain, item.version) for item in model.opset_import],
        "inputs": [
            item.SerializeToString(deterministic=True).hex()
            for item in model.graph.input
        ],
        "outputs": [
            item.SerializeToString(deterministic=True).hex()
            for item in model.graph.output
        ],
        "nodes": nodes,
        "initializers": initializers,
        "functions": [
            item.SerializeToString(deterministic=True).hex() for item in model.functions
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
        "node_count": len(nodes),
        "initializer_count": len(initializers),
        "payload": payload,
    }


def write_tsv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(row), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    options = parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)
    frozen_hash = sha256(options.frozen)
    candidate_hash = sha256(options.candidate)
    frozen_inference_hash = sha256(options.frozen_inference)
    candidate_inference_hash = sha256(options.candidate_inference)
    tail_hash = sha256(options.tail)
    frozen = graph_identity(options.frozen)
    candidate = graph_identity(options.candidate)
    byte_identical = frozen_hash == candidate_hash
    canonical_identical = frozen["canonical_sha256"] == candidate["canonical_sha256"]
    inference_identical = frozen_inference_hash == candidate_inference_hash
    status = (
        "pass"
        if (byte_identical or canonical_identical) and inference_identical
        else "fail"
    )
    classification = (
        "byte-identical"
        if byte_identical
        else "metadata-only"
        if canonical_identical
        else "semantic-or-qparam-difference"
    )
    row = {
        "status": status,
        "classification": classification,
        "frozen_deployable": str(options.frozen),
        "frozen_deployable_sha256": frozen_hash,
        "development_deployable": str(options.candidate),
        "development_deployable_sha256": candidate_hash,
        "byte_identical": int(byte_identical),
        "canonical_graph_identical": int(canonical_identical),
        "frozen_canonical_sha256": frozen["canonical_sha256"],
        "development_canonical_sha256": candidate["canonical_sha256"],
        "frozen_inference_sha256": frozen_inference_hash,
        "development_inference_sha256": candidate_inference_hash,
        "inference_byte_identical": int(inference_identical),
        "tail_sha256": tail_hash,
        "tail_identity_match": int(tail_hash == options.expected_tail_sha256),
    }
    if not row["tail_identity_match"]:
        row["status"] = "fail"
    write_tsv(options.output_dir / "no_override_neutrality.tsv", row)
    (options.output_dir / "no_override_graph_diff.md").write_text(
        "# No-override graph comparison\n\n"
        f"- Status: `{row['status']}`\n"
        f"- Classification: `{classification}`\n"
        f"- Raw deployable byte identity: `{bool(byte_identical)}`\n"
        f"- Canonical graph/node/initializer identity: `{bool(canonical_identical)}`\n"
        f"- Split inference byte identity: `{bool(inference_identical)}`\n"
        "- The canonical comparison includes every node, attribute, initializer "
        "dtype/shape/value, graph input/output, opset and FunctionProto.\n",
        encoding="utf-8",
    )
    return 0 if row["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
