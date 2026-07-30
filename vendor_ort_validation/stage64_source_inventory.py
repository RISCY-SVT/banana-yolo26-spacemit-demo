#!/usr/bin/env python3
"""Inventory candidate YOLO26 ONNX sources without modifying them."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto

CANONICAL_SHA256 = (
    "d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2"
)
SPLIT_NAMES = {
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("models", nargs="+", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_info_text(value: onnx.ValueInfoProto) -> str:
    tensor = value.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor.elem_type)
    dimensions: list[str] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(str(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return f"{value.name}:{dtype}:{'x'.join(dimensions)}"


def initializer_digest(model: onnx.ModelProto) -> str:
    digest = hashlib.sha256()
    for tensor in sorted(model.graph.initializer, key=lambda item: item.name):
        digest.update(tensor.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.SerializeToString(deterministic=True))
        digest.update(b"\n")
    return digest.hexdigest()


def classify(
    model_hash: str,
    qdq_count: int,
    qlinear_count: int,
    split_count: int,
    initializer_types: Counter[str],
    fixed_640_input: bool,
) -> str:
    if model_hash == CANONICAL_SHA256:
        return "selected-canonical-fp32"
    if qdq_count or qlinear_count:
        return "already-quantized-not-static-ptq-input"
    if split_count != len(SPLIT_NAMES):
        return "split-tensor-contract-missing"
    if not fixed_640_input:
        return "not-fixed-640-input"
    if set(initializer_types) - {"FLOAT", "INT64"}:
        return "mixed-or-fp16-graph"
    return "eligible-fp32-candidate-not-selected"


def main() -> int:
    options = parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(options.models):
        model_hash = sha256(path)
        model = onnx.load(path, load_external_data=True)
        operators = Counter(node.op_type for node in model.graph.node)
        produced = {
            output
            for node in model.graph.node
            for output in node.output
            if output
        }
        visible = produced | {
            value.name
            for value in [
                *model.graph.input,
                *model.graph.output,
                *model.graph.value_info,
            ]
        }
        initializer_types = Counter(
            TensorProto.DataType.Name(tensor.data_type)
            for tensor in model.graph.initializer
        )
        qdq_count = operators["QuantizeLinear"] + operators["DequantizeLinear"]
        qlinear_count = sum(
            count
            for operator, count in operators.items()
            if operator.startswith("QLinear")
        )
        split_count = len(SPLIT_NAMES & visible)
        fixed_640_input = any(
            value_info_text(value).endswith(":FLOAT:1x3x640x640")
            for value in model.graph.input
        )
        try:
            relative = path.resolve().relative_to(options.root.resolve()).as_posix()
        except ValueError:
            relative = str(path.resolve())
        rows.append(
            {
                "relative_path": relative,
                "sha256": model_hash,
                "bytes": path.stat().st_size,
                "ir_version": model.ir_version,
                "opsets": ",".join(
                    f"{item.domain or 'ai.onnx'}:{item.version}"
                    for item in model.opset_import
                ),
                "inputs": ",".join(value_info_text(v) for v in model.graph.input),
                "outputs": ",".join(value_info_text(v) for v in model.graph.output),
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "initializer_payload_sha256": initializer_digest(model),
                "initializer_dtypes": ",".join(
                    f"{name}:{count}"
                    for name, count in sorted(initializer_types.items())
                ),
                "qdq_count": qdq_count,
                "qlinear_count": qlinear_count,
                "vendor_split_tensor_count": split_count,
                "status": classify(
                    model_hash,
                    qdq_count,
                    qlinear_count,
                    split_count,
                    initializer_types,
                    fixed_640_input,
                ),
                "provenance_status": (
                    "existing-read-only-project-evidence;"
                    "external-publication-not-authorized"
                ),
            }
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
